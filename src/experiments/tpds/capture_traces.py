"""Capture per-layer selective-scan inputs during prefill of real Mamba checkpoints (E2 input).

For every prompt of the corpus (mixed-regime serving mix + LongBench), the
model runs one prefill at batch 1. Each mixer's forward is replaced by an
equivalent prefill implementation that calls the model's own submodules
(in_proj, conv1d, x_proj, dt_proj, norm, out_proj) and a recurrence-preserving
scan kernel; right before the scan, a capture callback stores the exact scan
inputs of the selected layers:

* Mamba-1 (``state-spaces/mamba-370m-hf``): u = SiLU(conv(x)) (b, d, L),
  delta before softplus (b, d, L; dt_proj bias already included), B, C (b, n, L),
  gate z (b, d, L, optional). Per-layer constants A = -exp(A_log), D.
* Mamba-2 (``AntonV/mamba2-370m-hf``): x (b, L, h, p), dt before bias and
  softplus (b, L, h), B, C (b, L, g, n). Constants A, D, dt_bias.

Before capture starts, the patched forward is checked against the stock
transformers forward on a short prompt (logit difference is recorded).

Storage: fp16, one file per (prompt, layer) under ``<trace-dir>/<model>/``,
plus ``manifest.jsonl`` (one fsynced row per completed prompt; completed
prompts are skipped on restart) and a byte-size guard.

Example (smoke):
    python -m src.experiments.tpds.capture_traces --model mamba1 --layers 0,24 \
        --samples-per-regime 1 --longbench-per-task 1 --max-tokens 2048 --limit-prompts 4 \
        --trace-dir src/outputs/tpds_traces_smoke
"""
from __future__ import annotations

import argparse
import os
import time
import types
from pathlib import Path

import torch

from src.experiments.tpds.common import DEFAULT_TRACE_DIR, JsonlWriter, Logger, check_disk, env_metadata, read_jsonl
from src.experiments.tpds.corpus import build_corpus, prompt_fold

MODEL_IDS = {"mamba1": "state-spaces/mamba-370m-hf", "mamba2": "AntonV/mamba2-370m-hf"}
FP16_MAX = 65504.0


# --------------------------------------------------------------------------- patched mixers
class _Capture:
    def __init__(self):
        self.layers: set[int] = set()
        self.enabled = False
        self.store: dict[int, dict[str, torch.Tensor]] = {}
        self.params: dict[int, dict[str, torch.Tensor]] = {}

    def __call__(self, layer: int, tensors: dict[str, torch.Tensor], params: dict[str, torch.Tensor]) -> None:
        if not self.enabled or layer not in self.layers:
            return
        packed, over = {}, 0
        for k, v in tensors.items():
            packed[k], o = _to_fp16(v.detach())
            over += o
        self.store[layer] = (packed, over)
        if layer not in self.params:
            self.params[layer] = {k: v.detach().float().cpu().clone() for k, v in params.items()}


def _mamba1_prefill_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    from src.experiments.tpds.kernels import mamba1_scan

    if cache_params is not None:
        return self._tpds_orig_forward(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)
    bsz, L, _ = hidden_states.shape
    proj = self.in_proj(hidden_states).transpose(1, 2)
    x, gate = proj.chunk(2, dim=1)
    if attention_mask is not None:
        x = x * attention_mask.unsqueeze(1)
    x = self.act(self.conv1d(x)[..., :L])
    if attention_mask is not None:
        x = x * attention_mask.unsqueeze(1)
    ssm = self.x_proj(x.transpose(1, 2))
    ts, B, C = torch.split(ssm, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)
    dt = self.dt_proj(ts)  # pre-softplus, bias included
    A = -torch.exp(self.A_log.float())
    u = x.contiguous()
    delta = dt.transpose(1, 2).contiguous()
    Bt = B.transpose(1, 2).contiguous()
    Ct = C.transpose(1, 2).contiguous()
    z = gate.contiguous()
    self._tpds_capture(self.layer_idx, {"u": u, "delta": delta, "B": Bt, "C": Ct, "z": z},
                       {"A": A, "D": self.D})
    y = mamba1_scan(u, delta, A.contiguous(), Bt, Ct, self.D.float(), z, None, True, chunk=256)
    return self.out_proj(y.transpose(1, 2))


def _mamba2_prefill_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    from src.experiments.tpds.kernels import mamba2_ssd

    if cache_params is not None:
        return self._tpds_orig_forward(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)
    bsz, L, _ = hidden_states.shape
    proj = self.in_proj(hidden_states)
    d_mlp = (proj.shape[-1] - 2 * self.intermediate_size - 2 * self.n_groups * self.ssm_state_size - self.num_heads) // 2
    _, _, gate, xBC, dt = proj.split([d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1)
    xBC = self.act(self.conv1d(xBC.transpose(1, 2))[..., :L].transpose(1, 2))
    gs = self.n_groups * self.ssm_state_size
    x, B, C = torch.split(xBC, [self.intermediate_size, gs, gs], dim=-1)
    A = -torch.exp(self.A_log.float())
    x4 = x.reshape(bsz, L, -1, self.head_dim).contiguous()
    B4 = B.reshape(bsz, L, self.n_groups, -1).contiguous()
    C4 = C.reshape(bsz, L, self.n_groups, -1).contiguous()
    dt = dt.contiguous()
    self._tpds_capture(self.layer_idx, {"x": x4, "dt": dt, "B": B4, "C": C4},
                       {"A": A, "D": self.D, "dt_bias": self.dt_bias})
    y = mamba2_ssd(x4, dt, A, B4, C4, self.chunk_size, D=self.D, dt_bias=self.dt_bias, dt_softplus=True)
    y = y.reshape(bsz, L, -1)
    y = self.norm(y, gate)
    return self.out_proj(y)


def _disable_hf_fused_kernels() -> None:
    """Keep transformers off its fused Mamba kernels (compiled CUDA extensions or hub kernels).

    The capture forward uses our own Triton kernels, so the fused path is never needed; the
    unpatched reference forward then runs the portable PyTorch path. This avoids depending on
    whichever mamba_ssm/causal_conv1d build a host has installed, which may lack kernels for the
    GPU architecture or be incompatible with the installed transformers version.
    """
    import importlib

    for name in ("transformers.models.mamba.modeling_mamba", "transformers.models.mamba2.modeling_mamba2"):
        mod = importlib.import_module(name)
        if hasattr(mod, "lazy_load_kernel"):
            mod.lazy_load_kernel = lambda *args, **kwargs: None


def load_model(kind: str, model_id: str, device: str = "cuda", dtype=torch.float32):
    from transformers import AutoTokenizer

    _disable_hf_fused_kernels()
    if kind == "mamba1":
        from transformers import MambaForCausalLM as Cls
    else:
        from transformers import Mamba2ForCausalLM as Cls
    tok = AutoTokenizer.from_pretrained(model_id)
    model = Cls.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return tok, model


def patch_model(kind: str, model, capture: _Capture) -> int:
    fwd = _mamba1_prefill_forward if kind == "mamba1" else _mamba2_prefill_forward
    count = 0
    for layer in model.backbone.layers:
        mixer = layer.mixer
        mixer._tpds_orig_forward = mixer.forward
        mixer._tpds_capture = capture
        mixer.forward = types.MethodType(fwd, mixer)
        count += 1
    return count


def unpatch_model(model) -> None:
    for layer in model.backbone.layers:
        mixer = layer.mixer
        if hasattr(mixer, "_tpds_orig_forward"):
            mixer.forward = mixer._tpds_orig_forward


@torch.no_grad()
def check_patch_equivalence(kind: str, model, tok, capture: _Capture, text: str, n_tokens: int = 48) -> dict:
    ids = tok(text, return_tensors="pt").input_ids[:, :n_tokens].to(model.device)
    unpatch_model(model)
    ref = model(ids, use_cache=False).logits.float()
    patch_model(kind, model, capture)
    out = model(ids, use_cache=False).logits.float()
    diff = (out - ref).abs()
    return {"tokens": int(ids.shape[1]), "max_abs_logit_diff": float(diff.max()),
            "ref_logit_max_abs": float(ref.abs().max()),
            "argmax_agreement": float((out.argmax(-1) == ref.argmax(-1)).float().mean())}


# --------------------------------------------------------------------------- storage helpers
def model_dir(trace_dir: Path, kind: str) -> Path:
    return Path(trace_dir) / kind


def call_path(trace_dir: Path, kind: str, prompt_id: str, layer: int) -> Path:
    return model_dir(trace_dir, kind) / prompt_id / f"L{layer:02d}.pt"


def param_path(trace_dir: Path, kind: str, layer: int) -> Path:
    return model_dir(trace_dir, kind) / "params" / f"L{layer:02d}.pt"


def load_manifest(trace_dir: Path, kind: str) -> list[dict]:
    rows = read_jsonl(model_dir(trace_dir, kind) / "manifest.jsonl")
    seen = {}
    for r in rows:
        seen[r["key"]] = r
    return list(seen.values())


def load_call(trace_dir: Path, kind: str, prompt_id: str, layer: int, device="cuda") -> dict[str, torch.Tensor]:
    d = torch.load(call_path(trace_dir, kind, prompt_id, layer), map_location="cpu", weights_only=True)
    return {k: v.to(device) for k, v in d.items()}


def load_params(trace_dir: Path, kind: str, layer: int, device="cuda") -> dict[str, torch.Tensor]:
    d = torch.load(param_path(trace_dir, kind, layer), map_location="cpu", weights_only=True)
    return {k: v.to(device) for k, v in d.items()}


def _atomic_save(obj, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path.stat().st_size


def _to_fp16(t: torch.Tensor) -> tuple[torch.Tensor, int]:
    over = int((t.abs() > FP16_MAX).sum())
    return t.clamp(-FP16_MAX, FP16_MAX).to(torch.float16).cpu().contiguous(), over


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(MODEL_IDS), required=True)
    p.add_argument("--model-id", default=None)
    p.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    p.add_argument("--layer-stride", type=int, default=6, help="capture layers 0, s, 2s, ...")
    p.add_argument("--layers", default=None, help="explicit comma-separated layer list (overrides --layer-stride)")
    p.add_argument("--samples-per-regime", type=int, default=20)
    p.add_argument("--longbench-per-task", type=int, default=8)
    p.add_argument("--longbench-lengths", default="1024,2048,4096,8192")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--limit-prompts", type=int, default=None)
    p.add_argument("--no-z", action="store_true", help="Mamba-1: do not store the gate z")
    p.add_argument("--max-gb", type=float, default=40.0, help="size guard for stored traces")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-name", default=None)
    return p.parse_args(argv)


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    kind = args.model
    model_id = args.model_id or MODEL_IDS[kind]
    log = Logger(args.log_name or f"capture_{kind}")
    mdir = model_dir(args.trace_dir, kind)
    mdir.mkdir(parents=True, exist_ok=True)
    check_disk(mdir, need_gb=min(args.max_gb, 200.0) * 1.1 + 1.0, need_inodes=20000, log=log)

    lengths = tuple(int(x) for x in args.longbench_lengths.split(",") if x)
    corpus = build_corpus(args.samples_per_regime, args.longbench_per_task, lengths, args.seed)
    if args.limit_prompts:
        # Keep the regime mix when limiting: take prompts round-robin over regimes.
        by_reg: dict[str, list] = {}
        for r in corpus:
            by_reg.setdefault(r["regime"], []).append(r)
        picked, i = [], 0
        while len(picked) < args.limit_prompts and any(by_reg.values()):
            for reg in list(by_reg):
                if by_reg[reg] and len(picked) < args.limit_prompts:
                    picked.append(by_reg[reg].pop(0))
        corpus = picked
    log(f"corpus: {len(corpus)} prompts, model {model_id}")

    tok, model = load_model(kind, model_id)
    n_layers = len(model.backbone.layers)
    layers = ([int(x) for x in args.layers.split(",")] if args.layers else list(range(0, n_layers, args.layer_stride)))
    capture = _Capture()
    capture.layers = set(layers)
    patch_model(kind, model, capture)
    equiv = check_patch_equivalence(kind, model, tok, capture, corpus[0]["text"])
    log(f"patched-forward equivalence: {equiv}")

    meta = env_metadata()
    meta.update({"model_kind": kind, "model_id": model_id, "layers": layers, "n_layers": n_layers,
                 "equivalence_check": equiv, "store_z": not args.no_z, "args": vars(args)})
    writer = JsonlWriter(mdir / "manifest.jsonl", meta={"key": None, **meta})
    total_bytes = sum(int(r.get("bytes", 0)) for r in load_manifest(args.trace_dir, kind))
    budget = args.max_gb * 2**30

    for idx, item in enumerate(corpus):
        pid = item["id"]
        if pid in writer.done:
            continue
        if total_bytes >= budget:
            log(f"size guard reached ({total_bytes / 2**30:.2f} GB >= {args.max_gb} GB); stopping capture")
            break
        max_tok = min(item["max_tokens"] or args.max_tokens, args.max_tokens)
        ids = tok(item["text"], return_tensors="pt").input_ids[:, :max_tok].to(model.device)
        L = int(ids.shape[1])
        capture.store.clear()
        capture.enabled = True
        t0 = time.perf_counter()
        model(ids, use_cache=False)
        torch.cuda.synchronize()
        fwd_s = time.perf_counter() - t0
        capture.enabled = False
        nbytes, overflow = 0, 0
        for layer in layers:
            packed, o = capture.store.pop(layer)
            overflow += o
            if kind == "mamba1" and args.no_z:
                packed.pop("z", None)
            nbytes += _atomic_save(packed, call_path(args.trace_dir, kind, pid, layer))
            pp = param_path(args.trace_dir, kind, layer)
            if not pp.exists():
                _atomic_save(capture.params[layer], pp)
        total_bytes += nbytes
        writer.write({"key": pid, "prompt_id": pid, "regime": item["regime"], "source": item["source"],
                      "fold": prompt_fold(pid), "L": L, "layers": layers, "bytes": nbytes,
                      "fp16_overflow": overflow, "prefill_s": round(fwd_s, 4)})
        log(f"[{idx + 1}/{len(corpus)}] {pid} L={L} {nbytes / 2**20:.1f} MB (total {total_bytes / 2**30:.2f} GB) "
            f"prefill {fwd_s:.2f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    writer.close()
    log("capture done")


if __name__ == "__main__":
    main()
