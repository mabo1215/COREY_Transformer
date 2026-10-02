"""Capture the Gated DeltaNet kernel inputs of a public checkpoint (review item M2b).

Loads a Gated DeltaNet checkpoint through the FLA model classes (``fla.models``), runs the prompt
corpus of ``capture_traces.py`` through it and stores, for the selected layers, exactly the tensors
the layer hands to the FLA kernel: q, k, v (the raw projections, after the short convolution or the
SiLU), g (the output of ``a_proj``, turned into the log-gate inside the kernel) and beta (the output
of ``b_proj``, passed through the sigmoid inside the kernel), plus the per-layer parameters A_log
and dt_bias. The capture wraps the two kernel entry points the layer imports
(``chunk_gated_delta_rule`` for prompts longer than 64 tokens, ``fused_recurrent_gated_delta_rule``
for shorter ones), so the model's computation is unchanged and every prompt is captured; the
manifest records which path the model took.

Storage is the format of ``capture_traces.py`` under ``<trace-dir>/gdn/`` (one ``.pt`` per prompt
and layer, ``params/Lxx.pt``, ``manifest.jsonl``), so ``e2_oracle_sweep.py --model gdn`` reads it
directly. Tensors are stored in fp16 (exact for in-range bf16 values; overflow is counted).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from src.experiments.tpds.capture_traces import _atomic_save, _to_fp16, call_path, load_manifest, model_dir, param_path
from src.experiments.tpds.common import DEFAULT_TRACE_DIR, JsonlWriter, Logger, check_disk, env_metadata, write_json
from src.experiments.tpds.corpus import build_corpus, prompt_fold

KIND = "gdn"
DEFAULT_MODEL_ID = "Idiap/gated-deltanet-swa-1.4B-30B"
TENSORS = ("q", "k", "v", "g", "beta")


class _Capture:
    def __init__(self):
        self.layers: set[int] = set()
        self.enabled = False
        self.current: int | None = None
        self.store: dict[int, tuple[dict, int, str]] = {}
        self.params: dict[int, dict[str, torch.Tensor]] = {}
        self.live_out: dict[int, torch.Tensor] = {}
        self.keep_out = False

    def record(self, tensors: dict, params: dict, path: str, out: torch.Tensor | None) -> None:
        layer = self.current
        if not self.enabled or layer is None or layer not in self.layers:
            return
        packed, over = {}, 0
        for k in TENSORS:
            packed[k], o = _to_fp16(tensors[k].detach())
            over += o
        self.store[layer] = (packed, over, path)
        if layer not in self.params:
            self.params[layer] = {k: v.detach().cpu().clone() for k, v in params.items() if torch.is_tensor(v)}
            self.params[layer]["allow_neg_eigval"] = torch.tensor(bool(params.get("allow_neg_eigval", False)))
        if self.keep_out and out is not None:
            self.live_out[layer] = out.detach().float().cpu()


_ACTIVE: _Capture | None = None


def _wrap(orig, path: str):
    def wrapped(*args, **kw):
        cap = _ACTIVE
        vals = dict(zip(TENSORS, args))
        vals.update({n: kw[n] for n in TENSORS if n in kw})
        res = orig(*args, **kw)
        if cap is not None and cap.enabled:
            out = res[0] if isinstance(res, (tuple, list)) else res
            cap.record(vals, {"A_log": kw.get("A_log"), "dt_bias": kw.get("dt_bias"),
                              "allow_neg_eigval": kw.get("allow_neg_eigval", False)}, path, out)
        return res
    wrapped._tpds_orig = orig  # type: ignore[attr-defined]
    return wrapped


def patch_kernels(layer_module) -> None:
    for name in ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"):
        fn = getattr(layer_module, name)
        if not hasattr(fn, "_tpds_orig"):
            setattr(layer_module, name, _wrap(fn, "chunk" if name.startswith("chunk") else "fused_recurrent"))


def gdn_layers(model):
    """(index, GatedDeltaNet module) for every block whose mixer is a Gated DeltaNet layer."""
    from fla.layers.gated_deltanet import GatedDeltaNet  # type: ignore

    blocks = model.model.layers if hasattr(model, "model") else model.layers
    return [(i, blk.attn) for i, blk in enumerate(blocks) if isinstance(getattr(blk, "attn", None), GatedDeltaNet)]


def load_model(model_id: str, dtype, device: str = "cuda"):
    import fla  # noqa: F401  (registers the FLA model classes with transformers)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from fla.models.gated_deltanet import GatedDeltaNetForCausalLM  # type: ignore

    # transformers >= 5 expects ``_tied_weights_keys`` as a {target: source} mapping; fla 0.5.x still
    # declares the list ["lm_head.weight"], which breaks post_init().
    keys = getattr(GatedDeltaNetForCausalLM, "_tied_weights_keys", None)
    if isinstance(keys, (list, tuple)):
        GatedDeltaNetForCausalLM._tied_weights_keys = {k: "model.embeddings.weight" for k in keys}
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return tok, model


@torch.no_grad()
def roundtrip_check(trace_dir: Path, pid: str, layer: int, capture: _Capture) -> dict:
    """Re-run the production kernel on the stored (fp16) inputs and compare with the live output."""
    from src.experiments.tpds import kernels as K
    from src.experiments.tpds.capture_traces import load_call, load_params

    call = load_call(trace_dir, KIND, pid, layer)
    prm = load_params(trace_dir, KIND, layer)
    inp = {k: call[k].to(torch.bfloat16) for k in TENSORS}
    o = K.gdn_chunk_prod(inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], prm["A_log"], prm["dt_bias"],
                         chunk_size=64, allow_neg_eigval=bool(prm["allow_neg_eigval"].item())).float().cpu()
    live = capture.live_out[layer]
    diff = (o - live).abs()
    return {"layer": layer, "L": int(o.shape[1]), "max_abs_diff": float(diff.max()),
            "max_rel_diff": float(diff.max() / live.abs().max().clamp(min=1e-6)),
            "live_abs_max": float(live.abs().max())}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    p.add_argument("--layer-stride", type=int, default=2, help="capture every s-th Gated DeltaNet layer (hybrids: attention layers are skipped)")
    p.add_argument("--layers", default=None, help="explicit comma-separated layer list (overrides --layer-stride)")
    p.add_argument("--samples-per-regime", type=int, default=20)
    p.add_argument("--longbench-per-task", type=int, default=8)
    p.add_argument("--longbench-lengths", default="1024,2048,4096,8192")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--limit-prompts", type=int, default=None)
    p.add_argument("--max-gb", type=float, default=20.0, help="size guard for stored traces")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-name", default=None)
    return p.parse_args(argv)


@torch.no_grad()
def main(argv=None):
    global _ACTIVE
    args = parse_args(argv)
    log = Logger(args.log_name or f"capture_{KIND}")
    mdir = model_dir(args.trace_dir, KIND)
    mdir.mkdir(parents=True, exist_ok=True)
    check_disk(mdir, need_gb=min(args.max_gb, 200.0) * 1.1 + 1.0, need_inodes=20000, log=log)

    lengths = tuple(int(x) for x in args.longbench_lengths.split(",") if x)
    corpus = build_corpus(args.samples_per_regime, args.longbench_per_task, lengths, args.seed)
    if args.limit_prompts:
        by_reg: dict[str, list] = {}
        for r in corpus:
            by_reg.setdefault(r["regime"], []).append(r)
        picked = []
        while len(picked) < args.limit_prompts and any(by_reg.values()):
            for reg in list(by_reg):
                if by_reg[reg] and len(picked) < args.limit_prompts:
                    picked.append(by_reg[reg].pop(0))
        corpus = picked
    log(f"corpus: {len(corpus)} prompts, model {args.model_id}")

    dtype = getattr(torch, args.dtype)
    tok, model = load_model(args.model_id, dtype)
    import fla.layers.gated_deltanet as layer_module  # type: ignore

    patch_kernels(layer_module)
    gl = gdn_layers(model)
    if not gl:
        raise SystemExit("no Gated DeltaNet layers found in the model")
    gdn_idx = [i for i, _ in gl]
    layers = ([int(x) for x in args.layers.split(",")] if args.layers else gdn_idx[::args.layer_stride])
    capture = _Capture()
    capture.layers = set(layers)
    _ACTIVE = capture
    for i, mod in gl:
        mod.register_forward_pre_hook(lambda m, a, kw, idx=i: setattr(capture, "current", idx), with_kwargs=True)
    first = gl[0][1]
    shapes = {"num_heads": int(getattr(first, "num_heads", 0)), "head_k_dim": int(getattr(first, "head_k_dim", 0)),
              "head_v_dim": int(getattr(first, "head_v_dim", 0)), "num_v_heads": int(getattr(first, "num_v_heads", 0)),
              "use_short_conv": bool(getattr(first, "use_short_conv", False)),
              "allow_neg_eigval": bool(getattr(first, "allow_neg_eigval", False)), "mode": str(getattr(first, "mode", ""))}
    log(f"model: {len(gl)} Gated DeltaNet layers of {len(model.model.layers)}; capturing {layers}; layer shapes {shapes}")

    meta = env_metadata()
    meta.update({"model_kind": KIND, "model_id": args.model_id, "dtype": args.dtype, "layers": layers,
                 "n_layers": len(model.model.layers), "gdn_layers": gdn_idx, "layer_shapes": shapes,
                 "kernel_flags": {"use_qk_l2norm_in_kernel": True, "use_gate_in_kernel": True,
                                  "use_beta_sigmoid_in_kernel": True, "state_v_first": True},
                 "args": vars(args)})
    writer = JsonlWriter(mdir / "manifest.jsonl", meta={"key": None, **meta})
    total_bytes = sum(int(r.get("bytes", 0)) for r in load_manifest(args.trace_dir, KIND))
    budget = args.max_gb * 2**30
    checked = False

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
        capture.live_out.clear()
        capture.keep_out = not checked
        capture.enabled = True
        t0 = time.perf_counter()
        model(ids, use_cache=False)
        torch.cuda.synchronize()
        fwd_s = time.perf_counter() - t0
        capture.enabled = False
        missing = [l for l in layers if l not in capture.store]
        if missing:
            raise SystemExit(f"layers {missing} were not captured for {pid}; the kernel wrappers did not fire")
        nbytes, overflow, paths = 0, 0, set()
        for layer in layers:
            packed, o, path = capture.store.pop(layer)
            overflow += o
            paths.add(path)
            nbytes += _atomic_save(packed, call_path(args.trace_dir, KIND, pid, layer))
            pp = param_path(args.trace_dir, KIND, layer)
            if not pp.exists():
                _atomic_save(capture.params[layer], pp)
        total_bytes += nbytes
        mode = ";".join(sorted(paths))
        writer.write({"key": pid, "prompt_id": pid, "regime": item["regime"], "source": item["source"],
                      "fold": prompt_fold(pid), "L": L, "layers": layers, "bytes": nbytes,
                      "fp16_overflow": overflow, "prefill_s": round(fwd_s, 4), "mode": mode})
        log(f"[{idx + 1}/{len(corpus)}] {pid} L={L} {mode} {nbytes / 2**20:.1f} MB (total {total_bytes / 2**30:.2f} GB) "
            f"prefill {fwd_s:.2f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
        if not checked and mode == "chunk":
            rt = roundtrip_check(args.trace_dir, pid, layers[0], capture)
            log(f"stored-input round trip (production kernel, chunk 64): {rt}")
            write_json(mdir / "roundtrip_check.json", rt)
            checked = True
            capture.keep_out = False
    writer.close()
    log("capture done")


if __name__ == "__main__":
    main()
