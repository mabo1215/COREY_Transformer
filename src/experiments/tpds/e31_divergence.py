"""E31 - output divergence of per-shape chunk switching (review item M8b).

Choosing the chunk of a chunked linear-recurrence scan per call shape changes the floating-point
summation order of the prefill scan. This experiment measures what that does to the model output:
for every prompt it compares the next-token distributions and the greedy continuation of a model
whose prefill scan runs at a fixed reference chunk with the same model at every alternative chunk.

Models (one per process, ``--model``):

  mamba2  Mamba-2 370M (``AntonV/mamba2-370m-hf``, the checkpoint of the captured traces). With
          mamba_ssm installed, transformers' Mamba-2 mixer calls the Triton SSD kernel
          ``mamba_chunk_scan_combined`` with its ``chunk_size`` attribute; the attribute is set on
          every mixer before the prefill, and a wrapper around the kernel entry point records the
          chunk every prefill call received. Chunks 32..512; reference = the library default 256.
          Without mamba_ssm the portable PyTorch path runs (it reads the same attribute); that is
          only allowed with ``--allow-slow-path`` (local smoke tests).
  mamba1  Mamba-1 370M (``state-spaces/mamba-370m-hf``). Reference = the unmodified model (stock
          mamba_ssm dispatch, whose chunk follows the length: 128/256/512/1024 for L up to that
          value, 2048 above). Alternatives = the routed static chunks of the routed study
          (128..2048): the machinery of ``e10_routed_interleaved`` is reused unchanged (the
          ``selective_scan_cuda`` stand-in with ``fwd_with_chunk_size`` from the runtime-chunk
          build, the per-chunk check against the stock scan, the patched mixer forward of the
          routed study with a constant chunk). Every routed prefill is checked to have routed
          that chunk into all layers.
  gdn     Gated DeltaNet 1.4B (``capture_gdn_traces.DEFAULT_MODEL_ID``; on the RTX 3090 host pass
          the local copy with ``--model-id``). The ``chunk_gated_delta_rule`` entry point that the
          FLA layer calls is wrapped so that ``chunk_size`` (16/32/64; reference 64, the FLA
          default) is injected. Prompts of at most 64 tokens take FLA's fused recurrent path in
          prefill (no chunk), which the row records. This path was not run locally (no FLA).

Prompts: the corpus of the captured traces (``capture_traces.select_corpus`` with the capture
arguments of the lanes: 20 samples per regime, 8 LongBench prompts per task, target lengths
1024-8192, at most 8192 tokens, seed 0; 192 prompts), tokenized exactly as the capture did and then
cropped to ``--max-len`` (default 2048) tokens. ``--n-prompts`` (default 200, i.e. all 192) takes a
deterministic subset stratified over power-of-two length buckets and regimes (buckets visited from
both ends: longest, shortest, second longest, ...; regimes round-robin inside a bucket; prompts in
a stratum ordered by a hash of their id); the same order is the execution order.

Per prompt (batch 1, greedy, ``torch.inference_mode``):

* reference run: prefill at the reference chunk, then ``--new-tokens`` (32) greedy decode steps
  with the recurrent cache (decode steps never use the chunked scan). The logits of positions
  0..32 are kept on the GPU (position 0 = the prefill's last position; position k = after feeding
  the k-th generated token). Stored: the 32 token ids, and per position the top-k (``--top-k``,
  20) token ids and log-probabilities and the entropy of the distribution. EOS is not a stop
  condition: every run generates exactly ``--new-tokens`` tokens.
* every alternative chunk and the reference chunk a second time (``rerun``, the run-to-run
  nondeterminism floor):
  (b) teacher-forced: prefill at that chunk, then the reference tokens are fed step by step
      through the decode path; at each of the 33 positions KL(ref || alt) of the next-token
      distributions over the full vocabulary (logits cast to float32, log-softmax and the sum in
      float64), the maximum |logit difference| and top-1 agreement.
  (a) free-running greedy: up to the first position whose top-1 differs from the reference token,
      the free-running inputs are the reference tokens, so its tokens equal the teacher-forced
      prefix. When the teacher-forced top-1 never departs from the reference tokens the
      free-running output is the reference output (recorded as derived); otherwise a separate
      free-running generation is run, its first divergent position and number of matching tokens
      are recorded, and its logits on the shared prefix are checked to be bitwise equal to the
      teacher-forced ones (``prefix_consistent``). ``--free-run always`` runs it for every chunk.

Every row holds all chunks of one (model, dtype, prompt), so all comparisons are within one process
(Triton autotuning picks a kernel configuration per chunk once per process, so the reference and
its rerun share it; a restart only re-runs whole prompts).

Resumable: one fsynced JSONL row per (model, dtype, new tokens, max length, prompt) with key
``e31|<model>|<dtype>|n<new>|L<max-len>|<prompt id>``; completed keys are skipped on restart. One
process per result file (an exclusive lock under the log directory); run the models one after the
other with the same ``--tag`` or concurrently with different tags (``--analyze --input-tags``
merges them).

``--analyze`` (CPU only) writes ``e31_divergence_<tag>.json``: per model, dtype and chunk the share
of prompts whose greedy output diverges within the new tokens, the median / minimum first
divergent position, KL at position 0 and over all positions (mean, median, max), maximum |logit
difference|, top-1 agreement, the share of bitwise-identical logits, and a breakdown by length
bucket; the same for the reference rerun; and a "per-shape table" row: for each prompt the chunk a
per-length table would choose and that chunk's divergence (the reference chunk -> the rerun). The
tables are batch-1, per power-of-two length bucket, the chunk with the lowest mean latency over the
bucket's captured calls: Mamba-2 from the RTX 3090 CUDA-graph three-round sweep of the 370M SSD
calls (round medians), Mamba-1 from the RTX 3090 reference-scan sweep (runtime-chunk CUDA scan, one
round of 50 repeats), Gated DeltaNet from the RTX 3090 three-round 1.4B sweep. A bucket missing from
a table falls back to the nearest bucket (log2 distance, ties to the smaller). The mode of the
per-call oracle chunk is reported as a second table variant.

``--dry-run`` prints the prompt plan (needs the tokenizer only, no model, no GPU) and a runtime
estimate. ``--download-only`` fetches the checkpoint and tokenizer into the Hugging Face cache (CPU;
``--hf-home`` or ``HF_HOME``/``HF_ENDPOINT`` from the environment). ``--smoke``: 3 prompts, 8 new
tokens, results in the smoke directory, slow path allowed.

Examples:
    python -m src.experiments.tpds.e31_divergence --model mamba2 --dry-run
    python -m src.experiments.tpds.e31_divergence --model mamba2 --download-only
    python -m src.experiments.tpds.e31_divergence --model mamba2 --smoke
    python -m src.experiments.tpds.e31_divergence --model mamba1 --dtype float16 --tag vgpu3090
    python -m src.experiments.tpds.e31_divergence --analyze --tag vgpu3090
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.experiments.tpds.common import (GDN_CHUNKS, LOG_DIR, MAMBA2_CHUNKS, RESULTS_DIR, SMOKE_DIR, JsonlWriter, Logger,
                                         check_disk, dedupe_by_key, env_metadata, host_name, read_jsonl, utc_now,
                                         write_json)
from src.experiments.tpds.corpus import prompt_fold
from src.experiments.tpds.e10_routed_interleaved import STATIC_CHUNKS as MAMBA1_ROUTED_CHUNKS
from src.experiments.tpds.e10_routed_interleaved import stock_chunk

MODELS = ("mamba2", "mamba1", "gdn")
DTYPES = ("float16", "bfloat16", "float32")
REFERENCE_CHUNK = {"mamba2": 256, "mamba1": None, "gdn": 64}   # None: the stock length-based dispatch
ALT_CHUNKS = {"mamba2": tuple(c for c in MAMBA2_CHUNKS if c != 256), "mamba1": tuple(MAMBA1_ROUTED_CHUNKS),
              "gdn": tuple(c for c in GDN_CHUNKS if c != 64)}
CHECKPOINT_GB = {"mamba2": 0.74, "mamba1": 1.49, "gdn": 5.36}    # safetensors sizes on the Hub
# Corpus arguments of the trace capture (CAP_ARGS of src/experiments/remote/tpds_common.sh).
CAPTURE_CORPUS = {"samples_per_regime": 20, "longbench_per_task": 8, "longbench_lengths": "1024,2048,4096,8192",
                  "seed": 0}
CAPTURE_MAX_TOKENS = 8192
TABLE_DIR = RESULTS_DIR
DEFAULT_TABLES = {
    # RTX 3090, CUDA-graph timing, three rounds (round medians): the 370M Mamba-2 SSD calls.
    "mamba2": TABLE_DIR / "e2_mamba2_vgpu3090_r4graph3.csv",
    # RTX 3090, runtime-chunk reference CUDA scan, 50 repeats in one round: the 370M Mamba-1 calls.
    "mamba1": TABLE_DIR / "e8_refscan_vgpu3090.csv",
    # RTX 3090, CUDA-graph timing, three rounds: the 1.4B Gated DeltaNet calls.
    "gdn": TABLE_DIR / "e2_gdn_vgpu3090_1p4b_r4graph3.csv",
}


def _length_bucket(L: int) -> int:
    """Power-of-two length bucket (floor), as ``e2_oracle_sweep.length_bucket``."""
    return 1 << max(0, int(math.floor(math.log2(max(int(L), 1)))))


def _g(x: Any, digits: int = 6) -> Any:
    """Round a float to ``digits`` significant digits for compact JSON (NaN/inf kept)."""
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return x
    return float(f"{x:.{digits}g}")


def _sha1(values: list[int]) -> str:
    return hashlib.sha1(",".join(map(str, values)).encode()).hexdigest()


def default_model_id(kind: str) -> str:
    if kind == "gdn":
        from src.experiments.tpds.capture_gdn_traces import DEFAULT_MODEL_ID

        return DEFAULT_MODEL_ID
    from src.experiments.tpds.capture_traces import MODEL_IDS

    return MODEL_IDS[kind]


def row_key(args, prompt_id: str) -> str:
    return f"e31|{args.model}|{args.dtype}|n{args.new_tokens}|L{args.max_len}|{prompt_id}"


def variant_specs(kind: str, alt_chunks: list[int]) -> list[dict[str, Any]]:
    """The reference, the alternatives and the reference rerun of one model."""
    ref = REFERENCE_CHUNK[kind]
    specs = [{"name": "ref", "role": "reference", "chunk": ref}]
    specs += [{"name": f"c{c}", "role": "alt", "chunk": int(c)} for c in alt_chunks if c != ref]
    specs.append({"name": "rerun", "role": "rerun", "chunk": ref})
    return specs


# --------------------------------------------------------------------------- prompt plan
def stratified_order(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic order stratified over length buckets and regimes (see the module docstring)."""
    by_b: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for it in items:
        by_b[it["bucket"]][it["regime"]].append(it)
    for regs in by_b.values():
        for lst in regs.values():
            lst.sort(key=lambda r: hashlib.sha1(r["prompt_id"].encode()).hexdigest())
    asc = sorted(by_b)
    cycle, lo, hi = [], 0, len(asc) - 1
    while lo <= hi:
        cycle.append(asc[hi])
        if lo != hi:
            cycle.append(asc[lo])
        lo, hi = lo + 1, hi - 1
    ptr = {b: 0 for b in asc}
    out: list[dict[str, Any]] = []
    while len(out) < len(items):
        for b in cycle:
            regs = [r for r in sorted(by_b[b]) if by_b[b][r]]
            if not regs:
                continue
            r = regs[ptr[b] % len(regs)]
            ptr[b] += 1
            out.append(by_b[b][r].pop(0))
    return out


def build_plan(tok: Any, args, log=print) -> list[dict[str, Any]]:
    """Tokenize the capture corpus as the capture did, crop to --max-len, take the stratified subset."""
    from src.experiments.tpds.capture_traces import select_corpus

    corpus = select_corpus(argparse.Namespace(**CAPTURE_CORPUS, limit_prompts=None))
    items = []
    for it in corpus:
        max_tok = min(it["max_tokens"] or CAPTURE_MAX_TOKENS, CAPTURE_MAX_TOKENS)
        ids = [int(i) for i in tok(it["text"])["input_ids"][:max_tok]]
        L_cap = len(ids)
        ids = ids[: args.max_len]
        if len(ids) < 2:
            log(f"skipping {it['id']}: {len(ids)} tokens")
            continue
        items.append({"prompt_id": it["id"], "regime": it["regime"], "source": it["source"],
                      "fold": prompt_fold(it["id"]), "ids": ids, "L": len(ids), "L_captured": L_cap,
                      "bucket": _length_bucket(len(ids))})
    order = stratified_order(items)
    return order[: args.n_prompts] if args.n_prompts else order


def plan_summary(plan: list[dict[str, Any]]) -> dict[str, Any]:
    return {"n": len(plan), "prompt_ids_sha1": hashlib.sha1("|".join(p["prompt_id"] for p in plan).encode()).hexdigest()[:16],
            "by_bucket": {str(b): n for b, n in sorted(Counter(p["bucket"] for p in plan).items())},
            "by_regime": dict(sorted(Counter(p["regime"] for p in plan).items())),
            "n_capped": sum(1 for p in plan if p["L"] < p["L_captured"]),
            "L_min": min((p["L"] for p in plan), default=None), "L_max": max((p["L"] for p in plan), default=None)}


# --------------------------------------------------------------------------- model runners
class _Runner:
    """Common interface: configure(variant), prefill(ids) -> (logits, cache), step(tok, cache) -> logits."""

    kind = ""

    def __init__(self, args, log):
        import torch

        self.args, self.log = args, log
        self.device = torch.device("cuda")
        self.dtype = getattr(torch, args.dtype)
        self.model_id = args.model_id or default_model_id(args.model)
        self.info: dict[str, Any] = {}
        self.variants = variant_specs(args.model, args.chunks)

    @property
    def reference(self) -> dict[str, Any]:
        return self.variants[0]

    @property
    def others(self) -> list[dict[str, Any]]:
        return self.variants[1:]

    def close(self) -> None:
        pass


class Mamba2Runner(_Runner):
    kind = "mamba2"

    def __init__(self, args, log):
        super().__init__(args, log)
        self.info["pinned_kernels"] = _pin_installed_kernels(log)
        from transformers import AutoTokenizer, Mamba2ForCausalLM
        from transformers.models.mamba2 import modeling_mamba2 as m2

        self.m2 = m2
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = Mamba2ForCausalLM.from_pretrained(self.model_id, dtype=self.dtype).to(self.device).eval()
        self.mixers = [layer.mixer for layer in self.model.backbone.layers]
        self.n_layers = len(self.mixers)
        self.fast_path = bool(getattr(m2, "is_fast_path_available", False))
        default_chunk = int(self.model.config.chunk_size)
        self.info.update({"fast_path": self.fast_path, "config_chunk_size": default_chunk, "n_layers": self.n_layers,
                          "ssd_kernel_module": getattr(getattr(m2, "mamba_chunk_scan_combined", None), "__module__",
                                                       None)})
        if default_chunk != REFERENCE_CHUNK["mamba2"]:
            log(f"WARNING: the checkpoint's chunk_size is {default_chunk}, not {REFERENCE_CHUNK['mamba2']}; "
                f"the reference stays {REFERENCE_CHUNK['mamba2']}")
        if not self.fast_path:
            if not args.allow_slow_path:
                raise SystemExit("transformers' Mamba-2 fast path (mamba_ssm + causal_conv1d) is unavailable; "
                                 "pass --allow-slow-path only for a plumbing smoke test")
            log("WARNING: Mamba-2 fast path unavailable; the portable PyTorch path runs (it honours chunk_size too)")
        self.hook_records: list[tuple[int, int, int]] = []
        self.kernel_records: list[int] = []
        for m in self.mixers:
            m.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        if self.fast_path:
            orig = m2.mamba_chunk_scan_combined

            def wrapped(*a, **kw):
                self.kernel_records.append(int(kw["chunk_size"]) if "chunk_size" in kw else int(a[5]))
                return orig(*a, **kw)

            wrapped._e31_orig = orig  # type: ignore[attr-defined]
            m2.mamba_chunk_scan_combined = wrapped

    def _pre_hook(self, module, args, kwargs):
        hs = args[0] if args else kwargs.get("hidden_states")
        if hs is not None and hs.shape[1] > 1:
            self.hook_records.append((int(module.layer_idx), int(hs.shape[1]), int(module.chunk_size)))
        return None

    def configure(self, variant):
        for m in self.mixers:
            m.chunk_size = int(variant["chunk"])
        self.hook_records.clear()
        self.kernel_records.clear()

    def prefill(self, ids):
        out = self.model(input_ids=ids, use_cache=True, logits_to_keep=1)
        return out.logits[0, -1].float(), out.cache_params

    def step(self, tok, cache):
        out = self.model(input_ids=tok, cache_params=cache, use_cache=True)
        return out.logits[0, -1].float()

    def prefill_check(self, variant, L: int) -> dict[str, Any]:
        c = int(variant["chunk"])
        hooks, kern = list(self.hook_records), list(self.kernel_records)
        self.hook_records.clear()
        self.kernel_records.clear()
        ok = len(hooks) == self.n_layers and all(r[1] == L and r[2] == c for r in hooks)
        if self.fast_path:
            ok &= len(kern) == self.n_layers and all(k == c for k in kern)
        if not ok:
            raise RuntimeError(f"Mamba-2 chunk {c} not honoured in prefill (L={L}): mixer records {hooks[:3]}..., "
                               f"kernel chunks {Counter(kern)}")
        counts = Counter(kern) if self.fast_path else Counter(r[2] for r in hooks)
        return {"path": "ssd_triton" if self.fast_path else "torch_forward",
                "chunk_counts": {str(k): v for k, v in sorted(counts.items())}}


class Mamba1Runner(_Runner):
    kind = "mamba1"

    def __init__(self, args, log):
        super().__init__(args, log)
        from src.experiments import run_integrated_end_to_end as rie
        from src.experiments.tpds import e10_routed_interleaved as e10

        self.e10, self.rie = e10, rie
        os.environ.setdefault("COREY_SELECTIVE_SCAN_BACKEND", "auto")
        self.info["adapter"] = e10.install_chunk_adapter(args.chunk_module, log)
        os.environ["COREY_SELECTIVE_SCAN_DISPATCH_MODULE"] = args.dispatch_module
        rie._prefer_installed_mamba_kernels()   # binds the dispatch module into transformers' Mamba module
        from transformers import AutoTokenizer, MambaForCausalLM
        from transformers.models.mamba import modeling_mamba as _mm

        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = MambaForCausalLM.from_pretrained(self.model_id, dtype=self.dtype).to(self.device).eval()
        self.n_layers = len(self.model.backbone.layers)
        missing = [n for n in e10.FAST_PATH_NAMES if getattr(_mm, n, None) is None]
        self.fast_path = not missing
        dispatch_info = rie.STATE.dispatch_info
        self.info.update({"fast_path": self.fast_path, "fast_path_missing": missing, "dispatch_module": args.dispatch_module,
                          "dispatch_info": dispatch_info, "n_layers": self.n_layers})
        routed_ok = (self.fast_path and self.info["adapter"]["available"]
                     and bool(dispatch_info and dispatch_info.get("runtime_cuda_available")))
        if routed_ok:
            dispatch = importlib.import_module(args.dispatch_module)
            valid = sorted(getattr(dispatch, "VALID_CHUNKS", MAMBA1_ROUTED_CHUNKS))
            bad = sorted(set(args.chunks) - set(valid))
            if bad:
                raise SystemExit(f"chunks {bad} are not supported by the runtime-chunk scan (valid: {valid})")
            self.info["routed_scan_validation"] = e10.validate_routed_scan(dispatch, valid, log)
            if not self.info["routed_scan_validation"]["passed"]:
                raise SystemExit("the runtime-chunk scan does not match the stock scan (see the log)")
        elif args.allow_slow_path:
            log(f"WARNING: routed scan unavailable (fast path {self.fast_path}, adapter "
                f"{self.info['adapter'].get('reason')}); only the reference and its rerun run")
            self.variants = [v for v in self.variants if v["role"] != "alt"]
        else:
            raise SystemExit(f"the Mamba-1 routed scan is unavailable (fast path missing {missing}, adapter "
                             f"{self.info['adapter'].get('reason')}, dispatch {dispatch_info}); "
                             "--allow-slow-path runs only the reference and its rerun")
        cfg_ns = argparse.Namespace(static_chunks=list(args.chunks), adaptive_chunk_min=None,
                                    adaptive_chunk_max=None, entropy_stride=1, configs="passive,static",
                                    hook_controls=False, random_seed=0)
        cfgs = {c.name: c for c in e10.build_configs(cfg_ns)}
        self.cfg_passive = cfgs["passive"]
        self.cfg_static = {c: cfgs[f"static_chunk_{c}"] for c in args.chunks}
        self.apply_ns = argparse.Namespace(num_bins=256, guard_fallback_chunk=512, guard_min_delta_buckets=2)
        self.original_forward = rie.install_patch()

    def configure(self, variant):
        cfg = self.cfg_passive if variant["chunk"] is None else self.cfg_static[int(variant["chunk"])]
        self.e10.apply_config(cfg, self.apply_ns, None, 0)
        # Route every prefill call (the routed study used 16); decode calls (one token) never record.
        self.rie.STATE.min_seq_len = 2

    def prefill(self, ids):
        out = self.model(input_ids=ids, use_cache=True, logits_to_keep=1)
        return out.logits[0, -1].float(), out.cache_params

    def step(self, tok, cache):
        out = self.model(input_ids=tok, cache_params=cache, use_cache=True)
        return out.logits[0, -1].float()

    def prefill_check(self, variant, L: int) -> dict[str, Any]:
        st = self.rie.STATE
        recs = list(st.records)
        st.records.clear()
        if variant["chunk"] is None:
            if recs:
                raise RuntimeError(f"the stock reference recorded {len(recs)} routed scan calls")
            return {"path": "stock" if self.fast_path else "hf_slow_path",
                    "chunk_counts": {str(stock_chunk(L)): self.n_layers} if self.fast_path else {}}
        c = int(variant["chunk"])
        ok = (st.chunk_size_kwarg_supported is True and len(recs) == self.n_layers
              and all(r.get("routed") and int(r["chunk"]) == c and int(r["seq_len"]) == L for r in recs))
        if not ok:
            raise RuntimeError(f"chunk {c} was not routed into every prefill scan (L={L}; honoured "
                               f"{st.chunk_size_kwarg_supported}; {len(recs)} records)")
        return {"path": "routed", "chunk_counts": {str(c): len(recs)}}

    def close(self):
        self.rie.STATE.enabled = False
        self.rie.restore(self.original_forward)


class GdnRunner(_Runner):
    kind = "gdn"

    def __init__(self, args, log):
        super().__init__(args, log)
        from src.experiments.tpds.capture_gdn_traces import gdn_layers, load_model

        self.tok, self.model = load_model(self.model_id, self.dtype)
        import fla.layers.gated_deltanet as layer_module  # type: ignore

        self.layer_module = layer_module
        self.n_layers = len(gdn_layers(self.model))
        self.chunk = REFERENCE_CHUNK["gdn"]
        self.records: list[tuple[str, int, int | None]] = []
        self.orig = {n: getattr(layer_module, n) for n in ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule")}
        orig_chunk, orig_fr = self.orig["chunk_gated_delta_rule"], self.orig["fused_recurrent_gated_delta_rule"]

        def _qlen(a, kw):
            q = kw.get("q", a[0] if a else None)
            return int(q.shape[1]) if q is not None else -1

        def chunk_wrapped(*a, **kw):
            kw["chunk_size"] = self.chunk
            if _qlen(a, kw) > 1:
                self.records.append(("chunk", _qlen(a, kw), self.chunk))
            return orig_chunk(*a, **kw)

        def fr_wrapped(*a, **kw):
            if _qlen(a, kw) > 1:
                self.records.append(("fused_recurrent", _qlen(a, kw), None))
            return orig_fr(*a, **kw)

        layer_module.chunk_gated_delta_rule = chunk_wrapped
        layer_module.fused_recurrent_gated_delta_rule = fr_wrapped
        self.info.update({"n_gdn_layers": self.n_layers, "n_layers": len(self.model.model.layers),
                          "fla_layer_module": getattr(layer_module, "__file__", None)})

    def configure(self, variant):
        self.chunk = int(variant["chunk"])
        self.records.clear()

    def prefill(self, ids):
        out = self.model(input_ids=ids, use_cache=True, logits_to_keep=1)
        return out.logits[0, -1].float(), out.past_key_values

    def step(self, tok, cache):
        out = self.model(input_ids=tok, past_key_values=cache, use_cache=True, logits_to_keep=1)
        return out.logits[0, -1].float()

    def prefill_check(self, variant, L: int) -> dict[str, Any]:
        recs = list(self.records)
        self.records.clear()
        c = int(variant["chunk"])
        if len(recs) != self.n_layers or any(r[1] != L for r in recs):
            raise RuntimeError(f"GDN prefill recorded {len(recs)} kernel calls for {self.n_layers} layers (L={L})")
        paths = Counter(r[0] for r in recs)
        if paths.get("chunk", 0) and any(r[2] != c for r in recs if r[0] == "chunk"):
            raise RuntimeError(f"GDN chunk {c} not injected: {recs[:3]}")
        return {"path": "+".join(sorted(paths)),
                "chunk_counts": {str(c): paths["chunk"]} if paths.get("chunk") else {}}

    def close(self):
        for n, f in self.orig.items():
            setattr(self.layer_module, n, f)


RUNNERS = {"mamba2": Mamba2Runner, "mamba1": Mamba1Runner, "gdn": GdnRunner}


def _pin_installed_kernels(log) -> dict[str, Any]:
    """Make transformers' lazy kernel loader use the installed mamba_ssm / causal_conv1d packages.

    transformers 5.x would fetch Hub kernels instead when the ``kernels`` package is installed; the
    installed packages are the ones whose kernels the sweeps timed. No-op where they are missing.
    """
    out: dict[str, Any] = {}
    try:
        from transformers.integrations import hub_kernels
    except Exception as exc:  # pragma: no cover
        return {"error": repr(exc)}
    for name, mod_name, subs in (("mamba-ssm", "mamba_ssm", ("mamba_ssm.ops.triton.ssd_combined",
                                                             "mamba_ssm.ops.triton.selective_state_update")),
                                 ("causal-conv1d", "causal_conv1d", ())):
        try:
            mod = importlib.import_module(mod_name)
            for s in subs:
                importlib.import_module(s)
            hub_kernels._KERNEL_MODULE_MAPPING[name] = mod
            out[name] = getattr(mod, "__file__", None)
        except Exception as exc:
            out[name] = f"unavailable: {type(exc).__name__}"
    log(f"kernel modules: {out}")
    return out


# --------------------------------------------------------------------------- per-prompt measurement
def run_sequence(runner: _Runner, variant: dict[str, Any], ids, n_new: int, forced=None):
    """Prefill at the variant's chunk, then ``n_new`` decode steps (fed ``forced`` tokens or own argmax).

    Returns (logits [n_new + 1, V] float32, fed token ids [n_new], prefill check).
    """
    import torch

    runner.configure(variant)
    logits0, cache = runner.prefill(ids)
    check = runner.prefill_check(variant, int(ids.shape[1]))
    out = torch.empty((n_new + 1, logits0.numel()), dtype=torch.float32, device=logits0.device)
    out[0] = logits0
    cur, fed = logits0, []
    for k in range(n_new):
        nxt = forced[:, k:k + 1] if forced is not None else cur.argmax(-1).view(1, 1)
        fed.append(nxt)
        cur = runner.step(nxt, cache)
        out[k + 1] = cur
    tokens = torch.cat(fed, dim=1)[0].tolist() if fed else []
    return out, [int(t) for t in tokens], check


def compare(ref_lp, ref_p, ref_logits, ref_top1, alt_logits) -> dict[str, list]:
    import torch

    alt_lp = torch.log_softmax(alt_logits.double(), dim=-1)
    kl = torch.where(ref_p > 0, ref_p * (ref_lp - alt_lp), torch.zeros_like(ref_p)).sum(-1)
    diff = (alt_logits - ref_logits).abs().amax(-1)
    alt_top1 = alt_logits.argmax(-1)
    return {"kl": kl.tolist(), "max_abs_logit_diff": diff.tolist(),
            "top1_agree": (alt_top1 == ref_top1).int().tolist(), "alt_top1": alt_top1.tolist()}


def first_mismatch(a: list[int], b: list[int]) -> int | None:
    return next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)


def process_prompt(runner: _Runner, it: dict[str, Any], args) -> dict[str, Any]:
    import torch

    n = args.new_tokens
    ids = torch.tensor([it["ids"]], dtype=torch.long, device=runner.device)
    L = int(ids.shape[1])
    t0 = time.perf_counter()
    ref_logits, ref_tokens, ref_check = run_sequence(runner, runner.reference, ids, n)
    ref_lp = torch.log_softmax(ref_logits.double(), dim=-1)
    ref_p = ref_lp.exp()
    ref_top1 = ref_logits.argmax(-1)
    topv, topi = ref_lp.topk(min(args.top_k, ref_lp.shape[-1]), dim=-1)
    entropy = -torch.where(ref_p > 0, ref_p * ref_lp, torch.zeros_like(ref_p)).sum(-1)
    forced = torch.tensor([ref_tokens], dtype=torch.long, device=runner.device)
    variants: dict[str, Any] = {}
    for v in runner.others:
        alt_logits, _, check = run_sequence(runner, v, ids, n, forced=forced)
        cmp = compare(ref_lp, ref_p, ref_logits, ref_top1, alt_logits)
        p = first_mismatch(cmp["alt_top1"][:n], ref_tokens)
        free: dict[str, Any]
        if p is None and args.free_run == "on-divergence":
            free = {"source": "teacher_forced", "first_div": None, "n_match": n, "tokens": None,
                    "prefix_consistent": None}
        else:
            fr_logits, fr_tokens, _ = run_sequence(runner, v, ids, n)
            upto = (p + 1) if p is not None else n + 1
            fd = first_mismatch(fr_tokens, ref_tokens)
            free = {"source": "run", "first_div": fd, "n_match": sum(1 for x, y in zip(fr_tokens, ref_tokens) if x == y),
                    "tokens": fr_tokens if fd is not None else None,
                    "prefix_consistent": bool(torch.equal(fr_logits[:upto], alt_logits[:upto])) and fd == p}
            del fr_logits
        variants[v["name"]] = {
            "role": v["role"], "chunk": v["chunk"], "scan": check,
            "kl": [_g(x) for x in cmp["kl"]], "max_abs_logit_diff": [_g(x) for x in cmp["max_abs_logit_diff"]],
            "top1_agree": cmp["top1_agree"], "alt_top1": cmp["alt_top1"],
            "bitwise_identical": all(x == 0.0 for x in cmp["max_abs_logit_diff"]),
            "first_div_tf": p, "free_run": free}
        del alt_logits
    row = {"kind": "prompt", "key": row_key(args, it["prompt_id"]), "experiment": "E31", "model": args.model,
           "dtype": args.dtype, "new_tokens": n, "max_len": args.max_len, "prompt_id": it["prompt_id"],
           "regime": it["regime"], "source": it["source"], "fold": it["fold"], "L": L, "L_captured": it["L_captured"],
           "capped": L < it["L_captured"], "L_bucket": it["bucket"], "ids_sha1": _sha1(it["ids"])[:16],
           "stock_chunk": stock_chunk(L) if args.model == "mamba1" else None,
           "reference": {"chunk": runner.reference["chunk"], "scan": ref_check, "tokens": ref_tokens,
                         "top_ids": topi.tolist(), "top_logprobs": [[round(float(x), 5) for x in r] for r in topv.tolist()],
                         "entropy": [round(float(x), 5) for x in entropy.tolist()]},
           "variants": variants, "elapsed_s": round(time.perf_counter() - t0, 3),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3), "utc": utc_now()}
    return row


# --------------------------------------------------------------------------- run
@contextlib.contextmanager
def exclusive_lock(args, log):
    path = Path(args.log_dir) / f"e31_divergence_{args.tag}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise SystemExit(f"another E31 process writes e31_divergence_{args.tag}.jsonl (lock {path}); "
                         "wait for it or use a different --tag")
    try:
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def run(args, log) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (use --dry-run or --download-only on a machine without a GPU)")
    jsonl = args.out_dir / f"e31_divergence_{args.tag}.jsonl"
    check_disk(args.out_dir, need_gb=0.2, need_inodes=100, log=log)
    with exclusive_lock(args, log):
        torch.manual_seed(args.seed)
        runner = RUNNERS[args.model](args, log)
        try:
            plan = build_plan(runner.tok, args, log)
            meta = env_metadata()
            meta.update({
                "experiment": "E31", "model": args.model, "model_id": runner.model_id, "dtype": args.dtype,
                "reference_chunk": REFERENCE_CHUNK[args.model], "variants": runner.variants, "runner": runner.info,
                "new_tokens": args.new_tokens, "max_len": args.max_len, "top_k": args.top_k,
                "free_run": args.free_run, "plan": plan_summary(plan), "capture_corpus": CAPTURE_CORPUS,
                "torch_flags": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
                "hf_home": os.environ.get("HF_HOME"), "hf_endpoint": os.environ.get("HF_ENDPOINT"),
                "args": {k: str(v) for k, v in vars(args).items()}})
            w = JsonlWriter(jsonl, meta=meta)
            todo = [it for it in plan if row_key(args, it["prompt_id"]) not in w.done]
            log(f"E31 {args.model} ({runner.model_id}) {args.dtype} on {meta.get('gpu')}: {len(plan)} prompts, "
                f"{len(plan) - len(todo)} done, {len(todo)} to run; variants "
                f"{[v['name'] for v in runner.variants]}; fast path {runner.info.get('fast_path')}")
            if args.max_prompts is not None:
                todo = todo[: args.max_prompts]
            with torch.inference_mode():
                for i, it in enumerate(todo):
                    row = process_prompt(runner, it, args)
                    w.write(row)
                    parts = []
                    for name, v in row["variants"].items():
                        fd = v["free_run"]["first_div"]
                        parts.append(f"{name}:kl0={v['kl'][0]:.1e},klmax={max(v['kl']):.1e},"
                                     f"div={'-' if fd is None else fd}")
                    log(f"[{i + 1}/{len(todo)}] {it['prompt_id']} L={row['L']} {row['elapsed_s']:.1f}s "
                        f"peak {row['peak_vram_gb']:.2f} GB | " + " ".join(parts))
            w.close()
        finally:
            runner.close()
    try:
        analyze(args)
    except Exception as exc:  # the analysis is CPU-side and can be re-run with --analyze
        log(f"analysis failed: {exc!r}")


# --------------------------------------------------------------------------- dry run / download
def dry_run(args, log) -> None:
    from transformers import AutoTokenizer

    model_id = args.model_id or default_model_id(args.model)
    tok = AutoTokenizer.from_pretrained(model_id)
    plan = build_plan(tok, args, log)
    specs = variant_specs(args.model, args.chunks)
    print(f"E31 plan (dry run): model {args.model} ({model_id}), dtype {args.dtype}, tag {args.tag}, "
          f"out {args.out_dir / f'e31_divergence_{args.tag}.jsonl'}")
    print(f"  variants: {[(v['name'], v['chunk'] if v['chunk'] is not None else 'stock') for v in specs]}")
    print(f"  {args.new_tokens} new tokens, top-k {args.top_k}, max length {args.max_len}, free run {args.free_run}")
    s = plan_summary(plan)
    print(f"  {s['n']} prompts (L {s['L_min']}..{s['L_max']}, {s['n_capped']} cropped to {args.max_len}); "
          f"by bucket {s['by_bucket']}")
    print(f"  by regime {s['by_regime']}")
    table = None
    try:
        table = load_table(args.model, args.tables[args.model])
    except Exception as exc:
        print(f"  per-shape table unavailable: {exc!r}")
    print(f"  {'#':>4s} {'prompt':34s} {'regime':28s} {'L':>5s} {'L_cap':>6s} {'bucket':>6s} {'table':>6s}")
    for i, p in enumerate(plan):
        tc = table_chunk(table, p["L"])[0] if table else None
        if i < args.show or i >= len(plan) - 3:
            print(f"  {i:4d} {p['prompt_id']:34s} {p['regime']:28s} {p['L']:5d} {p['L_captured']:6d} {p['bucket']:6d} "
                  f"{str(tc):>6s}")
        elif i == args.show:
            print("  ...")
    if table:
        print(f"  per-shape table ({table['source']}): " + ", ".join(f"{b}:{v['chunk']}" for b, v in table["buckets"].items()))
    n_runs = 1 + (len(specs) - 1) + args.est_div_share * (len(specs) - 2)
    per_run = args.est_prefill_s + args.new_tokens * args.est_step_ms / 1000.0
    total = len(plan) * n_runs * per_run
    print(f"  estimate: {n_runs:.1f} runs per prompt x {per_run:.2f} s (prefill {args.est_prefill_s:.2f} s + "
          f"{args.new_tokens} decode steps x {args.est_step_ms:.0f} ms) -> {total / 60:.0f} min")
    jsonl = args.out_dir / f"e31_divergence_{args.tag}.jsonl"
    if jsonl.exists():
        done = {r.get("key") for r in read_jsonl(jsonl)}
        print(f"  existing result file: {sum(1 for p in plan if row_key(args, p['prompt_id']) in done)} of {len(plan)} done")


def download_only(args, log) -> None:
    model_id = args.model_id or default_model_id(args.model)
    if Path(model_id).is_dir():
        files = sorted(p.name for p in Path(model_id).iterdir())
        ok = "config.json" in files and any(f.endswith(".safetensors") for f in files)
        log(f"{model_id} is a local directory ({'complete' if ok else 'INCOMPLETE'}): {files}")
        if not ok:
            raise SystemExit(1)
        return
    from huggingface_hub import constants, snapshot_download

    cache = Path(constants.HF_HUB_CACHE)
    log(f"downloading {model_id} into {cache} (HF_ENDPOINT={os.environ.get('HF_ENDPOINT')})")
    check_disk(cache, need_gb=CHECKPOINT_GB[args.model] * 1.1 + 0.5, need_inodes=200, log=log)
    path = Path(snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model",
                                                            "*.txt"]))
    files = {p.name: p.stat().st_size for p in path.iterdir() if p.is_file()}
    log(f"snapshot {path}: " + ", ".join(f"{k} ({v / 2**20:.1f} MB)" for k, v in sorted(files.items())))
    if not any(k.endswith(".safetensors") for k in files):
        raise SystemExit("no safetensors weights in the snapshot")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(path))
    log(f"tokenizer OK ({type(tok).__name__}, vocab {len(tok)})")


# --------------------------------------------------------------------------- analysis (CPU only)
def load_table(model: str, path: Path, batch: int = 1) -> dict[str, Any] | None:
    """Per-length-bucket chunk table of one batch size (default 1) from a stored sweep (lowest mean latency
    over the bucket)."""
    from src.experiments.tpds.analysis import chunk_columns, load_e2, round_median_view

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = load_e2([path])
    if df.empty:
        return None
    view = "single round (mean of repeats)"
    if "oracle_chunk_med" in df.columns:
        df = round_median_view(df)
        view = "round medians"
    prefix = {"mamba1": "mamba1|", "mamba2": "mamba2|", "gdn": "gdn|"}[model]
    df = df[(df["batch"] == batch) & (df["layer"] >= 0) & df["key"].astype(str).str.startswith(prefix)]
    chunks = chunk_columns(df)
    buckets: dict[str, Any] = {}
    for b, g in df.groupby("L_bucket"):
        full = [c for c in chunks if g[f"lat_c{c}_mean"].notna().all()] or chunks
        means = {c: float(g[f"lat_c{c}_mean"].mean()) for c in full}
        best = min(means, key=lambda c: (means[c], c))
        oc = Counter(int(x) for x in g["oracle_chunk"].dropna())
        top = max(oc.values()) if oc else 0
        mode = min((c for c in oc if oc[c] == top), key=lambda c: (means.get(c, float("inf")), c)) if oc else best
        buckets[str(int(b))] = {"chunk": int(best), "oracle_mode": int(mode), "n_calls": int(len(g)),
                                "mean_ms": {str(c): round(v, 5) for c, v in means.items()},
                                "oracle_counts": {str(c): n for c, n in sorted(oc.items())}}
    return {"source": path.name, "latency": view, "batch": batch,
            "selection": f"lowest mean latency over the bucket's batch-{batch} calls",
            "chunks": chunks, "buckets": buckets}


def table_chunk(table: dict[str, Any], L: int, field: str = "chunk") -> tuple[int, int, bool]:
    """(chunk, table bucket used, fallback?) for a prompt of length L."""
    b = _length_bucket(L)
    bk = {int(k): v for k, v in table["buckets"].items()}
    if b in bk:
        return bk[b][field], b, False
    near = min(bk, key=lambda x: (abs(math.log2(x) - math.log2(b)), x))
    return bk[near][field], near, True


def _stats(vals: list[float]) -> dict[str, Any] | None:
    v = [float(x) for x in vals if x is not None and not math.isnan(float(x))]
    if not v:
        return None
    return {"n": len(v), "mean": statistics.fmean(v), "median": statistics.median(v), "min": min(v), "max": max(v)}


def summarize(recs: list[tuple[dict[str, Any], dict[str, Any]]], n_new: int) -> dict[str, Any]:
    """Aggregate per-prompt variant records [(row, variant record)]."""
    if not recs:
        return {"n_prompts": 0}
    fds = [v["free_run"]["first_div"] for _, v in recs]
    div = [f for f in fds if f is not None]
    kl0 = [v["kl"][0] for _, v in recs]
    kl_all = [x for _, v in recs for x in v["kl"]]
    kl_prompt_mean = [statistics.fmean(v["kl"]) for _, v in recs]
    d0 = [v["max_abs_logit_diff"][0] for _, v in recs]
    d_prompt_max = [max(v["max_abs_logit_diff"]) for _, v in recs]
    agree0 = [v["top1_agree"][0] for _, v in recs]
    agree_all = [x for _, v in recs for x in v["top1_agree"]]
    out = {
        "n_prompts": len(recs),
        "diverged_share": len(div) / len(recs), "n_diverged": len(div),
        "first_div": ({"median": statistics.median(div), "min": min(div), "mean": statistics.fmean(div)} if div else None),
        "n_match_mean": statistics.fmean(v["free_run"]["n_match"] for _, v in recs),
        "kl_pos0": _stats(kl0), "kl_all_positions": _stats(kl_all), "kl_prompt_mean": _stats(kl_prompt_mean),
        "max_abs_logit_diff_pos0": _stats(d0), "max_abs_logit_diff_prompt_max": _stats(d_prompt_max),
        "top1_agree_pos0": statistics.fmean(agree0), "top1_agree_all": statistics.fmean(agree_all),
        "bitwise_identical_share": statistics.fmean(1.0 if v["bitwise_identical"] else 0.0 for _, v in recs),
        "pos0_identical_share": statistics.fmean(1.0 if v["max_abs_logit_diff"][0] == 0 else 0.0 for _, v in recs),
        "free_runs": sum(1 for _, v in recs if v["free_run"]["source"] == "run"),
        "prefix_inconsistent": sum(1 for _, v in recs if v["free_run"].get("prefix_consistent") is False),
        "n_nan_kl": sum(1 for x in kl_all if x is None or math.isnan(float(x))),
    }
    by_b: dict[int, list] = defaultdict(list)
    for r, v in recs:
        by_b[int(r["L_bucket"])].append(v)
    out["by_bucket"] = {str(b): {"n": len(vs), "diverged_share": statistics.fmean(
        1.0 if v["free_run"]["first_div"] is not None else 0.0 for v in vs),
        "kl_pos0_mean": statistics.fmean(v["kl"][0] for v in vs),
        "kl_mean": statistics.fmean(x for v in vs for x in v["kl"]),
        "top1_agree_all": statistics.fmean(x for v in vs for x in v["top1_agree"])} for b, vs in sorted(by_b.items())}
    return out


def analyze(args) -> dict[str, Any]:
    tags = [t for t in (args.input_tags or args.tag).split(",") if t]
    rows, sources = [], []
    for t in tags:
        path = args.out_dir / f"e31_divergence_{t}.jsonl"
        sources.append(path.name)
        rows += [r for r in read_jsonl(path) if r.get("kind") == "prompt"]
    rows = dedupe_by_key(rows)
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["dtype"], int(r["new_tokens"]), int(r["max_len"]))].append(r)
    tables: dict[str, Any] = {}
    for model in sorted({g[0] for g in groups}):
        try:
            tables[model] = load_table(model, args.tables[model])
        except Exception as exc:
            tables[model] = None
            print(f"per-shape table for {model} unavailable: {exc!r}")
    res: dict[str, Any] = {
        "experiment": "E31-analysis", "tag": args.tag, "sources": sources, "n_rows": len(rows), "utc": utc_now(),
        "definitions": {
            "position": "0 = the prefill's last position; k = after feeding the k-th reference token",
            "kl": "KL(ref || variant) of the next-token distributions over the full vocabulary, teacher-forced",
            "first_div": "first generated position whose free-running greedy token differs from the reference "
                         "(0-based); prompts without divergence are excluded from its median/min",
            "diverged_share": "share of prompts whose free-running greedy output differs within the new tokens",
            "rerun": "the reference chunk run a second time (run-to-run nondeterminism floor)",
            "per_shape": "per prompt the chunk of the per-length-bucket table (reference chunk -> the rerun)",
            "per_shape_oracle_mode": "the same with the per-bucket mode of the per-call oracle chunk",
        },
        "tables": tables, "groups": {},
    }
    for (model, dtype, n_new, max_len), rs in sorted(groups.items()):
        gid = f"{model}|{dtype}" + ("" if (n_new, max_len) == (32, 2048) else f"|n{n_new}|L{max_len}")
        names = sorted({n for r in rs for n in r["variants"]}, key=_variant_sort_key)
        g: dict[str, Any] = {"model": model, "dtype": dtype, "new_tokens": n_new, "max_len": max_len,
                             "n_prompts": len(rs), "reference_chunk": REFERENCE_CHUNK.get(model),
                             "L": _stats([r["L"] for r in rs]), "variants": {}}
        for name in names:
            recs = [(r, r["variants"][name]) for r in rs if name in r["variants"]]
            s = summarize(recs, n_new)
            s["chunk"] = recs[0][1]["chunk"] if recs else None
            if model == "mamba1" and name.startswith("c"):
                same = [(r, v) for r, v in recs if r.get("stock_chunk") == v["chunk"]]
                s["stock_equivalent"] = {"n": len(same), "bitwise_identical": sum(1 for _, v in same
                                                                                  if v["bitwise_identical"])}
            g["variants"][name] = s
        table = tables.get(model)
        if table:
            for label, field in (("per_shape", "chunk"), ("per_shape_oracle_mode", "oracle_mode")):
                recs, chosen, fallback, missing = [], Counter(), 0, 0
                for r in rs:
                    c, _, fb = table_chunk(table, r["L"], field)
                    fallback += int(fb)
                    name = "rerun" if c == REFERENCE_CHUNK.get(model) else f"c{c}"
                    if name not in r["variants"]:
                        missing += 1
                        continue
                    chosen[c] += 1
                    recs.append((r, r["variants"][name]))
                s = summarize(recs, n_new)
                s.update({"chosen_chunks": {str(c): n for c, n in sorted(chosen.items())},
                          "reference_chunk_share": (chosen.get(REFERENCE_CHUNK.get(model), 0) / len(recs)) if recs else None,
                          "bucket_fallbacks": fallback, "prompts_missing_variant": missing, "table": table["source"]})
                g[label] = s
        res["groups"][gid] = g
    out = args.out_dir / f"e31_divergence_{args.tag}.json"
    write_json(out, res)
    _print_summary(res)
    print(f"analysis written to {out}")
    return res


def _variant_sort_key(name: str):
    if name == "rerun":
        return (1, 0)
    try:
        return (0, int(name[1:]))
    except ValueError:
        return (2, name)


def _fmt(x, spec=".2e"):
    return "n/a" if x is None else format(x, spec)


def _print_summary(res: dict[str, Any]) -> None:
    for gid, g in res["groups"].items():
        ref = g["reference_chunk"] if g["reference_chunk"] is not None else "stock"
        print(f"\n{gid}: {g['n_prompts']} prompts, reference chunk {ref}, {g['new_tokens']} new tokens")
        print(f"  {'variant':22s} {'n':>4s} {'div%':>6s} {'med1st':>6s} {'min1st':>6s} {'KL@0 mean':>10s} "
              f"{'KL mean':>9s} {'KL max':>9s} {'max|dlogit|':>11s} {'top1%':>6s} {'bitwise%':>8s}")
        rows = [(f"{n} (chunk {s.get('chunk') if s.get('chunk') is not None else 'stock'})", s)
                for n, s in g["variants"].items()]
        for label in ("per_shape", "per_shape_oracle_mode"):
            if label in g:
                rows.append((label, g[label]))
        for label, s in rows:
            if not s.get("n_prompts"):
                print(f"  {label:22s} {0:4d}")
                continue
            fd = s.get("first_div") or {}
            print(f"  {label:22s} {s['n_prompts']:4d} {100 * s['diverged_share']:6.1f} "
                  f"{str(fd.get('median', '-')):>6s} {str(fd.get('min', '-')):>6s} "
                  f"{_fmt((s['kl_pos0'] or {}).get('mean')):>10s} {_fmt((s['kl_all_positions'] or {}).get('mean')):>9s} "
                  f"{_fmt((s['kl_all_positions'] or {}).get('max')):>9s} "
                  f"{_fmt((s['max_abs_logit_diff_prompt_max'] or {}).get('max'), '.3g'):>11s} "
                  f"{100 * s['top1_agree_all']:6.2f} {100 * s['bitwise_identical_share']:8.1f}")
        if "per_shape" in g:
            print(f"  per-shape table chunks {g['per_shape']['chosen_chunks']} ({g['per_shape']['table']})")


# --------------------------------------------------------------------------- CLI
def _parse_ints(raw: str) -> list[int]:
    return [int(x) for x in str(raw).split(",") if x.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, default="mamba2")
    p.add_argument("--model-id", default=None, help="Hub id or local directory (default: the captured-trace checkpoint)")
    p.add_argument("--dtype", choices=DTYPES, default="float16", help="weights and activations")
    p.add_argument("--chunks", type=_parse_ints, default=None,
                   help="alternative chunks (default: mamba2 32,64,128,512; mamba1 128..2048; gdn 16,32)")
    p.add_argument("--n-prompts", type=int, default=200, help="stratified subset size (the corpus has 192)")
    p.add_argument("--max-len", type=int, default=2048, help="prompt length cap in tokens")
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--top-k", type=int, default=20, help="reference top-k stored per position")
    p.add_argument("--free-run", choices=("on-divergence", "always"), default="on-divergence")
    p.add_argument("--max-prompts", type=int, default=None, help="process at most this many new prompts in this call")
    p.add_argument("--allow-slow-path", action="store_true",
                   help="allow the portable PyTorch path (Mamba-2: all chunks; Mamba-1: reference and rerun only)")
    p.add_argument("--dispatch-module", default="src.corey_selective_scan_dispatch", help="Mamba-1 routed dispatch")
    p.add_argument("--chunk-module", default="selective_scan_chunk_cuda", help="Mamba-1 runtime-chunk extension")
    p.add_argument("--hf-home", default=None, help="Hugging Face cache root (sets HF_HOME before any HF import)")
    p.add_argument("--smoke", action="store_true", help="3 prompts, 8 new tokens, smoke directory, slow path allowed")
    p.add_argument("--dry-run", action="store_true", help="print the prompt plan (tokenizer only, no model, no GPU)")
    p.add_argument("--download-only", action="store_true", help="fetch the checkpoint and tokenizer, then exit (CPU)")
    p.add_argument("--analyze", action="store_true", help="only (re)write the analysis from the JSONL (CPU)")
    p.add_argument("--input-tags", default=None, help="--analyze: comma-separated result tags to merge (default --tag)")
    for m in MODELS:
        p.add_argument(f"--{m}-table", type=Path, default=DEFAULT_TABLES[m], help=f"sweep CSV of the {m} per-shape table")
    p.add_argument("--show", type=int, default=12, help="--dry-run: plan rows to print")
    p.add_argument("--est-step-ms", type=float, default=40.0, help="--dry-run: decode step time (RTX 3090 host)")
    p.add_argument("--est-prefill-s", type=float, default=0.08, help="--dry-run: prefill time")
    p.add_argument("--est-div-share", type=float, default=0.3, help="--dry-run: share of chunks needing a free run")
    p.add_argument("--out-dir", type=Path, default=None, help=f"default {RESULTS_DIR} ({SMOKE_DIR} with --smoke)")
    p.add_argument("--log-dir", type=Path, default=LOG_DIR)
    p.add_argument("--tag", default=None, help="result-file tag (default: host name; smoke-<host> with --smoke)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.smoke:
        args.n_prompts, args.new_tokens, args.allow_slow_path = 3, 8, True
    args.chunks = sorted(set(args.chunks if args.chunks is not None else ALT_CHUNKS[args.model]))
    args.out_dir = Path(args.out_dir) if args.out_dir else (SMOKE_DIR if args.smoke else RESULTS_DIR)
    args.tag = args.tag or (f"smoke-{host_name()}" if args.smoke else host_name())
    args.tables = {m: getattr(args, f"{m}_table") for m in MODELS}
    if args.new_tokens < 1 or args.max_len < 2 or args.top_k < 1:
        p.error("--new-tokens, --top-k must be >= 1 and --max-len >= 2")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.hf_home:
        if any(m.startswith(("huggingface_hub", "transformers")) for m in sys.modules):
            raise SystemExit("--hf-home must be applied before huggingface_hub is imported")
        os.environ["HF_HOME"] = str(args.hf_home)
    if args.analyze:
        analyze(args)
        return
    log = Logger(f"e31_divergence_{args.tag}", args.log_dir)
    if args.dry_run:
        dry_run(args, log)
    elif args.download_only:
        download_only(args, log)
    else:
        run(args, log)


if __name__ == "__main__":
    main()
