"""E32 - chunk selection inside a full-model prefill captured as one CUDA graph (review taco-r2 M3).

The per-call studies replay one scan call per CUDA graph. This experiment measures what a chunk choice
does to a whole prefill forward pass when the entire forward is captured as one CUDA graph (the
device-bound regime of a serving engine that captures its prefill step), and how large the scan's share
of that forward pass is.

Models (one per process, ``--model``):

  mamba2  a Mamba-2 checkpoint in transformers' format (``--model-id``; 370M ``AntonV/mamba2-370m-hf``
          by default, 2.7B ``benchang1110/mamba2-2.7b-hf``). The fast path calls the Triton SSD kernel
          ``mamba_chunk_scan_combined`` with every mixer's ``chunk_size`` attribute; the attribute is set
          on all mixers before a capture. Default chunk 256 (the library default).
  gdn     Gated DeltaNet 1.4B (``Idiap/gated-deltanet-swa-1.4B-30B``, a hybrid with sliding-window
          attention). ``chunk_gated_delta_rule`` as the FLA layer calls it is wrapped so that
          ``chunk_size`` is injected. Default chunk 64 (the FLA default).

Every scan call goes through one wrapper that records the chunk it received, so each capture is checked
to have run every layer at the intended chunk.

Variants per (length L, batch B): every chunk of the model's candidate set, and ``noscan``: the scan
entry point returns a stored output of the right shape instead of running (the rest of the forward pass
is unchanged), so that (T(default) - T(noscan)) / T(default) is the scan's share of the forward pass.

Inputs: real text. The capture corpus (as the trace capture selected it) is tokenized and concatenated
into one token stream; row b of a batch is the slice of L tokens starting at b * max(L). Falls back to
seeded random token ids if the corpus cannot be loaded (recorded in the metadata).

Shapes: every L in ``--lengths`` and B in ``--batches`` with L * B <= ``--max-tokens`` (a prefill token budget).

Protocol per (L, B):
  1. eager forward at the default chunk (reference logits of the last position, stored scan output for
     ``noscan``), then one eager forward per chunk (Triton autotuning, chunk check, logits difference to
     the default);
  2. ``--rounds`` rounds; in each round the variants run in a seeded random order; per variant and mode:
     ``graph``: warm-up forwards on a side stream, capture of the whole forward (``use_cache=False``,
     last-position logits), ``--warmup`` replays, then ``--repeats`` replays timed with CUDA events after
     a GPU-side pre-spin (device time of the captured forward; the graph is captured anew in every round,
     so buffer placement varies between rounds); ``eager``: ``--eager-repeats`` eager forwards timed with
     CUDA events and no pre-spin (launch-inclusive).
  SM clock, power and throttle reasons before and after each cell (``--record-gpu-state``); with
  ``--exclusive-guard`` a cell waits while another process holds the GPU (``--guard-gpu``: the physical
  index) and is re-timed once if one appeared during it.

Resumable: one fsynced JSONL row per (model label, dtype, L, B, mode, round, variant); completed keys are
skipped on restart. Out-of-memory cells are recorded and skipped.

``--analyze`` (CPU) reads one or more tags (``--input-tags``; one tag = one session) and writes
``e32_model_graph_<tag>.json``: per (label, L, B) the round-median latency of every variant, the scan
share, the gains of the shape table and of the in-session best chunk over the default, bootstrap
intervals over rounds, and across sessions the reproducibility of the best chunk and of the gains, plus
the gain of a table tuned in one session and evaluated in the other.

The shape table is the per-length-bucket chunk with the lowest mean latency over the captured calls of
that batch size in a stored per-call CUDA-graph sweep on another RTX 3090 board (``--table``): it was
built from isolated scan calls of other prompts, so it is held out from this experiment.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.experiments.tpds.common import (GDN_CHUNKS, LOG_DIR, MAMBA2_CHUNKS, RESULTS_DIR, JsonlWriter, Logger,
                                         check_disk, cuda_time, dedupe_by_key, env_metadata, foreign_gpu_pids,
                                         gpu_state, host_name, read_jsonl, utc_now, wait_for_exclusive_gpu,
                                         write_json)

DEFAULT_CHUNK = {"mamba2": 256, "gdn": 64}
GDN_RECURRENT_MAX_LEN = 64   # fla GatedDeltaNet: mode = fused_recurrent when q_len <= 64 and not training
CHUNKS = {"mamba2": MAMBA2_CHUNKS, "gdn": GDN_CHUNKS}
DEFAULT_DTYPE = {"mamba2": "float16", "gdn": "bfloat16"}
DEFAULT_MODEL_ID = {"mamba2": "AntonV/mamba2-370m-hf", "gdn": "Idiap/gated-deltanet-swa-1.4B-30B"}
# Stored per-call CUDA-graph sweeps (RTX 3090, three rounds) that define the held-out shape tables.
TABLES = {"ssd370m": RESULTS_DIR / "e2_mamba2_vgpu3090_r4graph3.csv",
          "ssd2p7b": RESULTS_DIR / "e2_mamba2_vgpu3090_2p7b_r4graph3.csv",
          "gdn1p4b": RESULTS_DIR / "e2_gdn_vgpu3090_1p4b_r4graph3.csv"}
CORPUS = {"samples_per_regime": 20, "longbench_per_task": 8, "longbench_lengths": "1024,2048,4096,8192", "seed": 0}


def _parse_ints(raw: str) -> list[int]:
    return [int(x) for x in str(raw).split(",") if x.strip()]


def _g(x: Any, digits: int = 6) -> Any:
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return x
    return float(f"{x:.{digits}g}")


def row_key(label: str, dtype: str, L: int, B: int, mode: str, rnd: int, variant: str) -> str:
    return f"e32|{label}|{dtype}|L{L}|B{B}|{mode}|r{rnd}|{variant}"


# --------------------------------------------------------------------------- model wrapper
class ScanModel:
    """A model whose scan entry point is wrapped: chunk control, chunk records, and the no-scan variant."""

    def __init__(self, kind: str, model_id: str, dtype_name: str, log):
        import torch

        self.kind, self.model_id, self.log = kind, model_id, log
        self.dtype = getattr(torch, dtype_name)
        self.info: dict[str, Any] = {}
        if kind == "mamba2":
            from src.experiments.tpds.e31_divergence import _pin_installed_kernels

            self.info["pinned_kernels"] = _pin_installed_kernels(log)
            from transformers import AutoTokenizer, Mamba2ForCausalLM
            from transformers.models.mamba2 import modeling_mamba2 as m2

            self.tok = AutoTokenizer.from_pretrained(model_id)
            self.model = Mamba2ForCausalLM.from_pretrained(model_id, dtype=self.dtype).to("cuda").eval()
            # transformers resolves the kernels lazily while building the mixers, so check after loading
            if not bool(getattr(m2, "is_fast_path_available", False)):
                raise SystemExit("transformers' Mamba-2 fast path (mamba_ssm + causal_conv1d) is unavailable")
            self.mixers = [layer.mixer for layer in self.model.backbone.layers]
            self.n_scan_layers = len(self.mixers)
            self.mod, self.entry = m2, "mamba_chunk_scan_combined"
            self.info["config_chunk_size"] = int(self.model.config.chunk_size)
        else:
            from src.experiments.tpds.capture_gdn_traces import gdn_layers, load_model

            self.tok, self.model = load_model(model_id, self.dtype)
            import fla.layers.gated_deltanet as layer_module  # type: ignore

            self.mixers = []
            self.n_scan_layers = len(gdn_layers(self.model))
            self.mod, self.entry = layer_module, "chunk_gated_delta_rule"
        self.orig = getattr(self.mod, self.entry)
        self.chunk = DEFAULT_CHUNK[kind]
        self.noscan = False
        self.stash = None
        self.records: list[int | None] = []
        self.info.update({"n_scan_layers": self.n_scan_layers, "entry": f"{self.mod.__name__}.{self.entry}",
                          "n_params_m": round(sum(p.numel() for p in self.model.parameters()) / 1e6, 1)})

        def wrapped(*a, **kw):
            if self.kind == "gdn":
                kw["chunk_size"] = self.chunk
            c = kw.get("chunk_size")
            self.records.append(None if self.noscan else (int(c) if c is not None else -1))
            if self.noscan:
                if self.stash is None:
                    raise RuntimeError("noscan requested before a stored scan output exists")
                return self.stash
            out = self.orig(*a, **kw)
            if self.stash is None:
                self.stash = out
            return out

        setattr(self.mod, self.entry, wrapped)
        self.configure(DEFAULT_CHUNK[kind])

    def configure(self, chunk: int) -> None:
        self.chunk = int(chunk)
        for m in self.mixers:
            m.chunk_size = int(chunk)

    def forward(self, ids):
        out = self.model(input_ids=ids, use_cache=False, logits_to_keep=1)
        return out.logits

    def close(self) -> None:
        setattr(self.mod, self.entry, self.orig)


# --------------------------------------------------------------------------- inputs
def token_stream(tok, need: int, log) -> tuple[list[int], str]:
    try:
        from src.experiments.tpds.capture_traces import select_corpus

        corpus = select_corpus(argparse.Namespace(**CORPUS, limit_prompts=None))
        eos = tok.eos_token_id if tok.eos_token_id is not None else 0
        ids: list[int] = []
        for it in corpus:
            ids.extend(int(i) for i in tok(it["text"])["input_ids"])
            ids.append(int(eos))
            if len(ids) >= need:
                break
        if len(ids) < need:
            ids = (ids * math.ceil(need / max(len(ids), 1)))[:need]
        return ids[:need], f"capture corpus ({len(corpus)} prompts), concatenated"
    except Exception as exc:  # pragma: no cover - fallback when the datasets are missing
        log(f"corpus unavailable ({exc!r}); using seeded random token ids")
        rng = random.Random(0)
        vocab = int(getattr(tok, "vocab_size", 32000) or 32000)
        return [rng.randrange(100, vocab - 100) for _ in range(need)], "random token ids (seed 0)"


# --------------------------------------------------------------------------- timing helpers
def capture_graph(fn):
    import torch

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def summ(ts: list[float]) -> dict[str, Any]:
    return {"n": len(ts), "median_ms": _g(statistics.median(ts)), "mean_ms": _g(statistics.fmean(ts)),
            "min_ms": _g(min(ts)), "max_ms": _g(max(ts)), "std_ms": _g(statistics.pstdev(ts) if len(ts) > 1 else 0.0)}


def free_cuda() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


@contextlib.contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise SystemExit(f"another E32 process holds {path}; wait for it or use a different --tag")
    try:
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


# --------------------------------------------------------------------------- run
def run(args, log) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    jsonl = args.out_dir / f"e32_model_graph_{args.tag}.jsonl"
    check_disk(args.out_dir, need_gb=0.2, need_inodes=100, log=log)
    lengths, batches = _parse_ints(args.lengths), _parse_ints(args.batches)
    chunks = _parse_ints(args.chunks) if args.chunks else list(CHUNKS[args.model])
    default = DEFAULT_CHUNK[args.model]
    if default not in chunks:
        chunks = sorted(set(chunks) | {default})
    variants = [f"c{c}" for c in chunks] + ([] if args.skip_noscan else ["noscan"])
    modes = [m for m in args.modes.split(",") if m]
    with exclusive_lock(Path(args.log_dir) / f"e32_model_graph_{args.tag}.lock"):
        torch.manual_seed(args.seed)
        sm = ScanModel(args.model, args.model_id, args.dtype, log)
        try:
            stream, stream_src = token_stream(sm.tok, max(lengths) * max(batches), log)
            meta = env_metadata()
            meta.update({"experiment": "E32", "model": args.model, "label": args.label, "model_id": args.model_id,
                         "dtype": args.dtype, "default_chunk": default, "chunks": chunks, "variants": variants,
                         "modes": modes, "lengths": lengths, "batches": batches, "max_tokens": args.max_tokens,
                         "rounds": args.rounds,
                         "warmup": args.warmup, "repeats": args.repeats, "eager_repeats": args.eager_repeats,
                         "pre_spin_ms": args.pre_spin_ms, "inputs": stream_src, "model_info": sm.info,
                         "guard_gpu": args.guard_gpu, "session": args.tag,
                         "torch_flags": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                         "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
                         "args": {k: str(v) for k, v in vars(args).items()}})
            w = JsonlWriter(jsonl, meta=meta)
            log(f"E32 {args.label} ({args.model_id}) {args.dtype} on {meta.get('gpu')}: L={lengths} B={batches} "
                f"variants={variants} modes={modes} rounds={args.rounds}; {len(w.done)} rows done")
            guard_until = args.guard_until
            n_cells = 0
            with torch.inference_mode():
                for B in batches:
                    for L in lengths:
                        if L * B > args.max_tokens:
                            continue
                        if args.model == "gdn" and L <= GDN_RECURRENT_MAX_LEN:
                            # The Gated DeltaNet layer runs a fused recurrent kernel for q_len <= 64 in inference,
                            # which has no chunk parameter: the chunk is not a tunable knob at these lengths.
                            log(f"L={L} B={B}: skipped, the layer uses the recurrent kernel (no chunk parameter)")
                            continue
                        keys = [row_key(args.label, args.dtype, L, B, m, r, v) for m in modes
                                for r in range(args.rounds) for v in variants]
                        if all(k in w.done for k in keys):
                            continue
                        if args.max_cells is not None and n_cells >= args.max_cells:
                            log(f"--max-cells {args.max_cells} reached")
                            break
                        n_cells += 1
                        try:
                            run_shape(args, log, w, sm, stream, L, B, chunks, variants, modes, guard_until)
                        except torch.cuda.OutOfMemoryError as exc:
                            log(f"L={L} B={B}: out of memory ({str(exc)[:120]}); recording and skipping")
                            for k in keys:
                                if k not in w.done:
                                    w.write({"key": k, "kind": "oom", "label": args.label, "L": L, "batch": B,
                                             "utc": utc_now()})
                        finally:
                            sm.stash = None
                            sm.noscan = False
                            free_cuda()
            w.close()
        finally:
            sm.close()
    try:
        analyze(argparse.Namespace(**{**vars(args), "input_tags": args.tag}))
    except Exception as exc:  # the analysis is CPU-side and can be re-run with --analyze
        log(f"analysis failed: {exc!r}")


def run_shape(args, log, w, sm: ScanModel, stream: list[int], L: int, B: int, chunks: list[int],
              variants: list[str], modes: list[str], guard_until) -> None:
    import torch

    lmax = max(_parse_ints(args.lengths))
    ids = torch.tensor([stream[b * lmax: b * lmax + L] for b in range(B)], dtype=torch.long, device="cuda")
    default = DEFAULT_CHUNK[sm.kind]
    t0 = time.perf_counter()
    # 1. eager reference at the default chunk (also stores the scan output for ``noscan``)
    sm.stash, sm.noscan = None, False
    sm.configure(default)
    sm.records.clear()
    ref = sm.forward(ids).float()
    torch.cuda.synchronize()
    ref_top1 = ref[:, -1].argmax(-1)
    eager_check: dict[str, Any] = {}
    for c in chunks:
        sm.configure(c)
        sm.records.clear()
        out = sm.forward(ids).float()
        torch.cuda.synchronize()
        recs = Counter(sm.records)
        ok = recs == Counter({c: sm.n_scan_layers})
        eager_check[f"c{c}"] = {"chunk_ok": bool(ok), "records": {str(k): v for k, v in recs.items()},
                                "max_abs_logit_diff": _g((out - ref).abs().max().item()),
                                "top1_agree": _g((out[:, -1].argmax(-1) == ref_top1).float().mean().item())}
        if not ok:
            raise RuntimeError(f"chunk {c} not honoured at L={L} B={B}: {dict(recs)}")
    log(f"L={L} B={B}: eager checks done in {time.perf_counter() - t0:.1f}s; "
        + " ".join(f"{k}:d={v['max_abs_logit_diff']:.2g}" for k, v in eager_check.items()))
    for rnd in range(args.rounds):
        order = list(variants)
        random.Random(f"{args.seed}|{L}|{B}|{rnd}").shuffle(order)
        for v in order:
            for mode in modes:
                key = row_key(args.label, args.dtype, L, B, mode, rnd, v)
                if key in w.done:
                    continue
                row = time_cell(args, sm, ids, L, B, v, mode, guard_until, ref, ref_top1)
                row.update({"key": key, "kind": "timing", "label": args.label, "model": sm.kind, "dtype": args.dtype,
                            "L": L, "batch": B, "mode": mode, "round": rnd, "variant": v,
                            "chunk": None if v == "noscan" else int(v[1:]), "default_chunk": default,
                            "eager_check": eager_check.get(v), "utc": utc_now()})
                if args.exclusive_guard and row.get("foreign_after"):
                    log(f"L={L} B={B} r{rnd} {v} {mode}: foreign process during the cell; re-timing once")
                    wait_for_exclusive_gpu(guard_until, log, index=args.guard_gpu)
                    row2 = time_cell(args, sm, ids, L, B, v, mode, guard_until, ref, ref_top1)
                    row.update({**row2, "retimed": True})
                w.write(row)
                log(f"L={L} B={B} r{rnd} {mode:5s} {v:6s} median {row['t']['median_ms']:.3f} ms "
                    f"(n={row['t']['n']}, chunk_ok={row.get('chunk_ok')})")
    del ids, ref


def time_cell(args, sm: ScanModel, ids, L, B, v, mode, guard_until, ref, ref_top1) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {}
    if args.exclusive_guard:
        waited = wait_for_exclusive_gpu(guard_until, None, index=args.guard_gpu)
        if waited is None:
            raise SystemExit("guard deadline passed while another process held the GPU")
        out["guard_wait_s"] = _g(waited, 4)
    if args.record_gpu_state:
        out["gpu_before"] = gpu_state(args.guard_gpu)
    sm.noscan = v == "noscan"
    if not sm.noscan:
        sm.configure(int(v[1:]))
    sm.records.clear()
    fn = lambda: sm.forward(ids)  # noqa: E731
    if mode == "graph":
        free_cuda()
        mem0 = torch.cuda.memory_reserved()
        tc = time.perf_counter()
        g, logits = capture_graph(fn)
        out["capture_s"] = _g(time.perf_counter() - tc, 4)          # 2 warm-up forwards + the capture
        out["graph_reserved_gb"] = _g((torch.cuda.memory_reserved() - mem0) / 2**30, 4)
        recs = Counter(sm.records)
        n_calls = sum(recs.values())
        # warm-up calls (2) + the capture call each record one entry per scan layer
        expect = Counter({None if sm.noscan else int(v[1:]): 3 * sm.n_scan_layers})
        out["chunk_ok"] = bool(recs == expect)
        out["records"] = {str(k): n for k, n in recs.items()}
        ts = cuda_time(g.replay, warmup=args.warmup, repeats=args.repeats, pre_spin_ms=args.pre_spin_ms)
        if not sm.noscan:
            lf = logits.float()
            out["max_abs_logit_diff"] = _g((lf - ref).abs().max().item())
            out["top1_agree"] = _g((lf[:, -1].argmax(-1) == ref_top1).float().mean().item())
        del g, logits
        out["n_scan_calls_recorded"] = n_calls
    else:
        ts = cuda_time(fn, warmup=2, repeats=args.eager_repeats, pre_spin_ms=0.0)
        recs = Counter(sm.records)
        out["chunk_ok"] = set(recs) == ({None} if sm.noscan else {int(v[1:])})
        out["records"] = {str(k): n for k, n in recs.items()}
    sm.noscan = False
    out["t"] = summ(ts)
    out["times_ms"] = [_g(t, 5) for t in ts]
    out["peak_vram_gb"] = _g(torch.cuda.max_memory_allocated() / 2**30, 4)
    torch.cuda.reset_peak_memory_stats()
    if args.record_gpu_state:
        out["gpu_after"] = gpu_state(args.guard_gpu)
    if args.exclusive_guard:
        out["foreign_after"] = bool(foreign_gpu_pids(args.guard_gpu))
    free_cuda()
    return out


# --------------------------------------------------------------------------- analysis
def _round_medians(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    by: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        by[r["variant"]].append((r["round"], r["t"]["median_ms"]))
    return {v: [t for _, t in sorted(x)] for v, x in by.items()}


def _boot_gain(dflt: list[float], alt: list[float], n: int = 2000, seed: int = 0) -> list[float] | None:
    """95% bootstrap interval (over rounds, resampled independently) of (median(dflt) - median(alt)) / median(dflt)."""
    if len(dflt) < 2 or len(alt) < 2:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        d = statistics.median(rng.choices(dflt, k=len(dflt)))
        a = statistics.median(rng.choices(alt, k=len(alt)))
        vals.append((d - a) / d)
    vals.sort()
    return [_g(100 * vals[int(0.025 * n)], 4), _g(100 * vals[int(0.975 * n) - 1], 4)]


def _table_for(label: str, table_path: Path | None, batch: int) -> dict[str, Any] | None:
    from src.experiments.tpds.e31_divergence import load_table

    path = table_path or TABLES.get(label)
    if path is None or not Path(path).exists():
        return None
    model = "gdn" if label.startswith("gdn") else "mamba2"
    try:
        return load_table(model, Path(path), batch=batch)
    except Exception:
        return None


def analyze(args) -> dict[str, Any]:
    from src.experiments.tpds.e31_divergence import table_chunk

    tags = [t for t in str(args.input_tags or args.tag).split(",") if t]
    out_tag = args.tag if len(tags) == 1 else "_".join(tags)
    sessions: dict[str, list[dict[str, Any]]] = {}
    metas: dict[str, dict[str, Any]] = {}
    for t in tags:
        rows = read_jsonl(args.out_dir / f"e32_model_graph_{t}.jsonl")
        mrows = read_jsonl(args.out_dir / f"e32_model_graph_{t}.jsonl", kind="meta")
        metas[t] = mrows[-1] if mrows else {}
        sessions[t] = dedupe_by_key([r for r in rows if r.get("kind") == "timing"])
    res: dict[str, Any] = {"experiment": "E32", "sessions": tags, "created_utc": utc_now(),
                           "meta": {t: {k: m.get(k) for k in ("gpu", "label", "model_id", "dtype", "default_chunk",
                                                             "chunks", "rounds", "repeats", "pre_spin_ms", "utc",
                                                             "inputs", "torch", "pkg_triton", "pkg_mamba_ssm",
                                                             "pkg_flash-linear-attention")}
                                    for t, m in metas.items()},
                           "cells": [], "summary": {}}
    tables: dict[tuple[str, int], Any] = {}
    cells_by_label: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for t, rows in sessions.items():
        groups: dict[tuple, list] = defaultdict(list)
        for r in rows:
            groups[(r["label"], r["L"], r["batch"], r["mode"])].append(r)
        for (label, L, B, mode), rs in sorted(groups.items()):
            if (label, B) not in tables:
                tables[(label, B)] = _table_for(label, getattr(args, "table", None), B)
            tab = tables[(label, B)]
            med = _round_medians(rs)
            dflt = f"c{rs[0]['default_chunk']}"
            if dflt not in med:
                continue
            T = {v: statistics.median(x) for v, x in med.items()}
            chunk_vs = {v: T[v] for v in T if v != "noscan"}
            best = min(chunk_vs, key=lambda v: (chunk_vs[v], v))
            cell = {"session": t, "label": label, "L": L, "batch": B, "mode": mode, "default": dflt,
                    "n_rounds": min(len(x) for x in med.values()),
                    "median_ms": {v: _g(x, 5) for v, x in sorted(T.items())},
                    "round_cv_default": _g(statistics.pstdev(med[dflt]) / statistics.fmean(med[dflt])
                                           if len(med[dflt]) > 1 else None, 4),
                    "best": best, "gain_best_pct": _g(100 * (T[dflt] - T[best]) / T[dflt], 4),
                    "all_chunk_ok": all(r.get("chunk_ok") for r in rs),
                    "max_abs_logit_diff": _g(max((r.get("max_abs_logit_diff") or 0.0) for r in rs), 4)}
            if "noscan" in T:
                cell["scan_share_pct"] = _g(100 * (T[dflt] - T["noscan"]) / T[dflt], 4)
                cell["scan_ms_default"] = _g(T[dflt] - T["noscan"], 5)
            if tab is not None:
                c, bucket, fb = table_chunk(tab, L)
                tv = f"c{c}"
                cell.update({"table_chunk": c, "table_bucket": bucket, "table_fallback": fb,
                             "table_source": tab.get("source")})
                if tv in T:
                    cell["gain_table_pct"] = _g(100 * (T[dflt] - T[tv]) / T[dflt], 4)
                    cell["gain_table_ci95"] = _boot_gain(med[dflt], med[tv])
                    cell["table_regret_vs_best_pct"] = _g(100 * (T[tv] - T[best]) / T[best], 4)
                    if "noscan" in T and T[dflt] > T["noscan"]:
                        cell["gain_table_pct_of_scan"] = _g(100 * (T[dflt] - T[tv]) / (T[dflt] - T["noscan"]), 4)
            res["cells"].append(cell)
            cells_by_label[(label, mode)].append(cell)
    # per (label, mode, session) summaries
    for (label, mode), cells in sorted(cells_by_label.items()):
        for t in tags:
            cs = [c for c in cells if c["session"] == t]
            if not cs:
                continue
            tot_d = sum(c["median_ms"][c["default"]] for c in cs)
            s: dict[str, Any] = {"n_cells": len(cs)}
            for name, field in (("table", "table_chunk"), ("best", "best")):
                have = [c for c in cs if (field in c) and (f"c{c[field]}" if name == "table" else c[field]) in c["median_ms"]]
                if not have:
                    continue
                def _alt(c):
                    v = f"c{c['table_chunk']}" if name == "table" else c["best"]
                    return c["median_ms"][v]
                td = sum(c["median_ms"][c["default"]] for c in have)
                s[f"gain_{name}_latency_weighted_pct"] = _g(100 * (td - sum(_alt(c) for c in have)) / td, 4)
                gains = [c[f"gain_{name}_pct"] for c in have if c.get(f"gain_{name}_pct") is not None]
                if gains:
                    s[f"gain_{name}_mean_pct"] = _g(statistics.fmean(gains), 4)
                    s[f"gain_{name}_range_pct"] = [_g(min(gains), 4), _g(max(gains), 4)]
            shares = [c["scan_share_pct"] for c in cs if c.get("scan_share_pct") is not None]
            if shares:
                s["scan_share_range_pct"] = [_g(min(shares), 4), _g(max(shares), 4)]
                s["scan_share_latency_weighted_pct"] = _g(
                    100 * sum(c["scan_ms_default"] for c in cs if c.get("scan_ms_default") is not None) / tot_d, 4)
            s["round_cv_default_median"] = _g(statistics.median(
                [c["round_cv_default"] for c in cs if c.get("round_cv_default") is not None] or [float("nan")]), 4)
            s["all_chunk_ok"] = all(c["all_chunk_ok"] for c in cs)
            res["summary"].setdefault(label, {}).setdefault(mode, {})[t] = s
        # cross-session reproducibility
        if len(tags) >= 2:
            a, b = tags[0], tags[1]
            ca = {(c["L"], c["batch"]): c for c in cells if c["session"] == a}
            cb = {(c["L"], c["batch"]): c for c in cells if c["session"] == b}
            common = sorted(set(ca) & set(cb))
            if common:
                same_best = sum(ca[k]["best"] == cb[k]["best"] for k in common)
                # a table tuned in-session in A (its best chunk per cell) evaluated on B, and vice versa
                xfer = []
                for x, y in ((ca, cb), (cb, ca)):
                    num = den = 0.0
                    for k in common:
                        d = y[k]["median_ms"][y[k]["default"]]
                        alt = y[k]["median_ms"].get(x[k]["best"])
                        if alt is None:
                            continue
                        num += d - alt
                        den += d
                    xfer.append(_g(100 * num / den, 4) if den else None)
                diffs = [abs((ca[k].get("gain_table_pct") or 0) - (cb[k].get("gain_table_pct") or 0)) for k in common]
                ddef = [abs(ca[k]["median_ms"][ca[k]["default"]] / cb[k]["median_ms"][cb[k]["default"]] - 1) * 100
                        for k in common]
                res["summary"][label][mode]["cross_session"] = {
                    "sessions": [a, b], "n_cells": len(common), "same_best_chunk": same_best,
                    "in_session_best_table_transferred_gain_latency_weighted_pct": xfer,
                    "abs_diff_gain_table_pct_median": _g(statistics.median(diffs), 4),
                    "abs_diff_gain_table_pct_max": _g(max(diffs), 4),
                    "default_latency_session_ratio_abs_pct_median": _g(statistics.median(ddef), 4),
                    "default_latency_session_ratio_abs_pct_max": _g(max(ddef), 4)}
    res["tables"] = {f"{lab}|B{B}": ({"source": t.get("source"), "buckets": {k: v["chunk"] for k, v in t["buckets"].items()}}
                                     if t else None) for (lab, B), t in tables.items()}
    path = args.out_dir / f"e32_model_graph_{out_tag}.json"
    write_json(path, res)
    print(f"wrote {path}")
    for label, modes in res["summary"].items():
        for mode, ss in modes.items():
            for t, s in ss.items():
                print(label, mode, t, {k: v for k, v in s.items() if not isinstance(v, dict)})
    return res


# --------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(DEFAULT_CHUNK), default="mamba2")
    p.add_argument("--model-id", default=None)
    p.add_argument("--label", default=None, help="model label used in keys and tables (ssd370m, ssd2p7b, gdn1p4b)")
    p.add_argument("--dtype", default=None)
    p.add_argument("--lengths", default="256,512,1000,1024,2048,3000,4096,6000,8192")
    p.add_argument("--batches", default="1,4")
    p.add_argument("--chunks", default=None, help="candidate chunks (default: the model's set)")
    p.add_argument("--modes", default="graph,eager")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--eager-repeats", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=8192, help="skip (L, B) with L * B above this prefill budget")
    p.add_argument("--pre-spin-ms", type=float, default=50.0)
    p.add_argument("--skip-noscan", action="store_true")
    p.add_argument("--max-cells", type=int, default=None, help="stop after this many (L, B) cells (smoke tests)")
    p.add_argument("--table", type=Path, default=None, help="per-call sweep CSV of the shape table (default by label)")
    p.add_argument("--exclusive-guard", action="store_true")
    p.add_argument("--guard-until", type=float, default=None)
    p.add_argument("--guard-gpu", type=int, default=None, help="physical GPU index for the guard and GPU state")
    p.add_argument("--record-gpu-state", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--log-dir", type=Path, default=LOG_DIR)
    p.add_argument("--tag", default=None)
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--input-tags", default=None, help="comma-separated session tags to analyse together")
    a = p.parse_args(argv)
    a.model_id = a.model_id or DEFAULT_MODEL_ID[a.model]
    a.dtype = a.dtype or DEFAULT_DTYPE[a.model]
    a.label = a.label or {"mamba2": "ssd370m", "gdn": "gdn1p4b"}[a.model]
    a.tag = a.tag or f"{host_name()}_{a.label}"
    if a.guard_gpu is None:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        a.guard_gpu = int(vis) if vis.isdigit() else 0
    return a


def main(argv=None):
    args = parse_args(argv)
    if args.analyze:
        analyze(args)
        return
    log = Logger(f"e32_model_graph_{args.tag}", args.log_dir)
    run(args, log)


if __name__ == "__main__":
    main()
