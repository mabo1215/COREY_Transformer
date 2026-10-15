"""E27 - chunk choice of packed variable-length prefill calls (review item M6b).

Serving engines with continuous batching and chunked prefill run the Mamba-2 SSD and Gated DeltaNet
kernels on packed variable-length calls: the prompt chunks of several requests are concatenated along
the sequence axis (batch 1) and the state is reset at every sequence boundary; the first packed sequence
may continue a prompt begun in an earlier step and then starts from a carried state. The study's other
sweeps time equal-length batches only. This experiment times every chunk of both production kernels on
packed steps sampled from public serving traces (``packed_steps.py``):

  ssd370m / ssd2p7b  ``mamba_chunk_scan_combined`` with ``seq_idx`` (captured 370M and 2.7B Mamba-2 inputs)
  gdn1p4b            ``chunk_gated_delta_rule`` with ``cu_seqlens`` (captured 1.4B Gated DeltaNet inputs)

Each packed sequence of a step is a slice of a captured prompt of at least its length (same layer, the
layer rotating across steps), so values and layer parameters are those of the checkpoints. Two variants
per step: ``fresh`` (no carried state) and ``carried`` (a carried state for the first sequence (SSD) or for
every sequence (Gated DeltaNet)). Timing: device time (GPU-side pre-spin), warm L2, ``--rounds`` rounds
with the chunk order shuffled per step and round; the per-step oracle is defined on round medians.

``--analyze`` writes the per-family summary: share of steps whose default chunk is within 2% of the oracle,
mean per-step and latency-weighted regret of the default, of the best single chunk, and of held-out tables
keyed on the packed-token bucket alone and on (bucket, number of sequences), and how often the carried state
moves the optimum.

Output: ``e27_varlen_<tag>.jsonl`` (resumable, one row per step, variant, round) and ``e27_varlen_<tag>.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.experiments.tpds.common import (GDN_CHUNKS, MAMBA2_CHUNKS, RESULTS_DIR, JsonlWriter, Logger, all_gpu_pids,
                                         check_disk, env_metadata, foreign_gpu_pids, host_name, read_jsonl,
                                         register_own_gpu_pids, summarize, wait_for_exclusive_gpu, write_json)

FAMILIES = {"ssd370m": ("mamba2", 256, MAMBA2_CHUNKS), "ssd2p7b": ("mamba2", 256, MAMBA2_CHUNKS),
            "gdn1p4b": ("gdn", 64, GDN_CHUNKS)}
VARIANTS = ("fresh", "carried")
TIE = 0.02


# --------------------------------------------------------------------------- inputs
def trace_index(trace_dir: Path, kind: str):
    from src.experiments.tpds.capture_traces import model_dir

    rows = read_jsonl(model_dir(trace_dir, kind) / "manifest.jsonl")
    meta = next((r for r in rows if r.get("layers")), {})
    calls = {}
    for r in rows:
        if r.get("prompt_id") is not None and r.get("L"):
            calls[r["prompt_id"]] = int(r["L"])
    layers = [int(x) for x in meta.get("layers", [])]
    if not calls or not layers:
        raise SystemExit(f"no captured {kind} calls in {trace_dir}")
    return sorted(calls.items(), key=lambda kv: kv[0]), layers


def build_step(family, trace_dir, index, layers, step, step_no):
    """Packed inputs of one step: each sequence is a slice of a captured prompt that is long enough."""
    import torch

    from src.experiments.tpds.capture_traces import load_call, load_params

    kind = FAMILIES[family][0]
    calls, _ = index
    layer = layers[step_no % len(layers)]
    rng = random.Random(f"e27|{family}|{step_no}")
    parts, lens = [], []
    for j, c in enumerate(step["lens"]):
        pool = [pid for pid, L in calls if L >= c] or [max(calls, key=lambda kv: kv[1])[0]]
        pid = rng.choice(pool)
        d = load_call(trace_dir, kind, pid, layer)
        n = min(c, next(L for p, L in calls if p == pid))
        keys = ("x", "dt", "B", "C") if kind == "mamba2" else ("q", "k", "v", "g", "beta")
        parts.append({k: d[k][:1, :n] for k in keys})
        lens.append(n)
    inp = {k: torch.cat([p[k] for p in parts], 1).contiguous() for k in parts[0]}
    params = load_params(trace_dir, kind, layer)
    T = int(sum(lens))
    if kind == "mamba2":
        inp.update({"A": params["A"], "D": params["D"].float(), "dt_bias": params["dt_bias"].float()})
        inp["seq_idx"] = torch.cat([torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(lens)])[None].cuda()
    else:
        inp = {k: v.to(torch.bfloat16) for k, v in inp.items()}
        inp.update({"A_log": params["A_log"], "dt_bias": params["dt_bias"],
                    "allow_neg_eigval": bool(params["allow_neg_eigval"].item()) if "allow_neg_eigval" in params
                    else False})
        inp["cu_seqlens"] = torch.tensor([0, *np.cumsum(lens).tolist()], dtype=torch.long, device="cuda")
    return inp, lens, layer, T


def carried_state(family, inp, n_seq):
    """A carried state of the right shape (small random values; the kernels are data-oblivious)."""
    import torch

    from src.experiments.tpds import kernels as K

    g = torch.Generator(device="cuda").manual_seed(7)
    if FAMILIES[family][0] == "mamba2":
        h, p, n = inp["x"].shape[2], inp["x"].shape[3], inp["B"].shape[3]
        return 0.1 * torch.randn(1, h, p, n, device="cuda", generator=g)
    _, st = K.gdn_chunk_prod_varlen(inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], inp["A_log"],
                                    inp["dt_bias"], inp["cu_seqlens"], 64, inp["allow_neg_eigval"],
                                    output_final_state=True)
    return 0.1 * torch.randn(st.shape, device="cuda", generator=g, dtype=torch.float32).to(st.dtype)


def make_fn(family, inp, chunk, state):
    from src.experiments.tpds import kernels as K

    if FAMILIES[family][0] == "mamba2":
        return lambda: K.mamba2_ssd_varlen(inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"], chunk, inp["seq_idx"],
                                           D=inp["D"], dt_bias=inp["dt_bias"], initial_states=state)
    return lambda: K.gdn_chunk_prod_varlen(inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], inp["A_log"],
                                           inp["dt_bias"], inp["cu_seqlens"], chunk, inp["allow_neg_eigval"],
                                           initial_state=state)[0]


# --------------------------------------------------------------------------- run
def run(args):
    import torch

    from src.experiments.tpds.common import cuda_time

    tag = args.tag or host_name()
    log = Logger(f"e27_varlen_{tag}")
    check_disk(args.out_dir, need_gb=0.2, log=log)
    baseline = all_gpu_pids()
    torch.zeros(1, device="cuda")
    own = register_own_gpu_pids(baseline, log=log) if args.exclusive_guard else set()
    steps = json.loads(Path(args.steps).read_text())["steps"]
    if args.max_steps:
        steps = steps[:args.max_steps]
    meta = env_metadata()
    meta.update({"experiment": "E27", "own_gpu_pids": sorted(own), "steps_file": str(args.steps),
                 "args": {k: str(v) for k, v in vars(args).items()}})
    w = JsonlWriter(args.out_dir / f"e27_varlen_{tag}.jsonl", meta=meta)
    traces = {"ssd370m": args.ssd370m_traces, "ssd2p7b": args.ssd2p7b_traces, "gdn1p4b": args.gdn_traces}
    for family in [f for f in args.families.split(",") if f]:
        kind, default, chunks = FAMILIES[family]
        tdir = traces[family]
        index = trace_index(tdir, kind)
        log(f"{family}: {len(index[0])} captured prompts, layers {index[1]}")
        for i, st in enumerate(steps):
            todo = [(v, r) for v in VARIANTS for r in range(args.rounds)
                    if f"e27|{family}|{st['trace']}|T{st['budget']}|s{st['step']}|{v}|r{r}" not in w.done]
            if not todo:
                continue
            inp, lens, layer, T = build_step(family, tdir, index, index[1], st, i)
            states = {"fresh": None, "carried": carried_state(family, inp, len(lens))}
            for v, r in todo:
                key = f"e27|{family}|{st['trace']}|T{st['budget']}|s{st['step']}|{v}|r{r}"
                order = list(chunks)
                random.Random(f"{key}|order").shuffle(order)
                res, err = {}, {}
                if args.exclusive_guard and wait_for_exclusive_gpu(args.guard_until, log) is None:
                    w.close()
                    raise SystemExit(3)
                busy_before = foreign_gpu_pids() if args.exclusive_guard else None
                for c in order:
                    fn = make_fn(family, inp, c, states[v])
                    try:
                        fn()
                        torch.cuda.synchronize()
                        res[c] = summarize(cuda_time(fn, args.warmup, args.repeats, False, args.pre_spin_ms))["mean_ms"]
                    except Exception as e:  # noqa: BLE001 - an unsupported (variant, chunk) is recorded, not fatal
                        err[c] = f"{type(e).__name__}: {e}"[:200]
                        torch.cuda.synchronize()
                busy_after = foreign_gpu_pids() if args.exclusive_guard else None
                w.write({"kind": "timing", "key": key, "family": family, "trace": st["trace"], "budget": st["budget"],
                         "step": st["step"], "variant": v, "round": r, "layer": layer, "lens": lens, "tokens": T,
                         "nseq": len(lens), "continued_first": bool(st["continued"][0]),
                         "lat_ms": {str(c): res[c] for c in res}, "errors": err,
                         "foreign_gpu": bool(busy_before or busy_after)})
            log(f"{family} step {i + 1}/{len(steps)} ({T} tokens, {len(lens)} seqs) done")
            del inp, states
            torch.cuda.empty_cache()
    w.close()
    analyze(args)


# --------------------------------------------------------------------------- analysis
def _bucket(T: int) -> int:
    return 1 << max(0, int(math.floor(math.log2(max(T, 1)))))


def _nseq_class(n: int) -> str:
    return "1" if n == 1 else "2-3" if n <= 3 else "4-7" if n <= 7 else "8+"


def _table_regret(rows, keyfn, chunks, folds=5, seed=0):
    """Held-out table: per key the chunk with the lowest summed relative latency on the other folds."""
    ids = sorted({r["id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(ids)
    fold_of = {s: i % folds for i, s in enumerate(ids)}
    out = []
    for f in range(folds):
        train = [r for r in rows if fold_of[r["id"]] != f]
        agg = defaultdict(lambda: defaultdict(float))
        glob = defaultdict(float)
        for r in train:
            for c in chunks:
                agg[keyfn(r)][c] += r["rel"][c]
                glob[c] += r["rel"][c]
        best_glob = min(chunks, key=lambda c: glob[c])
        for r in rows:
            if fold_of[r["id"]] == f:
                k = keyfn(r)
                c = min(chunks, key=lambda c: agg[k][c]) if k in agg else best_glob
                out.append((r, c))
    return out


def _summ(pairs):
    reg = [r["rel"][c] - 1 for r, c in pairs]
    lat = sum(r["lat"][c] for r, c in pairs)
    opt = sum(r["lat"][r["oracle"]] for r, _ in pairs)
    return {"n": len(pairs), "mean_regret_pct": 100 * statistics.fmean(reg),
            "p95_regret_pct": 100 * float(np.percentile(reg, 95)),
            "latency_weighted_pct": 100 * (lat / opt - 1),
            "share_within_2pct": float(np.mean([x <= TIE for x in reg]))}


def analyze(args):
    tag = args.tag or host_name()
    rows = [r for r in read_jsonl(args.out_dir / f"e27_varlen_{tag}.jsonl") if r.get("kind") == "timing"]
    res = {"experiment": "E27-analysis", "tag": tag, "tie_margin": TIE, "families": {}}
    for family in sorted({r["family"] for r in rows}):
        _, default, chunks_all = FAMILIES[family]
        out = {}
        per_step = {}
        for v in VARIANTS:
            groups = defaultdict(list)
            for r in rows:
                if r["family"] == family and r["variant"] == v:
                    groups[(r["trace"], r["budget"], r["step"])].append(r)
            items = []
            for sid, rs in groups.items():
                chunks = [c for c in chunks_all if all(str(c) in x["lat_ms"] for x in rs)]
                if default not in chunks or len(chunks) < 2:
                    continue
                lat = {c: statistics.median(x["lat_ms"][str(c)] for x in rs) for c in chunks}
                oracle = min(chunks, key=lambda c: lat[c])
                items.append({"id": sid, "tokens": rs[0]["tokens"], "nseq": rs[0]["nseq"], "lat": lat,
                              "oracle": oracle, "rel": {c: lat[c] / lat[oracle] for c in chunks},
                              "chunks": chunks})
            if not items:
                continue
            common = [c for c in chunks_all if all(c in it["chunks"] for it in items)]
            single = min(common, key=lambda c: sum(it["rel"][c] for it in items))
            out[v] = {"n_steps": len(items),
                      "oracle_hist": {str(c): sum(it["oracle"] == c for it in items) for c in common},
                      "default": {"chunk": default, **_summ([(it, default) for it in items])},
                      "best_single": {"chunk": single, **_summ([(it, single) for it in items])},
                      "table_bucket": _summ(_table_regret(items, lambda it: _bucket(it["tokens"]), common)),
                      "table_bucket_nseq": _summ(_table_regret(items, lambda it: (_bucket(it["tokens"]),
                                                                                  _nseq_class(it["nseq"])), common)),
                      "oracle_ms_median": float(np.median([it["lat"][it["oracle"]] for it in items]))}
            per_step[v] = {it["id"]: it for it in items}
        if "fresh" in per_step and "carried" in per_step:
            both = [s for s in per_step["fresh"] if s in per_step["carried"]]
            moved = [s for s in both if per_step["carried"][s]["rel"][per_step["fresh"][s]["oracle"]] - 1 > TIE]
            out["carried_moves_optimum_share"] = float(len(moved) / max(len(both), 1))
            out["carried_over_fresh_median_ratio"] = float(np.median(
                [per_step["carried"][s]["lat"][per_step["carried"][s]["oracle"]]
                 / per_step["fresh"][s]["lat"][per_step["fresh"][s]["oracle"]] for s in both])) if both else None
        res["families"][family] = out
        for v in VARIANTS:
            if v in out:
                o = out[v]
                print(f"{family}/{v}: {o['n_steps']} steps; default {o['default']['mean_regret_pct']:.1f}% "
                      f"(lw {o['default']['latency_weighted_pct']:.1f}%, within 2%: {o['default']['share_within_2pct']:.0%}); "
                      f"bucket table {o['table_bucket']['mean_regret_pct']:.1f}%, +nseq {o['table_bucket_nseq']['mean_regret_pct']:.1f}%")
    write_json(args.out_dir / f"e27_varlen_{tag}.json", res)
    return res


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=Path, default=RESULTS_DIR / "packed_steps_taco.json")
    p.add_argument("--families", default="ssd2p7b,ssd370m,gdn1p4b")
    p.add_argument("--ssd370m-traces", type=Path, default=None)
    p.add_argument("--ssd2p7b-traces", type=Path, default=None)
    p.add_argument("--gdn-traces", type=Path, default=None)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--pre-spin-ms", type=float, default=20.0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--exclusive-guard", action="store_true")
    p.add_argument("--guard-until", type=float, default=None)
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--tag", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.analyze:
        analyze(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
