"""E24: chunk sweep of an independently written public chunked kernel (review item M1b).

The headline chunk effect of the paper is measured on the study's own Triton Mamba-1 scan; the
production kernels it also measures (Mamba-2 SSD, Gated DeltaNet, the reference CUDA scan) either do
not expose a wide chunk range or are flat. This experiment asks whether the optimal chunk moves with
the shape on a public kernel that was written by others and exposes a wide chunk range: the Tiled
Flash Linear Attention (TFLA) mLSTM forward of NX-AI's ``mlstm_kernels`` package
(``mlstm_chunkwise__xl_chunk``, forward only, as in prefill). Its ``chunk_size`` sets the intra-chunk
tile of the parallel kernel; the recurrent kernel that materializes the chunk states uses
min(128, chunk) (the library's own heuristic, unchanged). The library default is chunk 128.

Shapes: three head layouts at the widths of the 370M, 1.4B and 2.8B Mamba checkpoints (value width
2048, 4096 and 5120 as 4, 8 and 10 heads of 512, with query/key head dimension 256, the xLSTM-7B head
shape), lengths 512 ... 8192 (including 3072 and 6144, so that not every length is a power of two;
TFLA needs the length to be a multiple of the chunk), batch 1, 4 and 16, and every chunk in
{32, ..., 2048} that divides the length. Inputs are synthetic (bf16 q, k, v; fp32 gate
pre-activations): the kernel is data-oblivious, and one draw per shape is reused for every chunk and
round. Every shape is timed in ``--rounds`` rounds; shapes are shuffled within a round and chunks
within a shape. Timing as in the other sweeps: CUDA events, ``--pre-spin-ms`` > 0 gives device time
(the host queues every repeat behind a GPU spin), 0 the launch-inclusive latency. A cell that runs
out of memory is recorded as infeasible.

``--analyze`` (CPU) writes ``e24_tfla_<tag>_analysis.json``: per shape the round-median oracle chunk
and its tie set (95% CIs of the round means overlap), the spread (slowest / fastest chunk), and over
all shapes the regret of the library default (128), of the best single chunk, of tables keyed by
length only, by batch only and by layout only (fitted on all shapes, an optimistic bound), and of an
analytical cost model of the Eq.-(5) kind (non-negative least squares on a launch term, the
intra-chunk quadratic work, the state materialization and the sequential chunk steps), evaluated
leave-one-shape-out so that it predicts the chunk of a shape it was not fitted on.

Output: ``e24_tfla_<tag>.jsonl`` (one fsynced row per shape and round; a restart skips completed
rows), ``e24_tfla_<tag>.json`` (meta and counts), ``e24_tfla_<tag>_analysis.json``.
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

from src.experiments.tpds.common import RESULTS_DIR, read_jsonl, t_crit, write_json

# name: (heads, qk head dim, v head dim); value width = heads * v head dim
DEFAULT_LAYOUTS = "370m:4:256:512,1.4b:8:256:512,2.8b:10:256:512"
DEFAULT_CHUNKS = "32,64,128,256,512,1024,2048"
LIBRARY_DEFAULT_CHUNK = 128


def parse_layouts(spec: str) -> dict[str, tuple[int, int, int]]:
    out = {}
    for item in spec.split(","):
        name, nh, dqk, dv = item.split(":")
        out[name] = (int(nh), int(dqk), int(dv))
    return out


def load_tfla():
    from mlstm_kernels.torch.chunkwise.triton_xl_chunk.fw import mlstm_chunkwise_fw  # type: ignore

    return mlstm_chunkwise_fw


def make_inputs(b, nh, dqk, dv, L, seed):
    import torch

    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(b, nh, L, dqk, device="cuda", generator=gen).to(torch.bfloat16)
    k = torch.randn(b, nh, L, dqk, device="cuda", generator=gen).to(torch.bfloat16)
    v = torch.randn(b, nh, L, dv, device="cuda", generator=gen).to(torch.bfloat16)
    i = torch.randn(b, nh, L, device="cuda", generator=gen)               # input-gate pre-activation
    f = 3.0 + torch.randn(b, nh, L, device="cuda", generator=gen)         # forget gate near 1 after the sigmoid
    return {"q": q, "k": k, "v": v, "i": i, "f": f}


def runner(fw, inp, chunk):
    def fn():
        return fw(matQ=inp["q"], matK=inp["k"], matV=inp["v"], vecI=inp["i"], vecF=inp["f"], chunk_size=chunk,
                  output_dtype=inp["q"].dtype)[0]
    return fn


def time_cell(fn, args, log):
    """Time one (shape, chunk) cell; with the guard, re-time if another process used the GPU."""
    from src.experiments.tpds.common import cuda_time, foreign_gpu_pids, summarize, wait_for_exclusive_gpu

    retimed = 0
    while True:
        if args.exclusive_guard:
            waited = wait_for_exclusive_gpu(args.guard_until, log)
            if waited is None:
                return None
            if waited > 0:
                cuda_time(fn, 5 * args.warmup, 1)
        before = foreign_gpu_pids() if args.exclusive_guard else None
        samples = cuda_time(fn, args.warmup, args.repeats, pre_spin_ms=args.pre_spin_ms)
        after = foreign_gpu_pids() if args.exclusive_guard else None
        if not (before or after):
            return {**summarize(samples), "retimed": retimed}
        retimed += 1
        log(f"foreign GPU processes {before or after}; re-timing")


def run(args):
    import torch

    from src.experiments.tpds.common import (JsonlWriter, Logger, all_gpu_pids, check_disk, env_metadata, gpu_state,
                                             host_name, peak_vram_gb, register_own_gpu_pids)

    tag = args.tag or host_name()
    log = Logger(f"e24_tfla_{tag}")
    gpu_baseline = all_gpu_pids()
    check_disk(args.out_dir, need_gb=0.2, log=log)
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    own = register_own_gpu_pids(gpu_baseline, log=log) if args.exclusive_guard else set()
    fw = load_tfla()
    import mlstm_kernels  # type: ignore

    layouts = parse_layouts(args.layouts)
    lengths = [int(x) for x in args.lengths.split(",") if x]
    batches = [int(x) for x in args.batches.split(",") if x]
    chunks = [int(x) for x in args.chunks.split(",") if x]
    meta = env_metadata()
    meta.update({"experiment": "E24", "kernel": "mlstm_kernels chunkwise--triton_xl_chunk (forward)",
                 "mlstm_kernels_version": getattr(mlstm_kernels, "__version__", None), "own_gpu_pids": sorted(own),
                 "layouts": {k: {"heads": v[0], "dqk": v[1], "dv": v[2]} for k, v in layouts.items()},
                 "lengths": lengths, "batches": batches, "chunks": chunks, "library_default_chunk":
                 LIBRARY_DEFAULT_CHUNK, "args": {k: str(v) for k, v in vars(args).items()}})
    w = JsonlWriter(args.out_dir / f"e24_tfla_{tag}.jsonl", meta=meta)
    shapes = [(name, L, b) for name in layouts for L in lengths for b in batches]
    rng = random.Random(args.seed)
    total = len(shapes) * args.rounds
    new_rows, t_start = 0, time.time()
    log(f"{len(shapes)} shapes x {args.rounds} rounds = {total} rows, {len(w.done)} done; pre-spin {args.pre_spin_ms} ms")
    for r in range(args.rounds):
        order = list(shapes)
        rng.shuffle(order)
        for name, L, b in order:
            key = f"e24|{name}|L{L}|b{b}|r{r}"
            if key in w.done:
                continue
            if args.max_rows is not None and new_rows >= args.max_rows:
                break
            nh, dqk, dv = layouts[name]
            inp = make_inputs(b, nh, dqk, dv, L, args.seed + 7 * L + 131 * b + 1009 * nh)
            cs = [c for c in chunks if L % c == 0 and c <= L]
            rng.shuffle(cs)
            s0 = gpu_state()
            cells, ref, max_diff, infeasible = {}, None, 0.0, []
            for c in cs:
                fn = runner(fw, inp, c)
                try:
                    y = fn().float()
                    torch.cuda.synchronize()
                except torch.cuda.OutOfMemoryError:
                    infeasible.append(c)
                    torch.cuda.empty_cache()
                    continue
                if ref is None:
                    ref = y
                else:
                    max_diff = max(max_diff, float((y - ref).abs().max()))
                    del y
                cell = time_cell(fn, args, log)
                if cell is None:
                    log("guard gave up; stopping (exit 3, restart resumes)")
                    w.close()
                    raise SystemExit(3)
                cells[str(c)] = cell
            del ref
            best = min(cells, key=lambda c: cells[c]["mean_ms"]) if cells else None
            w.write({"kind": "timing", "key": key, "layout": name, "heads": nh, "dqk": dqk, "dv": dv, "L": L,
                     "batch": b, "round": r, "cells": cells, "infeasible_chunks": infeasible,
                     "best_chunk": int(best) if best else None, "max_out_diff_across_chunks": max_diff, "gpu_state_before": s0,
                     "gpu_state_after": gpu_state(), "peak_vram_gb": peak_vram_gb()})
            new_rows += 1
            del inp
            torch.cuda.empty_cache()
            log(f"[{len(w.done)}/{total}] {name} L={L} b={b} r{r}: best {best} "
                f"{cells[best]['mean_ms'] if best else float('nan'):.4f} ms over {sorted(int(c) for c in cells)} "
                f"({(time.time() - t_start) / 60:.1f} min)")
    w.close()
    write_json(args.out_dir / f"e24_tfla_{tag}.json", {**meta, "rows_done": len(w.done), "rows_planned": total,
                                                       "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                     time.gmtime())})
    log(f"done: {len(w.done)}/{total} rows")


# --------------------------------------------------------------------------- analysis (CPU)
def shape_table(rows):
    """Per shape: round means per chunk -> median, round CI, oracle on the medians and its tie set."""
    by_shape = defaultdict(list)
    for r in rows:
        by_shape[(r["layout"], r["L"], r["batch"])].append(r)
    out = {}
    for key, rs in by_shape.items():
        chunks = sorted({int(c) for r in rs for c in r["cells"]})
        med, lo, hi = {}, {}, {}
        for c in chunks:
            means = [r["cells"][str(c)]["mean_ms"] for r in rs if str(c) in r["cells"]]
            if len(means) != len(rs):
                continue
            med[c] = statistics.median(means)
            if len(means) > 1:
                half = t_crit(len(means) - 1) * statistics.stdev(means) / math.sqrt(len(means))
                m = statistics.fmean(means)
                lo[c], hi[c] = m - half, m + half
            else:
                lo[c] = hi[c] = means[0]
        if not med:
            continue
        o = min(med, key=med.get)
        out[key] = {"layout": key[0], "L": key[1], "batch": key[2], "heads": rs[0]["heads"], "dqk": rs[0]["dqk"],
                    "dv": rs[0]["dv"], "rounds": len(rs), "median_ms": med, "oracle": o,
                    "tied": sorted(c for c in med if lo[c] <= hi[o]),
                    "spread": max(med.values()) / med[o]}
    return out


def regret_of(table, choose):
    """Mean / p95 / max percent regret and tied share of a chunk choice per shape (None = not applicable)."""
    regs, tied, n = [], 0, 0
    for s in table.values():
        c = choose(s)
        if c is None or c not in s["median_ms"]:
            continue
        n += 1
        regs.append(100.0 * (s["median_ms"][c] / s["median_ms"][s["oracle"]] - 1.0))
        tied += c in s["tied"]
    if not regs:
        return None
    regs.sort()
    return {"n": n, "mean_pct": statistics.fmean(regs), "p95_pct": regs[min(len(regs) - 1, int(0.95 * len(regs)))],
            "max_pct": regs[-1], "tied_share": tied / n}


def best_table(table, keyfn):
    """Chunk per key value that minimizes the summed relative latency over the shapes with that value."""
    groups = defaultdict(list)
    for s in table.values():
        groups[keyfn(s)].append(s)
    best = {}
    for k, ss in groups.items():
        common = set.intersection(*(set(s["median_ms"]) for s in ss))
        best[k] = min(common, key=lambda c: sum(s["median_ms"][c] / s["median_ms"][s["oracle"]] for s in ss))
    return lambda s: best[keyfn(s)]


def features(s, c):
    """Cost-model terms of one (shape, chunk): launch, intra-chunk quadratic work, state materialization,
    sequential chunk steps of the recurrent kernel, linear streaming work."""
    b, nh, L, dqk, dv = s["batch"], s["heads"], s["L"], s["dqk"], s["dv"]
    c_inter = min(128, c)
    return [1.0,
            b * nh * L * c * (dqk + dv) * 1e-12,          # intra-chunk attention-like work, O(L C)
            b * nh * (L / c_inter) * dqk * dv * 1e-9,     # chunk-state materialization
            L / c_inter * 1e-3,                           # sequential steps of the recurrent kernel
            b * nh * L * (dqk + dv) * 1e-9]               # streaming of q, k, v and h


def fit_nnls(rows):
    import numpy as np
    from scipy.optimize import nnls

    X = np.array([f for f, _ in rows])
    y = np.array([t for _, t in rows])
    wgt = 1.0 / y                                       # relative (log-like) error
    coef, _ = nnls(X * wgt[:, None], y * wgt)
    return coef


def model_analysis(table):
    import numpy as np

    keys = sorted(table)
    pts = {k: [(features(table[k], c), table[k]["median_ms"][c]) for c in table[k]["median_ms"]] for k in keys}
    coef_all = fit_nnls([p for k in keys for p in pts[k]])
    pred_choice = {}
    rel_err = []
    for k in keys:                                   # leave one shape out
        coef = fit_nnls([p for kk in keys if kk != k for p in pts[kk]])
        s = table[k]
        pred = {c: float(np.dot(coef, features(s, c))) for c in s["median_ms"]}
        pred_choice[k] = min(pred, key=pred.get)
        rel_err += [abs(pred[c] / s["median_ms"][c] - 1.0) for c in s["median_ms"]]
    reg = regret_of(table, lambda s: pred_choice[(s["layout"], s["L"], s["batch"])])
    return {"coef_all": [float(x) for x in coef_all],
            "terms": ["launch_ms", "intra_chunk_work", "state_materialization", "sequential_steps", "streaming"],
            "loso_regret": reg, "loso_median_rel_error": float(np.median(rel_err)),
            "loso_p90_rel_error": float(np.percentile(rel_err, 90))}


def analyze(args):
    tag = args.tag
    rows = [r for r in read_jsonl(args.out_dir / f"e24_tfla_{tag}.jsonl") if r.get("kind") == "timing"]
    table = shape_table(rows)
    chunks = sorted({c for s in table.values() for c in s["median_ms"]})
    statics = {c: regret_of(table, lambda s, c=c: c) for c in chunks}
    statics = {c: v for c, v in statics.items() if v is not None}
    full = [c for c in chunks if statics.get(c, {}).get("n") == len(table)]
    best_static = min(full, key=lambda c: statics[c]["mean_pct"]) if full else None
    oracles = defaultdict(int)
    for s in table.values():
        oracles[s["oracle"]] += 1
    res = {"experiment": "E24-analysis", "tag": tag, "n_rows": len(rows), "n_shapes": len(table),
           "chunks": chunks, "oracle_counts": dict(sorted(oracles.items())),
           "n_distinct_oracles": len(oracles),
           "spread": {"median": statistics.median(s["spread"] for s in table.values()),
                      "max": max(s["spread"] for s in table.values())},
           "library_default": {"chunk": LIBRARY_DEFAULT_CHUNK, **(statics.get(LIBRARY_DEFAULT_CHUNK) or {})},
           "best_static": {"chunk": best_static, **(statics.get(best_static) or {})},
           "static": {str(c): v for c, v in statics.items()},
           "table_by_length": regret_of(table, best_table(table, lambda s: s["L"])),
           "table_by_batch": regret_of(table, best_table(table, lambda s: s["batch"])),
           "table_by_layout": regret_of(table, best_table(table, lambda s: s["layout"])),
           "cost_model": model_analysis(table) if len(table) >= 6 else None,
           "shapes": [{**{k: v for k, v in s.items() if k != "median_ms"},
                       "median_ms": {str(c): t for c, t in s["median_ms"].items()}} for s in table.values()]}
    write_json(args.out_dir / f"e24_tfla_{tag}_analysis.json", res)
    print(f"E24 {tag}: {len(table)} shapes, oracle counts {res['oracle_counts']}, spread median "
          f"{res['spread']['median']:.2f}x; default 128 mean regret "
          f"{res['library_default'].get('mean_pct', float('nan')):.2f}%; best static {best_static} "
          f"{res['best_static'].get('mean_pct', float('nan')):.2f}%"
          + (f"; cost model (leave one shape out) {res['cost_model']['loso_regret']['mean_pct']:.2f}%"
             if res["cost_model"] and res["cost_model"]["loso_regret"] else ""))
    return res


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layouts", default=DEFAULT_LAYOUTS, help="name:heads:qk_dim:v_dim,...")
    p.add_argument("--lengths", default="512,1024,2048,3072,4096,6144,8192")
    p.add_argument("--batches", default="1,4,16")
    p.add_argument("--chunks", default=DEFAULT_CHUNKS)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--pre-spin-ms", type=float, default=0.0,
                   help="GPU-side spin before every timing block (device time); 0 = launch-inclusive")
    p.add_argument("--exclusive-guard", action="store_true",
                   help="pause while other processes use the GPU and re-time any cell during which one appeared")
    p.add_argument("--guard-until", type=float, default=None, help="epoch seconds after which the guard gives up")
    p.add_argument("--max-rows", type=int, default=None, help="stop after this many new rows (smoke)")
    p.add_argument("--analyze", action="store_true", help="only write the analysis of an existing table (CPU)")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--tag", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.analyze:
        if not args.tag:
            from src.experiments.tpds.common import host_name
            args.tag = host_name()
        analyze(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
