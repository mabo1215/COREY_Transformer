"""E17 analysis (review item M2b): production kernels out of the launch floor, real Gated DeltaNet traces.

CPU only. Two parts:

synthetic   ``e17_workbound_<tag>.jsonl`` (E17c): per (family, shape, L, batch) the round means of every
            chunk, the best chunk on the round means, its tie set (chunks within ``--tie-pct`` of the best,
            the practical margin of the paper, and chunks whose 95% CI over rounds overlaps the best),
            the spread (slowest / fastest chunk), the agreement of the per-round argmin, and whether the
            call is above the launch floor (latency of the best chunk > ``--floor-factor`` x the empty
            kernel). Per family and shape: the regret of the single best chunk and of the chunk tuned at
            the smallest token count, and the token count from which the spread first leaves the margin.

per-call    ``e2_<model>_<tag>.csv`` (E17a, E17b): the same quantities as Table II/III of the paper for the
            captured calls (per batch: oracle != best static share, best static chunk, static regret mean /
            p95, rule regret; spread median / p95; tied-chunk count), plus the out-of-sample exact-shape
            headroom (``analysis.cv_exact_headroom``) against one host synchronization.

Output: ``e17_analysis_<tag>.json``. The E3 predictor table for the per-call files is produced by
``e3_predictors.py --models mamba2,gdn --hosts <host_suffix>``.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.experiments.tpds.common import RESULTS_DIR, read_jsonl, write_json

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def _ci(xs):
    if len(xs) < 2:
        return (xs[0], xs[0]) if xs else (math.nan, math.nan)
    m, s = statistics.fmean(xs), statistics.stdev(xs)
    h = T95.get(len(xs) - 1, 1.96) * s / math.sqrt(len(xs))
    return m - h, m + h


def synthetic(rows: list[dict], floor_ms: float | None, tie_pct: float, floor_factor: float) -> dict:
    cells = defaultdict(list)
    for r in rows:
        cells[(r["family"], r["shape"], r["L"], r["batch"])].append(r)
    per_cell = []
    for (fam, shape, L, b), rs in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] * kv[0][3])):
        chunks = sorted({int(c) for r in rs for c in r["cells"]})
        lat = {c: [r["cells"][str(c)]["mean_ms"] for r in rs if str(c) in r["cells"]] for c in chunks}
        mean = {c: statistics.fmean(v) for c, v in lat.items()}
        best = min(mean, key=mean.get)
        ci = {c: _ci(v) for c, v in lat.items()}
        tied_margin = sorted(c for c in chunks if mean[c] <= mean[best] * (1 + tie_pct / 100))
        tied_ci = sorted(c for c in chunks if ci[c][0] <= ci[best][1])
        argmins = [min(r["cells"], key=lambda c: r["cells"][c]["mean_ms"]) for r in rs]
        per_cell.append({
            "family": fam, "shape": shape, "L": L, "batch": b, "tokens": L * b, "rounds": len(rs),
            "chunks": chunks, "mean_ms": {str(c): mean[c] for c in chunks}, "best_chunk": best,
            "best_ms": mean[best], "tied_margin": tied_margin, "tied_ci": tied_ci,
            "spread": max(mean.values()) / mean[best],
            "round_argmin_agreement": sum(1 for a in argmins if int(a) == best) / len(argmins),
            "above_launch_floor": None if floor_ms is None else bool(mean[best] > floor_factor * floor_ms)})
    by_shape = defaultdict(list)
    for c in per_cell:
        by_shape[(c["family"], c["shape"])].append(c)
    shapes = []
    for (fam, shape), cs in sorted(by_shape.items()):
        cs = sorted(cs, key=lambda c: (c["tokens"], c["L"]))
        chunks = cs[0]["chunks"]
        reg = {ch: [c["mean_ms"][str(ch)] / c["best_ms"] - 1 for c in cs] for ch in chunks}
        best_static = min(chunks, key=lambda ch: statistics.fmean(reg[ch]))
        small_tuned = cs[0]["best_chunk"]
        opened = next((c for c in cs if c["spread"] > 1 + tie_pct / 100 and len(c["tied_margin"]) < len(chunks)), None)
        shapes.append({
            "family": fam, "shape": shape, "n_cells": len(cs),
            "best_chunk_counts": {str(k): sum(1 for c in cs if c["best_chunk"] == k) for k in chunks},
            "max_spread": max(c["spread"] for c in cs), "median_spread": statistics.median(c["spread"] for c in cs),
            "best_static_chunk": best_static,
            "best_static_regret_pct": {"mean": 100 * statistics.fmean(reg[best_static]),
                                       "max": 100 * max(reg[best_static])},
            "small_tuned_chunk": small_tuned,
            "small_tuned_regret_pct": {"mean": 100 * statistics.fmean(reg[small_tuned]),
                                       "max": 100 * max(reg[small_tuned])},
            "largest_chunk_regret_pct": {"mean": 100 * statistics.fmean(reg[max(chunks)]),
                                         "max": 100 * max(reg[max(chunks)])},
            "first_tokens_spread_beyond_margin": None if opened is None else opened["tokens"],
            "cells_above_launch_floor": sum(1 for c in cs if c["above_launch_floor"]),
        })
    return {"cells": per_cell, "shapes": shapes}


def per_call(csv: Path, sync_ms: float | None) -> dict:
    from src.experiments.tpds.analysis import chunk_columns, cv_exact_headroom, latency_matrix, load_e2, \
        static_regret_table

    df = load_e2([csv])
    if df.empty:
        return {"source": csv.name, "n_calls": 0}
    chunks = chunk_columns(df)
    lat = latency_matrix(df, chunks)
    spread = lat.max(1) / lat.min(1)
    tied_n = df["tied_chunks"].fillna("").map(lambda s: len([c for c in str(s).split(";") if c]))
    out = {"source": csv.name, "n_calls": int(len(df)), "chunks": chunks,
           "spread_median": float(np.median(spread)), "spread_p95": float(np.percentile(spread, 95)),
           "tied_with_oracle_mean_count": float(tied_n.mean()), "by_batch": {}}
    for b, sub in df.groupby("batch"):
        t = static_regret_table(sub, chunks)
        bs = int(t["best_static"].split("_")[1])
        out["by_batch"][int(b)] = {
            "calls": int(len(sub)),
            "oracle_ne_best_static_pct": float(100 * (sub["oracle_chunk"] != bs).mean()),
            "best_static_chunk": bs,
            "static_regret_pct": [t[t["best_static"]]["mean_regret_pct"], t[t["best_static"]]["p95_regret_pct"]],
            "rule_regret_pct": [t["entropy_rule"]["mean_regret_pct"], t["entropy_rule"]["p95_regret_pct"]]
            if "entropy_rule" in t else None,
            "largest_chunk_regret_pct": [t[f"static_{max(chunks)}"]["mean_regret_pct"],
                                         t[f"static_{max(chunks)}"]["p95_regret_pct"]],
            "oracle_distribution": {str(int(k)): int(v) for k, v in sub["oracle_chunk"].value_counts().items()},
            "median_latency_ms": float(np.median(lat[sub.index].min(1)))}
    if df["fold"].nunique() > 1:
        out["cv_exact_headroom"] = cv_exact_headroom(df, chunks, threshold_ms=sync_ms)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--tag", required=True, help="host tag of the E17c file (e.g. vgpu3090)")
    p.add_argument("--per-call", default=None,
                   help="comma list of e2 CSV names (default: e2_mamba2_<tag>_2p7b_r3.csv,e2_gdn_<tag>_1p4b_r3.csv)")
    p.add_argument("--tie-pct", type=float, default=2.0, help="practical tie margin in percent")
    p.add_argument("--floor-factor", type=float, default=2.0)
    p.add_argument("--sync-ms", type=float, default=None,
                   help="one host synchronization in ms (default: 0.0214 for vgpu3090, 0.020 otherwise)")
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = args.out_dir or args.in_dir
    sync = args.sync_ms if args.sync_ms is not None else (0.0214 if "3090" in args.tag else 0.020)
    res = {"experiment": "E17-analysis", "tag": args.tag, "tie_pct": args.tie_pct, "sync_ms": sync}
    syn = args.in_dir / f"e17_workbound_{args.tag}.jsonl"
    if syn.exists():
        meta = next(iter(read_jsonl(syn, kind="meta")), {})
        floor = (meta.get("empty_kernel") or {}).get("mean_ms")
        rows = [r for r in read_jsonl(syn) if r.get("kind") == "timing"]
        res["synthetic"] = {"source": syn.name, "n_rows": len(rows), "launch_floor_ms": floor,
                            **synthetic(rows, floor, args.tie_pct, args.floor_factor)}
    names = (args.per_call.split(",") if args.per_call else
             [f"e2_mamba2_{args.tag}_2p7b_r3.csv", f"e2_gdn_{args.tag}_1p4b_r3.csv"])
    res["per_call"] = {n: per_call(args.in_dir / n, sync) for n in names if (args.in_dir / n).exists()}
    path = out_dir / f"e17_analysis_{args.tag}.json"
    write_json(path, res)
    for s in res.get("synthetic", {}).get("shapes", []):
        print(f"{s['family']} {s['shape']}: max spread {s['max_spread']:.2f}x, best static {s['best_static_chunk']} "
              f"(regret {s['best_static_regret_pct']['mean']:.1f}% / max {s['best_static_regret_pct']['max']:.1f}%), "
              f"largest chunk {s['largest_chunk_regret_pct']['mean']:.1f}% / {s['largest_chunk_regret_pct']['max']:.1f}%, "
              f"best counts {s['best_chunk_counts']}")
    for n, pc in res["per_call"].items():
        hb = {b: (v['best_static_chunk'], round(v['static_regret_pct'][0], 1)) for b, v in pc.get("by_batch", {}).items()}
        h = pc.get("cv_exact_headroom", {}).get("covered", {})
        print(f"{n}: {pc['n_calls']} calls, spread median {pc.get('spread_median', float('nan')):.2f}x; "
              f"best static per batch {hb}; covered headroom {h.get('mean_ms')} ms (max key {h.get('max_key_ms')})")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
