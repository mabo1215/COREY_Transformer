"""Ties as the share within 2% of the oracle (taco-r1, review items N7 and M9e).

The manuscript's convention is that a chunk is tied with the oracle when its latency is within 2% of the
oracle latency. Two places still used overlapping 95% confidence intervals; this script recomputes them with
the 2% margin:

  value invariance   per kernel and length (RTX PRO 6000, E1): share of (input, round) cells in which the
                     input-agnostic chunk C* (median over inputs and rounds, per perturbation mode) is within 2%
                     of the cell's own best chunk; minimum over the two modes (the column "Tied" of the
                     value-invariance table)
  per-call oracle    mean number of chunks within 2% of the per-call oracle (oracle included) on the captured
                     370M calls of the RTX PRO 6000: Mamba-1 in device time on the round medians of three rounds,
                     Mamba-2 launch-inclusive in one block (the timings of the oracle-regret table)

CPU only; reads stored result files. Output: ``taco_within2_ties.json`` in the results directory.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.experiments.tpds.analysis import chunk_columns, latency_matrix, load_e2, round_median_view
from src.experiments.tpds.common import RESULTS_DIR, utc_now, write_json

MARGIN = 0.02


def invariance_ties(e1: dict) -> dict:
    cells: dict[tuple, dict] = defaultdict(dict)
    for r in e1["rows"]:
        if r.get("kind", "timing") not in ("timing", None) and "mean_ms" not in r:
            continue
        cells[(r["family"], r["L"], r["mode"], r["input"], r["round"])][r["chunk"]] = r["mean_ms"]
    groups: dict[tuple, list] = defaultdict(list)
    for (fam, L, mode, _inp, _rnd), per_chunk in cells.items():
        groups[(fam, L, mode)].append(per_chunk)
    per_mode = {}
    for (fam, L, mode), members in groups.items():
        chunks = sorted(set.intersection(*[set(m) for m in members]))
        agg = {c: statistics.median(m[c] for m in members) for c in chunks}
        c_star = min(agg, key=agg.get)
        within = sum(m[c_star] <= (1 + MARGIN) * min(m[c] for c in chunks) for m in members)
        per_mode[(fam, L, mode)] = {"c_star": c_star, "n_cells": len(members), "share_within_2pct": within / len(members)}
    out = {}
    for (fam, L, mode), rec in sorted(per_mode.items()):
        key = f"{fam}|L{L}"
        o = out.setdefault(key, {"family": fam, "L": L, "modes": {}})
        o["modes"][mode] = rec
    for o in out.values():
        o["min_over_modes_share_within_2pct"] = min(m["share_within_2pct"] for m in o["modes"].values())
    total = sum(m["n_cells"] for o in out.values() for m in o["modes"].values())
    within = sum(m["n_cells"] * m["share_within_2pct"] for o in out.values() for m in o["modes"].values())
    return {"per_kernel_length": out, "pooled_share_within_2pct": within / total, "n_cells": total}


def oracle_tie_count(path: Path, round_median: bool) -> dict:
    df = load_e2([path])
    if round_median:
        df = round_median_view(df)
    chunks = chunk_columns(df)
    lat = latency_matrix(df, chunks)
    lat = np.where(np.isfinite(lat), lat, np.inf)
    orc = lat.min(1)
    n_within = (lat <= (1 + MARGIN) * orc[:, None]).sum(1)
    return {"file": path.name, "round_median": round_median, "n_calls": int(len(orc)),
            "mean_chunks_within_2pct": float(n_within.mean()),
            "share_calls_single_chunk_within_2pct": float((n_within == 1).mean())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv)
    e1 = json.loads((args.in_dir / "e1_pro6000.json").read_text())
    res = {"experiment": "taco-r1 within-2% ties", "created_utc": utc_now(), "margin": MARGIN,
           "value_invariance_pro6000": invariance_ties(e1),
           "oracle_ties_pro6000": {
               "mamba1_device_round_medians": oracle_tie_count(args.in_dir / "e2_mamba1_pro6000_r4spin.csv", True),
               "mamba2_launch_inclusive": oracle_tie_count(args.in_dir / "e2_mamba2_pro6000.csv", False)}}
    write_json(args.in_dir / "taco_within2_ties.json", res)
    for k, o in res["value_invariance_pro6000"]["per_kernel_length"].items():
        print(f"{k:16s} C* {[m['c_star'] for m in o['modes'].values()]} within 2% (min over modes) "
              f"{100 * o['min_over_modes_share_within_2pct']:.0f}%")
    print(f"pooled {100 * res['value_invariance_pro6000']['pooled_share_within_2pct']:.1f}% of "
          f"{res['value_invariance_pro6000']['n_cells']} cells")
    for k, v in res["oracle_ties_pro6000"].items():
        print(k, f"mean chunks within 2%: {v['mean_chunks_within_2pct']:.2f}")


if __name__ == "__main__":
    main()
