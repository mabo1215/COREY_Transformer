"""Summary of the Nsight Compute counters of the mechanism cells (E35b, taco-r2 M5e; CPU only).

Reads ``e35_counters_<tag>.json`` (one row per profiled kernel, joined with its cell) and reports, per kernel
family and cell: duration, achieved and theoretical occupancy, warps per scheduler, issue-slot utilization, the
three largest warp-stall reasons (warps stalled per issued instruction), local-memory (spill) sectors loaded and
stored, L1 and L2 hit rates and DRAM traffic. For the Triton scan it also compares the default launch table with
the fixed (BLOCK_D, num_warps) = (4, 2) configuration and the two buffer placements of one cell.

Output: ``taco_r2_counters.json`` in the results directory, and a printed digest.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from src.experiments.tpds.common import RESULTS_DIR

STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active.ratio"


def digest(r: dict) -> dict:
    stalls = {k[len(STALL_PREFIX):-len(STALL_SUFFIX)]: r[k] for k in r
              if k.startswith(STALL_PREFIX) and k.endswith(STALL_SUFFIX) and r[k] is not None}
    top = sorted(stalls.items(), key=lambda kv: -kv[1])[:3]
    g = lambda k: r.get(k)  # noqa: E731
    return {"kernel": r["kernel_name"][:60], "us": (g("gpu__time_duration.sum") or 0) / 1e3,
            "grid": g("launch__grid_size"), "block": g("launch__block_size"),
            "regs": g("launch__registers_per_thread"),
            "occ_achieved_pct": g("sm__warps_active.avg.pct_of_peak_sustained_active"),
            "occ_theoretical_pct": g("sm__maximum_warps_per_active_cycle_pct"),
            "warps_per_scheduler": g("smsp__warps_active.avg.per_cycle_active"),
            "issue_active_pct": g("smsp__issue_active.avg.pct_of_peak_sustained_active"),
            "waves": g("launch__waves_per_multiprocessor"),
            "top_stalls": [(k, round(v, 2)) for k, v in top],
            "local_ld_sectors": g("l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"),
            "local_st_sectors": g("l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"),
            "l1_hit_pct": g("l1tex__t_sector_hit_rate.pct"), "l2_hit_pct": g("lts__t_sector_hit_rate.pct"),
            "dram_mb": ((g("dram__bytes_read.sum") or 0) + (g("dram__bytes_write.sum") or 0)) / 1e6}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="4c_e35")
    a = ap.parse_args(argv)
    d = json.loads((RESULTS_DIR / f"e35_counters_{a.tag}.json").read_text())
    cells = defaultdict(list)
    for r in d["rows"]:
        cells[(r["family"], r["L"], r["batch"], r["chunk"], r["launch"], r["placement"])].append(digest(r))
    out = {"source": f"e35_counters_{a.tag}.json", "cells": [{"family": k[0], "L": k[1], "batch": k[2], "chunk": k[3],
                                                             "launch": k[4], "placement": k[5], "kernels": v}
                                                            for k, v in sorted(cells.items())]}
    (RESULTS_DIR / "taco_r2_counters.json").write_text(json.dumps(out, indent=1))
    for c in out["cells"]:
        if c["family"] == "m1" and c["batch"] == 1 and c["launch"] in ("default", "d4_w2"):
            k = c["kernels"][0]
            print(f"m1 L={c['L']} c={c['chunk']} {c['launch']}/{c['placement']}: {k['us']:.1f} us grid {k['grid']} "
                  f"regs {k['regs']} occ {k['occ_achieved_pct']:.1f}/{k['occ_theoretical_pct']:.1f}% "
                  f"warps/sched {k['warps_per_scheduler']:.2f} issue {k['issue_active_pct']:.1f}% "
                  f"spill ld/st {k['local_ld_sectors']:.0f}/{k['local_st_sectors']:.0f} L2 {k['l2_hit_pct']:.1f}% "
                  f"stalls {k['top_stalls']}")
    for c in out["cells"]:
        if c["family"] in ("ssd", "gdn") and c["L"] == 4096:
            print(f"{c['family']} L={c['L']} c={c['chunk']}:")
            for k in sorted(c["kernels"], key=lambda k: -k["us"])[:3]:
                print(f"   {k['kernel'][:34]:34s} {k['us']:7.1f} us occ {k['occ_achieved_pct']:.1f}% "
                      f"issue {k['issue_active_pct']:.1f}% stalls {k['top_stalls'][:2]} L2 {k['l2_hit_pct']:.1f}%")


if __name__ == "__main__":
    main()
