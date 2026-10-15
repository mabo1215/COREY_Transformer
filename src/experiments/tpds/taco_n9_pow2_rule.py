"""Review item N9 (taco-r1): the length-capped default rule C = min(C_default, nextpow2(L)) on the graph timings.

A chunk larger than the sequence pads every call shorter than the chunk to a full chunk. A simple fix that
needs no tuning is to cap the default at the next power of two of the length (and at the smallest supported
chunk from below). This script scores that rule on the CUDA-graph replays of the production kernels (RTX 3090,
three rounds, round medians; ``e22_graph_analysis`` tables ``ssd370m_graph``, ``ssd2p7b_graph``,
``gdn1p4b_graph``) and, for reference, on the device-time tables of the same calls, next to the library default
and the held-out exact-shape table of ``taco_report_numbers.json``. With ``--pro6000`` the RTX PRO 6000 graph replays
of E26 (``taco_e26_analysis`` tables ``*_graph_pro6000``) are scored as well.

CPU only; reads stored result files. Output: ``taco_n9_pow2_rule.json`` in the results directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.experiments.tpds import e22_graph_analysis as E22
from src.experiments.tpds.common import RESULTS_DIR, utc_now, write_json

TABLES = ("ssd370m_graph", "ssd2p7b_graph", "gdn1p4b_graph", "ssd370m_device", "ssd2p7b_device", "gdn1p4b_device")
PRO6000_TABLES = ("ssd370m_graph_pro6000", "ssd2p7b_graph_pro6000", "gdn1p4b_graph_pro6000")


def capped_rule(t: "E22.Table") -> np.ndarray:
    """Index of min(C_default, nextpow2(L)), rounded up to the smallest available chunk."""
    chunks = np.array(t.chunks)
    target = np.minimum(int(t.default), 2 ** np.ceil(np.log2(np.maximum(t.L, 1))).astype(int))
    target = np.maximum(target, chunks.min())
    idx = np.array([int(np.where(chunks >= c)[0][0]) for c in target])
    return idx


def stats(t: "E22.Table", pick: np.ndarray, mask: np.ndarray) -> dict:
    lat = t.lat[np.arange(len(pick)), pick]
    ok = mask & np.isfinite(lat)
    reg = lat[ok] / t.orc[ok] - 1.0
    excess = (lat[ok] - t.orc[ok]).sum()
    return {"n": int(ok.sum()), "mean_regret_pct": float(100 * reg.mean()),
            "p95_regret_pct": float(100 * np.percentile(reg, 95)), "max_regret_pct": float(100 * reg.max()),
            "within_2pct_share": float((reg <= 0.02).mean()),
            "lw_share_of_oracle_pct": float(100 * excess / t.orc[ok].sum()),
            "ms_per_forward_pass": float(1e3 * excess / ok.sum() * t.layers / 1e3) if ok.any() else None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--pro6000", action="store_true", help="also score the RTX PRO 6000 graph replays (E26)")
    args = p.parse_args(argv)
    E22.use_three_rounds(args.in_dir)
    tables = list(TABLES)
    if args.pro6000:
        from src.experiments.tpds import taco_e26_analysis as E26
        registered = E26.register(args.in_dir, "pro6000", "pro6000")
        tables += [t for t in PRO6000_TABLES if t in registered]
    out = {"experiment": "taco-r1 N9 length-capped default rule", "created_utc": utc_now(),
           "rule": "C = max(min chunk, min(C_default, nextpow2(L)))", "tables": {}}
    for name in tables:
        t = E22.Table(name, args.in_dir)
        t.set_floor()
        rule = capped_rule(t)
        default = np.full(len(rule), t.default_idx())
        rec = {"timing": t.timing, "checkpoint": t.checkpoint, "layers": t.layers, "default_chunk": t.default,
               "floor_ms": t.floor_ms, "share_rule_differs_from_default": float((rule != default).mean())}
        for sub, mask in t.masks().items():
            rec[sub] = {"default": stats(t, default, mask), "capped_rule": stats(t, rule, mask)}
        out["tables"][name] = rec
        a = rec["all"]
        print(f"{name:16s} default {a['default']['mean_regret_pct']:6.2f}% (lw {a['default']['lw_share_of_oracle_pct']:.2f}%,"
              f" {a['default']['ms_per_forward_pass']:.3f} ms/fp) | capped {a['capped_rule']['mean_regret_pct']:6.2f}%"
              f" (lw {a['capped_rule']['lw_share_of_oracle_pct']:.2f}%, {a['capped_rule']['ms_per_forward_pass']:.3f} ms/fp);"
              f" rule differs on {100 * rec['share_rule_differs_from_default']:.0f}% of calls")
    write_json(args.in_dir / "taco_n9_pow2_rule.json", out)


if __name__ == "__main__":
    main()
