"""E26 analysis (taco-r1, review item M6a): CUDA-graph replays of the captured calls on the RTX PRO 6000.

Applies the analysis of the RTX 3090 graph tables (``taco_report_numbers.graph_set``: per-call and
latency-weighted regret of the library default, the best static chunk, the held-out shape tree and exact-shape
table and, for the multi-kernel pipelines, the work-term model with its eager calibration and refitted on the
graph timings; launch floor and share of calls above it; prompt-cluster bootstrap intervals) to the three-round
RTX PRO 6000 replays of the same calls (round-median view):

  ssd370m_graph_pro6000   e2_mamba2_<tag>_taco_graph3.csv        (E26a)
  gdn1p4b_graph_pro6000   e2_gdn_<tag>_1p4b_taco_graph3.csv      (E26b)
  ssd2p7b_graph_pro6000   e2_mamba2_<tag>_2p7b_taco_graph3.csv   (E26c)
  m1_370m_graph_pro6000   e2_mamba1_<tag>_taco_graph3.csv        (E26d)

Tables whose file is missing are skipped. Also prints the RTX 3090 rows of ``taco_report_numbers.json`` next to
them. CPU only. Output: ``taco_e26_analysis_<tag>.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiments.tpds import e22_graph_analysis as E22
from src.experiments.tpds import taco_report_numbers as TRN
from src.experiments.tpds.common import RESULTS_DIR, utc_now, write_json

# name: (csv pattern, family, checkpoint, layers, default chunk, refit work terms)
SPECS = {
    "ssd370m_graph_pro6000": ("e2_mamba2_{tag}_taco_graph3.csv", "mamba2", "Mamba-2 370M", 48, 256, True),
    "gdn1p4b_graph_pro6000": ("e2_gdn_{tag}_1p4b_taco_graph3.csv", "gdn", "Gated DeltaNet 1.4B hybrid", 12, 64, True),
    "ssd2p7b_graph_pro6000": ("e2_mamba2_{tag}_2p7b_taco_graph3.csv", "mamba2", "Mamba-2 2.7B", 64, 256, True),
    "m1_370m_graph_pro6000": ("e2_mamba1_{tag}_taco_graph3.csv", "mamba1", "Mamba-1 370M (Triton scan)", 48, None, False),
}
RTX3090 = {"ssd370m_graph_pro6000": "ssd370m_graph", "gdn1p4b_graph_pro6000": "gdn1p4b_graph",
           "ssd2p7b_graph_pro6000": "ssd2p7b_graph", "m1_370m_graph_pro6000": "m1_370m_graph"}


def register(in_dir: Path, tag: str, gpu: str) -> list[str]:
    names = []
    for nm, (pat, fam, ckpt, layers, default, refit) in SPECS.items():
        csv = pat.format(tag=tag)
        if not (in_dir / csv).exists():
            continue
        E22.TABLES[nm] = (csv, fam, ckpt, "CUDA graph, warm L2, three rounds (round medians), RTX PRO 6000",
                          gpu, layers, default, None, True, refit)
        names.append(nm)
    return names


def headline(rec: dict) -> dict:
    """Default, shape tree, exact table and work-term rows of one table (all calls)."""
    pol = rec["subsets"]["all"]["policies"]
    keep = ("default", "best_static_in_sample", "shape_tree_held_out", "exact_table_held_out",
            "work_term_model_eager_calibration", "work_term_model_refit_on_graph_timings_cv")
    return {k: {f: pol[k].get(f) for f in ("mean_regret_pct", "p95_regret_pct", "max_regret_pct", "within_2pct_share",
                                           "lw_share_of_oracle_pct", "ms_per_forward_pass",
                                           "oracle_scan_ms_per_forward_pass")}
            for k in keep if k in pol}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--tag", default="pro6000", help="host tag of the E26 result files")
    p.add_argument("--gpu", default="pro6000", help="host whose stored work-term calibration is used")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    names = register(args.in_dir, args.tag, args.gpu)
    if not names:
        raise SystemExit(f"no E26 result files with tag {args.tag} in {args.in_dir}")
    checks, cache = TRN.Checks(), {}
    out = {"experiment": "E26-analysis", "created_utc": utc_now(), "tag": args.tag, "n_boot": args.n_boot,
           "tables": {}, "headline": {}, "rtx3090_headline": {}}
    ref = TRN._load_json(args.in_dir / "taco_report_numbers.json") if hasattr(TRN, "_load_json") else None
    for nm in names:
        rec = TRN.graph_set(nm, args, checks, cache)
        rec["gpu"] = "RTX PRO 6000"
        out["tables"][nm] = rec
        out["headline"][nm] = {"floor_us": rec["floor_us"], "share_above_floor": rec["share_above_floor"],
                               "n_calls": rec["n_calls"], "default_chunk": rec["default_chunk"], **headline(rec)}
        if ref:
            r3 = ref.get("M4b_latency_weighted", {}).get("a_cuda_graph", {}).get(RTX3090[nm])
            if r3:
                out["rtx3090_headline"][nm] = {"floor_us": r3["floor_us"], "share_above_floor": r3["share_above_floor"],
                                               **headline(r3)}
        h = out["headline"][nm]
        d = h["default"]
        msg = (f"{nm:24s} floor {h['floor_us']:.0f} us, {100 * h['share_above_floor']:.0f}% above | default "
               f"{d['mean_regret_pct']:.1f}/{d['p95_regret_pct']:.1f}% lw {d['lw_share_of_oracle_pct']:.2f}% "
               f"{d['ms_per_forward_pass']:.3f} ms (scan {d['oracle_scan_ms_per_forward_pass']:.1f} ms)")
        for k in ("shape_tree_held_out", "exact_table_held_out", "work_term_model_eager_calibration",
                  "work_term_model_refit_on_graph_timings_cv"):
            if k in h:
                msg += f" | {k.split('_held')[0].replace('work_term_model_', 'wt_')} {h[k]['mean_regret_pct']:.2f}%"
        print(msg)
    out["checks_summary"] = checks.summary()
    write_json(args.out or args.in_dir / f"taco_e26_analysis_{args.tag}.json", E22._clean(out))


if __name__ == "__main__":
    main()
