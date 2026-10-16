"""LaTeX rows of the taco-r2 main-text tables from the stored E32-E36 analyses (CPU only).

  e33   value invariance in graph replay (Table "Values versus placement")
  e32   whole prefill step as one CUDA graph (Table "model_graph")
Each section prints the rows and the summary numbers the text quotes; ``--json`` also writes them to
``taco_r2_table_numbers.json`` in the results directory.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from src.experiments.tpds.common import RESULTS_DIR

LABEL = {"m1triton": "Triton scan", "ssd370m": "SSD 370M", "ssd2p7b": "SSD 2.7B", "gdn1p4b": "GDN 1.4B"}
ORDER = ("m1triton", "ssd370m", "ssd2p7b", "gdn1p4b")


def _f(x, d=1):
    return "--" if x is None else f"{x:.{d}f}"


def e33(tags: list[str]) -> dict:
    """Per (family, L, B): chunk spread, value spread at the best chunk, real vs Gaussian, best-chunk changes, placement."""
    out = {"rows": [], "summary": {}}
    cfg, plc, flips = [], [], {}
    for t in tags:
        p = RESULTS_DIR / f"e33_value_graph_{t}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        cfg += d["configs"]
        plc += d["placements"]
        flips.update(d.get("best_chunk_by_input", {}))
    shapes = sorted({(c["family"], c["L"], c["batch"]) for c in cfg},
                    key=lambda k: (ORDER.index(k[0]) if k[0] in ORDER else 9, k[2], k[1]))
    for fam, L, B in shapes:
        cs = {c["chunk"]: c for c in cfg if (c["family"], c["L"], c["batch"]) == (fam, L, B)}
        g0 = {ch: c["median_ms"]["gaussian#0"] for ch, c in cs.items()}
        best = min(g0, key=lambda ch: (g0[ch], ch))
        c = cs[best]
        fl = flips.get(f"{fam}|L{L}|B{B}", {})
        pl = [p for p in plc if (p["family"], p["L"], p["batch"], p["chunk"]) == (fam, L, B, best)]
        row = {"family": fam, "L": L, "batch": B, "best_chunk": best,
               "chunk_spread": max(g0.values()) / min(g0.values()),
               "values_spread_pct": c["spread_inputs_pct"], "draws_spread_pct": c["spread_between_draws_pct_max"],
               "dist_spread_pct": c["spread_distributions_pct"],
               "real_vs_gaussian_pct": c["real_vs_gaussian0_pct"], "round_cv_pct": 100 * c["round_cv_median"],
               "tost1": f"{c['tost_equivalent_1pct']}/{c['n_tost']}", "tost2": f"{c['tost_equivalent_2pct']}/{c['n_tost']}",
               "friedman_p": c["friedman"].get("p"),
               "n_inputs_other_best": fl.get("n_inputs_other_best"), "n_inputs": len(fl.get("best_by_input", {})),
               "regret_gauss_choice_max_pct": fl.get("regret_of_gaussian_choice_pct_max"),
               "placement_spread_pct": pl[0]["spread_placements_pct"] if pl else None,
               "max_values_spread_any_chunk_pct": max(x["spread_inputs_pct"] for x in cs.values()),
               "max_dist_spread_any_chunk_pct": max(x["spread_distributions_pct"] for x in cs.values())}
        out["rows"].append(row)
        print(f"{LABEL.get(fam, fam)} & {L}$\\times${B} & {best} & ${row['chunk_spread']:.2f}\\times$ & "
              f"{_f(row['values_spread_pct'])} ({_f(row['draws_spread_pct'])}) & {_f(row['real_vs_gaussian_pct'], 1)} & "
              f"{row['tost2']} & {row['n_inputs_other_best']}; {_f(row['regret_gauss_choice_max_pct'])} & "
              f"{_f(row['placement_spread_pct'])} \\\\   % cv {row['round_cv_pct']:.2f}%, any chunk values "
              f"{row['max_values_spread_any_chunk_pct']:.1f} dist {row['max_dist_spread_any_chunk_pct']:.1f}")
    return out


def e32(labels: list[str], tags: tuple[str, str]) -> dict:
    """Whole-prefill CUDA-graph rows: prefill range, scan share, table and best gains (graph mode), sessions."""
    out = {"models": {}}
    names = {"ssd370m": "SSD 370M", "ssd2p7b": "SSD 2.7B", "gdn1p4b": "GDN 1.4B"}
    for lab in labels:
        p2 = RESULTS_DIR / f"e32_model_graph_{tags[0]}_{lab}_{tags[1]}_{lab}.json"
        p1 = RESULTS_DIR / f"e32_model_graph_{tags[0]}_{lab}.json"
        p = p2 if p2.exists() else p1
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        cells = [c for c in d["cells"] if c["mode"] == "graph" and c["session"] == f"{tags[0]}_{lab}"]
        summ = d["summary"].get(lab, {}).get("graph", {})
        s1 = summ.get(f"{tags[0]}_{lab}", {})
        s2 = summ.get(f"{tags[1]}_{lab}", {})
        xs = summ.get("cross_session", {})
        pre = [c["median_ms"][c["default"]] for c in cells]
        row = {"label": names.get(lab, lab), "n_shapes": len(cells),
               "prefill_ms_range": [min(pre), max(pre)] if pre else None,
               "scan_share_range": s1.get("scan_share_range_pct"), "scan_share_lw": s1.get("scan_share_latency_weighted_pct"),
               "table_gain_lw": [s1.get("gain_table_latency_weighted_pct"), s2.get("gain_table_latency_weighted_pct")],
               "table_gain_range": s1.get("gain_table_range_pct"),
               "best_gain_lw": [s1.get("gain_best_latency_weighted_pct"), s2.get("gain_best_latency_weighted_pct")],
               "best_gain_range": s1.get("gain_best_range_pct"),
               "same_best": f"{xs.get('same_best_chunk')}/{xs.get('n_cells')}" if xs else None,
               "transferred_best_gain_lw": xs.get("in_session_best_table_transferred_gain_latency_weighted_pct"),
               "default_session_ratio_median_pct": xs.get("default_latency_session_ratio_abs_pct_median"),
               "capture_s": None, "graph_gb": None}
        out["models"][lab] = row
        print(json.dumps(row))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--e33-tags", default="4c_e33,4c_e33m1")
    ap.add_argument("--e32-labels", default="ssd370m,ssd2p7b,gdn1p4b")
    ap.add_argument("--e32-sessions", default="4c_e32s1,4c_e32s2")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    res = {"e33": e33(a.e33_tags.split(","))}
    res["e32"] = e32(a.e32_labels.split(","), tuple(a.e32_sessions.split(",")))
    if a.json:
        (RESULTS_DIR / "taco_r2_table_numbers.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
