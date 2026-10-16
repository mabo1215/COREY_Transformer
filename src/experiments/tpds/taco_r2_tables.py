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
    out = {"models": {}}
    for lab in labels:
        p = RESULTS_DIR / f"e32_model_graph_{tags[0]}_{lab}_{tags[1]}_{lab}.json"
        if not p.exists():
            p = RESULTS_DIR / f"e32_model_graph_{tags[0]}_{lab}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        out["models"][lab] = {"summary": d["summary"].get(lab), "file": p.name}
        print(lab, json.dumps(d["summary"].get(lab, {}), indent=None)[:1500])
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
