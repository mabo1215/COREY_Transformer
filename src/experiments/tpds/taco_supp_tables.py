"""LaTeX tables of the supplement for the TACO revision (review items M4c, M5a, M5b).

Reads ``taco_model_baselines.json`` (top-1/2/3 of every ranker, learning curve over the number of
calibration keys) and ``taco_report_numbers.json`` (paired prompt-cluster bootstrap intervals of the
cost model minus each selector) and prints two tables to stdout:

  tab:supp_topk_all      top-1 / top-2 / top-3 mean regret of every calibration-only ranker in device time
                         on the three GPUs, with the paired difference cost model minus ranker (top-1)
  tab:supp_learning      cost-model mean regret versus the number of calibration keys (mean, min and max
                         over key subsets) next to the nearest-calibration-key table

Device-time tags: pro6000_r4spin (RTX PRO 6000, round medians), vgpu3090_r3spin (RTX 3090, single block),
h800_r3spin (H800, single block); paired intervals use the selector tables of ``taco_report_numbers``
(pro6000_device, vgpu3090_device, h800_device).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiments.tpds.common import RESULTS_DIR

TAGS = ("pro6000_r4spin", "vgpu3090_r3spin", "h800_r3spin")
PAIRED = ("pro6000_device", "vgpu3090_device", "h800_device")
RANKERS = [("cost_model", "Cost model, Eq.~(\\Mref{eq:cost_model})", "cost_model"),
           ("work_only", "Three-parameter restriction", "work_only"),
           ("compile_facts_rule", "Compile-facts rule (no parameters)", None),
           ("nearest_calibrated_key", "Nearest calibration key", "nearest_calibrated_key"),
           ("loglinear_interp_logL_logB", "Log-linear interpolation", "loglinear_interp_logL_logB"),
           ("length_tree_calibration", "Length tree", "length_tree_calibration"),
           ("gbr_structural", "Gradient boosting, struct.", "gbr_structural"),
           ("mlp_basic_mean_over_seeds", "MLP (mean of 5 seeds)", None),
           ("mlp_structural_mean_over_seeds", "MLP, struct.\\ (mean of 5 seeds)", None),
           ("tfla_runtime", "TFLA expression", "tfla_runtime"),
           ("random_k", "Random chunks", None)]


def topk_cell(rec, k):
    v = rec.get(f"top{k}")
    if v is None:
        return "--"
    m = v[0] if isinstance(v, list) else v.get("mean", v)
    if isinstance(m, dict):
        m = m.get("mean")
    return f"{m:.2f}" if m < 10 else f"{m:.1f}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv)
    B = json.loads((args.in_dir / "taco_model_baselines.json").read_text())
    A = json.loads((args.in_dir / "taco_report_numbers.json").read_text())
    st = B["M5a"]["summary_table"]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Calibration-only rankers with the $k$ best-ranked chunks profiled per exact shape (device "
             r"time; RTX PRO~6000: round medians, RTX~3090 and H800: one block): mean per-call relative regret "
             r"(\%) at top-1 / top-2 / top-3, and the paired difference cost model minus ranker at top-1 (points, "
             r"95\% prompt-cluster bootstrap interval; RTX~3090 and H800 on the selector tables of the main paper). "
             r"Random chunks: expectation over random rankings.}",
             r"\label{tab:supp_topk_all}", r"\scriptsize", r"\setlength{\tabcolsep}{2pt}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{RTX PRO~6000} & \multicolumn{2}{c}{RTX~3090} & \multicolumn{2}{c}{H800} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Ranker & top-1/2/3 & Paired & top-1/2/3 & Paired & top-1/2/3 & Paired \\", r"\midrule"]
    for key, label, pkey in RANKERS:
        cells = [label]
        for tag, ptag in zip(TAGS, PAIRED):
            rec = st[tag].get(key)
            cells.append("--" if rec is None else " / ".join(topk_cell(rec, k) for k in (1, 2, 3)))
            if pkey is None or pkey == "cost_model":
                cells.append("--")
            else:
                pm = A["M4a_selector_tables"]["mamba1"][ptag]["paired_cost_model_minus"]["all_calls"]
                r = pm.get(f"cost_model - {pkey}")
                cells.append("--" if r is None else
                             f"{r['diff_mean_regret_pp']:+.2f} [{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}]")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    lc = B["M5b"]["learning_curve_summary_mean_regret_pct"]
    lines += [r"\begin{table}[t]", r"\centering",
              r"\caption{Learning curve of the cost model over the number of calibration keys (device time): mean "
              r"per-call relative regret (\%) on the captured calls, mean over random subsets of $k$ of the twelve "
              r"tuning keys (minimum--maximum over subsets), next to the table of the nearest calibrated key.}",
              r"\label{tab:supp_learning}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
              r"\begin{tabular}{rrrrrrr}", r"\toprule",
              r" & \multicolumn{2}{c}{RTX PRO~6000} & \multicolumn{2}{c}{RTX~3090} & \multicolumn{2}{c}{H800} \\",
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
              r"Keys & Model & Nearest & Model & Nearest & Model & Nearest \\", r"\midrule"]
    for k in ("1", "2", "3", "4", "6", "8", "10", "12"):
        cells = [k]
        for tag in TAGS:
            m, lo, hi, nk = lc[tag][k]
            cells += [f"{m:.2f} ({lo:.2f}--{hi:.2f})", f"{nk:.1f}"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
