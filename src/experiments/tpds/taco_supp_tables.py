"""LaTeX tables of the supplement for the TACO revision (review items M4c, M5a, M5b).

Reads ``taco_model_baselines.json`` (top-1/2/3 of every ranker, learning curve over the number of
calibration keys) and ``taco_report_numbers.json`` (paired prompt-cluster bootstrap intervals of the
cost model minus each selector) and prints two tables to stdout:

  tab:supp_topk_all      top-1 / top-2 / top-3 mean regret of every calibration-only ranker in device time
                         on the three GPUs, with the paired difference cost model minus ranker (top-1)
  tab:supp_learning      cost-model mean regret versus the number of calibration keys (mean, min and max
                         over key subsets) next to the nearest-calibration-key table
  tab:supp_params_ci     the eight fitted parameters per GPU with 95% intervals over a bootstrap of the twelve
                         calibration shapes, and the resulting captured-call regret interval (review item M3c)
  tab:supp_ci            95% prompt-cluster bootstrap intervals of the main selector table (device time, three
                         GPUs), of the CUDA-graph defaults on both GPUs and of the pooled tensor-parallel regrets
                         (shape bootstrap) (review item M4e)

``--only NAME`` prints one table.

Device-time tags: pro6000_r4spin (RTX PRO 6000, round medians), vgpu3090_r3spin (RTX 3090, single block),
h800_r3spin (H800, single block); paired intervals use the selector tables of ``taco_report_numbers``
(pro6000_device, vgpu3090_device, h800_device).
"""
from __future__ import annotations

import argparse
import json
import re
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
    p.add_argument("--only", default=None, help="print only the table with this label")
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
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    lines += params_ci_table(B) + [""] + ci_table(A, args.in_dir)
    out = "\n".join(lines)
    if args.only:
        start = out.find(r"\label{" + args.only + "}")
        if start < 0:
            raise SystemExit(f"no table {args.only}")
        b = out.rfind(r"\begin{table", 0, start)
        m = re.compile(r"\\end\{table\*?\}").search(out, start)
        out = out[b:m.end()]
    print(out)


PARAMS = [("t0", r"$t_0$ ($\mu$s)", 1e3), ("tau_m", r"$\tau_m$ ($\mu$s)", 1e3), ("tau_e", r"$\tau_e$ (ns)", 1e6),
          ("tau_s", r"$\tau_s$ (ns)", 1e6), ("tau_q", r"$\tau_q$ (ns)", 1e6), ("tau_p", r"$\tau_p$ (ns)", 1e6),
          ("W0", r"$W_0$ (warps)", 1.0), ("W1", r"$W_1$ (warps)", 1.0)]


def _num(x: float) -> str:
    if x == 0 or abs(x) < 5e-3:
        return "0.00"
    return f"{x:.2f}" if abs(x) < 100 else f"{x:.0f}"


def params_ci_table(B: dict) -> list[str]:
    """Fitted parameters (stored three-restart fit) with 95% intervals over 200 bootstrap replicates of the twelve
    calibration shapes; the parameters are stored in ms (times) and warps."""
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Cost-model parameters per GPU (device time): the stored fit and the 95\% interval over 200 "
             r"bootstrap replicates of the twelve calibration shapes (all their timings resampled together, refitted "
             r"with the stored protocol), and the resulting interval of the mean per-call regret on the captured "
             r"calls. Times per state and tile ($\tau_m$), per element ($\tau_e$), per scan level ($\tau_s$) and per "
             r"spilled word ($\tau_q$, $\tau_p$).}",
             r"\label{tab:supp_params_ci}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrrr}", r"\toprule",
             r"Parameter & RTX PRO~6000 & RTX~3090 & H800 \\", r"\midrule"]
    for key, label, scale in PARAMS:
        cells = [label]
        for tag in TAGS:
            rec = B["per_tag"][tag]["M3c"]["cluster_bootstrap"]["params"][key]
            lo, hi = rec["ci95"]
            cells.append(f"{_num(scale * rec['stored'])} [{_num(scale * lo)}, {_num(scale * hi)}]")
        lines.append(" & ".join(cells) + r" \\")
    cells = ["Mean regret (\\%)"]
    for tag in TAGS:
        rec = B["per_tag"][tag]["M3c"]["cluster_bootstrap"]["captured_mean_regret_pct"]
        cells.append(f"{rec['stored']:.2f} [{rec['ci95'][0]:.2f}, {rec['ci95'][1]:.2f}]")
    lines.append(r"\midrule")
    lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return lines


SELECTORS = [("best_static_per_batch_held_out", "Best chunk per batch, held out"), ("entropy_only", "Entropy only"),
             ("shape", "Shape"), ("shape+entropy", "Shape + entropy"),
             ("exact_shape_table_held_out", "Exact-shape table"), ("cost_model", "Cost model"),
             ("work_only", "Three-parameter restriction"), ("mlp_basic", "MLP"), ("mlp_structural", "MLP, struct."),
             ("cost_model_top2", "Cost model, top-2")]
GRAPH = [("ssd370m_graph", "SSD 370M"), ("ssd2p7b_graph", "SSD 2.7B"), ("gdn1p4b_graph", "GDN 1.4B"),
         ("m1_370m_graph", "Triton Mamba-1 370M")]


def _ci(rec: dict, key: str = "mean_regret_pct") -> str:
    ci = rec.get("ci95", {}).get(key)
    v = rec.get(key)
    if v is None:
        return "--"
    return f"{v:.2f}" if not ci else f"{v:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def ci_table(A: dict, in_dir: Path) -> list[str]:
    sel = A["M4a_selector_tables"]["mamba1"]
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{95\% prompt-cluster bootstrap intervals (2000 replicates) of the mean per-call relative regret "
             r"(\%). Top: selector table of the main paper (device time; RTX PRO~6000 and RTX~3090: round medians, "
             r"H800: one block). Middle: library default under CUDA-graph replay (Triton: best single chunk). Bottom: "
             r"pooled tensor-parallel shard shapes (126 per GPU, shape bootstrap).}",
             r"\label{tab:supp_ci}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrrr}", r"\toprule",
             r"Selector & RTX PRO~6000 & RTX~3090 & H800 \\", r"\midrule"]
    for key, label in SELECTORS:
        cells = [label] + [_ci(sel[t]["selectors"][key]) if key in sel[t]["selectors"] else "--"
                           for t in ("pro6000_device", "vgpu3090_device", "h800_device")]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\midrule", r"CUDA graph, default & RTX PRO~6000 & RTX~3090 & \\", r"\midrule"]
    g3090 = A["M4b_latency_weighted"]["a_cuda_graph"]
    e26p = in_dir / "taco_e26_analysis_pro6000.json"
    gpro = json.loads(e26p.read_text())["tables"] if e26p.exists() else {}
    for key, label in GRAPH:
        r3 = g3090[key]["subsets"]["all"]["policies"]["default"]
        rp = gpro.get(key + "_pro6000", {}).get("subsets", {}).get("all", {}).get("policies", {}).get("default")
        lines.append(" & ".join([label, "--" if rp is None else _ci(rp), _ci(r3), ""]) + r" \\")
    lines += [r"\midrule", r"Tensor parallel, pooled & & RTX~3090 & H800 \\", r"\midrule"]
    tp = A["M7a_tensor_parallel"]["hosts"]
    for key, label in [("cost_model", "Cost model"), ("calibration_width_table", "Table at the calibration width"),
                       ("unsharded_table", "Unsharded table"), ("best_static", "Best single chunk")]:
        cells = [label, ""]
        for host in ("vgpu3090", "h800"):
            rec = tp[host]["triton"]["pooled_shards"].get(key)
            ci = None if rec is None else rec.get("mean_regret_pct_ci95_shape_bootstrap")
            cells.append("--" if rec is None else (f"{rec['mean_regret_pct']:.2f}" if not ci else
                                                   f"{rec['mean_regret_pct']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return lines


if __name__ == "__main__":
    main()
