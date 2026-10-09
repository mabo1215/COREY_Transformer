"""Bootstrap intervals for every regret cell of main-text Tables IV and V (review item M7d).

CPU only, deterministic; reads stored result files and times nothing.

Tables (RTX PRO 6000, 2360 captured 370M calls each):
  Mamba-1  ``e2_mamba1_pro6000_r4spin.csv`` on its round medians (``analysis.round_median_view``: device
           time, three shuffled rounds, oracle and ties on the medians), as in the main text;
  Mamba-2  ``e2_mamba2_pro6000.csv`` (launch-inclusive, one block per chunk).

Resampling unit: prompt clusters (connected components of the call-prompt graph,
``e3b_information.prompt_clusters``), ``--n-boot`` replicates, 95% percentile intervals of the mean and of
the 95th percentile (weighted inverted-CDF percentile, as in E3b).

Table IV: per batch size, the in-sample best single chunk of that batch (lowest summed normalized latency)
and the entropy rule; clusters are resampled among the calls of that batch.
Table V: every predictor row, held out with the capture-time prompt folds exactly as in E3b (best single
chunk, entropy rule, batch only, the E3 trees, the exact-shape table); the Mamba-1 cost model is the stored
per-call choice of ``cost_model_pro6000_r4spin.json`` (device-time calibration), the Mamba-2 work-term model
``cost_model.pipe_select`` with the parameters stored in ``cost_model_pro6000.json`` (launch-bound exact ties
to the work-bound optimum). Point values are checked against ``r4_rounds_analysis.json`` (Mamba-1) and
``e3b_information_mamba2_pro6000.json`` / ``cost_model_pro6000.json`` (Mamba-2).

Output: ``tables_iv_v_ci.json`` in the results directory.

Usage:  python3 -m src.experiments.tpds.tables_iv_v_ci [--n-boot 2000] [--seed 20261009]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.tpds import cost_model as CM  # noqa: E402
from src.experiments.tpds import e3_predictors as e3  # noqa: E402
from src.experiments.tpds import e3b_information as E3B  # noqa: E402
from src.experiments.tpds.analysis import load_e2, round_median_view  # noqa: E402
from src.experiments.tpds.common import RESULTS_DIR, utc_now, write_json  # noqa: E402

TABLES = {"mamba1": ("e2_mamba1_pro6000_r4spin.csv", True, "pro6000_r4spin"),
          "mamba2": ("e2_mamba2_pro6000.csv", False, "pro6000")}
ROWS_V = ["global_static", "entropy_rule", "batch_only", "L_only", "shape", "shape+entropy", "all",
          "entropy_only", "moments", "exact_key_table", "cost_model"]


def load_table(in_dir: Path, model: str) -> E3B.Table:
    csv, rmed, host = TABLES[model]
    df = load_e2([in_dir / csv])
    if "regimes" in df and (df["regimes"] == "e4_key").any():
        df = df[df["regimes"] != "e4_key"].reset_index(drop=True)
    if rmed:
        df = round_median_view(df)
    return E3B.Table(df, f"{model}_{host}", model, host, csv)


def choices_v(t: E3B.Table, in_dir: Path) -> dict[str, np.ndarray]:
    ch = {"global_static": E3B.cv_static(t), "entropy_rule": E3B.entropy_rule_choice(t)}
    if len(np.unique(t.batch)) > 1:
        ch["batch_only"] = E3B.cv_table(t, [t.batch_code])[0]
    for name, (cols, depth) in e3.FEATURE_SETS.items():
        ch[name] = E3B.cv_tree(t, cols, depth)
    ch["exact_key_table"] = E3B.ExactKeySplit(t).base_choice
    if t.model == "mamba1":
        cm, info = E3B.cost_model_choice(t, in_dir)
        if cm is None:
            raise RuntimeError(f"no stored cost-model choice: {info}")
        ch["cost_model"] = cm
    else:
        pm = json.loads((in_dir / f"cost_model_{t.host}.json").read_text())["V4_pipelines"]["mamba2"]
        th = [pm["params"][k] for k in CM.PIPE_PARAMS]
        P = np.stack([CM.pipe_predict(th, t.df["L"].to_numpy(float), t.df["batch"].to_numpy(float), c)
                      for c in t.chunks], 1)
        P = np.where(np.isfinite(t.lat), P, np.inf)
        ch["cost_model"] = CM.pipe_select(th, P, t.chunks)
    return ch


def summarize(reg: np.ndarray, tie: np.ndarray, clusters: np.ndarray, B: int, rng: np.random.Generator) -> dict:
    _, cl = np.unique(clusters, return_inverse=True)
    reps = E3B.cluster_bootstrap({"m": reg}, cl, B, rng, {"p": reg})
    return {"n": int(len(reg)), "n_clusters": int(cl.max() + 1),
            "mean_pct": float(100 * reg.mean()), "mean_ci95_pct": [100 * x for x in E3B._pct_ci(reps["m"], 0.95)],
            "p95_pct": float(100 * np.percentile(reg, 95)), "p95_ci95_pct": [100 * x for x in E3B._pct_ci(reps["p"], 0.95)],
            "tied_share": float(tie.mean())}


def analyze(in_dir: Path, B: int, seed: int) -> dict:
    out = {}
    for k, model in enumerate(TABLES):
        t = load_table(in_dir, model)
        rng = np.random.default_rng(seed + k)
        ar = np.arange(t.N)
        rec = {"source": t.source, "round_median": TABLES[model][1], "n_calls": t.N,
               "n_prompt_clusters": int(t.clusters.max() + 1), "table_IV": {}, "table_V": {}}
        rule = E3B.entropy_rule_choice(t)
        for b in sorted(np.unique(t.batch)):
            m = t.batch == b
            j = int(t.norm_fit[m].sum(0).argmin())
            rec["table_IV"][str(int(b))] = {
                "best_static_chunk": int(t.chunks[j]),
                "oracle_ne_static_share": float((t.oracle_idx[m] != j).mean()),
                "static": summarize(t.norm[m, j] - 1, t.tied_mask[m, j], t.clusters[m], B, rng),
                "entropy_rule": summarize(t.norm[ar[m], rule[m]] - 1, t.tied_mask[ar[m], rule[m]], t.clusters[m], B, rng)}
        for name, c in choices_v(t, in_dir).items():
            rec["table_V"][name] = summarize(t.norm[ar, c] - 1, t.tied_mask[ar, c], t.clusters, B, rng)
        out[model] = rec
    return out


def checks(res: dict, in_dir: Path) -> dict:
    """Largest absolute difference (points) between recomputed and stored point values."""
    r4 = json.loads((in_dir / "r4_rounds_analysis.json").read_text())["tags"]["pro6000_r4spin"]["views"]["round_median"]
    m1 = res["mamba1"]
    d = []
    for b, v in r4["table_II"].items():
        d.append(abs(m1["table_IV"][b]["static"]["mean_pct"] - v["static"]["mean_regret_pct"]))
        d.append(abs(m1["table_IV"][b]["entropy_rule"]["mean_pct"] - v["entropy_rule"]["mean_regret_pct"]))
    names = {"exact_key_table": "exact_shape_table_held_out"}
    for n, v in m1["table_V"].items():
        d.append(abs(v["mean_pct"] - r4["predictors"][names.get(n, n)]["mean_regret_pct"]))
    e3b = json.loads((in_dir / "e3b_information_mamba2_pro6000.json").read_text())["predictors"]
    m2 = res["mamba2"]
    d2 = [abs(m2["table_V"][n]["mean_pct"] - e3b[n]["mean_regret_pct"]) for n in m2["table_V"] if n in e3b]
    pm = json.loads((in_dir / "cost_model_pro6000.json").read_text())["V4_pipelines"]["mamba2"]["e2"]
    d2.append(abs(m2["table_V"]["cost_model"]["mean_pct"] - pm["mean_regret_pct"]))
    return {"mamba1_max_abs_diff_pp": float(max(d)), "mamba2_max_abs_diff_pp": float(max(d2))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261009)
    args = p.parse_args(argv)
    t0 = time.time()
    res = {"experiment": "tables_iv_v_ci", "item": "M7d", "created_utc": utc_now(), "n_boot": args.n_boot,
           "seed": args.seed, "unit": "prompt clusters", "interval": "95% percentile"}
    res.update(analyze(args.in_dir, args.n_boot, args.seed))
    res["checks"] = checks(res, args.in_dir)
    res["runtime_s"] = round(time.time() - t0, 1)
    out = args.out or args.in_dir / "tables_iv_v_ci.json"
    write_json(out, res)
    for model in TABLES:
        r = res[model]
        print(f"{model}: Table IV")
        for b, v in r["table_IV"].items():
            s, e = v["static"], v["entropy_rule"]
            print(f"  batch {b:>2}: static {v['best_static_chunk']:4d} {s['mean_pct']:6.1f} [{s['mean_ci95_pct'][0]:.1f}, "
                  f"{s['mean_ci95_pct'][1]:.1f}] / {s['p95_pct']:6.1f} [{s['p95_ci95_pct'][0]:.1f}, {s['p95_ci95_pct'][1]:.1f}]"
                  f"   rule {e['mean_pct']:6.1f} [{e['mean_ci95_pct'][0]:.1f}, {e['mean_ci95_pct'][1]:.1f}]")
        print(f"{model}: Table V")
        for n, s in r["table_V"].items():
            print(f"  {n:16s} {s['mean_pct']:6.2f} [{s['mean_ci95_pct'][0]:.2f}, {s['mean_ci95_pct'][1]:.2f}] / "
                  f"{s['p95_pct']:6.1f} [{s['p95_ci95_pct'][0]:.1f}, {s['p95_ci95_pct'][1]:.1f}]  tied {s['tied_share']:.2f}")
    print("checks:", res["checks"], f"({res['runtime_s']} s)")


if __name__ == "__main__":
    main()
