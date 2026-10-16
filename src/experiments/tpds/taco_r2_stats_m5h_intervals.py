"""taco-r2 item M5h: intervals and per-batch breakdowns, cluster structure, refit-in-bootstrap and multiplicity.

CPU only, deterministic; reads stored result files and times nothing.

Contents
--------
clusters          number and size distribution of the prompt clusters (connected components of the call-prompt
                  graph, ``e3b_information.prompt_clusters``) of every captured-call table
table5            the selector rows of Table 5 on the three primary device-time Mamba-1 tables: mean per-call regret
                  and latency-weighted share with 95% prompt-cluster percentile intervals (the generator of
                  ``taco_report_numbers.evaluate_picks``, so the stored intervals are reproduced) and BCa intervals,
                  all calls and per batch; paired differences (shape + entropy minus shape; cost model minus the
                  exact-shape table) in points and microseconds
table6            the rows of Table 6 (reference CUDA scan, launch-inclusive; Triton scan on the 1.4B and 2.8B
                  checkpoints, device time) with intervals, all calls and per batch: best single chunk (held out),
                  entropy rule, stock dispatch (reference), shape, entropy only, shape + entropy, cost model (reference:
                  Eq. (5) refitted to the reference scan; Triton: the 370M fit of the RTX 3090, not refitted), table
                  tuned on the 370M calls (Triton), the in-sample exact-shape floor and the held-out headroom in us
                  per call at covered shapes. The per-batch rows answer "the pooled 1.4B regret of 3.7% is 1.0 / 13.9
                  / 0.9% at batches 1 / 4 / 16" (Triton Mamba-1 1.4B cost-model transfer)
table8            every row of Table 8 in ms per forward pass with intervals (default, best chunk per batch by total
                  latency, model, held-out exact-shape table); conventions as in Table 8
refit             refit-inside-the-bootstrap interval of the paired difference shape + entropy minus shape (the trees
                  re-trained in every replicate on resampled prompt clusters with fresh random folds), next to the
                  fixed-prediction interval and the stored repeated grouped cross-validation (50 fold assignments)
multiplicity      Holm and Benjamini-Hochberg adjustment of the information tests over all 19 data sets: family E
                  (entropy: exact-key permutation test in three strata, conditional mutual information given the exact
                  key and the exact key and layer, two discretizations) and family A (all six activation statistics)

Output: ``taco_r2_stats_m5h.json``.
Usage:  PYTHONPATH=. python3 -m src.experiments.tpds.taco_r2_stats_m5h_intervals [--n-boot 2000] [--n-refit 200]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.experiments.tpds import cost_model as CM  # noqa: E402
from src.experiments.tpds import cost_model_baselines_r4 as CMB4  # noqa: E402
from src.experiments.tpds import deployment_units as DU  # noqa: E402
from src.experiments.tpds import e22_graph_analysis as E22  # noqa: E402
from src.experiments.tpds import e3_predictors as E3  # noqa: E402
from src.experiments.tpds import e3b_information as E3B  # noqa: E402
from src.experiments.tpds import taco_r2_stats_common as C  # noqa: E402
from src.experiments.tpds import taco_report_numbers as TRN  # noqa: E402
from src.experiments.tpds.analysis import chunk_columns, cv_exact_headroom, latency_matrix, load_e2  # noqa: E402
from src.experiments.tpds.common import RESULTS_DIR  # noqa: E402

T5_SETS = ("pro6000_device", "vgpu3090_device", "h800_device")          # TRN.M1_SETS names (Table 5)
T5_ROWS = ("default_chunk_256", "best_static_global_held_out", "best_static_per_batch_held_out", "L_only", "entropy_only",
           "moments", "shape", "shape+entropy", "exact_shape_table_held_out", "cost_model", "cost_model_top2",
           "cost_model_top3")
T6_SETS = {"reference_370m": ("e8_refscan_vgpu3090.csv", "refscan", 48, "mamba1_reference_cuda_370m_vgpu3090"),
           "triton_1p4b": ("e2_mamba1_vgpu3090_1p4b_r3.csv", "mamba1", 48, "mamba1_triton_1p4b_vgpu3090"),
           "triton_2p8b": ("e2_mamba1_vgpu3090_2p8b_r3.csv", "mamba1", 64, "mamba1_triton_2p8b_vgpu3090")}
# Table 6 as printed (mean per-call regret %)
T6_PRINTED = {"reference_370m": {"best_single_chunk": "30.2", "entropy_rule": "185.5", "stock_dispatch": "6.6", "shape": "3.0",
                                 "entropy_only": "28.9", "shape+entropy": "3.1", "cost_model": "4.9", "exact_floor_in_sample": "2.5",
                                 "headroom_covered_us": "2.2"},
              "triton_1p4b": {"best_single_chunk": "15.4", "entropy_rule": "119.2", "shape": "2.1", "entropy_only": "13.4",
                              "shape+entropy": "2.1", "cost_model": "3.7", "table_tuned_on_370m": "5.3",
                              "exact_floor_in_sample": "0.07", "headroom_covered_us": "0.18"},
              "triton_2p8b": {"best_single_chunk": "15.5", "entropy_rule": "128.7", "shape": "2.9", "entropy_only": "14.3",
                              "shape+entropy": "2.9", "cost_model": "0.8", "table_tuned_on_370m": "6.8",
                              "exact_floor_in_sample": "0.05", "headroom_covered_us": "0.32"}}
# Table 8 as printed (ms per forward pass): default, best static, model, table
T8_PRINTED = {"Triton Mamba-1 370M, PRO 6000": ("4.42", "2.03", "0.30", "0.02"),
              "Triton Mamba-1 370M, RTX 3090": ("6.40", "0.79", "2.90", "0.70"),
              "Triton Mamba-1 370M, H800": ("4.42", "1.76", "1.29", "0.02"),
              "Triton Mamba-1 1.4B, RTX 3090": ("14.5", "2.91", "2.66", "0.30"),
              "Triton Mamba-1 2.8B, RTX 3090": ("25.2", "4.04", "1.53", "0.53"),
              "Reference scan 370M, RTX 3090": ("1.75", None, "0.66", "0.30"),
              "SSD 370M, PRO 6000": ("0.37", None, "0.34", "0.28"),
              "SSD 370M, RTX 3090": ("0.98", None, "0.92", "0.89"),
              "SSD 2.7B, RTX 3090": ("0.24", None, "0.26", "0.15"),
              "GDN 1.4B, RTX 3090": ("<0.01", None, "<0.01", "<0.01")}
T8_TRITON = {"Triton Mamba-1 370M, PRO 6000": ("e2_mamba1_pro6000_r4spin.csv", True, "pro6000_r4spin", 48, None),
             "Triton Mamba-1 370M, RTX 3090": ("e2_mamba1_vgpu3090_r4spin.csv", True, "vgpu3090_r3spin", 48, None),
             "Triton Mamba-1 370M, H800": ("e2_mamba1_h800_r3spin.csv", False, "h800_r3spin", 48, None),
             "Triton Mamba-1 1.4B, RTX 3090": ("e2_mamba1_vgpu3090_1p4b_r3.csv", False, None, 48, "mamba1_triton_1p4b_vgpu3090"),
             "Triton Mamba-1 2.8B, RTX 3090": ("e2_mamba1_vgpu3090_2p8b_r3.csv", False, None, 64, "mamba1_triton_2p8b_vgpu3090")}
T8_DU = {"Reference scan 370M, RTX 3090": "mamba1_reference_cuda_370m_vgpu3090", "SSD 370M, PRO 6000": "mamba2_ssd_370m_pro6000",
         "SSD 370M, RTX 3090": "mamba2_ssd_370m_vgpu3090", "SSD 2.7B, RTX 3090": "mamba2_ssd_2p7b_vgpu3090",
         "GDN 1.4B, RTX 3090": "gdn_1p4b_vgpu3090"}
REFIT_SETS = {"pro6000_device": ("e2_mamba1_pro6000_r4spin.csv", True), "vgpu3090_device": ("e2_mamba1_vgpu3090_r4spin.csv", True),
              "h800_device": ("e2_mamba1_h800_r3spin.csv", False), "mamba2_vgpu3090_launch": ("e2_mamba2_vgpu3090.csv", False),
              "refscan_vgpu3090": ("e8_refscan_vgpu3090.csv", False)}
E3B_TAGS = ("gdn_vgpu3090_1p4b_r3", "mamba1_h800", "mamba1_h800_r3spin", "mamba1_pro6000", "mamba1_pro6000_r4spin_rmed",
            "mamba1_vgpu3090", "mamba1_vgpu3090_1p4b", "mamba1_vgpu3090_1p4b_r3", "mamba1_vgpu3090_1p4b_r3plain",
            "mamba1_vgpu3090_2p8b", "mamba1_vgpu3090_2p8b_r3", "mamba1_vgpu3090_2p8b_r3plain", "mamba1_vgpu3090_r3spin",
            "mamba1_vgpu3090_r4spin_rmed", "mamba2_pro6000", "mamba2_vgpu3090", "mamba2_vgpu3090_2p7b_r3",
            "mamba2_vgpu3090_r3spin", "refscan_vgpu3090")


# --------------------------------------------------------------------------- generic interval block
def interval_block(lat, orc, picks: dict, clusters, label: str, n_boot: int, layers: int | None, batch: np.ndarray,
                   pairs=(), extra: dict | None = None) -> dict:
    """Percentile and BCa intervals of mean regret, LW share and us per call (and ms per forward pass) per pick,
    all calls (generator of ``taco_report_numbers.evaluate_picks``) and per batch; paired differences.

    ``extra``: name -> (per-call latency of a policy that is not a chunk index, e.g. a separately timed binary)."""
    pol_lat = {n: lat[np.arange(len(orc)), np.asarray(p)] for n, p in picks.items()}
    pol_lat.update(extra or {})
    subsets = {"all": None, **{f"batch{int(b)}": batch == b for b in sorted(np.unique(batch))}}
    out = {}
    for sub, sel in subsets.items():
        cb = C.ClusterBoot(clusters, n_boot, TRN.make_rng(label, "all" if sub == "all" else sub.replace("batch", "batch")),
                           sel)
        A = {"orc": orc}
        for n, v in pol_lat.items():
            A[f"{n}|exc"] = v - orc
            A[f"{n}|reg"] = v / orc - 1.0
        rec = {"n_calls": int(len(cb.idx)), "n_prompt_clusters": int(cb.K), "policies": {}, "paired": {}}
        for n in pol_lat:
            r = {"mean_regret_pct": C.summarize(cb.evaluate(A, C.mean_g(f"{n}|reg", 100))),
                 "lw_share_of_oracle_pct": C.summarize(cb.evaluate(A, C.lw_g(f"{n}|exc"))),
                 "us_per_call": C.summarize(cb.evaluate(A, C.mean_g(f"{n}|exc", 1000)))}
            if layers:
                r["ms_per_forward_pass"] = C.summarize(cb.evaluate(A, C.mean_g(f"{n}|exc", layers)))
            rec["policies"][n] = r
        for a, b in pairs:
            if a in pol_lat and b in pol_lat:
                rec["paired"][f"{a} - {b}"] = {
                    "pp": C.summarize(cb.evaluate(A, lambda T, n, a=a, b=b: 100 * (T[f"{a}|reg"] - T[f"{b}|reg"]) / n)),
                    "us": C.summarize(cb.evaluate(A, lambda T, n, a=a, b=b: 1000 * (T[f"{a}|exc"] - T[f"{b}|exc"]) / n))}
        out[sub] = rec
    return out


def _cell(rec: dict, key: str = "mean_regret_pct", nd: int = 2) -> str:
    r = rec[key]
    return (f"{r['point']:.{nd}f} [{r['ci95_percentile'][0]:.{nd}f}, {r['ci95_percentile'][1]:.{nd}f}] "
            f"BCa [{r['ci95_bca'][0]:.{nd}f}, {r['ci95_bca'][1]:.{nd}f}]")


# --------------------------------------------------------------------------- clusters
def cluster_structure(in_dir: Path) -> dict:
    files = {"Mamba-1 370M (all GPUs and conventions, same calls)": "e2_mamba1_pro6000_r4spin.csv",
             "Mamba-2 370M": "e2_mamba2_vgpu3090.csv", "Mamba-2 2.7B": "e2_mamba2_vgpu3090_2p7b_r3.csv",
             "Gated DeltaNet 1.4B": "e2_gdn_vgpu3090_1p4b_r3.csv", "Mamba-1 1.4B": "e2_mamba1_vgpu3090_1p4b_r3.csv",
             "Mamba-1 2.8B": "e2_mamba1_vgpu3090_2p8b_r3.csv", "reference scan 370M": "e8_refscan_vgpu3090.csv"}
    out = {}
    for label, csv in files.items():
        df = DU.load_calls(in_dir, {"csv": csv})
        cl, n_prompts = E3B.prompt_clusters(df)
        rec = C.cluster_sizes(cl, df)
        rec["source"] = csv
        rec["n_calls"] = int(len(df))
        rec["calls_by_batch"] = {str(int(b)): int((df["batch"] == b).sum()) for b in sorted(df["batch"].unique())}
        rec["clusters_with_batched_calls"] = int(pd.Series(df["batch"].to_numpy() > 1).groupby(cl).any().sum())
        rec["clusters_nested_in_folds"] = bool((pd.Series(df["fold"].to_numpy()).groupby(cl).nunique() == 1).all())
        out[label] = rec
    return out


# --------------------------------------------------------------------------- Table 5
def table5(args, checks: list) -> dict:
    stored = (C.load_json(args.in_dir / "taco_report_numbers.json") or {}).get("M4a_selector_tables", {}).get("mamba1", {})
    out = {}
    for name in T5_SETS:
        spec = TRN.M1_SETS[name]
        t = TRN.load_table(args.in_dir, spec["csv"], spec["rmed"], name, "mamba1", spec["cal"])
        picks = TRN.base_picks(t)
        res = json.loads((args.in_dir / f"cost_model_{spec['cal']}.json").read_text())
        m = dict(zip(res["per_call_choice"]["mamba1"]["keys"], res["per_call_choice"]["mamba1"]["chunk"]))
        picks["cost_model"] = np.array([t.chunks.index(int(m[k])) for k in t.df["key"]])
        cv = pd.read_csv(args.in_dir / f"cost_model_curves_{spec['cal']}.csv")
        piv = cv.pivot_table(index=["batch", "L"], columns="chunk", values="predicted_ms")
        P = np.array([[piv.loc[(int(b), int(L)), c] for c in t.chunks] for b, L in zip(t.batch, t.df["L"])])
        if not np.array_equal(P.argmin(1), picks["cost_model"]):
            checks.append({"check": f"table5.{name}: curve argmin == stored per-call cost-model choice",
                           "agreement": float((P.argmin(1) == picks["cost_model"]).mean()), "match": False})
        for k in (2, 3):
            picks[f"cost_model_top{k}"] = CMB4.topk_picks(t.df, t.chunks, P, k)
        sel = {k: picks[k] for k in T5_ROWS if k in picks}
        blk = interval_block(t.lat, t.orc, sel, t.clusters, name, args.n_boot, 48, t.batch,
                             pairs=(("shape+entropy", "shape"), ("cost_model", "exact_shape_table_held_out"),
                                    ("cost_model", "best_static_per_batch_held_out")))
        st = stored.get(name, {}).get("selectors", {})
        for k in sel:
            if k in st:
                a = blk["all"]["policies"][k]["mean_regret_pct"]
                ok = abs(a["point"] - st[k]["mean_regret_pct"]) < 1e-9
                ok_ci = np.allclose(a["ci95_percentile"], st[k]["ci95"]["mean_regret_pct"], atol=1e-9)
                checks.append({"check": f"table5.{name}.{k}: point and percentile interval == taco_report_numbers",
                               "computed": [a["point"], a["ci95_percentile"]],
                               "stored": [st[k]["mean_regret_pct"], st[k]["ci95"]["mean_regret_pct"]], "match": bool(ok and ok_ci)})
        out[name] = {"gpu": spec["gpu"], "convention": spec["convention"], "source": spec["csv"], "intervals": blk,
                     "compact": {k: _cell(blk["all"]["policies"][k]) for k in sel}}
    return out


# --------------------------------------------------------------------------- Table 6
def table6(args, checks: list) -> dict:
    out = {}
    specs = {s["name"]: s for s in DU.kernel_specs()}
    for name, (csv, kind, layers, du_name) in T6_SETS.items():
        t = TRN.load_table(args.in_dir, csv, False, name, "mamba1", "vgpu3090")
        bp = TRN.base_picks(t)
        picks = {"best_single_chunk": bp["best_static_global_held_out"], "entropy_rule": bp["entropy_rule"],
                 "shape": bp["shape"], "entropy_only": bp["entropy_only"], "shape+entropy": bp["shape+entropy"],
                 "exact_floor_in_sample": bp["exact_shape_floor_in_sample"]}
        lat_fin = np.where(np.isfinite(t.lat), t.lat, np.inf)
        picks["cost_model"], minfo = DU.model_choice(args.in_dir, specs[du_name], t.df, t.chunks, lat_fin)
        extra = {}
        if kind == "refscan":
            extra["stock_dispatch"] = t.df["lat_stock_mean"].to_numpy(float)
            picks["stock_chunk_runtime_build"] = np.array([t.chunks.index(int(c)) for c in t.df["stock_chunk"]])
        else:
            ref = load_e2([args.in_dir / "e2_mamba1_vgpu3090_r3spin.csv"])
            key = ["L_bucket", "batch", "aligned"]
            ref["aligned"] = (ref["L"] % 16 == 0).astype(int)
            d = t.df.assign(aligned=(t.df["L"] % 16 == 0).astype(int))
            rlat = latency_matrix(ref, t.chunks)
            table = {k: int(np.argmin(rlat[g.index.to_numpy()].mean(0))) for k, g in ref.groupby(key)}
            glob = int(np.argmin((rlat / rlat.min(1, keepdims=True)).sum(0)))
            picks["table_tuned_on_370m"] = np.array([table.get(tuple(r), glob) for r in d[key].itertuples(index=False)])
        blk = interval_block(t.lat, t.orc, picks, t.clusters, f"m5h_t6_{name}", args.n_boot, layers, t.batch,
                             pairs=(("shape+entropy", "shape"), ("cost_model", "shape")), extra=extra)
        h = cv_exact_headroom(t.df.reset_index(drop=True), t.chunks, threshold_ms=C.SYNC_US["RTX 3090"] / 1000)
        rec = {"source": csv, "scan_layers": layers, "cost_model_source": minfo, "intervals": blk,
               "headroom_covered_us": 1000 * h["covered"]["mean_ms"], "headroom_covered_max_shape_us": 1000 * h["covered"]["max_key_ms"],
               "headroom_covered_keys_above_sync": h["covered"].get("n_keys_above_threshold")}
        rec["per_batch_mean_regret_pct"] = {
            p: {b: _cell(blk[f"batch{b}"]["policies"][p]) for b in (1, 4, 16)} for p in blk["all"]["policies"]}
        rec["compact"] = {p: _cell(blk["all"]["policies"][p]) for p in blk["all"]["policies"]}
        # checks against the printed table
        for p, printed in T6_PRINTED[name].items():
            if p == "headroom_covered_us":
                val = rec["headroom_covered_us"]
            else:
                val = blk["all"]["policies"][p]["mean_regret_pct"]["point"]
            checks.append({"check": f"table6.{name}.{p} vs printed", "computed": val, "printed": printed,
                           "match": bool(E22._matches(val, printed))})
        out[name] = rec
    return out


# --------------------------------------------------------------------------- Table 8
def table8(args, checks: list) -> dict:
    out = {}
    specs = {s["name"]: s for s in DU.kernel_specs()}
    for label, (csv, rmed, cal, layers, du_name) in T8_TRITON.items():
        t = TRN.load_table(args.in_dir, csv, rmed, f"m5h_t8_{label}", "mamba1", cal or "vgpu3090")
        bp = TRN.base_picks(t)
        lat_fin = np.where(np.isfinite(t.lat), t.lat, np.inf)
        if cal:
            res = json.loads((args.in_dir / f"cost_model_{cal}.json").read_text())
            m = dict(zip(res["per_call_choice"]["mamba1"]["keys"], res["per_call_choice"]["mamba1"]["chunk"]))
            model = np.array([t.chunks.index(int(m[k])) for k in t.df["key"]])
        else:
            model, _ = DU.model_choice(args.in_dir, specs[du_name], t.df, t.chunks, lat_fin)
        picks = {"default": bp["default_chunk_256"], "best_static": bp["best_static_per_batch_by_total_latency"],
                 "model": model, "table": bp["exact_shape_table_held_out_mean_latency"]}
        out[label] = _t8_row(label, t.lat, t.orc, picks, t.clusters, layers, t.batch, None, args, checks, csv)
    for label, du_name in T8_DU.items():
        spec = specs[du_name]
        df = DU.load_calls(args.in_dir, spec)
        chunks = chunk_columns(df)
        lat = DU.lat_inf(df, chunks)
        orc = lat.min(1)
        picks = {"model": DU.model_choice(args.in_dir, spec, df, chunks, lat)[0],
                 "table": DU.cv_exact_pick(df, lat)}
        extra = {}
        if spec["default"] == "stock":
            extra["default"] = df["lat_stock_mean"].to_numpy(float)
        else:
            picks["default"] = np.full(len(df), chunks.index(int(spec["default"])))
        sel = None
        if spec["floor"]:
            small = (df["batch"] == 1).to_numpy() & (df["L"] <= 256).to_numpy()
            sel = orc > 2 * np.median(orc[small])
        cl = E3B.prompt_clusters(df)[0]
        out[label] = _t8_row(label, lat, orc, picks, cl, spec["layers"], df["batch"].to_numpy(), sel, args, checks,
                             spec["csv"], extra)
    return out


def _t8_row(label, lat, orc, picks, clusters, layers, batch, sel, args, checks, src, extra=None) -> dict:
    pol_lat = {n: lat[np.arange(len(orc)), np.asarray(p)] for n, p in picks.items()}
    pol_lat.update(extra or {})
    cb = C.ClusterBoot(clusters, args.n_boot, C.make_rng("m5h_t8", label), sel)
    A = {"orc": orc, **{f"{n}|exc": v - orc for n, v in pol_lat.items()}}
    rec = {"source": src, "scan_layers": layers, "subset": "calls above the launch floor" if sel is not None else "all calls",
           "n_calls": int(len(cb.idx)), "n_prompt_clusters": int(cb.K),
           "oracle_scan_ms_per_forward_pass": C.summarize(cb.evaluate(A, C.mean_g("orc", layers)))["point"],
           "ms_per_forward_pass": {}}
    for n in ("default", "best_static", "model", "table"):
        if n not in pol_lat:
            continue
        rec["ms_per_forward_pass"][n] = C.summarize(cb.evaluate(A, C.mean_g(f"{n}|exc", layers)))
    for n, printed in zip(("default", "best_static", "model", "table"), T8_PRINTED[label]):
        if printed is None or n not in rec["ms_per_forward_pass"]:
            continue
        v = rec["ms_per_forward_pass"][n]["point"]
        ok = v < 0.01 if printed == "<0.01" else E22._matches(v, printed)
        checks.append({"check": f"table8.{label}.{n} vs printed", "computed": v, "printed": printed, "match": bool(ok)})
    rec["compact"] = {n: _cell(rec["ms_per_forward_pass"], n, 3) for n in rec["ms_per_forward_pass"]}
    return rec


# --------------------------------------------------------------------------- refit inside the bootstrap
def _cv_trees(X: dict, oracle_idx, norm, folds, names) -> dict:
    preds = {n: np.full(len(folds), -1) for n in names}
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if tr.sum() == 0 or te.sum() == 0:
            continue
        for n in names:
            cols, depth = E3.FEATURE_SETS[n]
            m = E3.LeafCostTree(depth).fit(X[n][tr], oracle_idx[tr], norm[tr])
            preds[n][te] = m.predict_idx(X[n][te])
    return preds


def _refit_worker(job) -> dict:
    name, csv, rmed, in_dir, n_rep, seed_label = job
    t = TRN.load_table(Path(in_dir), csv, rmed, name, "", "")
    names = ("shape", "shape+entropy")
    X = {n: t.df[E3.FEATURE_SETS[n][0]].to_numpy(float) for n in names}
    lat = np.where(np.isfinite(t.lat), t.lat, np.inf)
    orc = lat.min(1)
    oracle_idx = lat.argmin(1)
    norm = np.minimum(lat / orc[:, None], 1e6)
    cl = t.clusters
    K = int(cl.max()) + 1
    members = [np.flatnonzero(cl == k) for k in range(K)]
    rng = C.make_rng("m5h_refit", seed_label)
    diffs, diffs_us = [], []
    for _ in range(n_rep):
        draw = rng.integers(0, K, K)
        idx = np.concatenate([members[k] for k in draw])
        copy_fold = rng.integers(0, 5, K)
        folds = np.concatenate([np.full(len(members[k]), copy_fold[j]) for j, k in enumerate(draw)])
        p = _cv_trees({n: X[n][idx] for n in names}, oracle_idx[idx], norm[idx], folds, names)
        r = {n: lat[idx, p[n]] for n in names}
        diffs.append(100 * ((r["shape+entropy"] - r["shape"]) / orc[idx]).mean())
        diffs_us.append(1000 * (r["shape+entropy"] - r["shape"]).mean())
    diffs, diffs_us = np.array(diffs), np.array(diffs_us)
    return {"name": name, "n_replicates": n_rep, "diff_pp_ci95": C.pct_ci(diffs), "diff_pp_mean": float(diffs.mean()),
            "diff_us_ci95": C.pct_ci(diffs_us), "share_replicates_entropy_better": float(np.mean(diffs < 0)),
            "method": "clusters resampled with replacement, each copy assigned to one of five folds at random, both trees "
                      "re-trained by grouped five-fold cross-validation on the replicate"}


def refit_block(args) -> dict:
    jobs = [(n, csv, rmed, str(args.in_dir), args.n_refit, n) for n, (csv, rmed) in REFIT_SETS.items()]
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as ex:
        res = {r["name"]: r for r in ex.map(_refit_worker, jobs)}
    stored_tags = {"pro6000_device": "mamba1_pro6000_r4spin_rmed", "vgpu3090_device": "mamba1_vgpu3090_r4spin_rmed",
                   "h800_device": "mamba1_h800_r3spin", "mamba2_vgpu3090_launch": "mamba2_vgpu3090",
                   "refscan_vgpu3090": "refscan_vgpu3090"}
    for n, tag in stored_tags.items():
        st = C.load_json(args.in_dir / f"e3b_information_{tag}.json") or {}
        d = st.get("differences", {}).get("shape+entropy - shape", {}).get("prompt_cluster")
        if d:
            res[n]["fixed_prediction_bootstrap"] = {"diff_pp": d["diff_pp"], "ci95_pp": d["ci95_pp"]}
        rc = st.get("repeated_cv", {}).get("depth4: shape+entropy - shape")
        if rc:
            res[n]["repeated_grouped_cv_50"] = rc
    rcv = {}
    for tag in E3B_TAGS:
        st = C.load_json(args.in_dir / f"e3b_information_{tag}.json") or {}
        rc = st.get("repeated_cv", {}).get("depth4: shape+entropy - shape")
        if rc:
            rcv[tag] = {"mean_pp": rc["mean_pp"], "p2.5_p97.5_pp": rc["p2.5_p97.5_pp"], "range_pp": rc["range_pp"],
                        "share_repeats_entropy_better": rc["share_repeats_other_better"]}
    return {"refit_in_bootstrap": res, "stored_repeated_grouped_cv_all_datasets": rcv}


# --------------------------------------------------------------------------- multiplicity
def multiplicity(args) -> dict:
    fam_e, fam_a, info = {}, {}, {}
    for tag in E3B_TAGS:
        st = C.load_json(args.in_dir / f"e3b_information_{tag}.json")
        if not st:
            continue
        ep = st["exact_key_permutation"]["strata"]
        for stratum, recs in ep.items():
            for stat, r in recs.items():
                if stat in ("layer",) or "p_value" not in r:
                    continue
                k = f"{tag}|perm|{stratum}|{stat}"
                fam_a[k] = r["p_value"]
                info[k] = {"T_obs_pp": r.get("T_obs_pp")}
                if stat == "entropy_bits":
                    fam_e[k] = r["p_value"]
        for cond in ("exact_key", "exact_key_x_layer"):
            for stat, r in st["conditional_mi"]["by_condition"].get(cond, {}).items():
                if stat == "layer_control" or "p_value" not in r:
                    continue
                k = f"{tag}|cmi|{cond}|{stat}"
                fam_a[k] = r["p_value"]
                info[k] = {"excess_bits": r.get("excess_bits"),
                           "excess_over_H_oracle": (r["excess_bits"] / r["H_oracle_given_cond_bits"])
                           if r.get("H_oracle_given_cond_bits") else None}
                if stat.startswith("entropy"):
                    fam_e[k] = r["p_value"]

    def summarize_family(fam: dict) -> dict:
        h, b = C.holm(fam), C.bh(fam)
        n_perm_min = 1.0 / 1001
        return {"n_tests": len(fam), "n_datasets": len({k.split("|")[0] for k in fam}),
                "n_raw_p_below_0.05": int(sum(p < 0.05 for p in fam.values())),
                "n_bh_below_0.05": int(sum(p < 0.05 for p in b.values())),
                "n_holm_below_0.05": int(sum(p < 0.05 for p in h.values())),
                "smallest_attainable_p_with_1000_permutations": n_perm_min,
                "raw_significant": {k: {"p": fam[k], "p_bh": b[k], "p_holm": h[k], **info.get(k, {})}
                                    for k in sorted(fam, key=fam.get) if fam[k] < 0.05}}
    return {"definition": ("family E: entropy tests (exact-key split permutation test in the strata exact key x fold, "
                           "exact key, exact key x layer; conditional mutual information of entropy (8 global bins and 4 "
                           "within-stratum bins) given the exact key and given the exact key and layer); family A: the "
                           "same for all six activation statistics and the moment vector; layer controls excluded. "
                           "Adjusted over all 19 data sets at once."),
            "family_entropy": summarize_family(fam_e), "family_all_statistics": summarize_family(fam_a)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--n-boot", type=int, default=C.N_BOOT)
    p.add_argument("--n-refit", type=int, default=200)
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--skip-refit", action="store_true")
    args = p.parse_args(argv)
    t0 = time.time()
    checks: list = []
    out = {"item": "M5h", "meta": C.meta(__name__, args), "definitions": __doc__}
    out["clusters"] = cluster_structure(args.in_dir)
    print(f"clusters ({time.time() - t0:.0f} s)", flush=True)
    out["table5"] = table5(args, checks)
    print(f"table 5 ({time.time() - t0:.0f} s)", flush=True)
    out["table6"] = table6(args, checks)
    print(f"table 6 ({time.time() - t0:.0f} s)", flush=True)
    out["table8"] = table8(args, checks)
    print(f"table 8 ({time.time() - t0:.0f} s)", flush=True)
    out["multiplicity"] = multiplicity(args)
    if not args.skip_refit:
        out["refit"] = refit_block(args)
        print(f"refit ({time.time() - t0:.0f} s)", flush=True)
    out["checks"] = checks
    out["n_checks"] = len(checks)
    out["n_checks_failed"] = sum(1 for c in checks if c.get("match") is False)
    out["runtime_s"] = round(time.time() - t0, 1)
    path = C.write_result("m5h", out, args.in_dir)
    for n, r in out["table6"].items():
        print(n, json.dumps(r["per_batch_mean_regret_pct"].get("cost_model")))
    for c in checks:
        if c.get("match") is False:
            print("CHECK FAILED:", c)
    print(f"wrote {path} ({out['n_checks']} checks, {out['n_checks_failed']} failed)")


if __name__ == "__main__":
    main()
