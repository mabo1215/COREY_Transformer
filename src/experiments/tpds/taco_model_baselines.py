"""TACO revision (cycle taco-r1): offline re-analysis of the Mamba-1 cost model and its selector
baselines (review items M3a, M3b, M3c, M4c, M5a, M5b, M5c). CPU only; reads existing result files and
writes ``taco_model_baselines.json``; nothing is timed.

Conventions: two result tags per GPU (RTX PRO 6000 = pro6000, RTX 3090 = vgpu3090, H800 PCIe = h800):
  device_time (primary)        pro6000_r4spin, vgpu3090_r3spin, h800_r3spin
  launch_inclusive (secondary) pro6000, vgpu3090, h800
Captured-call latencies are the ``lat_c<C>_mean`` columns read by every published table; per-call
relative regret = lat(chosen chunk) / min over the six chunks - 1. Every regret summary below gives the
mean, 95th percentile and maximum in percent and the share of calls within 2% of the per-call oracle.

The cost-model predictor and fit are re-implemented on numpy arrays (``predict_np``, ``fit_fast``)
for speed; both are checked to reproduce ``cost_model.predict`` and the stored parameters exactly.

Items (per result tag unless stated):
  M3b  bandwidth floor binding on calibration and captured calls; implementation vs Eq. (3).
  M5a  top-1/2/3 for every ranker: rank the six chunks, profile the k best-ranked chunks per exact
       shape (L, batch) and serve every call of the shape with the fastest of them by shape-mean
       latency (``cost_model_baselines_r4.topk_picks``). Rankers: the selectors of
       ``cost_model_baselines_r4.fit_selectors`` (cost model, three-parameter restriction, work-term
       model, nearest calibrated key, log-linear interpolation, length and shape trees, TFLA variants
       as a sanity check), the learned regressors with the hyperparameters chosen by the stored
       12-fold calibration CV of round 4 (GBR deterministic; MLP over seeds 0-4), a zero-parameter
       compile-facts rule (among the default-table chunks whose compile facts show no spill, minimize
       waves x ceil(L / C); ties to the larger chunk; spilling chunks rank last) and random-k (k
       chunks drawn uniformly per exact shape: exact expectation of the mean and of the within-2%
       share over all C(6, k) subsets, Monte Carlo for the 95th percentile and the maximum).
  M4c  a priori model-form selection over the twelve ablation forms (+ the three-parameter
       restriction): leave-one-GPU-out (same convention) on the other GPUs' captured calls, on the
       other GPUs' leave-one-calibration-shape-out (LOSO) chunk regret, and on the target's own
       calibration LOSO. Tie rule: the fewest free parameters among the forms within 0.1 points of
       the best criterion value; then the lowest criterion value; remaining exact ties are listed.
       Exact-tie check of drop tau_e / drop tau_p against the full form.
  M3c  calibration fit quality: refit on full-protocol timings only (72 E4 static + 18 E1 medians),
       inverse-variance weighting (source-level variances, see ``inverse_variance_weights``),
       excluding the search timings of the first (cold-cache) E4 key, OLS next to soft-l1, 50 random
       multi-starts, a cluster bootstrap over the 12 calibration shapes (200 replicates: parameter and
       captured-call regret CIs) and R^2 of the minimized objective (1 - cost / cost of the best
       constant under the same loss).
  M3a  launch-table coupling from the E4 184-configuration sweeps (search medians, 3 warm-ups / 10
       repeats): chunk spread at the default table, at every fixed (BLOCK_D, num_warps, num_stages)
       and against the best configuration; two-path Shapley split of the regret of a static entry
       (chunk 256, or the best static chunk of the GPU) into a chunk-at-fixed-launch part and a launch
       part; registers / spills / occupancy per default entry (kernel_stats_sm<arch>.json).
  M5b  learning curve over k = 1..12 calibration keys (20 subsets per k, all when fewer; fit from X0;
       the nearest-key table of the same keys alongside), a fit on the 72 static re-measurements
       only, cold-start regret of the model against the default chunk and the best static chunk, and
       (launch-inclusive tags) the one-off cost of the model's calibration against profiling every key
       (E5b scaling of the profiling cost, per-variant compile cost from the cold first E4 key), with
       and without a warm compile cache, for the captured workload and the replayed serving traces.
  M5c  TFLA's runtime expression with its own dimensions (d_qk 256, d_hv 512) fitted on TFLA's own
       kernel timings (E24, RTX 3090, device time r4spinb and launch-inclusive r4plainb): in-sample and
       leave-one-shape-out regret; TFLA's expression on the Mamba-1 scan only as a sanity check.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.experiments.tpds import cost_model as cm  # noqa: E402
from src.experiments.tpds import cost_model_baselines_r4 as r4  # noqa: E402
from src.experiments.tpds import e24_tfla_chunk as e24  # noqa: E402
from src.experiments.tpds import e5b_key_churn as e5b  # noqa: E402
from src.experiments.tpds.analysis import latency_matrix, load_e2  # noqa: E402
from src.experiments.tpds.common import RESULTS_DIR, read_jsonl, write_json  # noqa: E402

GPUS = ("pro6000", "vgpu3090", "h800")
TAGS = {"device_time": {"pro6000": "pro6000_r4spin", "vgpu3090": "vgpu3090_r3spin", "h800": "h800_r3spin"},
        "launch_inclusive": {"pro6000": "pro6000", "vgpu3090": "vgpu3090", "h800": "h800"}}
TAG_INFO = {tag: (conv, gpu) for conv, d in TAGS.items() for gpu, tag in d.items()}
PUBLISHED_MEAN_PCT = {"device_time": {"pro6000": 0.38, "vgpu3090": 3.55, "h800": 3.01},
                      "launch_inclusive": {"pro6000": 0.9, "vgpu3090": 3.2, "h800": 3.5}}
FEAT = ("l2_miss", "loads", "e", "e_d", "lanes_c", "warps_c", "spill_words", "l1_miss", "m", "nw", "waves",
        "n_tiles", "t_mem_ms")
ARCH = {"pro6000": 120, "vgpu3090": 86, "h800": 90}
CHUNKS = sorted(cm.DEFAULT_LAUNCH)
SEED = 0
TIE_POINTS = 0.1                    # M4c tie rule: forms within 0.1 points of the best criterion value
CHECK_TOL = 1e-6                    # absolute tolerance (percentage points) of reproduction checks
RTX3090_BF16_INTENSITY = 76.0       # dense BF16 tensor peak / memory bandwidth (71 TFLOP/s / 936 GB/s), as in r4
TFLA_E24_MAP = {"d_qk": 256, "d_hv": 512, "bytes_qkv": 2, "bytes_if": 4, "bytes_cmn": 4, "f_causal": 0.5}
TFLA_E24_TAGS = {"device_time": "vgpu3090_r4spinb", "launch_inclusive": "vgpu3090_r4plainb"}
PROVENANCE_QUOTE = "short offline search on the RTX 3070 at d=2048, L in {512, 2048, 8192}"

PLANNED = {
    "M5a": "top-1/2/3 regret (rank chunks, profile best k per exact shape, keep the fastest by shape-mean latency; "
           "mean, p95, max, share within 2%) for cost model, work_only (3 parameters), work_term_model, nearest key, "
           "log-linear interpolation, length/shape tree, GBR/MLP basic and structural (MLP over seeds 0-4), random-k "
           "(exact expectation over subsets, Monte Carlo for p95/max), compile-facts rule (non-spilling default-table "
           "chunks, minimize waves x ceil(L/C), ties to the larger chunk); rankers via cost_model_baselines_r4.fit_selectors.",
    "M4c": "a priori form selection over the 12 ablation forms (+ work_only): leave-one-GPU-out on the other GPUs' "
           "captured calls, on the other GPUs' calibration LOSO, and target-calibration LOSO; tie rule: fewest "
           "parameters among forms within 0.1 points of the best; exact-tie check of drop tau_e / drop tau_p.",
    "M3c": "refits: full-protocol only (72 static + 18 E1), inverse-variance weighting, excluding first-key search "
           "timings; OLS vs soft-l1; 50 random multi-starts; cluster bootstrap over the 12 keys (200 replicates); "
           "R^2 of the minimized objective (1 - cost / cost of a constant model).",
    "M3a": "decomposition of the default-table chunk effect from the E4 184-configuration search (spread at the "
           "default table, at fixed (BLOCK_D, num_warps, num_stages), best overall; two-path Shapley split of "
           "static regret into chunk-at-fixed-launch and launch parts); registers/spills per default entry.",
    "M5b": "learning curve (k = 1..12 keys, 20 repetitions, fit from X0), 72-static-only fit, cost table vs "
           "profile-every-key (e5b scaling, per-variant compile cost), serving-trace key counts, cold-start regret.",
    "M5c": "fit TFLA's runtime expression (d_qk 256, d_hv 512) to the E24 TFLA chunk sweep (vgpu3090_r4spinb / "
           "r4plainb), in-sample and leave-one-shape-out regret; TFLA-on-Mamba-1 only as a sanity check.",
}


# =========================================================================== numpy cost model
def farr(f) -> dict:
    return {k: f[k].to_numpy(float) for k in FEAT}


def sub(F: dict, m) -> dict:
    """Rows of a feature dict by boolean mask or integer index."""
    return {k: v[m] for k, v in F.items()}


def tile_time(theta, F):
    """Eq. (3) before the bandwidth floor (same operations and order as ``cost_model.predict``)."""
    t0, tf, td, tm, tl, te, ts, tx, tq, tp, W0, W1 = theta
    N = cm.D_STATE
    A = tf + td * F["l2_miss"] + N * tm
    I = tl * F["loads"] + N * (te * F["e"] + ts * F["e_d"] * np.log2(np.clip(F["lanes_c"], 1, None))
                               + tx * F["e_d"] * (F["warps_c"] > 1) + F["spill_words"] * (tq + tp * F["l1_miss"]))
    W = F["m"] * F["nw"]
    return t0 + F["waves"] * F["n_tiles"] * (A * np.minimum(1.0, W1 / W) + I * np.maximum(1.0, W / W0))


def predict_np(theta, F):
    return np.maximum(tile_time(theta, F), F["t_mem_ms"])


def _free_full(fixed: dict):
    free = [i for i in range(len(cm.PARAMS)) if i not in fixed]

    def full(v):
        th = np.empty(len(cm.PARAMS))
        th[free] = v
        for i, val in fixed.items():
            th[i] = val
        return th
    return free, full


def _residual_fn(F, y, full, w):
    def res(v):
        r = np.log(predict_np(full(v), F)) - y
        return r if w is None else w * r
    return res


def fit_fast(F, y_ms, x0=None, fixed=None, loss="soft_l1", weights=None, scales=(1.0, 0.3, 3.0)):
    """``cost_model.fit`` on numpy feature arrays (identical restarts, bounds, loss and tolerances).
    ``weights`` multiplies the squared log residuals (as ``cost_model.fit``); ``loss`` = "linear" is OLS."""
    y = np.log(y_ms)
    x0 = cm.X0 if x0 is None else np.asarray(x0, float)
    fixed = cm.DEFAULT_FIXED if fixed is None else fixed
    free, full = _free_full(fixed)
    w = None if weights is None else np.sqrt(np.asarray(weights, float))
    res = _residual_fn(F, y, full, w)
    best = None
    for scale in scales:
        start = np.clip(x0 * np.r_[np.full(cm.N_SCALE, scale), np.ones(len(cm.PARAMS) - cm.N_SCALE)], cm.LO, cm.HI)
        sol = least_squares(res, start[free], bounds=(cm.LO[free], cm.HI[free]), loss=loss, f_scale=0.1,
                            x_scale="jac", max_nfev=5000)
        if best is None or sol.cost < best.cost:
            best = sol
    return full(best.x), {"cost": float(best.cost), "nfev": int(best.nfev)}


def fit_single(F, y_ms, start, fixed=None, loss="soft_l1"):
    """One least-squares run from ``start`` (full 12-vector), no restarts (multi-start analysis)."""
    fixed = cm.DEFAULT_FIXED if fixed is None else fixed
    free, full = _free_full(fixed)
    res = _residual_fn(F, np.log(y_ms), full, None)
    sol = least_squares(res, np.clip(start, cm.LO, cm.HI)[free], bounds=(cm.LO[free], cm.HI[free]), loss=loss,
                        f_scale=0.1, x_scale="jac", max_nfev=5000)
    return full(sol.x), float(sol.cost)


def loss_cost(r, loss="soft_l1", f_scale=0.1) -> float:
    """scipy's least-squares cost 0.5 * sum(f^2 rho(r^2 / f^2)) for the linear and soft-l1 losses."""
    if loss == "linear":
        return 0.5 * float(np.sum(r ** 2))
    return float(np.sum(f_scale ** 2 * (np.sqrt(1.0 + (r / f_scale) ** 2) - 1.0)))


def objective_r2(theta, F, y_ms, loss="soft_l1", weights=None) -> dict:
    """R^2 of the minimized objective: 1 - cost(theta) / cost of the best constant under the same loss."""
    y = np.log(y_ms)
    w = np.ones_like(y) if weights is None else np.sqrt(np.asarray(weights, float))
    r = w * (np.log(predict_np(theta, F)) - y)
    c = least_squares(lambda v: w * (v[0] - y), [float(np.median(y))], loss=loss, f_scale=0.1).cost
    cost = loss_cost(r, loss)
    return {"loss": loss, "cost": cost, "cost_constant_model": float(c), "r2_objective": float(1 - cost / c)}


# =========================================================================== shared metrics
def reg_stats(reg) -> dict:
    return {"n": int(len(reg)), "mean_regret_pct": float(100 * reg.mean()),
            "p95_regret_pct": float(100 * np.percentile(reg, 95)), "max_regret_pct": float(100 * reg.max()),
            "share_within_2pct": float((reg <= 0.02).mean())}


def call_regret(lat: np.ndarray, choice: np.ndarray) -> np.ndarray:
    return lat[np.arange(len(lat)), choice] / lat.min(1) - 1


def chk(stored, got, tol=CHECK_TOL) -> dict:
    if stored is None or got is None:
        return {"stored": stored, "recomputed": got, "pass": None}
    d = abs(float(got) - float(stored))
    return {"stored": float(stored), "recomputed": float(got), "abs_diff": d, "tol": tol, "pass": bool(d <= tol)}


def mean_ci(vals) -> dict:
    v = np.asarray(vals, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "min": float(v.min()),
            "max": float(v.max()), "n": int(len(v))}


def pct_ci(vals, lo=2.5, hi=97.5) -> dict:
    v = np.asarray(vals, float)
    return {"median": float(np.median(v)), "ci95": [float(np.percentile(v, lo)), float(np.percentile(v, hi))]}


# =========================================================================== M5a: top-k for every ranker
def make_learner(kind: str, hp: dict, seed: int):
    """``cost_model_baselines_r4._make_learner`` with an explicit random_state."""
    if kind == "gbr":
        return HistGradientBoostingRegressor(learning_rate=0.05, early_stopping=False, random_state=seed, **hp)
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), solver="adam", batch_size=128,
                                                        learning_rate_init=1e-3, max_iter=300, tol=0.0,
                                                        n_iter_no_change=10 ** 9, random_state=seed, **hp))


def compile_facts_matrix(ctx) -> np.ndarray:
    """Zero-parameter compile-facts rule as a score (lower = better): non-spilling chunks first, then
    waves x ceil(L / C) (both from the compile facts and the SM limits), ties to the larger chunk."""
    fc = ctx.calls.fc
    spill = fc["spill_words"].to_numpy(float) > 0
    v = fc["waves"].to_numpy(float) * fc["n_tiles"].to_numpy(float)
    C = ctx.calls.cand.loc[fc.index, "C"].to_numpy(float)
    return ctx.calls.matrix(np.where(spill, 1e9, 0.0) + v - 1e-3 * np.log2(C))


def shape_groups(df: pd.DataFrame) -> dict:
    return {k: np.asarray(v) for k, v in df.groupby(["L", "batch"]).indices.items()}


def random_k(ctx, ks, draws: int, rng) -> dict:
    """k chunks drawn uniformly per exact shape, the fastest by shape-mean latency served to the shape."""
    lat, orc = ctx.lat, ctx.lat.min(1)
    groups = shape_groups(ctx.df)
    n, nch = len(lat), lat.shape[1]
    out = {}
    for k in ks:
        subsets = list(combinations(range(nch), k))
        exp_reg, exp_w2 = np.zeros(n), np.zeros(n)
        per_shape = []
        for idx in groups.values():
            m = lat[idx].mean(0)
            picks = np.array([s[int(np.argmin(m[list(s)]))] for s in subsets])
            R = lat[idx][:, picks].T / orc[idx] - 1           # subsets x calls of the shape
            exp_reg[idx], exp_w2[idx] = R.mean(0), (R <= 0.02).mean(0)
            per_shape.append((idx, R))
        mc = np.empty((draws, n))
        for idx, R in per_shape:
            mc[:, idx] = R[rng.integers(len(subsets), size=draws)]
        p95, mx = np.percentile(mc, 95, axis=1), mc.max(1)
        out[f"top{k}"] = {"n": int(n), "n_subsets_per_shape": len(subsets),
                          "mean_regret_pct": float(100 * exp_reg.mean()),
                          "share_within_2pct": float(exp_w2.mean()),
                          "p95_regret_pct": float(100 * p95.mean()),
                          "p95_regret_pct_mc_interval": [float(100 * np.percentile(p95, 2.5)), float(100 * np.percentile(p95, 97.5))],
                          "max_regret_pct": float(100 * mx.mean()),
                          "max_regret_pct_mc_interval": [float(100 * np.percentile(mx, 2.5)), float(100 * np.percentile(mx, 97.5))],
                          "mc_draws": int(draws), "mc_mean_regret_pct": float(100 * mc.mean()),
                          "note": "mean and within-2% share exact over all subsets; p95/max are means over Monte Carlo draws"}
    return out


def topk_rows(ctx, P: np.ndarray, ks=(1, 2, 3)) -> dict:
    return {f"top{k}": reg_stats(call_regret(ctx.lat, r4.topk_picks(ctx.df, ctx.chunks, P, k))) for k in ks}


def m5a_tag(ctx, stored: dict, opts) -> tuple[dict, dict, dict]:
    t0 = time.time()
    sels, sel_info, _extra = r4.fit_selectors(ctx, do_learned=False)
    checks = {}
    X = r4.learn_features(ctx.points, ctx.f, ctx.stats)
    ylog = np.log(ctx.y)
    learned = {}
    for kind in ("gbr", "mlp"):
        for fs_name, fs in (("basic", r4.BASIC_COLS), ("structural", r4.BASIC_COLS + r4.STRUCT_EXTRA)):
            name = f"{kind}_{fs_name}"
            si = stored["selector_info"][name]
            cols = [c for c in fs if X[c].nunique() > 1]
            checks[f"{name}_columns_equal_stored"] = bool(cols == si["columns"])
            seeds = list(range(opts.mlp_seeds)) if kind == "mlp" else [0]
            for s in seeds:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    model = make_learner(kind, si["chosen"], s).fit(X[cols].to_numpy(float), ylog)
                sels[name if s == 0 else f"{name}_seed{s}"] = r4.Learned(model, cols)
            learned[name] = {"hyperparameters": si["chosen"], "seeds": seeds,
                             "hyperparameter_source": "12-fold leave-one-calibration-shape-out CV of round 4 (seed 0; "
                                                      "cost_model_baselines_r4_<tag>.json selector_info); the final model "
                                                      "is retrained per seed with these hyperparameters",
                             "n_columns": len(cols)}
    mats = {n: s.scores(ctx.calls) for n, s in sels.items()}
    mats["compile_facts_rule"] = compile_facts_matrix(ctx)
    rankers, n_top1_mismatch = {}, {}
    for n, P in mats.items():
        rankers[n] = topk_rows(ctx, P)
        n_top1_mismatch[n] = int((r4.topk_picks(ctx.df, ctx.chunks, P, 1) != P.argmin(1)).sum())
    groups = shape_groups(ctx.df)
    pick = np.empty(len(ctx.df), dtype=int)
    for idx in groups.values():
        pick[idx] = int(np.argmin(ctx.lat[idx].mean(0)))
    exact_table = reg_stats(call_regret(ctx.lat, pick))
    rng = np.random.default_rng(SEED)
    rk = random_k(ctx, (1, 2, 3), opts.mc_draws, rng)
    # MLP over seeds
    seed_summary = {}
    for fs_name in ("basic", "structural"):
        names = [f"mlp_{fs_name}"] + [f"mlp_{fs_name}_seed{s}" for s in range(1, opts.mlp_seeds)]
        seed_summary[f"mlp_{fs_name}"] = {
            f"top{k}": {"mean_regret_pct": mean_ci([rankers[n][f"top{k}"]["mean_regret_pct"] for n in names]),
                        "p95_regret_pct": mean_ci([rankers[n][f"top{k}"]["p95_regret_pct"] for n in names]),
                        "max_regret_pct": mean_ci([rankers[n][f"top{k}"]["max_regret_pct"] for n in names]),
                        "share_within_2pct": mean_ci([rankers[n][f"top{k}"]["share_within_2pct"] for n in names])}
            for k in (1, 2, 3)}
    # reproduction checks against the stored round-4 rows
    for n, r in stored["baselines"].items():
        if n in rankers:
            checks[f"top1_vs_r4_{n}"] = chk(r["mean_regret_pct"], rankers[n]["top1"]["mean_regret_pct"])
    for n, r in stored["topk"].items():
        for k, v in r.items():
            if n in rankers and k in rankers[n]:
                checks[f"{k}_vs_r4_{n}"] = chk(v["mean_regret_pct"], rankers[n][k]["mean_regret_pct"])
    checks["all_chunks_profiled_vs_r4"] = chk(stored["topk"]["cost_model"]["all_chunks_profiled"]["mean_regret_pct"],
                                              exact_table["mean_regret_pct"])
    checks["top1_equals_argmin_all_rankers"] = bool(sum(n_top1_mismatch.values()) == 0)
    checks["random_k_mc_mean_close_to_exact"] = bool(all(abs(v["mc_mean_regret_pct"] - v["mean_regret_pct"]) < 0.05 * max(1.0, v["mean_regret_pct"])
                                                         for v in rk.values()))
    # the roofline floor alone cannot rank chunks: bytes / bandwidth does not depend on the chunk
    Tm = ctx.calls.matrix(ctx.calls.fc["t_mem_ms"].to_numpy(float))
    Tm = np.where(np.isfinite(Tm), Tm, np.nan)
    roofline_spread = float(np.nanmax(np.nanmax(Tm, 1) - np.nanmin(Tm, 1)))
    order = ["cost_model", "work_only", "work_term_model", "nearest_calibrated_key", "loglinear_interp_logL_logB",
             "length_tree_calibration", "shape_tree_calibration", "gbr_basic", "gbr_structural", "mlp_basic",
             "mlp_structural", "compile_facts_rule", "tfla_runtime", "tfla_runtime_overlap", "tfla_runtime_T_over_C",
             "tfla_runtime_launch_floor"]
    table = {n: {f"top{k}": [round(rankers[n][f"top{k}"][m], 4) for m in ("mean_regret_pct", "p95_regret_pct",
                                                                            "max_regret_pct", "share_within_2pct")]
                 for k in (1, 2, 3)} for n in order if n in rankers}
    table["random_k"] = {f"top{k}": [round(rk[f"top{k}"][m], 4) for m in ("mean_regret_pct", "p95_regret_pct",
                                                                         "max_regret_pct", "share_within_2pct")]
                         for k in (1, 2, 3)}
    for fs in ("basic", "structural"):
        table[f"mlp_{fs}_mean_over_seeds"] = {f"top{k}": [round(seed_summary[f"mlp_{fs}"][f"top{k}"][m]["mean"], 4)
                                                          for m in ("mean_regret_pct", "p95_regret_pct",
                                                                    "max_regret_pct", "share_within_2pct")]
                                              for k in (1, 2, 3)}
    out = {"table_columns": ["mean_regret_pct", "p95_regret_pct", "max_regret_pct", "share_within_2pct"],
           "table": table, "rankers": rankers, "random_k": rk, "exact_shape_table_all_chunks_profiled": exact_table,
           "mlp_over_seeds": seed_summary, "learned": learned,
           "tfla_fit_info": {n: v for n, v in sel_info.items() if n.startswith("tfla_runtime") and "intensity" not in n},
           "compile_facts_rule_definition": "candidates: the six default-table chunks; a chunk whose compiled kernel "
                                            "(target arch, alignment of L) reports spill bytes ranks after every "
                                            "non-spilling chunk; within a class minimize waves x ceil(L/C) (waves from "
                                            "registers, warps and block limits); ties to the larger chunk; no fitted "
                                            "parameter",
           "roofline_note": {"max_over_calls_of_chunk_spread_of_bytes_over_bandwidth_ms": roofline_spread,
                             "note": "the bandwidth floor (bytes / peak bandwidth) is identical for every chunk of a "
                                     "call, so a roofline model alone ranks all chunks equal; the occupancy part is "
                                     "the compile-facts rule (waves) and the three-parameter restriction"},
           "checks": checks, "seconds": round(time.time() - t0, 1)}
    return out, mats, rankers


# =========================================================================== M4c: forms and LOSO per tag
def loso_fast(ctx, F, fixed: dict, x0) -> dict:
    """``cost_model_baselines_r4.loso`` on numpy arrays."""
    pts = ctx.points
    L, B, d = pts["L"].to_numpy(), pts["B"].to_numpy(), pts["d"].to_numpy()
    src, key, C, lat = pts["src"].to_numpy(), pts["key"].to_numpy(), pts["C"].to_numpy(int), ctx.y
    rows = []
    for k in sorted(pts.loc[pts["src"] == "e4_static", "key"].unique()):
        i0 = int(np.flatnonzero(key == k)[0])
        hold = (L == L[i0]) & (B == B[i0]) & (d == d[i0])
        th, _ = fit_fast(sub(F, ~hold), lat[~hold], x0=x0, fixed=fixed)
        st = (key == k) & (src == "e4_static")
        pst = predict_np(th, sub(F, st))
        meas = dict(zip(C[st], lat[st]))
        c_pred, c_best = int(C[st][int(np.argmin(pst))]), int(min(meas, key=meas.get))
        ph = predict_np(th, sub(F, hold))
        rows.append({"key": k, "chunk_pred": c_pred, "chunk_best": c_best,
                     "chunk_regret_pct": float(100 * (meas[c_pred] / meas[c_best] - 1)),
                     "heldout_rmse_log": float(np.sqrt(np.mean((np.log(ph) - np.log(lat[hold])) ** 2)))})
    r = np.array([x["chunk_regret_pct"] for x in rows])
    return {"mean_chunk_regret_pct": float(r.mean()), "max_chunk_regret_pct": float(r.max()),
            "se_chunk_regret_pct": float(r.std(ddof=1) / math.sqrt(len(r))), "n_shapes": len(rows),
            "heldout_rmse_log_mean": float(np.mean([x["heldout_rmse_log"] for x in rows])), "per_shape": rows}


def m4c_tag(ctx, F, Fc, stored: dict) -> tuple[dict, dict]:
    t0 = time.time()
    specs = r4.form_specs(ctx.theta)
    forms, choices = {}, {}
    for name in r4.ALL_FORMS:
        fixed, x0 = specs[name]
        th = ctx.theta.copy() if name == "full" else fit_fast(F, ctx.y, x0=x0, fixed=fixed)[0]
        S = ctx.calls.matrix(predict_np(th, Fc))
        choices[name] = S.argmin(1)
        forms[name] = {"n_free_params": len(cm.PARAMS) - len(fixed),
                       "params": {k: float(v) for k, v in zip(cm.PARAMS, th)},
                       "calibration_objective_cost": loss_cost(np.log(predict_np(th, F)) - np.log(ctx.y)),
                       "captured_calls": reg_stats(call_regret(ctx.lat, choices[name])),
                       "loso": loso_fast(ctx, F, fixed, th)}
    checks = {}
    for n, r in forms.items():
        sr = stored["forms"][n]
        checks[f"form_{n}_captured_mean_vs_r4"] = chk(sr["captured_calls"]["mean_regret_pct"],
                                                      r["captured_calls"]["mean_regret_pct"], 1e-4)
        checks[f"form_{n}_loso_mean_vs_r4"] = chk(sr["leave_one_shape_out"]["mean_chunk_regret_pct"],
                                                  r["loso"]["mean_chunk_regret_pct"], 1e-4)
    full_cost = forms["full"]["calibration_objective_cost"]
    tie = {}
    for alt, par in (("drop_scan_work", "tau_e"), ("drop_spill_l1_miss", "tau_p")):
        diff = int((choices[alt] != choices["full"]).sum())
        tie[alt] = {"dropped_parameter": par, "full_fit_value_of_dropped_parameter": forms["full"]["params"][par],
                    "n_calls_choice_differs_from_full": diff, "identical_choices": diff == 0,
                    "mean_regret_pct": forms[alt]["captured_calls"]["mean_regret_pct"],
                    "full_mean_regret_pct": forms["full"]["captured_calls"]["mean_regret_pct"],
                    "calibration_objective_cost": forms[alt]["calibration_objective_cost"],
                    "full_calibration_objective_cost": full_cost,
                    "loso_mean_chunk_regret_pct": forms[alt]["loso"]["mean_chunk_regret_pct"],
                    "full_loso_mean_chunk_regret_pct": forms["full"]["loso"]["mean_chunk_regret_pct"]}
    for n in forms:
        forms[n]["loso"].pop("per_shape")
    return {"forms": forms, "exact_tie_check": tie, "checks": checks, "seconds": round(time.time() - t0, 1)}, choices


# =========================================================================== M3c: calibration fit quality
def default_pairs(ctx) -> dict:
    """Log ratios of timings of the same configuration under two protocols: E4 search medians (3/10)
    vs E4 static re-measurements (10/100) at the default launch entries, and E1 medians vs E4 static."""
    pts = ctx.points
    st = {(r.L, r.B, int(r.C)): r.lat for r in pts[pts["src"] == "e4_static"].itertuples()}
    se = [math.log(r.lat / st[(r.L, r.B, int(r.C))]) for r in pts[pts["src"] == "e4_search"].itertuples()
          if (int(r.BD), int(r.nw)) == cm.DEFAULT_LAUNCH[int(r.C)] and int(r.ns) == 1 and (r.L, r.B, int(r.C)) in st]
    e1 = [math.log(r.lat / st[(r.L, r.B, int(r.C))]) for r in pts[pts["src"] == "e1"].itertuples()
          if (r.L, r.B, int(r.C)) in st]
    return {"search_vs_static": np.array(se), "e1_vs_static": np.array(e1)}


def inverse_variance_weights(ctx) -> tuple[np.ndarray, dict]:
    """Source-level inverse-variance weights. Full-protocol variance s2_full = mean square of
    log(E1 median / E4 static) over the 18 matched configurations / 2 (two independent full-protocol
    timings); search variance s2_search = mean square of log(search median / E4 static) over the 72
    default-entry configurations minus s2_full (floored at s2_full). Weights 1/s2, normalized to mean 1."""
    pr = default_pairs(ctx)
    s2_full = float(np.mean(pr["e1_vs_static"] ** 2) / 2)
    s2_search = float(max(np.mean(pr["search_vs_static"] ** 2) - s2_full, s2_full))
    src = ctx.points["src"].to_numpy()
    w = np.where(src == "e4_search", 1.0 / s2_search, 1.0 / s2_full)
    w = w / w.mean()
    return w, {"s2_full_protocol_log": s2_full, "s2_search_log": s2_search, "n_pairs_search": int(len(pr["search_vs_static"])),
               "n_pairs_e1": int(len(pr["e1_vs_static"])), "weight_ratio_full_over_search": s2_search / s2_full}


def variant_record(ctx, F, Fc, th, stored_choice, fit_info=None, fit_mask=None, loss="soft_l1") -> dict:
    S = ctx.calls.matrix(predict_np(th, Fc))
    ch = S.argmin(1)
    rec = {"params": {cm.PARAMS[i]: float(th[i]) for i in range(len(cm.PARAMS)) if i not in cm.DEFAULT_FIXED},
           "calibration_fit_all_2298": cm.fit_quality(predict_np(th, F), ctx.y),
           "captured_calls": reg_stats(call_regret(ctx.lat, ch)),
           "share_calls_same_choice_as_stored_fit": float((ch == stored_choice).mean())}
    if fit_mask is not None:
        Fm = sub(F, fit_mask)
        rec["n_fit_points"] = int(fit_mask.sum()) if fit_mask.dtype == bool else int(len(fit_mask))
        rec["calibration_fit_on_fit_points"] = cm.fit_quality(predict_np(th, Fm), ctx.y[fit_mask])
        rec["objective_r2_on_fit_points"] = objective_r2(th, Fm, ctx.y[fit_mask], loss)
    if fit_info is not None:
        rec["fit_cost"] = fit_info["cost"]
    return rec


def m3c_tag(ctx, F, Fc, opts, stored_choice) -> dict:
    t0 = time.time()
    pts = ctx.points
    src = pts["src"].to_numpy()
    allm = np.ones(len(src), bool)
    out = {}
    th0, info0 = fit_fast(F, ctx.y)
    out["stored_soft_l1"] = variant_record(ctx, F, Fc, ctx.theta, stored_choice, None, allm)
    out["stored_soft_l1"]["refit_cost_from_X0"] = info0["cost"]
    out["stored_soft_l1"]["loss_cost_formula_equals_scipy_cost"] = bool(
        abs(loss_cost(np.log(predict_np(th0, F)) - np.log(ctx.y)) - info0["cost"]) <= 1e-9 * max(1.0, info0["cost"]))
    # (1) full-protocol timings only
    m = np.isin(src, ["e4_static", "e1"])
    th, info = fit_fast(sub(F, m), ctx.y[m])
    out["full_protocol_only"] = variant_record(ctx, F, Fc, th, stored_choice, info, m)
    # (2) inverse-variance weighting (source-level)
    w, winfo = inverse_variance_weights(ctx)
    th, info = fit_fast(F, ctx.y, weights=w)
    rec = variant_record(ctx, F, Fc, th, stored_choice, info)
    rec["weights"] = winfo
    rec["objective_r2"] = objective_r2(th, F, ctx.y, "soft_l1", w)
    out["inverse_variance_weighted"] = rec
    # (3) without the search timings of the first (cold-cache) E4 key
    first = read_jsonl(ctx.in_dir / f"e4_{ctx.tag}.jsonl", kind="m1tune")[0]["key"]
    m = ~((src == "e4_search") & (pts["key"].to_numpy() == first))
    th, info = fit_fast(sub(F, m), ctx.y[m])
    out["exclude_first_key_search"] = variant_record(ctx, F, Fc, th, stored_choice, info, m)
    out["exclude_first_key_search"]["excluded_key"] = first
    # (4) OLS on all timings
    th, info = fit_fast(F, ctx.y, loss="linear")
    out["ols_all"] = variant_record(ctx, F, Fc, th, stored_choice, info, allm, loss="linear")
    # (5) random multi-starts (one least-squares run each)
    rng = np.random.default_rng(SEED)
    free = [i for i in range(len(cm.PARAMS)) if i not in cm.DEFAULT_FIXED]
    starts = []
    for _ in range(opts.n_multistart):
        x = cm.X0.copy()
        for i in free:
            if i < cm.N_SCALE:
                x[i] = cm.X0[i] * 10.0 ** rng.uniform(-2, 2)
            elif cm.PARAMS[i] == "W0":
                x[i] = rng.uniform(1.0, 64.0)
            else:
                x[i] = rng.uniform(1.0, 256.0)
        starts.append(np.clip(x, cm.LO, cm.HI))
    ms = []
    for x in starts:
        th, cost = fit_single(F, ctx.y, x)
        ch = ctx.calls.matrix(predict_np(th, Fc)).argmin(1)
        ms.append({"cost": cost, "mean_regret_pct": float(100 * call_regret(ctx.lat, ch).mean()),
                   "same_choice_share": float((ch == stored_choice).mean()), "theta": th})
    costs = np.array([x["cost"] for x in ms])
    best = int(np.argmin(costs))
    near = costs <= costs[best] * 1.01
    regs = np.array([x["mean_regret_pct"] for x in ms])
    out["multistart"] = {
        "n_starts": len(ms), "start_distribution": "time constants X0 x 10^U(-2, 2), W0 ~ U(1, 64), W1 ~ U(1, 256), "
                                                   "clipped to the bounds; one soft-l1 run each (seed 0)",
        "stored_protocol_cost": info0["cost"], "best_cost": float(costs[best]),
        "best_cost_rel_improvement_over_stored_protocol": float(1 - costs[best] / info0["cost"]),
        "n_within_1e-6_rel_of_best": int((costs <= costs[best] * (1 + 1e-6)).sum()),
        "n_within_1pct_of_best": int(near.sum()),
        "n_reaching_stored_protocol_cost_1e-6": int((costs <= info0["cost"] * (1 + 1e-6)).sum()),
        "best_start_captured_mean_regret_pct": float(regs[best]),
        "best_start_same_choice_share": ms[best]["same_choice_share"],
        "best_start_params": {cm.PARAMS[i]: float(ms[best]["theta"][i]) for i in free},
        "regret_among_starts_within_1pct_of_best": {"min": float(regs[near].min()), "median": float(np.median(regs[near])),
                                                    "max": float(regs[near].max())},
        "regret_all_starts": {"min": float(regs.min()), "median": float(np.median(regs)), "max": float(regs.max())},
        "cost_all_starts": {"min": float(costs.min()), "median": float(np.median(costs)), "max": float(costs.max())}}
    # (6) cluster bootstrap over the 12 calibration shapes
    shapes = sorted(set(zip(pts["L"], pts["B"])))
    cl = np.array([shapes.index(s) for s in zip(pts["L"], pts["B"])])
    cl_idx = [np.flatnonzero(cl == j) for j in range(len(shapes))]
    rng = np.random.default_rng(SEED + 1)
    boots = []
    for _ in range(opts.n_boot):
        draw = rng.integers(len(shapes), size=len(shapes))
        idx = np.concatenate([cl_idx[j] for j in draw])
        th, _ = fit_fast(sub(F, idx), ctx.y[idx])
        ch = ctx.calls.matrix(predict_np(th, Fc)).argmin(1)
        reg = call_regret(ctx.lat, ch)
        boots.append({"theta": th, "mean": 100 * reg.mean(), "p95": 100 * np.percentile(reg, 95),
                      "w2": float((reg <= 0.02).mean()), "same": float((ch == stored_choice).mean()),
                      "n_distinct": int(len(set(draw.tolist())))})
    out["cluster_bootstrap"] = {
        "n_replicates": len(boots), "clusters": [f"L{l}|b{b}" for l, b in shapes],
        "protocol": "resample the 12 calibration shapes with replacement (all their E4 search, E4 static and E1 "
                    "points), refit from X0 (stored three-restart protocol), score on the captured calls; "
                    "percentile intervals",
        "params": {cm.PARAMS[i]: {"stored": float(ctx.theta[i]), **pct_ci([b["theta"][i] for b in boots])} for i in free},
        "captured_mean_regret_pct": {"stored": out["stored_soft_l1"]["captured_calls"]["mean_regret_pct"],
                                     **pct_ci([b["mean"] for b in boots])},
        "captured_p95_regret_pct": pct_ci([b["p95"] for b in boots]),
        "captured_share_within_2pct": pct_ci([b["w2"] for b in boots]),
        "share_same_choice_as_stored_fit": pct_ci([b["same"] for b in boots]),
        "distinct_clusters_per_replicate_mean": float(np.mean([b["n_distinct"] for b in boots])),
        "replicate_mean_regret_pct": [round(b["mean"], 4) for b in boots]}
    # (7) R^2 of the minimized objective (stored soft-l1 fit)
    out["objective_r2_stored_fit"] = out["stored_soft_l1"].pop("objective_r2_on_fit_points")
    pr = default_pairs(ctx)
    out["search_vs_full_protocol"] = {
        "n_default_entry_pairs": int(len(pr["search_vs_static"])),
        "max_pct": float(100 * (np.exp(pr["search_vs_static"].max()) - 1)),
        "min_pct": float(100 * (np.exp(pr["search_vs_static"].min()) - 1)),
        "median_abs_pct": float(100 * np.median(np.abs(np.exp(pr["search_vs_static"]) - 1))),
        "share_abs_above_10pct": float((np.abs(np.exp(pr["search_vs_static"]) - 1) > 0.10).mean()),
        "e1_vs_static_median_abs_pct": float(100 * np.median(np.abs(np.exp(pr["e1_vs_static"]) - 1))),
        "e1_vs_static_max_abs_pct": float(100 * np.max(np.abs(np.exp(pr["e1_vs_static"]) - 1))),
        "n_points_by_source": {s: int((src == s).sum()) for s in ("e4_search", "e4_static", "e1")}}
    out["summary_captured_mean_regret_pct"] = {
        k: out[k]["captured_calls"]["mean_regret_pct"] for k in ("stored_soft_l1", "full_protocol_only",
                                                                 "inverse_variance_weighted", "exclude_first_key_search",
                                                                 "ols_all")}
    out["seconds"] = round(time.time() - t0, 1)
    return out


# =========================================================================== M3a: launch-table coupling
def shapley_split(S: dict, a: tuple, b: tuple) -> dict | None:
    """Two-path Shapley split of f(b) - f(a) between the chunk and the launch configuration; a, b are
    (C, BD, nw, ns). Parts are reported as gains (f(a) - f(b) > 0 when b is faster) in % of f(b)."""
    ab, ba = (b[0],) + a[1:], (a[0],) + b[1:]
    if any(k not in S for k in (a, b, ab, ba)):
        return None
    fa, fb, fx, fy = S[a], S[b], S[ab], S[ba]
    chunk = 0.5 * ((fx - fa) + (fb - fy))
    launch = 0.5 * ((fy - fa) + (fb - fx))
    return {"from": list(a), "to": list(b), "regret_pct": 100 * (fa / fb - 1),
            "chunk_part_pct": -100 * chunk / fb, "launch_part_pct": -100 * launch / fb,
            "gain_ms": fa - fb, "chunk_gain_ms": -chunk, "launch_gain_ms": -launch}


def m3a_tag(tag: str, in_dir: Path) -> dict:
    rows = read_jsonl(in_dir / f"e4_{tag}.jsonl", kind="m1tune")
    dl = {c: (c, *cm.DEFAULT_LAUNCH[c], 1) for c in CHUNKS}
    ratio = {c: float(np.mean([r["final"][f"static_{c}"]["mean_ms"] / r["final"]["tuned"]["mean_ms"] for r in rows]))
             for c in CHUNKS}
    c_gs = min(ratio, key=ratio.get)
    per_key, fixed_all, dec = [], [], {}
    pairs = ("default256_to_best_default_table", "default256_to_best_overall", "gpu_best_static_to_best_default_table",
             "gpu_best_static_to_best_overall", "best_default_table_to_best_overall")
    for p in pairs:
        dec[p] = []
    for r in rows:
        S = {(s["block_t"], s["block_d"], s["num_warps"], s["num_stages"]): s["median_ms"] for s in r["search"]
             if s.get("median_ms")}
        dflt = {c: S[dl[c]] for c in CHUNKS}
        full = {c: r["final"][f"static_{c}"]["median_ms"] for c in CHUNKS}
        c_def = min(dflt, key=dflt.get)
        best = min(S, key=S.get)
        launches = sorted({k[1:] for k in S})
        fixed = {}
        for la in launches:
            vals = {c: S[(c,) + la] for c in CHUNKS if (c,) + la in S}
            if len(vals) >= 2:
                cb = min(vals, key=vals.get)
                fixed[la] = {"spread_pct": 100 * (max(vals.values()) / vals[cb] - 1), "best_chunk": cb,
                             "n_chunks": len(vals)}
                fixed_all.append({"key": r["key"], "launch": la, **fixed[la], "same_best_as_default_table": cb == c_def})
        spreads = np.array([v["spread_pct"] for v in fixed.values()])
        rec = {"key": r["key"], "L": r["L"], "batch": r["batch"],
               "default_table_spread_search_pct": 100 * (max(dflt.values()) / dflt[c_def] - 1),
               "default_table_spread_full_protocol_pct": 100 * (max(full.values()) / min(full.values()) - 1),
               "best_default_chunk_search": c_def, "best_default_chunk_full_protocol": min(full, key=full.get),
               "fixed_launch_spread_median_pct": float(np.median(spreads)),
               "fixed_launch_spread_max_pct": float(spreads.max()),
               "fixed_launch_spread_at_default_launches_pct": {f"BD{la[0]}_nw{la[1]}": fixed[la]["spread_pct"]
                                                               for la in sorted({dl[c][1:] for c in CHUNKS})},
               "fixed_launch_spread_at_launch_of_best_default_entry_pct": fixed[dl[c_def][1:]]["spread_pct"],
               "best_overall_config": list(best),
               "best_default_over_best_overall_search_pct": 100 * (dflt[c_def] / S[best] - 1),
               "best_static_over_tuned_full_protocol_pct": 100 * (min(r["final"][f"static_{c}"]["mean_ms"] for c in CHUNKS)
                                                                  / r["final"]["tuned"]["mean_ms"] - 1)}
        targets = {"best_default_table": dl[c_def], "best_overall": best}
        for p in pairs:
            a_name, b_name = p.split("_to_")
            a = {"default256": dl[256], "gpu_best_static": dl[c_gs], "best_default_table": dl[c_def]}[a_name]
            dec[p].append(dict(key=r["key"], **(shapley_split(S, a, targets[b_name]) or {"missing_corner": True})))
        per_key.append(rec)

    def agg(lst):
        ok = [x for x in lst if "missing_corner" not in x]
        g = np.array([x["gain_ms"] for x in ok])
        cg = np.array([x["chunk_gain_ms"] for x in ok])
        return {"n_keys": len(ok), "n_keys_missing_corner": len(lst) - len(ok),
                "mean_regret_pct": float(np.mean([x["regret_pct"] for x in ok])),
                "mean_chunk_part_pct": float(np.mean([x["chunk_part_pct"] for x in ok])),
                "mean_launch_part_pct": float(np.mean([x["launch_part_pct"] for x in ok])),
                "max_launch_part_pct": float(np.max([x["launch_part_pct"] for x in ok])),
                "max_chunk_part_pct": float(np.max([x["chunk_part_pct"] for x in ok])),
                "chunk_share_of_summed_gain": float(cg.sum() / g.sum()) if g.sum() > 0 else None,
                "n_keys_launch_part_above_chunk_part": int(sum(x["launch_part_pct"] > x["chunk_part_pct"] for x in ok)),
                "per_key": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()} for x in lst]}
    tvs = np.array([k["best_static_over_tuned_full_protocol_pct"] for k in per_key])
    summ = {
        "default_table_spread_search_pct_median": float(np.median([k["default_table_spread_search_pct"] for k in per_key])),
        "default_table_spread_full_protocol_pct_median": float(np.median([k["default_table_spread_full_protocol_pct"] for k in per_key])),
        "fixed_launch_spread_pct_median_over_key_launch": float(np.median([x["spread_pct"] for x in fixed_all])),
        "fixed_launch_spread_pct_median_of_per_key_max": float(np.median([k["fixed_launch_spread_max_pct"] for k in per_key])),
        "fixed_launch_spread_at_launch_of_best_default_entry_pct_median":
            float(np.median([k["fixed_launch_spread_at_launch_of_best_default_entry_pct"] for k in per_key])),
        "share_key_launch_pairs_same_best_chunk_as_default_table": float(np.mean([x["same_best_as_default_table"] for x in fixed_all])),
        "best_default_over_best_overall_search_pct": {"geomean": float(100 * (np.exp(np.mean(np.log1p(
            np.array([k["best_default_over_best_overall_search_pct"] for k in per_key]) / 100))) - 1)),
            "max": float(max(k["best_default_over_best_overall_search_pct"] for k in per_key))},
        "best_static_over_tuned_full_protocol_pct": {"geomean": float(100 * (np.exp(np.mean(np.log1p(tvs / 100))) - 1)),
                                                     "max": float(tvs.max())}}
    return {"timing_protocol": "E4 search medians (3 warm-ups, 10 repeats) for every configuration and Shapley corner; "
                               "full-protocol E4 re-measurements (10/100) only where stated",
            "gpu_best_static_chunk": c_gs,
            "gpu_best_static_definition": "default-table chunk minimizing the mean over the 12 keys of static / tuned "
                                          "(e4_autotune.summarize_e4 'global_static_chunk')",
            "gpu_best_static_mean_ratio_to_tuned": ratio[c_gs], "summary": summ,
            "shapley": {p: agg(v) for p, v in dec.items()}, "per_key": per_key}


def m3a_registers(in_dir: Path) -> dict:
    """Registers, spills, shared memory and occupancy of the six default-table entries per architecture,
    and the register count against the chunk at every fixed launch configuration."""
    out = {}
    for gpu, arch in ARCH.items():
        d = json.loads((in_dir / f"kernel_stats_sm{arch}.json").read_text())
        hw = cm.hardware(gpu, in_dir)
        rows = {(bool(r["aligned"]), r["block_t"], r["block_d"], r["num_warps"], r["num_stages"]): r for r in d["configs"]}
        entries = []
        for al in (True, False):
            for c in CHUNKS:
                bd, nw = cm.DEFAULT_LAUNCH[c]
                r = rows.get((al, c, bd, nw, 1))
                if r is None:
                    continue
                alloc = int(math.ceil(r["regs"] / 8.0) * 8) * 32
                k = max(1, int(min(hw["regs_per_sm"] // (alloc * nw), hw["max_warps_per_sm"] // nw, hw["max_blocks_per_sm"])))
                entries.append({"aligned": al, "chunk": c, "block_d": bd, "num_warps": nw, "regs": r["regs"],
                                "spill_store_bytes": r.get("spill_store_bytes", 0), "spill_load_bytes": r.get("spill_load_bytes", 0),
                                "shared_bytes": r.get("shared_bytes"), "programs_per_batch_at_d2048": int(math.ceil(2048 / bd)),
                                "resident_programs_per_sm": k, "resident_warps_per_sm": k * nw})
        series = {}
        for la in sorted({k[2:] for k in rows if k[0]}):
            regs = [(c, rows[(True, c) + la]["regs"], rows[(True, c) + la].get("spill_store_bytes", 0)) for c in CHUNKS
                    if (True, c) + la in rows]
            series[f"BD{la[0]}_nw{la[1]}_ns{la[2]}"] = {
                "regs_by_chunk": {str(c): g for c, g, _ in regs},
                "first_spilling_chunk": next((c for c, _, s in regs if s > 0), None),
                "nondecreasing": bool(all(b[1] >= a[1] for a, b in zip(regs, regs[1:]))),
                "strictly_increasing": bool(all(b[1] > a[1] for a, b in zip(regs, regs[1:])))}
        al_entries = [e for e in entries if e["aligned"]]
        out[gpu] = {"arch": arch, "compile_facts_meta": {k: d["meta"].get(k) for k in ("gpu", "target_arch", "compile_only",
                                                                                     "pkg_triton", "dim_for_compile")},
                    "default_entries": entries,
                    "aligned_regs_by_chunk": [e["regs"] for e in al_entries],
                    "aligned_first_spilling_chunk": next((e["chunk"] for e in al_entries
                                                          if e["spill_store_bytes"] or e["spill_load_bytes"]), None),
                    "fixed_launch_series_aligned": series,
                    "n_fixed_launch_series": len(series),
                    "n_series_regs_nondecreasing_in_chunk": int(sum(v["nondecreasing"] for v in series.values())),
                    "n_series_regs_strictly_increasing_in_chunk": int(sum(v["strictly_increasing"] for v in series.values()))}
    ktxt = (Path(cm.__file__).parent / "kernels.py").read_text()
    m = re.search(r"DEFAULT_MAMBA1_LAUNCH\s*=\s*(\{[^}]*\})", ktxt)
    table = {int(k): tuple(v) for k, v in eval(m.group(1), {}).items()} if m else None   # literal dict in source
    flat = " ".join(line.strip().lstrip("#").strip() for line in ktxt.splitlines())
    out["launch_table_provenance"] = {
        "kernels_py_comment": "Per-chunk default launch configuration (BLOCK_D, num_warps), chosen once by a "
                              + PROVENANCE_QUOTE + " (num_stages 1)",
        "comment_found_in_kernels_py": PROVENANCE_QUOTE in flat,
        "kernels_py_table_equals_cost_model_DEFAULT_LAUNCH": table == {c: tuple(v) for c, v in cm.DEFAULT_LAUNCH.items()},
        "table": {str(c): list(v) for c, v in cm.DEFAULT_LAUNCH.items()}}
    return out


# =========================================================================== M5b: learning curve, cold start
def learning_curve(ctx, F, Fc, reps: int, rng) -> dict:
    pts = ctx.points
    PL, PB = pts["L"].to_numpy(), pts["B"].to_numpy()
    shapes = sorted(set(zip(PL, PB)))
    key_of = {(k["L"], k["B"]): k for k in ctx.keys}
    res = {}
    for k in range(1, len(shapes) + 1):
        n_all = math.comb(len(shapes), k)
        if n_all <= reps:
            subsets = list(combinations(range(len(shapes)), k))
        else:
            seen, subsets = set(), []
            while len(subsets) < reps:
                s = tuple(sorted(rng.choice(len(shapes), k, replace=False).tolist()))
                if s not in seen:
                    seen.add(s)
                    subsets.append(s)
        rows = []
        for s in subsets:
            sel = {shapes[i] for i in s}
            m = np.array([(a, b) in sel for a, b in zip(PL, PB)])
            th, _ = fit_fast(sub(F, m), ctx.y[m])
            reg = call_regret(ctx.lat, ctx.calls.matrix(predict_np(th, Fc)).argmin(1))
            nk = r4.NearestKey([key_of[sh] for sh in sorted(sel)])
            reg_nk = call_regret(ctx.lat, nk.scores(ctx.calls).argmin(1))
            rows.append({"keys": [f"L{a}|b{b}" for a, b in sorted(sel)], "n_points": int(m.sum()),
                         "model": reg_stats(reg), "nearest_key": reg_stats(reg_nk),
                         "theta_max_rel_diff_vs_stored": float(np.max(np.abs(th - ctx.theta) / np.maximum(np.abs(ctx.theta), 1e-12)))})

        def summ(who, metric):
            v = np.array([r[who][metric] for r in rows])
            return {"mean": float(v.mean()), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}
        res[str(k)] = {"n_subsets": len(subsets), "all_subsets_enumerated": n_all <= reps,
                       "model_mean_regret_pct": summ("model", "mean_regret_pct"),
                       "model_p95_regret_pct": summ("model", "p95_regret_pct"),
                       "model_share_within_2pct": summ("model", "share_within_2pct"),
                       "nearest_key_mean_regret_pct": summ("nearest_key", "mean_regret_pct"),
                       "nearest_key_p95_regret_pct": summ("nearest_key", "p95_regret_pct"),
                       "subsets": [{"keys": r["keys"], "n_points": r["n_points"],
                                    "model_mean_regret_pct": round(r["model"]["mean_regret_pct"], 4),
                                    "nearest_key_mean_regret_pct": round(r["nearest_key"]["mean_regret_pct"], 4)}
                                   for r in rows]}
        if k == len(shapes):
            res[str(k)]["full_set_theta_max_rel_diff_vs_stored"] = rows[0]["theta_max_rel_diff_vs_stored"]
    return res


def cold_start(ctx, choice_model) -> dict:
    lat, orc = ctx.lat, ctx.lat.min(1)
    ar = np.arange(len(lat))
    cal = np.mean([k["mean_ms"] / k["mean_ms"].min() - 1 for k in ctx.keys], axis=0)
    j_cal = int(np.argmin(cal))
    j_in = int(np.argmin((lat / orc[:, None] - 1).mean(0)))
    pols = {"cost_model": choice_model, "default_chunk_256": np.full(len(lat), ctx.chunks.index(256)),
            "best_static_from_calibration": np.full(len(lat), j_cal),
            "best_static_in_sample_captured": np.full(len(lat), j_in),
            "nearest_calibrated_key": r4.NearestKey(ctx.keys).scores(ctx.calls).argmin(1)}
    out = {}
    for n, c in pols.items():
        exc = lat[ar, c] - orc
        out[n] = {**reg_stats(call_regret(lat, c)), "mean_excess_us_per_call": float(1000 * exc.mean())}
    out["best_static_from_calibration"]["chunk"] = ctx.chunks[j_cal]
    out["best_static_in_sample_captured"]["chunk"] = ctx.chunks[j_in]
    m = out["cost_model"]
    benefit = {f"model_vs_{n}": {"regret_points": out[n]["mean_regret_pct"] - m["mean_regret_pct"],
                                 "us_per_call": out[n]["mean_excess_us_per_call"] - m["mean_excess_us_per_call"]}
               for n in pols if n != "cost_model"}
    return {"policies": out, "cold_start_benefit": benefit,
            "definition": "every captured call treated as the first call of an unprofiled key: the serving default is "
                          "the policy's chunk; excess = latency above the per-call oracle"}


def m5b_tag(ctx, F, Fc, opts, stored_choice, mats) -> tuple[dict, dict]:
    t0 = time.time()
    out = {"learning_curve": learning_curve(ctx, F, Fc, opts.lc_reps, np.random.default_rng(SEED))}
    m = (ctx.points["src"] == "e4_static").to_numpy()
    th, info = fit_fast(sub(F, m), ctx.y[m])
    ch72 = ctx.calls.matrix(predict_np(th, Fc)).argmin(1)
    out["static72_only_fit"] = {"n_fit_points": int(m.sum()), "fit_from": "X0",
                                "params": {cm.PARAMS[i]: float(th[i]) for i in range(len(cm.PARAMS)) if i not in cm.DEFAULT_FIXED},
                                "calibration_fit_on_fit_points": cm.fit_quality(predict_np(th, sub(F, m)), ctx.y[m]),
                                "captured_calls": reg_stats(call_regret(ctx.lat, ch72)),
                                "top2": topk_rows(ctx, ctx.calls.matrix(predict_np(th, Fc)), (2,))["top2"]}
    out["cold_start"] = cold_start(ctx, stored_choice)
    P = mats["cost_model"]
    rank = np.argsort(P, 1)
    choices = {"cost_model": stored_choice.tolist(), "static72": ch72.tolist(),
               "top2": r4.topk_picks(ctx.df, ctx.chunks, P, 2).tolist(),
               "top2_candidates": rank[:, :2].tolist()}
    out["seconds"] = round(time.time() - t0, 1)
    return out, choices


# =========================================================================== per-tag driver
def analyze_tag(tag: str, in_dir: Path, opts) -> tuple[dict, dict]:
    t_start = time.time()
    ctx = r4.Ctx(tag, in_dir)
    F, Fc = farr(ctx.f), farr(ctx.calls.fc)
    out = {"tag": tag, "n_calibration_points": int(len(ctx.y)), "n_calls": int(len(ctx.df))}
    # checks: numpy predictor and fit reproduce the stored model
    out["check_predict_np_equals_cost_model_predict"] = bool(
        np.array_equal(cm.predict(ctx.theta, ctx.f), predict_np(ctx.theta, F))
        and np.array_equal(cm.predict(ctx.theta, ctx.calls.fc), predict_np(ctx.theta, Fc)))
    th, info = fit_fast(F, ctx.y)
    out["check_refit_from_X0_max_rel_param_diff"] = float(
        np.max(np.abs(th - ctx.theta) / np.maximum(np.abs(ctx.theta), 1e-12)))
    S = ctx.calls.matrix(predict_np(ctx.theta, Fc))
    choice = S.argmin(1)
    reg = ctx.lat[np.arange(len(ctx.lat)), choice] / ctx.lat.min(1) - 1
    out["cost_model_top1"] = reg_stats(reg)
    out["check_choice_equals_stored_per_call_choice"] = bool(
        [int(ctx.chunks[i]) for i in choice] == ctx.stored["per_call_choice"]["mamba1"]["chunk"])
    out["check_calibration_r2_log_vs_stored"] = chk(ctx.stored["V1"]["all"]["r2_log"],
                                                    cm.fit_quality(predict_np(ctx.theta, F), ctx.y)["r2_log"], 1e-9)
    # M3b: does the bandwidth floor bind?
    T_cal, T_call = tile_time(ctx.theta, F), tile_time(ctx.theta, Fc)
    S_nofloor = ctx.calls.matrix(T_call)
    out["M3b_floor"] = {
        "calibration_points_floor_binds": int((F["t_mem_ms"] > T_cal).sum()),
        "calibration_max_ratio_floor_over_T": float(np.max(F["t_mem_ms"] / T_cal)),
        "captured_call_candidates_floor_binds": int((Fc["t_mem_ms"] > T_call).sum()),
        "captured_call_candidates": int(len(T_call)),
        "captured_calls_max_ratio_floor_over_T": float(np.max(Fc["t_mem_ms"] / T_call)),
        "captured_calls_choice_changed_by_floor": int((S.argmin(1) != S_nofloor.argmin(1)).sum()),
        "fixed_zero_terms_in_stored_params": {cm.PARAMS[i]: float(ctx.theta[i]) for i in cm.DEFAULT_FIXED}}
    stored = json.loads((in_dir / f"cost_model_baselines_r4_{tag}.json").read_text())
    out["M5a"], mats, _ = m5a_tag(ctx, stored, opts)
    print(f"  {tag}: M5a done ({out['M5a']['seconds']} s)", flush=True)
    out["M4c"], _ = m4c_tag(ctx, F, Fc, stored)
    print(f"  {tag}: M4c done ({out['M4c']['seconds']} s)", flush=True)
    out["M3c"] = m3c_tag(ctx, F, Fc, opts, choice)
    print(f"  {tag}: M3c done ({out['M3c']['seconds']} s)", flush=True)
    out["M5b"], choices = m5b_tag(ctx, F, Fc, opts, choice, mats)
    lc12 = out["M5b"]["learning_curve"]["12"]
    out["M5b"]["checks"] = {"learning_curve_k12_regret_vs_stored_fit": chk(out["cost_model_top1"]["mean_regret_pct"],
                                                                           lc12["model_mean_regret_pct"]["mean"], 1e-6),
                            "learning_curve_k12_nearest_key_vs_r4": chk(stored["baselines"]["nearest_calibrated_key"]["mean_regret_pct"],
                                                                        lc12["nearest_key_mean_regret_pct"]["mean"]),
                            "learning_curve_k12_theta_max_rel_diff": lc12["full_set_theta_max_rel_diff_vs_stored"]}
    print(f"  {tag}: M5b done ({out['M5b']['seconds']} s)", flush=True)
    out["M3a"] = m3a_tag(tag, in_dir)
    out["seconds"] = round(time.time() - t_start, 1)
    return out, choices


# =========================================================================== M4c cross-GPU selection
def select_tie_rule(cands: list[str], crit: dict, nfree: dict) -> tuple[str, list[str], list[str]]:
    """Fewest free parameters among the forms within TIE_POINTS of the best criterion value; then the
    lowest criterion; forms with the same parameter count and an identical criterion are exact ties."""
    best = min(crit[c] for c in cands)
    within = [c for c in cands if crit[c] <= best + TIE_POINTS + 1e-12]
    fewest = min(nfree[c] for c in within)
    pool = sorted([c for c in within if nfree[c] == fewest], key=lambda c: (crit[c], cands.index(c)))
    sel = pool[0]
    return sel, [c for c in pool[1:] if crit[c] == crit[sel]], within


def m4c_cross(per_tag: dict, in_dir: Path) -> dict:
    out, checks = {}, {}
    stored_sum = json.loads((in_dir / "cost_model_baselines_r4_summary.json").read_text())["clean_selection_cross_gpu"]
    for conv, tags in TAGS.items():
        if not all(t in per_tag for t in tags.values()):
            continue
        forms = {t: per_tag[t]["M4c"]["forms"] for t in tags.values()}
        nfree = {n: r["n_free_params"] for n, r in next(iter(forms.values())).items()}
        rows = {}
        for gpu, tgt in tags.items():
            srcs = [tags[g] for g in tags if g != gpu]

            def tstat(n):
                return forms[tgt][n]["captured_calls"]
            for set_name, cands in r4.CANDIDATE_SETS.items():
                crits = {"logo_captured_calls": {n: float(np.mean([forms[s][n]["captured_calls"]["mean_regret_pct"] for s in srcs]))
                                                 for n in cands},
                         "logo_calibration_loso": {n: float(np.mean([forms[s][n]["loso"]["mean_chunk_regret_pct"] for s in srcs]))
                                                   for n in cands},
                         "target_calibration_loso": {n: forms[tgt][n]["loso"]["mean_chunk_regret_pct"] for n in cands}}
                for cname, crit in crits.items():
                    sel, alts, within = select_tie_rule(cands, crit, nfree)
                    rows[f"{gpu}|{set_name}|{cname}"] = {
                        "target": tgt, "sources": srcs if cname != "target_calibration_loso" else [tgt],
                        "selected_form": sel, "n_free_params": nfree[sel], "criterion_value": crit[sel],
                        "criterion_best": min(crit.values()), "forms_within_0p1": within,
                        "target_captured_calls": tstat(sel),
                        "exact_tie_alternatives": {a: tstat(a) for a in alts},
                        "frozen_full_form_target": tstat("full")}
                if conv == "launch_inclusive":   # reproduce round 4's strict leave-one-GPU-out choice
                    second = {n: float(np.mean([forms[s][n]["loso"]["heldout_rmse_log_mean"] for s in srcs])) for n in cands}
                    n_strict = r4.select_strict(cands, crits["logo_captured_calls"], nfree, second)
                    st = stored_sum.get(tgt, {}).get(f"{set_name}|strict|leave_one_gpu_out", {}).get("selected_form")
                    checks[f"r4_strict_logo_{tgt}_{set_name}"] = {"stored": st, "recomputed": n_strict, "pass": st == n_strict}
        out[conv] = rows
    acc = {conv: {gpu: {"selected_form": out[conv][f"{gpu}|ablation12+work_only|logo_captured_calls"]["selected_form"],
                        **{k: out[conv][f"{gpu}|ablation12+work_only|logo_captured_calls"]["target_captured_calls"][k]
                           for k in ("mean_regret_pct", "p95_regret_pct", "max_regret_pct", "share_within_2pct")},
                        "frozen_full_form_mean_regret_pct":
                            out[conv][f"{gpu}|ablation12+work_only|logo_captured_calls"]["frozen_full_form_target"]["mean_regret_pct"]}
                  for gpu in TAGS[conv]} for conv in out}
    return {"protocol": "leave-one-GPU-out within one convention; criterion = mean over the two other GPUs; tie rule: "
                        "fewest free parameters among forms within 0.1 points of the best criterion, then lowest "
                        "criterion; primary = candidate set ablation12+work_only, criterion logo_captured_calls",
            "method_accuracy_primary": acc, "selections": out,
            "exact_tie_check": {t: per_tag[t]["M4c"]["exact_tie_check"] for t in per_tag}, "checks": checks}


# =========================================================================== M5b cost table
def workload_costs(tag: str, in_dir: Path, ch: dict) -> dict:
    """One-off costs of the model's calibration against profiling every key of the captured workload."""
    rows = read_jsonl(in_dir / f"e4_{tag}.jsonl")
    m1 = [r for r in rows if r.get("kind") == "m1tune"]
    scale = e5b._tuning_scale(rows)
    c_v = m1[0]["tuning_first_call_s"] / max(m1[0].get("n_configs", 1), 1)
    T = {"full_calibration_cold_s": float(sum(r["tuning_wall_s"] for r in m1)),
         "full_calibration_warm_s": float(sum(r["tuning_bench_s"] for r in m1)),
         "full_calibration_compile_s": float(sum(r["tuning_first_call_s"] for r in m1)),
         "static72_calibration_warm_s": float(sum(r["profile_static_wall_s"] for r in m1)),
         "compile_s_per_variant_cold": float(c_v)}
    T["static72_calibration_cold_s"] = T["static72_calibration_warm_s"] + len(CHUNKS) * c_v
    df = load_e2([in_dir / f"e2_mamba1_{tag}.csv"])
    df["aligned"] = (df["L"] % 16 == 0).astype(int)
    lat = latency_matrix(df, CHUNKS)
    orc, ar = lat.min(1), np.arange(len(df))
    al = df["aligned"].to_numpy()
    choices = {"cost_model": np.array(ch["cost_model"]), "cost_model_static72": np.array(ch["static72"]),
               "cost_model_top2": np.array(ch["top2"]), "default_chunk_256": np.full(len(df), CHUNKS.index(256))}
    out = {"one_off_costs": T, "tuning_scale": scale, "modes": {}}
    for mode, cols in (("exact", ["L", "batch"]), ("bucket_aligned", ["L_bucket", "batch", "aligned"])):
        groups = {k: np.asarray(v) for k, v in df.groupby(cols).indices.items()}
        tab = np.empty(len(df), dtype=int)
        P = {}
        for kv, idx in groups.items():
            mlat = lat[idx].mean(0)
            P[kv] = scale["profile_s_per_ms"] * float(mlat.sum())
            tab[idx] = int(np.argmin(mlat))
        sumP, meanP = float(sum(P.values())), float(np.mean(list(P.values())))
        n_spec = int(len(set(al.tolist())))
        nv = {n: int(len(set(zip(c.tolist(), al.tolist())))) for n, c in choices.items()}
        exc = {n: float((lat[ar, c] - lat[ar, tab]).mean() * 1000) for n, c in choices.items()}   # us per call
        strat = {"profile_every_key": {"warm_s": sumP, "cold_s": sumP + len(CHUNKS) * n_spec * c_v,
                                       "excess_over_profiled_table_us_per_call": 0.0},
                 "cost_model_only": {"warm_s": T["full_calibration_warm_s"],
                                     "cold_s": T["full_calibration_cold_s"] + nv["cost_model"] * c_v,
                                     "excess_over_profiled_table_us_per_call": exc["cost_model"]},
                 "cost_model_static72_only": {"warm_s": T["static72_calibration_warm_s"],
                                              "cold_s": T["static72_calibration_cold_s"] + nv["cost_model_static72"] * c_v,
                                              "excess_over_profiled_table_us_per_call": exc["cost_model_static72"]},
                 "default_chunk_256": {"warm_s": 0.0, "cold_s": n_spec * c_v,
                                       "excess_over_profiled_table_us_per_call": exc["default_chunk_256"]}}
        if mode == "exact":
            cand = np.array(ch["top2_candidates"])
            P2 = 0.0
            for kv, idx in groups.items():
                P2 += scale["profile_s_per_ms"] * float(lat[idx][:, cand[idx[0]]].mean(0).sum())
            nv2 = int(len({(int(c), int(a)) for row, a in zip(cand, al) for c in row}))
            strat["cost_model_top2"] = {"warm_s": T["full_calibration_warm_s"] + P2,
                                        "cold_s": T["full_calibration_cold_s"] + P2 + nv2 * c_v,
                                        "per_key_top2_profiling_s": P2,
                                        "excess_over_profiled_table_us_per_call": exc["cost_model_top2"]}
        for s in strat.values():
            s["model_cheaper_than_profile_every_key_warm"] = None
        for n in ("cost_model_only", "cost_model_static72_only", "cost_model_top2"):
            if n in strat:
                s = strat[n]
                s["one_off_cheaper_than_profile_every_key_warm"] = bool(s["warm_s"] < sumP)
                s["one_off_cheaper_than_profile_every_key_cold"] = bool(s["cold_s"] < strat["profile_every_key"]["cold_s"])
                s.pop("model_cheaper_than_profile_every_key_warm")
        out["modes"][mode] = {
            "n_keys": len(groups), "n_alignment_specializations": n_spec, "sum_P_profile_s": sumP, "mean_P_profile_s": meanP,
            "n_calls": int(len(df)), "table_excess_over_oracle_us_per_call": float((lat[ar, tab] - orc).mean() * 1000),
            "strategies": {k: {kk: vv for kk, vv in v.items() if vv is not None} for k, v in strat.items()},
            "payback_keys": {"full_calibration_warm": T["full_calibration_warm_s"] / meanP,
                             "full_calibration_cold": (T["full_calibration_cold_s"] + nv["cost_model"] * c_v
                                                       - len(CHUNKS) * n_spec * c_v) / meanP,
                             "static72_calibration_warm": T["static72_calibration_warm_s"] / meanP,
                             "definition": "number of keys (at this workload's mean per-key profiling cost) above which "
                                           "profiling every key costs more one-off time than the model's calibration; "
                                           "cold adds the compile of the variants each strategy launches"},
            "P_per_key_s": {"|".join(str(int(x)) for x in (kv if isinstance(kv, tuple) else (kv,))): v for kv, v in P.items()},
            "variants_launched": nv}
    return out


def m5b_cross(per_tag: dict, choices: dict, in_dir: Path) -> dict:
    out, checks = {}, {}
    li = TAGS["launch_inclusive"]
    for gpu, tag in li.items():
        if tag not in choices:
            continue
        w = workload_costs(tag, in_dir, choices[tag])
        out[gpu] = w
        e5b_p = in_dir / f"e5b_{tag}.json"
        if e5b_p.exists():
            st = json.loads(e5b_p.read_text())
            Pk = w["modes"]["bucket_aligned"]["P_per_key_s"]
            d = max(abs(Pk[f"{k['L_bucket']}|{k['batch']}|{k['aligned']}"] - k["P_s"]["profile"]) for k in st["keys"])
            checks[f"e5b_per_key_P_profile_{gpu}_max_abs_diff_s"] = {"value": d, "pass": d <= 1e-9}
            checks[f"e5b_tuning_scale_{gpu}"] = chk(st["tuning_scale"]["profile_s_per_ms"], w["tuning_scale"]["profile_s_per_ms"], 1e-12)
            checks[f"e5b_n_keys_{gpu}"] = {"stored": st["workload"]["n_keys"], "recomputed": w["modes"]["bucket_aligned"]["n_keys"],
                                           "pass": st["workload"]["n_keys"] == w["modes"]["bucket_aligned"]["n_keys"]}
            ex = st["by_key_mode"]["exact"]["n_keys"]
            checks[f"e5b_n_exact_keys_{gpu}"] = {"stored": ex, "recomputed": w["modes"]["exact"]["n_keys"],
                                                 "pass": ex == w["modes"]["exact"]["n_keys"]}
        e5_p = in_dir / f"e5_{tag}.json"
        if e5_p.exists():
            st = json.loads(e5_p.read_text())["view_e4_keys"]["mamba1"]["policies"]
            checks[f"e5_compile_per_config_ms_{gpu}"] = chk(st["profiled_static"]["compile_per_config_ms"],
                                                            1000 * w["one_off_costs"]["compile_s_per_variant_cold"], 1e-6)
            checks[f"e5_autotune_T_ms_{gpu}"] = chk(st["autotuned"]["T_ms"], 1000 * w["one_off_costs"]["full_calibration_cold_s"], 1e-6)
            checks[f"e5_profiled_static_T_ms_{gpu}"] = chk(st["profiled_static"]["T_ms"],
                                                           1000 * w["one_off_costs"]["static72_calibration_warm_s"], 1e-6)
    du_p, skc_p = in_dir / "deployment_units.json", in_dir / "serving_key_churn.json"
    if du_p.exists():
        du = json.loads(du_p.read_text())
        for gpu in out:
            st = du["m6b_captured_keys"].get(gpu)
            if st:
                checks[f"deployment_units_P_sum_33_keys_{gpu}"] = chk(st["P_profile_s_sum_33_keys"],
                                                                      out[gpu]["modes"]["bucket_aligned"]["sum_P_profile_s"], 1e-9)
    serving = {}
    if du_p.exists() and skc_p.exists():
        skc = json.loads(skc_p.read_text())
        du_rows = {(r["trace"], r["window"], r["policy"], r["T"], r["sched"], r["rho"], r["kernel"]): r
                   for r in du["serving"]["rows"]}
        prim = [r for r in skc["results"] if r["policy"] in ("per_request", "fixed_slice")
                or (r["sched"] == "colocated" and r["rho"] == skc["config"]["rho_primary"])]
        maxdiff = 0.0
        for gpu in [g for g in out if g in skc["hosts"]]:
            oc = out[gpu]["one_off_costs"]
            Kw = {m: out[gpu]["modes"][m]["payback_keys"]["full_calibration_warm"] for m in ("exact", "bucket_aligned")}
            res = {}
            for mode in ("exact", "bucket_aligned"):
                recs = []
                for r in prim:
                    k = r["keys"][mode]
                    sumP = k["by_host"][gpu]["cache_tuning_total_s"]["profile"]
                    dr = du_rows[(r["trace"], r["window"], r["policy"], r["T"], r["sched"], r["rho"], r["kernel"])]
                    m6 = dr["by_host"][gpu]["key_modes"][mode]["m6b"]
                    maxdiff = max(maxdiff, abs(m6["startup_profile_all_keys_s"] - sumP))
                    nreq = dr["n_requests"]
                    prof = nreq * m6["cost_ms_per_request"]["profile_all_keys_at_start"] / 1000
                    mod = nreq * m6["cost_ms_per_request"]["model_only"] / 1000
                    recs.append({"row": f"{r['trace']}/{r['window']}/{r['policy']}/T{r['T']}/{r['kernel']}",
                                 "n_keys": k["n_keys"], "n_calls": k["n_calls"], "sum_P_profile_s": sumP,
                                 "cold_share_warm10": k["cold_start"]["warm_10pct"]["share_later_calls_on_unseen_keys"],
                                 "total_profile_every_key_s": prof,
                                 "total_model_only_warm_s": oc["full_calibration_warm_s"] + mod,
                                 "total_model_only_cold_s": oc["full_calibration_cold_s"] + mod})
                sp = np.array([x["sum_P_profile_s"] for x in recs])
                nk = np.array([x["n_keys"] for x in recs])
                res[mode] = {"n_rows": len(recs),
                             "n_keys": {"median": float(np.median(nk)), "max": int(nk.max()), "min": int(nk.min())},
                             "sum_P_profile_s": {"median": float(np.median(sp)), "max": float(sp.max())},
                             "share_rows_profiling_every_key_cheaper_than_full_calibration_warm":
                                 float((sp < oc["full_calibration_warm_s"]).mean()),
                             "share_rows_profiling_every_key_cheaper_than_static72_calibration_warm":
                                 float((sp < oc["static72_calibration_warm_s"]).mean()),
                             "share_rows_n_keys_at_or_above_payback_warm": float((nk >= Kw[mode]).mean()),
                             "share_rows_model_total_cheaper_warm": float(np.mean([x["total_model_only_warm_s"] < x["total_profile_every_key_s"] for x in recs])),
                             "share_rows_model_total_cheaper_cold": float(np.mean([x["total_model_only_cold_s"] < x["total_profile_every_key_s"] for x in recs])),
                             "cold_share_warm10": {"median": float(np.median([x["cold_share_warm10"] for x in recs])),
                                                   "max": float(np.max([x["cold_share_warm10"] for x in recs]))},
                             "rows": recs}
            serving[gpu] = res
        checks["serving_sum_P_equals_deployment_units_startup_max_abs_diff_s"] = {"value": maxdiff, "pass": maxdiff <= 1e-9}
    cold = {t: per_tag[t]["M5b"]["cold_start"]["cold_start_benefit"] for t in per_tag}
    lc = {t: {k: [round(v["model_mean_regret_pct"]["mean"], 3), round(v["model_mean_regret_pct"]["min"], 3),
                  round(v["model_mean_regret_pct"]["max"], 3), round(v["nearest_key_mean_regret_pct"]["mean"], 3)]
              for k, v in per_tag[t]["M5b"]["learning_curve"].items()} for t in per_tag}
    return {"learning_curve_summary_columns": ["model_mean_over_subsets", "model_min", "model_max", "nearest_key_mean"],
            "learning_curve_summary_mean_regret_pct": lc,
            "captured_workload_costs": out, "serving_traces": serving, "cold_start_benefit": cold,
            "cost_conventions": "one-off costs are wall-clock seconds of the launch-inclusive E4 runs: full calibration "
                                "cold = E4 tuning wall (search incl. compile of 184 configurations, 12 keys), warm = "
                                "search timing only; static72 = profiling the six default entries at 12 keys "
                                "(profile_static_wall_s); per-key profiling P_k = E5b scaling (seconds per ms of the "
                                "key's summed six-chunk latency); compile per variant = first-key first-call time / 184 "
                                "(E5); serving rows: P_k sums and per-request costs from serving_key_churn.json and "
                                "deployment_units.json (cold compile of the served variants charged equally to both "
                                "strategies there, so cold differs from warm only by the calibration's own compile)",
            "checks": checks}


# =========================================================================== M5c: TFLA on TFLA
def tfla_counts_g(C, m: dict):
    """TFLA per-head, per-chunk byte and FLOP counts (same expressions as cost_model_baselines_r4.tfla_counts,
    general head dimensions)."""
    dh, dq, C = m["d_hv"], m["d_qk"], np.asarray(C, float)
    nbytes = 4 * C * m["bytes_if"] + 3 * C * (dh + dq) * m["bytes_qkv"] + 2 * dh * dq * m["bytes_cmn"]
    flops = (C ** 2 * m["f_causal"] * (2 * dq + 2 * dh + 6) + C ** 2 + C * (4 * dh * dq + 2 * dq + dh + 11)
             + 2 * dh * dq + 5)
    return nbytes, flops


def tfla_lopt_g(i_acc: float, m: dict) -> float:
    """argmin_C (Bytes(C) / beta + FLOPs(C) / alpha) / C for alpha / beta = i_acc (TFLA App. G.4)."""
    dh, dq = m["d_hv"], m["d_qk"]
    return float(math.sqrt((i_acc * 2 * dh * dq * m["bytes_cmn"] + 2 * dh * dq + 5)
                           / (m["f_causal"] * (2 * dq + 2 * dh + 6) + 1)))


class TFLAg:
    """TFLA's runtime N_batch N_head (T / L) (Bytes / beta + FLOPs / alpha) for TFLA's own kernel;
    th = (log alpha [FLOP/ms], log beta [byte/ms][, t0 ms])."""

    def __init__(self, th, mode: str, limit: str | None, m: dict):
        self.th, self.mode, self.limit, self.m = np.asarray(th, float), mode, limit, m

    def predict(self, B, H, L, C):
        nbytes, flops = tfla_counts_g(C, self.m)
        n = B * H * L / C                     # every timed E24 chunk divides L
        tb = 0.0 if self.limit == "no_bytes" else n * nbytes / math.exp(self.th[1])
        tf = 0.0 if self.limit == "no_flops" else n * flops / math.exp(self.th[0])
        t = np.maximum(tb, tf) if self.mode == "max" else tb + tf
        return t + self.th[2] if self.mode == "sum_floor" else t


def fit_tfla_g(B, H, L, C, t, mode: str, m: dict) -> tuple[TFLAg, dict]:
    """``cost_model_baselines_r4.fit_tfla`` (same starts, bounds, loss and boundary rule) for TFLAg."""
    y = np.log(t)
    nfloor = mode == "sum_floor"
    lo = [math.log(1e3), math.log(1e3)] + ([0.0] if nfloor else [])
    hi = [math.log(1e30), math.log(1e30)] + ([float(np.exp(y.max()))] if nfloor else [])
    fits = {}
    for limit in (None, "no_flops", "no_bytes"):
        best = None
        for la in (math.log(1e8), math.log(1e10), math.log(1e12), math.log(1e16)):
            for lb in (math.log(1e7), math.log(1e9), math.log(1e11)):
                x0 = [la, lb] + ([float(np.exp(y.min())) / 2] if nfloor else [])

                def res(th, limit=limit):
                    return np.log(TFLAg(th, mode, limit, m).predict(B, H, L, C)) - y
                sol = least_squares(res, x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.1, x_scale="jac", max_nfev=20000)
                if best is None or sol.cost < best.cost - 1e-12:
                    best = sol
        fits[limit] = best
    chosen = None
    for limit in ("no_flops", "no_bytes"):
        if fits[limit].cost <= fits[None].cost * (1 + 1e-6) + 1e-12 and (chosen is None or fits[limit].cost < fits[chosen].cost):
            chosen = limit
    best = fits[chosen]
    sel = TFLAg(best.x, mode, chosen, m)
    alpha = None if chosen == "no_flops" else math.exp(best.x[0])
    beta = None if chosen == "no_bytes" else math.exp(best.x[1])
    i_acc = alpha / beta if alpha is not None and beta is not None else (float("inf") if alpha is None else 0.0)
    info = {"mode": mode, "boundary": chosen, "fit_cost": {str(k): float(v.cost) for k, v in fits.items()},
            "alpha_tflops": None if alpha is None else alpha * 1e3 / 1e12,
            "beta_gbs": None if beta is None else beta * 1e3 / 1e9,
            "accelerator_intensity_flop_per_byte": i_acc if math.isfinite(i_acc) else "inf",
            "closed_form_runtime_optimal_chunk": tfla_lopt_g(i_acc, m) if math.isfinite(i_acc) and i_acc > 0 else None,
            "calibration_fit": cm.fit_quality(sel.predict(B, H, L, C), t)}
    if nfloor:
        info["t0_ms"] = float(best.x[2])
    return sel, info


def m5c_tfla(in_dir: Path, opts) -> dict:
    t0 = time.time()
    m = TFLA_E24_MAP
    out, checks = {"map": m, "map_note": "E24 inputs: bf16 q, k, v (2 bytes), fp32 gate pre-activations (4 bytes), fp32 "
                                       "chunk states (4 bytes); F_causal 0.5 and all F_OP = 1 as in r4"}, {}
    # the general counts reproduce r4's Mamba-1 instantiation
    Cg = np.array(CHUNKS, float)
    rb, rf = r4.tfla_counts(Cg)
    gb, gf = tfla_counts_g(Cg, {**r4.TFLA_MAP, "d_hv": r4.TFLA_MAP["d_hv"], "d_qk": r4.TFLA_MAP["d_qk"]})
    checks["tfla_counts_general_equals_r4"] = bool(np.allclose(rb, gb, rtol=0, atol=0) and np.allclose(rf, gf, rtol=0, atol=0))
    checks["tfla_lopt_general_equals_r4"] = bool(all(abs(tfla_lopt_g(i, r4.TFLA_MAP) - r4.tfla_lopt(i)) < 1e-9
                                                     for i in (64, 128, 256)))
    for conv, tag in TFLA_E24_TAGS.items():
        rows = [r for r in read_jsonl(in_dir / f"e24_tfla_{tag}.jsonl") if r.get("kind") == "timing"]
        table = e24.shape_table(rows)
        keys = sorted(table)
        assert all(table[k]["dqk"] == m["d_qk"] and table[k]["dv"] == m["d_hv"] for k in keys)
        pts = [(k, table[k]["batch"], table[k]["heads"], table[k]["L"], c, t) for k in keys
               for c, t in table[k]["median_ms"].items()]
        kk = np.array([keys.index(p[0]) for p in pts])
        B, H, L, C, T = (np.array([p[i] for p in pts], float) for i in range(1, 6))
        assert np.all(L % C == 0)

        def score(choose) -> dict:
            st = e24.regret_of(table, choose)
            w2 = [table[k]["median_ms"][choose(table[k])] / table[k]["median_ms"][table[k]["oracle"]] - 1 <= 0.02 for k in keys]
            return {**st, "share_within_2pct": float(np.mean(w2))}

        def choices_of(sel) -> dict:
            pr = sel.predict(B, H, L, C)
            ch = {}
            for j, k in enumerate(keys):
                idx = np.flatnonzero(kk == j)
                ch[k] = int(C[idx[int(np.argmin(pr[idx]))]])
            return ch
        res = {"n_shapes": len(keys), "n_points": int(len(T))}
        for mode in ("sum", "max", "sum_floor"):
            sel, info = fit_tfla_g(B, H, L, C, T, mode, m)
            ch = choices_of(sel)
            rec = {"fit": info, "in_sample": score(lambda s, ch=ch: ch[(s["layout"], s["L"], s["batch"])]),
                   "chosen_chunk_counts": {str(c): int(v) for c, v in zip(*np.unique(list(ch.values()), return_counts=True))}}
            if opts.tfla_loso:
                lch = {}
                for j, k in enumerate(keys):
                    tr = kk != j
                    s2, _ = fit_tfla_g(B[tr], H[tr], L[tr], C[tr], T[tr], mode, m)
                    idx = np.flatnonzero(kk == j)
                    lch[k] = int(C[idx[int(np.argmin(s2.predict(B[idx], H[idx], L[idx], C[idx])))]])
                rec["leave_one_shape_out"] = score(lambda s, lch=lch: lch[(s["layout"], s["L"], s["batch"])])
                rec["leave_one_shape_out"]["chosen_chunk_counts"] = {
                    str(c): int(v) for c, v in zip(*np.unique(list(lch.values()), return_counts=True))}
            res[f"tfla_{mode}"] = rec
        fixed = {}
        for i_acc in (64, RTX3090_BF16_INTENSITY, 128, 256, 512):
            sel = TFLAg([math.log(i_acc), 0.0], "sum", None, m)
            ch = choices_of(sel)
            fixed[str(i_acc)] = {"closed_form_runtime_optimal_chunk": tfla_lopt_g(i_acc, m),
                                 "regret": score(lambda s, ch=ch: ch[(s["layout"], s["L"], s["batch"])]),
                                 "chosen_chunk_counts": {str(c): int(v) for c, v in zip(*np.unique(list(ch.values()), return_counts=True))}}
        res["tfla_unfitted_intensity"] = fixed
        chunks = sorted({c for s in table.values() for c in s["median_ms"]})
        statics = {c: score(lambda s, c=c: c) for c in chunks if all(c in s["median_ms"] for s in table.values())}
        best_static = min(statics, key=lambda c: statics[c]["mean_pct"])
        res["library_default_128"] = statics.get(e24.LIBRARY_DEFAULT_CHUNK)
        res["best_static"] = {"chunk": best_static, **statics[best_static]}
        res["static_all_shapes"] = {str(c): v for c, v in statics.items()}
        stored_p = in_dir / f"e24_tfla_{tag}_analysis.json"
        if stored_p.exists():
            sa = json.loads(stored_p.read_text())
            res["e24_own_nnls_model_loso_stored"] = sa["cost_model"]["loso_regret"]
            checks[f"{tag}_n_shapes"] = {"stored": sa["n_shapes"], "recomputed": len(keys), "pass": sa["n_shapes"] == len(keys)}
            checks[f"{tag}_library_default_mean_pct"] = chk(sa["library_default"]["mean_pct"], res["library_default_128"]["mean_pct"], 1e-9)
            checks[f"{tag}_best_static_chunk"] = {"stored": sa["best_static"]["chunk"], "recomputed": best_static,
                                                  "pass": sa["best_static"]["chunk"] == best_static}
            checks[f"{tag}_best_static_mean_pct"] = chk(sa["best_static"]["mean_pct"], res["best_static"]["mean_pct"], 1e-9)
        res["single_static_chunk_note"] = (
            "TFLA's expression is B H (L/C) g(C): without a launch floor its argmin over C does not depend on the shape, "
            "so it selects one chunk wherever that chunk is timed (chosen_chunk_counts)")
        out[conv] = {"tag": tag, **res}
    out["checks"] = checks
    out["seconds"] = round(time.time() - t0, 1)
    return out


# =========================================================================== driver
def _job(spec):
    kind, arg, in_dir, opts = spec
    if kind == "tag":
        return kind, arg, analyze_tag(arg, in_dir, opts)
    return kind, arg, m5c_tfla(in_dir, opts)


def collect_checks(res: dict) -> dict:
    """Flatten every reproduction check into name -> pass (True / False / None)."""
    flat = {}

    def walk(prefix, obj):
        if isinstance(obj, dict) and "pass" in obj and not isinstance(obj["pass"], dict):
            flat[prefix] = obj["pass"]
        elif isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}", v)
        elif isinstance(obj, bool):
            flat[prefix] = obj
    walk("checks", res["checks"])
    for t, r in res["per_tag"].items():
        for item in ("M5a", "M4c", "M5b"):
            walk(f"per_tag.{t}.{item}.checks", r[item]["checks"])
        walk(f"per_tag.{t}.check_choice_equals_stored_per_call_choice", r["check_choice_equals_stored_per_call_choice"])
        walk(f"per_tag.{t}.check_calibration_r2_log_vs_stored", r["check_calibration_r2_log_vs_stored"])
        walk(f"per_tag.{t}.check_predict_np_equals_cost_model_predict", r["check_predict_np_equals_cost_model_predict"])
    for item in ("M4c", "M5b", "M5c"):
        if item in res and "checks" in res[item]:
            walk(f"{item}.checks", res[item]["checks"])
    return flat


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--out", type=Path, default=None, help="output JSON (default <in-dir>/taco_model_baselines.json)")
    p.add_argument("--tags", default=",".join(t for d in TAGS.values() for t in d.values()))
    p.add_argument("--jobs", type=int, default=2, help="worker processes (at most 2 are used)")
    p.add_argument("--n-boot", type=int, default=200, help="M3c cluster-bootstrap replicates")
    p.add_argument("--n-multistart", type=int, default=50, help="M3c random multi-starts")
    p.add_argument("--lc-reps", type=int, default=20, help="M5b learning-curve subsets per k")
    p.add_argument("--mc-draws", type=int, default=2000, help="M5a random-k Monte Carlo draws")
    p.add_argument("--mlp-seeds", type=int, default=5, help="M5a MLP seeds")
    p.add_argument("--no-tfla-loso", dest="tfla_loso", action="store_false", help="skip M5c leave-one-shape-out refits")
    opts = p.parse_args(argv)
    t0 = time.time()
    tags = [t for t in opts.tags.split(",") if t]
    res = {"experiment": "taco-r1-model-baselines", "definitions": __doc__, "tags": TAGS, "per_tag": {}, "checks": {},
           "replicate_counts": {"M3c_cluster_bootstrap": opts.n_boot, "M3c_multistarts": opts.n_multistart,
                                "M5b_learning_curve_subsets_per_k": opts.lc_reps, "M5a_random_k_mc_draws": opts.mc_draws,
                                "M5a_mlp_seeds": opts.mlp_seeds, "M5c_leave_one_shape_out": bool(opts.tfla_loso)}}
    jobs = [("tag", t, opts.in_dir, opts) for t in tags] + [("tfla", None, opts.in_dir, opts)]
    choices, tfla = {}, None
    with ProcessPoolExecutor(max_workers=max(1, min(opts.jobs, 2))) as ex:
        for kind, arg, r in ex.map(_job, jobs):
            if kind == "tag":
                res["per_tag"][arg], choices[arg] = r
                print(f"{arg}: done in {r[0]['seconds']} s ({time.time() - t0:.0f} s elapsed)", flush=True)
            else:
                tfla = r
                print(f"M5c TFLA: done in {r['seconds']} s", flush=True)
    for tag in tags:
        conv, gpu = TAG_INFO[tag]
        r = res["per_tag"][tag]
        pub, got = PUBLISHED_MEAN_PCT[conv][gpu], r["cost_model_top1"]["mean_regret_pct"]
        tol = 0.05 if conv == "launch_inclusive" else 0.005
        res["checks"][f"published_regret_{conv}_{gpu}"] = {"published": pub, "recomputed": got,
                                                            "pass": abs(round(got, 2 if tol < 0.01 else 1) - pub) <= tol}
        res["checks"][f"predict_np_exact_{tag}"] = r["check_predict_np_equals_cost_model_predict"]
        res["checks"][f"refit_reproduces_stored_{tag}"] = {"max_rel_param_diff": r["check_refit_from_X0_max_rel_param_diff"],
                                                           "pass": r["check_refit_from_X0_max_rel_param_diff"] < 1e-6}
        f = r["M3b_floor"]
        print(f"{conv:16s} {gpu:9s} cost model {got:.3f}% (published {pub}); floor binds: calibration "
              f"{f['calibration_points_floor_binds']}/{r['n_calibration_points']}, call candidates "
              f"{f['captured_call_candidates_floor_binds']}/{f['captured_call_candidates']}, choice changes "
              f"{f['captured_calls_choice_changed_by_floor']}")
    # ---- cross-tag parts
    res["M5a"] = {"protocol": "top-k: per exact shape (L, batch) profile the k best-ranked chunks, serve the fastest by "
                              "shape-mean latency; learned hyperparameters from the stored round-4 CV",
                  "summary_table_columns": ["mean_regret_pct", "p95_regret_pct", "max_regret_pct", "share_within_2pct"],
                  "summary_table": {t: res["per_tag"][t]["M5a"]["table"] for t in tags}}
    res["M4c"] = m4c_cross(res["per_tag"], opts.in_dir)
    res["M3c"] = {t: {"summary_captured_mean_regret_pct": res["per_tag"][t]["M3c"]["summary_captured_mean_regret_pct"],
                      "objective_r2_stored_fit": res["per_tag"][t]["M3c"]["objective_r2_stored_fit"]["r2_objective"],
                      "ols_r2_log": res["per_tag"][t]["M3c"]["ols_all"]["calibration_fit_all_2298"]["r2_log"],
                      "soft_l1_r2_log": res["per_tag"][t]["M3c"]["stored_soft_l1"]["calibration_fit_all_2298"]["r2_log"],
                      "bootstrap_mean_regret_ci95": res["per_tag"][t]["M3c"]["cluster_bootstrap"]["captured_mean_regret_pct"]["ci95"],
                      "multistart_best_cost_rel_improvement":
                          res["per_tag"][t]["M3c"]["multistart"]["best_cost_rel_improvement_over_stored_protocol"],
                      "search_vs_full_protocol_max_pct": res["per_tag"][t]["M3c"]["search_vs_full_protocol"]["max_pct"]}
                  for t in tags}
    if "vgpu3090" in res["per_tag"]:
        res["checks"]["M3c_review_226pct_search_vs_full_protocol_vgpu3090"] = {
            "review": 226.0, "recomputed": res["per_tag"]["vgpu3090"]["M3c"]["search_vs_full_protocol"]["max_pct"],
            "pass": abs(res["per_tag"]["vgpu3090"]["M3c"]["search_vs_full_protocol"]["max_pct"] - 226.0) < 0.5}
    res["M3a"] = {"registers": m3a_registers(opts.in_dir),
                  "per_tag_summary": {t: {"gpu_best_static_chunk": res["per_tag"][t]["M3a"]["gpu_best_static_chunk"],
                                          **res["per_tag"][t]["M3a"]["summary"],
                                          "shapley_mean": {p: {k: v[k] for k in ("mean_regret_pct", "mean_chunk_part_pct",
                                                                                "mean_launch_part_pct", "max_launch_part_pct",
                                                                                "chunk_share_of_summed_gain",
                                                                                "n_keys_launch_part_above_chunk_part", "n_keys")}
                                                           for p, v in res["per_tag"][t]["M3a"]["shapley"].items()}}
                                      for t in tags}}
    reg = res["M3a"]["registers"]
    res["checks"]["M3a_sm120_aligned_regs_64_126_202_202_255_255"] = reg["pro6000"]["aligned_regs_by_chunk"] == [64, 126, 202, 202, 255, 255]
    res["checks"]["M3a_launch_table_provenance_comment"] = reg["launch_table_provenance"]["comment_found_in_kernels_py"]
    res["checks"]["M3a_launch_table_equal"] = reg["launch_table_provenance"]["kernels_py_table_equals_cost_model_DEFAULT_LAUNCH"]
    for t in tags:
        e4s = json.loads((opts.in_dir / f"e4_{t}.json").read_text())["summary"]["mamba1"]
        a = res["per_tag"][t]["M3a"]
        res["checks"][f"M3a_e4_global_static_chunk_{t}"] = {"stored": e4s["global_static_chunk"], "recomputed": a["gpu_best_static_chunk"],
                                                            "pass": e4s["global_static_chunk"] == a["gpu_best_static_chunk"]}
        res["checks"][f"M3a_e4_global_static_ratio_{t}"] = chk(e4s["global_static_mean_ratio_to_tuned"],
                                                               a["gpu_best_static_mean_ratio_to_tuned"], 1e-12)
    if "pro6000" in res["per_tag"]:
        s = res["per_tag"]["pro6000"]["M3a"]["summary"]["best_static_over_tuned_full_protocol_pct"]
        res["checks"]["M3a_paper_text_2p8_gm_11p8_max_pro6000_li"] = {"paper": [2.8, 11.8], "recomputed": [s["geomean"], s["max"]],
                                                                       "pass": round(s["geomean"], 1) == 2.8 and round(s["max"], 1) == 11.8}
    res["M5b"] = m5b_cross(res["per_tag"], choices, opts.in_dir)
    if tfla is not None:
        sanity = {t: {n: {k: res["per_tag"][t]["M5a"]["rankers"][n][k]["mean_regret_pct"] for k in ("top1", "top2", "top3")}
                      for n in ("tfla_runtime", "tfla_runtime_overlap", "tfla_runtime_T_over_C", "tfla_runtime_launch_floor")}
                  for t in tags}
        tfla["sanity_check_tfla_expression_on_mamba1_scan"] = {
            "note": "TFLA's expression instantiated for the Mamba-1 scan (N_head = d, d_hv = 1, d_qk = 16) and fitted on its "
                    "calibration timings (cost_model_baselines_r4); a category mismatch kept only as a sanity check",
            "mean_regret_pct": sanity}
        res["M5c"] = tfla
    res["unavailable"] = {
        "M5b_serving_trace_costs_h800": {
            "reason": "the serving replay (serving_key_churn.json) and deployment_units.json hold per-host costs for pro6000 "
                      "and vgpu3090 only, and the raw traces (src/data/serving_traces) are not in the repository, so the "
                      "replay cannot be re-run with H800 costs; the H800 captured-workload cost table is given"},
        "M5b_h800_online_decision_overheads": {
            "reason": "e4_h800.jsonl has no host-synchronization or histogram-overhead rows (no E5/E5b run on the H800), "
                      "so N* against an online policy cannot be formed for the H800; the model-vs-profiling costs need "
                      "no overhead and are reported"},
        "M5b_calibration_wall_clock_of_static_and_e1_timings": {
            "reason": "the wall-clock time of the 72 full-protocol static re-measurements (10/100) and of the 18 E1 "
                      "medians is not recorded; the calibration cost uses the E4 tuning wall clock (search incl. compile), "
                      "as the paper does"},
        "M5b_device_time_cost_table": {
            "reason": "one-off costs are wall clocks of the launch-inclusive E4 runs; the device-time E4 runs add a GPU "
                      "spin before every timing block, so their wall clocks are not deployment costs; device-time "
                      "regrets are reported in the learning curve and cold-start sections"},
        "M3c_per_timing_inverse_variance": {
            "reason": "the E4 search rows store only the median of 10 repeats (no per-timing variance); weights are "
                      "source-level variances estimated from repeated configurations (search vs static, E1 vs static)"},
        "M5c_other_gpus": {
            "reason": "TFLA's kernel (E24) was timed only on the RTX 3090 (vgpu3090_r4spinb device time, vgpu3090_r4plainb "
                      "launch-inclusive; r4spin/r4plain are one-shape smoke runs)"},
        "M5a_M4c_mlp_per_seed_hyperparameter_cv": {
            "reason": "not run to stay within the runtime budget (12-fold CV x 3 penalties x 2 feature sets x 5 seeds "
                      "~ 18 min per tag); hyperparameters come from the stored seed-0 CV of round 4, and only the final "
                      "model is retrained per seed"},
    }
    res["checks_flat"] = collect_checks(res)
    failed = [k for k, v in res["checks_flat"].items() if v is False]
    res["checks_summary"] = {"n_checks": len(res["checks_flat"]),
                             "n_pass": sum(v is True for v in res["checks_flat"].values()),
                             "n_fail": len(failed), "failed": failed,
                             "n_not_applicable": sum(v is None for v in res["checks_flat"].values())}
    res["runtime_s"] = round(time.time() - t0, 1)
    write_json(opts.out or (opts.in_dir / "taco_model_baselines.json"), res)
    print("checks:", {k: v for k, v in res["checks"].items() if k.startswith("published")})
    print("check summary:", res["checks_summary"])
    print(f"runtime {res['runtime_s']} s")


if __name__ == "__main__":
    main()
