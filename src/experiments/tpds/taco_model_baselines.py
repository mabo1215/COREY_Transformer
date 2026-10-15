"""TACO revision (cycle taco-r1): offline re-analysis of the Mamba-1 cost model and its selector
baselines (review items M3a, M3b, M3c, M4c, M5a, M5b, M5c). CPU only; reads existing result files and
writes ``taco_model_baselines.json``; nothing is timed.

Conventions: two result tags per GPU (RTX PRO 6000 = pro6000, RTX 3090 = vgpu3090, H800 PCIe = h800):
  device_time (primary)        pro6000_r4spin, vgpu3090_r3spin, h800_r3spin
  launch_inclusive (secondary) pro6000, vgpu3090, h800
Captured-call latencies are the ``lat_c<C>_mean`` columns read by every published table; per-call
relative regret = lat(chosen chunk) / min over the six chunks - 1.

The cost-model predictor and fit are re-implemented on numpy arrays (``predict_np``, ``fit_fast``)
for speed; both are checked to reproduce ``cost_model.predict`` and the stored parameters exactly.

STATUS (session interrupted): implemented are the reproduction checks and item M3b (bandwidth floor
binding on calibration and captured calls). Items M5a, M4c, M3c, M3a, M5b, M5c are recorded under
"unavailable" with reason "not finished"; the planned definitions are kept in ``PLANNED`` below so a
later session can resume them.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from src.experiments.tpds import cost_model as cm  # noqa: E402
from src.experiments.tpds import cost_model_baselines_r4 as r4  # noqa: E402
from src.experiments.tpds.common import RESULTS_DIR, write_json  # noqa: E402

GPUS = ("pro6000", "vgpu3090", "h800")
TAGS = {"device_time": {"pro6000": "pro6000_r4spin", "vgpu3090": "vgpu3090_r3spin", "h800": "h800_r3spin"},
        "launch_inclusive": {"pro6000": "pro6000", "vgpu3090": "vgpu3090", "h800": "h800"}}
PUBLISHED_MEAN_PCT = {"device_time": {"pro6000": 0.38, "vgpu3090": 3.55, "h800": 3.01},
                      "launch_inclusive": {"pro6000": 0.9, "vgpu3090": 3.2, "h800": 3.5}}
FEAT = ("l2_miss", "loads", "e", "e_d", "lanes_c", "warps_c", "spill_words", "l1_miss", "m", "nw", "waves",
        "n_tiles", "t_mem_ms")

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


def farr(f) -> dict:
    return {k: f[k].to_numpy(float) for k in FEAT}


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


def fit_fast(F, y_ms, x0=None, fixed=None, loss="soft_l1"):
    """``cost_model.fit`` on numpy feature arrays (identical restarts, bounds, loss and tolerances)."""
    y = np.log(y_ms)
    x0 = cm.X0 if x0 is None else np.asarray(x0, float)
    fixed = cm.DEFAULT_FIXED if fixed is None else fixed
    free = [i for i in range(len(cm.PARAMS)) if i not in fixed]

    def full(v):
        th = np.empty(len(cm.PARAMS))
        th[free] = v
        for i, val in fixed.items():
            th[i] = val
        return th

    def res(v):
        return np.log(predict_np(full(v), F)) - y

    best = None
    for scale in (1.0, 0.3, 3.0):
        start = np.clip(x0 * np.r_[np.full(cm.N_SCALE, scale), np.ones(len(cm.PARAMS) - cm.N_SCALE)], cm.LO, cm.HI)
        sol = least_squares(res, start[free], bounds=(cm.LO[free], cm.HI[free]), loss=loss, f_scale=0.1,
                            x_scale="jac", max_nfev=5000)
        if best is None or sol.cost < best.cost:
            best = sol
    return full(best.x), {"cost": float(best.cost), "nfev": int(best.nfev)}


def reg_stats(reg) -> dict:
    return {"n": int(len(reg)), "mean_regret_pct": float(100 * reg.mean()),
            "p95_regret_pct": float(100 * np.percentile(reg, 95)), "max_regret_pct": float(100 * reg.max()),
            "share_within_2pct": float((reg <= 0.02).mean())}


def analyze_tag(tag: str, in_dir: Path) -> dict:
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
    reg = ctx.lat[np.arange(len(ctx.lat)), S.argmin(1)] / ctx.lat.min(1) - 1
    out["cost_model_top1"] = reg_stats(reg)
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
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv)
    t0 = time.time()
    res = {"experiment": "taco-r1-model-baselines", "definitions": __doc__, "tags": TAGS, "per_tag": {}, "checks": {}}
    for conv, tags in TAGS.items():
        for gpu, tag in tags.items():
            r = analyze_tag(tag, args.in_dir)
            res["per_tag"][tag] = r
            pub, got = PUBLISHED_MEAN_PCT[conv][gpu], r["cost_model_top1"]["mean_regret_pct"]
            tol = 0.05 if conv == "launch_inclusive" else 0.005
            res["checks"][f"published_regret_{conv}_{gpu}"] = {"published": pub, "recomputed": got,
                                                                "pass": abs(round(got, 2 if tol < 0.01 else 1) - pub) <= tol}
            res["checks"][f"predict_np_exact_{tag}"] = r["check_predict_np_equals_cost_model_predict"]
            res["checks"][f"refit_reproduces_stored_{tag}"] = r["check_refit_from_X0_max_rel_param_diff"]
            f = r["M3b_floor"]
            print(f"{conv:16s} {gpu:9s} cost model {got:.3f}% (published {pub}); floor binds: calibration "
                  f"{f['calibration_points_floor_binds']}/{r['n_calibration_points']}, call candidates "
                  f"{f['captured_call_candidates_floor_binds']}/{f['captured_call_candidates']}, choice changes "
                  f"{f['captured_calls_choice_changed_by_floor']}")
    res["unavailable"] = {k: {"reason": "not finished (session interrupted)", "plan": v} for k, v in PLANNED.items()}
    res["runtime_s"] = round(time.time() - t0, 1)
    write_json(args.in_dir / "taco_model_baselines.json", res)
    print("checks:", {k: v for k, v in res["checks"].items() if k.startswith("published")})


if __name__ == "__main__":
    main()
