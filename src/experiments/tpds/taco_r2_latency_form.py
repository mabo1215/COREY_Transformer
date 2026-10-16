"""taco-r2 item M4d: physical consistency of the latency term of Eq. (3) and a max(latency, throughput) refit.
CPU only; reads stored result files and times nothing.

Eq. (3) predicts the time per tile and wave as A min(1, W1 / W) + I max(1, W / W0), with A = N_s tau_m the serial
per-tile latency of the state loop, I the issue part, W the resident warps per SM. Beyond W1 warps the latency
term falls below A, so the model can predict a wave makespan below the critical path of one program. This script
(i) measures how often that happens with the stored fits, on the calibration timings and on the captured calls'
candidates (default launch table), and (ii) refits alternative forms with the same data, loss (soft-l1 on log
latency, f_scale 0.1), bounds, three-restart protocol and fixed terms, and reports their fit and selection:

  eq3                     the stored form (refit from X0 and from the stored parameters, the better kept)
  eq3_floored             max(A, A min(1, W1/W) + I max(1, W/W0)): Eq. (3) with the critical path as a floor
  max_latency_issue       max(A + I, I W / W0): one warp's dependent chain (latency plus its own issue) or the
                          issue throughput shared by W warps beyond W0, whichever binds (Volkov; Hong and Kim)
  max_latency_throughput  max(A, I max(1, W/W0)): latency and issue fully overlapped
  additive_no_hiding      A + I max(1, W/W0): no latency hiding at all (W1 -> infinity)

The forms without W1 have seven free parameters. For each form: calibration objective and its R^2 against the
best constant, R^2 of log latency, the leave-one-calibration-shape-out chunk regret (12 folds, a calibration-only
criterion), and on the 2360 captured calls the per-call and latency-weighted regret (with prompt-cluster
intervals and per batch) and the share of calls whose selected chunk differs from the stored fit. Also the ratio
of programs to SMs at batch 1 (the last-wave question).

Output: ``taco_r2_baselines_m4d.json``.

Usage:  PYTHONPATH=. python3 -m src.experiments.tpds.taco_r2_latency_form [--jobs 6]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from src.experiments.tpds import cost_model as cm  # noqa: E402
from src.experiments.tpds import cost_model_baselines_r4 as r4  # noqa: E402
from src.experiments.tpds import taco_model_baselines as TMB  # noqa: E402
from src.experiments.tpds import taco_r2_baselines_common as C  # noqa: E402
from src.experiments.tpds.common import RESULTS_DIR  # noqa: E402

FORMS = ("eq3", "eq3_floored", "max_latency_issue", "max_latency_throughput", "additive_no_hiding")
NO_W1 = ("max_latency_issue", "max_latency_throughput", "additive_no_hiding")
I_W1 = cm.PARAMS.index("W1")
TAGS = {"device_time": {"pro6000": "pro6000_r4spin", "vgpu3090": "vgpu3090_r3spin", "h800": "h800_r3spin"},
        "launch_inclusive": {"pro6000": "pro6000", "vgpu3090": "vgpu3090", "h800": "h800"}}
EVAL = {"device_time": {"pro6000": ("pro6000_device", "pro6000_device_pooled"),
                        "vgpu3090": ("vgpu3090_device", "vgpu3090_device_one_block"), "h800": ("h800_device",)},
        "launch_inclusive": {"pro6000": ("pro6000_launch",), "vgpu3090": ("vgpu3090_launch",), "h800": ("h800_launch",)}}


def parts(theta, F):
    """A (latency per tile), I (issue per tile), W (resident warps): the operations of ``cost_model.predict``."""
    t0, tf, td, tm, tl, te, ts, tx, tq, tp, W0, W1 = theta
    N = cm.D_STATE
    A = tf + td * F["l2_miss"] + N * tm
    I = tl * F["loads"] + N * (te * F["e"] + ts * F["e_d"] * np.log2(np.clip(F["lanes_c"], 1, None))
                               + tx * F["e_d"] * (F["warps_c"] > 1) + F["spill_words"] * (tq + tp * F["l1_miss"]))
    return A, I, F["m"] * F["nw"]


def tile(form: str, theta, F):
    A, I, W = parts(theta, F)
    W0, W1 = theta[10], theta[11]
    if form == "eq3":
        return A * np.minimum(1.0, W1 / W) + I * np.maximum(1.0, W / W0)
    if form == "eq3_floored":
        return np.maximum(A, A * np.minimum(1.0, W1 / W) + I * np.maximum(1.0, W / W0))
    if form == "max_latency_issue":
        return np.maximum(A + I, I * W / W0)
    if form == "max_latency_throughput":
        return np.maximum(A, I * np.maximum(1.0, W / W0))
    if form == "additive_no_hiding":
        return A + I * np.maximum(1.0, W / W0)
    raise ValueError(form)


def predict(form: str, theta, F):
    return np.maximum(theta[0] + F["waves"] * F["n_tiles"] * tile(form, theta, F), F["t_mem_ms"])


def fixed_of(form: str) -> dict:
    fx = dict(cm.DEFAULT_FIXED)
    if form in NO_W1:
        fx[I_W1] = cm.X0[I_W1]       # unused by the form; fixed so that it is not a free parameter
    return fx


def fit(form: str, F, y_ms, starts: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """``cost_model.fit`` protocol (soft-l1, f_scale 0.1, bounds, restarts with the time constants of each start
    scaled by 1, 0.3 and 3) for any form; the lowest cost over all starts and scales is kept."""
    y = np.log(y_ms)
    fx = fixed_of(form)
    free = [i for i in range(len(cm.PARAMS)) if i not in fx]

    def full(v):
        th = np.empty(len(cm.PARAMS))
        th[free] = v
        for i, val in fx.items():
            th[i] = val
        return th

    def res(v):
        return np.log(predict(form, full(v), F)) - y
    best = None
    for x0 in starts:
        for scale in (1.0, 0.3, 3.0):
            st = np.clip(np.asarray(x0, float) * np.r_[np.full(cm.N_SCALE, scale), np.ones(len(cm.PARAMS) - cm.N_SCALE)],
                         cm.LO, cm.HI)
            sol = least_squares(res, st[free], bounds=(cm.LO[free], cm.HI[free]), loss="soft_l1", f_scale=0.1,
                                x_scale="jac", max_nfev=5000)
            if best is None or sol.cost < best.cost:
                best = sol
    return full(best.x), float(best.cost)


def objective_r2(form, theta, F, y_ms) -> float:
    y = np.log(y_ms)
    r = np.log(predict(form, theta, F)) - y
    c = least_squares(lambda v: v[0] - y, [float(np.median(y))], loss="soft_l1", f_scale=0.1).cost
    return float(1 - TMB.loss_cost(r) / c)


def loso(form: str, ctx, F, theta) -> dict:
    pts = ctx.points
    L, B, d = pts["L"].to_numpy(), pts["B"].to_numpy(), pts["d"].to_numpy()
    src, key, Cc, lat = pts["src"].to_numpy(), pts["key"].to_numpy(), pts["C"].to_numpy(int), ctx.y
    regs = []
    for k in sorted(pts.loc[pts["src"] == "e4_static", "key"].unique()):
        i0 = int(np.flatnonzero(key == k)[0])
        hold = (L == L[i0]) & (B == B[i0]) & (d == d[i0])
        th, _ = fit(form, TMB.sub(F, ~hold), lat[~hold], [theta])
        st = (key == k) & (src == "e4_static")
        pst = predict(form, th, TMB.sub(F, st))
        meas = dict(zip(Cc[st], lat[st]))
        cp, cb = int(Cc[st][int(np.argmin(pst))]), int(min(meas, key=meas.get))
        regs.append(100 * (meas[cp] / meas[cb] - 1))
    r = np.array(regs)
    return {"mean_chunk_regret_pct": float(r.mean()), "max_chunk_regret_pct": float(r.max()), "n_shapes": len(r)}


def below_critical_path(theta, F) -> dict:
    A, I, W = parts(theta, F)
    t_eq3 = tile("eq3", theta, F)
    hidden = W > theta[11]
    below = t_eq3 < A
    return {"n": int(len(A)), "share_W_above_W1": float(hidden.mean()), "share_tile_time_below_A": float(below.mean()),
            "min_ratio_tile_time_to_A": float(np.min(t_eq3 / A)),
            "median_ratio_tile_time_to_A_where_below": float(np.median((t_eq3 / A)[below])) if below.any() else None,
            "W_range": [float(W.min()), float(W.max())]}


def tag_worker(job) -> dict:
    tag, in_dir = job
    t0 = time.time()
    ctx = r4.Ctx(tag, Path(in_dir))
    F, Fc = TMB.farr(ctx.f), TMB.farr(ctx.calls.fc)
    S0 = ctx.calls.matrix(TMB.predict_np(ctx.theta, Fc))
    stored_choice = S0.argmin(1)
    out = {"tag": tag, "keys": [str(k) for k in ctx.df["key"]], "chunks": [int(c) for c in ctx.chunks],
           "stored_params": {k: float(v) for k, v in zip(cm.PARAMS, ctx.theta)},
           "stored_fit_own_table_mean_regret_pct": float(100 * TMB.call_regret(ctx.lat, stored_choice).mean()),
           "check_eq3_equals_stored_predictor": bool(np.array_equal(predict("eq3", ctx.theta, F), TMB.predict_np(ctx.theta, F))),
           "critical_path_stored_fit": {"calibration_points": below_critical_path(ctx.theta, F),
                                        "captured_call_candidates": below_critical_path(ctx.theta, Fc)},
           "forms": {}, "scores": {}}
    # programs per SM at batch 1 (captured calls, default-table entries)
    cand = ctx.calls.cand.loc[ctx.calls.fc.index]
    nsm = ctx.hw["n_sm"]
    npg = np.ceil(cand["d"].to_numpy() / cand["BD"].to_numpy()) * cand["B"].to_numpy()
    b1 = cand["B"].to_numpy() == 1
    out["programs_per_sm_batch1"] = {"n_sm": int(nsm), "min": float((npg[b1] / nsm).min()), "max": float((npg[b1] / nsm).max()),
                                     "values": sorted({round(float(v), 3) for v in npg[b1] / nsm})}
    for form in FORMS:
        starts = [cm.X0.copy(), ctx.theta.copy()]
        if form in NO_W1:
            for s in starts:
                s[I_W1] = cm.X0[I_W1]
        th, cost = fit(form, F, ctx.y, starts)
        S = ctx.calls.matrix(predict(form, th, Fc))
        ch = S.argmin(1)
        out["forms"][form] = {
            "n_free_params": len(cm.PARAMS) - len(fixed_of(form)),
            "params": {cm.PARAMS[i]: float(th[i]) for i in range(len(cm.PARAMS)) if i not in fixed_of(form)},
            "objective_cost": cost, "objective_r2": objective_r2(form, th, F, ctx.y),
            "r2_log_calibration": cm.fit_quality(predict(form, th, F), ctx.y)["r2_log"],
            "r2_log_captured_calls": float(1 - np.nanvar(np.log(S[np.isfinite(ctx.lat)]) - np.log(ctx.lat[np.isfinite(ctx.lat)]))
                                           / np.var(np.log(ctx.lat[np.isfinite(ctx.lat)]))),
            "share_calls_choice_differs_from_stored_fit": float((ch != stored_choice).mean()),
            "own_table_mean_regret_pct": float(100 * TMB.call_regret(ctx.lat, ch).mean()),
            "loso": loso(form, ctx, F, th),
            "critical_path": None if form != "eq3" else below_critical_path(th, Fc)}
        out["scores"][form] = S
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    t0 = time.time()
    tags = [t for d in TAGS.values() for t in d.values()]
    per = {}
    with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, len(tags)))) as ex:
        for r in ex.map(tag_worker, [(t, str(args.in_dir)) for t in tags]):
            per[r["tag"]] = r
            print(f"{r['tag']}: {r['seconds']} s", flush=True)
    checks = C.Checks()
    res = {"experiment": "taco-r2 M4d latency term of Eq. (3)", "definitions": __doc__, "per_gpu": {}}
    for conv, tg in TAGS.items():
        res["per_gpu"][conv] = {}
        for gpu, tag in tg.items():
            r = per[tag]
            checks.add(f"{tag}: eq3 predictor equals stored", int(r["check_eq3_equals_stored_predictor"]), 1, "cost_model.predict", 0)
            checks.add(f"{tag}: eq3 refit regret equals stored fit", r["forms"]["eq3"]["own_table_mean_regret_pct"],
                       r["stored_fit_own_table_mean_regret_pct"], "stored fit (refit from X0 and the stored start)", 0.01)
            g = {k: r[k] for k in ("stored_params", "critical_path_stored_fit", "programs_per_sm_batch1")}
            g["forms"] = {}
            for nm in EVAL[conv][gpu]:
                T = C.triton_table(nm, args.in_dir)
                pos = {k: i for i, k in enumerate(r["keys"])}
                rows = np.array([pos[k] for k in T.keys()])
                picks = {f: r["scores"][f][rows].argmin(1) for f in FORMS}
                recs, paired, _ = C.evaluate(T, picks, nm, args.n_boot, [(f, "eq3") for f in FORMS if f != "eq3"])
                for f in FORMS:
                    g["forms"].setdefault(f, {k: v for k, v in r["forms"][f].items()})[f"eval_{nm}"] = C.compact(recs[f])
                g.setdefault("paired_minus_eq3", {})[nm] = paired["all_calls"]
            res["per_gpu"][conv][gpu] = g
    res["checks"] = checks.summary()
    res["runtime_s"] = round(time.time() - t0, 1)
    path = C.save("m4d", res, args.in_dir)
    print(f"wrote {path}; checks {res['checks']['n_match']}/{res['checks']['n']}")
    for m in res["checks"]["mismatches"]:
        print("  MISMATCH", m["check"], m["computed"], m["reference"])
    for conv in TAGS:
        for gpu, g in res["per_gpu"][conv].items():
            prim = EVAL[conv][gpu][0]
            cp = g["critical_path_stored_fit"]["captured_call_candidates"]
            print(f"{conv:16s} {gpu:9s} W1 {g['stored_params']['W1']:.2f}; candidates with W > W1 {100 * cp['share_W_above_W1']:.0f}%, "
                  f"tile time < A {100 * cp['share_tile_time_below_A']:.0f}% (min ratio {cp['min_ratio_tile_time_to_A']:.2f}); "
                  f"programs/SM at batch 1 {g['programs_per_sm_batch1']['values']}")
            for f in FORMS:
                x = g["forms"][f]
                e = x[f"eval_{prim}"]
                print(f"    {f:24s} k={x['n_free_params']} cost {x['objective_cost']:.4f} R2obj {x['objective_r2']:.3f} "
                      f"LOSO {x['loso']['mean_chunk_regret_pct']:.2f} | per call {e['mean_regret_pct']:.2f} "
                      f"LW {e['lw_share_of_oracle_pct']:.2f} | differs {100 * x['share_calls_choice_differs_from_stored_fit']:.0f}%")


if __name__ == "__main__":
    main()
