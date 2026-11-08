"""jpdc-r1 item M2: parameter correlations and identifiability of the eight-parameter Triton-scan cost model.

CPU only; times nothing. For every GPU it takes the stored calibration data and fit (and the refit on the
full-protocol timings alone) and reports

* the local sensitivity matrix of the log-latency residuals with respect to the log of each free parameter,
  its singular values, condition number and numerical rank;
* the asymptotic parameter correlation matrix and relative standard errors from the Gauss-Newton covariance
  sigma^2 (J^T J)^+ (reported only as a local indicator: the objective is piecewise smooth);
* the parameter correlations over a 12-key cluster bootstrap of the fit (keys resampled with replacement,
  refit from the standard starts), with the share of resamples that end on a bound.

Usage: PYTHONPATH=. python3 -m src.experiments.tpds.jpdc_m2_identifiability [--n-boot 200]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.experiments.tpds import cost_model as cm
from src.experiments.tpds import cost_model_baselines_r4 as r4
from src.experiments.tpds import taco_model_baselines as TMB
from src.experiments.tpds.common import RESULTS_DIR

FREE = [i for i in range(len(cm.PARAMS)) if i not in cm.DEFAULT_FIXED]
NAMES = [cm.PARAMS[i] for i in FREE]


def log_sensitivity(theta, F, y):
    """d residual / d log(theta_i) for the free parameters, by central differences (relative step 1e-4)."""
    cols = []
    for i in FREE:
        h = 1e-4
        up, dn = theta.copy(), theta.copy()
        base = theta[i] if theta[i] > 0 else 1e-12
        up[i], dn[i] = base * (1 + h), base * (1 - h)
        cols.append((np.log(TMB.predict_np(up, F)) - np.log(TMB.predict_np(dn, F))) / (2 * h))
    return np.column_stack(cols)


def local_report(theta, F, y):
    J = log_sensitivity(theta, F, y)
    n = len(y)
    resid = np.log(TMB.predict_np(theta, F)) - np.log(y)
    active = np.array([np.linalg.norm(J[:, k]) > 1e-9 for k in range(J.shape[1])])
    Ja = J[:, active]
    sv = np.linalg.svd(Ja, compute_uv=False)
    rank = int((sv > sv[0] * 1e-8).sum())
    s2 = float((resid ** 2).sum() / max(n - Ja.shape[1], 1))
    cov = s2 * np.linalg.pinv(Ja.T @ Ja)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    names = [nm for nm, a in zip(NAMES, active) if a]
    pairs = sorted(((abs(corr[a, b]), names[a], names[b], float(corr[a, b]))
                    for a in range(len(names)) for b in range(a + 1, len(names))), reverse=True)
    return {"n": n, "sigma_log": float(np.sqrt(s2)), "insensitive_parameters": [nm for nm, a in zip(NAMES, active) if not a],
            "singular_values": [float(x) for x in sv], "condition_number": float(sv[0] / sv[-1]), "numerical_rank": rank,
            "n_active_parameters": int(active.sum()),
            "rel_se_log_param": {nm: float(s) for nm, s in zip(names, sd)},
            "correlation": {"names": names, "matrix": corr.tolist()},
            "top_correlated_pairs": [{"a": a, "b": b, "corr": c} for _, a, b, c in pairs[:6]]}


def bootstrap(ctx, F, n_boot, seed=0):
    keys = ctx.points["key"].to_numpy()
    uniq = np.array(sorted(set(keys.tolist())))
    by = {k: np.where(keys == k)[0] for k in uniq}
    rng = np.random.default_rng(seed)
    thetas, at_bound = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by[k] for k in pick])
        th, _ = TMB.fit_fast(TMB.sub(F, idx), ctx.y[idx])
        thetas.append(th[FREE])
        at_bound.append([bool(th[i] <= cm.LO[i] * (1 + 1e-6) + 1e-12 or th[i] >= cm.HI[i] * (1 - 1e-6)) for i in FREE])
    T = np.array(thetas)
    L = np.log(np.clip(T, 1e-12, None))
    keep = L.std(0) > 1e-9
    C = np.corrcoef(L[:, keep].T)
    names = [nm for nm, k in zip(NAMES, keep) if k]
    pairs = sorted(((abs(C[a, b]), names[a], names[b], float(C[a, b]))
                    for a in range(len(names)) for b in range(a + 1, len(names))), reverse=True)
    return {"n_boot": n_boot, "n_keys": int(len(uniq)),
            "share_at_bound": {nm: float(np.mean(np.array(at_bound)[:, k])) for k, nm in enumerate(NAMES)},
            "rel_spread_iqr_over_median": {nm: float((np.percentile(T[:, k], 75) - np.percentile(T[:, k], 25))
                                                     / max(np.median(T[:, k]), 1e-12)) for k, nm in enumerate(NAMES)},
            "log_param_correlation": {"names": names, "matrix": C.tolist()},
            "top_correlated_pairs": [{"a": a, "b": b, "corr": c} for _, a, b, c in pairs[:6]]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args(argv)
    out = {}
    for gpu, tag in TMB.TAGS["device_time"].items():
        ctx = r4.Ctx(tag, RESULTS_DIR)
        F = TMB.farr(ctx.f)
        src = ctx.points["src"].to_numpy()
        m = np.isin(src, ["e4_static", "e1"])
        th_full, _ = TMB.fit_fast(TMB.sub(F, m), ctx.y[m])
        rec = {"tag": tag, "params_stored": {k: float(v) for k, v in zip(cm.PARAMS, ctx.theta)},
               "stored_fit": local_report(ctx.theta, F, ctx.y),
               "full_protocol_fit": local_report(th_full, TMB.sub(F, m), ctx.y[m]),
               "bootstrap_stored_data": bootstrap(ctx, F, args.n_boot)}
        out[gpu] = rec
        s = rec["stored_fit"]
        print(gpu, "cond", f"{s['condition_number']:.3g}", "rank", s["numerical_rank"], "/", s["n_active_parameters"],
              "top pairs", [(p["a"], p["b"], round(p["corr"], 3)) for p in s["top_correlated_pairs"][:3]])
        print("   bootstrap", [(p["a"], p["b"], round(p["corr"], 3)) for p in rec["bootstrap_stored_data"]["top_correlated_pairs"][:3]],
              "at bound", {k: round(v, 2) for k, v in rec["bootstrap_stored_data"]["share_at_bound"].items() if v > 0})
    (RESULTS_DIR / "jpdc_m2_identifiability.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
