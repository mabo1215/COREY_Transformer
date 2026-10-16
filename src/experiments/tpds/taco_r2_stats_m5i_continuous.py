"""Continuous-outcome power analysis of "entropy adds nothing beyond the exact shape" (taco-r2, item M5i, offline).

CPU only and deterministic; reads stored result files and times nothing. Output: ``taco_r2_stats_m5i.json``.

Why. Where the exact-shape headroom exceeds one host synchronization (Mamba-2 SSD 370M, launch-inclusive, RTX 3090)
the exclusion of a data-dependent chunk rests on the information tests, which predict the per-call oracle *label*.
That label is tie- and noise-dominated there, so the label tests have little power. This script tests a continuous
outcome instead and states effect sizes in microseconds with intervals, a minimum detectable effect and an
absolute equivalence verdict.

Data sets (the same 2360 captured calls)
  mamba2_vgpu3090_launch   Mamba-2 SSD 370M, RTX 3090, launch-inclusive, one block per chunk (primary)
  mamba2_vgpu3090_graph    the same calls under CUDA-graph replay, warm L2, three rounds (round medians)
  mamba1_pro6000_device    Triton Mamba-1 370M, RTX PRO 6000, device time, three rounds (round medians)
Subsets: the covered calls (the call's exact shape (L, batch) occurs in the other prompt folds, as
``headroom_noise.TableStructure``) and the covered shapes of the primary data set whose held-out headroom exceeds one
host synchronization (21.4 us).

Tests
  Test A (top-2 contrast). Per call of a covered key, d = log T(c1) - log T(c2), where c1, c2 are the key's two
          chunks with the lowest mean latency over its calls. Linear mixed model
              d = alpha_key + beta * entropy + u_layer + v_cluster + e
          (key fixed effects, so beta is a within-key slope; crossed random intercepts for the scan layer and the
          prompt cluster), fitted by REML. beta is reported per bit of entropy and per within-key standard deviation
          of entropy, with a Wald interval and a prompt-cluster bootstrap interval (clusters resampled with
          replacement, every copy a new cluster level, the model refitted in every replicate).
          Microseconds: the entropy-induced change of the top-2 latency difference T(c1) - T(c2) over the within-key
          5th-95th percentile range of entropy, beta * Tbar_key * range_key, averaged over calls (Tbar_key = mean of
          the two chunks' latencies over the key's calls). Minimum detectable effect (two-sided alpha 0.05, power 0.8):
          2.80 x SE. Equivalence (TOST, alpha 0.05): the 90% interval of that change inside +/- half a host
          synchronization (10.7 us on the RTX 3090, 10.1 us on the RTX PRO 6000). Implied gain: an entropy-aware rule
          that serves c2 instead of c1 on the calls whose fitted contrast alpha_key + beta * z is positive gains
          Tbar_key * (exp(fitted d) - 1) there; reported at the point estimate and as the maximum over the 95%
          interval of beta (the gain is convex in beta), minus the gain at beta = 0.
  Test B (per-chunk log latency). log T_ic = a_{key,c} + b_c * entropy + call_i + layer x chunk + cluster x chunk + e.
          The call intercept is eliminated exactly by projecting each call's chunk vector on orthonormal (Helmert)
          within-call contrasts; random intercepts that are common to all chunks of a call (layer, cluster) cancel in
          that projection as well, so chunk-specific layer and cluster intercepts are used (variance components
          sigma_layer, sigma_cluster, REML). H0: b_c equal for all chunks (no chunk x entropy interaction): Wald
          chi-square and a likelihood-ratio test on ML fits. Effect sizes: every pairwise slope difference b_c - b_c'
          with Wald interval (and Bonferroni over the pairs), and the largest one in microseconds over the 5th-95th
          percentile entropy range (level sqrt(T_c T_c') per call). Also the variant without layer and cluster terms
          (ordinary least squares on the within-call contrasts).
  Context  plug-in H(oracle | exact key) and I(entropy; oracle | exact key) with a within-key permutation null
          (``e3b_information.cmi_test``, 1000 permutations), mean number of chunks within 2% of the oracle, the covered
          shapes above one synchronization, and the reviewer's exploratory Spearman check (seven largest-headroom
          covered shapes; log T(table chunk) / T(runner-up) against entropy).
  Graph check  the five shapes' headroom under CUDA-graph replay of the same calls on the same board: in sample,
          held out (covered), and cross-round (table and oracle chosen on the median of two rounds, scored on the third;
          ``r4_rounds_analysis``).
  Margin   the 0.5-point equivalence margin of the paired shape+entropy minus shape difference against the shape
          tree's own residual regret (the difference cannot fall below minus that residual), and the same paired
          difference in microseconds per call with intervals against an absolute margin of half a synchronization.

Usage:  PYTHONPATH=. python3 -m src.experiments.tpds.taco_r2_stats_m5i_continuous [--n-boot-a 1000] [--n-boot-b 200]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.linalg import cho_factor, cho_solve  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import chi2, norm, spearmanr  # noqa: E402

from src.experiments.tpds import e3_predictors as E3  # noqa: E402
from src.experiments.tpds import e3b_information as E3B  # noqa: E402
from src.experiments.tpds import headroom_noise as HN  # noqa: E402
from src.experiments.tpds import taco_r2_stats_common as C  # noqa: E402
from src.experiments.tpds.analysis import chunk_columns, latency_matrix, load_e2, round_median_view  # noqa: E402
from src.experiments.tpds.common import RESULTS_DIR  # noqa: E402

Z975, Z95, Z80 = norm.ppf(0.975), norm.ppf(0.95), norm.ppf(0.80)
MDE_FACTOR = Z975 + Z80          # 2.80: two-sided alpha 0.05, power 0.80
HALF_SYNC_US = {"RTX 3090": 21.4 / 2, "RTX PRO 6000": 20.2 / 2, "H800": 20.0 / 2}

DATA_SETS = {
    "mamba2_vgpu3090_launch": dict(csv="e2_mamba2_vgpu3090.csv", rmed=False, graph=None, gpu="RTX 3090",
                                   label="Mamba-2 SSD 370M, RTX 3090, launch-inclusive, one block per chunk"),
    "mamba2_vgpu3090_graph": dict(csv=None, rmed=True, graph="ssd370m_graph", gpu="RTX 3090",
                                  label="Mamba-2 SSD 370M, RTX 3090, CUDA-graph replay, three rounds (round medians)"),
    "mamba1_pro6000_device": dict(csv="e2_mamba1_pro6000_r4spin.csv", rmed=True, graph=None, gpu="RTX PRO 6000",
                                  label="Triton Mamba-1 370M, RTX PRO 6000, device time, three rounds (round medians)"),
}
PRIMARY = "mamba2_vgpu3090_launch"

# data sets of the paired shape + entropy minus shape difference (margin check)
MARGIN_SETS = {
    "mamba1_pro6000_device": ("e2_mamba1_pro6000_r4spin.csv", True, "mamba1", "pro6000_r4spin", "RTX PRO 6000"),
    "mamba1_vgpu3090_device": ("e2_mamba1_vgpu3090_r4spin.csv", True, "mamba1", "vgpu3090_r4spin", "RTX 3090"),
    "mamba1_h800_device": ("e2_mamba1_h800_r3spin.csv", False, "mamba1", "h800_r3spin", "H800"),
    "mamba2_vgpu3090_launch": ("e2_mamba2_vgpu3090.csv", False, "mamba2", "vgpu3090", "RTX 3090"),
    "mamba2_pro6000_launch": ("e2_mamba2_pro6000.csv", False, "mamba2", "pro6000", "RTX PRO 6000"),
}


# =========================================================================== data
class Data:
    """Per-call arrays of one data set: latency matrix, exact key, coverage, entropy, layer, prompt cluster."""

    def __init__(self, name: str, in_dir: Path):
        spec = DATA_SETS[name]
        self.name, self.gpu, self.label = name, spec["gpu"], spec["label"]
        if spec["graph"]:
            t = C.graph_table(spec["graph"], in_dir)
            df, self.chunks, lat = t.df, list(t.chunks), np.where(np.isfinite(t.lat), t.lat, np.nan)
            self.source = t.csv
        else:
            df = load_e2([in_dir / spec["csv"]])
            if "regimes" in df and (df["regimes"] == "e4_key").any():
                df = df[df["regimes"] != "e4_key"]
            if spec["rmed"]:
                df = round_median_view(df)
            df = df.reset_index(drop=True)
            self.chunks = chunk_columns(df)
            lat = latency_matrix(df, self.chunks)
            self.source = spec["csv"]
        self.df = df.reset_index(drop=True)
        self.lat = lat                                  # ms, NaN where a chunk did not run
        self.n = len(self.df)
        self.st = HN.TableStructure(self.df)
        self.key = self.st.exact_id
        self.keys = self.st.exact_keys
        self.covered = self.st.covered
        self.entropy = self.df["entropy_bits"].to_numpy(float)
        self.layer = pd.factorize(self.df["layer"])[0]
        self.cluster = E3B.prompt_clusters(self.df)[0]
        self.R = C.round_matrix(self.df, self.chunks) if spec["rmed"] else None

    def headroom(self) -> dict:
        lat = np.where(np.isfinite(self.lat), self.lat, np.inf)
        h = self.st.headroom(lat, lat)
        return {"in": self.st.per_shape(h["in"]), "cv": self.st.per_shape(h["cv"]), "calls": h}


def indicator(codes: np.ndarray, n_levels: int | None = None) -> sp.csr_matrix:
    codes = np.asarray(codes, int)
    q = int(codes.max()) + 1 if n_levels is None else n_levels
    return sp.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), q))


# =========================================================================== linear mixed model
class LMM:
    """y = X beta + sum_j Z_j u_j + e, u_j ~ N(0, theta_j sigma^2 I), e ~ N(0, sigma^2 I).

    Profiled (RE)ML on cross-products: with D = diag(theta) and A = I + D^(1/2) Z'Z D^(1/2) (Woodbury identity),
    log|H| = log|A| and X'H^-1 X = X'X - X'Z D^(1/2) A^-1 D^(1/2) Z'X, so one evaluation costs O(q^3) with
    q = number of random-effect levels. theta = 0 gives ordinary least squares.
    """

    def __init__(self, y: np.ndarray, X: np.ndarray, Zs: list[sp.csr_matrix]):
        self.y, self.X = np.asarray(y, float), np.asarray(X, float)
        self.n, self.p = self.X.shape
        self.qs = [z.shape[1] for z in Zs]
        Z = sp.hstack(Zs).tocsr() if Zs else sp.csr_matrix((self.n, 0))
        self.XtX, self.Xty, self.yty = self.X.T @ self.X, self.X.T @ self.y, float(self.y @ self.y)
        self.ZtZ = (Z.T @ Z).toarray()
        self.ZtX = np.asarray(Z.T @ self.X)
        self.Zty = np.asarray(Z.T @ self.y).ravel()

    def evaluate(self, theta, reml: bool = True) -> dict:
        theta = np.maximum(np.asarray(theta, float), 0.0)
        Q = sum(self.qs)
        if Q:
            d = np.sqrt(np.repeat(theta, self.qs))
            A = np.eye(Q) + d[:, None] * self.ZtZ * d[None, :]
            cf = cho_factor(A, lower=True)
            logdet_h = 2.0 * np.log(np.diag(cf[0])).sum()
            ZX, Zy = d[:, None] * self.ZtX, d * self.Zty
            SX, Sy = cho_solve(cf, ZX), cho_solve(cf, Zy)
            XHX, XHy, yHy = self.XtX - ZX.T @ SX, self.Xty - ZX.T @ Sy, self.yty - Zy @ Sy
        else:
            logdet_h, XHX, XHy, yHy = 0.0, self.XtX, self.Xty, self.yty
        cx = cho_factor(XHX, lower=True)
        beta = cho_solve(cx, XHy)
        rss = float(yHy - XHy @ beta)
        if reml:
            dof = self.n - self.p
            s2 = rss / dof
            ll = -0.5 * (dof * np.log(2 * np.pi * s2) + dof + logdet_h + 2.0 * np.log(np.diag(cx[0])).sum())
        else:
            s2 = rss / self.n
            ll = -0.5 * (self.n * np.log(2 * np.pi * s2) + self.n + logdet_h)
        return {"ll": float(ll), "beta": beta, "s2": float(s2), "XHX": XHX, "theta": theta}

    def fit(self, reml: bool = True, start=None) -> dict:
        k = len(self.qs)
        if k == 0:
            r = self.evaluate([], reml)
        else:
            best = self.evaluate(np.zeros(k), reml)
            starts = [np.full(k, 0.3), np.full(k, 1.0), np.full(k, 0.05)]
            if start is not None:
                starts.insert(0, np.sqrt(np.maximum(start, 0)))
            for s0 in starts:
                res = minimize(lambda s: -self.evaluate(s ** 2, reml)["ll"], s0, method="L-BFGS-B",
                               bounds=[(0.0, 100.0)] * k)
                cand = self.evaluate(res.x ** 2, reml)
                if cand["ll"] > best["ll"] + 1e-9:
                    best = cand
            r = best
        r["cov"] = r["s2"] * np.linalg.inv(r["XHX"])
        r["sigma2"] = r["s2"]
        r["var_components"] = [float(t * r["s2"]) for t in np.atleast_1d(r["theta"])]
        return r


# =========================================================================== test A
def top2_design(D: Data, sel: np.ndarray) -> dict:
    """Per call of ``sel``: d = log T(c1) - log T(c2) with the key's two fastest chunks (by mean latency)."""
    idx = np.flatnonzero(sel)
    keys = np.unique(D.key[idx])
    d = np.empty(len(idx))
    tbar = np.empty(len(idx))
    rng_e = np.empty(len(idx))
    zc = np.empty(len(idx))
    per_key = {}
    for k in keys:
        m = D.key[idx] == k
        rows = idx[m]
        L = D.lat[rows]
        ok = np.isfinite(L).all(0)
        mean = np.where(ok, np.nanmean(L, 0), np.inf)
        o = np.argsort(mean)
        c1, c2 = int(o[0]), int(o[1])
        d[m] = np.log(L[:, c1]) - np.log(L[:, c2])
        tb = float(0.5 * (L[:, c1].mean() + L[:, c2].mean()))
        tbar[m] = tb
        e = D.entropy[rows]
        rng_e[m] = float(np.percentile(e, 95) - np.percentile(e, 5))
        zc[m] = e - e.mean()
        per_key[f"L={D.keys[k][0]},B={D.keys[k][1]}"] = {
            "n_calls": int(m.sum()), "c1": D.chunks[c1], "c2": D.chunks[c2], "tbar_us": 1000 * tb,
            "mean_d": float(d[m].mean()), "entropy_sd_bits": float(e.std(ddof=1)) if m.sum() > 1 else None,
            "entropy_p5_p95_range_bits": float(rng_e[m][0])}
    return {"idx": idx, "d": d, "z": zc, "key": D.key[idx], "keys": keys, "layer": D.layer[idx],
            "cluster": D.cluster[idx], "tbar": tbar, "range": rng_e, "per_key": per_key}


def fit_top2(y, z, key, layer, cluster, random_effects=True, start=None) -> dict:
    _, kc = np.unique(key, return_inverse=True)
    X = np.column_stack([indicator(kc).toarray(), z])
    Zs = [indicator(pd.factorize(layer)[0]), indicator(pd.factorize(cluster)[0])] if random_effects else []
    return LMM(y, X, Zs).fit(True, start)


def implied_gain_us(alpha_key: np.ndarray, kc: np.ndarray, beta: float, z: np.ndarray, tbar: np.ndarray) -> float:
    dhat = alpha_key[kc] + beta * z
    return float(1000 * np.mean(np.where(dhat > 0, tbar * np.expm1(dhat), 0.0)))


def test_a(D: Data, sel: np.ndarray, n_boot: int, label: str) -> dict:
    g = top2_design(D, sel)
    y, z = g["d"], g["z"]
    _, kc = np.unique(g["key"], return_inverse=True)
    full = fit_top2(y, z, g["key"], g["layer"], g["cluster"])
    ols = fit_top2(y, z, g["key"], g["layer"], g["cluster"], random_effects=False)
    # check: theta = 0 equals ordinary least squares
    Xo = np.column_stack([indicator(kc).toarray(), z])
    b_ols = np.linalg.lstsq(Xo, y, rcond=None)[0]
    beta, se = float(full["beta"][-1]), float(np.sqrt(full["cov"][-1, -1]))
    sd_within = float(z.std(ddof=1))
    fac_us = float(1000 * np.mean(g["tbar"] * g["range"]))      # us of top-2 difference per unit beta over 5-95%
    fac_bit_us = float(1000 * np.mean(g["tbar"]))                # us per bit per unit beta
    half = HALF_SYNC_US[D.gpu]
    eff = beta * fac_us
    ci95 = [(beta - Z975 * se) * fac_us, (beta + Z975 * se) * fac_us]
    ci90 = [(beta - Z95 * se) * fac_us, (beta + Z95 * se) * fac_us]
    # prompt-cluster bootstrap with refit
    rng = C.make_rng("m5i_testA", D.name, label)
    clusters = np.unique(g["cluster"])
    rows_of = {c: np.flatnonzero(g["cluster"] == c) for c in clusters}
    boot_beta = []
    t0 = time.time()
    for _ in range(n_boot):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        rows = np.concatenate([rows_of[c] for c in pick])
        cl = np.concatenate([np.full(len(rows_of[c]), j) for j, c in enumerate(pick)])
        try:
            r = fit_top2(y[rows], z[rows], g["key"][rows], g["layer"][rows], cl, start=full["theta"])
            boot_beta.append(float(r["beta"][-1]))
        except np.linalg.LinAlgError:
            continue
    boot_beta = np.array(boot_beta)
    alpha_key = full["beta"][:-1]
    gain_hat = implied_gain_us(alpha_key, kc, beta, z, g["tbar"])
    gain0 = implied_gain_us(alpha_key, kc, 0.0, z, g["tbar"])
    lo, hi = beta - Z975 * se, beta + Z975 * se
    gain_ub = max(implied_gain_us(alpha_key, kc, b, z, g["tbar"]) for b in (lo, hi))
    bq = np.percentile(boot_beta, [2.5, 97.5]) if len(boot_beta) else [np.nan, np.nan]
    gain_ub_boot = max(implied_gain_us(alpha_key, kc, b, z, g["tbar"]) for b in bq)
    p_beta = float(2 * norm.sf(abs(beta / se)))
    per_shape = {}
    if len(g["keys"]) <= 8:                      # small subsets: the same model per shape (Wald only)
        for k in g["keys"]:
            m = g["key"] == k
            r = fit_top2(y[m], z[m], g["key"][m], g["layer"][m], g["cluster"][m])
            bk, sk = float(r["beta"][-1]), float(np.sqrt(r["cov"][-1, -1]))
            fk = float(1000 * np.mean(g["tbar"][m] * g["range"][m]))
            ci90k = [(bk - Z95 * sk) * fk, (bk + Z95 * sk) * fk]
            ak, kk = r["beta"][:-1], np.zeros(int(m.sum()), int)
            g0 = implied_gain_us(ak, kk, 0.0, z[m], g["tbar"][m])
            gk = implied_gain_us(ak, kk, bk, z[m], g["tbar"][m])
            gub = max(implied_gain_us(ak, kk, b, z[m], g["tbar"][m]) for b in (bk - Z975 * sk, bk + Z975 * sk))
            per_shape[f"L={D.keys[k][0]},B={D.keys[k][1]}"] = {
                "implied_gain_due_to_entropy_us": gk - g0, "implied_gain_upper_bound_wald95_us": gub - g0,
                "n_calls": int(m.sum()), "beta_per_bit": bk, "beta_ci95_wald": [bk - Z975 * sk, bk + Z975 * sk],
                "p_wald": float(2 * norm.sf(abs(bk / sk))), "effect_us_p5_p95": bk * fk,
                "effect_us_ci95": [(bk - Z975 * sk) * fk, (bk + Z975 * sk) * fk], "mde_us": float(MDE_FACTOR * sk * fk),
                "tost_half_sync_equivalent": bool(-half < ci90k[0] and ci90k[1] < half)}
    return {
        "n_calls": int(len(y)), "n_keys": int(len(g["keys"])), "n_layers": int(len(np.unique(g["layer"]))),
        "n_prompt_clusters": int(len(clusters)),
        "model": "d = alpha_key + beta * entropy + u_layer + v_cluster + e (REML)",
        "variance_components": {"layer": full["var_components"][0], "prompt_cluster": full["var_components"][1],
                                "residual": full["sigma2"]},
        "beta_per_bit": beta, "beta_se": se, "beta_ci95_wald": [lo, hi],
        "beta_ci95_cluster_bootstrap_refit": [float(bq[0]), float(bq[1])], "n_boot_ok": int(len(boot_beta)),
        "bootstrap_seconds": round(time.time() - t0, 1),
        "beta_p_value_wald": p_beta,
        "beta_per_within_key_sd": beta * sd_within, "within_key_entropy_sd_bits": sd_within,
        "ols_check": {"beta_ols_lstsq": float(b_ols[-1]), "beta_lmm_theta0": float(ols["beta"][-1]),
                      "match": bool(abs(b_ols[-1] - ols["beta"][-1]) < 1e-9), "beta_ols_se": float(np.sqrt(ols["cov"][-1, -1]))},
        "us_conversion": {"per_bit_us_per_unit_beta": fac_bit_us, "p5_p95_range_us_per_unit_beta": fac_us,
                          "mean_tbar_us": float(1000 * g["tbar"].mean()),
                          "mean_entropy_p5_p95_range_bits": float(g["range"].mean())},
        "effect_us_per_bit": beta * fac_bit_us,
        "effect_us_p5_p95": eff, "effect_us_ci95_wald": ci95, "effect_us_ci90_wald": ci90,
        "effect_us_ci95_bootstrap": [float(bq[0] * fac_us), float(bq[1] * fac_us)],
        "mde_us_p5_p95": float(MDE_FACTOR * se * fac_us), "mde_us_per_bit": float(MDE_FACTOR * se * fac_bit_us),
        "mde_beta_per_bit": float(MDE_FACTOR * se),
        "tost": {"margin_us": half, "margin_definition": "half of one host synchronization",
                 "ci90_us": ci90, "equivalent": bool(-half < ci90[0] and ci90[1] < half)},
        "implied_gain_us_per_call": {
            "definition": "entropy-aware switch between c1 and c2 within key on the fitted contrast; minus the gain "
                          "at beta = 0 gives the part due to entropy",
            "at_point_estimate": gain_hat, "at_beta_0": gain0, "due_to_entropy": gain_hat - gain0,
            "upper_bound_wald95": gain_ub, "upper_bound_bootstrap95": gain_ub_boot,
            "upper_bound_due_to_entropy_wald95": gain_ub - gain0},
        "per_key": g["per_key"],
        "per_shape_fits": per_shape,
    }


# =========================================================================== test B
def helmert(c: int) -> np.ndarray:
    """c x (c-1) orthonormal contrasts (columns orthogonal to the ones vector)."""
    H = np.zeros((c, c - 1))
    for j in range(1, c):
        H[:j, j - 1] = 1.0
        H[j, j - 1] = -j
        H[:, j - 1] /= np.sqrt(j * (j + 1))
    return H


def test_b(D: Data, sel: np.ndarray, label: str) -> dict:
    idx = np.flatnonzero(sel)
    ok = np.isfinite(D.lat[idx]).all(0)
    cols = np.flatnonzero(ok)
    chunks = [D.chunks[j] for j in cols]
    Y = np.log(D.lat[np.ix_(idx, cols)])
    c = len(cols)
    Hm = helmert(c)
    Ys = Y @ Hm                                          # (n, c-1) within-call contrasts
    n = len(idx)
    z_raw = D.entropy[idx]
    _, kc = np.unique(D.key[idx], return_inverse=True)
    K = int(kc.max()) + 1
    zc = z_raw - np.bincount(kc, weights=z_raw)[kc] / np.bincount(kc)[kc]
    j = np.tile(np.arange(c - 1), n)
    call = np.repeat(np.arange(n), c - 1)
    y = Ys.ravel()
    Xk = indicator(kc[call] * (c - 1) + j, K * (c - 1)).toarray()
    Xs = indicator(j, c - 1).toarray() * zc[call][:, None]
    X_full, X_red = np.column_stack([Xk, Xs]), Xk
    lay = pd.factorize(D.layer[idx])[0]
    clu = pd.factorize(D.cluster[idx])[0]
    nl, nc = lay.max() + 1, clu.max() + 1
    Zs = [indicator(lay[call] * (c - 1) + j, nl * (c - 1)), indicator(clu[call] * (c - 1) + j, nc * (c - 1))]
    res = {"n_calls": int(n), "chunks": chunks, "n_keys": int(K),
           "model": "log T_ic = a_key,c + b_c * entropy + call + layer x chunk + cluster x chunk + e; call eliminated "
                    "by within-call orthonormal contrasts; REML"}
    tbar = D.lat[np.ix_(idx, cols)]
    rng_k = {k: np.percentile(z_raw[kc == k], 95) - np.percentile(z_raw[kc == k], 5) for k in range(K)}
    rng_call = np.array([rng_k[k] for k in kc])
    for variant, zs in (("layer_and_cluster_by_chunk", Zs), ("no_layer_cluster_terms", [])):
        m_full = LMM(y, X_full, zs)
        fit = m_full.fit(True)
        g = fit["beta"][-(c - 1):]
        V = fit["cov"][-(c - 1):, -(c - 1):]
        wald = float(g @ np.linalg.solve(V, g))
        p_wald = float(chi2.sf(wald, c - 1))
        ml_full = m_full.fit(False, start=fit["theta"] if zs else None)
        ml_red = LMM(y, X_red, zs).fit(False, start=fit["theta"] if zs else None)
        lr = 2.0 * (ml_full["ll"] - ml_red["ll"])
        b = Hm @ g                                       # slopes relative to their mean
        Vb = Hm @ V @ Hm.T
        pairs = {}
        n_pairs = c * (c - 1) // 2
        zb = norm.ppf(1 - 0.025 / n_pairs)
        for a in range(c):
            for bb in range(a + 1, c):
                e = np.zeros(c)
                e[a], e[bb] = 1.0, -1.0
                diff, sdd = float(e @ b), float(np.sqrt(e @ Vb @ e))
                lvl = np.sqrt(tbar[:, a] * tbar[:, bb])
                fac = float(1000 * np.mean(lvl * rng_call))
                pairs[f"{chunks[a]}-{chunks[bb]}"] = {
                    "diff_per_bit": diff, "se": sdd, "ci95": [diff - Z975 * sdd, diff + Z975 * sdd],
                    "ci95_bonferroni": [diff - zb * sdd, diff + zb * sdd], "p_wald": float(2 * norm.sf(abs(diff / sdd))),
                    "us_p5_p95_per_unit": fac, "effect_us_p5_p95": diff * fac,
                    "effect_us_ci95": [(diff - Z975 * sdd) * fac, (diff + Z975 * sdd) * fac],
                    "effect_us_ci95_bonferroni": [(diff - zb * sdd) * fac, (diff + zb * sdd) * fac],
                    "mde_us": float(MDE_FACTOR * sdd * fac)}
        largest = max(pairs, key=lambda k: abs(pairs[k]["effect_us_p5_p95"]))
        largest_abs_bound = max(max(abs(x) for x in p["effect_us_ci95_bonferroni"]) for p in pairs.values())
        res[variant] = {
            "variance_components": ({"layer_x_chunk": fit["var_components"][0], "cluster_x_chunk": fit["var_components"][1],
                                     "residual": fit["sigma2"]} if zs else {"residual": fit["sigma2"]}),
            "slopes_minus_mean_per_bit": dict(zip(map(str, chunks), map(float, b))),
            "interaction_wald_chi2": wald, "df": c - 1, "interaction_p_wald": p_wald,
            "interaction_lr_chi2_ml": float(lr), "interaction_p_lr": float(chi2.sf(max(lr, 0.0), c - 1)),
            "pairwise": pairs, "largest_pair_by_us": largest, "largest_pair": pairs[largest],
            "max_abs_us_bonferroni_bound_over_pairs": float(largest_abs_bound)}
    return res


# =========================================================================== context
def context(D: Data, in_dir: Path) -> dict:
    hr = D.headroom()
    cov = D.st.shape_covered
    sync_us = C.SYNC_US[D.gpu]
    order = [k for k in np.argsort(-hr["cv"]) if cov[k]]
    above = [k for k in order if 1000 * hr["cv"][k] > sync_us]
    lat_inf = np.where(np.isfinite(D.lat), D.lat, np.inf)
    within2 = (lat_inf <= 1.02 * lat_inf.min(1, keepdims=True)).sum(1)
    # plug-in conditional entropy and CMI of entropy given the exact key (e3b_information.cmi_test)
    t = E3B.Table(D.df, D.name, "", D.gpu, D.source)
    cmi = E3B.cmi_test(E3B._disc_global(t.df["entropy_bits"].to_numpy(float)), t.df["oracle_chunk"].to_numpy(),
                       t.key, 1000, C.make_rng("m5i_cmi", D.name))
    stored = C.load_json(in_dir / "e3b_information_mamba2_vgpu3090.json") if D.name == PRIMARY else None
    st_cmi = stored["conditional_mi"]["by_condition"]["exact_key"]["entropy_g8"] if stored else None
    spear = {}
    for k in order[:7]:
        m = D.key == k
        L = lat_inf[m]
        o = np.argsort(L.mean(0))
        yv = np.log(L[:, o[0]] / L[:, o[1]])
        r = spearmanr(D.entropy[m], yv)
        spear[f"L={D.keys[k][0]},B={D.keys[k][1]}"] = {
            "n_calls": int(m.sum()), "table_chunk": D.chunks[o[0]], "runner_up": D.chunks[o[1]],
            "headroom_held_out_us": float(1000 * hr["cv"][k]), "rho": float(r.statistic), "p": float(r.pvalue)}
    rhos = [v["rho"] for v in spear.values()]
    ps = [v["p"] for v in spear.values()]
    return {
        "n_calls": D.n, "n_covered_calls": int(D.covered.sum()), "n_covered_shapes": int(cov.sum()),
        "held_out_covered_mean_us": float(1000 * hr["calls"]["cv"][D.covered].mean()),
        "host_sync_us": sync_us,
        "covered_shapes_above_sync": [{"L": D.keys[k][0], "batch": D.keys[k][1], "n_calls": int(D.st.shape_counts[k]),
                                       "held_out_us": float(1000 * hr["cv"][k]), "in_sample_us": float(1000 * hr["in"][k])}
                                      for k in above],
        "mean_chunks_within_2pct": float(within2.mean()),
        "H_oracle_given_exact_key_bits": cmi["H_oracle_given_cond_bits"],
        "cmi_entropy_given_exact_key": {k: cmi[k] for k in ("cmi_bits", "null_mean_bits", "excess_bits", "p_value")},
        "cmi_stored": ({k: st_cmi[k] for k in ("cmi_bits", "excess_bits", "p_value", "H_oracle_given_cond_bits")}
                       if st_cmi else None),
        "max_possible_H_bits": float(np.log2(len(D.chunks))),
        "spearman_seven_largest_headroom_shapes": spear,
        "spearman_rho_range": [float(min(rhos)), float(max(rhos))] if rhos else None,
        "spearman_min_p": float(min(ps)) if ps else None,
        "above_sync_key_ids": [int(k) for k in above],
    }


def graph_headroom(G: Data, shape_keys: list[tuple[int, int]]) -> dict:
    """Headroom of the given shapes under CUDA-graph replay: in sample, held out (covered), cross-round."""
    lat = np.where(np.isfinite(G.lat), G.lat, np.inf)
    h = G.st.headroom(lat, lat)
    ps_in, ps_cv = G.st.per_shape(h["in"]), G.st.per_shape(h["cv"])
    R = np.where(np.isfinite(G.R), G.R, np.inf)
    acc = {"in": np.zeros(G.n), "cv": np.zeros(G.n)}
    for r in range(R.shape[0]):
        s = np.median(np.delete(R, r, 0), 0)
        hh = G.st.headroom(s, R[r])
        for k in acc:
            acc[k] += hh[k] / R.shape[0]
    cr_in, cr_cv = G.st.per_shape(acc["in"]), G.st.per_shape(acc["cv"])
    sync = C.SYNC_US[G.gpu]
    out = {"n_rounds": int(R.shape[0]), "host_sync_us": sync, "shapes": {}}
    pos = {k: i for i, k in enumerate(G.keys)}
    for s in shape_keys:
        k = pos[s]
        out["shapes"][f"L={s[0]},B={s[1]}"] = {
            "n_calls": int(G.st.shape_counts[k]), "covered": bool(G.st.shape_covered[k]),
            "in_sample_us": float(1000 * ps_in[k]), "held_out_us": float(1000 * ps_cv[k]),
            "cross_round_in_sample_us": float(1000 * cr_in[k]), "cross_round_held_out_us": float(1000 * cr_cv[k]),
            "mean_oracle_latency_us": float(1000 * lat[G.key == k].min(1).mean())}
    v = out["shapes"].values()
    out["max_held_out_us"] = max(x["held_out_us"] for x in v)
    out["max_cross_round_held_out_us"] = max(x["cross_round_held_out_us"] for x in v)
    out["n_above_sync_held_out"] = sum(x["held_out_us"] > sync for x in v)
    out["n_above_sync_cross_round"] = sum(x["cross_round_held_out_us"] > sync for x in v)
    cov = G.st.shape_covered
    out["all_covered_shapes"] = {
        "n": int(cov.sum()), "max_held_out_us": float(1000 * ps_cv[cov].max()),
        "max_cross_round_held_out_us": float(1000 * cr_cv[cov].max()),
        "n_above_sync_held_out": int((1000 * ps_cv[cov] > sync).sum()),
        "n_above_sync_cross_round": int((1000 * cr_cv[cov] > sync).sum()),
        "covered_mean_held_out_us": float(1000 * h["cv"][G.covered].mean()),
        "covered_mean_cross_round_us": float(1000 * acc["cv"][G.covered].mean())}
    return out


# =========================================================================== margin check
def margin_check(in_dir: Path, n_boot: int) -> dict:
    from src.experiments.tpds import taco_report_numbers as TRN
    out = {}
    for name, (csv, rmed, model, host, gpu) in MARGIN_SETS.items():
        t = TRN.load_table(in_dir, csv, rmed, name, model, host)
        cols_s, depth_s = E3.FEATURE_SETS["shape"]
        cols_se, depth_se = E3.FEATURE_SETS["shape+entropy"]
        ps, pse = E3B.cv_tree(t, cols_s, depth_s), E3B.cv_tree(t, cols_se, depth_se)
        lat = np.where(np.isfinite(t.lat), t.lat, np.inf)
        A = {**C.policy_arrays(lat, t.orc, ps, "s"), **C.policy_arrays(lat, t.orc, pse, "se")}
        cb = C.ClusterBoot(t.clusters, n_boot, C.make_rng("m5i_margin", name))
        d_pp = C.summarize(cb.evaluate(A, lambda T, n: 100 * (T["se_reg"] - T["s_reg"]) / n))
        d_us = C.summarize(cb.evaluate(A, lambda T, n: 1000 * (T["se_exc"] - T["s_exc"]) / n))
        reps_us = cb.evaluate(A, lambda T, n: 1000 * (T["se_exc"] - T["s_exc"]) / n)["reps"]
        reps_pp = cb.evaluate(A, lambda T, n: 100 * (T["se_reg"] - T["s_reg"]) / n)["reps"]
        shape_reg = float(100 * A["s_reg"].mean())
        shape_us = float(1000 * A["s_exc"].mean())
        half = HALF_SYNC_US[gpu]
        ci90_us, ci90_pp = C.pct_ci(reps_us, 0.90), C.pct_ci(reps_pp, 0.90)
        rel = 0.5 * shape_reg
        out[name] = {
            "gpu": gpu, "source": csv, "round_median_view": rmed, "n_calls": int(t.N),
            "shape_tree_mean_regret_pct": shape_reg, "shape_tree_us_per_call": shape_us,
            "shape_plus_entropy_mean_regret_pct": float(100 * A["se_reg"].mean()),
            "paired_diff_pp": d_pp, "paired_diff_us_per_call": d_us,
            "lowest_possible_diff_pp": -shape_reg,
            "margin_0p5pp_can_fail_in_beneficial_direction": bool(-shape_reg < -0.5),
            "tost_0p5pp": bool(-0.5 < ci90_pp[0] and ci90_pp[1] < 0.5),
            "absolute_margin_us": half, "ci90_us": ci90_us,
            "tost_absolute_half_sync": bool(-half < ci90_us[0] and ci90_us[1] < half),
            "relative_margin_pp_half_of_shape_residual": rel, "ci90_pp": ci90_pp,
            "tost_relative_half_residual": bool(-rel < ci90_pp[0] and ci90_pp[1] < rel)}
    return out


# =========================================================================== main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--n-boot-a", type=int, default=1000, help="cluster-bootstrap refits of test A")
    p.add_argument("--n-boot-margin", type=int, default=2000)
    args = p.parse_args(argv)
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)  # noqa: E731
    res = {"experiment": "taco-r2 M5i offline: continuous-outcome entropy test", **C.meta(__file__, args),
           "definitions": __doc__, "data_sets": {}}
    data = {n: Data(n, args.in_dir) for n in DATA_SETS}
    prim = data[PRIMARY]
    ctx = context(prim, args.in_dir)
    res["context_primary"] = ctx
    above_keys = [(s["L"], s["batch"]) for s in ctx["covered_shapes_above_sync"]]
    log("context done; above-sync shapes", above_keys)
    for name, D in data.items():
        rec = {"label": D.label, "source": D.source, "gpu": D.gpu, "subsets": {}}
        subsets = {"covered_all": D.covered}
        if name.startswith("mamba2"):
            pos = {k: i for i, k in enumerate(D.keys)}
            subsets["five_above_sync_shapes_of_primary"] = np.isin(D.key, [pos[s] for s in above_keys])
        for sub, sel in subsets.items():
            nb = args.n_boot_a if (name == PRIMARY or sub == "covered_all") else args.n_boot_a
            a = test_a(D, sel, nb, sub)
            log(name, sub, "test A beta %.4g se %.3g eff %.2f us" % (a["beta_per_bit"], a["beta_se"], a["effect_us_p5_p95"]))
            b = test_b(D, sel, sub)
            log(name, sub, "test B p_wald %.3g" % b["layer_and_cluster_by_chunk"]["interaction_p_wald"])
            rec["subsets"][sub] = {"test_a_top2": a, "test_b_per_chunk": b}
        res["data_sets"][name] = rec
    res["graph_check_five_shapes"] = graph_headroom(data["mamba2_vgpu3090_graph"], above_keys)
    log("graph check done")
    res["margin_check"] = margin_check(args.in_dir, args.n_boot_margin)
    log("margin check done")
    res["reproduction"] = {
        "H_oracle_given_exact_key_bits": {"computed": ctx["H_oracle_given_exact_key_bits"], "review": 1.97},
        "cmi_excess_bits": {"computed": ctx["cmi_entropy_given_exact_key"]["excess_bits"], "review": -0.0003,
                            "stored": ctx["cmi_stored"]["excess_bits"] if ctx["cmi_stored"] else None},
        "cmi_p": {"computed": ctx["cmi_entropy_given_exact_key"]["p_value"], "review": 0.50},
        "n_shapes_above_sync": {"computed": len(above_keys), "paper": 5},
        "max_headroom_us": {"computed": ctx["covered_shapes_above_sync"][0]["held_out_us"] if above_keys else None,
                            "paper": 43.0},
        "spearman_rho_range": {"computed": ctx["spearman_rho_range"], "review": [-0.06, 0.31]},
        "spearman_min_p": {"computed": ctx["spearman_min_p"], "review": ">= 0.13"},
        "mean_chunks_within_2pct": {"computed": ctx["mean_chunks_within_2pct"],
                                    "review": 3.2, "note": "the review's 3.2 is the RTX PRO 6000 launch-inclusive Mamba-2 "
                                    "value of the main text; the RTX 3090 table gives the computed value"},
    }
    d2 = load_e2([args.in_dir / "e2_mamba2_pro6000.csv"])
    l2 = latency_matrix(d2, chunk_columns(d2))
    res["reproduction"]["mean_chunks_within_2pct_mamba2_pro6000_launch"] = {
        "computed": float((l2 <= 1.02 * l2.min(1, keepdims=True)).sum(1).mean()), "paper": 3.2}
    res["runtime_s"] = round(time.time() - t0, 1)
    path = C.write_result("m5i", res, args.in_dir)
    log("wrote", path)
    print_summary(res)


def print_summary(res: dict) -> None:
    print("\n=== M5i summary ===")
    c = res["context_primary"]
    print(f"H(oracle|key) {c['H_oracle_given_exact_key_bits']:.3f} bits; CMI excess {c['cmi_entropy_given_exact_key']['excess_bits']:.4f} "
          f"p={c['cmi_entropy_given_exact_key']['p_value']:.3f}; within 2%: {c['mean_chunks_within_2pct']:.2f}")
    print("above-sync shapes:", [(s["L"], s["batch"], round(s["held_out_us"], 1)) for s in c["covered_shapes_above_sync"]])
    print("Spearman rho range", c["spearman_rho_range"], "min p", round(c["spearman_min_p"], 3))
    for name, rec in res["data_sets"].items():
        for sub, r in rec["subsets"].items():
            a, b = r["test_a_top2"], r["test_b_per_chunk"]["layer_and_cluster_by_chunk"]
            print(f"{name} / {sub}: n={a['n_calls']} keys={a['n_keys']}")
            print(f"  A: beta {a['beta_per_bit']:.4f}/bit [{a['beta_ci95_wald'][0]:.4f}, {a['beta_ci95_wald'][1]:.4f}] "
                  f"boot [{a['beta_ci95_cluster_bootstrap_refit'][0]:.4f}, {a['beta_ci95_cluster_bootstrap_refit'][1]:.4f}] p={a['beta_p_value_wald']:.3f}; "
                  f"effect {a['effect_us_p5_p95']:.2f} us [{a['effect_us_ci95_wald'][0]:.2f}, {a['effect_us_ci95_wald'][1]:.2f}], "
                  f"MDE {a['mde_us_p5_p95']:.2f} us, TOST +/-{a['tost']['margin_us']:.1f}: {a['tost']['equivalent']}; "
                  f"gain {a['implied_gain_us_per_call']['due_to_entropy']:.3f} us (ub {a['implied_gain_us_per_call']['upper_bound_due_to_entropy_wald95']:.3f})")
            lp = b["largest_pair"]
            print(f"  B: interaction chi2 {b['interaction_wald_chi2']:.2f} df {b['df']} p={b['interaction_p_wald']:.3f}, "
                  f"LR p={b['interaction_p_lr']:.3f}; largest pair {b['largest_pair_by_us']}: {lp['effect_us_p5_p95']:.2f} us "
                  f"[{lp['effect_us_ci95'][0]:.2f}, {lp['effect_us_ci95'][1]:.2f}], MDE {lp['mde_us']:.2f}; "
                  f"max |bound| (Bonf.) {b['max_abs_us_bonferroni_bound_over_pairs']:.2f} us")
    g = res["graph_check_five_shapes"]
    for s, v in g["shapes"].items():
        print(f"graph {s}: in {v['in_sample_us']:.2f}, held-out {v['held_out_us']:.2f}, cross-round {v['cross_round_held_out_us']:.2f} us")
    print("graph all covered:", g["all_covered_shapes"])
    for n, m in res["margin_check"].items():
        print(f"margin {n}: shape {m['shape_tree_mean_regret_pct']:.3f}% ; diff {m['paired_diff_pp']['point']:.3f} pp "
              f"{[round(x, 3) for x in m['paired_diff_pp']['ci95_percentile']]}; us {m['paired_diff_us_per_call']['point']:.3f} "
              f"{[round(x, 3) for x in m['paired_diff_us_per_call']['ci95_percentile']]}; 0.5pp can fail: "
              f"{m['margin_0p5pp_can_fail_in_beneficial_direction']}; TOST abs: {m['tost_absolute_half_sync']}; "
              f"TOST rel: {m['tost_relative_half_residual']}")


if __name__ == "__main__":
    main()
