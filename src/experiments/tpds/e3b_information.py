"""E3b - uncertainty and conditional-information tests of "activation statistics add nothing" (M7d, M2c).

CPU-only analysis of existing result files; nothing is timed. Inputs are the per-call E2 tables
``e2_<model>_<host>.csv`` and the reference CUDA scan table ``e8_refscan_<host>.csv`` (calls only,
the tuning-key rows are dropped as in E8). For every table ("tag", e.g. ``mamba1_pro6000``,
``mamba1_vgpu3090_1p4b_r3``, ``refscan_vgpu3090``) the script writes
``e3b_information_<tag>.json``; at the end it writes ``e3b_information_summary.json`` and
``e3b_information_summary.csv`` with one compact row per tag. A tag whose output already exists is
skipped unless ``--force`` is given, so an interrupted run resumes where it stopped.

Definitions
-----------
For call i and chunk c, lat[i, c] is the mean of the call's 50-repeat timing block at chunk c. The
per-call oracle is argmin_c lat[i, c], and the relative regret of a choice c is
lat[i, c] / min_c lat[i, c] - 1 (reported in percent). "CI tie" means the chosen chunk lies in the
call's tied set recorded at capture time (95% confidence intervals overlapping the oracle's).
The exact shape key is (L, batch, width, L mod 16 == 0); width is constant within a table, so the
key reduces to (L, batch). The length bucket is the power-of-two bucket L_bucket of the E2 table.
A chunk without a timing has infinite latency in the table (24 Mamba-2 2.7B calls at chunk 32); for
fitting, relative latencies are capped at 1e6 so such a chunk is never chosen, and regrets are always
computed from the uncapped values.

Resampling units
----------------
Every prompt belongs to one capture-time fold, but batched calls mix several prompts, so prompts are
linked through shared batched calls. The prompt clusters are the connected components of the
bipartite call-prompt graph (42 clusters for the 370M tables, 192 single-prompt clusters for the
batch-1-only tables); they nest inside the folds. The primary bootstrap resamples prompt clusters
with replacement (all calls of a cluster move together). A secondary bootstrap resamples request
groups (``group_id``: one prompt, or one batch of prompts, with all its captured layers); it ignores
that the same prompt appears in several requests and therefore gives narrower intervals.

Predictors (held out with the capture-time prompt folds, grouped 5-fold cross-validation as in E3)
------------------------------------------------------------------------------------------------
  global_static, entropy_rule, L_only, shape, entropy_only, moments, shape+entropy, all
      exactly the E3 predictors as implemented: leaf-cost trees of maximum depth 4 (depth 3 for
      L_only) with at least 5 calls per leaf and regret-optimal leaf labels. The manuscript text says
      "depth 3", so true depth-3 trees are reported as well (tree_d3:<set>).
  batch_only            per-batch table, regret-optimal label per batch (Table III row).
  tree_d3:<set>, tree_d6:<set>, tree_unbounded:<set>
                        the E3 leaf-cost tree with maximum depth 3, 6 and unbounded depth
                        (min 5 calls per leaf), for <set> in shape, shape+entropy, all.
  gbt_cls:<set>         histogram gradient-boosted classifier of the oracle chunk (200 rounds,
                        learning rate 0.05, at most 15 leaves per tree, at least 10 calls per leaf).
  gbt_reg:<set>         one gradient-boosted regressor per chunk of log(lat / oracle lat), same
                        hyperparameters; the chunk with the lowest prediction is chosen (cost-aware).
  exact_key_table       table keyed on the exact shape key, regret-optimal entry per key from the
                        training folds; unseen keys fall back to (length bucket, batch, aligned) and
                        then to the best single chunk (as the held-out headroom of the manuscript).
  exact_key+<stat>      exact_key_table whose entry may be split once, within each key, on one
                        statistic (threshold and the two chunks chosen on the training folds, at
                        least 5 training calls per side, split only if it lowers the training
                        regret); exact_key+any_moment chooses the best split over all six
                        activation statistics.
  exact_key_floor_in_sample
                        the same table chosen on all calls (the in-sample exact-shape floor).
  best_static_per_batch_in_sample
                        the in-sample best chunk per batch size (Table II "best static").
  cost_model            Eq. (5), per-call choice as stored by the cost-model run (Mamba-1 on the
                        RTX PRO 6000, RTX 3090 and H800 tables only); checked against the argmin of
                        the stored per-shape prediction curves.
  stock_dispatch        reference scan only: the stock length-based dispatch, timed separately.
The predictors use the paper's shape features (L, batch, width, layer); the activation statistics
are entropy, variance, kurtosis, sparsity, mean |x| and delta entropy.

Tests
-----
T0  Reproduction: every E3 predictor is recomputed and compared with the stored E3 (or E8) result;
    the batch-only row, the in-sample exact-shape floor, the held-out exact-shape headroom of
    ``analysis.cv_exact_headroom`` and the bucket-conditional mutual information of
    ``analysis.conditional_mutual_information`` (0.046 and 1.23 bits on Mamba-1, RTX PRO 6000) are
    recomputed with the paper's own functions.
T1  Bootstrap (M7d-i): mean regret, 95th-percentile regret (weighted inverted-CDF percentile), CI-tie
    share and the shares within 2% and 5% of the oracle, each with a 95% percentile interval over
    ``--n-boot`` cluster-bootstrap replicates of the fixed held-out predictions. Paired differences
    (e.g. shape+entropy minus shape) use the same replicates; the equivalence verdict at margin
    delta (0.5 and 1.0 percentage points) is "equivalent" if the 95% interval lies inside
    (-delta, delta), "not_equivalent" if it lies entirely outside [-delta, delta] and "inconclusive"
    otherwise; the TOST verdict at alpha = 0.05 (90% interval inside the margin) is also given.
    Training-set variability is reported separately by repeating the cross-validation with
    ``--n-cv-repeats`` random assignments of prompt clusters to five folds (trees only).
T2  Exact-key stratified permutation test (M7d-ii): statistic T = held-out mean regret of the
    exact_key_table minus that of exact_key+<stat> (percentage points; positive means the statistic
    helps). Null: the rows of activation statistics are permuted jointly within strata and the whole
    held-out procedure is repeated ``--n-perm`` times; one-sided p = (1 + #{T_null >= T_obs}) /
    (1 + n_perm). Three strata: (a) exact key x fold, the primary test: since T is a cross-validated
    statistic, permuting within folds keeps each fold's set of statistic values, so the null is
    exact when latency is independent of the statistics given the key; (b) exact key, the plain
    within-key permutation (it also moves values between folds and therefore between prompts);
    (c) exact key x layer, which asks whether a statistic helps beyond the key and the layer index.
    Layer itself is included as a control statistic (not under (c), where it is constant).
    Holm-adjusted p-values over the six activation statistics.
T3  Conditional mutual information (M7d-iii): plug-in I(X; oracle chunk | Z) in bits, with X
    discretized (8 global quantile bins as in the manuscript; 4 quantile bins within each stratum;
    the moment vector as a joint within-stratum median split of the five moments and as 8 k-means
    clusters of the standardized moments) and Z the length bucket, the length bucket and batch, the
    exact key, or the exact key and layer. Null: X permuted within Z, ``--n-cmi-perm`` permutations;
    reported are the plug-in value, the null mean, the excess over the null mean, the null 95th
    percentile, the p-value and H(oracle | Z). References: log L given batch (the paper's 1.23 bits)
    and log L given length bucket and batch (what the exact length adds within a bucket).
T4  Higher-capacity predictors (M7d-iv): depth-3, the E3 depth 4 (reproduced), depth-6 and
    unbounded leaf-cost trees and both gradient-boosted variants on shape, shape+entropy and shape+all statistics,
    with the bootstrap intervals and paired differences of T1.
T5  Tie shares under practical-equivalence margins (M2c): for every predictor, the CI-tie share and
    the shares of calls whose chosen chunk is within 2% and 5% of the oracle latency, how often
    the two criteria disagree (CI tie but outside the margin; inside the margin but not a CI tie),
    per batch for the main selectors, and for the oracle tie sets themselves (mean set size and
    disagreement over all non-oracle (call, chunk) pairs).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as _dt  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

from src.experiments.tpds import e3_predictors as e3  # noqa: E402
from src.experiments.tpds.analysis import (chunk_columns, conditional_mutual_information,  # noqa: E402
                                           cv_exact_headroom, latency_matrix, load_e2)
from src.experiments.tpds.common import RESULTS_DIR, write_json  # noqa: E402

SHAPE = list(e3.SHAPE)
MOMENTS = list(e3.MOMENTS)
ACT_STATS = MOMENTS + ["delta_entropy_bits"]
HC_SETS = {"shape": SHAPE, "shape+entropy": SHAPE + ["entropy_bits"], "all": SHAPE + ACT_STATS}
EQ_MARGINS_PP = (0.5, 1.0)
TIE_MARGINS = (0.02, 0.05)
GBT_PARAMS = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10,
                  early_stopping=False, random_state=0)
MIN_LEAF = 5


# --------------------------------------------------------------------------- data
def prompt_clusters(df: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Connected components of the bipartite call-prompt graph (calls sharing a prompt are linked)."""
    ids: dict[str, int] = {}
    rows, cols = [], []
    for i, p in enumerate(df["prompt_ids"].astype(str)):
        for q in set(p.split(";")):
            rows.append(i)
            cols.append(ids.setdefault(q, len(ids)))
    n, m = len(df), len(ids)
    a = coo_matrix((np.ones(len(rows)), (np.array(rows), np.array(cols) + n)), shape=(n + m, n + m))
    _, lab = connected_components(a, directed=False)
    _, codes = np.unique(lab[:n], return_inverse=True)
    return codes.ravel(), m


class Table:
    """One per-call table with the derived arrays every test needs."""

    def __init__(self, df: pd.DataFrame, tag: str, model: str, host: str, source: str):
        self.df, self.tag, self.model, self.host, self.source = df, tag, model, host, source
        self.N = len(df)
        self.chunks = chunk_columns(df)
        self.C = len(self.chunks)
        self.lat = latency_matrix(df, self.chunks)
        self.orc = self.lat.min(1)
        self.norm = self.lat / self.orc[:, None]
        self.norm_fit = np.minimum(self.norm, 1e6)          # finite copy for fitting (missing timings)
        self.n_nonfinite_latencies = int((~np.isfinite(self.lat)).sum())
        self.oracle_idx = self.lat.argmin(1)
        tied = [set(int(c) for c in str(t).split(";") if c) for t in df["tied_chunks"].fillna("")]
        self.tied_mask = np.array([[c in t for c in self.chunks] for t in tied], dtype=bool)
        self.folds = df["fold"].to_numpy()
        self.batch = df["batch"].to_numpy()
        self.layer = df["layer"].to_numpy()
        L = df["L"].to_numpy()
        self.aligned = (L % 16 == 0).astype(int)
        keys = pd.DataFrame({"L": L, "batch": self.batch, "width": df["width"].to_numpy(), "aligned": self.aligned})
        self.key = keys.groupby(list(keys.columns), sort=True).ngroup().to_numpy()
        self.key_layer = pd.DataFrame({"k": self.key, "l": self.layer}).groupby(["k", "l"]).ngroup().to_numpy()
        self.key_fold = pd.DataFrame({"k": self.key, "f": self.folds}).groupby(["k", "f"]).ngroup().to_numpy()
        self.bucket_key = pd.DataFrame({"lb": df["L_bucket"].to_numpy(), "b": self.batch, "a": self.aligned}) \
            .groupby(["lb", "b", "a"]).ngroup().to_numpy()
        self.batch_code = pd.factorize(self.batch, sort=True)[0]
        self.Lb = np.log2(df["L_bucket"].to_numpy()).astype(int)
        self.clusters, self.n_prompts = prompt_clusters(df)
        self.requests = pd.factorize(df["group_id"])[0]
        per_cluster_folds = pd.Series(self.folds).groupby(self.clusters).nunique()
        self.clusters_nested_in_folds = bool((per_cluster_folds == 1).all())


# --------------------------------------------------------------------------- held-out predictors
def _fold_iter(folds):
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if tr.any() and te.any():
            yield tr, te


def cv_table(t: Table, levels: list[np.ndarray], folds=None) -> tuple[np.ndarray, dict]:
    """Hierarchical lookup table (finest level first, then coarser levels, then the best single chunk).

    Each table entry is the chunk with the lowest summed normalized latency over its training calls.
    """
    folds = t.folds if folds is None else folds
    ch = np.empty(t.N, dtype=int)
    used = np.zeros(len(levels) + 1, dtype=int)
    for tr, te in _fold_iter(folds):
        out = np.full(te.sum(), int(np.argmin(t.norm_fit[tr].sum(0))))
        lev_of = np.full(te.sum(), len(levels))
        for li in reversed(range(len(levels))):
            codes = levels[li]
            sums = np.zeros((codes.max() + 1, t.C))
            np.add.at(sums, codes[tr], t.norm_fit[tr])
            have = np.zeros(codes.max() + 1, dtype=bool)
            have[codes[tr]] = True
            c = codes[te]
            m = have[c]
            out[m] = sums.argmin(1)[c[m]]
            lev_of[m] = li
        ch[te] = out
        used += np.bincount(lev_of, minlength=len(levels) + 1)
    return ch, {"calls_per_level": used.tolist()}


def cv_static(t: Table, folds=None) -> np.ndarray:
    return cv_table(t, [], folds)[0]


def entropy_rule_choice(t: Table) -> np.ndarray:
    chunks = t.chunks
    rule = t.df["entropy_rule_chunk"].clip(min(chunks), max(chunks)).to_numpy()
    return np.array([chunks.index(int(c)) if int(c) in chunks else
                     int(np.argmin([abs(np.log2(c) - np.log2(x)) for x in chunks])) for c in rule])


def cv_tree(t: Table, cols: list[str], depth, folds=None) -> np.ndarray:
    folds = t.folds if folds is None else folds
    X = t.df[cols].to_numpy(float)
    ch = np.empty(t.N, dtype=int)
    for tr, te in _fold_iter(folds):
        m = e3.LeafCostTree(depth).fit(X[tr], t.oracle_idx[tr], t.norm[tr])
        ch[te] = m.predict_idx(X[te])
    return ch


def cv_gbt_cls(t: Table, cols: list[str]) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = t.df[cols].to_numpy(float)
    ch = np.empty(t.N, dtype=int)
    for tr, te in _fold_iter(t.folds):
        y = t.oracle_idx[tr]
        if len(np.unique(y)) == 1:
            ch[te] = int(y[0])
            continue
        ch[te] = HistGradientBoostingClassifier(**GBT_PARAMS).fit(X[tr], y).predict(X[te]).astype(int)
    return ch


def cv_gbt_reg(t: Table, cols: list[str]) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = t.df[cols].to_numpy(float)
    ch = np.empty(t.N, dtype=int)
    logn = np.log(t.norm_fit)
    for tr, te in _fold_iter(t.folds):
        P = np.column_stack([HistGradientBoostingRegressor(**GBT_PARAMS).fit(X[tr], logn[tr, c]).predict(X[te])
                             for c in range(t.C)])
        ch[te] = P.argmin(1)
    return ch


class ExactKeySplit:
    """Held-out exact-key table, optionally split once per key on a statistic (T2).

    The base table and its fallbacks are computed once; ``run`` evaluates the split model for a
    matrix of statistics (one column per statistic), so it can be called for every permutation.
    """

    def __init__(self, t: Table, min_leaf: int = MIN_LEAF, eps: float = 1e-9):
        self.t, self.min_leaf, self.eps = t, min_leaf, eps
        self.base_choice, self.base_info = cv_table(t, [t.key, t.bucket_key])
        self.base_regret = t.norm[np.arange(t.N), self.base_choice] - 1
        self.covered = np.zeros(t.N, dtype=bool)
        self.cells = []                     # (train idx, test idx, base training cost)
        for tr, te in _fold_iter(t.folds):
            tr_keys = np.unique(t.key[tr])
            for k in np.unique(t.key[te]):
                te_idx = np.where(te & (t.key == k))[0]
                if k not in tr_keys:
                    continue
                self.covered[te_idx] = True
                tr_idx = np.where(tr & (t.key == k))[0]
                if len(tr_idx) < 2 * min_leaf:
                    continue
                self.cells.append((tr_idx, te_idx, float(t.norm_fit[tr_idx].sum(0).min())))

    def run(self, S: np.ndarray, any_cols: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (choices per statistic [s, N], choice of the best-of-any_cols split [N], splits per statistic)."""
        t, lo = self.t, self.min_leaf - 1
        s = S.shape[1]
        choice = np.repeat(self.base_choice[None, :], s, axis=0)
        any_choice = self.base_choice.copy()
        n_split = np.zeros(s, dtype=int)
        any_mask = np.zeros(s, dtype=bool)
        any_mask[any_cols] = True
        ar = np.arange(s)
        for tr_idx, te_idx, base_cost in self.cells:
            n = len(tr_idx)
            hi = n - self.min_leaf - 1
            X = S[tr_idx]
            order = np.argsort(X, axis=0, kind="stable")
            xs = np.take_along_axis(X, order, 0)
            cs = np.cumsum(t.norm_fit[tr_idx][order], axis=0)      # (n, s, C)
            left = cs[lo:hi + 1]
            right = cs[-1][None] - left
            cost = left.min(-1) + right.min(-1)                      # (m, s)
            cost = np.where(xs[lo:hi + 1] < xs[lo + 1:hi + 2], cost, np.inf)
            bi = cost.argmin(0)
            gain = base_cost - cost[bi, ar]
            split = gain > self.eps
            if not split.any():
                continue
            n_split += split
            thr = 0.5 * (xs[lo + bi, ar] + xs[lo + bi + 1, ar])
            lch = left[bi, ar].argmin(-1)
            rch = right[bi, ar].argmin(-1)
            xt = S[te_idx]                                           # (q, s)
            ch = np.where(xt <= thr[None, :], lch[None, :], rch[None, :])
            for j in np.where(split)[0]:
                choice[j, te_idx] = ch[:, j]
            cand = np.where(split & any_mask)[0]
            if len(cand):
                j = cand[np.argmax(gain[cand])]
                any_choice[te_idx] = ch[:, j]
        return choice, any_choice, n_split


def cost_model_choice(t: Table, in_dir: Path) -> tuple[np.ndarray | None, dict]:
    """Stored per-call cost-model choice (Mamba-1 only), checked against the stored curves."""
    info: dict = {"available": False}
    if t.model != "mamba1":
        info["reason"] = "per-call choice stored for the Mamba-1 Triton tables only"
        return None, info
    path = in_dir / f"cost_model_{t.host}.json"
    if not path.exists():
        info["reason"] = f"no stored cost-model run for this table ({path.name} absent)"
        return None, info
    rec = json.loads(path.read_text()).get("per_call_choice", {}).get("mamba1")
    if not rec:
        info["reason"] = "no per-call choice in the stored run"
        return None, info
    m = dict(zip(rec["keys"], rec["chunk"]))
    if not set(t.df["key"]).issubset(m):
        info["reason"] = "stored keys do not cover this table"
        return None, info
    ch = np.array([t.chunks.index(int(m[k])) for k in t.df["key"]])
    info.update({"available": True, "source": path.name})
    curves = in_dir / f"cost_model_curves_{t.host}.csv"
    if curves.exists():
        cv = pd.read_csv(curves)
        best = cv.loc[cv.groupby(["batch", "L"])["predicted_ms"].idxmin()].set_index(["batch", "L"])["chunk"]
        alt = np.array([best.get((int(b), int(L)), -1) for b, L in zip(t.batch, t.df["L"])])
        info["agreement_with_curve_argmin"] = float((alt == np.array(t.chunks)[ch]).mean())
        info["curves_source"] = curves.name
    return ch, info


# --------------------------------------------------------------------------- bootstrap (T1)
def _pct_ci(x: np.ndarray, level: float) -> list[float]:
    a = (1 - level) / 2
    return [float(np.percentile(x, 100 * a)), float(np.percentile(x, 100 * (1 - a)))]


def _weighted_p95(r: np.ndarray, W: np.ndarray) -> np.ndarray:
    order = np.argsort(r, kind="stable")
    cw = np.cumsum(W[:, order], axis=1)
    pos = (cw < 0.95 * cw[:, -1:]).sum(1)
    return r[order][np.minimum(pos, len(r) - 1)]


def cluster_bootstrap(values: dict[str, np.ndarray], clusters: np.ndarray, B: int, rng: np.random.Generator,
                      p95_of: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Replicates (length B) of the mean of every value array and of the 95th percentile of every p95_of array."""
    K = int(clusters.max()) + 1
    M = rng.multinomial(K, np.full(K, 1.0 / K), size=B).astype(float)
    names = list(values)
    X = np.stack([values[n] for n in names], 1).astype(float)
    Sx = np.zeros((K, X.shape[1]))
    np.add.at(Sx, clusters, X)
    cnt = np.bincount(clusters, minlength=K).astype(float)
    means = (M @ Sx) / (M @ cnt)[:, None]
    out = {n: means[:, j] for j, n in enumerate(names)}
    if p95_of:
        W = M[:, clusters]
        for n, r in p95_of.items():
            out[n] = _weighted_p95(r, W)
    return out


def equivalence(point: float, reps: np.ndarray) -> dict:
    ci95, ci90 = _pct_ci(reps, 0.95), _pct_ci(reps, 0.90)
    rec = {"diff_pp": point, "ci95_pp": ci95, "ci90_pp": ci90, "verdicts": {}}
    for d in EQ_MARGINS_PP:
        if -d < ci95[0] and ci95[1] < d:
            v = "equivalent"
        elif ci95[0] > d or ci95[1] < -d:
            v = "not_equivalent"
        else:
            v = "inconclusive"
        rec["verdicts"][f"margin_{d:g}pp"] = {"verdict_95ci": v, "tost90_equivalent": bool(-d < ci90[0] and ci90[1] < d)}
    return rec


# --------------------------------------------------------------------------- permutations
def within_strata_perm(strata: np.ndarray, rng: np.random.Generator, base_order: np.ndarray) -> np.ndarray:
    """Index array that permutes rows uniformly within each stratum."""
    src = np.lexsort((rng.random(len(strata)), strata))
    perm = np.empty(len(strata), dtype=int)
    perm[base_order] = src
    return perm


def _holm(p: dict[str, float]) -> dict[str, float]:
    items = sorted(p.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v))
        out[k] = run
    return out


def permutation_test(t: Table, model: ExactKeySplit, n_perm: int, rng: np.random.Generator) -> dict:
    stats = [c for c in ACT_STATS if c in t.df.columns] + ["layer"]
    any_cols = list(range(len(stats) - 1))
    S = t.df[stats].to_numpy(float)
    ar = np.arange(t.N)

    def _T(S_):
        ch, any_ch, n_split = model.run(S_, any_cols)
        reg = t.norm[ar[None, :], ch] - 1                              # (s, N)
        any_reg = t.norm[ar, any_ch] - 1
        all_reg = np.vstack([reg, any_reg[None]])
        T = 100 * (model.base_regret[None, :] - all_reg).mean(1)
        Tc = 100 * (model.base_regret[None, model.covered] - all_reg[:, model.covered]).mean(1)
        return T, Tc, n_split, ch, any_ch

    T_obs, Tc_obs, split_obs, ch_obs, any_obs = _T(S)
    names = stats + ["any_moment"]
    out = {"statistics": names, "min_leaf": model.min_leaf, "n_perm": n_perm,
           "base_heldout_mean_regret_pct": float(100 * model.base_regret.mean()),
           "base_calls_per_level": model.base_info["calls_per_level"],
           "n_covered_calls": int(model.covered.sum()), "n_cells_splittable": len(model.cells),
           "observed": {n: {"heldout_mean_regret_pct": float(100 * model.base_regret.mean() - T_obs[j]),
                            "T_pp": float(T_obs[j]), "T_covered_pp": float(Tc_obs[j]),
                            "n_cells_split": int(split_obs[j]) if j < len(stats) else None}
                        for j, n in enumerate(names)},
           "strata": {}}
    for sname, strata in (("exact_key_x_fold", t.key_fold), ("exact_key", t.key), ("exact_key_x_layer", t.key_layer)):
        base_order = np.argsort(strata, kind="stable")
        null = np.empty((n_perm, len(names)))
        null_split = np.empty((n_perm, len(stats)))
        for p in range(n_perm):
            perm = within_strata_perm(strata, rng, base_order)
            T, _, ns, _, _ = _T(S[perm])
            null[p], null_split[p] = T, ns
        rec = {}
        for j, n in enumerate(names):
            if sname == "exact_key_x_layer" and n == "layer":
                rec[n] = {"note": "layer is constant within the strata; not testable"}
                continue
            rec[n] = {"T_obs_pp": float(T_obs[j]), "null_mean_pp": float(null[:, j].mean()),
                      "null_p95_pp": float(np.percentile(null[:, j], 95)),
                      "p_value": float((1 + (null[:, j] >= T_obs[j] - 1e-12).sum()) / (1 + n_perm))}
            if j < len(stats):
                rec[n]["null_mean_cells_split"] = float(null_split[:, j].mean())
        holm = _holm({n: rec[n]["p_value"] for n in names[:len(stats) - 1]})
        for n, v in holm.items():
            rec[n]["p_holm_over_6_statistics"] = v
        out["strata"][sname] = rec
    return out, ch_obs, any_obs, stats


# --------------------------------------------------------------------------- conditional MI (T3)
def _disc_global(v: np.ndarray, bins: int = 8) -> np.ndarray:
    """Identical to the discretization of analysis.conditional_mutual_information."""
    qs = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.searchsorted(qs, v, side="right")


def _within_bins(v: np.ndarray, strata: np.ndarray, q: int) -> np.ndarray:
    order = np.lexsort((v, strata))
    s_sorted = strata[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s_sorted)) + 1]
    sizes = np.diff(np.r_[starts, len(v)])
    pos = np.arange(len(v)) - np.repeat(starts, sizes)
    out = np.empty(len(v), dtype=int)
    out[order] = (pos * q // np.repeat(sizes, sizes)).astype(int)
    return out


def _H(codes: np.ndarray) -> float:
    c = np.bincount(codes)
    c = c[c > 0].astype(float)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def _codes(v) -> tuple[np.ndarray, int]:
    c = pd.factorize(np.asarray(v), sort=True)[0]
    return c, int(c.max()) + 1


def cmi_test(x, y, z, n_perm: int, rng: np.random.Generator, perms: list[np.ndarray] | None = None) -> dict:
    """Plug-in I(X; Y | Z) in bits with a within-Z permutation null of X."""
    xc, nx = _codes(x)
    yc, ny = _codes(y)
    zc, _ = _codes(z)
    hxz, hyz, hz = _H(zc * nx + xc), _H(zc * ny + yc), _H(zc)
    obs = hxz + hyz - _H((zc * nx + xc) * ny + yc) - hz
    if perms is None:
        base_order = np.argsort(zc, kind="stable")
        perms = [within_strata_perm(zc, rng, base_order) for _ in range(n_perm)]
    null = np.array([hxz + hyz - _H((zc * nx + xc[p]) * ny + yc) - hz for p in perms])
    return {"cmi_bits": float(obs), "null_mean_bits": float(null.mean()), "excess_bits": float(obs - null.mean()),
            "null_p95_bits": float(np.percentile(null, 95)),
            "p_value": float((1 + (null >= obs - 1e-12).sum()) / (1 + len(null))),
            "H_oracle_given_cond_bits": float(hyz - hz), "x_levels": nx, "n_strata": int(zc.max()) + 1}


def cmi_block(t: Table, n_perm: int, rng: np.random.Generator) -> dict:
    from sklearn.cluster import KMeans
    df, y = t.df, t.df["oracle_chunk"].to_numpy()
    mom = df[MOMENTS].to_numpy(float)
    z_std = (mom - mom.mean(0)) / np.where(mom.std(0) > 0, mom.std(0), 1)
    kmeans8 = KMeans(n_clusters=8, n_init=10, random_state=0).fit_predict(z_std)
    conds = {"L_bucket": t.Lb, "L_bucket_batch": t.Lb * 100 + t.batch, "exact_key": t.key,
             "exact_key_x_layer": t.key_layer}
    out: dict = {"estimator": "plug-in, bits; null: X permuted within the conditioning strata", "by_condition": {}}
    for cname, z in conds.items():
        zc, _ = _codes(z)
        base_order = np.argsort(zc, kind="stable")
        perms = [within_strata_perm(zc, rng, base_order) for _ in range(n_perm)]
        xs = {"entropy_g8": _disc_global(df["entropy_bits"].to_numpy(float)),
              "entropy_q4_within": _within_bins(df["entropy_bits"].to_numpy(float), zc, 4)}
        for m in ACT_STATS[1:]:
            xs[f"{m}_g8"] = _disc_global(df[m].to_numpy(float))
        joint = np.zeros(t.N, dtype=int)
        for j in range(len(MOMENTS)):
            joint = joint * 2 + _within_bins(mom[:, j], zc, 2)
        xs["moments_joint_median_within"] = joint
        xs["moments_kmeans8"] = kmeans8
        if cname != "exact_key_x_layer":
            xs["layer_control"] = t.layer
        out["by_condition"][cname] = {k: cmi_test(v, y, zc, n_perm, rng, perms) for k, v in xs.items()}
    logL = _disc_global(np.log2(df["L"].to_numpy(float)))
    out["reference"] = {
        "logL_g8_given_batch": cmi_test(logL, y, t.batch, n_perm, rng),
        "logL_g8_given_L_bucket_batch": cmi_test(logL, y, t.Lb * 100 + t.batch, n_perm, rng)}
    return out


# --------------------------------------------------------------------------- tie margins (T5)
def tie_margin_block(t: Table, outcomes: dict[str, dict], main: list[str]) -> dict:
    out: dict = {"predictors": {}, "per_batch": {}, "oracle_tie_sets": {}}
    for name, o in outcomes.items():
        r, a = o["regret"], o["tie"]
        rec = {"ci_tie_share": float(a.mean())}
        for m in TIE_MARGINS:
            b = r <= m + 1e-12
            k = f"{round(100 * m)}pct"
            rec[f"within_{k}_share"] = float(b.mean())
            rec[f"disagree_{k}_share"] = float((a != b).mean())
            rec[f"ci_tie_but_outside_{k}_share"] = float((a & ~b).mean())
            rec[f"within_{k}_but_not_ci_tie_share"] = float((~a & b).mean())
        out["predictors"][name] = rec
    for name in main:
        if name not in outcomes:
            continue
        r, a = outcomes[name]["regret"], outcomes[name]["tie"]
        out["per_batch"][name] = {}
        for b in np.unique(t.batch):
            s = t.batch == b
            rec = {"n": int(s.sum()), "ci_tie_share": float(a[s].mean())}
            for m in TIE_MARGINS:
                k = f"{round(100 * m)}pct"
                w = r[s] <= m + 1e-12
                rec[f"within_{k}_share"] = float(w.mean())
                rec[f"disagree_{k}_share"] = float((a[s] != w).mean())
            out["per_batch"][name][int(b)] = rec
    non_oracle = np.ones_like(t.tied_mask)
    non_oracle[np.arange(t.N), t.oracle_idx] = False
    rec = {"mean_ci_tie_set_size": float(t.tied_mask.sum(1).mean())}
    for m in TIE_MARGINS:
        k = f"{round(100 * m)}pct"
        within = t.norm <= 1 + m + 1e-12
        a, b = t.tied_mask[non_oracle], within[non_oracle]
        rec[f"mean_within_{k}_set_size"] = float(within.sum(1).mean())
        rec[f"pairs_disagree_{k}_share"] = float((a != b).mean())
        rec[f"pairs_ci_tie_but_outside_{k}_share"] = float((a & ~b).mean())
        rec[f"pairs_within_{k}_but_not_ci_tie_share"] = float((~a & b).mean())
        inter = (t.tied_mask & within).sum(1)
        union = (t.tied_mask | within).sum(1)
        rec[f"mean_jaccard_ci_vs_{k}"] = float((inter / union).mean())
    rec["n_non_oracle_pairs"] = int(non_oracle.sum())
    out["oracle_tie_sets"] = rec
    return out


# --------------------------------------------------------------------------- per-tag driver
def _meta(args) -> dict:
    import scipy
    import sklearn
    return {"utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "python": sys.version.split()[0],
            "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "argv": sys.argv}


def _stored_e3(t: Table, in_dir: Path) -> dict | None:
    if t.model == "mamba1-ref":
        p = in_dir / f"e8_analysis_{t.host}.json"
        return json.loads(p.read_text()).get("predictors") if p.exists() else None
    p = in_dir / f"e3_{t.tag}.json"
    return json.loads(p.read_text()).get("predictors_all_batches") if p.exists() else None


def analyze_tag(t: Table, args) -> dict:
    t0 = time.time()
    seed = int(args.seed) + zlib.crc32(t.tag.encode()) % 100_000
    ss = np.random.SeedSequence(seed)
    rng_boot, rng_req, rng_perm, rng_cmi, rng_cv = (np.random.default_rng(s) for s in ss.spawn(5))
    ar = np.arange(t.N)
    choices: dict[str, np.ndarray] = {}
    extra: dict = {}

    # E3 predictors (depth as in E3) + the Table III batch-only row
    choices["global_static"] = cv_static(t)
    choices["entropy_rule"] = entropy_rule_choice(t)
    choices["batch_only"], extra["batch_only"] = cv_table(t, [t.batch_code])
    for name, (cols, depth) in e3.FEATURE_SETS.items():
        choices[name] = cv_tree(t, cols, depth)
    # higher capacity
    for sname, cols in HC_SETS.items():
        choices[f"tree_d3:{sname}"] = cv_tree(t, cols, 3)
        choices[f"tree_d6:{sname}"] = cv_tree(t, cols, 6)
        choices[f"tree_unbounded:{sname}"] = cv_tree(t, cols, None)
        if not args.skip_gbt:
            choices[f"gbt_cls:{sname}"] = cv_gbt_cls(t, cols)
            choices[f"gbt_reg:{sname}"] = cv_gbt_reg(t, cols)
    # exact key, its in-sample floor and the in-sample per-batch best static chunk
    ek = ExactKeySplit(t)
    choices["exact_key_table"] = ek.base_choice
    floor = np.empty(t.N, dtype=int)
    for codes in (t.key,):
        sums = np.zeros((codes.max() + 1, t.C))
        np.add.at(sums, codes, t.norm_fit)
        floor = sums.argmin(1)[codes]
    choices["exact_key_floor_in_sample"] = floor
    sums = np.zeros((t.batch_code.max() + 1, t.C))
    np.add.at(sums, t.batch_code, t.norm_fit)
    choices["best_static_per_batch_in_sample"] = sums.argmin(1)[t.batch_code]
    cm_ch, cm_info = cost_model_choice(t, args.in_dir)
    if cm_ch is not None:
        choices["cost_model"] = cm_ch
    t1 = time.time()

    # T2 permutation test (also yields the observed exact_key+<stat> predictors)
    perm, ch_obs, any_obs, stats = permutation_test(t, ek, args.n_perm, rng_perm)
    for j, s in enumerate(stats):
        choices[f"exact_key+{s}"] = ch_obs[j]
    choices["exact_key+any_moment"] = any_obs
    t2 = time.time()

    outcomes = {n: {"regret": t.norm[ar, c] - 1, "tie": t.tied_mask[ar, c]} for n, c in choices.items()}
    if t.model == "mamba1-ref" and "lat_stock_mean" in t.df:
        stock_tie = np.array([int(c) in set(int(x) for x in str(s).split(";") if x)
                              for c, s in zip(t.df["stock_chunk"], t.df["tied_chunks"].fillna(""))])
        outcomes["stock_dispatch"] = {"regret": t.df["lat_stock_mean"].to_numpy(float) / t.orc - 1, "tie": stock_tie}

    # T0 reproduction
    repro: dict = {}
    stored = _stored_e3(t, args.in_dir)
    fresh = e3.evaluate(t.df, t.chunks)
    rows = {}
    for name in ["global_static", "entropy_rule", *e3.FEATURE_SETS]:
        o = outcomes[name]
        rec = {"mean_regret_pct": float(100 * o["regret"].mean()), "tie_share": float(o["tie"].mean()),
               "fresh_e3_mean_regret_pct": fresh[name]["mean_regret_pct"]}
        if stored and name in stored:
            rec["stored_mean_regret_pct"] = stored[name]["mean_regret_pct"]
            rec["stored_tie_share"] = stored[name]["tie_aware_accuracy"]
            rec["stored_p95_regret_pct"] = stored[name]["p95_regret_pct"]
        rows[name] = rec
    repro["e3_predictors"] = rows
    diffs = [abs(r["mean_regret_pct"] - r["stored_mean_regret_pct"]) for r in rows.values() if "stored_mean_regret_pct" in r]
    repro["max_abs_diff_vs_stored_pp"] = float(max(diffs)) if diffs else None
    repro["max_abs_diff_vs_fresh_e3_pp"] = float(max(abs(r["mean_regret_pct"] - r["fresh_e3_mean_regret_pct"])
                                                    for r in rows.values()))
    repro["batch_only"] = {"mean_regret_pct": float(100 * outcomes["batch_only"]["regret"].mean()),
                           "p95_regret_pct": float(100 * np.percentile(outcomes["batch_only"]["regret"], 95))}
    repro["exact_shape_floor_in_sample_pct"] = float(100 * outcomes["exact_key_floor_in_sample"]["regret"].mean())
    cvh = cv_exact_headroom(t.df, t.chunks)
    repro["cv_exact_headroom_paper_function"] = {"mean_pct": cvh["mean_pct"], "mean_ms": cvh["mean_ms"],
                                                 "fallbacks": cvh["fallbacks"]}
    repro["exact_key_table_regret_optimal_heldout_pct"] = float(100 * ek.base_regret.mean())
    oc = t.df["oracle_chunk"].to_numpy()
    repro["cmi_paper_function"] = {
        "entropy_given_L_bucket": conditional_mutual_information(t.df["entropy_bits"].to_numpy(), oc, t.Lb),
        "logL_given_batch": conditional_mutual_information(np.log2(t.df["L"].to_numpy()), oc, t.batch)}
    if cm_ch is not None:
        stored_cm = json.loads((args.in_dir / cm_info["source"]).read_text()).get("V2_e2_calls", {})
        repro["cost_model"] = {"mean_regret_pct": float(100 * outcomes["cost_model"]["regret"].mean()),
                               "tie_share": float(outcomes["cost_model"]["tie"].mean()),
                               "stored_mean_regret_pct": stored_cm.get("mean_regret_pct"),
                               "stored_tie_share": stored_cm.get("tie_aware_accuracy")}

    # T1 bootstrap (prompt clusters; request groups)
    vals, p95s = {}, {}
    for n, o in outcomes.items():
        vals[f"{n}|mean"] = 100 * o["regret"]
        vals[f"{n}|tie"] = o["tie"].astype(float)
        for m in TIE_MARGINS:
            vals[f"{n}|w{round(100 * m)}"] = (o["regret"] <= m + 1e-12).astype(float)
        p95s[f"{n}|p95"] = 100 * o["regret"]
    pairs = []
    for fam in ["", "tree_d3:", "tree_d6:", "tree_unbounded:", "gbt_cls:", "gbt_reg:"]:
        for other in ["shape+entropy", "all"]:
            a, b = f"{fam}{other}", f"{fam}shape"
            if a in outcomes and b in outcomes:
                pairs.append((a, b))
    pairs += [(f"exact_key+{s}", "exact_key_table") for s in [*stats, "any_moment"]]
    pairs += [("moments", "entropy_only"), ("shape", "L_only")]
    if "cost_model" in outcomes:
        pairs.append(("cost_model", "shape"))
    for a, b in pairs:
        vals[f"{a} - {b}|diff"] = 100 * (outcomes[a]["regret"] - outcomes[b]["regret"])
    reps = cluster_bootstrap(vals, t.clusters, args.n_boot, rng_boot, p95s)
    reps_req = cluster_bootstrap({k: v for k, v in vals.items() if k.endswith("|mean") or k.endswith("|diff")},
                                 t.requests, args.n_boot, rng_req, {})
    preds = {}
    for n, o in outcomes.items():
        r = o["regret"]
        rec = {"mean_regret_pct": float(100 * r.mean()), "mean_ci95_prompt_cluster": _pct_ci(reps[f"{n}|mean"], 0.95),
               "mean_ci95_request_group": _pct_ci(reps_req[f"{n}|mean"], 0.95),
               "p95_regret_pct": float(100 * np.percentile(r, 95)), "p95_ci95_prompt_cluster": _pct_ci(reps[f"{n}|p95"], 0.95),
               "ci_tie_share": float(o["tie"].mean()), "ci_tie_ci95_prompt_cluster": _pct_ci(reps[f"{n}|tie"], 0.95),
               "oracle_hit_rate": float((r <= 1e-12).mean())}
        for m in TIE_MARGINS:
            k = round(100 * m)
            rec[f"within_{k}pct_share"] = float((r <= m + 1e-12).mean())
            rec[f"within_{k}pct_ci95_prompt_cluster"] = _pct_ci(reps[f"{n}|w{k}"], 0.95)
        if n in choices:
            rec["chosen_distribution"] = {int(k): int(v) for k, v in
                                          pd.Series(np.array(t.chunks)[choices[n]]).value_counts().items()}
        preds[n] = rec
    differences = {}
    for a, b in pairs:
        k = f"{a} - {b}"
        point = float(vals[f"{k}|diff"].mean())
        differences[k] = {"prompt_cluster": equivalence(point, reps[f"{k}|diff"]),
                          "request_group": equivalence(point, reps_req[f"{k}|diff"])}
    t3 = time.time()

    # T1 training variability: repeated grouped CV over prompt clusters (trees)
    rcv: dict = {"n_repeats": args.n_cv_repeats, "unit": "prompt clusters assigned at random to 5 folds"}
    if args.n_cv_repeats > 0:
        K = int(t.clusters.max()) + 1
        depths = (3, 4, 6)          # 4 is the depth of the E3 implementation
        rec = {f"d{d}:{s}": [] for d in depths for s in HC_SETS}
        for _ in range(args.n_cv_repeats):
            fold_of = rng_cv.permutation(K) % 5
            folds = fold_of[t.clusters]
            for d in depths:
                for s, cols in HC_SETS.items():
                    ch = cv_tree(t, cols, d, folds)
                    rec[f"d{d}:{s}"].append(100 * (t.norm[ar, ch] - 1).mean())
        for d in depths:
            for other in ("shape+entropy", "all"):
                diff = np.array(rec[f"d{d}:{other}"]) - np.array(rec[f"d{d}:shape"])
                rcv[f"depth{d}: {other} - shape"] = {
                    "mean_pp": float(diff.mean()), "range_pp": [float(diff.min()), float(diff.max())],
                    "p2.5_p97.5_pp": _pct_ci(diff, 0.95), "share_repeats_other_better": float((diff < -1e-9).mean()),
                    "share_repeats_identical": float((np.abs(diff) <= 1e-9).mean())}
            rcv[f"depth{d}: shape mean regret pct"] = {"mean": float(np.mean(rec[f"d{d}:shape"])),
                                                       "range": [float(np.min(rec[f"d{d}:shape"])),
                                                                 float(np.max(rec[f"d{d}:shape"]))]}
    t4 = time.time()

    # T3 conditional mutual information
    cmi = cmi_block(t, args.n_cmi_perm, rng_cmi)
    t5 = time.time()

    # T5 tie margins
    main = [n for n in ("global_static", "best_static_per_batch_in_sample", "shape", "cost_model", "stock_dispatch",
                        "exact_key_table") if n in outcomes]
    ties = tie_margin_block(t, outcomes, main)

    res = {"experiment": "E3b", "tag": t.tag, "model": t.model, "host": t.host, "source": t.source,
           "n_calls": t.N, "chunks": t.chunks, "n_prompts": t.n_prompts,
           "n_nonfinite_latencies": t.n_nonfinite_latencies,
           "n_prompt_clusters": int(t.clusters.max()) + 1, "n_request_groups": int(t.requests.max()) + 1,
           "prompt_clusters_nested_in_folds": t.clusters_nested_in_folds,
           "n_exact_keys": int(t.key.max()) + 1, "n_exact_key_x_layer_strata": int(t.key_layer.max()) + 1,
           "settings": {"seed": seed, "n_boot": args.n_boot, "n_perm": args.n_perm, "n_cmi_perm": args.n_cmi_perm,
                        "n_cv_repeats": args.n_cv_repeats, "equivalence_margins_pp": list(EQ_MARGINS_PP),
                        "tie_margins": list(TIE_MARGINS), "gbt_params": GBT_PARAMS if not args.skip_gbt else None,
                        "min_leaf_split": MIN_LEAF},
           "meta": _meta(args), "cost_model": cm_info, "extra": extra,
           "reproduction": repro, "predictors": preds, "differences": differences,
           "repeated_cv": rcv, "exact_key_permutation": perm, "conditional_mi": cmi, "tie_margins": ties,
           "timing_s": {"predictors": t1 - t0, "permutation": t2 - t1, "bootstrap": t3 - t2,
                        "repeated_cv": t4 - t3, "cmi": t5 - t4, "total": time.time() - t0}}
    return res


# --------------------------------------------------------------------------- summary
def _fmt_ci(v, ci):
    return f"{v:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def summary_row(res: dict) -> dict:
    p, d = res["predictors"], res["differences"]
    row = {"tag": res["tag"], "n_calls": res["n_calls"], "n_prompt_clusters": res["n_prompt_clusters"],
           "n_exact_keys": res["n_exact_keys"]}
    for n in ("global_static", "batch_only", "L_only", "shape", "shape+entropy", "all", "entropy_only", "moments",
              "tree_d3:shape", "tree_d3:shape+entropy", "tree_d3:all", "exact_key_table", "exact_key+entropy_bits", "exact_key+any_moment", "exact_key_floor_in_sample",
              "tree_d6:shape", "tree_d6:shape+entropy", "tree_d6:all", "gbt_cls:shape", "gbt_cls:shape+entropy",
              "gbt_cls:all", "gbt_reg:shape", "gbt_reg:shape+entropy", "gbt_reg:all", "cost_model",
              "stock_dispatch"):
        if n in p:
            row[f"{n} mean% [95% CI]"] = _fmt_ci(p[n]["mean_regret_pct"], p[n]["mean_ci95_prompt_cluster"])
    for k in ("shape+entropy - shape", "all - shape", "tree_d3:shape+entropy - tree_d3:shape",
              "tree_d3:all - tree_d3:shape", "tree_d6:shape+entropy - tree_d6:shape",
              "tree_d6:all - tree_d6:shape", "gbt_cls:shape+entropy - gbt_cls:shape", "gbt_cls:all - gbt_cls:shape",
              "gbt_reg:shape+entropy - gbt_reg:shape", "gbt_reg:all - gbt_reg:shape",
              "exact_key+entropy_bits - exact_key_table", "exact_key+any_moment - exact_key_table"):
        if k in d:
            e = d[k]["prompt_cluster"]
            row[f"{k} pp [95% CI]"] = _fmt_ci(e["diff_pp"], e["ci95_pp"])
            row[f"{k} verdict 0.5/1pp"] = "/".join(e["verdicts"][f"margin_{m:g}pp"]["verdict_95ci"] for m in EQ_MARGINS_PP)
    pt = res["exact_key_permutation"]["strata"]
    for s in ("exact_key_x_fold", "exact_key", "exact_key_x_layer"):
        for st in ("entropy_bits", "any_moment", "layer"):
            if st in pt[s] and "p_value" in pt[s][st]:
                row[f"perm[{s}] {st} T_pp; p"] = f"{pt[s][st]['T_obs_pp']:.3f}; {pt[s][st]['p_value']:.3f}"
        ps = [pt[s][st]["p_value"] for st in ACT_STATS if st in pt[s]]
        row[f"perm[{s}] min p over 6 stats (Holm)"] = (
            f"{min(ps):.3f} ({min(pt[s][st]['p_holm_over_6_statistics'] for st in ACT_STATS if st in pt[s]):.3f})")
    c = res["conditional_mi"]["by_condition"]
    for cond in ("L_bucket", "exact_key", "exact_key_x_layer"):
        for x in ("entropy_g8", "entropy_q4_within", "moments_joint_median_within", "moments_kmeans8"):
            r = c[cond][x]
            row[f"CMI {x}|{cond} excess bits; p"] = f"{r['excess_bits']:.3f}; {r['p_value']:.3f}"
        row[f"H(oracle|{cond}) bits"] = f"{c[cond]['entropy_g8']['H_oracle_given_cond_bits']:.3f}"
    ref = res["conditional_mi"]["reference"]
    row["CMI logL|batch excess bits"] = f"{ref['logL_g8_given_batch']['excess_bits']:.3f}"
    row["CMI logL|bucket,batch excess bits"] = f"{ref['logL_g8_given_L_bucket_batch']['excess_bits']:.3f}"
    tm = res["tie_margins"]["predictors"]
    for n in ("global_static", "shape", "cost_model", "stock_dispatch"):
        if n in tm:
            r = tm[n]
            row[f"ties {n} CI/2%/5%"] = (f"{100 * r['ci_tie_share']:.0f}/{100 * r['within_2pct_share']:.0f}/"
                                         f"{100 * r['within_5pct_share']:.0f}")
            row[f"ties {n} disagree 2%/5%"] = f"{100 * r['disagree_2pct_share']:.0f}/{100 * r['disagree_5pct_share']:.0f}"
    o = res["tie_margins"]["oracle_tie_sets"]
    row["oracle tie-set size CI/2%/5%"] = (f"{o['mean_ci_tie_set_size']:.2f}/{o['mean_within_2pct_set_size']:.2f}/"
                                           f"{o['mean_within_5pct_set_size']:.2f}")
    row["repro max |diff| vs stored E3 pp"] = res["reproduction"]["max_abs_diff_vs_stored_pp"]
    return row


# --------------------------------------------------------------------------- main
def discover(in_dir: Path, tags: list[str] | None) -> list[tuple[str, str, str, Path]]:
    out = []
    for f in sorted(in_dir.glob("e2_*.csv")):
        tag = f.stem[3:]
        model, host = tag.split("_", 1)
        out.append((tag, model, host, f))
    for f in sorted(in_dir.glob("e8_refscan_*.csv")):
        host = f.stem[len("e8_refscan_"):]
        out.append((f"refscan_{host}", "mamba1-ref", host, f))
    if tags:
        out = [x for x in out if x[0] in tags]
    return out


def _load(item) -> Table | None:
    tag, model, host, path = item
    df = load_e2([path])
    if df.empty:
        return None
    if "regimes" in df and (df["regimes"] == "e4_key").any():
        df = df[df["regimes"] != "e4_key"].reset_index(drop=True)
    if df["fold"].nunique() < 2:
        return None
    return Table(df, tag, model, host, path.name)


def _worker(payload):
    item, args = payload
    out = args.out_dir / f"e3b_information_{item[0]}.json"
    if out.exists() and not args.force:
        return item[0], "skipped (exists)", out
    t = _load(item)
    if t is None:
        return item[0], "skipped (no data)", None
    res = analyze_tag(t, args)
    write_json(out, res)
    return item[0], f"done in {res['timing_s']['total']:.0f}s", out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--tags", default=None, help="comma list of tags (default: every e2_*.csv and e8_refscan_*.csv)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--n-cmi-perm", type=int, default=1000)
    p.add_argument("--n-cv-repeats", type=int, default=50)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    p.add_argument("--skip-gbt", action="store_true")
    p.add_argument("--force", action="store_true", help="recompute tags whose output already exists")
    p.add_argument("--no-summary", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.out_dir = args.out_dir or args.in_dir
    items = discover(args.in_dir, args.tags.split(",") if args.tags else None)
    print(f"E3b: {len(items)} tables, {args.workers} workers, n_boot={args.n_boot}, n_perm={args.n_perm}, "
          f"n_cmi_perm={args.n_cmi_perm}, n_cv_repeats={args.n_cv_repeats}", flush=True)
    payloads = [(it, args) for it in items]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for tag, status, _ in pool.imap_unordered(_worker, payloads):
                print(f"  {tag}: {status}", flush=True)
    else:
        for pl in payloads:
            tag, status, _ = _worker(pl)
            print(f"  {tag}: {status}", flush=True)
    if args.no_summary:
        return
    rows, full = [], {}
    for tag, *_ in items:
        f = args.out_dir / f"e3b_information_{tag}.json"
        if f.exists():
            res = json.loads(f.read_text())
            rows.append(summary_row(res))
            full[tag] = rows[-1]
    write_json(args.out_dir / "e3b_information_summary.json", {"experiment": "E3b-summary", "rows": full})
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with open(args.out_dir / "e3b_information_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"summary: {args.out_dir / 'e3b_information_summary.csv'}")


if __name__ == "__main__":
    main()
