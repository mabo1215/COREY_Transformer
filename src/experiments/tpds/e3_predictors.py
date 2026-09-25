"""E3 - does any per-call feature predict the oracle chunk beyond sequence length and shape? (item C3)

CPU analysis of the E2 per-call tables. Predictors of the oracle chunk:
  global_static   best single chunk on the training folds
  entropy_rule    Eq. (2) of the manuscript, C in [32, 512] (no training)
  L_only          decision tree on L (depth <= 3)
  shape           tree on (L, batch, width, layer)
  entropy_only    tree on entropy
  moments         tree on (entropy, variance, kurtosis, sparsity, mean |x|)
  shape+entropy   tree on shape features plus entropy
  all             tree on shape, moments and delta entropy
Each tree leaf is assigned the chunk with the lowest total normalized latency
among its training calls (regret-optimal leaf labels). Evaluation is grouped
5-fold cross-validation, where the folds are the prompt folds assigned at
capture time (a prompt never appears in both training and test folds).
Metrics: oracle accuracy, tie-aware accuracy (prediction statistically tied
with the oracle), mean / p95 regret relative to the per-call oracle, and
the conditional mutual information I(entropy; oracle | L bucket [, batch])
with a within-stratum permutation baseline.

Output: ``e3_<model>_<host>.json``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.tpds.analysis import (chunk_columns, conditional_mutual_information, latency_matrix, load_e2,
                                           regret_stats)
from src.experiments.tpds.common import RESULTS_DIR, env_metadata, write_json

try:
    from sklearn.tree import DecisionTreeClassifier  # type: ignore

    HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    HAVE_SKLEARN = False


SHAPE = ["L", "batch", "width", "layer"]
MOMENTS = ["entropy_bits", "variance", "kurtosis", "sparsity", "mean_abs"]
FEATURE_SETS = {
    "L_only": (["L"], 3),
    "shape": (SHAPE, 4),
    "entropy_only": (["entropy_bits"], 4),
    "moments": (MOMENTS, 4),
    "shape+entropy": (SHAPE + ["entropy_bits"], 4),
    "all": (SHAPE + MOMENTS + ["delta_entropy_bits"], 4),
}


# --------------------------------------------------------------------------- minimal CART fallback
class _TinyTree:
    """Depth-limited Gini CART; used only when scikit-learn is unavailable."""

    def __init__(self, max_depth=3, min_leaf=5):
        self.max_depth, self.min_leaf = max_depth, min_leaf

    def fit(self, X, y):
        self.nodes = []
        self._grow(np.asarray(X, float), np.asarray(y), np.arange(len(y)), 0)
        return self

    def _gini(self, y):
        _, c = np.unique(y, return_counts=True)
        p = c / c.sum()
        return 1 - (p * p).sum()

    def _grow(self, X, y, idx, depth):
        nid = len(self.nodes)
        self.nodes.append(None)
        best = None
        if depth < self.max_depth and len(idx) >= 2 * self.min_leaf and len(np.unique(y[idx])) > 1:
            g0 = self._gini(y[idx])
            for f in range(X.shape[1]):
                vals = np.unique(X[idx, f])
                for thr in (vals[:-1] + vals[1:]) / 2 if len(vals) > 1 else []:
                    left = idx[X[idx, f] <= thr]
                    right = idx[X[idx, f] > thr]
                    if len(left) < self.min_leaf or len(right) < self.min_leaf:
                        continue
                    g = (len(left) * self._gini(y[left]) + len(right) * self._gini(y[right])) / len(idx)
                    if g < g0 - 1e-12 and (best is None or g < best[0]):
                        best = (g, f, thr, left, right)
        if best is None:
            self.nodes[nid] = ("leaf", nid)
            return nid
        _, f, thr, left, right = best
        li = self._grow(X, y, left, depth + 1)
        ri = self._grow(X, y, right, depth + 1)
        self.nodes[nid] = ("split", f, thr, li, ri)
        return nid

    def apply(self, X):
        X = np.asarray(X, float)
        out = np.empty(len(X), dtype=int)
        for i, x in enumerate(X):
            n = 0
            while self.nodes[n][0] == "split":
                _, f, thr, li, ri = self.nodes[n]
                n = li if x[f] <= thr else ri
            out[i] = n
        return out


def _make_tree(depth):
    if HAVE_SKLEARN:
        return DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5, random_state=0)
    return _TinyTree(max_depth=depth, min_leaf=5)


class LeafCostTree:
    """Tree on the oracle label whose leaves are relabeled with the regret-optimal chunk."""

    def __init__(self, depth):
        self.tree = _make_tree(depth)

    def fit(self, X, y, norm_lat):
        self.tree.fit(X, y)
        leaves = self.tree.apply(X)
        self.global_best = int(np.argmin(norm_lat.sum(0)))
        self.leaf_choice = {int(l): int(np.argmin(norm_lat[leaves == l].sum(0))) for l in np.unique(leaves)}
        return self

    def predict_idx(self, X):
        return np.array([self.leaf_choice.get(int(l), self.global_best) for l in self.tree.apply(X)])


# --------------------------------------------------------------------------- evaluation
def evaluate(df: pd.DataFrame, chunks: list[int], keep_per_call: bool = False) -> dict:
    lat = latency_matrix(df, chunks)
    oracle_idx = lat.argmin(1)
    norm = lat / lat.min(1, keepdims=True)
    tied = [set(int(c) for c in str(t).split(";") if c) for t in df["tied_chunks"].fillna("")]
    folds = df["fold"].to_numpy()
    uniq = np.unique(folds)
    preds = {name: np.full(len(df), -1) for name in ["global_static", "entropy_rule", *FEATURE_SETS]}
    rule = df["entropy_rule_chunk"].clip(min(chunks), max(chunks)).to_numpy()
    preds["entropy_rule"] = np.array([chunks.index(int(c)) if int(c) in chunks else
                                      int(np.argmin([abs(np.log2(c) - np.log2(x)) for x in chunks])) for c in rule])
    for f in uniq:
        tr, te = folds != f, folds == f
        if tr.sum() == 0 or te.sum() == 0:
            continue
        preds["global_static"][te] = int(np.argmin(norm[tr].sum(0)))
        for name, (cols, depth) in FEATURE_SETS.items():
            m = LeafCostTree(depth).fit(df.loc[tr, cols].to_numpy(float), oracle_idx[tr], norm[tr])
            preds[name][te] = m.predict_idx(df.loc[te, cols].to_numpy(float))
    out = {}
    for name, p in preds.items():
        ok = p >= 0
        if not ok.any():
            continue
        reg = norm[ok, :][np.arange(ok.sum()), p[ok]] - 1.0
        chosen = np.array(chunks)[p[ok]]
        st = regret_stats(reg)
        st["accuracy"] = float((p[ok] == oracle_idx[ok]).mean())
        st["tie_aware_accuracy"] = float(np.mean([c in t for c, t in zip(chosen, np.array(tied, dtype=object)[ok])]))
        if keep_per_call:
            st["per_call_regret"] = [round(float(v), 5) for v in reg]
        st["chosen_distribution"] = {int(k): int(v) for k, v in pd.Series(chosen).value_counts().items()}
        out[name] = st
    return out


def analyze(df: pd.DataFrame) -> dict:
    chunks = chunk_columns(df)
    res = {"n_calls": int(len(df)), "chunks": chunks, "n_folds": int(df["fold"].nunique()),
           "oracle_distribution": {int(k): int(v) for k, v in df["oracle_chunk"].value_counts().items()},
           "tied_with_oracle_mean_count": float(df["tied_chunks"].fillna("").map(lambda s: len([c for c in str(s).split(';') if c])).mean())}
    res["predictors_all_batches"] = evaluate(df, chunks, keep_per_call=True)
    res["predictors_by_batch"] = {int(b): evaluate(g, chunks) for b, g in df.groupby("batch") if g["fold"].nunique() > 1}
    Lb = np.log2(df["L_bucket"].to_numpy()).astype(int)
    res["mi_entropy_oracle_given_Lbucket"] = conditional_mutual_information(
        df["entropy_bits"].to_numpy(), df["oracle_chunk"].to_numpy(), Lb)
    res["mi_entropy_oracle_given_Lbucket_batch"] = conditional_mutual_information(
        df["entropy_bits"].to_numpy(), df["oracle_chunk"].to_numpy(), Lb * 100 + df["batch"].to_numpy())
    res["mi_logL_oracle_given_batch"] = conditional_mutual_information(
        np.log2(df["L"].to_numpy()), df["oracle_chunk"].to_numpy(), df["batch"].to_numpy())
    return res


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--models", default="mamba1,mamba2")
    p.add_argument("--hosts", default=None, help="comma list; default: every e2_<model>_<host>.csv found")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = args.out_dir or args.in_dir
    for model in args.models.split(","):
        files = sorted(args.in_dir.glob(f"e2_{model}_*.csv"))
        for f in files:
            host = f.stem[len(f"e2_{model}_"):]
            if args.hosts and host not in args.hosts.split(","):
                continue
            df = load_e2([f])
            if df.empty or df["fold"].nunique() < 2:
                print(f"skip {f}: not enough data/folds")
                continue
            res = {"experiment": "E3", "model": model, "host": host, "source": f.name,
                   "sklearn": HAVE_SKLEARN, "meta": env_metadata(), **analyze(df)}
            path = out_dir / f"e3_{model}_{host}.json"
            write_json(path, res)
            pa = res["predictors_all_batches"]
            print(f"{path.name}: n={res['n_calls']} " + ", ".join(
                f"{k}: acc {v['accuracy']:.2f} regret {v['mean_regret_pct']:.1f}%/{v['p95_regret_pct']:.1f}%"
                for k, v in pa.items()))
            print(f"  CMI(entropy; oracle | L bucket) = {res['mi_entropy_oracle_given_Lbucket']['cmi_bits']:.3f} bits "
                  f"(perm baseline {res['mi_entropy_oracle_given_Lbucket']['cmi_perm_baseline_bits']:.3f})")


if __name__ == "__main__":
    main()
