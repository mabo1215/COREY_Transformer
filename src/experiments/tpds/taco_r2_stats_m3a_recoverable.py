"""taco-r2 item M3a: recoverable gains of a held-out shape table under CUDA-graph replay, with prompt-cluster intervals.

CPU only, deterministic; reads the stored three-round CUDA-graph tables of the captured calls (Table 7: SSD 370M,
SSD 2.7B, Gated DeltaNet 1.4B and the Triton Mamba-1 scan, RTX 3090 and RTX PRO 6000; round-median view).

Definitions (identical to Table 7 and ``taco_report_numbers.graph_set``)
---------------------------------------------------------------------------
default            library default chunk (SSD 256, Gated DeltaNet 64); the Triton scan has none, so its "default"
                   is the best single chunk chosen in sample (as Table 7)
exact table        exact-shape (L, batch) table chosen on the held-out capture-time prompt folds with the
                   (length bucket, batch, alignment) and best-single-chunk fallbacks (``deployment_units.cv_exact_pick``)
shape tree         held-out leaf-cost tree on the shape features (``e22_graph_analysis.e3_predictions``)
in-sample table    exact-shape table chosen on the scored calls themselves (``deployment_units.insample_exact_pick``):
                   every key profiled on its own calls; an upper bound of what a tuned table delivers (it also
                   selects on timing noise)
covered calls      calls whose exact key occurs in the other capture-time folds; the remaining calls are served by
                   the bucket or best-single-chunk fallback of the held-out table, i.e. by an untuned entry
LW share           100 * sum(excess latency over the per-call oracle) / sum(oracle latency): "% of scan time" (Table 7)
recoverable        LW(default) - LW(table): the part of the default's latency-weighted loss that a held-out table
                   delivers; also per call (percentage points of mean regret), in microseconds per call, in ms per
                   forward pass (x scan layers) and as a share of the default's own scan time
                   (100 * sum(excess_default - excess_table) / sum(latency_default))

Intervals
---------
percentile         2000 prompt-cluster bootstrap replicates of the fixed held-out choices; the generator of
                   ``taco_report_numbers.boot`` (same label and subset), so the stored intervals of Table 7's
                   default LW are reproduced exactly (checked)
BCa                bias-corrected and accelerated, acceleration from the leave-one-cluster-out jackknife
refit in bootstrap the exact table is re-chosen inside every replicate: each resampled cluster copy is assigned to
                   one of five folds at random, the table is chosen on the other folds' weighted calls (same
                   estimator and fallbacks) and scored on the copy's own fold; 1000 replicates. This interval
                   includes the training variability of the table.
repeated CV        50 random assignments of the prompt clusters to five folds (no resampling): spread of the
                   recoverable share due to the fold assignment alone (exact table and shape tree)

Output: ``taco_r2_stats_m3a.json``.
Usage:  PYTHONPATH=. python3 -m src.experiments.tpds.taco_r2_stats_m3a_recoverable [--n-boot 2000] [--n-refit 1000]
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

from src.experiments.tpds import deployment_units as DU  # noqa: E402
from src.experiments.tpds import e22_graph_analysis as E22  # noqa: E402
from src.experiments.tpds import e3_predictors as E3  # noqa: E402
from src.experiments.tpds import e3b_information as E3B  # noqa: E402
from src.experiments.tpds import taco_r2_stats_common as C  # noqa: E402
from src.experiments.tpds.common import RESULTS_DIR  # noqa: E402

N_FOLDS = 5
# R1's recomputation (taco-r2 review, M3a table): recoverable LW [95% CI]
R1_RECOVERABLE = {"ssd370m_graph": (2.79, 1.51, 5.30), "ssd370m_graph_pro6000": (1.72, 0.89, 3.57),
                  "ssd2p7b_graph": (2.17, 1.15, 4.61), "ssd2p7b_graph_pro6000": (5.39, 2.77, 7.63),
                  "gdn1p4b_graph": (0.45, -0.90, 1.78), "gdn1p4b_graph_pro6000": (2.94, 1.88, 4.93)}
# Table 7 default LW as printed
TABLE7_DEFAULT_LW = {"ssd370m_graph": "2.8", "ssd370m_graph_pro6000": "1.7", "ssd2p7b_graph": "2.2",
                     "ssd2p7b_graph_pro6000": "5.9", "gdn1p4b_graph": "1.0", "gdn1p4b_graph_pro6000": "3.2",
                     "m1_370m_graph": "17.9", "m1_370m_graph_pro6000": "32.3"}


# --------------------------------------------------------------------------- weighted exact-table cross-validation
class TableCV:
    """Vectorized exact-shape table with bucket and global fallbacks, chosen on weighted training calls.

    Same estimator as ``deployment_units.cv_exact_pick``: entry = argmin of the (weighted) mean latency of the
    training calls of the exact (L, batch) key, else of the (length bucket, batch, alignment) key, else the chunk
    with the lowest summed normalized latency.
    """

    def __init__(self, df: pd.DataFrame, lat: np.ndarray):
        self.lat = np.where(np.isfinite(lat), lat, 1e30)
        self.orc = lat.min(1)
        self.norm = self.lat / self.orc[:, None]
        aligned = (df["L"].to_numpy() % 16 == 0).astype(int)
        self.ex = df.groupby(["L", "batch"], sort=True).ngroup().to_numpy()
        self.bu = pd.DataFrame({"lb": df["L_bucket"].to_numpy(), "b": df["batch"].to_numpy(), "a": aligned}) \
            .groupby(["lb", "b", "a"], sort=True).ngroup().to_numpy()
        self.n, self.C = self.lat.shape

    def _pick(self, tw: np.ndarray, aw: np.ndarray | None = None) -> np.ndarray:
        """Per-call chunk index of the table trained with per-call weights ``tw`` (>= 0).

        With ``aw`` (weights of all calls, training and test), a key without training calls is served by its own
        entry chosen on ``aw`` (the key is profiled when it first appears) instead of the bucket / global fallback.
        """
        out = np.full(self.n, int(np.argmin((tw[:, None] * self.norm).sum(0))))
        have_b = np.zeros(self.bu.max() + 1, bool)
        sb = np.zeros((self.bu.max() + 1, self.C))
        np.add.at(sb, self.bu, tw[:, None] * self.lat)
        np.add.at(have_b, self.bu, tw > 0)
        pb = sb.argmin(1)
        m = have_b[self.bu]
        out[m] = pb[self.bu[m]]
        have_e = np.zeros(self.ex.max() + 1, bool)
        se = np.zeros((self.ex.max() + 1, self.C))
        np.add.at(se, self.ex, tw[:, None] * self.lat)
        np.add.at(have_e, self.ex, tw > 0)
        pe = se.argmin(1)
        m = have_e[self.ex]
        out[m] = pe[self.ex[m]]
        if aw is not None:
            sa = np.zeros((self.ex.max() + 1, self.C))
            np.add.at(sa, self.ex, aw[:, None] * self.lat)
            pa = sa.argmin(1)
            out[~m] = pa[self.ex[~m]]
        return out

    def excess(self, fold_w: np.ndarray, profile_unseen: bool = False) -> np.ndarray:
        """Weighted excess latency per call: fold_w (calls x folds) = weight of each call in each test fold.

        ``profile_unseen``: keys without training calls take their own entry (profiled on first sight)."""
        tot = fold_w.sum(1)
        exc = np.zeros(self.n)
        for f in range(fold_w.shape[1]):
            te = fold_w[:, f]
            if not te.any():
                continue
            p = self._pick(tot - te, tot if profile_unseen else None)
            exc += te * (self.lat[np.arange(self.n), p] - self.orc)
        return exc

    def check_against(self, df: pd.DataFrame, lat: np.ndarray) -> float:
        """Share of calls on which the capture-time folds give the choice of ``cv_exact_pick``."""
        folds = df["fold"].to_numpy()
        ref = DU.cv_exact_pick(df, lat)
        mine = np.full(self.n, -1)
        for f in np.unique(folds):
            te = folds == f
            p = self._pick((~te).astype(float))
            mine[te] = p[te]
        return float((mine == ref).mean())


def covered_mask(df: pd.DataFrame) -> np.ndarray:
    """Calls whose exact (L, batch) key occurs in the other capture-time folds (``cv_exact_headroom`` "covered")."""
    folds = df["fold"].to_numpy()
    key = list(zip(df["L"].to_numpy(), df["batch"].to_numpy()))
    seen: dict = {}
    for k, f in zip(key, folds):
        seen.setdefault(k, set()).add(int(f))
    return np.array([len(seen[k] - {int(f)}) > 0 for k, f in zip(key, folds)])


def shape_tree_pick(df: pd.DataFrame, chunks: list[int], lat: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Held-out shape tree for an arbitrary fold assignment (the loop of ``e22_graph_analysis.e3_predictions``)."""
    lat = np.where(np.isfinite(lat), lat, np.inf)
    oracle_idx = lat.argmin(1)
    norm = np.minimum(lat / lat.min(1, keepdims=True), 1e6)
    cols, depth = E3.FEATURE_SETS["shape"]
    out = np.full(len(df), -1)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        m = E3.LeafCostTree(depth).fit(df.loc[tr, cols].to_numpy(float), oracle_idx[tr], norm[tr])
        out[te] = m.predict_idx(df.loc[te, cols].to_numpy(float))
    return out


# --------------------------------------------------------------------------- one table
def stats_block(t, picks: dict, clusters: np.ndarray, rng, n_boot: int, sel: np.ndarray | None) -> dict:
    A = {"orc": t.orc}
    for k, p in picks.items():
        A.update(C.policy_arrays(t.lat, t.orc, p, k))
        A[f"{k}_lat"] = t.lat[np.arange(len(t.orc)), p]
    cb = C.ClusterBoot(clusters, n_boot, rng, sel)
    L = t.layers
    g = {
        "default_lw_pct": C.lw_g("default_exc"),
        "table_lw_pct": C.lw_g("table_exc"),
        "tree_lw_pct": C.lw_g("tree_exc"),
        "recoverable_lw_table_pp": lambda T, n: 100 * (T["default_exc"] - T["table_exc"]) / T["orc"],
        "recoverable_lw_tree_pp": lambda T, n: 100 * (T["default_exc"] - T["tree_exc"]) / T["orc"],
        "table_insample_lw_pct": C.lw_g("table_insample_exc"),
        "recoverable_lw_table_insample_pp": lambda T, n: 100 * (T["default_exc"] - T["table_insample_exc"]) / T["orc"],
        "recoverable_share_of_default_scan_time_pct": lambda T, n: 100 * (T["default_exc"] - T["table_exc"]) / T["default_lat"],
        "default_mean_regret_pct": C.mean_g("default_reg", 100),
        "table_mean_regret_pct": C.mean_g("table_reg", 100),
        "recoverable_mean_regret_table_pp": lambda T, n: 100 * (T["default_reg"] - T["table_reg"]) / n,
        "recoverable_us_per_call_table": lambda T, n: 1000 * (T["default_exc"] - T["table_exc"]) / n,
        "recoverable_ms_per_forward_pass_table": lambda T, n: L * (T["default_exc"] - T["table_exc"]) / n,
        "default_ms_per_forward_pass": lambda T, n: L * T["default_exc"] / n,
        "oracle_scan_ms_per_forward_pass": lambda T, n: L * T["orc"] / n,
    }
    out = {"n_calls": int(len(cb.idx)), "n_prompt_clusters": int(cb.K)}
    for name, fn in g.items():
        out[name] = C.summarize(cb.evaluate(A, fn))
    return out


def refit_bootstrap(tcv: TableCV, default_exc: np.ndarray, clusters: np.ndarray, n_rep: int, rng, sel=None) -> dict:
    """Recoverable LW with the exact table re-chosen inside every cluster-bootstrap replicate (random folds)."""
    idx = np.arange(len(default_exc)) if sel is None else np.flatnonzero(sel)
    _, cl = np.unique(clusters[idx], return_inverse=True)
    K = int(cl.max()) + 1
    full_cl = np.full(len(default_exc), -1)
    full_cl[idx] = cl
    reps, defs, tabs, reps_p = [], [], [], []
    for _ in range(n_rep):
        m = rng.multinomial(K, np.full(K, 1.0 / K))
        nf = np.stack([rng.multinomial(int(mk), np.full(N_FOLDS, 1.0 / N_FOLDS)) for mk in m])    # K x folds
        fw = np.zeros((len(default_exc), N_FOLDS))
        fw[idx] = nf[cl]
        w = fw.sum(1)
        xe = tcv.excess(fw)
        xp = tcv.excess(fw, profile_unseen=True)
        so = (w * tcv.orc).sum()
        defs.append(100 * (w * default_exc).sum() / so)
        tabs.append(100 * xe.sum() / so)
        reps.append(defs[-1] - tabs[-1])
        reps_p.append(defs[-1] - 100 * xp.sum() / so)
    reps = np.array(reps)
    reps_p = np.array(reps_p)
    return {"n_replicates": n_rep, "recoverable_lw_table_pp_ci95": C.pct_ci(reps),
            "recoverable_lw_table_profile_unseen_pp_ci95": C.pct_ci(reps_p),
            "recoverable_lw_table_profile_unseen_pp_median": float(np.median(reps_p)),
            "note": ("random folds per resampled cluster copy; keys without training calls fall back to the bucket / "
                     "best-single-chunk entry (as the paper's held-out table) or, in the profile_unseen variant, take "
                     "their own entry chosen on all copies (a key is profiled when first seen)"),
            "recoverable_lw_table_pp_median": float(np.median(reps)),
            "table_lw_pct_ci95": C.pct_ci(np.array(tabs)), "default_lw_pct_ci95": C.pct_ci(np.array(defs)),
            "share_replicates_le_0": float(np.mean(reps <= 0))}


def repeated_cv(t, tcv: TableCV, default_exc: np.ndarray, clusters: np.ndarray, n_rep: int, rng, with_tree: bool) -> dict:
    K = int(clusters.max()) + 1
    tab, tree = [], []
    so = t.orc.sum()
    for _ in range(n_rep):
        order = rng.permutation(K)
        fold_of_cluster = np.empty(K, int)
        fold_of_cluster[order] = np.arange(K) % N_FOLDS
        folds = fold_of_cluster[clusters]
        fw = np.zeros((len(t.orc), N_FOLDS))
        fw[np.arange(len(t.orc)), folds] = 1.0
        tab.append(100 * (default_exc.sum() - tcv.excess(fw).sum()) / so)
        if with_tree:
            p = shape_tree_pick(t.df, t.chunks, t.lat, folds)
            tree.append(100 * (default_exc.sum() - (t.lat[np.arange(len(t.orc)), p] - t.orc).sum()) / so)
    out = {"n_repeats": n_rep, "unit": "prompt clusters assigned at random to five folds",
           "recoverable_lw_table_pp": {"mean": float(np.mean(tab)), "min": float(np.min(tab)), "max": float(np.max(tab)),
                                       "p2.5_p97.5": C.pct_ci(np.array(tab))}}
    if with_tree:
        out["recoverable_lw_tree_pp"] = {"mean": float(np.mean(tree)), "min": float(np.min(tree)),
                                         "max": float(np.max(tree)), "p2.5_p97.5": C.pct_ci(np.array(tree))}
    return out


def drivers(t, pick: np.ndarray, top: int = 8) -> list[dict]:
    """Calls with the largest excess latency of a policy (diagnostic of a wide interval)."""
    exc = t.lat[np.arange(len(t.orc)), pick] - t.orc
    order = np.argsort(-exc)[:top]
    tot = exc.sum()
    return [{"L": int(t.L[i]), "batch": int(t.batch[i]), "fold": int(t.df["fold"].iat[i]), "layer": int(t.df["layer"].iat[i]),
             "chosen_chunk": int(t.chunks[pick[i]]), "oracle_chunk": int(t.chunks[t.oidx[i]]),
             "excess_us": float(1000 * exc[i]), "oracle_us": float(1000 * t.orc[i]),
             "share_of_policy_total_excess": float(exc[i] / tot) if tot > 0 else None} for i in order]


def analyze(nm: str, args, checks: list) -> dict:
    gpu, kernel = C.GRAPH_TABLES[nm]
    t = C.graph_table(nm, args.in_dir)
    clusters = E3B.prompt_clusters(t.df)[0]
    picks = {"default": np.full(len(t.df), t.default_idx()),
             "table": DU.cv_exact_pick(t.df, t.lat),
             "tree": E22.e3_predictions(t.df, t.chunks)["shape"],
             "table_insample": DU.insample_exact_pick(t.df, np.where(np.isfinite(t.lat), t.lat, 1e30))}
    cov = covered_mask(t.df)
    tcv = TableCV(t.df, t.lat)
    agree = tcv.check_against(t.df, t.lat)
    checks.append({"check": f"{nm}: vectorized exact table == deployment_units.cv_exact_pick (capture-time folds)",
                   "agreement": agree, "match": agree == 1.0})
    rec = {"gpu": gpu, "kernel": kernel, "source": t.csv, "scan_layers": t.layers,
           "default_chunk": int(t.chunks[t.default_idx()]),
           "default_note": "library default" if t.default is not None else "best single chunk in sample (no library default)",
           "floor_us": 1000 * t.floor_ms, "share_above_floor": float(t.above.mean()),
           "cluster_sizes": C.cluster_sizes(clusters, t.df), "subsets": {}}
    rec["n_covered_calls"] = int(cov.sum())
    subsets = {"all": None, "above_floor": t.above, "below_floor": ~t.above, "covered": cov}
    for b in sorted(np.unique(t.batch)):
        subsets[f"batch{int(b)}"] = t.batch == b
    for sub, sel in subsets.items():
        rng = C.make_rng(nm, sub)        # same generator as taco_report_numbers.graph_set for all / above / below
        rec["subsets"][sub] = stats_block(t, picks, clusters, rng, args.n_boot, sel)
    d_exc = t.lat[np.arange(len(t.orc)), picks["default"]] - t.orc
    rec["refit_in_bootstrap_all_calls"] = refit_bootstrap(tcv, d_exc, clusters, args.n_refit, C.make_rng("m3a_refit", nm))
    rec["repeated_cv_all_calls"] = repeated_cv(t, tcv, d_exc, clusters, args.n_cv, C.make_rng("m3a_rcv", nm),
                                               with_tree=args.tree_cv)
    rec["largest_excess_calls_table"] = drivers(t, picks["table"])
    t_exc = t.lat[np.arange(len(t.orc)), picks["table"]] - t.orc
    rec["table_excess_uncovered_calls"] = {
        "n_uncovered_calls": int((~cov).sum()),
        "share_of_table_excess": float(t_exc[~cov].sum() / t_exc.sum()) if t_exc.sum() > 0 else None,
        "table_lw_pct_excluding_uncovered_excess": float(100 * t_exc[cov].sum() / t.orc.sum()),
        "note": "uncovered calls have no exact-key entry in the training folds and are served by the bucket or global "
                "fallback, i.e. by an untuned entry"}
    # reproduction checks
    a = rec["subsets"]["all"]
    stored = (C.load_json(args.in_dir / ("taco_e26_analysis_pro6000.json" if nm.endswith("pro6000") else
                                         "taco_report_numbers.json")) or {})
    tabs = stored.get("tables") if nm.endswith("pro6000") else stored.get("M4b_latency_weighted", {}).get("a_cuda_graph")
    if tabs and nm in tabs:
        pol = tabs[nm]["subsets"]["all"]["policies"]
        for mine, theirs in (("default_lw_pct", "default"), ("table_lw_pct", "exact_table_held_out"),
                             ("tree_lw_pct", "shape_tree_held_out")):
            ok = abs(a[mine]["point"] - pol[theirs]["lw_share_of_oracle_pct"]) < 1e-9
            checks.append({"check": f"{nm}: {mine} point == stored {theirs}.lw_share_of_oracle_pct", "computed": a[mine]["point"],
                           "stored": pol[theirs]["lw_share_of_oracle_pct"], "match": ok})
        ci = pol["default"]["ci95"]["lw_share_of_oracle_pct"]
        ok = np.allclose(a["default_lw_pct"]["ci95_percentile"], ci, atol=1e-9)
        checks.append({"check": f"{nm}: default LW percentile interval == stored (same replicates)",
                       "computed": a["default_lw_pct"]["ci95_percentile"], "stored": ci, "match": bool(ok)})
    printed = TABLE7_DEFAULT_LW[nm]
    ok = E22._matches(a["default_lw_pct"]["point"], printed)
    checks.append({"check": f"{nm}: default LW vs Table 7", "computed": a["default_lw_pct"]["point"], "printed": printed,
                   "match": bool(ok)})
    if nm in R1_RECOVERABLE:
        p, lo, hi = R1_RECOVERABLE[nm]
        r = a["recoverable_lw_table_pp"]
        checks.append({"check": f"{nm}: recoverable LW vs R1 {p} [{lo}, {hi}]", "computed_point": round(r["point"], 3),
                       "computed_ci95_percentile": [round(x, 3) for x in r["ci95_percentile"]],
                       "point_match_2dp": round(r["point"], 2) == p,
                       "note": "R1's interval uses another bootstrap seed; endpoints differ by Monte Carlo error only"})
    return rec


def headline(res: dict) -> dict:
    out = {}
    for nm, r in res.items():
        a = r["subsets"]["all"]
        out[nm] = {"gpu": r["gpu"], "kernel": r["kernel"],
                   "default_lw_pct": C.fmt_ci(a["default_lw_pct"]),
                   "table_lw_pct": C.fmt_ci(a["table_lw_pct"], 3),
                   "recoverable_lw_table_pp": C.fmt_ci(a["recoverable_lw_table_pp"]),
                   "recoverable_lw_table_pp_bca": C.fmt_ci(a["recoverable_lw_table_pp"], kind="ci95_bca"),
                   "recoverable_lw_table_pp_refit_ci95": [round(x, 2) for x in
                                                          r["refit_in_bootstrap_all_calls"]["recoverable_lw_table_pp_ci95"]],
                   "recoverable_lw_tree_pp": C.fmt_ci(a["recoverable_lw_tree_pp"]),
                   "recoverable_lw_table_covered_calls_pp": C.fmt_ci(r["subsets"]["covered"]["recoverable_lw_table_pp"]),
                   "recoverable_lw_table_insample_pp": C.fmt_ci(a["recoverable_lw_table_insample_pp"]),
                   "recoverable_lw_table_profile_unseen_refit_ci95": [round(x, 2) for x in r["refit_in_bootstrap_all_calls"][
                       "recoverable_lw_table_profile_unseen_pp_ci95"]],
                   "recoverable_mean_regret_table_pp": C.fmt_ci(a["recoverable_mean_regret_table_pp"]),
                   "recoverable_us_per_call_table": C.fmt_ci(a["recoverable_us_per_call_table"]),
                   "recoverable_ms_per_forward_pass_table": C.fmt_ci(a["recoverable_ms_per_forward_pass_table"], 3),
                   "oracle_scan_ms_per_forward_pass": round(a["oracle_scan_ms_per_forward_pass"]["point"], 2),
                   "recoverable_by_batch_lw_pp": {b: C.fmt_ci(r["subsets"][f"batch{b}"]["recoverable_lw_table_pp"])
                                                  for b in (1, 4, 16) if f"batch{b}" in r["subsets"]},
                   "recoverable_above_floor_lw_pp": C.fmt_ci(r["subsets"]["above_floor"]["recoverable_lw_table_pp"]),
                   "share_replicates_le_0": a["recoverable_lw_table_pp"]["share_replicates_le_0"]}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--n-boot", type=int, default=C.N_BOOT)
    p.add_argument("--n-refit", type=int, default=1000)
    p.add_argument("--n-cv", type=int, default=50)
    p.add_argument("--no-tree-cv", dest="tree_cv", action="store_false")
    args = p.parse_args(argv)
    t0 = time.time()
    names = C.register_graph_tables(args.in_dir)
    checks, res = [], {}
    for nm in names:
        res[nm] = analyze(nm, args, checks)
        print(f"{nm}: done ({time.time() - t0:.0f} s)", flush=True)
    out = {"item": "M3a", "meta": C.meta(__name__, args), "definitions": __doc__, "headline": headline(res),
           "tables": res, "checks": checks, "n_checks": len(checks),
           "n_checks_failed": sum(1 for c in checks if c.get("match") is False), "runtime_s": round(time.time() - t0, 1)}
    path = C.write_result("m3a", out, args.in_dir)
    for nm, h in out["headline"].items():
        print(f"{h['kernel']:22s} {h['gpu']:13s} default LW {h['default_lw_pct']:24s} table LW {h['table_lw_pct']:26s} "
              f"recoverable {h['recoverable_lw_table_pp']:24s} BCa {h['recoverable_lw_table_pp_bca']:24s} "
              f"refit {h['recoverable_lw_table_pp_refit_ci95']}")
    for c in checks:
        if c.get("match") is False:
            print("CHECK FAILED:", c)
    print(f"wrote {path} ({out['n_checks']} checks, {out['n_checks_failed']} failed)")


if __name__ == "__main__":
    main()
