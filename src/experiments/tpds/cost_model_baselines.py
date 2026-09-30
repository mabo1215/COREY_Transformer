"""T1 baselines - what the Mamba-1 cost model adds over simpler selectors (items M4a and N3).

CPU-only; reads existing result files and writes ``cost_model_baselines_<host>.json``.
Every selector below picks one chunk per captured call (the 2360 E2 calls per GPU) and is
scored exactly as in E3 and in the cost-model validation (``e2_eval``): relative regret
lat(choice) / lat(per-call oracle) - 1 (mean, p95), and the tied share, i.e. the share of
calls whose chosen chunk lies in the call's statistically tied set (95% CIs overlapping the
oracle's, as recorded at capture time).

Selectors that never see a captured-call timing (calibration data only: the E4 launch-space
sweep, the E4 static re-measurements and the E1 medians):
  cost_model              Eq. (5) with its eight fitted parameters (the stored fit; a refit
                          is checked to reproduce it).
  work_only               Eq. (5) restricted to its launch and work terms:
                              T = max(t0 + waves * ceil(L / C) * N * (tau_m + tau_e * e), bytes / bw)
                          i.e. launch floor t0, a per-tile state cost tau_m and a per-element scan
                          cost tau_e, with the grid and the per-thread element count e as compiled.
                          Shuffle (tau_s), spill (tau_q, tau_p), issue sharing (W0) and latency
                          hiding (W1) are removed (W0 = 64 and W1 = 256 make both factors equal 1
                          for every resident-warp count). Three free parameters, refit on the same
                          calibration points as the full model.
  work_only_no_reg_occupancy
                          as work_only, but the resident programs per SM (hence the number of
                          waves) follow the warp and block limits only, ignoring the register
                          count from ptxas (no compile-time occupancy fact).
  work_only_w1_eq_1       as work_only with the drop-one ablation's latency-hiding convention
                          (W1 = 1, tile latency divided by the resident warps); sensitivity only.
  work_term_model         the work-term form used for the multi-kernel pipelines,
                              T = (F^p + W^p)^(1/p),  W = B * L * (a + b * C) + B * ceil(L / C) * g,
                          which uses only (L, B, C): no compile facts, no occupancy, no launch
                          configuration. Fitted, as for Mamba-2, on the calibration timings taken
                          at the default launch configuration (E4 static re-measurements and E1
                          medians); a fit on all calibration points is reported as a variant.
  nearest_calibrated_key  a lookup table built from the calibration timings: every captured call
                          is served with the best chunk (lowest E4 static re-measurement at the
                          default launch configuration) of its nearest calibrated key. Distance is
                          lexicographic: first |log2 B - log2 B_k|, then |log2 L - log2 L_k|, then
                          |log2 d - log2 d_k|; a tie in log-length (a call halfway between two
                          calibrated lengths in log2) goes to the shorter key. Every captured
                          batch size is calibrated at every calibrated length, so ordering batch
                          before length or length before batch gives the same key (checked).
                          This is the "12-key table" of the break-even analysis (E5).
                          Variants: ties to the longer key; E4 medians instead of means; the chunk of
                          the autotuned launch configuration; E1 medians pooled in; and a log-length
                          interpolation of the normalized latency curves of the two calibrated
                          lengths that bracket the call (same batch), which needs no tie-break.

Selectors trained on captured-call timings (grouped 5-fold cross-validation over the prompt
folds, exactly as in E3; numbers reproduced from ``e3_predictors.evaluate``):
  best single chunk, entropy rule, length only, shape tree, entropy only, moments,
  shape + entropy, all features, and
  batch_only              a table keyed by batch size: for each fold, every batch value gets the
                          chunk with the lowest total normalized latency over its training calls
                          (the same regret-optimal label rule as the E3 tree leaves).

Ranking quality within a shape: for every call, Kendall tau-b between the selector's score
over the six chunks (predicted latency for the models, the nearest key's measured latencies for
the lookup table, the training-cell mean normalized latency for the trees and tables) and the
call's measured mean latencies. Reported per call (mean, median) and per exact shape (L, batch)
with the latencies averaged over the shape's calls (mean, median over shapes).

Also reported for every selector: the ratio-of-means headroom used by E5,
100 * (mean lat(choice) - mean lat(oracle)) / mean lat(choice), which is the 2.0% (RTX PRO 6000)
quoted for the nearest-key table; it differs from the mean per-call regret.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.tpds import cost_model as cm
from src.experiments.tpds import e3_predictors as e3
from src.experiments.tpds.analysis import chunk_columns, latency_matrix, load_e2
from src.experiments.tpds.common import RESULTS_DIR, read_jsonl, write_json

# Eq. (5) parameter indices (cost_model.PARAMS order)
_I = {k: i for i, k in enumerate(cm.PARAMS)}
WORK_ONLY_FIXED = {_I["tau_f"]: 0.0, _I["tau_d"]: 0.0, _I["tau_l"]: 0.0, _I["tau_x"]: 0.0,
                   _I["tau_s"]: 0.0, _I["tau_q"]: 0.0, _I["tau_p"]: 0.0,
                   _I["W0"]: 64.0,    # max(1, W / 64) = 1 for W <= 48 resident warps: no issue sharing
                   _I["W1"]: 256.0}   # min(1, 256 / W) = 1: the tile latency is never hidden


# --------------------------------------------------------------------------- metrics
def kendall_tau_b_rows(S: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Row-wise Kendall tau-b between score rows S and measured rows M (NaN if either row is constant)."""
    n = S.shape[1]
    j, k = np.triu_indices(n, 1)
    dS = np.sign(S[:, j] - S[:, k])
    dM = np.sign(M[:, j] - M[:, k])
    n0 = len(j)
    n1 = (dS == 0).sum(1)
    n2 = (dM == 0).sum(1)
    den = np.sqrt((n0 - n1) * (n0 - n2).astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (dS * dM).sum(1) / den, np.nan)


def _tau_summary(t: np.ndarray) -> dict:
    ok = ~np.isnan(t)
    if not ok.any():
        return {"mean": None, "median": None, "n": 0, "n_undefined": int(len(t))}
    return {"mean": float(t[ok].mean()), "median": float(np.median(t[ok])), "n": int(ok.sum()),
            "n_undefined": int((~ok).sum()), "share_tau_ge_0p6": float((t[ok] >= 0.6).mean())}


def evaluate(df: pd.DataFrame, chunks: list[int], choice: np.ndarray, scores: np.ndarray | None = None,
             pred_ms: np.ndarray | None = None) -> dict:
    """E3/e2_eval metrics for a choice, plus ranking agreement and the E5 ratio-of-means headroom."""
    lat = latency_matrix(df, chunks)
    st = cm.e2_eval(df, chunks, choice, pred_ms)
    reg = lat[np.arange(len(df)), choice] / lat.min(1) - 1
    tied = [set(int(c) for c in str(t).split(";") if c) for t in df["tied_chunks"].fillna("")]
    in_tie = np.array([int(chunks[c]) in t for c, t in zip(choice, tied)])
    for b, rec in st["by_batch"].items():
        sel = (df["batch"] == b).to_numpy()
        rec["tie_aware_accuracy"] = float(in_tie[sel].mean())
        rec["n"] = int(sel.sum())
    l_pol, l_orc = float(lat[np.arange(len(df)), choice].mean()), float(lat.min(1).mean())
    st["ratio_of_means_headroom_ms"] = l_pol - l_orc
    st["ratio_of_means_headroom_pct"] = 100 * (l_pol - l_orc) / l_pol
    st["policy_mean_latency_ms"] = l_pol
    st["oracle_mean_latency_ms"] = l_orc
    if scores is not None:
        assert np.array_equal(np.argmin(scores, 1), choice), "scores and choice disagree"
        st["kendall_tau_per_call"] = _tau_summary(kendall_tau_b_rows(scores, lat))
        keys = df[["L", "batch"]].to_numpy()
        shapes, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.ravel()
        S_sh = np.stack([scores[inv == s].mean(0) for s in range(len(shapes))])
        M_sh = np.stack([lat[inv == s].mean(0) for s in range(len(shapes))])
        st["kendall_tau_per_shape"] = _tau_summary(kendall_tau_b_rows(S_sh, M_sh))
        st["kendall_tau_per_call_by_batch"] = {
            int(b): _tau_summary(kendall_tau_b_rows(scores[(df["batch"] == b).to_numpy()],
                                                    lat[(df["batch"] == b).to_numpy()]))
            for b in sorted(df["batch"].unique())}
        st["predicted_argmin_in_tied_set"] = float(in_tie.mean())
    return st


def _brief(st: dict) -> dict:
    keep = ("mean_regret_pct", "p95_regret_pct", "max_regret_pct", "tie_aware_accuracy", "accuracy",
            "ratio_of_means_headroom_pct")
    out = {k: st[k] for k in keep if k in st}
    for k in ("kendall_tau_per_call", "kendall_tau_per_shape"):
        if k in st:
            out[k] = {"mean": st[k]["mean"], "median": st[k]["median"]}
    return out


# --------------------------------------------------------------------------- model-based selectors
def model_selectors(host: str, in_dir: Path, df: pd.DataFrame, chunks: list[int]) -> tuple[dict, dict]:
    hw = cm.HARDWARE[host]
    stats = cm.load_kernel_stats(in_dir, hw["arch"])
    points = cm.load_points(host, in_dir)
    f = cm.features(points, stats, hw)
    fitted = points.loc[f.index]
    y = fitted["lat"].to_numpy()
    stored = json.loads((in_dir / f"cost_model_{host}.json").read_text())
    theta = np.array([stored["params"][k] for k in cm.PARAMS])
    rows, checks = {}, {}

    # full model: stored parameters; check that a refit reproduces them
    choice, P = cm.select_e2(theta, df, chunks, stats, hw)
    rows["cost_model"] = evaluate(df, chunks, choice, P, P)
    rows["cost_model"].update({"n_free_params": 8, "uses_captured_call_timings": False,
                               "calibration_fit": cm.fit_quality(cm.predict(theta, f), y)})
    th_re, _ = cm.fit(f, y)
    checks["cost_model_refit_max_rel_param_diff"] = float(np.max(np.abs(th_re - theta) / np.maximum(np.abs(theta), 1e-12)))
    checks["cost_model_choice_equals_stored"] = bool(
        [int(chunks[i]) for i in choice] == stored["per_call_choice"]["mamba1"]["chunk"])
    sv = stored["V2_e2_calls"]
    checks["cost_model_stored_vs_recomputed"] = {
        k: [sv[k], rows["cost_model"][k]] for k in ("mean_regret_pct", "p95_regret_pct", "tie_aware_accuracy")}

    # work-only restriction of Eq. (5), with compiled occupancy
    x0 = cm.X0.copy()
    for i, v in WORK_ONLY_FIXED.items():
        x0[i] = v
    x0[[_I["t0"], _I["tau_m"], _I["tau_e"]]] = theta[[_I["t0"], _I["tau_m"], _I["tau_e"]]]
    th_w, info_w = cm.fit(f, y, x0=x0, fixed=WORK_ONLY_FIXED)
    c_w, P_w = cm.select_e2(th_w, df, chunks, stats, hw)
    rows["work_only"] = evaluate(df, chunks, c_w, P_w, P_w)
    rows["work_only"].update({"n_free_params": 3, "uses_captured_call_timings": False,
                              "params": {k: float(v) for k, v in zip(cm.PARAMS, th_w)},
                              "calibration_fit": cm.fit_quality(cm.predict(th_w, f), y)})
    # sensitivity: the drop-one ablation's convention for latency hiding (W1 = 1: the tile latency is
    # divided by the resident warps) instead of W1 = 256 (never hidden)
    fx1 = dict(WORK_ONLY_FIXED)
    fx1[_I["W1"]] = 1.0
    x1 = th_w.copy()
    x1[_I["W1"]] = 1.0
    th_1, _ = cm.fit(f, y, x0=x1, fixed=fx1)
    c_1, P_1 = cm.select_e2(th_1, df, chunks, stats, hw)
    rows["work_only_w1_eq_1"] = evaluate(df, chunks, c_1, P_1, P_1)
    rows["work_only_w1_eq_1"].update({"n_free_params": 3, "uses_captured_call_timings": False,
                                      "params": {k: float(v) for k, v in zip(cm.PARAMS, th_1)},
                                      "calibration_fit": cm.fit_quality(cm.predict(th_1, f), y)})

    # the same, with register-free occupancy (warp and block limits only)
    def no_reg(frame: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
        g = feats.copy()
        src = frame.loc[g.index]
        k0 = np.maximum(1, np.minimum(hw["max_warps_per_sm"] // src["nw"].to_numpy(int), hw["max_blocks_per_sm"]))
        P_prog = np.ceil(src["d"].to_numpy(float) / src["BD"].to_numpy(float)) * src["B"].to_numpy(float)
        g["waves"] = np.ceil(P_prog / (hw["n_sm"] * k0))
        g["m"] = np.minimum(k0, np.ceil(P_prog / hw["n_sm"]))
        return g

    f_nr = no_reg(fitted, f)
    th_nr, _ = cm.fit(f_nr, y, x0=th_w, fixed=WORK_ONLY_FIXED)
    cand = cm.e2_candidates(df, chunks)
    fc = no_reg(cand, cm.features(cand, stats, hw))
    pr = np.full(len(cand), np.inf)
    pr[fc.index] = cm.predict(th_nr, fc)
    P_nr = pr.reshape(len(chunks), len(df)).T
    rows["work_only_no_reg_occupancy"] = evaluate(df, chunks, P_nr.argmin(1), P_nr, P_nr)
    rows["work_only_no_reg_occupancy"].update({"n_free_params": 3, "uses_captured_call_timings": False,
                                               "params": {k: float(v) for k, v in zip(cm.PARAMS, th_nr)},
                                               "calibration_fit": cm.fit_quality(cm.predict(th_nr, f_nr), y)})

    # work-term form of the pipelines: (L, B, C) only
    Lc, Bc = df["L"].to_numpy(float), df["batch"].to_numpy(float)
    for name, sel in (("work_term_model", fitted["src"].isin(["e4_static", "e1"]).to_numpy()),
                      ("work_term_model_all_points", np.ones(len(fitted), bool))):
        d = fitted.loc[sel, ["L", "B", "C", "lat"]]
        th_p, r2 = cm.pipe_fit(d)
        Pp = np.stack([cm.pipe_predict(th_p, Lc, Bc, float(c)) for c in chunks], 1)
        rows[name] = evaluate(df, chunks, Pp.argmin(1), Pp, Pp)
        rows[name].update({"n_free_params": 5, "uses_captured_call_timings": False, "n_fit_points": int(sel.sum()),
                           "params": dict(zip(cm.PIPE_PARAMS, map(float, th_p))),
                           "calibration_fit": cm.fit_quality(
                               cm.pipe_predict(th_p, d["L"].to_numpy(float), d["B"].to_numpy(float),
                                               d["C"].to_numpy(float)), d["lat"].to_numpy()),
                           "unconstrained_optimal_chunk": float(np.sqrt(th_p[3] / th_p[2])) if th_p[2] > 0 else None})
    return rows, checks


# --------------------------------------------------------------------------- nearest calibrated key
def calibration_keys(host: str, in_dir: Path, chunks: list[int]) -> list[dict]:
    """E4 Mamba-1 keys with their default-launch static latencies (and E1 medians where they exist)."""
    keys = []
    for r in read_jsonl(in_dir / f"e4_{host}.jsonl", kind="m1tune"):
        keys.append({"key": r["key"], "L": int(r["L"]), "B": int(r["batch"]), "d": int(r["width"]),
                     "mean_ms": np.array([r["final"][f"static_{c}"]["mean_ms"] for c in chunks]),
                     "median_ms": np.array([r["final"][f"static_{c}"]["median_ms"] for c in chunks]),
                     "tuned_chunk": int(r["best_config"]["block_t"])})
    e1 = pd.DataFrame(read_jsonl(in_dir / f"e1_{host}.jsonl", kind="timing"))
    e1 = e1[e1["family"] == "mamba1"]
    med = e1.groupby(["L", "batch", "chunk"])["median_ms"].median()
    for k in keys:
        if all((k["L"], k["B"], c) in med.index for c in chunks):
            k["e1_ms"] = np.array([med[(k["L"], k["B"], c)] for c in chunks])
    return keys


def nearest_key(L: int, B: int, d: int, keys: list[dict], order: str = "batch_first", tie: str = "shorter") -> int:
    def dist(k):
        dl = abs(math.log2(max(L, 1)) - math.log2(k["L"]))
        db = abs(math.log2(B) - math.log2(k["B"]))
        dd = abs(math.log2(d) - math.log2(k["d"]))
        t = k["L"] if tie == "shorter" else -k["L"]
        return (db, dl, dd, t) if order == "batch_first" else (dl, db, dd, t)
    return min(range(len(keys)), key=lambda i: dist(keys[i]))


def nearest_key_selectors(host: str, in_dir: Path, df: pd.DataFrame, chunks: list[int]) -> tuple[dict, dict, dict]:
    keys = calibration_keys(host, in_dir, chunks)
    Ls, Bs, ds = df["L"].to_numpy(int), df["batch"].to_numpy(int), df["width"].to_numpy(int)
    idx = np.array([nearest_key(L, B, d, keys) for L, B, d in zip(Ls, Bs, ds)])
    idx_lf = np.array([nearest_key(L, B, d, keys, order="length_first") for L, B, d in zip(Ls, Bs, ds)])
    idx_long = np.array([nearest_key(L, B, d, keys, tie="longer") for L, B, d in zip(Ls, Bs, ds)])
    checks = {"nearest_key_batch_first_equals_length_first": bool(np.array_equal(idx, idx_lf)),
              "n_calls_on_log_length_tie": int((idx != idx_long).sum())}
    ci = {c: i for i, c in enumerate(chunks)}

    def run(ix, field):
        S = np.stack([keys[i][field] for i in ix])
        return evaluate(df, chunks, S.argmin(1), S)

    rows = {"nearest_calibrated_key": run(idx, "mean_ms")}
    rows["nearest_calibrated_key"].update({
        "uses_captured_call_timings": False,
        "table": [{"key": k["key"], "L": k["L"], "batch": k["B"], "best_chunk": int(chunks[int(np.argmin(k["mean_ms"]))]),
                   "n_calls_served": int((idx == i).sum())} for i, k in enumerate(keys)]})
    var = {"tie_to_longer_key": run(idx_long, "mean_ms"), "e4_static_median": run(idx, "median_ms")}
    # chunk of the autotuned launch configuration (not the default launch): choice only, no ranking
    tuned = np.array([ci[keys[i]["tuned_chunk"]] for i in idx])
    var["tuned_launch_config_chunk"] = evaluate(df, chunks, tuned)
    # E1 medians pooled with the E4 static re-measurements where both exist (batch 1)
    pooled = []
    for k in keys:
        s = k["mean_ms"] / k["mean_ms"].min()
        pooled.append((s + k["e1_ms"] / k["e1_ms"].min()) / 2 if "e1_ms" in k else s)
    S = np.stack([pooled[i] for i in idx])
    var["pooled_with_e1_medians"] = evaluate(df, chunks, S.argmin(1), S)
    var["pooled_with_e1_medians"]["n_keys_with_e1"] = int(sum("e1_ms" in k for k in keys))
    # log-length interpolation between the two calibrated lengths that bracket the call (same batch):
    # normalized latency curves weighted by the distance in log2 L (no tie-break needed)
    S = np.empty((len(df), len(chunks)))
    for n, (L, B) in enumerate(zip(Ls, Bs)):
        ks = sorted((k for k in keys if k["B"] == B), key=lambda k: k["L"])
        x, xs = math.log2(max(L, 1)), [math.log2(k["L"]) for k in ks]
        if x <= xs[0] or x >= xs[-1]:
            k = ks[0] if x <= xs[0] else ks[-1]
            S[n] = k["mean_ms"] / k["mean_ms"].min()
            continue
        j = int(np.searchsorted(xs, x))
        w = (x - xs[j - 1]) / (xs[j] - xs[j - 1])
        S[n] = (1 - w) * ks[j - 1]["mean_ms"] / ks[j - 1]["mean_ms"].min() + w * ks[j]["mean_ms"] / ks[j]["mean_ms"].min()
    var["interpolated_log_length"] = evaluate(df, chunks, S.argmin(1), S)
    return rows, var, checks


# --------------------------------------------------------------------------- captured-call (cross-validated) selectors
class ScoredLeafTree(e3.LeafCostTree):
    """E3 leaf-cost tree that also keeps each leaf's mean normalized latency per chunk (for the ranking)."""

    def fit(self, X, y, norm_lat):
        super().fit(X, y, norm_lat)
        leaves = self.tree.apply(X)
        self.leaf_score = {int(l): norm_lat[leaves == l].mean(0) for l in np.unique(leaves)}
        self.global_score = norm_lat.mean(0)
        return self

    def predict_scores(self, X):
        return np.stack([self.leaf_score.get(int(l), self.global_score) for l in self.tree.apply(X)])


def cv_selectors(df: pd.DataFrame, chunks: list[int]) -> dict:
    """E3 predictors with ranking scores, plus the batch-only table; same prompt folds as E3."""
    lat = latency_matrix(df, chunks)
    norm = lat / lat.min(1, keepdims=True)
    oracle_idx = lat.argmin(1)
    folds = df["fold"].to_numpy()
    names = ["global_static", "batch_only", "batch_only_tree", *e3.FEATURE_SETS]
    S = {n: np.full((len(df), len(chunks)), np.nan) for n in names}
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if tr.sum() == 0 or te.sum() == 0:
            continue
        S["global_static"][te] = norm[tr].mean(0)
        b_tr, b_te = df["batch"].to_numpy()[tr], df["batch"].to_numpy()[te]
        cell = {int(b): norm[tr][b_tr == b].mean(0) for b in np.unique(b_tr)}
        S["batch_only"][te] = np.stack([cell.get(int(b), norm[tr].mean(0)) for b in b_te])
        m = ScoredLeafTree(2).fit(df.loc[tr, ["batch"]].to_numpy(float), oracle_idx[tr], norm[tr])
        S["batch_only_tree"][te] = m.predict_scores(df.loc[te, ["batch"]].to_numpy(float))
        for name, (cols, depth) in e3.FEATURE_SETS.items():
            m = ScoredLeafTree(depth).fit(df.loc[tr, cols].to_numpy(float), oracle_idx[tr], norm[tr])
            S[name][te] = m.predict_scores(df.loc[te, cols].to_numpy(float))
    out = {}
    for name, s in S.items():
        out[name] = evaluate(df, chunks, s.argmin(1), s)
        out[name]["uses_captured_call_timings"] = True
    rule = df["entropy_rule_chunk"].clip(min(chunks), max(chunks)).to_numpy()
    ridx = np.array([chunks.index(int(c)) if int(c) in chunks else
                     int(np.argmin([abs(np.log2(c) - np.log2(x)) for x in chunks])) for c in rule])
    out["entropy_rule"] = evaluate(df, chunks, ridx)
    out["entropy_rule"]["uses_captured_call_timings"] = False
    return out


# --------------------------------------------------------------------------- driver
def analyze(host: str, in_dir: Path) -> dict:
    df = load_e2([in_dir / f"e2_mamba1_{host}.csv"])
    chunks = chunk_columns(df)
    rows, checks = model_selectors(host, in_dir, df, chunks)
    nk_rows, nk_var, nk_checks = nearest_key_selectors(host, in_dir, df, chunks)
    rows.update(nk_rows)
    checks.update(nk_checks)
    cv_all = cv_selectors(df, chunks)
    rows.update(cv_all)

    # consistency with the stored E3 and E5 results
    e3_path = in_dir / f"e3_mamba1_{host}.json"
    if e3_path.exists():
        e3s = json.loads(e3_path.read_text())
        e3_now = e3.evaluate(df, chunks)
        checks["e3_stored_vs_recomputed_mean_regret_pct"] = {
            k: [e3s["predictors_all_batches"][k]["mean_regret_pct"], e3_now[k]["mean_regret_pct"], rows[k]["mean_regret_pct"]]
            for k in e3_now}
        checks["e3_scored_trees_match_e3"] = bool(all(
            abs(e3_now[k]["mean_regret_pct"] - rows[k]["mean_regret_pct"]) < 1e-9 for k in e3_now))
    e5_path = in_dir / f"e5_{host}.json"
    if e5_path.exists():
        v = json.loads(e5_path.read_text())["view_e2_traces"]["mamba1"]
        checks["e5_key12_headroom_pct_stored_vs_recomputed"] = [
            v["oracle_headroom_over_key12_pct"], rows["nearest_calibrated_key"]["ratio_of_means_headroom_pct"]]
        checks["e5_key12_table_ms_stored_vs_recomputed"] = [
            v["key12_table_ms"], rows["nearest_calibrated_key"]["policy_mean_latency_ms"]]

    # batch 1: selectors trained within batch 1 only (the E3 "by batch" protocol)
    b1 = df[df["batch"] == 1].reset_index(drop=True)
    within_b1 = {k: _brief(v) for k, v in cv_selectors(b1, chunks).items()}

    res = {"experiment": "T1-baselines", "host": host, "n_calls": int(len(df)), "chunks": chunks,
           "definitions": __doc__, "rows": rows, "variants_nearest_calibrated_key": nk_var, "checks": checks,
           "batch1_trained_within_batch1": within_b1,
           "summary": {k: _brief(v) for k, v in rows.items()},
           "summary_batch1_calls": {k: {"mean_regret_pct": v["by_batch"][1]["mean_regret_pct"],
                                        "p95_regret_pct": v["by_batch"][1]["p95_regret_pct"],
                                        "tie_aware_accuracy": v["by_batch"][1]["tie_aware_accuracy"],
                                        "kendall_tau_per_call_mean": (v.get("kendall_tau_per_call_by_batch", {})
                                                                      .get(1, {}).get("mean"))}
                                    for k, v in rows.items()}}
    return res


ORDER = ["cost_model", "work_only", "work_only_no_reg_occupancy", "work_only_w1_eq_1", "work_term_model",
         "work_term_model_all_points",
         "nearest_calibrated_key", "shape", "batch_only", "batch_only_tree", "L_only", "entropy_only", "shape+entropy",
         "moments", "all", "global_static", "entropy_rule"]


def _fmt(x, spec):
    return "   -  " if x is None else format(x, spec)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--hosts", default="pro6000,vgpu3090")
    args = p.parse_args(argv)
    out_dir = args.out_dir or args.in_dir
    for h in args.hosts.split(","):
        res = analyze(h, args.in_dir)
        write_json(out_dir / f"cost_model_baselines_{h}.json", res)
        print(f"== {h} (n={res['n_calls']}); regret mean/p95 %, tied share, Kendall tau per call mean/median, "
              f"per shape mean; batch-1 calls mean/tied")
        for k in ORDER:
            r = res["rows"][k]
            t = r.get("kendall_tau_per_call", {})
            ts = r.get("kendall_tau_per_shape", {})
            b1 = res["summary_batch1_calls"][k]
            print(f"  {k:28s} {r['mean_regret_pct']:6.2f} / {r['p95_regret_pct']:6.2f}  tied {r['tie_aware_accuracy']:.3f}"
                  f"  tau {_fmt(t.get('mean'), '6.3f')} / {_fmt(t.get('median'), '6.3f')}  shape {_fmt(ts.get('mean'), '6.3f')}"
                  f"  | b1 {b1['mean_regret_pct']:5.2f} tied {b1['tie_aware_accuracy']:.3f}")
        for k, v in res["variants_nearest_calibrated_key"].items():
            print(f"  nearest-key variant {k:24s} {v['mean_regret_pct']:6.2f} / {v['p95_regret_pct']:6.2f} "
                  f"tied {v['tie_aware_accuracy']:.3f} ratio-of-means {v['ratio_of_means_headroom_pct']:.2f}%")
        print("  checks:", json.dumps(res["checks"], default=float)[:1500])


if __name__ == "__main__":
    main()
