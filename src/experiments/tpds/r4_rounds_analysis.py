"""Round-median oracle on re-timed captured calls (review items M2b, M2c, M8a; tpds-r4 E19).

CPU only, deterministic; reads stored result files and times nothing.

Data. For every tag T of ``--tags`` (default ``vgpu3090_r4spin,vgpu3090_r4plain``; the RTX PRO 6000 tags
``pro6000_r4spin,pro6000_r4plain`` run unchanged once their files exist) the table ``e2_mamba1_<T>.csv``
written by ``e2_oracle_sweep --rounds N`` (N >= 2): the 2360 captured 370M Mamba-1 calls, every chunk of
every call timed in N rounds of one warmup + 50-repeat block, the chunk order shuffled per call and round.
Tags containing ``spin`` are device time (GPU pre-spin), all others launch-inclusive. Per tag the script
also reads
  * the published single-block table of the same GPU and convention (``--published``; default
    ``e2_mamba1_<gpu>_r3spin.csv`` for device time and ``e2_mamba1_<gpu>.csv`` for launch-inclusive);
    its calls are the same calls (key, shape, fold and activation features are checked);
  * the frozen cost model of the same convention, ``cost_model_<cal>.json`` (``--cost-tags``; default
    <cal> = <gpu>_r3spin for device time, <gpu> for launch-inclusive; the first existing of these and T),
    with the compile facts and the calibration data (E4, E1) it was fitted on;
  * the host synchronization of the GPU (``e5b_<gpu>.json``) and the cost of one device-side reduction
    kernel before the scan (``e9_placement_vgpu3090.json``; 7.3 us at L = 2048, RTX 3090 only) as
    headroom thresholds, exactly as ``headroom_r4``.

Views of one table (the same calls; only the latency matrix, the oracle and the tie sets differ):
  round_median   ``analysis.round_median_view``: latency = median over rounds of the round means; oracle
                 on these medians; CI tie set = chunks whose 95% CI of the round means (between-round
                 SD, t with N - 1 dof) overlaps the oracle's (``tied_chunks_med`` of the sweep).
  single_block   round 0 alone: latency = the round-0 block mean, i.e. one 50-repeat block per chunk,
                 the protocol of the published tables; CI tie set = 95% CI overlap of the round-0 blocks
                 (t with n - 1 dof), the rule of the sweep. Rounds run back to back per call, round 0
                 directly after the call's inputs were loaded, so every round is also evaluated as the single
                 block (``comparison.single_block_by_round``). The round-0 within-block SD is read from
                 ``<stem>_rounds.jsonl`` when that file is present; otherwise it is the root mean square
                 of the N within-round SDs, recovered exactly from the pooled SD and the round means:
                     s_w^2 = [(N n - 1) s_pool^2 - n sum_r (m_r - m_bar)^2] / (N (n - 1)).
                 ``--single-round`` selects another round.
  pooled         the pooled N x 50 samples as stored (``lat_c<C>_mean``, ``tied_chunks``), i.e. what
                 the single-round readers see when given the multi-round CSV unchanged.
  published      the published single-block table (another session, same calls).

Metrics per view (the definitions of the published tables and scripts):
  table_II       per batch: best static chunk (in sample, lowest mean relative regret,
                 ``analysis.static_regret_table``), share of calls whose oracle is not that chunk, its
                 regret (mean / p95) and tie shares, and the entropy rule.
  predictors     held-out predictors of ``e3_predictors`` / ``e3b_information`` (grouped 5-fold
                 cross-validation over the capture-time prompt folds; leaf-cost trees of depth 4, length
                 only depth 3): best single chunk, entropy rule, batch only, length only, shape,
                 shape + entropy, all features, entropy only, moments; the held-out exact-shape table
                 (regret-optimal entries, bucket and global fallback) and its in-sample floor; the best
                 static chunk per batch (in sample); the frozen cost model (no refit: its per-call choice
                 is recomputed from the stored parameters and checked against the stored choice), the
                 cost model with its 2 and 3 best-ranked chunks profiled per exact shape and all six
                 profiled (``cost_model.topk_refinement``); the nearest calibration key
                 (``cost_model_baselines``). Per predictor: mean / p95 / p99 / max relative regret and
                 the tie shares under three criteria: CI overlap (the view's tie set), within 2% and within
                 5% of the oracle latency. Shape + entropy minus shape: paired difference with its 95%
                 prompt-cluster bootstrap interval and the equivalence verdicts at 0.5 and 1 point
                 (``e3b_information``; same seed rule, so the published intervals are reproduced).
  refit          the cost model refitted on its calibration data (``cost_model.fit``). The calibration
                 timings are not re-timed by the multi-round sweep, so the refit reproduces the frozen
                 parameters (maximum relative difference reported) and the regrets do not change.
  tie_sets       mean size of the oracle's tie set under the three criteria.
  headroom       absolute oracle headroom over an exact-shape table (us per call;
                 ``headroom_noise.TableStructure``): in sample (mean, largest shape), held out (all calls,
                 covered calls, fallback calls; held-out per-shape maximum over covered shapes),
                 split-half (the Gaussian split of ``headroom_noise`` with the view's standard error,
                 ``--draws`` draws, the random stream of ``headroom_r4``), and covered shapes above one host
                 synchronization and above 7.3 us (naive / split-half). Multi-round views add the
                 cross-round estimate: the table and the per-call oracle are chosen on the median of the
                 other rounds and scored on the held-out round (each round held out once, averaged per
                 call), so neither within-block nor block-level noise can select a chunk that it then
                 rewards; like split-half it is conservative in expectation.
  noise          median within-block coefficient of variation (multi-round: also the between-round one).

Per tag, ``comparison``: round_median against single_block (same session and calls; only the oracle
definition differs) and against the published table: every mean / p95 regret and tie share side by
side; how often the round-0 oracle differs from the round-median oracle, lies outside its tie set and
outside 2% / 5% of it; the regret of the round-0 oracle measured on the round medians and the apparent
regret of the round-median oracle measured on round 0; how often the oracle changes between two rounds;
and the block-level noise between two round means of the identical kernel (median and 90th percentile
of the relative difference, share of pairs with |z| > 1.96 under the within-block standard error, the
statistics quoted for the reference scan's triplicate timings in ``headroom_noise``).

Reproduction check: the published view must reproduce the stored single-block results (Table II rows
of ``e2_mamba1_<pub>.json``, every E3 predictor, the stored cost-model regret, top-k and choice, the
nearest key, the headroom row of ``headroom_r4.json`` with its random stream, and the tie shares and
paired difference of ``e3b_information_mamba1_<pub>.json``). Mismatches are listed in ``checks``.

Output: ``--out`` (default ``r4_rounds_analysis.json`` in the results directory).

Usage:  python3 -m src.experiments.tpds.r4_rounds_analysis [--tags vgpu3090_r4spin,vgpu3090_r4plain]
"""
from __future__ import annotations

import argparse
import json
import math
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.tpds import cost_model as cm
from src.experiments.tpds import cost_model_baselines as cmb
from src.experiments.tpds import e3_predictors as e3
from src.experiments.tpds import e3b_information as e3b
from src.experiments.tpds import headroom_noise as HN
from src.experiments.tpds import headroom_r4 as HR
from src.experiments.tpds.analysis import chunk_columns, latency_matrix, load_e2, round_median_view, static_regret_table
from src.experiments.tpds.common import RESULTS_DIR, read_jsonl, t_crit, utc_now, write_json

DEFAULT_TAGS = "vgpu3090_r4spin,vgpu3090_r4plain"
HEADROOM_SEED = 20260930          # headroom_r4 / headroom_noise
BOOT_SEED = 20261008              # e3b_information
MARGINS = e3b.TIE_MARGINS         # 2% and 5%
VIEWS = ("published", "single_block", "round_median", "pooled")
# predictors reported in the compact comparison tables
KEY_PREDICTORS = ("global_static", "best_static_per_batch_in_sample", "entropy_rule", "batch_only", "L_only", "shape",
                  "shape+entropy", "all", "entropy_only", "moments", "exact_shape_table_held_out",
                  "exact_shape_floor_in_sample", "cost_model", "cost_model_top2", "cost_model_top3",
                  "all_chunks_profiled", "nearest_calibrated_key")


# --------------------------------------------------------------------------- tags and files
def gpu_of(tag: str) -> str:
    return tag.split("_")[0]


def convention_of(tag: str) -> str:
    return "device" if "spin" in tag else "launch-inclusive"


def default_published(tag: str) -> str:
    g = gpu_of(tag)
    return f"{g}_r3spin" if convention_of(tag) == "device" else g


def default_cost_tag(tag: str, in_dir: Path) -> str | None:
    g = gpu_of(tag)
    cands = [f"{g}_r3spin", tag] if convention_of(tag) == "device" else [g, tag]
    return next((c for c in cands if (in_dir / f"cost_model_{c}.json").exists()), None)


def _overrides(spec: str | None) -> dict:
    if not spec:
        return {}
    return dict(kv.split("=", 1) for kv in spec.split(",") if "=" in kv)


# --------------------------------------------------------------------------- views
def round_matrix(df: pd.DataFrame, chunks: list[int]) -> np.ndarray:
    """(rounds, calls, chunks) round means parsed from ``lat_c<C>_rounds``."""
    cols = []
    for c in chunks:
        cols.append(np.array([[float(x) for x in str(s).split(";")] for s in df[f"lat_c{c}_rounds"]]))
    return np.stack(cols, 2).transpose(1, 0, 2)


def block_size(df: pd.DataFrame) -> int:
    return int(round(float(df["repeats"].iloc[0]) / float(df["rounds"].iloc[0])))


def within_block_std(df: pd.DataFrame, chunks: list[int], R: np.ndarray) -> np.ndarray:
    """RMS of the within-round SDs, recovered from the pooled SD and the round means (equal block sizes)."""
    N, n = R.shape[0], block_size(df)
    s_pool = df[[f"lat_c{c}_std" for c in chunks]].to_numpy(float)
    dev2 = ((R - R.mean(0, keepdims=True)) ** 2).sum(0)
    ss_w = (N * n - 1) * s_pool ** 2 - n * dev2
    return np.sqrt(np.clip(ss_w, 0.0, None) / (N * (n - 1)))


def rounds_jsonl_stats(csv_path: Path, df: pd.DataFrame, chunks: list[int], rnd: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(mean, std) of round ``rnd`` per call and chunk from ``<stem>_rounds.jsonl`` (None if absent or incomplete)."""
    p = csv_path.with_name(csv_path.stem + "_rounds.jsonl")
    if not p.exists():
        return None
    rows = {r["call"]: r for r in read_jsonl(p, kind="round") if int(r["round"]) == rnd}
    if not all(k in rows for k in df["key"]):
        return None
    mean = np.array([[rows[k]["stats"][str(c)]["mean_ms"] for c in chunks] for k in df["key"]], float)
    std = np.array([[rows[k]["stats"][str(c)]["std_ms"] for c in chunks] for k in df["key"]], float)
    return mean, std


def single_block_view(df: pd.DataFrame, chunks: list[int], mean: np.ndarray, std: np.ndarray, n: int) -> pd.DataFrame:
    """Single-round E2 table: one block per chunk, tie set by 95% CI overlap with the oracle (the sweep's rule)."""
    out = df.copy()
    half = t_crit(n - 1) * std / math.sqrt(n)
    lo, hi = mean - half, mean + half
    for j, c in enumerate(chunks):
        out[f"lat_c{c}_mean"], out[f"lat_c{c}_std"] = mean[:, j], std[:, j]
        out[f"lat_c{c}_ci_lo"], out[f"lat_c{c}_ci_hi"] = lo[:, j], hi[:, j]
    o = mean.argmin(1)
    ar = np.arange(len(df))
    out["oracle_chunk"] = np.array(chunks)[o]
    out["oracle_mean_ms"] = mean[ar, o]
    tied = lo <= hi[ar, o][:, None]
    out["tied_chunks"] = [";".join(str(c) for c, t in zip(chunks, row) if t) for row in tied]
    out["repeats"] = n
    return out


def tied_mask(df: pd.DataFrame, chunks: list[int]) -> np.ndarray:
    tied = [set(int(c) for c in str(t).split(";") if c) for t in df["tied_chunks"].fillna("")]
    return np.array([[c in t for c in chunks] for t in tied], dtype=bool)


# --------------------------------------------------------------------------- cost model
class CostModel:
    """Frozen Eq. (5) parameters of one calibration tag, its compile facts and calibration keys."""

    def __init__(self, cal: str, in_dir: Path, chunks: list[int]):
        self.cal = cal
        self.stored = json.loads((in_dir / f"cost_model_{cal}.json").read_text())
        self.theta = np.array([self.stored["params"][k] for k in cm.PARAMS])
        self.hw = cm.hardware(cal, in_dir)
        self.stats = cm.load_kernel_stats(in_dir, self.hw["arch"])
        self.keys = cmb.calibration_keys(cal, in_dir, chunks)
        pts = cm.load_points(cal, in_dir)
        self.f = cm.features(pts, self.stats, self.hw)
        self.y = pts.loc[self.f.index, "lat"].to_numpy(float)
        th, info = cm.fit(self.f, self.y)
        self.refit = {"n_calibration_points": int(len(self.y)), "fit_info": info,
                      "max_rel_param_diff_vs_frozen": float(np.max(np.abs(th - self.theta)
                                                                   / np.maximum(np.abs(self.theta), 1e-12))),
                      "params_refit": dict(zip(cm.PARAMS, map(float, th)))}
        self.theta_refit = th

    def choice(self, df: pd.DataFrame, chunks: list[int], theta=None) -> tuple[np.ndarray, np.ndarray]:
        return cm.select_e2(self.theta if theta is None else theta, df, chunks, self.stats, self.hw)

    def nearest_key_choice(self, df: pd.DataFrame) -> np.ndarray:
        idx = [cmb.nearest_key(int(L), int(B), int(d), self.keys)
               for L, B, d in df[["L", "batch", "width"]].itertuples(index=False)]
        S = np.stack([self.keys[i]["mean_ms"] for i in idx])
        return S.argmin(1)


# --------------------------------------------------------------------------- metrics of one view
def regret_record(reg: np.ndarray, tie: np.ndarray) -> dict:
    rec = {"n": int(len(reg)), "mean_regret_pct": float(100 * reg.mean()),
           "p95_regret_pct": float(100 * np.percentile(reg, 95)), "p99_regret_pct": float(100 * np.percentile(reg, 99)),
           "max_regret_pct": float(100 * reg.max()), "oracle_hit_rate": float((reg <= 1e-12).mean()),
           "ci_tie_share": float(tie.mean())}
    for m in MARGINS:
        rec[f"within_{round(100 * m)}pct_share"] = float((reg <= m + 1e-12).mean())
    return rec


def table_ii(df: pd.DataFrame, chunks: list[int], norm: np.ndarray, tmask: np.ndarray) -> dict:
    out = {}
    for b in sorted(df["batch"].unique()):
        sel = (df["batch"] == b).to_numpy()
        sub = df[sel]
        st = static_regret_table(sub, chunks)
        best = st["best_static"]
        j = chunks.index(int(best.split("_")[1]))
        rec = {"n_calls": int(sel.sum()), "best_static_chunk": int(chunks[j]),
               "oracle_ne_static_share": float(1.0 - st[best]["oracle_hit_rate"]),
               "static": regret_record(norm[sel, j] - 1, tmask[sel, j]),
               "entropy_rule": {k: st["entropy_rule"][k] for k in ("mean_regret_pct", "p95_regret_pct")},
               "oracle_distribution": {int(k): int(v) for k, v in sub["oracle_chunk"].value_counts().items()}}
        out[int(b)] = rec
    return out


def predictor_choices(t: e3b.Table, model: CostModel | None) -> tuple[dict, dict]:
    """Per-call chunk index of every predictor (held out as in E3 / E3b), plus side information."""
    ch, info = {}, {}
    ch["global_static"] = e3b.cv_static(t)
    ch["entropy_rule"] = e3b.entropy_rule_choice(t)
    ch["batch_only"], _ = e3b.cv_table(t, [t.batch_code])
    for name, (cols, depth) in e3.FEATURE_SETS.items():
        ch[name] = e3b.cv_tree(t, cols, depth)
    ek_choice, ek_info = e3b.cv_table(t, [t.key, t.bucket_key])
    ch["exact_shape_table_held_out"] = ek_choice
    info["exact_shape_table_calls_per_level"] = ek_info["calls_per_level"]
    sums = np.zeros((t.key.max() + 1, t.C))
    np.add.at(sums, t.key, t.norm_fit)
    ch["exact_shape_floor_in_sample"] = sums.argmin(1)[t.key]
    sums = np.zeros((t.batch_code.max() + 1, t.C))
    np.add.at(sums, t.batch_code, t.norm_fit)
    ch["best_static_per_batch_in_sample"] = sums.argmin(1)[t.batch_code]
    if model is not None:
        choice, P = model.choice(t.df, t.chunks)
        ch["cost_model"] = choice
        lat = t.lat
        rank = np.argsort(P, 1)
        for k in (2, 3, len(t.chunks)):
            pick = np.empty(t.N, dtype=int)
            for _, g in t.df.groupby(["L", "batch"]):      # cost_model.topk_refinement
                i = g.index.to_numpy()
                cand = rank[i[0], :k]
                pick[i] = cand[np.argmin(lat[i][:, cand].mean(0))]
            ch[f"cost_model_top{k}" if k < len(t.chunks) else "all_chunks_profiled"] = pick
        ch["nearest_calibrated_key"] = model.nearest_key_choice(t.df)
        stored = model.stored["per_call_choice"]["mamba1"]
        m = dict(zip(stored["keys"], stored["chunk"]))
        info["cost_model_choice_equals_stored"] = bool(all(int(t.chunks[c]) == int(m.get(k, -1))
                                                           for c, k in zip(choice, t.df["key"])))
        rc, _ = model.choice(t.df, t.chunks, model.theta_refit)
        info["cost_model_refit_choice_equals_frozen"] = bool(np.array_equal(rc, choice))
    return ch, info


def headroom_block(df: pd.DataFrame, chunks: list[int], thresholds: dict, dev_pts, draws: int, rng,
                   R: np.ndarray | None = None) -> dict:
    """Exact-shape headroom estimators of ``headroom_noise`` / ``headroom_r4`` (us per call)."""
    m = latency_matrix(df, chunks)
    s = df[[f"lat_c{c}_std" for c in chunks]].to_numpy(float)
    se = s / np.sqrt(df["repeats"].to_numpy(float))[:, None]
    st = HN.TableStructure(df)
    hd = st.headroom(m, m)
    naive = HN.summarize_protocol(hd, st, thresholds)
    col = HN.DrawCollector(st)
    acc_cv = np.zeros(st.n_exact)
    for _ in range(draws):
        rng.standard_normal(m.shape)                     # E: keeps the random stream of headroom_noise.analyze_dataset
        D = rng.standard_normal(m.shape) * se
        h = st.cross_fit(m + D, m - D)
        col.add(h)
        acc_cv += st.per_shape(h["cv"])
    split = col.summary(thresholds)
    ps_in, ps_cv, sh_cv = st.per_shape(hd["in"]), st.per_shape(hd["cv"]), acc_cv / draws
    cov = st.shape_covered
    sync = thresholds.get("host_sync")

    def counts(ps: np.ndarray, mask: np.ndarray) -> dict:
        v = ps[mask]
        out = {"n_shapes": int(mask.sum()), "above_sync_nominal_20us": int((v > HR.NOMINAL_SYNC_MS).sum()),
               "above_device_7p3us": int((v > dev_pts[0][1]).sum()), "above_device_42us": int((v > dev_pts[-1][1]).sum())}
        if sync is not None:
            out["above_host_sync_measured"] = int((v > sync).sum())
        return out

    def argmax_shape(ps: np.ndarray, mask: np.ndarray) -> dict:
        k = np.where(mask)[0][int(np.argmax(ps[mask]))]
        return {"L": st.exact_keys[k][0], "batch": st.exact_keys[k][1], "n_calls": int(st.shape_counts[k]),
                "headroom_us": float(1000 * ps[k])}

    ho = naive["held_out"]
    rec = {"n_calls": int(st.n), "n_shapes": int(st.n_exact), "n_shapes_covered": int(cov.sum()),
           "n_calls_covered": int(st.covered.sum()),
           "median_within_block_cv_pct": float(100 * np.median(s / m)),
           "cv_basis": "lat_c<C>_std of the view (round_median: between-round SD of the round means)",
           "in_sample": {"mean_us": naive["in_sample"]["all"]["mean_us"],
                         "max_shape_us": naive["in_sample"]["all"]["max_shape_us"],
                         "argmax_shape": argmax_shape(ps_in, np.ones(st.n_exact, bool))},
           "held_out_all": {"mean_us": ho["all"]["mean_us"], "mean_pct": ho["all"]["mean_pct"],
                            "max_shape_us": ho["all"]["max_shape_us"]},
           "held_out_covered": {"mean_us": ho["covered"].get("mean_us"), "mean_pct": ho["covered"].get("mean_pct"),
                                "max_shape_us": ho["covered"].get("max_shape_us"),
                                "argmax_shape": argmax_shape(ps_cv, cov)},
           "held_out_fallback": {"n_calls": ho["fallback"].get("n_calls"), "mean_us": ho["fallback"].get("mean_us"),
                                 "max_shape_us": ho["fallback"].get("max_shape_us")},
           "split_half_covered": {k: split["cv_covered"].get(k) for k in
                                  ("mean_us", "ci95_lo_us", "ci95_hi_us", "max_shape_of_draw_mean_us", "n_draws")},
           "split_half_note": ("Gaussian split of each latency with the view's standard error; valid for one block of "
                               "i.i.d. repeats. For round medians the standard error is the between-round one of 3 "
                               "rounds, for which the split is not a valid model: use cross_round there."),
           "split_half_in_sample": {k: split["in_all"].get(k) for k in ("mean_us", "ci95_lo_us", "ci95_hi_us")},
           "shapes_above": {"in_sample_all_shapes": counts(ps_in, np.ones(st.n_exact, bool)),
                            "held_out_covered": counts(ps_cv, cov), "split_half_held_out_covered": counts(sh_cv, cov)}}
    if R is not None and R.shape[0] >= 2:
        acc = {"in": np.zeros(st.n), "cv": np.zeros(st.n), "ref": np.zeros(st.n)}
        for r in range(R.shape[0]):
            sel = np.median(np.delete(R, r, 0), 0)
            h = st.headroom(sel, R[r])
            for k in acc:
                acc[k] += h[k] / R.shape[0]
        cr = HN.summarize_protocol(acc, st, thresholds)
        ps_cr = st.per_shape(acc["cv"])
        rec["cross_round"] = {
            "definition": "table and oracle chosen on the median of the other rounds, scored on the held-out round",
            "in_sample_mean_us": cr["in_sample"]["all"]["mean_us"], "in_sample_max_shape_us": cr["in_sample"]["all"]["max_shape_us"],
            "held_out_all_mean_us": cr["held_out"]["all"]["mean_us"],
            "held_out_covered_mean_us": cr["held_out"]["covered"].get("mean_us"),
            "held_out_covered_max_shape_us": cr["held_out"]["covered"].get("max_shape_us"),
            "held_out_covered_argmax_shape": argmax_shape(ps_cr, cov),
            "shapes_above_held_out_covered": counts(ps_cr, cov)}
    return rec


def view_metrics(df: pd.DataFrame, chunks: list[int], view: str, label: str, model: CostModel | None,
                 thresholds: dict, dev_pts, draws: int, rng_headroom, boot_tag: str, n_boot: int,
                 R: np.ndarray | None = None) -> tuple[dict, dict]:
    t = e3b.Table(df.reset_index(drop=True), label, "mamba1", label, view)
    ar = np.arange(t.N)
    choices, info = predictor_choices(t, model)
    preds = {n: regret_record(t.norm[ar, c] - 1, t.tied_mask[ar, c]) for n, c in choices.items()}
    for n, c in choices.items():
        preds[n]["chosen_distribution"] = {int(k): int(v) for k, v in pd.Series(np.array(chunks)[c]).value_counts().items()}
    # paired difference shape + entropy minus shape, prompt-cluster bootstrap (e3b seed rule)
    seed = BOOT_SEED + zlib.crc32(boot_tag.encode()) % 100_000
    rng_boot = np.random.default_rng(np.random.SeedSequence(seed).spawn(5)[0])
    diffs = {}
    pairs = [("shape+entropy", "shape"), ("all", "shape"), ("cost_model", "shape")] if "cost_model" in choices else \
        [("shape+entropy", "shape"), ("all", "shape")]
    vals = {f"{a} - {b}": 100 * ((t.norm[ar, choices[a]] - 1) - (t.norm[ar, choices[b]] - 1)) for a, b in pairs}
    reps = e3b.cluster_bootstrap(vals, t.clusters, n_boot, rng_boot, {})
    for k, v in vals.items():
        diffs[k] = e3b.equivalence(float(v.mean()), reps[k])
    # tie-set sizes under the three criteria
    within = {round(100 * m): t.norm <= 1 + m + 1e-12 for m in MARGINS}
    tie_sets = {"mean_ci_tie_set_size": float(t.tied_mask.sum(1).mean()),
                **{f"mean_within_{k}pct_set_size": float(w.sum(1).mean()) for k, w in within.items()},
                "share_calls_with_ci_tie": float((t.tied_mask.sum(1) > 1).mean())}
    lat = t.lat
    spread = lat.max(1) / lat.min(1)
    res = {"view": view, "n_calls": int(t.N), "chunks": chunks,
           "oracle_distribution": {int(k): int(v) for k, v in df["oracle_chunk"].value_counts().items()},
           "spread_slowest_over_fastest": {"median": float(np.median(spread)), "p95": float(np.percentile(spread, 95))},
           "table_II": table_ii(df.reset_index(drop=True), chunks, t.norm, t.tied_mask),
           "predictors": preds, "paired_differences": diffs, "tie_sets": tie_sets, "info": info,
           "headroom": headroom_block(df.reset_index(drop=True), chunks, thresholds, dev_pts, draws, rng_headroom, R)}
    return res, choices


# --------------------------------------------------------------------------- reproduction of published values
def _close(a, b, tol=1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)))


def check_published(pub_tag: str, cal: str | None, res: dict, in_dir: Path) -> dict:
    out, mism, n = {}, [], 0

    def cmp(name, stored, new, tol=1e-9):
        nonlocal n
        n += 1
        if not _close(stored, new, tol):
            mism.append(f"{name}: stored {stored} vs recomputed {new}")

    pr = res["predictors"]
    p = in_dir / f"e3_mamba1_{pub_tag}.json"
    if p.exists():
        s = json.loads(p.read_text())["predictors_all_batches"]
        for k in ["global_static", "entropy_rule", *e3.FEATURE_SETS]:
            cmp(f"e3.{k}.mean", s[k]["mean_regret_pct"], pr[k]["mean_regret_pct"])
            cmp(f"e3.{k}.p95", s[k]["p95_regret_pct"], pr[k]["p95_regret_pct"])
            cmp(f"e3.{k}.tie", s[k]["tie_aware_accuracy"], pr[k]["ci_tie_share"])
    p = in_dir / f"e2_mamba1_{pub_tag}.json"
    if p.exists():
        s = json.loads(p.read_text())
        s = s.get("summary", s)["by_batch"]
        for b, rec in res["table_II"].items():
            sr = s[str(b)]["static_regret"]
            cmp(f"tableII.b{b}.best", int(sr["best_static"].split("_")[1]), rec["best_static_chunk"])
            cmp(f"tableII.b{b}.mean", sr[sr["best_static"]]["mean_regret_pct"], rec["static"]["mean_regret_pct"])
            cmp(f"tableII.b{b}.p95", sr[sr["best_static"]]["p95_regret_pct"], rec["static"]["p95_regret_pct"])
            cmp(f"tableII.b{b}.rule", sr["entropy_rule"]["mean_regret_pct"], rec["entropy_rule"]["mean_regret_pct"])
    if cal is not None:
        s = json.loads((in_dir / f"cost_model_{cal}.json").read_text())
        v = s["V2_e2_calls"]
        cmp("cost_model.mean", v["mean_regret_pct"], pr["cost_model"]["mean_regret_pct"])
        cmp("cost_model.p95", v["p95_regret_pct"], pr["cost_model"]["p95_regret_pct"])
        cmp("cost_model.tie", v["tie_aware_accuracy"], pr["cost_model"]["ci_tie_share"])
        for k, name in (("top2", "cost_model_top2"), ("top3", "cost_model_top3"), ("all_chunks_profiled", "all_chunks_profiled")):
            cmp(f"cost_model.{k}.mean", s["V2_topk_profiling"][k]["mean_regret_pct"], pr[name]["mean_regret_pct"])
            cmp(f"cost_model.{k}.p95", s["V2_topk_profiling"][k]["p95_regret_pct"], pr[name]["p95_regret_pct"])
        p = in_dir / f"cost_model_baselines_{cal}.json"
        if p.exists():
            nk = json.loads(p.read_text())["summary"]["nearest_calibrated_key"]
            cmp("nearest_key.mean", nk["mean_regret_pct"], pr["nearest_calibrated_key"]["mean_regret_pct"])
            cmp("nearest_key.p95", nk["p95_regret_pct"], pr["nearest_calibrated_key"]["p95_regret_pct"])
            cmp("nearest_key.tie", nk["tie_aware_accuracy"], pr["nearest_calibrated_key"]["ci_tie_share"])
    p = in_dir / f"e3b_information_mamba1_{pub_tag}.json"
    if p.exists():
        s = json.loads(p.read_text())
        tm = s["tie_margins"]
        for k_new, k_old in (("global_static", "global_static"), ("shape", "shape"), ("cost_model", "cost_model"),
                             ("exact_shape_table_held_out", "exact_key_table")):
            if k_old in tm["predictors"] and k_new in pr:
                cmp(f"e3b.{k_old}.ci", tm["predictors"][k_old]["ci_tie_share"], pr[k_new]["ci_tie_share"])
                for m in (2, 5):
                    cmp(f"e3b.{k_old}.w{m}", tm["predictors"][k_old][f"within_{m}pct_share"], pr[k_new][f"within_{m}pct_share"])
        cmp("e3b.exact_key_table.mean", s["predictors"]["exact_key_table"]["mean_regret_pct"],
            pr["exact_shape_table_held_out"]["mean_regret_pct"])
        ts = tm["oracle_tie_sets"]
        cmp("e3b.tie_set.ci", ts["mean_ci_tie_set_size"], res["tie_sets"]["mean_ci_tie_set_size"])
        for m in (2, 5):
            cmp(f"e3b.tie_set.w{m}", ts[f"mean_within_{m}pct_set_size"], res["tie_sets"][f"mean_within_{m}pct_set_size"])
        d = s["differences"]["shape+entropy - shape"]["prompt_cluster"]
        cmp("e3b.diff.point", d["diff_pp"], res["paired_differences"]["shape+entropy - shape"]["diff_pp"])
        for i in (0, 1):
            cmp(f"e3b.diff.ci{i}", d["ci95_pp"][i], res["paired_differences"]["shape+entropy - shape"]["ci95_pp"][i])
    p = in_dir / "headroom_r4.json"
    name = next((s[0] for s in HR.HEADROOM_SETS if s[1] == f"e2_mamba1_{pub_tag}.csv"), None)
    if p.exists() and name is not None:
        rows = json.loads(p.read_text())["M8a"]["rows"]
        if name in rows:
            r, h = rows[name], res["headroom"]
            for a, b in (("in_sample", "mean_us"), ("in_sample", "max_shape_us"), ("held_out_all", "mean_us"),
                         ("held_out_covered", "mean_us"), ("held_out_covered", "max_shape_us"),
                         ("split_half_covered", "mean_us"), ("split_half_covered", "ci95_lo_us"),
                         ("split_half_covered", "ci95_hi_us")):
                cmp(f"headroom.{a}.{b}", r[a][b], h[a][b], tol=1e-9)
            for a in ("held_out_covered", "split_half_held_out_covered"):
                for b in ("above_host_sync_measured", "above_device_7p3us"):
                    if b in r["shapes_above"][a]:
                        cmp(f"headroom.{a}.{b}", r["shapes_above"][a][b], h["shapes_above"][a][b])
            cmp("headroom.cv_pct", r["median_within_block_cv_pct"], h["median_within_block_cv_pct"])
    out.update({"n_compared": n, "n_mismatches": len(mism), "mismatches": mism})
    return out


def headroom_rng_index(pub_csv: str) -> int | None:
    """Random-stream index that ``headroom_r4`` uses for a published table (None if it does not use it)."""
    names = [d[0] for d in HN.DATASETS]
    for k, s in enumerate(HR.HEADROOM_SETS):
        if s[1] == pub_csv:
            return names.index(s[0]) if s[0] in names else HR.NEW_INDEX_BASE + k
    return None


# --------------------------------------------------------------------------- comparison of views
def block_noise(R: np.ndarray, se: np.ndarray, df: pd.DataFrame) -> dict:
    """Differences between round means of the identical kernel (all calls and chunks, every pair of rounds)."""
    N = R.shape[0]
    rel, z = [], []
    for a in range(N):
        for b in range(a + 1, N):
            d = R[a] - R[b]
            rel.append(d / (0.5 * (R[a] + R[b])))
            z.append(d / np.sqrt(2.0) / se)
    rel, z = np.stack(rel), np.stack(z)            # (pairs, calls, chunks)
    lvl = R.mean(0)
    var_r = R.var(0, ddof=1)
    block_sd = np.sqrt(np.clip(var_r - se ** 2, 0, None)) / lvl
    out = {"n_pairs": int(rel.size), "median_abs_rel_diff_pct": float(100 * np.median(np.abs(rel))),
           "p90_abs_rel_diff_pct": float(100 * np.percentile(np.abs(rel), 90)),
           "share_abs_z_above_1p96": float(np.mean(np.abs(z) > 1.96)),
           "median_block_level_rel_sd_pct": float(100 * np.median(block_sd)),
           "rms_block_level_rel_sd_pct": float(100 * np.sqrt(np.mean(block_sd ** 2))),
           "note": "z uses the within-block standard error (RMS over rounds when the per-round SDs are not stored)",
           "by_batch": {}}
    for b in sorted(df["batch"].unique()):
        s = (df["batch"] == b).to_numpy()
        out["by_batch"][int(b)] = {"median_abs_rel_diff_pct": float(100 * np.median(np.abs(rel[:, s]))),
                                   "p90_abs_rel_diff_pct": float(100 * np.percentile(np.abs(rel[:, s]), 90)),
                                   "share_abs_z_above_1p96": float(np.mean(np.abs(z[:, s]) > 1.96))}
    return out


def oracle_comparison(chunks, ref_df: pd.DataFrame, other_df: pd.DataFrame, ref_name: str, other_name: str) -> dict:
    """How the oracle of ``other`` relates to that of ``ref`` (both latency matrices on the same calls)."""
    Lr, Lo = latency_matrix(ref_df, chunks), latency_matrix(other_df, chunks)
    o_r, o_o = Lr.argmin(1), Lo.argmin(1)
    ar = np.arange(len(Lr))
    tr = tied_mask(ref_df, chunks)
    reg_on_ref = Lr[ar, o_o] / Lr[ar, o_r] - 1           # the other view's oracle scored on the reference latencies
    reg_on_other = Lo[ar, o_r] / Lo[ar, o_o] - 1          # the reference oracle scored on the other view
    return {"reference": ref_name, "other": other_name,
            "share_oracle_differs": float((o_r != o_o).mean()),
            "share_other_oracle_outside_reference_ci_tie_set": float((~tr[ar, o_o]).mean()),
            **{f"share_other_oracle_outside_{round(100 * m)}pct_of_reference": float((reg_on_ref > m + 1e-12).mean())
               for m in MARGINS},
            "regret_of_other_oracle_on_reference_pct": {"mean": float(100 * reg_on_ref.mean()),
                                                        "p95": float(100 * np.percentile(reg_on_ref, 95)),
                                                        "max": float(100 * reg_on_ref.max())},
            "apparent_regret_of_reference_oracle_on_other_pct": {"mean": float(100 * reg_on_other.mean()),
                                                                 "p95": float(100 * np.percentile(reg_on_other, 95)),
                                                                 "max": float(100 * reg_on_other.max())}}


def compact(views: dict) -> dict:
    """Metric -> {view: value}: the numbers a writer needs, side by side."""
    rows = {}

    def put(key, fn):
        rows[key] = {}
        for v, r in views.items():
            try:
                rows[key][v] = fn(r)
            except (KeyError, TypeError):
                rows[key][v] = None

    for p in KEY_PREDICTORS:
        for f in ("mean_regret_pct", "p95_regret_pct", "ci_tie_share", "within_2pct_share", "within_5pct_share"):
            put(f"{p}.{f}", lambda r, p=p, f=f: r["predictors"][p][f])
    put("shape+entropy - shape.diff_pp", lambda r: r["paired_differences"]["shape+entropy - shape"]["diff_pp"])
    put("shape+entropy - shape.ci95_pp", lambda r: r["paired_differences"]["shape+entropy - shape"]["ci95_pp"])
    put("shape+entropy - shape.verdict_0.5pp",
        lambda r: r["paired_differences"]["shape+entropy - shape"]["verdicts"]["margin_0.5pp"]["verdict_95ci"])
    for b in (1, 4, 16):
        put(f"table_II.b{b}.best_static_chunk", lambda r, b=b: r["table_II"][b]["best_static_chunk"])
        put(f"table_II.b{b}.oracle_ne_static_share", lambda r, b=b: r["table_II"][b]["oracle_ne_static_share"])
        put(f"table_II.b{b}.static_mean_pct", lambda r, b=b: r["table_II"][b]["static"]["mean_regret_pct"])
        put(f"table_II.b{b}.static_p95_pct", lambda r, b=b: r["table_II"][b]["static"]["p95_regret_pct"])
        put(f"table_II.b{b}.static_ci_tie_share", lambda r, b=b: r["table_II"][b]["static"]["ci_tie_share"])
        put(f"table_II.b{b}.rule_mean_pct", lambda r, b=b: r["table_II"][b]["entropy_rule"]["mean_regret_pct"])
        put(f"table_II.b{b}.rule_p95_pct", lambda r, b=b: r["table_II"][b]["entropy_rule"]["p95_regret_pct"])
    for k in ("mean_ci_tie_set_size", "mean_within_2pct_set_size", "mean_within_5pct_set_size"):
        put(f"tie_sets.{k}", lambda r, k=k: r["tie_sets"][k])
    h = "headroom"
    put("headroom.median_within_block_cv_pct", lambda r: r[h]["median_within_block_cv_pct"])
    put("headroom.in_sample_mean_us", lambda r: r[h]["in_sample"]["mean_us"])
    put("headroom.in_sample_max_shape_us", lambda r: r[h]["in_sample"]["max_shape_us"])
    put("headroom.held_out_all_mean_us", lambda r: r[h]["held_out_all"]["mean_us"])
    put("headroom.held_out_covered_mean_us", lambda r: r[h]["held_out_covered"]["mean_us"])
    put("headroom.held_out_covered_max_shape_us", lambda r: r[h]["held_out_covered"]["max_shape_us"])
    put("headroom.held_out_fallback_max_shape_us", lambda r: r[h]["held_out_fallback"]["max_shape_us"])
    put("headroom.split_half_covered_mean_us", lambda r: r[h]["split_half_covered"]["mean_us"])
    put("headroom.split_half_covered_ci95_us", lambda r: [r[h]["split_half_covered"]["ci95_lo_us"],
                                                         r[h]["split_half_covered"]["ci95_hi_us"]])
    put("headroom.above_host_sync_naive_split", lambda r: [r[h]["shapes_above"]["held_out_covered"].get("above_host_sync_measured"),
                                                           r[h]["shapes_above"]["split_half_held_out_covered"].get("above_host_sync_measured")])
    put("headroom.above_7p3us_naive_split", lambda r: [r[h]["shapes_above"]["held_out_covered"]["above_device_7p3us"],
                                                       r[h]["shapes_above"]["split_half_held_out_covered"]["above_device_7p3us"]])
    put("headroom.cross_round_held_out_covered_mean_us", lambda r: r[h]["cross_round"]["held_out_covered_mean_us"])
    put("headroom.cross_round_in_sample_mean_us", lambda r: r[h]["cross_round"]["in_sample_mean_us"])
    put("headroom.cross_round_held_out_covered_max_shape_us", lambda r: r[h]["cross_round"]["held_out_covered_max_shape_us"])
    put("spread.median", lambda r: r["spread_slowest_over_fastest"]["median"])
    put("spread.p95", lambda r: r["spread_slowest_over_fastest"]["p95"])
    return rows


# --------------------------------------------------------------------------- per tag
def analyze_tag(tag: str, args) -> dict:
    t0 = time.time()
    in_dir = args.in_dir
    path = in_dir / f"e2_mamba1_{tag}.csv"
    df = load_e2([path]).reset_index(drop=True)
    if df.empty or "oracle_chunk_med" not in df.columns:
        return {"tag": tag, "skipped": f"{path.name} missing or not a multi-round table"}
    chunks = chunk_columns(df)
    gpu, conv = gpu_of(tag), convention_of(tag)
    pub_tag = _overrides(args.published).get(tag, default_published(tag))
    cal = _overrides(args.cost_tags).get(tag) or default_cost_tag(tag, in_dir)
    R = round_matrix(df, chunks)
    n_block = block_size(df)
    s_w = within_block_std(df, chunks, R)
    res = {"tag": tag, "gpu": gpu, "convention": conv, "source": path.name, "n_calls": int(len(df)),
           "rounds": int(R.shape[0]), "repeats_per_block": n_block, "chunks": chunks,
           "published_table": f"e2_mamba1_{pub_tag}.csv", "cost_model_tag": cal, "views": {}, "checks": {}}
    # consistency of the stored round statistics with the parsed round means
    med = df[[f"lat_c{c}_med" for c in chunks]].to_numpy(float)
    res["checks"]["round_median_matches_round_means_max_rel_diff"] = float(np.max(np.abs(np.median(R, 0) / med - 1)))
    pooled_mean = df[[f"lat_c{c}_mean" for c in chunks]].to_numpy(float)
    res["checks"]["pooled_mean_matches_round_means_max_rel_diff"] = float(np.max(np.abs(R.mean(0) / pooled_mean - 1)))
    # single-block view: round r, its own SD when the rounds file is present, else the RMS within-round SD
    rnd = args.single_round
    js = rounds_jsonl_stats(path, df, chunks, rnd)
    if js is not None:
        sb_mean, sb_std = js
        res["single_block_std_source"] = "rounds.jsonl (own SD of the round)"
        res["checks"]["rounds_jsonl_mean_matches_csv_max_rel_diff"] = float(np.max(np.abs(sb_mean / R[rnd] - 1)))
        allr = [rounds_jsonl_stats(path, df, chunks, r) for r in range(R.shape[0])]
        rms = np.sqrt(np.mean([a[1] ** 2 for a in allr], 0))
        res["checks"]["derived_rms_within_sd_vs_rounds_jsonl_max_rel_diff"] = float(np.max(np.abs(s_w / rms - 1)))
        alt = single_block_view(df, chunks, sb_mean, s_w, n_block)
        own = single_block_view(df, chunks, sb_mean, sb_std, n_block)
        res["checks"]["single_block_tie_sets_rms_sd_vs_own_sd_agreement"] = float((alt["tied_chunks"] == own["tied_chunks"]).mean())
    else:
        sb_mean, sb_std = R[rnd], s_w
        res["single_block_std_source"] = ("RMS of the within-round SDs, recovered from the pooled SD and the round means "
                                          "(the per-round file is not available)")
    views = {}
    pub_path = in_dir / f"e2_mamba1_{pub_tag}.csv"
    if pub_path.exists():
        pub = load_e2([pub_path]).set_index("key").loc[df["key"]].reset_index()
        same = all((pub[c].to_numpy() == df[c].to_numpy()).all() for c in ("L", "batch", "width", "fold", "layer", "group_id"))
        res["checks"]["published_same_calls"] = bool(same and np.allclose(pub["entropy_bits"], df["entropy_bits"]))
        views["published"] = pub
    else:
        res["checks"]["published_same_calls"] = f"{pub_path.name} not found; published view skipped"
    views["single_block"] = single_block_view(df, chunks, sb_mean, sb_std, n_block)
    views["round_median"] = round_median_view(df)
    views["pooled"] = df.copy()
    model = CostModel(cal, in_dir, chunks) if cal is not None else None
    if model is not None:
        res["refit"] = model.refit
    thr = HN.host_thresholds(in_dir).get(gpu)
    if thr is None:
        e5b_p = in_dir / f"e5b_{gpu}.json"
        thr = {"host_sync": float(json.loads(e5b_p.read_text())["host_sync_ms"]) if e5b_p.exists() else None,
               "device_reduction_L2048": None}
    dev_pts = HR.device_cost_points(in_dir)
    thresholds = {**thr, "sync_nominal_20us": HR.NOMINAL_SYNC_MS, "device_reduction_7p3us": dev_pts[0][1],
                  "device_reduction_42us": dev_pts[-1][1]}
    res["thresholds_us"] = {k: (None if v is None else 1000 * v) for k, v in thresholds.items()}
    choices = {}
    for v, vdf in views.items():
        if v == "published":
            idx = headroom_rng_index(pub_path.name)
            rng_h = np.random.default_rng([args.seed, idx if idx is not None else 900])
            boot_tag = f"mamba1_{pub_tag}"
        else:
            rng_h = np.random.default_rng([args.seed, 1000 + zlib.crc32(f"{tag}|{v}".encode()) % 100_000])
            boot_tag = f"mamba1_{tag}"        # common bootstrap weights for the views of this tag
        Rv = R if v == "round_median" else None     # the cross-round estimate does not depend on the view
        res["views"][v], choices[v] = view_metrics(vdf, chunks, v, f"{tag}|{v}", model, thresholds, dev_pts,
                                                   args.draws, rng_h, boot_tag, args.n_boot, Rv)
        if v == "round_median":
            bc = np.median(R, 0)                     # between-round CV of the round medians
            rsd = df[[f"lat_c{c}_rsd" for c in chunks]].to_numpy(float)
            res["views"][v]["noise"] = {"median_between_round_cv_pct": float(100 * np.median(rsd / bc)),
                                        "p90_between_round_cv_pct": float(100 * np.percentile(rsd / bc, 90)),
                                        "median_within_block_cv_pct": float(100 * np.median(s_w / R.mean(0)))}
    if "published" in views:
        res["checks"]["published_reproduction"] = check_published(pub_tag, cal, res["views"]["published"], in_dir)
    # comparison of the views
    rm = views["round_median"]
    comp = {"oracle": {v: oracle_comparison(chunks, rm, views[v], "round_median", v) for v in views if v != "round_median"}}
    if "published" in views:
        comp["oracle"]["single_block_vs_published"] = oracle_comparison(chunks, views["published"], views["single_block"],
                                                                         "published", "single_block")
    ch_r = [R[r].argmin(1) for r in range(R.shape[0])]
    comp["oracle_changes_between_rounds"] = {
        "share_calls_oracle_not_identical_in_all_rounds": float(np.mean([len(set(x)) > 1 for x in zip(*ch_r)])),
        "share_round_pairs_oracle_differs": float(np.mean([(ch_r[a] != ch_r[b]).mean() for a in range(len(ch_r))
                                                           for b in range(a + 1, len(ch_r))]))}
    se_w = s_w / math.sqrt(n_block)
    comp["block_noise_between_rounds"] = block_noise(R, se_w, df)
    # every round as the single block (round 0 is the first block after the call's inputs were loaded)
    slow = np.median(R / np.median(R, 0)[None], 2)                 # (rounds, calls): median over chunks
    per_round, per_round_oracle = {}, {}
    for r in range(R.shape[0]):
        if r == rnd:
            vr, met = views["single_block"], res["views"]["single_block"]
        else:
            std_r = rounds_jsonl_stats(path, df, chunks, r)[1] if js is not None else s_w
            vr = single_block_view(df, chunks, R[r], std_r, n_block)
            rng_h = np.random.default_rng([args.seed, 1000 + zlib.crc32(f"{tag}|single_block_r{r}".encode()) % 100_000])
            met, _ = view_metrics(vr, chunks, f"single_block_r{r}", f"{tag}|r{r}", model, thresholds, dev_pts,
                                  args.draws, rng_h, f"mamba1_{tag}", args.n_boot)
        per_round[f"r{r}"] = met
        per_round_oracle[f"r{r}"] = oracle_comparison(chunks, rm, vr, "round_median", f"single_block_r{r}")
    rows_r = compact(per_round)
    for k, v in rows_r.items():
        vals = [x for x in v.values() if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if len(vals) == len(v):
            v["mean_over_rounds"] = float(np.mean(vals))
    comp["single_block_by_round"] = {
        "note": ("each round's block means used as a single-block table; rounds run back to back per call, round 0 "
                 "directly after the call's inputs were loaded and its features computed"),
        "share_calls_slowed_gt5pct_all_chunks": {f"r{r}": float((slow[r] > 1.05).mean()) for r in range(R.shape[0])},
        "share_calls_slowed_gt2pct_all_chunks": {f"r{r}": float((slow[r] > 1.02).mean()) for r in range(R.shape[0])},
        "slowed_gt5pct_calls_by_batch": {f"r{r}": {int(b): int(((slow[r] > 1.05) & (df["batch"] == b).to_numpy()).sum())
                                                   for b in sorted(df["batch"].unique())} for r in range(R.shape[0])},
        "oracle_vs_round_median": per_round_oracle, "metrics": rows_r}
    comp["metrics"] = compact(res["views"])
    # the M2b quantities: change from the single-block oracle to the round-median oracle
    delta = {}
    for p in ("global_static", "best_static_per_batch_in_sample", "shape", "shape+entropy", "exact_shape_table_held_out",
              "cost_model", "cost_model_top2", "nearest_calibrated_key", "all_chunks_profiled"):
        a = res["views"]["single_block"]["predictors"].get(p)
        b = res["views"]["round_median"]["predictors"].get(p)
        if a is None or b is None:
            continue
        delta[p] = {f: {"single_block": a[f], "round_median": b[f], "change": b[f] - a[f]}
                    for f in ("mean_regret_pct", "p95_regret_pct", "ci_tie_share", "within_2pct_share", "within_5pct_share")}
    comp["single_block_to_round_median"] = delta
    for b in (1, 4, 16):
        sb, rmb = res["views"]["single_block"]["table_II"].get(b), res["views"]["round_median"]["table_II"].get(b)
        if sb and rmb:
            comp.setdefault("table_II_single_block_to_round_median", {})[b] = {
                "best_static_chunk": [sb["best_static_chunk"], rmb["best_static_chunk"]],
                "static_mean_pct": [sb["static"]["mean_regret_pct"], rmb["static"]["mean_regret_pct"]],
                "static_p95_pct": [sb["static"]["p95_regret_pct"], rmb["static"]["p95_regret_pct"]],
                "rule_p95_pct": [sb["entropy_rule"]["p95_regret_pct"], rmb["entropy_rule"]["p95_regret_pct"]]}
    res["comparison"] = comp
    res["seconds"] = round(time.time() - t0, 1)
    return res


# --------------------------------------------------------------------------- main
def _fmt(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x, nd) for x in v) + "]"
    if isinstance(v, str):
        return v
    return f"{v:.{nd}f}"


def print_report(out: dict) -> None:
    for tag, r in out["tags"].items():
        if "skipped" in r:
            print(f"{tag}: {r['skipped']}")
            continue
        print(f"\n== {tag} ({r['convention']}, {r['n_calls']} calls, {r['rounds']} rounds); published = {r['published_table']}")
        ch = r["checks"].get("published_reproduction")
        if ch:
            print(f"  reproduction of the published table: {ch['n_compared'] - ch['n_mismatches']}/{ch['n_compared']} values match")
            for m in ch["mismatches"][:10]:
                print("   ", m)
        rows = r["comparison"]["metrics"]
        views = [v for v in VIEWS if v in r["views"]]
        print("  " + f"{'metric':52s}" + "".join(f"{v:>14s}" for v in views))
        for k in ("global_static.mean_regret_pct", "global_static.p95_regret_pct", "global_static.ci_tie_share",
                  "global_static.within_2pct_share", "global_static.within_5pct_share",
                  "shape.mean_regret_pct", "shape.p95_regret_pct", "shape.ci_tie_share", "shape.within_2pct_share",
                  "shape.within_5pct_share", "shape+entropy.mean_regret_pct", "shape+entropy - shape.diff_pp",
                  "shape+entropy - shape.ci95_pp", "entropy_only.mean_regret_pct",
                  "cost_model.mean_regret_pct", "cost_model.p95_regret_pct", "cost_model.ci_tie_share",
                  "cost_model.within_2pct_share", "cost_model.within_5pct_share", "cost_model_top2.mean_regret_pct",
                  "nearest_calibrated_key.mean_regret_pct", "exact_shape_table_held_out.mean_regret_pct",
                  "tie_sets.mean_ci_tie_set_size", "tie_sets.mean_within_2pct_set_size",
                  "headroom.held_out_covered_mean_us", "headroom.held_out_covered_max_shape_us",
                  "headroom.split_half_covered_mean_us", "headroom.cross_round_held_out_covered_mean_us"):
            print("  " + f"{k:52s}" + "".join(f"{_fmt(rows[k].get(v)):>14s}" for v in views))
        oc = r["comparison"]["oracle"]["single_block"]
        print(f"  round-0 oracle vs round-median oracle: differs {100 * oc['share_oracle_differs']:.1f}%, outside the "
              f"round-median CI tie set {100 * oc['share_other_oracle_outside_reference_ci_tie_set']:.1f}%, outside 2% "
              f"{100 * oc['share_other_oracle_outside_2pct_of_reference']:.1f}%")
        sbr = r["comparison"]["single_block_by_round"]
        print("  single block by round: slowed >5% " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in
                                                               sbr["share_calls_slowed_gt5pct_all_chunks"].items()))
        for k in ("shape.mean_regret_pct", "shape.p95_regret_pct", "cost_model.mean_regret_pct",
                  "cost_model.p95_regret_pct", "headroom.held_out_covered_mean_us"):
            print("    " + f"{k:40s}" + "".join(f"{kk}={_fmt(vv)}  " for kk, vv in sbr["metrics"][k].items()))
        bn = r["comparison"]["block_noise_between_rounds"]
        print(f"  block noise between rounds: median |rel diff| {bn['median_abs_rel_diff_pct']:.2f}%, p90 "
              f"{bn['p90_abs_rel_diff_pct']:.2f}%, |z| > 1.96 in {100 * bn['share_abs_z_above_1p96']:.1f}% of pairs")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--out", type=Path, default=None, help="output JSON (default: <in-dir>/r4_rounds_analysis.json)")
    p.add_argument("--tags", default=DEFAULT_TAGS, help="comma list of multi-round E2 tags (e2_mamba1_<tag>.csv)")
    p.add_argument("--published", default=None,
                   help="tag=published_tag overrides, e.g. pro6000_r4plain=pro6000 (default: <gpu>_r3spin / <gpu>)")
    p.add_argument("--cost-tags", default=None,
                   help="tag=calibration_tag overrides of the frozen cost model (default: <gpu>_r3spin / <gpu>)")
    p.add_argument("--single-round", type=int, default=0, help="round used as the single-block oracle")
    p.add_argument("--draws", type=int, default=200, help="split-half draws (headroom_noise)")
    p.add_argument("--seed", type=int, default=HEADROOM_SEED)
    p.add_argument("--n-boot", type=int, default=2000, help="prompt-cluster bootstrap replicates (e3b_information)")
    args = p.parse_args(argv)
    t0 = time.time()
    out = {"experiment": "r4_rounds_analysis (review tpds-r4 M2b, M2c, M8a; E19)", "created_utc": utc_now(),
           "definitions": __doc__, "settings": {"draws": args.draws, "seed": args.seed, "n_boot": args.n_boot,
                                                "boot_seed": BOOT_SEED, "single_round": args.single_round,
                                                "tie_margins": list(MARGINS)},
           "units": "regrets and shares in percent or fractions as named; headroom in us per call", "tags": {}}
    for tag in [t for t in args.tags.split(",") if t]:
        print(f"analyzing {tag} ...", flush=True)
        out["tags"][tag] = analyze_tag(tag, args)
    # cross-tag summary: the writer's table (round-median vs single block vs published, per tag)
    summ = {}
    for tag, r in out["tags"].items():
        if "skipped" in r:
            continue
        rows = r["comparison"]["metrics"]
        summ[tag] = {k: rows[k] for k in rows}
        oc = r["comparison"]["oracle"]["single_block"]
        summ[tag]["oracle.round0_vs_round_median"] = {
            "share_differs": oc["share_oracle_differs"],
            "share_outside_round_median_ci_tie_set": oc["share_other_oracle_outside_reference_ci_tie_set"],
            "share_outside_2pct": oc["share_other_oracle_outside_2pct_of_reference"],
            "share_outside_5pct": oc["share_other_oracle_outside_5pct_of_reference"],
            "regret_of_round0_oracle_on_round_medians_pct": oc["regret_of_other_oracle_on_reference_pct"]}
        if "published" in r["comparison"]["oracle"]:
            op = r["comparison"]["oracle"]["published"]
            summ[tag]["oracle.published_vs_round_median"] = {
                "share_differs": op["share_oracle_differs"],
                "share_outside_round_median_ci_tie_set": op["share_other_oracle_outside_reference_ci_tie_set"],
                "share_outside_2pct": op["share_other_oracle_outside_2pct_of_reference"]}
        summ[tag]["block_noise_between_rounds"] = {k: v for k, v in r["comparison"]["block_noise_between_rounds"].items()
                                                   if k != "by_batch"}
        summ[tag]["oracle_changes_between_rounds"] = r["comparison"]["oracle_changes_between_rounds"]
        sbr = r["comparison"]["single_block_by_round"]
        summ[tag]["single_block_by_round"] = {
            "share_calls_slowed_gt5pct_all_chunks": sbr["share_calls_slowed_gt5pct_all_chunks"],
            **{k: sbr["metrics"][k] for k in ("global_static.mean_regret_pct", "global_static.p95_regret_pct",
                                              "shape.mean_regret_pct", "shape.p95_regret_pct", "shape.ci_tie_share",
                                              "shape.within_2pct_share", "cost_model.mean_regret_pct",
                                              "cost_model.p95_regret_pct", "cost_model.within_2pct_share",
                                              "headroom.held_out_covered_mean_us",
                                              "headroom.held_out_covered_max_shape_us") if k in sbr["metrics"]},
            "oracle_differs_from_round_median": {k: v["share_oracle_differs"] for k, v in sbr["oracle_vs_round_median"].items()},
            "oracle_outside_2pct_of_round_median": {k: v["share_other_oracle_outside_2pct_of_reference"]
                                                    for k, v in sbr["oracle_vs_round_median"].items()}}
    out["comparison"] = summ
    out["runtime_s"] = round(time.time() - t0, 1)
    path = args.out or (args.in_dir / "r4_rounds_analysis.json")
    write_json(path, out)
    print_report(out)
    print(f"\nwrote {path} ({out['runtime_s']} s)")


if __name__ == "__main__":
    main()
