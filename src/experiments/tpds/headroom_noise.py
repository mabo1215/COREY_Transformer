"""Selection noise in the oracle headroom over a shape-keyed chunk table (review items M3b, M3c).

The oracle headroom of a chunk table keyed on the exact shape (L, batch) is

    h = mean over calls of [ latency(table chunk) - latency(per-call oracle chunk) ],

where every latency is the mean of ``repeats`` (= 50) timed launches. Two estimation biases act
on the naive estimate that uses the same timing means for every step:

  * the per-call oracle is the argmin over noisy means, so its latency is biased low and the
    headroom is biased HIGH (winner's curse; ties between chunks count as headroom);
  * a table chosen on the same calls it is scored on is biased low, so the headroom is biased
    LOW. The held-out protocol of ``analysis.cv_exact_headroom`` (table chosen on the other
    prompt folds) removes this second bias only.

This script reports, per dataset (Mamba-1 and Mamba-2 on the RTX PRO 6000 and the RTX 3090,
the reference CUDA scan on the RTX 3090, and the 1.4B / 2.8B Mamba-1 checkpoints on the
RTX 3090):

 1. ``naive``: the in-sample exact-shape headroom and the held-out headroom of
    ``cv_exact_headroom`` (imported and re-run; covered calls are served by their own shape entry,
    fallback calls by the bucket or global entry), with the per-shape maximum and the number of
    shapes above one host synchronization (the value measured on that GPU in E4 / E5b). A
    vectorized re-implementation of the same protocol is checked against the imported function.

 2. ``split_half`` (primary noise correction): the 50 repeats of every timing are split into two
    halves of 25; the table (in-sample or held-out) and the per-call oracle are chosen on half A and
    scored on the independent half B, and vice versa (cross-fitting; the two directions are
    averaged per call). Only the mean and standard deviation of each timing were stored, so the
    split is simulated from the exact conditional law of the two half-means given the full-sample
    mean under i.i.d. Gaussian repeats: A = m + D, B = m - D with D ~ N(0, s^2 / n), n = repeats.
    (If the halves are i.i.d. N(mu, 2 s^2 / n), then m = (A + B) / 2 and D = (A - B) / 2 are
    independent and D ~ N(0, s^2 / n).) Because the selection never sees the scoring half, within-block noise
    between truly tied chunks contributes zero in expectation; the estimate is conservative
    (a lower bound on the true headroom in expectation), since an oracle chosen on 25 repeats is
    itself sometimes wrong. The reported interval is the 2.5 / 97.5 percentile over B random
    splits.

 3. ``split_half_plugin``: the construction as first specified, two independent replicate means
    A, B ~ N(m, 2 s^2 / n) around the observed means, choose on A, score on B (cross-fitted). It
    treats the observed means as the truth; because those means carry the same noise that drives
    the selection, it retains part of the winner's curse and sits between ``naive`` and
    ``split_half``. Reported for transparency only.

 4. ``bootstrap_bias``: parametric bootstrap of the naive estimator, m* = m + E, E ~ N(0, s^2 / n);
    bias = mean(naive(m*)) - naive(m); bias-corrected headroom = naive(m) - bias.

 5. ``null_zero_headroom``: the naive headroom that timing noise alone produces when the true
    headroom is exactly zero. Null truth: mu0[i, c] = mean over the calls of i's exact shape of
    m[., c] + (mean_c m[i, c] - mean over the shape of the same). Every call keeps its own latency
    level, and all calls of a shape share one chunk profile, so the shape's best chunk is every
    call's best chunk (h = 0, the premise of the value-invariance proposition). Data are drawn as
    X0 = mu0 + E with each call's own measured noise and the naive estimator is applied. The
    one-sided p-value is the share of null draws whose naive headroom reaches the observed one.
    For held-out fallback calls the null truth is not zero (a bucket entry can be wrong for an
    untuned shape), so the null is reported for covered calls and in-sample only.

Noise shares: ``split`` = 1 - split_half / naive (clipped to [0, 1]; an upper bound on the share of
the naive headroom due to within-block selection noise, since split_half is conservative),
``null`` = null mean / naive (clipped; share of the observed headroom that pure noise reproduces),
and ``bootstrap`` = bias / naive.

 6. ``block_noise``: items 2-5 use the within-block standard error, i.e. they assume the 50
    repeats of a timing are independent draws around the call's true latency. The repeats run back
    to back in one block, so anything that shifts a whole block (clock or power state, host load,
    the block's position within the call) is invisible to that error and survives any split of the
    repeats. It is measured from independent re-timings of identical kernels: the reference-scan
    table times the stock-dispatch configuration three times per call, and the value-invariance
    and placement experiments time every cell in several rounds (``block_noise_calibration_rounds``).
    The null and the bootstrap bias correction are repeated with a block-level term, Gaussian with
    relative SD rho in {0.5, 1, 2}% (sensitivity), and, for the reference scan, resampled from the
    triplicate deviations of the call's (batch, length bucket) stratum (empirical).

 7. ``degraded_period_filter``: calls timed while the device or host was slowed are slower at every
    chunk than the other calls of their shape. Calls whose median-over-chunks latency ratio to the
    shape median exceeds 1.10 or 1.05 are removed and the naive protocol is re-run.

Thresholds: the host synchronization per GPU (``host_sync_ms`` of ``e5b_<host>.json``), and, for the
RTX 3090 datasets, the marginal cost of one device-side reduction kernel before the scan at
L = 2048 (``e9_placement_vgpu3090.json``), as a reference only (it grows with L).

Output: ``headroom_noise.json`` in the results directory (all latencies in ms and us, keys suffixed
``_ms`` / ``_us``). CPU only; fixed RNG seed.

Usage:  python3 -m src.experiments.tpds.headroom_noise [--draws 200] [--seed 20260930]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.tpds.analysis import chunk_columns, cv_exact_headroom, latency_matrix, load_e2
from src.experiments.tpds.common import RESULTS_DIR, write_json

# name, label, csv file, GPU tag (selects the host synchronization threshold)
DATASETS = [
    ("mamba1_pro6000", "Mamba-1 370M, Triton scan, RTX PRO 6000", "e2_mamba1_pro6000.csv", "pro6000"),
    ("mamba1_vgpu3090", "Mamba-1 370M, Triton scan, RTX 3090", "e2_mamba1_vgpu3090.csv", "vgpu3090"),
    ("mamba2_pro6000", "Mamba-2 370M, SSD scan, RTX PRO 6000", "e2_mamba2_pro6000.csv", "pro6000"),
    ("mamba2_vgpu3090", "Mamba-2 370M, SSD scan, RTX 3090", "e2_mamba2_vgpu3090.csv", "vgpu3090"),
    ("refscan_vgpu3090", "Mamba-1 370M, reference CUDA scan, RTX 3090", "e8_refscan_vgpu3090.csv", "vgpu3090"),
    ("mamba1_1p4b_vgpu3090", "Mamba-1 1.4B, Triton scan, RTX 3090, batch 1", "e2_mamba1_vgpu3090_1p4b.csv", "vgpu3090"),
    ("mamba1_2p8b_vgpu3090", "Mamba-1 2.8B, Triton scan, RTX 3090, batch 1", "e2_mamba1_vgpu3090_2p8b.csv", "vgpu3090"),
    # tpds-r3 E14/E15: re-timed larger checkpoints (batch 1, 4, 16) in device time and launch-inclusive,
    # and the 370M calls in device time
    ("mamba1_1p4b_r3_vgpu3090", "Mamba-1 1.4B, Triton scan, RTX 3090, device time", "e2_mamba1_vgpu3090_1p4b_r3.csv", "vgpu3090"),
    ("mamba1_2p8b_r3_vgpu3090", "Mamba-1 2.8B, Triton scan, RTX 3090, device time", "e2_mamba1_vgpu3090_2p8b_r3.csv", "vgpu3090"),
    ("mamba1_1p4b_r3plain_vgpu3090", "Mamba-1 1.4B, Triton scan, RTX 3090, launch-inclusive", "e2_mamba1_vgpu3090_1p4b_r3plain.csv", "vgpu3090"),
    ("mamba1_2p8b_r3plain_vgpu3090", "Mamba-1 2.8B, Triton scan, RTX 3090, launch-inclusive", "e2_mamba1_vgpu3090_2p8b_r3plain.csv", "vgpu3090"),
    ("mamba1_r3spin_vgpu3090", "Mamba-1 370M, Triton scan, RTX 3090, device time", "e2_mamba1_vgpu3090_r3spin.csv", "vgpu3090"),
    # tpds-r3 E17 (M2b): production kernels on larger checkpoints, RTX 3090, device time
    ("mamba2_2p7b_r3_vgpu3090", "Mamba-2 2.7B, SSD scan, RTX 3090, device time", "e2_mamba2_vgpu3090_2p7b_r3.csv", "vgpu3090"),
    ("gdn_1p4b_r3_vgpu3090", "Gated DeltaNet 1.4B, FLA chunk kernel, RTX 3090, device time", "e2_gdn_vgpu3090_1p4b_r3.csv", "vgpu3090"),
]

# Existing published values used as a reproduction check: (dataset, source json, path, recomputed path)
REFERENCE_CHECKS = [
    ("mamba1_pro6000", "e5b_pro6000.json", ("by_key_mode", "exact", "headroom_pooled_ms"), ("naive", "in_sample", "all", "mean_ms")),
    ("mamba1_pro6000", "e5b_pro6000.json", ("by_key_mode", "exact", "max_key_headroom_ms"), ("naive", "in_sample", "all", "max_shape_ms")),
    ("mamba1_vgpu3090", "e5b_vgpu3090.json", ("by_key_mode", "exact", "headroom_pooled_ms"), ("naive", "in_sample", "all", "mean_ms")),
    ("mamba1_vgpu3090", "e5b_vgpu3090.json", ("by_key_mode", "exact", "max_key_headroom_ms"), ("naive", "in_sample", "all", "max_shape_ms")),
    ("refscan_vgpu3090", "e8_analysis_vgpu3090.json", ("headroom", "exact", "mean_ms"), ("naive", "in_sample", "all", "mean_ms")),
    ("refscan_vgpu3090", "e8_analysis_vgpu3090.json", ("headroom", "exact", "max_key_ms"), ("naive", "in_sample", "all", "max_shape_ms")),
]
for _ds, _js, _key in (("mamba1_pro6000", "e5b_pro6000.json", ("exact_cv_headroom",)),
                       ("mamba1_vgpu3090", "e5b_vgpu3090.json", ("exact_cv_headroom",)),
                       ("refscan_vgpu3090", "e8_analysis_vgpu3090.json", ("headroom", "exact_cv"))):
    REFERENCE_CHECKS += [
        (_ds, _js, _key + ("mean_ms",), ("naive", "held_out", "all", "mean_ms")),
        (_ds, _js, _key + ("covered", "mean_ms"), ("naive", "held_out", "covered", "mean_ms")),
        (_ds, _js, _key + ("covered", "max_key_ms"), ("naive", "held_out", "covered", "max_shape_ms")),
        (_ds, _js, _key + ("fallback", "mean_ms"), ("naive", "held_out", "fallback", "mean_ms")),
        (_ds, _js, _key + ("fallback", "max_key_ms"), ("naive", "held_out", "fallback", "max_shape_ms")),
    ]


# ----------------------------------------------------------------------------- data and structure

def load_dataset(path: Path) -> pd.DataFrame:
    df = load_e2([path])
    if "regimes" in df.columns:  # the reference-scan table also holds the twelve E4 tuning keys
        df = df[df["regimes"] != "e4_key"]
    df = df.reset_index(drop=True)
    # Chunks the kernel cannot run at a call's shape (Mamba-2 SSD beyond its 32-bit indexing) are stored as
    # +inf. The estimators average and difference latencies, so they get a finite stand-in instead: ten times
    # the slowest available chunk of that call with zero spread, which no table or oracle ever selects.
    chunks = chunk_columns(df)
    mean = df[[f"lat_c{c}_mean" for c in chunks]].to_numpy(dtype=float)
    bad = ~np.isfinite(mean)
    if bad.any():
        fill = 10.0 * np.nanmax(np.where(bad, np.nan, mean), axis=1)
        for j, c in enumerate(chunks):
            rows = np.where(bad[:, j])[0]
            df.loc[rows, f"lat_c{c}_mean"] = fill[rows]
            df.loc[rows, f"lat_c{c}_std"] = 0.0
        df.attrs["unavailable_cells"] = int(bad.sum())
    return df


class TableStructure:
    """Index structure of the exact-shape table protocols; it depends only on the call shapes.

    Mirrors ``analysis.cv_exact_headroom``: for every prompt fold, a call is served by the entry of
    its exact shape (L, batch) chosen on the other folds if that shape occurs there ("covered"),
    else by the (length bucket, batch, L mod 16 == 0) entry, else by the single chunk with the
    least summed relative latency over the other folds. The in-sample table uses all calls.
    """

    def __init__(self, df: pd.DataFrame):
        n = len(df)
        self.n = n
        L, b = df["L"].to_numpy(), df["batch"].to_numpy()
        aligned = (L % 16 == 0).astype(int)
        self.exact_id, exact_keys = pd.MultiIndex.from_arrays([L, b]).factorize()
        self.bucket_id, _ = pd.MultiIndex.from_arrays([df["L_bucket"].to_numpy(), b, aligned]).factorize()
        self.exact_keys = [(int(x), int(y)) for x, y in exact_keys]
        self.n_exact, self.n_bucket = len(exact_keys), int(self.bucket_id.max()) + 1
        folds = df["fold"].to_numpy()
        self.folds = np.unique(folds)
        self.fold_idx = np.searchsorted(self.folds, folds)
        F = len(self.folds)
        self.F = F
        train = self.fold_idx[None, :] != np.arange(F)[:, None]                     # (F, n)
        self.W_in = (self.exact_id[None, :] == np.arange(self.n_exact)[:, None]).astype(float)   # (Ke, n)
        W_b = (self.bucket_id[None, :] == np.arange(self.n_bucket)[:, None]).astype(float)       # (Kb, n)
        self.W_cv_exact = (train[:, None, :] * self.W_in[None]).reshape(F * self.n_exact, n)
        self.W_cv_bucket = (train[:, None, :] * W_b[None]).reshape(F * self.n_bucket, n)
        self.W_cv_global = train.astype(float)                                        # (F, n)
        cnt_e = self.W_cv_exact.sum(1).reshape(F, self.n_exact)
        cnt_b = self.W_cv_bucket.sum(1).reshape(F, self.n_bucket)
        has_e = cnt_e[self.fold_idx, self.exact_id] > 0
        has_b = cnt_b[self.fold_idx, self.bucket_id] > 0
        self.source = np.where(has_e, 0, np.where(has_b, 1, 2))                      # 0 exact, 1 bucket, 2 global
        self.covered = self.source == 0
        self.shape_counts = np.bincount(self.exact_id, minlength=self.n_exact).astype(float)
        cov_per_shape = np.bincount(self.exact_id, weights=self.covered.astype(float), minlength=self.n_exact)
        assert np.all((cov_per_shape == 0) | (cov_per_shape == self.shape_counts)), "shape both covered and not"
        self.shape_covered = cov_per_shape > 0

    def choose(self, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-call table chunk (in-sample, held-out) and per-call oracle chunk, all chosen on ``sel``."""
        F, Ke, Kb, C = self.F, self.n_exact, self.n_bucket, sel.shape[1]
        oracle = sel.argmin(1)
        # argmin of a group sum equals argmin of the group mean (same count for every chunk)
        j_in = (self.W_in @ sel).argmin(1)[self.exact_id]
        ce = (self.W_cv_exact @ sel).reshape(F, Ke, C).argmin(2)
        cb = (self.W_cv_bucket @ sel).reshape(F, Kb, C).argmin(2)
        cg = (self.W_cv_global @ (sel / sel.min(1, keepdims=True))).argmin(1)
        f = self.fold_idx
        j_cv = np.where(self.source == 0, ce[f, self.exact_id],
                        np.where(self.source == 1, cb[f, self.bucket_id], cg[f]))
        return j_in, j_cv, oracle

    def headroom(self, sel: np.ndarray, score: np.ndarray) -> dict:
        """Per-call headroom of the tables over the per-call oracle, both chosen on sel, scored on score."""
        j_in, j_cv, oracle = self.choose(sel)
        r = np.arange(self.n)
        ref = score[r, oracle]
        return {"in": score[r, j_in] - ref, "cv": score[r, j_cv] - ref, "ref": ref}

    def cross_fit(self, a: np.ndarray, b: np.ndarray) -> dict:
        """Choose on a, score on b and vice versa; average the two directions per call."""
        h1, h2 = self.headroom(a, b), self.headroom(b, a)
        return {k: 0.5 * (h1[k] + h2[k]) for k in h1}

    def per_shape(self, h: np.ndarray) -> np.ndarray:
        return np.bincount(self.exact_id, weights=h, minlength=self.n_exact) / self.shape_counts


# ----------------------------------------------------------------------------- summaries

def _ms_us(name: str, v: float | None) -> dict:
    if v is None or not np.isfinite(v):
        return {f"{name}_ms": None, f"{name}_us": None}
    return {f"{name}_ms": float(v), f"{name}_us": float(1000.0 * v)}


def summarize_calls(h: np.ndarray, ref: np.ndarray, st: TableStructure, sel: np.ndarray | None,
                    thresholds: dict) -> dict:
    """Mean, relative mean, per-shape maximum and shapes above each threshold for a call subset."""
    if sel is None:
        sel = np.ones(len(h), dtype=bool)
    if not sel.any():
        return {"n_calls": 0, "n_shapes": 0}
    ps = pd.Series(h[sel]).groupby(st.exact_id[sel]).mean()
    out = {"n_calls": int(sel.sum()), "n_shapes": int(len(ps)), **_ms_us("mean", h[sel].mean()),
           "mean_pct": float(100 * np.mean(h[sel] / ref[sel])), **_ms_us("max_shape", ps.max())}
    for tname, tv in thresholds.items():
        if tv is None:
            continue
        above = ps[ps > tv]
        out[f"n_shapes_above_{tname}"] = int(len(above))
        out[f"calls_on_shapes_above_{tname}"] = int(st.shape_counts[above.index.to_numpy()].sum())
    return out


def summarize_protocol(hd: dict, st: TableStructure, thresholds: dict) -> dict:
    return {"in_sample": {"all": summarize_calls(hd["in"], hd["ref"], st, None, thresholds)},
            "held_out": {"all": summarize_calls(hd["cv"], hd["ref"], st, None, thresholds),
                         "covered": summarize_calls(hd["cv"], hd["ref"], st, st.covered, thresholds),
                         "fallback": summarize_calls(hd["cv"], hd["ref"], st, ~st.covered, thresholds)}}


class DrawCollector:
    """Accumulates per-draw headroom statistics of one estimator and summarizes them."""

    def __init__(self, st: TableStructure):
        self.st = st
        self.rows = {k: [] for k in ("in_all", "cv_all", "cv_covered", "cv_fallback")}
        self.pct = {k: [] for k in self.rows}
        self.shape_in, self.shape_cv = [], []

    def add(self, hd: dict):
        st = self.st
        for k, h, m in (("in_all", hd["in"], None), ("cv_all", hd["cv"], None),
                        ("cv_covered", hd["cv"], st.covered), ("cv_fallback", hd["cv"], ~st.covered)):
            m = np.ones(st.n, dtype=bool) if m is None else m
            self.rows[k].append(float(h[m].mean()) if m.any() else np.nan)
            self.pct[k].append(float(100 * np.mean(h[m] / hd["ref"][m])) if m.any() else np.nan)
        self.shape_in.append(st.per_shape(hd["in"]))
        self.shape_cv.append(st.per_shape(hd["cv"]))

    def summary(self, thresholds: dict, observed: dict | None = None) -> dict:
        st = self.st
        out = {}
        for k, vals in self.rows.items():
            v = np.asarray(vals, dtype=float)
            if np.all(np.isnan(v)):
                out[k] = {"n_calls": 0}
                continue
            shapes = np.asarray(self.shape_in if k == "in_all" else self.shape_cv)       # (draws, Ke)
            smask = {"in_all": np.ones(st.n_exact, bool), "cv_all": np.ones(st.n_exact, bool),
                     "cv_covered": st.shape_covered, "cv_fallback": ~st.shape_covered}[k]
            sm = shapes[:, smask]
            shape_mean = sm.mean(0)                          # per-shape headroom averaged over draws
            per_draw_max = sm.max(1)
            rec = {"n_draws": int(len(v)), **_ms_us("mean", v.mean()), **_ms_us("ci95_lo", np.percentile(v, 2.5)),
                   **_ms_us("ci95_hi", np.percentile(v, 97.5)), "mean_pct": float(np.nanmean(self.pct[k])),
                   **_ms_us("max_shape_of_draw_mean", shape_mean.max()),
                   **_ms_us("max_shape_per_draw_mean", per_draw_max.mean()),
                   **_ms_us("max_shape_per_draw_ci95_hi", np.percentile(per_draw_max, 97.5))}
            for tname, tv in thresholds.items():
                if tv is None:
                    continue
                rec[f"n_shapes_above_{tname}_draw_mean"] = int((shape_mean > tv).sum())
                rec[f"n_shapes_above_{tname}_per_draw_mean"] = float((sm > tv).sum(1).mean())
            if observed is not None and k in observed and observed[k] is not None:
                rec["p_value_null_ge_observed"] = float((1 + (v >= observed[k]).sum()) / (1 + len(v)))
            out[k] = rec
        return out

    def shape_draw_mean(self, kind: str) -> np.ndarray:
        return np.asarray(self.shape_in if kind == "in" else self.shape_cv).mean(0)


# ----------------------------------------------------------------------------- block-level timing noise
#
# The 50 repeats of one timing run back to back in one block. Anything that shifts a whole block
# (clock or power state, co-running host work, the position of the block within the call) is
# invisible in the within-block standard deviation, and a split of the repeats cannot remove it,
# because both halves share the block. It is measured here from independent re-timings of the
# same kernel on the same input: the reference-scan table times the stock-dispatch configuration
# three times per call (installed stock kernel, rebuilt stock kernel, and the runtime-chunk kernel
# at the stock chunk), and the value-invariance and placement experiments time every cell in
# several rounds.

RHO_GRID = (0.005, 0.01, 0.02)      # relative block-level noise (SD / latency) of the sensitivity analysis


def _robust_sd(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def refscan_replicates(df: pd.DataFrame) -> np.ndarray | None:
    """(n, 3) block means of the stock-dispatch configuration (stock, rebuilt stock, runtime chunk)."""
    need = ["lat_stock_mean", "lat_stock_rebuilt_mean", "stock_chunk"]
    if not all(c in df.columns for c in need):
        return None
    sc = df["stock_chunk"].to_numpy().astype(int)
    chunk_col = np.array([df.at[i, f"lat_c{c}_mean"] for i, c in enumerate(sc)], dtype=float)
    return np.stack([df["lat_stock_mean"].to_numpy(float), df["lat_stock_rebuilt_mean"].to_numpy(float), chunk_col], 1)


def calibrate_refscan(df: pd.DataFrame) -> tuple[dict, dict, np.ndarray]:
    """Block-to-block noise of identical kernels from the reference-scan triplicates.

    Returns a summary, the pooled per-block relative deviations per stratum (batch, length
    bucket), and the stratum index of every call. A deviation is r_k / mean(r) - 1 of one of the
    three re-timings, scaled by sqrt(3/2) to undo the shrinkage of deviations from a 3-sample mean.
    """
    rep = refscan_replicates(df)
    n = df["repeats"].to_numpy(float)
    sc = df["stock_chunk"].to_numpy().astype(int)
    sd = np.stack([df["lat_stock_std"].to_numpy(float), df["lat_stock_rebuilt_std"].to_numpy(float),
                   np.array([df.at[i, f"lat_c{c}_std"] for i, c in enumerate(sc)], dtype=float)], 1)
    se = sd / np.sqrt(n)[:, None]
    pairs = {"stock_vs_rebuilt": (0, 1), "rebuilt_vs_runtime_chunk": (1, 2), "stock_vs_runtime_chunk": (0, 2)}
    out = {"n_calls": int(len(df)), "note": "three independent timings (50 repeats each) of the identical kernel "
           "configuration per call; if the within-block standard error described all noise, z would be N(0, 1)",
           "pairs": {}}
    for name, (a, b) in pairs.items():
        d = rep[:, a] - rep[:, b]
        lvl = 0.5 * (rep[:, a] + rep[:, b])
        z = d / np.sqrt(se[:, a] ** 2 + se[:, b] ** 2)
        rel = d / lvl
        out["pairs"][name] = {"median_abs_rel_diff_pct": float(100 * np.median(np.abs(rel))),
                              "p90_abs_rel_diff_pct": float(100 * np.percentile(np.abs(rel), 90)),
                              **_ms_us("median_abs_diff", np.median(np.abs(d))),
                              **_ms_us("median_within_block_se_of_diff", np.median(np.sqrt(se[:, a] ** 2 + se[:, b] ** 2))),
                              "robust_sd_z": _robust_sd(z), "share_abs_z_above_1p96": float(np.mean(np.abs(z) > 1.96)),
                              "mean_rel_diff_pct": float(100 * rel.mean())}
    rel_all = np.concatenate([(rep[:, a] - rep[:, b]) / (0.5 * (rep[:, a] + rep[:, b])) for a, b in pairs.values()])
    out["per_block_relative_sd_robust_pct"] = float(100 * _robust_sd(rel_all) / np.sqrt(2))
    dev = np.sqrt(1.5) * (rep / rep.mean(1, keepdims=True) - 1.0)
    stratum, strata = pd.MultiIndex.from_arrays([df["batch"].to_numpy(), df["L_bucket"].to_numpy()]).factorize()
    pools = {s: dev[stratum == s].ravel() for s in range(len(strata))}
    out["by_stratum"] = [{"batch": int(b), "L_bucket": int(lb), "n_calls": int((stratum == s).sum()),
                          "rms_rel_block_dev_pct": float(100 * np.sqrt(np.mean(pools[s] ** 2))),
                          "robust_rel_block_sd_pct": float(100 * _robust_sd(pools[s])),
                          "share_blocks_more_than_10pct_slow": float(np.mean(pools[s] > 0.10))}
                         for s, (b, lb) in enumerate(strata)]
    out["rms_rel_block_dev_pct"] = float(100 * np.sqrt(np.mean(dev ** 2)))
    return out, pools, stratum


def calibrate_rounds(in_dir: Path) -> dict:
    """Round-to-round (block-level) noise of identical cells in the value-invariance / placement runs."""
    out = {}
    for tag, fname in (("pro6000", "e1_pro6000.jsonl"), ("vgpu3090", "e1_vgpu3090.jsonl"),
                       ("vgpu3090_e9", "e9_placement_vgpu3090.jsonl")):
        p = in_dir / fname
        if not p.exists():
            continue
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        df = pd.DataFrame([r for r in rows if r.get("kind") == "timing"])
        if df.empty:
            continue
        df["se"] = df["std_ms"] / np.sqrt(df["n"])
        keys = [c for c in ("family", "design", "L", "batch", "chunk", "input", "mode") if c in df.columns]
        g = df.groupby(keys).agg(k=("mean_ms", "size"), m=("mean_ms", "mean"), var=("mean_ms", "var"),
                                 se2=("se", lambda x: float(np.mean(np.square(x)))))
        g = g[g["k"] > 1]
        g["block_var"] = (g["var"] - g["se2"]).clip(lower=0)      # between-round variance beyond within-block noise
        g["ratio"] = np.sqrt(g["var"] / g["se2"])
        rec = {}
        for (fam, L), h in g.groupby(["family", "L"]):
            rec[f"{fam}|L{int(L)}"] = {
                "n_cells": int(len(h)), "rounds": int(h["k"].median()), **_ms_us("median_latency", h["m"].median()),
                "pooled_rel_block_sd_pct": float(100 * np.sqrt(np.mean(h["block_var"] / h["m"] ** 2))),
                "median_rel_block_sd_pct": float(100 * np.median(np.sqrt(h["block_var"]) / h["m"])),
                "median_sd_between_over_within_se": float(h["ratio"].median())}
        out[tag] = {"source": fname, "repeats_per_block": int(df["n"].median()), "by_family_length": rec}
    return out


def gaussian_block_noise(se: np.ndarray, rho: float):
    """Within-block standard error plus a Gaussian block-level term with relative SD rho."""
    def draw(level: np.ndarray, rng) -> np.ndarray:
        return rng.standard_normal(level.shape) * np.sqrt(se ** 2 + (rho * level) ** 2)
    return draw


def empirical_block_noise(pools: dict, stratum: np.ndarray):
    """Multiplicative block noise resampled from the triplicate deviations of the call's stratum."""
    def draw(level: np.ndarray, rng) -> np.ndarray:
        e = np.empty_like(level)
        for s, pool in pools.items():
            idx = np.where(stratum == s)[0]
            e[idx] = pool[rng.integers(0, len(pool), size=(len(idx), level.shape[1]))]
        return level * e
    return draw


def run_noise_model(st: TableStructure, m: np.ndarray, mu0: np.ndarray, noise, draws: int, rng,
                    thresholds: dict, obs: dict) -> dict:
    """Null (zero true headroom) and bootstrap bias correction of the naive estimator under one noise model."""
    boot, null = DrawCollector(st), DrawCollector(st)
    for _ in range(draws):
        mb = m + noise(m, rng)
        boot.add(st.headroom(mb, mb))
        x0 = mu0 + noise(mu0, rng)
        null.add(st.headroom(x0, x0))
    bs = boot.summary(thresholds)
    ns = null.summary(thresholds, observed=obs)
    out = {"null_zero_headroom": {k: ns[k] for k in ("in_all", "cv_covered")}, "bias_corrected": {}}
    for k in ("in_all", "cv_all", "cv_covered", "cv_fallback"):
        if obs.get(k) is None or bs[k].get("mean_ms") is None:
            continue
        b = bs[k]["mean_ms"] - obs[k]
        out["bias_corrected"][k] = {**_ms_us("bias", b), **_ms_us("corrected", obs[k] - b),
                                    "bias_share_of_naive": _share(b, obs[k], clip=False)}
    out["_null_shapes_cv"] = null.shape_draw_mean("cv")
    return out


def _get(d: dict, path: tuple):
    for p in path:
        d = d[p]
    return d


def _share(num: float, den: float, clip: bool = True) -> float | None:
    if den is None or den <= 0:
        return None
    x = num / den
    return float(min(max(x, 0.0), 1.0)) if clip else float(x)


# ----------------------------------------------------------------------------- per dataset

def analyze_dataset(label: str, path: Path, thresholds: dict, draws: int, rng) -> dict:
    df = load_dataset(path)
    chunks = chunk_columns(df)
    m = latency_matrix(df, chunks)
    s = df[[f"lat_c{c}_std" for c in chunks]].to_numpy(dtype=float)
    reps = df["repeats"].to_numpy(dtype=float)
    se = s / np.sqrt(reps)[:, None]                                  # standard error of each timing mean
    st = TableStructure(df)

    # 1. naive estimates; held-out via the imported protocol and the vectorized re-implementation
    ref_cv = cv_exact_headroom(df, chunks, threshold_ms=thresholds["host_sync"])
    naive_hd = st.headroom(m, m)
    naive = summarize_protocol(naive_hd, st, thresholds)
    fb = {"exact": int((st.source == 0).sum()), "bucket_aligned": int((st.source == 1).sum()),
          "global": int((st.source == 2).sum())}
    impl_check = {
        "cv_exact_headroom_mean_ms": ref_cv["mean_ms"], "vectorized_mean_ms": naive["held_out"]["all"]["mean_ms"],
        "cv_exact_headroom_covered_mean_ms": ref_cv["covered"]["mean_ms"],
        "vectorized_covered_mean_ms": naive["held_out"]["covered"]["mean_ms"],
        "cv_exact_headroom_max_key_ms": ref_cv["max_key_ms"], "vectorized_max_key_ms": naive["held_out"]["all"]["max_shape_ms"],
        "fallback_counts_match": ref_cv["fallbacks"] == fb,
        "n_keys_above_host_sync_match": ref_cv.get("n_keys_above_threshold") ==
        naive["held_out"]["all"].get("n_shapes_above_host_sync")}
    impl_check["max_abs_diff_ms"] = float(max(abs(impl_check["cv_exact_headroom_mean_ms"] - impl_check["vectorized_mean_ms"]),
                                              abs(impl_check["cv_exact_headroom_covered_mean_ms"]
                                                  - impl_check["vectorized_covered_mean_ms"]),
                                              abs(impl_check["cv_exact_headroom_max_key_ms"] - impl_check["vectorized_max_key_ms"])))
    impl_check["passed"] = bool(impl_check["max_abs_diff_ms"] < 1e-12 and impl_check["fallback_counts_match"]
                                and impl_check["n_keys_above_host_sync_match"])

    # null truth with exactly zero headroom (per call level + shared per-shape chunk profile)
    prof = (st.W_in @ m) / st.shape_counts[:, None]                                  # (Ke, C) shape means
    lvl = m.mean(1)
    lvl_shape = np.bincount(st.exact_id, weights=lvl, minlength=st.n_exact) / st.shape_counts
    mu0 = prof[st.exact_id] + (lvl - lvl_shape[st.exact_id])[:, None]
    null_truth = st.headroom(mu0, mu0)

    # 2-5. bootstrap draws (common random numbers across estimators)
    col = {k: DrawCollector(st) for k in ("split_half", "split_half_plugin", "bootstrap_naive", "null_naive",
                                          "null_split_half")}
    for _ in range(draws):
        E = rng.standard_normal(m.shape) * se
        D = rng.standard_normal(m.shape) * se
        col["split_half"].add(st.cross_fit(m + D, m - D))
        mb = m + E
        col["bootstrap_naive"].add(st.headroom(mb, mb))
        col["split_half_plugin"].add(st.cross_fit(mb + D, mb - D))
        x0 = mu0 + E
        col["null_naive"].add(st.headroom(x0, x0))
        col["null_split_half"].add(st.cross_fit(x0 + D, x0 - D))

    obs = {"in_all": naive["in_sample"]["all"]["mean_ms"], "cv_all": naive["held_out"]["all"]["mean_ms"],
           "cv_covered": naive["held_out"]["covered"].get("mean_ms"),
           "cv_fallback": naive["held_out"]["fallback"].get("mean_ms")}
    split = col["split_half"].summary(thresholds)
    plugin = col["split_half_plugin"].summary(thresholds)
    boot = col["bootstrap_naive"].summary(thresholds)
    null = col["null_naive"].summary(thresholds, observed=obs)
    null_split = col["null_split_half"].summary(thresholds)
    del null["cv_fallback"], null["cv_all"]   # fallback null truth is not zero headroom (untuned shapes)

    bias = {}
    for k in ("in_all", "cv_all", "cv_covered", "cv_fallback"):
        if obs.get(k) is None or boot[k].get("mean_ms") is None:
            continue
        b = boot[k]["mean_ms"] - obs[k]
        bias[k] = {**_ms_us("bias", b), **_ms_us("corrected", obs[k] - b)}

    shares = {}
    for k in ("in_all", "cv_covered", "cv_all"):
        if obs.get(k) is None:
            continue
        shares[k] = {"split": _share(obs[k] - split[k]["mean_ms"], obs[k]),
                     "split_unclipped": _share(obs[k] - split[k]["mean_ms"], obs[k], clip=False),
                     "null": _share(null[k]["mean_ms"], obs[k]) if k in null else None,
                     "bootstrap": _share(bias[k]["bias_ms"], obs[k]) if k in bias else None}

    # block-level noise: Gaussian sensitivity grid, and the empirical triplicate model where available
    models = [(f"gaussian_rho_{100 * r:g}pct", gaussian_block_noise(se, r)) for r in RHO_GRID]
    block = {"note": "null and bootstrap bias correction with within-block noise plus block-level noise; a split of "
                     "the repeats cannot remove block-level noise, so no split-half estimate is given here",
             "models": {}}
    rep = refscan_replicates(df)
    if rep is not None:
        cal, pools, stratum = calibrate_refscan(df)
        block["refscan_triplicate_calibration"] = cal
        models.append(("empirical_refscan_triplicates", empirical_block_noise(pools, stratum)))
        # model-free check: the stock-chunk column replaced by the median of its three independent timings
        sc_idx = np.array([chunks.index(int(c)) for c in df["stock_chunk"]])
        m_med = m.copy()
        m_med[np.arange(len(df)), sc_idx] = np.median(rep, 1)
        block["stock_chunk_median_of_three"] = {
            "note": "naive headroom after replacing the timing of the stock-dispatch chunk by the median of its three "
                    "independent timings (less block noise on one of five chunks)",
            **summarize_protocol(st.headroom(m_med, m_med), st, thresholds)}
    null_shapes = {}
    for mname, noise in models:
        res_m = run_noise_model(st, m, mu0, noise, draws, rng, thresholds, obs)
        null_shapes[mname] = res_m.pop("_null_shapes_cv")
        block["models"][mname] = res_m

    # time-correlated slowdowns: a call timed in a degraded period is slower at every chunk than the
    # other calls of its shape. The median over chunks of latency / shape median does not depend on
    # which chunk is fastest; calls above a threshold are removed and the naive protocol is re-run.
    shape_med = pd.DataFrame(m).groupby(st.exact_id).transform("median").to_numpy()
    slowdown = np.median(m / shape_med, 1)
    degraded = {"note": "naive headroom after removing calls whose median-over-chunks latency exceeds the shape "
                        "median by more than the threshold (calls timed during a degraded period)",
                "slowdown_quantiles": {f"p{q}": float(np.percentile(slowdown, q)) for q in (50, 90, 99)}}
    if "layer" in df.columns:
        degraded["mean_slowdown_by_layer"] = {int(k): float(v) for k, v in
                                              pd.Series(slowdown).groupby(df["layer"].to_numpy()).mean().items()}
    for thr in (1.10, 1.05):
        keep = slowdown <= thr
        d2 = df[keep].reset_index(drop=True)
        st2, m2 = TableStructure(d2), m[keep]
        degraded[f"slowdown_le_{thr:.2f}"] = {"n_calls_kept": int(keep.sum()), "share_calls_kept": float(keep.mean()),
                                              **summarize_protocol(st2.headroom(m2, m2), st2, thresholds)}

    # per-shape view of the covered shapes with the largest held-out headroom
    ps_naive_cv = st.per_shape(naive_hd["cv"])
    ps_naive_in = st.per_shape(naive_hd["in"])
    ps_split_cv = col["split_half"].shape_draw_mean("cv")
    ps_null_cv = col["null_naive"].shape_draw_mean("cv")
    cov_idx = np.where(st.shape_covered)[0]
    top = cov_idx[np.argsort(-ps_naive_cv[cov_idx])][:8]
    top_shapes = []
    for k in top:
        rec = {"L": st.exact_keys[k][0], "batch": st.exact_keys[k][1], "n_calls": int(st.shape_counts[k]),
               **_ms_us("naive_in_sample", ps_naive_in[k]), **_ms_us("naive_held_out", ps_naive_cv[k]),
               **_ms_us("split_half_held_out", ps_split_cv[k]), **_ms_us("null_within_block", ps_null_cv[k])}
        for mname, arr in null_shapes.items():
            rec.update(_ms_us(f"null_{mname}", arr[k]))
        top_shapes.append(rec)

    tied = df["tied_chunks"].fillna("").astype(str).str.split(";").map(lambda t: len([c for c in t if c]))
    rel_se = se / m
    return {
        "label": label, "source_table": path.name, "n_calls": int(len(df)), "chunks": chunks,
        "repeats": sorted(int(r) for r in np.unique(reps)), "n_folds": int(st.F),
        "n_shapes": int(st.n_exact), "n_shapes_covered": int(st.shape_covered.sum()),
        "n_shapes_fallback": int((~st.shape_covered).sum()), "fallback_counts": fb,
        "thresholds": {k: _ms_us("value", v) for k, v in thresholds.items()},
        "timing_noise": {**_ms_us("median_se", np.median(se)), "median_relative_se_pct": float(100 * np.median(rel_se)),
                         "share_calls_with_tied_chunks": float((tied > 1).mean()),
                         "mean_tied_chunks_per_call": float(tied.mean())},
        "naive": naive,
        "cv_exact_headroom_imported": ref_cv,
        "implementation_check": impl_check,
        "split_half": split,
        "split_half_plugin": plugin,
        "bootstrap_naive": boot,
        "bootstrap_bias_correction": bias,
        "null_zero_headroom": {"truth_check_max_abs_in_sample_ms": float(np.abs(null_truth["in"]).max()),
                               "truth_check_max_abs_cv_covered_ms": float(np.abs(null_truth["cv"][st.covered]).max()),
                               "naive": null, "split_half": null_split},
        "noise_share": shares,
        "block_noise": block,
        "degraded_period_filter": degraded,
        "top_covered_shapes_held_out": top_shapes,
    }


# ----------------------------------------------------------------------------- main

def host_thresholds(in_dir: Path) -> dict:
    out = {}
    for tag in ("pro6000", "vgpu3090"):
        out[tag] = {"host_sync": float(json.loads((in_dir / f"e5b_{tag}.json").read_text())["host_sync_ms"])}
    dev = None
    e9 = in_dir / "e9_placement_vgpu3090.json"
    if e9.exists():
        rows = json.loads(e9.read_text()).get("decision_cost", [])
        dev = next((float(r["marginal_ms"]) for r in rows if int(r["L"]) == 2048), None)
    out["pro6000"]["device_reduction_L2048"] = None          # measured on the RTX 3090 only
    out["vgpu3090"]["device_reduction_L2048"] = dev
    return out


def reference_checks(in_dir: Path, results: dict) -> list[dict]:
    out = []
    for ds, js, jpath, rpath in REFERENCE_CHECKS:
        if ds not in results:
            continue
        exp = float(_get(json.loads((in_dir / js).read_text()), jpath))
        got = float(_get(results[ds], rpath))
        out.append({"dataset": ds, "reference": f"{js}:{'.'.join(jpath)}", "expected_ms": exp, "recomputed_ms": got,
                    "abs_diff_us": 1000 * abs(exp - got), "match": bool(abs(exp - got) < 1e-9)})
    return out


def _fmt(x, nd=2):
    return "  -  " if x is None else f"{x:.{nd}f}"


def print_summary(res: dict) -> None:
    hdr = (f"{'dataset':22s} {'naive in':>8s} {'max':>6s} | {'held-out all/cov/fb (us)':>24s} | "
           f"{'split in [95%]':>22s} | {'split cov [95%]':>22s} | {'null in / cov':>13s} | "
           f"{'share':>5s} | {'>sync in/cov/fb; split cov/fb':>30s}")
    print(hdr)
    print("-" * len(hdr))
    for name, r in res["datasets"].items():
        n, sp, nl = r["naive"], r["split_half"], r["null_zero_headroom"]["naive"]
        ni, ho = n["in_sample"]["all"], n["held_out"]
        cov, fbk = ho["covered"], ho["fallback"]
        spi, spc, spf = sp["in_all"], sp["cv_covered"], sp["cv_fallback"]
        share = r["noise_share"].get("cv_covered", r["noise_share"]["in_all"])["split"]
        print(f"{name:22s} {ni['mean_us']:8.2f} {ni['max_shape_us']:6.1f} | "
              f"{ho['all']['mean_us']:6.2f}/{_fmt(cov.get('mean_us')):>6s}/{_fmt(fbk.get('mean_us')):>7s}   | "
              f"{spi['mean_us']:6.2f} [{spi['ci95_lo_us']:5.2f},{spi['ci95_hi_us']:5.2f}] | "
              f"{spc['mean_us']:6.2f} [{spc['ci95_lo_us']:5.2f},{spc['ci95_hi_us']:5.2f}] | "
              f"{nl['in_all']['mean_us']:5.2f} / {nl['cv_covered']['mean_us']:5.2f} | {_fmt(share):>5s} | "
              f"{ni['n_shapes_above_host_sync']}/{cov.get('n_shapes_above_host_sync', 0)}/"
              f"{fbk.get('n_shapes_above_host_sync', 0)}; "
              f"{spc.get('n_shapes_above_host_sync_draw_mean', 0)}/{spf.get('n_shapes_above_host_sync_draw_mean', 0)}")
    print("\nnaive in: in-sample exact-shape headroom (us/call), max: largest per-shape headroom (us);"
          " held-out: table chosen on the other prompt folds, all / covered shapes / fallback shapes;"
          " split: cross-fitted split-half (choose on 25 repeats, score on the other 25), mean and"
          " 95% interval over random splits; null: naive headroom under zero true headroom;"
          " share: 1 - split/naive on held-out covered calls (upper bound on the selection-noise share);"
          " >sync: shapes whose headroom exceeds one host synchronization.")

    names = None
    print("\nBlock-level noise: null (zero true headroom) naive headroom, us/call, in-sample / held-out covered")
    for name, r in res["datasets"].items():
        models = r["block_noise"]["models"]
        if names is None:
            names = list(models)
            print(f"{'dataset':22s} {'observed':>13s} | {'within-block':>13s} | "
                  + " | ".join(f"{n.replace('gaussian_', '').replace('empirical_refscan_', 'emp_'):>13s}"
                               for n in names + (["empirical_refscan_triplicates"] if "empirical_refscan_triplicates"
                                                 not in names else [])))
        ni = r["naive"]["in_sample"]["all"]["mean_us"]
        nc = r["naive"]["held_out"]["covered"].get("mean_us")
        nl = r["null_zero_headroom"]["naive"]
        cells = [f"{ni:6.2f}/{_fmt(nc):>6s}", f"{nl['in_all']['mean_us']:6.2f}/{nl['cv_covered']['mean_us']:6.2f}"]
        for mname in names + ["empirical_refscan_triplicates"]:
            if mname in models:
                z = models[mname]["null_zero_headroom"]
                cells.append(f"{z['in_all']['mean_us']:6.2f}/{z['cv_covered']['mean_us']:6.2f}")
            elif mname == "empirical_refscan_triplicates":
                cells.append(f"{'-':>13s}")
        print(f"{name:22s} " + " | ".join(f"{c:>13s}" for c in cells))

    print("\nDegraded-period filter: held-out covered headroom, us/call (largest shape; shapes above host sync)")
    print(f"{'dataset':22s} {'all calls':>22s} | {'slowdown <= 1.10':>28s} | {'slowdown <= 1.05':>28s}")
    for name, r in res["datasets"].items():
        c0 = r["naive"]["held_out"]["covered"]
        cells = [f"{c0['mean_us']:6.2f} ({c0['max_shape_us']:6.1f}; {c0['n_shapes_above_host_sync']})"]
        for k in ("slowdown_le_1.10", "slowdown_le_1.05"):
            f = r["degraded_period_filter"][k]
            c = f["held_out"]["covered"]
            cells.append(f"{f['share_calls_kept']:.2f} kept: {_fmt(c.get('mean_us'))} ({_fmt(c.get('max_shape_us'), 1)}; "
                         f"{c.get('n_shapes_above_host_sync', 0)})")
        print(f"{name:22s} {cells[0]:>22s} | {cells[1]:>28s} | {cells[2]:>28s}")
    bad = [c for c in res["reference_checks"] if not c["match"]]
    print(f"\nreference checks: {len(res['reference_checks']) - len(bad)}/{len(res['reference_checks'])} match"
          + ("" if not bad else "; MISMATCH: " + "; ".join(f"{c['dataset']} {c['reference']} expected "
                                                          f"{c['expected_ms']:.6g} got {c['recomputed_ms']:.6g}"
                                                          for c in bad)))
    impl = {k: v["implementation_check"]["passed"] for k, v in res["datasets"].items()}
    print("vectorized held-out protocol equals cv_exact_headroom: " + ", ".join(f"{k}={v}" for k, v in impl.items()))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--out", type=Path, default=None, help="output JSON (default: <in-dir>/headroom_noise.json)")
    p.add_argument("--draws", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--datasets", default=",".join(d[0] for d in DATASETS))
    args = p.parse_args(argv)
    thr = host_thresholds(args.in_dir)
    want = set(args.datasets.split(","))
    t0 = time.time()
    res = {"experiment": "headroom-noise (review M3b, M3c)", "draws": args.draws, "seed": args.seed,
           "units": "latencies in ms (keys *_ms) and microseconds (keys *_us) per call",
           "method": {
               "naive": "table and per-call oracle chosen and scored on the same 50-repeat timing means; "
                        "in_sample: exact-shape table chosen on all calls; held_out: exact-shape table chosen on the "
                        "other prompt folds with bucket / global fallback (cv_exact_headroom).",
               "split_half": "cross-fitted split-half: halves A = m + D, B = m - D, D ~ N(0, s^2/n) per call and chunk "
                             "(exact conditional law of two 25-repeat half-means given the 50-repeat mean under "
                             "Gaussian repeats); table and oracle chosen on one half, scored on the other, both "
                             "directions averaged; conservative (lower bound in expectation).",
               "split_half_plugin": "A, B ~ N(m, 2 s^2/n) independently around the observed means; treats the observed "
                                    "means as truth and keeps part of the selection bias.",
               "bootstrap_bias_correction": "m* = m + N(0, s^2/n); bias = E[naive(m*)] - naive(m); corrected = naive - bias.",
               "null_zero_headroom": "truth mu0 = per-shape chunk profile + per-call level (true headroom exactly 0); "
                                     "X0 = mu0 + N(0, s^2/n); naive estimator applied; p = share of draws >= observed.",
               "noise_share": "split = 1 - split_half/naive (clipped, upper bound); null = null mean / naive (clipped); "
                              "bootstrap = bias / naive.",
               "intervals": "2.5 and 97.5 percentiles over the draws",
           },
           "thresholds_by_gpu": {g: {k: _ms_us("value", v) for k, v in t.items()} for g, t in thr.items()},
           "datasets": {}}
    for idx, (name, label, csv, gpu) in enumerate(DATASETS):
        if name not in want:
            continue
        rng = np.random.default_rng([args.seed, idx])
        res["datasets"][name] = analyze_dataset(label, args.in_dir / csv, thr[gpu], args.draws, rng)
        res["datasets"][name]["gpu"] = gpu
    res["block_noise_calibration_rounds"] = calibrate_rounds(args.in_dir)
    res["reference_checks"] = reference_checks(args.in_dir, res["datasets"])
    res["runtime_s"] = round(time.time() - t0, 1)
    out = args.out or (args.in_dir / "headroom_noise.json")
    write_json(out, res)
    print_summary(res)
    print(f"\nwrote {out} ({res['runtime_s']} s)")


if __name__ == "__main__":
    main()
