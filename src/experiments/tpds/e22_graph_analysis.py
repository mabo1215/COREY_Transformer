"""E22 analysis (review item M3b): CUDA-graph timing of the production kernels and the Mamba-1 scan.

CPU only, deterministic, reads only stored result files.

Input: the E2 tables of the CUDA-graph sweep on the RTX 3090 (``e2_oracle_sweep.py --cuda-graph``: every
(call, chunk) is captured once into a CUDA graph and its replays are timed; 10 warmup and 50 timed replays,
one round, batch 1, 4 and 16), with a warm L2 and with the L2 flushed before every replay:
  e2_mamba2_vgpu3090_r4graph[cold].csv         Mamba-2 370M SSD calls
  e2_mamba2_vgpu3090_2p7b_r4graph[cold].csv    Mamba-2 2.7B SSD calls (flushed: calls with L >= 2048 only)
  e2_gdn_vgpu3090_1p4b_r4graph[cold].csv       Gated DeltaNet 1.4B hybrid calls (chunks 16, 32, 64)
  e2_mamba1_vgpu3090_r4graph.csv               Mamba-1 370M Triton scan calls (warm L2 only)
and the earlier timings of the same calls (same keys): launch-inclusive eager timings
(``e2_mamba2_vgpu3090.csv``, the Mamba-2 rows of the paper on this GPU; ``e2_mamba2_pro6000.csv``, the Mamba-2
rows of Tables II and III; ``e2_mamba1_vgpu3090.csv``) and device time with a GPU-side pre-spin
(``e2_mamba2_vgpu3090_r3spin.csv``, discarded in the paper; ``e2_mamba2_vgpu3090_2p7b_r3.csv`` and
``e2_gdn_vgpu3090_1p4b_r3.csv``, Table S-largewidth-calls; ``e2_mamba1_vgpu3090_r3spin.csv`` and the
three-round ``e2_mamba1_vgpu3090_r4spin.csv``, read through ``analysis.round_median_view``).

Per table, for all calls and for the calls above / below the launch floor (twice the median oracle latency
of the batch-1 calls with L <= 256, the definition of Table S-largewidth-calls; the flushed 2.7B table has no
short calls and takes the floor of the warm one):
  * floor, share of calls above it, within-block coefficient of variation (std / mean of each
    (call, chunk) timing block: median over cells, as in Table S-conventions, and median over calls of the
    per-call median, which reproduces Table S-largewidth-calls), chunk spread, oracle distribution;
  * Table II quantities per batch (best static chunk, oracle != best static, static / rule / default regret);
  * Table III quantities: the E3 predictors (grouped 5-fold cross-validation over the prompt folds, trees of
    depth 4, length only 3; ``e3_predictors``), scored on all and on above-floor calls; the library default
    (SSD 256, Gated DeltaNet 64; Mamba-1 has none, its best static chunk is used); the best single chunk; the
    exact-shape table chosen on held-out folds (``deployment_units.cv_exact_pick``); and, for the multi-kernel
    pipelines, the work-term model with the parameters stored for the GPU (``cost_model_<gpu>.json``, fitted
    on eager calibration timings) under the selection rules of ``headroom_r4`` (``original``: exact ties at
    the floor go to the smallest chunk; ``launch_default``: to the work-bound optimum, 256 / 64, review M3c;
    ``work_argmin``), with its floor F replaced by the measured floor of the table, and refitted on the graph
    timings of the training folds (graph tables only);
  * the held-out exact-shape headroom (``analysis.cv_exact_headroom``: covered mean, per-shape maximum,
    shapes above one host synchronization of 21.4 us) on all calls and on the above-floor calls;
  * deployment units (regret in us per call and ms per forward pass: 48, 64 and 12 scan layers);
  * a Table I analog on real activations: per exact batch-1 shape L in {512, 2048, 8192}, the input-agnostic
    best chunk (lowest per-chunk median over the calls of that shape), the spread of the per-chunk medians,
    the share of calls whose statistical tie set contains it and its maximum regret;
  * per (batch, length bucket) and per token bin: per-chunk median latencies, oracle counts, spread, default
    regret and share above the floor.
Comparisons (joined by call key): latency ratio by length bucket and batch (oracle latency, default chunk,
all cells), oracle changes (exact and tie-aware), the regret under one convention of the per-call oracle and
of the exact-shape table chosen under the other, and the rank agreement of the per-call chunk profiles.
Claims: the production-kernel statements of the paper re-evaluated under CUDA-graph timing. Launch-bound
boundary: the work-term formula F / (a + 2 sqrt(b g)) at the measured graph floor (eager a, b, g) and with
parameters refitted on all graph timings, next to the eager values (9,513 / 17,321 tokens; 3,878 at 320 us),
and the empirical token count from which most calls lie above the floor.
Reproduction: applied to the eager tables, the same code must give the printed Mamba-2 / Gated DeltaNet
numbers (Tables II, III, S-conventions, S-headroom-noise, S-largewidth-calls, deployment, M3c) and the stored
``e3_*.json`` / ``headroom_r4.json`` values; every check is part of the output.

Output: ``e22_graph_analysis_vgpu3090.json`` (or ``--out``).

Usage:  PYTHONPATH=. python3 -m src.experiments.tpds.e22_graph_analysis [--in-dir DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.experiments.tpds import cost_model as CM
from src.experiments.tpds import e3_predictors as E3
from src.experiments.tpds.analysis import (chunk_columns, cv_exact_headroom, latency_matrix, load_e2,
                                           round_median_view, static_regret_table)
from src.experiments.tpds.common import RESULTS_DIR, env_metadata, write_json
from src.experiments.tpds.deployment_units import cv_exact_pick, insample_exact_pick
from src.experiments.tpds.headroom_r4 import select_pipeline

SYNC_MS = 0.0214            # one host synchronization on the RTX 3090 (Table S-headroom-noise: 21.4 us)
TABLE1_LENGTHS = (512, 2048, 8192)
WORK_RULES = ("original", "launch_default", "work_argmin")

# name: csv, family, checkpoint, timing convention, gpu, scan layers, library default chunk (None: no
# default, best static chunk instead), floor taken from another table, round-median view, refit work terms
TABLES = {
    "ssd370m_graph": ("e2_mamba2_vgpu3090_r4graph.csv", "mamba2", "Mamba-2 370M", "CUDA graph, warm L2",
                      "vgpu3090", 48, 256, None, False, True),
    "ssd370m_graphcold": ("e2_mamba2_vgpu3090_r4graphcold.csv", "mamba2", "Mamba-2 370M", "CUDA graph, L2 flushed",
                          "vgpu3090", 48, 256, None, False, True),
    "ssd370m_eager": ("e2_mamba2_vgpu3090.csv", "mamba2", "Mamba-2 370M", "launch-inclusive (eager)",
                      "vgpu3090", 48, 256, None, False, False),
    "ssd370m_device": ("e2_mamba2_vgpu3090_r3spin.csv", "mamba2", "Mamba-2 370M", "device time (pre-spin)",
                       "vgpu3090", 48, 256, None, False, False),
    "ssd370m_pro6000_eager": ("e2_mamba2_pro6000.csv", "mamba2", "Mamba-2 370M",
                              "launch-inclusive (eager), RTX PRO 6000", "pro6000", 48, 256, None, False, False),
    "ssd2p7b_graph": ("e2_mamba2_vgpu3090_2p7b_r4graph.csv", "mamba2", "Mamba-2 2.7B", "CUDA graph, warm L2",
                      "vgpu3090", 64, 256, None, False, True),
    "ssd2p7b_graphcold": ("e2_mamba2_vgpu3090_2p7b_r4graphcold.csv", "mamba2", "Mamba-2 2.7B",
                          "CUDA graph, L2 flushed (calls with L >= 2048)", "vgpu3090", 64, 256, "ssd2p7b_graph",
                          False, True),
    "ssd2p7b_device": ("e2_mamba2_vgpu3090_2p7b_r3.csv", "mamba2", "Mamba-2 2.7B", "device time (pre-spin)",
                       "vgpu3090", 64, 256, None, False, False),
    "gdn1p4b_graph": ("e2_gdn_vgpu3090_1p4b_r4graph.csv", "gdn", "Gated DeltaNet 1.4B hybrid", "CUDA graph, warm L2",
                      "vgpu3090", 12, 64, None, False, True),
    "gdn1p4b_graphcold": ("e2_gdn_vgpu3090_1p4b_r4graphcold.csv", "gdn", "Gated DeltaNet 1.4B hybrid",
                          "CUDA graph, L2 flushed", "vgpu3090", 12, 64, None, False, True),
    "gdn1p4b_device": ("e2_gdn_vgpu3090_1p4b_r3.csv", "gdn", "Gated DeltaNet 1.4B hybrid", "device time (pre-spin)",
                       "vgpu3090", 12, 64, None, False, False),
    "m1_370m_graph": ("e2_mamba1_vgpu3090_r4graph.csv", "mamba1", "Mamba-1 370M (Triton scan)", "CUDA graph, warm L2",
                      "vgpu3090", 48, None, None, False, False),
    "m1_370m_device": ("e2_mamba1_vgpu3090_r3spin.csv", "mamba1", "Mamba-1 370M (Triton scan)",
                       "device time (pre-spin)", "vgpu3090", 48, None, None, False, False),
    "m1_370m_device_r4": ("e2_mamba1_vgpu3090_r4spin.csv", "mamba1", "Mamba-1 370M (Triton scan)",
                          "device time, three rounds (round medians)", "vgpu3090", 48, None, None, True, False),
    "m1_370m_eager": ("e2_mamba1_vgpu3090.csv", "mamba1", "Mamba-1 370M (Triton scan)", "launch-inclusive (eager)",
                      "vgpu3090", 48, None, None, False, False),
}

# --rounds3 (re-check P3, E25): the warm-L2 graph tables timed in three rounds replace the single-round ones
# (read through the round-median view); the single-round tables stay as "<name>1" for comparison
GRAPH3 = {"ssd370m_graph": "e2_mamba2_vgpu3090_r4graph3.csv", "ssd2p7b_graph": "e2_mamba2_vgpu3090_2p7b_r4graph3.csv",
          "gdn1p4b_graph": "e2_gdn_vgpu3090_1p4b_r4graph3.csv", "m1_370m_graph": "e2_mamba1_vgpu3090_r4graph3.csv"}


def use_three_rounds(in_dir: Path) -> list[str]:
    """Switch the warm graph tables to the three-round sweeps that exist in ``in_dir``; returns their names."""
    switched = []
    for nm, csv in GRAPH3.items():
        if not (in_dir / csv).exists():
            continue
        spec = list(TABLES[nm])
        TABLES[nm + "1"] = tuple(spec[:3] + [spec[3] + ", one round"] + spec[4:])
        spec[0], spec[3], spec[8] = csv, spec[3] + ", three rounds (round medians)", True
        TABLES[nm] = tuple(spec)
        COMPARISONS.append((nm, nm + "1"))
        switched.append(nm)
    return switched


# (new, reference): latency ratio new / reference, oracle changes, cross regret
COMPARISONS = [
    ("ssd370m_graph", "ssd370m_eager"), ("ssd370m_graph", "ssd370m_device"), ("ssd370m_graphcold", "ssd370m_graph"),
    ("ssd370m_graphcold", "ssd370m_eager"),
    ("ssd2p7b_graph", "ssd2p7b_device"), ("ssd2p7b_graphcold", "ssd2p7b_graph"), ("ssd2p7b_graphcold", "ssd2p7b_device"),
    ("gdn1p4b_graph", "gdn1p4b_device"), ("gdn1p4b_graphcold", "gdn1p4b_graph"), ("gdn1p4b_graphcold", "gdn1p4b_device"),
    ("m1_370m_graph", "m1_370m_device"), ("m1_370m_graph", "m1_370m_device_r4"), ("m1_370m_device_r4", "m1_370m_device"),
    ("m1_370m_graph", "m1_370m_eager"),
]


# ----------------------------------------------------------------------------- helpers
def _clean(obj):
    """JSON-safe copy: numpy scalars to Python, non-finite floats to None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    return obj


def _pct(x):
    return float(100.0 * x)


def _tied_sets(df: pd.DataFrame) -> list[set]:
    return [set(int(c) for c in str(t).split(";") if c) for t in df["tied_chunks"].fillna("")]


def _matches(value, printed: str) -> bool:
    """Printed value reproduced at its printed precision ('<0.01' style bounds allowed)."""
    if value is None:
        return False
    if printed.startswith("<"):
        return float(value) < float(printed[1:])
    nd = len(printed.split(".")[1]) if "." in printed else 0
    return bool(abs(round(float(value) + 1e-12, nd) - float(printed)) < 1e-9)


class Table:
    """One E2 table with its latency matrix (+inf where a chunk cannot run), oracle, tie sets and floor."""

    def __init__(self, name: str, in_dir: Path, keys=None):
        (self.csv, self.family, self.checkpoint, self.timing, self.gpu, self.layers, self.default,
         self.floor_from, self.round_median, self.refit) = TABLES[name]
        self.name = name
        df = load_e2([in_dir / self.csv])
        if keys is not None:            # only these calls (e.g. the calls of a table restricted to L >= 2048)
            df = df[df["key"].isin(set(keys))]
        if self.round_median:
            df = round_median_view(df)
        self.df = df.reset_index(drop=True)
        self.chunks = chunk_columns(self.df)
        lat = latency_matrix(self.df, self.chunks)
        self.avail = np.isfinite(lat)
        self.lat = np.where(self.avail, lat, np.inf)
        self.orc = self.lat.min(1)
        self.oidx = self.lat.argmin(1)
        self.tied = _tied_sets(self.df)
        std = self.df[[f"lat_c{c}_std" for c in self.chunks]].to_numpy(float)
        cv = std / np.where(self.avail, lat, np.nan)
        self.cv = np.where(np.isfinite(cv), cv, np.nan)
        self.batch = self.df["batch"].to_numpy()
        self.L = self.df["L"].to_numpy()
        self.floor_ms = None
        self.above = None

    def set_floor(self, floor_ms: float | None = None) -> None:
        if floor_ms is None:
            small = (self.batch == 1) & (self.L <= 256)
            floor_ms = 2.0 * float(np.median(self.orc[small]))
        self.floor_ms = float(floor_ms)
        self.above = self.orc > self.floor_ms

    def default_idx(self) -> int:
        """Library default (SSD 256, GDN 64); Mamba-1 has none: the best static chunk in sample."""
        if self.default is not None:
            return self.chunks.index(int(self.default))
        return self.best_static_idx(np.ones(len(self.df), bool))

    def best_static_idx(self, mask: np.ndarray) -> int:
        norm = self.lat[mask] / self.orc[mask, None]
        return int(np.argmin(norm.mean(0)))

    def masks(self) -> dict:
        return {"all": np.ones(len(self.df), bool), "above_floor": self.above, "below_floor": ~self.above}


# ----------------------------------------------------------------------------- policies
def e3_predictions(df: pd.DataFrame, chunks: list[int]) -> dict[str, np.ndarray]:
    """Per-call chunk index of every E3 predictor (the loop of ``e3_predictors.evaluate``, kept per call)."""
    lat = latency_matrix(df, chunks)
    oracle_idx = lat.argmin(1)
    norm = lat / lat.min(1, keepdims=True)
    folds = df["fold"].to_numpy()
    preds = {name: np.full(len(df), -1) for name in ["global_static", "entropy_rule", *E3.FEATURE_SETS]}
    rule = df["entropy_rule_chunk"].clip(min(chunks), max(chunks)).to_numpy()
    preds["entropy_rule"] = np.array([chunks.index(int(c)) if int(c) in chunks else
                                      int(np.argmin([abs(np.log2(c) - np.log2(x)) for x in chunks])) for c in rule])
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if tr.sum() == 0 or te.sum() == 0:
            continue
        preds["global_static"][te] = int(np.argmin(norm[tr].sum(0)))
        for name, (cols, depth) in E3.FEATURE_SETS.items():
            m = E3.LeafCostTree(depth).fit(df.loc[tr, cols].to_numpy(float), oracle_idx[tr], norm[tr])
            preds[name][te] = m.predict_idx(df.loc[te, cols].to_numpy(float))
    return preds


def pipe_params(in_dir: Path, gpu: str, family: str) -> list[float]:
    pm = json.loads((in_dir / f"cost_model_{gpu}.json").read_text())["V4_pipelines"][family]
    return [float(pm["params"][k]) for k in CM.PIPE_PARAMS]


def pipe_points(t: Table, rows: np.ndarray) -> pd.DataFrame:
    pts = []
    for j, c in enumerate(t.chunks):
        sel = rows & t.avail[:, j]
        pts.append(pd.DataFrame({"L": t.L[sel], "B": t.batch[sel], "C": c, "lat": t.lat[sel, j]}))
    return pd.concat(pts, ignore_index=True)


def boundary(th) -> dict:
    F, a, b, g, p = th
    den = a + 2.0 * math.sqrt(b * g)
    return {"F_us": 1000.0 * F, "C_star": math.sqrt(g / b) if b > 0 else None, "BL_star_tokens": F / den,
            "L_star_by_batch": {str(bb): F / den / bb for bb in (1, 4, 16)}}


def work_term_policies(t: Table, in_dir: Path) -> tuple[dict, dict]:
    """Chunk index per call of the work-term model variants, and their parameters."""
    th = pipe_params(in_dir, t.gpu, t.family)
    Lf, Bf = t.L.astype(float), t.batch.astype(float)
    picks, info = {}, {"stored_params": dict(zip(CM.PIPE_PARAMS, th)), "stored_boundary": boundary(th)}
    for rule in WORK_RULES:
        picks[f"work_term_{rule}"] = select_pipeline(th, Lf, Bf, t.chunks, t.avail, rule)
    small = (t.batch == 1) & (t.L <= 256)
    F_meas = float(np.median(t.orc[small])) if small.any() else None
    if F_meas is not None:
        th_low = [F_meas] + list(th[1:])
        picks["work_term_measured_floor"] = select_pipeline(th_low, Lf, Bf, t.chunks, t.avail, "launch_default")
        info["measured_floor"] = {"F_us": 1000.0 * F_meas, "note": "F = median oracle latency of the batch-1 calls "
                                  "with L <= 256 of this table; a, b, g, p stored (eager calibration); ties to the "
                                  "work-bound optimum", "boundary": boundary(th_low)}
    if t.refit:
        folds = t.df["fold"].to_numpy()
        pick = np.full(len(t.df), -1)
        fold_params = {}
        for f in np.unique(folds):
            th_f, r2 = CM.pipe_fit(pipe_points(t, folds != f))
            te = folds == f
            pick[te] = select_pipeline(th_f, Lf[te], Bf[te], t.chunks, t.avail[te], "launch_default")
            fold_params[str(int(f))] = {**dict(zip(CM.PIPE_PARAMS, map(float, th_f))), "r2_log_train": r2}
        picks["work_term_refit_cv"] = pick
        th_all, r2_all = CM.pipe_fit(pipe_points(t, np.ones(len(t.df), bool)))
        info["refit"] = {"note": "work-term model refitted on the timings of this table (all chunks of the "
                                 "training-fold calls; selection on the held-out fold, ties to the work-bound "
                                 "optimum); in-sample fit on all calls for the parameters and the boundary",
                         "fold_params": fold_params, "all_calls_params": dict(zip(CM.PIPE_PARAMS, map(float, th_all))),
                         "all_calls_r2_log": r2_all, "boundary_all_calls_fit": boundary(th_all)}
    return picks, info


def policy_stats(t: Table, pick: np.ndarray, mask: np.ndarray) -> dict:
    idx = np.where(mask & (pick >= 0))[0]
    if len(idx) == 0:
        return {"n": 0}
    p = pick[idx]
    picked, o = t.lat[idx, p], t.orc[idx]
    reg = picked / o - 1.0
    d_us = 1000.0 * (picked - o)
    chosen = np.array(t.chunks)[p]
    return {"n": int(len(idx)), "mean_pct": _pct(reg.mean()), "p95_pct": _pct(np.percentile(reg, 95)),
            "max_pct": _pct(reg.max()), "tied_share": float(np.mean([c in t.tied[i] for c, i in zip(chosen, idx)])),
            "within_2pct_share": float((reg <= 0.02).mean()), "oracle_hit_share": float((p == t.oidx[idx]).mean()),
            "us_per_call": float(d_us.mean()), "ms_per_forward_pass": float(d_us.mean() * t.layers / 1000.0),
            "latency_weighted_share_pct": _pct((picked - o).sum() / picked.sum()), "oracle_us_mean": float(1000 * o.mean()),
            "chosen": {str(int(k)): int(v) for k, v in zip(*np.unique(chosen, return_counts=True))}}


# ----------------------------------------------------------------------------- per-table summary
def _spread(t: Table) -> np.ndarray:
    fin = np.where(t.avail, t.lat, np.nan)
    return np.nanmax(fin, 1) / np.nanmin(fin, 1)


def _dist(v) -> dict:
    s = pd.Series(v).value_counts()
    return {str(int(k)): int(n) for k, n in sorted(s.items())}


def table2(t: Table, mask: np.ndarray) -> dict:
    out = {}
    for b in sorted(np.unique(t.batch)):
        sel = mask & (t.batch == b)
        if not sel.any():
            continue
        sub = t.df[sel].reset_index(drop=True)
        tab = static_regret_table(sub, t.chunks)
        bs = int(tab["best_static"].split("_")[1])
        rec = {"calls": int(sel.sum()), "best_static_chunk": bs,
               "oracle_ne_best_static_pct": _pct((sub["oracle_chunk"] != bs).mean()),
               "static_regret_pct": [tab[tab["best_static"]]["mean_regret_pct"], tab[tab["best_static"]]["p95_regret_pct"]],
               "rule_regret_pct": [tab["entropy_rule"]["mean_regret_pct"], tab["entropy_rule"]["p95_regret_pct"]],
               "oracle_distribution": _dist(sub["oracle_chunk"])}
        if t.default is not None:
            d = tab[f"static_{t.default}"]
            rec["default_regret_pct"] = [d["mean_regret_pct"], d["p95_regret_pct"], d["max_regret_pct"]]
        out[str(int(b))] = rec
    return out


def table1_analog(t: Table) -> dict:
    out = {}
    for L in TABLE1_LENGTHS:
        sel = (t.L == L) & (t.batch == 1)
        if sel.sum() < 2:
            continue
        idx = np.where(sel)[0]
        med = np.array([np.median(t.lat[idx, j]) for j in range(len(t.chunks))])
        fin = np.isfinite(med)
        js = int(np.argmin(np.where(fin, med, np.inf)))
        reg = t.lat[idx, js] / t.orc[idx] - 1.0
        out[str(L)] = {"calls": int(len(idx)), "prompt_groups": int(t.df.loc[sel, "group_id"].nunique()),
                       "c_star": int(t.chunks[js]), "spread_of_chunk_medians": float(med[fin].max() / med[fin].min()),
                       "tied_share": float(np.mean([t.chunks[js] in t.tied[i] for i in idx])),
                       "max_regret_pct": _pct(reg.max()), "mean_regret_pct": _pct(reg.mean()),
                       "median_us_per_chunk": {str(c): 1000.0 * m for c, m in zip(t.chunks, med)},
                       "oracle_distribution": _dist(np.array(t.chunks)[t.oidx[idx]])}
    return out


def curves(t: Table) -> dict:
    """Per (batch, length bucket) and per token bin: per-chunk median latencies, oracle counts, spread."""
    spread = _spread(t)
    dj = t.default_idx()
    reg_def = t.lat[:, dj] / t.orc - 1.0
    by_cell = []
    for (b, lb), g in t.df.groupby(["batch", "L_bucket"]):
        i = g.index.to_numpy()
        by_cell.append({"batch": int(b), "L_bucket": int(lb), "calls": int(len(i)),
                        "median_us": {str(c): float(1000 * np.median(t.lat[i, j])) for j, c in enumerate(t.chunks)},
                        "oracle_distribution": _dist(np.array(t.chunks)[t.oidx[i]]),
                        "spread_median": float(np.median(spread[i])),
                        "default_regret_pct_median": _pct(np.median(reg_def[i])),
                        "default_regret_pct_mean": _pct(reg_def[i].mean()),
                        "default_regret_us_mean": float(1000 * (t.lat[i, dj] - t.orc[i]).mean()),
                        "share_above_floor": float(t.above[i].mean())})
    tokens = t.L * t.batch
    tbin = 2 ** np.floor(np.log2(tokens)).astype(int)
    by_tokens = []
    for tb in np.unique(tbin):
        i = np.where(tbin == tb)[0]
        by_tokens.append({"tokens_bin": int(tb), "calls": int(len(i)), "share_above_floor": float(t.above[i].mean()),
                          "oracle_us_median": float(1000 * np.median(t.orc[i])),
                          "spread_median": float(np.median(spread[i])),
                          "default_regret_pct_mean": _pct(reg_def[i].mean()),
                          "oracle_distribution": _dist(np.array(t.chunks)[t.oidx[i]])})
    # empirical launch-bound boundary: first token bin from which at least half the calls of every larger bin
    # lie above the floor
    shares = [r["share_above_floor"] for r in by_tokens]
    first = next((r["tokens_bin"] for k, r in enumerate(by_tokens) if all(s >= 0.5 for s in shares[k:])), None)
    return {"by_batch_L_bucket": by_cell, "by_tokens": by_tokens,
            "empirical_boundary_tokens_bin": first,
            "empirical_boundary_note": "smallest power-of-two bin of B*L from which >= 50% of the calls of that and "
                                       "every larger bin have an oracle latency above the floor (2 x median "
                                       "batch-1 L <= 256); the work-term BL* is where the work equals F, about "
                                       "half the token count at which latency reaches 2 F"}


def summarize(t: Table, in_dir: Path) -> dict:
    """Per-table summary (floor, CV, spread, oracle, Table II / III quantities, headroom, curves)."""
    n = len(t.df)
    spread = _spread(t)
    m = t.masks()
    small = (t.batch == 1) & (t.L <= 256)
    res = {"source": t.csv, "family": t.family, "checkpoint": t.checkpoint, "timing": t.timing, "gpu": t.gpu,
           "layers": t.layers, "n_calls": n, "chunks": t.chunks,
           "calls_with_unavailable_chunks": int((~t.avail).any(1).sum()),
           "library_default": t.default, "default_used": int(t.chunks[t.default_idx()]),
           "floor": {"floor_us": 1000 * t.floor_ms, "floor_from": t.floor_from or t.name,
                     "median_b1_L_le_256_us": 1000 * float(np.median(t.orc[small])) if small.any() else None,
                     "min_oracle_us": 1000 * float(t.orc.min()),
                     "median_oracle_us_smallest_bucket_by_batch": {
                         str(int(b)): 1000 * float(np.median(t.orc[(t.batch == b) & (t.df["L_bucket"] == t.df["L_bucket"].min()).to_numpy()]))
                         for b in np.unique(t.batch)},
                     "n_above": int(t.above.sum()), "share_above": float(t.above.mean())}}
    res["within_block_cv_pct"] = {
        k: {"cell_median": _pct(np.nanmedian(t.cv[v])) if v.any() else None,
            "per_call_median": _pct(np.nanmedian(np.nanmedian(t.cv[v], 1))) if v.any() else None}
        for k, v in m.items()}
    res["spread"] = {k: {"median": float(np.median(spread[v])), "p95": float(np.percentile(spread[v], 95))}
                     for k, v in m.items() if v.any()}
    res["oracle_distribution"] = {k: _dist(np.array(t.chunks)[t.oidx[v]]) for k, v in m.items() if v.any()}
    res["tied_with_oracle_mean_count"] = float(np.mean([len(s) for s in t.tied]))
    if t.above.any():
        top = int(pd.Series(np.array(t.chunks)[t.oidx[t.above]]).value_counts().index[0])
        jt = t.chunks.index(top)
        reg = t.lat[:, jt] / t.orc - 1.0
        res["above_floor_static"] = {
            "chunk_most_often_oracle": top, "oracle_count": int((t.oidx[t.above] == jt).sum()),
            "n_above": int(t.above.sum()),
            "regret_pct_by_batch": {str(int(b)): {"n": int((t.above & (t.batch == b)).sum()),
                                                  "mean": _pct(reg[t.above & (t.batch == b)].mean()),
                                                  "max": _pct(reg[t.above & (t.batch == b)].max())}
                                    for b in np.unique(t.batch) if (t.above & (t.batch == b)).any()}}
    res["table2"] = {k: table2(t, v) for k, v in (("all", m["all"]), ("above_floor", m["above_floor"]))}

    # policies
    preds = e3_predictions(t.df, t.chunks)
    picks = {"default": np.full(n, t.default_idx()), "best_static_insample": np.full(n, t.best_static_idx(m["all"])),
             **{f"e3_{k}": v for k, v in preds.items()}, "exact_table_heldout": cv_exact_pick(t.df, t.lat),
             "exact_table_insample": insample_exact_pick(t.df, t.lat)}
    if t.family in ("mamba2", "gdn"):
        wp, winfo = work_term_policies(t, in_dir)
        picks.update(wp)
        res["work_term_model"] = winfo
    res["policies"] = {name: {k: policy_stats(t, p, v) for k, v in m.items()} for name, p in picks.items()}
    # consistency with e3_predictors.evaluate
    ev = E3.evaluate(t.df, t.chunks)
    res["check_e3_evaluate_max_abs_diff_pp"] = float(max(
        abs(ev[k]["mean_regret_pct"] - res["policies"][f"e3_{k}"]["all"]["mean_pct"]) for k in ev))

    # exact-shape held-out headroom (all calls; calls above the floor)
    res["headroom"] = {}
    for k in ("all", "above_floor"):
        sub = t.df[m[k]].reset_index(drop=True)
        if len(sub) < 10 or sub["fold"].nunique() < 2:
            continue
        h = cv_exact_headroom(sub, chunk_columns(sub), threshold_ms=SYNC_MS)
        cov = h.get("covered", {})
        res["headroom"][k] = {"held_out_all_us": 1000 * h["mean_ms"], "held_out_all_pct": h["mean_pct"],
                              "max_key_all_us": 1000 * h["max_key_ms"], "n_keys": h["n_keys"],
                              "covered_calls": cov.get("n_calls"), "covered_keys": cov.get("n_keys"),
                              "covered_us": 1000 * cov["mean_ms"] if cov.get("n_calls") else None,
                              "covered_pct": cov.get("mean_pct"),
                              "covered_max_key_us": 1000 * cov["max_key_ms"] if cov.get("n_calls") else None,
                              "covered_keys_above_sync": cov.get("n_keys_above_threshold"),
                              "keys_above_sync_all": h.get("n_keys_above_threshold"), "sync_us": 1000 * SYNC_MS}
    res["table1_analog"] = table1_analog(t)
    res["curves"] = curves(t)
    return res


# ----------------------------------------------------------------------------- comparisons
def compare(a: Table, b: Table) -> dict:
    """Table a (new convention) against table b (reference) on the calls present in both."""
    pos_b = {k: i for i, k in enumerate(b.df["key"])}
    ia = np.array([i for i, k in enumerate(a.df["key"]) if k in pos_b])
    ib = np.array([pos_b[k] for k in a.df["key"].iloc[ia]])
    common = [c for c in a.chunks if c in b.chunks]
    ja, jb = [a.chunks.index(c) for c in common], [b.chunks.index(c) for c in common]
    la, lb = a.lat[np.ix_(ia, ja)], b.lat[np.ix_(ib, jb)]
    oa, ob = la.min(1), lb.min(1)
    xa, xb = la.argmin(1), lb.argmin(1)
    ca, cb = np.array(common)[xa], np.array(common)[xb]
    ratio = oa / ob
    Lb_ = a.df["L_bucket"].to_numpy()[ia]
    bat = a.batch[ia]
    fin = np.isfinite(la) & np.isfinite(lb)
    cell_ratio = (la / lb)[fin]

    def qs(v):
        return {"n": int(len(v)), "median": float(np.median(v)), "p5": float(np.percentile(v, 5)),
                "p95": float(np.percentile(v, 95))} if len(v) else {"n": 0}

    diff_us = 1000.0 * (oa - ob)
    out = {"a": a.name, "b": b.name, "n_joined": int(len(ia)), "common_chunks": common,
           "oracle_latency_ratio_a_over_b": qs(ratio), "all_cells_ratio_a_over_b": qs(cell_ratio),
           "oracle_diff_us_a_minus_b": qs(diff_us),
           "oracle_diff_us_by_L_bucket": {str(int(L)): qs(diff_us[Lb_ == L]) for L in np.unique(Lb_)},
           "oracle_ratio_by_L_bucket": {str(int(L)): qs(ratio[Lb_ == L]) for L in np.unique(Lb_)},
           "oracle_ratio_by_batch": {str(int(x)): qs(ratio[bat == x]) for x in np.unique(bat)},
           "oracle_ratio_by_batch_L_bucket": {f"{int(x)}x{int(L)}": float(np.median(ratio[(bat == x) & (Lb_ == L)]))
                                              for x in np.unique(bat) for L in np.unique(Lb_)
                                              if ((bat == x) & (Lb_ == L)).any()}}
    if a.default is not None and a.default in common:
        jd = common.index(a.default)
        out["default_chunk_ratio_a_over_b"] = qs((la[:, jd] / lb[:, jd])[np.isfinite(lb[:, jd])])
    ta = [a.tied[i] for i in ia]
    tb = [b.tied[i] for i in ib]
    out["oracle_same_share"] = float((ca == cb).mean())
    out["oracle_a_in_tieset_b_share"] = float(np.mean([c in s for c, s in zip(ca, tb)]))
    out["oracle_b_in_tieset_a_share"] = float(np.mean([c in s for c, s in zip(cb, ta)]))
    out["oracle_disjoint_share"] = float(np.mean([(x not in sb) and (y not in sa)
                                                  for x, y, sa, sb in zip(ca, cb, ta, tb)]))
    out["oracle_transitions_b_to_a"] = {f"{int(x)}->{int(y)}": int(v) for (x, y), v in
                                        pd.Series(list(zip(cb, ca))).value_counts().sort_index().items()}
    out["oracle_change_by_batch"] = {str(int(x)): float((ca[bat == x] != cb[bat == x]).mean()) for x in np.unique(bat)}
    # regret under a of the chunk chosen under b
    reg_b_oracle = la[np.arange(len(ia)), xb] / oa - 1.0
    sub_b = b.df.iloc[ib].reset_index(drop=True)
    pick_b_tab = insample_exact_pick(sub_b, lb)
    pick_a_tab = insample_exact_pick(a.df.iloc[ia].reset_index(drop=True), la)
    reg_b_tab = la[np.arange(len(ia)), pick_b_tab] / oa - 1.0
    reg_a_tab = la[np.arange(len(ia)), pick_a_tab] / oa - 1.0
    out["regret_under_a_pct"] = {
        "per_call_oracle_of_b": {"mean": _pct(reg_b_oracle.mean()), "p95": _pct(np.percentile(reg_b_oracle, 95)),
                                 "max": _pct(reg_b_oracle.max()),
                                 "us_per_call": float(1000 * (la[np.arange(len(ia)), xb] - oa).mean())},
        "exact_table_tuned_on_b_insample": {"mean": _pct(reg_b_tab.mean()), "p95": _pct(np.percentile(reg_b_tab, 95)),
                                            "us_per_call": float(1000 * (la[np.arange(len(ia)), pick_b_tab] - oa).mean())},
        "exact_table_tuned_on_a_insample": {"mean": _pct(reg_a_tab.mean()), "p95": _pct(np.percentile(reg_a_tab, 95)),
                                            "us_per_call": float(1000 * (la[np.arange(len(ia)), pick_a_tab] - oa).mean())}}
    rho = []
    for r in range(len(ia)):
        ok = fin[r]
        if ok.sum() >= 3:
            rho.append(spearmanr(la[r, ok], lb[r, ok])[0])
    rho = np.array([x for x in rho if np.isfinite(x)])
    out["per_call_spearman_chunk_profiles"] = {"median": float(np.median(rho)), "mean": float(rho.mean()),
                                               "share_ge_0p9": float((rho >= 0.9).mean()), "n": int(len(rho))}
    # subsets: above the reference floor and above both floors
    for name, sel in (("above_b_floor", b.above[ib]), ("above_both_floors", b.above[ib] & a.above[ia])):
        if sel.any():
            out[name] = {"n": int(sel.sum()), "oracle_ratio": qs(ratio[sel]),
                         "oracle_same_share": float((ca[sel] == cb[sel]).mean()),
                         "oracle_a_in_tieset_b_share": float(np.mean([c in s for c, s, k in zip(ca, tb, sel) if k])),
                         "regret_under_a_of_b_oracle_mean_pct": _pct(reg_b_oracle[sel].mean())}
    return out


# ----------------------------------------------------------------------------- claims
def claims(T: dict, S: dict) -> dict:
    out = {}

    def default_by_batch(t: Table, mask: np.ndarray) -> dict:
        j = t.chunks.index(int(t.default))
        reg = t.lat[:, j] / t.orc - 1.0
        return {str(int(b)): {"n": int((mask & (t.batch == b)).sum()), "mean_pct": _pct(reg[mask & (t.batch == b)].mean()),
                              "p95_pct": _pct(np.percentile(reg[mask & (t.batch == b)], 95)),
                              "max_pct": _pct(reg[mask & (t.batch == b)].max()),
                              "default_is_oracle": int((t.oidx[mask & (t.batch == b)] == j).sum())}
                for b in np.unique(t.batch) if (mask & (t.batch == b)).any()}

    def keyed_mask(t: Table, keys: set) -> np.ndarray:
        return t.df["key"].isin(keys).to_numpy()

    def misclassified(t: Table, km: np.ndarray) -> dict:
        """Calls above the device-time floor whose graph-timed oracle latency lies below the graph floor."""
        sel = km & ~t.above
        j = t.chunks.index(int(t.default))
        short = km & (t.L <= 256)
        return {"n": int(sel.sum()), "n_short_L_le_256_in_reference_set": int(short.sum()),
                "short_calls": [{"L": int(t.L[i]), "batch": int(t.batch[i]), "graph_oracle_us": float(1000 * t.orc[i]),
                                 "default_regret_pct": _pct(t.lat[i, j] / t.orc[i] - 1)} for i in np.where(short)[0]],
                "calls": [{"L": int(t.L[i]), "batch": int(t.batch[i]), "graph_oracle_us": float(1000 * t.orc[i]),
                           "graph_oracle_chunk": int(t.chunks[t.oidx[i]]),
                           "default_regret_pct": _pct(t.lat[i, j] / t.orc[i] - 1)} for i in np.where(sel)[0]]}

    # (1) SSD 2.7B: chunk 256 above the floor
    dev = T["ssd2p7b_device"]
    keys397 = set(dev.df.loc[dev.above, "key"])
    rec = {"paper": "chunk 256 is the oracle of 375 of the 397 captured calls above the floor and loses at most "
                    "0.24% per batch size (device time, floor 0.76 ms)",
           "device_time_recomputed": default_by_batch(dev, dev.above)}
    for nm in ("ssd2p7b_graph", "ssd2p7b_graphcold"):
        t = T[nm]
        km = keyed_mask(t, keys397)
        rec[nm] = {"same_397_keys": {"n": int(km.sum()), "by_batch": default_by_batch(t, km),
                                     "oracle_distribution": _dist(np.array(t.chunks)[t.oidx[km]])},
                   "same_397_keys_above_graph_floor": {"n": int((km & t.above).sum()),
                                                       "by_batch": default_by_batch(t, km & t.above),
                                                       "oracle_distribution": _dist(np.array(t.chunks)[t.oidx[km & t.above]])},
                   "device_above_but_graph_below_floor": misclassified(t, km),
                   "own_floor_above": {"floor_us": 1000 * t.floor_ms, "n": int(t.above.sum()),
                                       "by_batch": default_by_batch(t, t.above),
                                       "oracle_distribution": _dist(np.array(t.chunks)[t.oidx[t.above]])},
                   "all_calls": default_by_batch(t, np.ones(len(t.df), bool))}
    out["ssd2p7b_chunk256_above_floor"] = rec

    # (2) SSD 370M: best static / default per batch (Table II; Limitations 'at most 2.0% at the 370M width')
    rec = {"paper": "Table II (RTX PRO 6000, eager): best static regret 1.5 / 1.0 / 2.0% at batch 1 / 4 / 16; "
                    "Limitations: 'a single chunk loses at most 2.0% at the 370M width'"}
    for nm in ("ssd370m_pro6000_eager", "ssd370m_eager", "ssd370m_graph", "ssd370m_graphcold"):
        t = T[nm]
        rec[nm] = {"table2_all": S[nm]["table2"]["all"], "default_256_by_batch": default_by_batch(t, np.ones(len(t.df), bool)),
                   "default_256_above_floor_by_batch": default_by_batch(t, t.above)}
    # small-L flatness: chunk 32 / chunk 256 and default / oracle at exact shapes
    flat = {}
    for nm in ("ssd370m_eager", "ssd370m_graph", "ssd370m_graphcold", "ssd2p7b_graph"):
        t = T[nm]
        j32, j256 = t.chunks.index(32), t.chunks.index(256)
        cells = {}
        for b, L in ((1, 512), (1, 1024), (1, 2048), (1, 8192), (4, 1024), (16, 25), (16, 256), (16, 1024)):
            sel = (t.batch == b) & (t.L == L) & t.avail[:, j32]
            if sel.any():
                cells[f"b{b}_L{L}"] = {"calls": int(sel.sum()),
                                       "c32_over_c256_median": float(np.median(t.lat[sel, j32] / t.lat[sel, j256])),
                                       "c256_regret_pct_median": _pct(np.median(t.lat[sel, j256] / t.orc[sel] - 1)),
                                       "c256_regret_us_median": float(1000 * np.median(t.lat[sel, j256] - t.orc[sel])),
                                       "oracle_us_median": float(1000 * np.median(t.orc[sel])),
                                       "oracle_distribution": _dist(np.array(t.chunks)[t.oidx[sel]])}
        flat[nm] = cells
    rec["exact_shape_cells"] = flat
    out["ssd370m_static_and_flatness"] = rec

    # (3) GDN: chunk 64 the oracle of 154 of 155 calls above the floor
    dev = T["gdn1p4b_device"]
    keys155 = set(dev.df.loc[dev.above, "key"])
    j64 = dev.chunks.index(64)
    rec = {"paper": "chunk 64 is the oracle of 154 of 155 captured calls above the floor (device time, floor 1.44 ms)",
           "device_time_recomputed": {"n": int(dev.above.sum()), "chunk64_oracle": int((dev.oidx[dev.above] == j64).sum())}}
    for nm in ("gdn1p4b_graph", "gdn1p4b_graphcold"):
        t = T[nm]
        km = keyed_mask(t, keys155)
        rec[nm] = {"same_155_keys": {"n": int(km.sum()), "chunk64_oracle": int((t.oidx[km] == j64).sum()),
                                     "by_batch": default_by_batch(t, km)},
                   "same_155_keys_above_graph_floor": {"n": int((km & t.above).sum()),
                                                       "chunk64_oracle": int((t.oidx[km & t.above] == j64).sum()),
                                                       "by_batch": default_by_batch(t, km & t.above)},
                   "device_above_but_graph_below_floor": misclassified(t, km),
                   "own_floor_above": {"floor_us": 1000 * t.floor_ms, "n": int(t.above.sum()),
                                       "chunk64_oracle": int((t.oidx[t.above] == j64).sum()),
                                       "by_batch": default_by_batch(t, t.above),
                                       "oracle_distribution": _dist(np.array(t.chunks)[t.oidx[t.above]])},
                   "all_calls": {"n": len(t.df), "chunk64_oracle": int((t.oidx == j64).sum()),
                                 "by_batch": default_by_batch(t, np.ones(len(t.df), bool))}}
    out["gdn_chunk64"] = rec

    # (4) library defaults suffice: default vs held-out exact table, in us/call and ms per forward pass
    rec = {"paper": "Table deployment (ms per forward pass, default / model / table): SSD 370M PRO 6000 0.37 / 0.32 / "
                    "0.28, SSD 370M 0.98 / 0.96 / 0.89, SSD 2.7B dev. 0.24 / 0.29 / 0.15 (above floor), GDN 1.4B dev. "
                    "<0.01 (above floor); 'on SSD and Gated DeltaNet a tuned table gains at most 0.09 ms over the "
                    "default'"}
    for nm in ("ssd370m_pro6000_eager", "ssd370m_eager", "ssd370m_graph", "ssd370m_graphcold", "ssd2p7b_device",
               "ssd2p7b_graph", "ssd2p7b_graphcold", "gdn1p4b_device", "gdn1p4b_graph", "gdn1p4b_graphcold"):
        pol = S[nm]["policies"]
        r = {}
        for sub in ("all", "above_floor"):
            d, tb = pol["default"][sub], pol["exact_table_heldout"][sub]
            mdl = pol.get("work_term_original", {}).get(sub, {})
            mld = pol.get("work_term_launch_default", {}).get(sub, {})
            if d.get("n", 0) == 0:
                continue
            r[sub] = {"n": d["n"], "default_ms_per_pass": d["ms_per_forward_pass"],
                      "model_original_ms_per_pass": mdl.get("ms_per_forward_pass"),
                      "model_launch_default_ms_per_pass": mld.get("ms_per_forward_pass"),
                      "table_ms_per_pass": tb["ms_per_forward_pass"],
                      "default_minus_table_ms_per_pass": d["ms_per_forward_pass"] - tb["ms_per_forward_pass"],
                      "default_us_per_call": d["us_per_call"], "table_us_per_call": tb["us_per_call"],
                      "default_mean_p95_pct": [d["mean_pct"], d["p95_pct"]], "table_mean_p95_pct": [tb["mean_pct"], tb["p95_pct"]],
                      "default_share_regret_gt_5pct": None}
            t = T[nm]
            j = t.default_idx()
            msk = t.masks()[sub]
            r[sub]["default_share_regret_gt_5pct"] = float(((t.lat[msk, j] / t.orc[msk] - 1) > 0.05).mean())
            r[sub]["default_share_regret_gt_10pct"] = float(((t.lat[msk, j] / t.orc[msk] - 1) > 0.10).mean())
        rec[nm] = r
    out["defaults_suffice_deployment_units"] = rec
    return out


# ----------------------------------------------------------------------------- launch-bound boundary
def launch_boundary(T: dict, S: dict, in_dir: Path) -> dict:
    hr = json.loads((in_dir / "headroom_r4.json").read_text())["M3c"]["hosts"]
    out = {"paper_eager": {
        "vgpu3090": {"fitted_F_us": 1000 * hr["vgpu3090"]["boundary"]["fitted_F_ms"],
                     "BL_star_tokens": hr["vgpu3090"]["boundary"]["BL_star_fitted_F"],
                     "BL_star_at_320us": hr["vgpu3090"]["boundary"]["BL_star_lowered_F"],
                     "L_star_by_batch_at_320us": hr["vgpu3090"]["boundary"]["L_star_lowered_F_by_batch"]},
        "pro6000": {"fitted_F_us": 1000 * hr["pro6000"]["boundary"]["fitted_F_ms"],
                    "BL_star_tokens": hr["pro6000"]["boundary"]["BL_star_fitted_F"],
                    "BL_star_at_320us": hr["pro6000"]["boundary"]["BL_star_lowered_F"]},
        "printed": "about 9,500 (RTX 3090) and 17,000 (RTX PRO 6000) tokens per call at the fitted floor; about "
                   "3,900 tokens (3,900 / 970 / 240 per sequence at batch 1 / 4 / 16) at a 320 us floor"}}
    for nm in ("ssd370m_graph", "ssd370m_graphcold", "ssd370m_eager", "ssd370m_device", "ssd2p7b_graph",
               "ssd2p7b_device", "gdn1p4b_graph", "gdn1p4b_device"):
        s = S[nm]
        w = s.get("work_term_model", {})
        out[nm] = {"floor_us": s["floor"]["floor_us"], "median_b1_L_le_256_us": s["floor"]["median_b1_L_le_256_us"],
                   "measured_floor_eager_terms": (w.get("measured_floor") or {}).get("boundary"),
                   "refit_all_calls": (w.get("refit") or {}).get("boundary_all_calls_fit"),
                   "refit_all_calls_r2_log": (w.get("refit") or {}).get("all_calls_r2_log"),
                   "empirical_boundary_tokens_bin": s["curves"]["empirical_boundary_tokens_bin"],
                   "share_above_floor_by_tokens": {str(r["tokens_bin"]): r["share_above_floor"]
                                                   for r in s["curves"]["by_tokens"]},
                   "spread_median_by_tokens": {str(r["tokens_bin"]): r["spread_median"] for r in s["curves"]["by_tokens"]}}
    return out


# ----------------------------------------------------------------------------- reproduction
PRINTED = {
    "table2_mamba2_pro6000": {  # main Table II (tab:oracle_regret), Mamba-2 rows
        "1": {"calls": 1536, "oracle_ne_static": "70", "best_static": 64, "static": ("1.5", "5.9"), "rule": ("1.8", "6.3")},
        "4": {"calls": 488, "oracle_ne_static": "73", "best_static": 256, "static": ("1.0", "2.8"), "rule": ("1.1", "2.9")},
        "16": {"calls": 336, "oracle_ne_static": "57", "best_static": 256, "static": ("2.0", "6.8"), "rule": ("2.3", "11.4")}},
    "table3_mamba2_pro6000": {  # main Table III (tab:predictors), Mamba-2 column: mean / p95 / tied %
        "e3_global_static": ("1.7", "5.8", "68"), "e3_entropy_rule": ("1.8", "6.1", "68"), "e3_L_only": ("1.7", "6.1", "75"),
        "e3_shape": ("2.2", "5.6", "79"), "e3_shape+entropy": ("2.2", "5.6", "79"), "e3_all": ("2.2", "5.4", "79"),
        "e3_entropy_only": ("1.7", "5.9", "69"), "e3_moments": ("1.8", "6.2", "70"),
        "exact_table_heldout": ("1.3", "5.2", "81"), "work_term_original": ("1.5", "5.3", "75")},
    "spread_tied_mamba2_pro6000": {"spread_median": "1.04", "spread_p95": "2.7", "tied_count": "3.1"},
    "conventions_mamba2_vgpu3090": {"cv": "4.5", "e3_global_static": "2.5", "e3_shape": "2.4",
                                    "e3_shape+entropy": "2.4", "e3_entropy_only": "2.5", "headroom_covered_us": "17.20"},
    "headroom_noise": {"ssd370m_pro6000_eager": {"all": "5.89", "covered": "6.06", "max": "13.7", "above_sync": 0},
                       "ssd370m_eager": {"all": "18.6", "covered": "17.2", "max": "43.0", "above_sync": 5}},
    "largewidth_calls": {
        "ssd2p7b_device": {"calls": 2360, "above": 397, "floor_ms": "0.76", "cv_below": "74", "cv_above": "0.9",
                           "spread_median": "2.05", "spread_p95": "2.19", "static": 256, "static_oracle": 375,
                           "static_regret": ("0.24", "0.16", "0.08"), "headroom_us": "0.9", "headroom_calls": 348,
                           "headroom_keys": 12, "keys_above_sync": 1},
        "gdn1p4b_device": {"calls": 1770, "above": 155, "floor_ms": "1.44", "cv_below": "64", "cv_above": "0.6",
                           "spread_median": "1.82", "spread_p95": "1.90", "static": 64, "static_oracle": 154,
                           "static_regret": ("0.00", "0.00", "0.05"), "headroom_us": "0.0", "headroom_calls": 144,
                           "headroom_keys": 6, "keys_above_sync": 0}},
    "deployment": {  # main Table deployment, ms per forward pass: default / model / table
        "ssd370m_pro6000_eager": ("all", "0.37", "0.32", "0.28"), "ssd370m_eager": ("all", "0.98", "0.96", "0.89"),
        "ssd2p7b_device": ("above_floor", "0.24", "0.29", "0.15"), "gdn1p4b_device": ("above_floor", "<0.01", "<0.01", "<0.01")},
    "m3c": {  # appendix, multi-kernel pipelines: original -> launch_default mean regret %
        "ssd370m_pro6000_eager": ("all", "1.45", "1.56"), "ssd370m_eager": ("all", "2.48", "2.36"),
        "ssd370m_device": ("all", "5.72", "5.61"), "ssd2p7b_device": ("all", "11.40", "12.74")},
    "m3c_above": {"ssd2p7b_device": ("above_floor", "0.24", "0.19")},
}


def reproduction(S: dict, in_dir: Path) -> dict:
    checks = []

    def add(what, value, printed):
        ok = _matches(value, printed) if isinstance(printed, str) else (value == printed)
        checks.append({"check": what, "recomputed": value, "printed": printed, "match": bool(ok)})

    t2 = S["ssd370m_pro6000_eager"]["table2"]["all"]
    for b, p in PRINTED["table2_mamba2_pro6000"].items():
        r = t2[b]
        add(f"Table II M2 b{b} calls", r["calls"], p["calls"])
        add(f"Table II M2 b{b} oracle != static %", r["oracle_ne_best_static_pct"], p["oracle_ne_static"])
        add(f"Table II M2 b{b} best static", r["best_static_chunk"], p["best_static"])
        add(f"Table II M2 b{b} static mean", r["static_regret_pct"][0], p["static"][0])
        add(f"Table II M2 b{b} static p95", r["static_regret_pct"][1], p["static"][1])
        add(f"Table II M2 b{b} rule mean", r["rule_regret_pct"][0], p["rule"][0])
        add(f"Table II M2 b{b} rule p95", r["rule_regret_pct"][1], p["rule"][1])
    pol = S["ssd370m_pro6000_eager"]["policies"]
    for k, (mn, p95, tied) in PRINTED["table3_mamba2_pro6000"].items():
        a = pol[k]["all"]
        add(f"Table III M2 {k} mean", a["mean_pct"], mn)
        add(f"Table III M2 {k} p95", a["p95_pct"], p95)
        add(f"Table III M2 {k} tied %", 100 * a["tied_share"], tied)
    s = S["ssd370m_pro6000_eager"]
    p = PRINTED["spread_tied_mamba2_pro6000"]
    add("Sec. oracle: M2 spread median (PRO 6000)", s["spread"]["all"]["median"], p["spread_median"])
    add("Sec. oracle: M2 spread p95 (PRO 6000)", s["spread"]["all"]["p95"], p["spread_p95"])
    add("Sec. oracle: M2 chunks tied with oracle (PRO 6000)", s["tied_with_oracle_mean_count"], p["tied_count"])
    s = S["ssd370m_eager"]
    p = PRINTED["conventions_mamba2_vgpu3090"]
    add("Table S-conventions M2 CV", s["within_block_cv_pct"]["all"]["cell_median"], p["cv"])
    for k in ("e3_global_static", "e3_shape", "e3_shape+entropy", "e3_entropy_only"):
        add(f"Table S-conventions M2 {k}", s["policies"][k]["all"]["mean_pct"], p[k])
    add("Table S-conventions M2 headroom", s["headroom"]["all"]["covered_us"], p["headroom_covered_us"])
    for nm, p in PRINTED["headroom_noise"].items():
        h = S[nm]["headroom"]["all"]
        add(f"Table S-headroom-noise {nm} held-out all", h["held_out_all_us"], p["all"])
        add(f"Table S-headroom-noise {nm} held-out covered", h["covered_us"], p["covered"])
        add(f"Table S-headroom-noise {nm} held-out max", h["covered_max_key_us"], p["max"])
        add(f"Table S-headroom-noise {nm} covered keys above sync", h["covered_keys_above_sync"], p["above_sync"])
    for nm, p in PRINTED["largewidth_calls"].items():
        s = S[nm]
        add(f"Table S-largewidth-calls {nm} calls", s["n_calls"], p["calls"])
        add(f"Table S-largewidth-calls {nm} above", s["floor"]["n_above"], p["above"])
        add(f"Table S-largewidth-calls {nm} floor ms", s["floor"]["floor_us"] / 1000, p["floor_ms"])
        add(f"Table S-largewidth-calls {nm} CV below (per-call median)", s["within_block_cv_pct"]["below_floor"]["per_call_median"], p["cv_below"])
        add(f"Table S-largewidth-calls {nm} CV above (per-call median)", s["within_block_cv_pct"]["above_floor"]["per_call_median"], p["cv_above"])
        add(f"Table S-largewidth-calls {nm} spread median", s["spread"]["above_floor"]["median"], p["spread_median"])
        add(f"Table S-largewidth-calls {nm} spread p95", s["spread"]["above_floor"]["p95"], p["spread_p95"])
        add(f"Table S-largewidth-calls {nm} static chunk", s["above_floor_static"]["chunk_most_often_oracle"], p["static"])
        add(f"Table S-largewidth-calls {nm} static oracle count", s["above_floor_static"]["oracle_count"], p["static_oracle"])
        for b, v in zip(("1", "4", "16"), p["static_regret"]):
            add(f"Table S-largewidth-calls {nm} static regret b{b}", s["above_floor_static"]["regret_pct_by_batch"][b]["mean"], v)
        h = s["headroom"]["above_floor"]
        add(f"Table S-largewidth-calls {nm} headroom us", h["covered_us"], p["headroom_us"])
        add(f"Table S-largewidth-calls {nm} headroom calls", h["covered_calls"], p["headroom_calls"])
        add(f"Table S-largewidth-calls {nm} headroom keys", h["covered_keys"], p["headroom_keys"])
        add(f"Table S-largewidth-calls {nm} keys above sync", h["covered_keys_above_sync"], p["keys_above_sync"])
    for nm, (sub, d, mdl, tb) in PRINTED["deployment"].items():
        pol = S[nm]["policies"]
        add(f"Table deployment {nm} default", pol["default"][sub]["ms_per_forward_pass"], d)
        add(f"Table deployment {nm} model", pol["work_term_original"][sub]["ms_per_forward_pass"], mdl)
        add(f"Table deployment {nm} table", pol["exact_table_heldout"][sub]["ms_per_forward_pass"], tb)
    for key in ("m3c", "m3c_above"):
        for nm, (sub, orig, ld) in PRINTED[key].items():
            pol = S[nm]["policies"]
            add(f"M3c {nm} {sub} original", pol["work_term_original"][sub]["mean_pct"], orig)
            add(f"M3c {nm} {sub} launch_default", pol["work_term_launch_default"][sub]["mean_pct"], ld)
    # stored E3 files (every predictor, all batches) and stored M3c rule values (exact)
    stored = []
    for nm, f in (("ssd370m_pro6000_eager", "e3_mamba2_pro6000.json"), ("ssd370m_eager", "e3_mamba2_vgpu3090.json"),
                  ("ssd370m_device", "e3_mamba2_vgpu3090_r3spin.json"), ("ssd2p7b_device", "e3_mamba2_vgpu3090_2p7b_r3.json"),
                  ("gdn1p4b_device", "e3_gdn_vgpu3090_1p4b_r3.json"), ("m1_370m_eager", "e3_mamba1_vgpu3090.json"),
                  ("m1_370m_device", "e3_mamba1_vgpu3090_r3spin.json")):
        pth = in_dir / f
        if not pth.exists():
            continue
        pa = json.loads(pth.read_text())["predictors_all_batches"]
        diffs = [abs(v["mean_regret_pct"] - S[nm]["policies"][f"e3_{k}"]["all"]["mean_pct"]) for k, v in pa.items()]
        stored.append({"table": nm, "file": f, "n_predictors": len(diffs), "max_abs_diff_pp": float(max(diffs))})
    hr = json.loads((in_dir / "headroom_r4.json").read_text())["M3c"]["hosts"]
    m3c_stored = []
    for nm, host, csv in (("ssd370m_pro6000_eager", "pro6000", "e2_mamba2_pro6000.csv"),
                          ("ssd370m_eager", "vgpu3090", "e2_mamba2_vgpu3090.csv"),
                          ("ssd370m_device", "vgpu3090", "e2_mamba2_vgpu3090_r3spin.csv"),
                          ("ssd2p7b_device", "vgpu3090", "e2_mamba2_vgpu3090_2p7b_r3.csv")):
        rules = hr[host]["datasets"][csv]["rules"]
        for rule in WORK_RULES:
            for sub in ("all", "above_floor"):
                if sub in rules[rule] and rules[rule][sub].get("n"):
                    m3c_stored.append({"table": nm, "rule": rule, "subset": sub,
                                       "abs_diff_pp": abs(rules[rule][sub]["mean_regret_pct"]
                                                          - S[nm]["policies"][f"work_term_{rule}"][sub]["mean_pct"])})
    notes = ["Table S-conventions CV is the median over (call, chunk) cells; Table S-largewidth-calls CV is the median "
             "over calls of the per-call median (the cell median gives " + ", ".join(
                 f"{nm} {S[nm]['within_block_cv_pct']['below_floor']['cell_median']:.2f} / "
                 f"{S[nm]['within_block_cv_pct']['above_floor']['cell_median']:.2f}"
                 for nm in ("ssd2p7b_device", "gdn1p4b_device")) + " below / above the floor, i.e. 63 / 0.7 instead "
             "of the printed 64 / 0.6 for Gated DeltaNet)"]
    return {"printed": checks, "n_printed": len(checks), "n_printed_match": sum(c["match"] for c in checks), "notes": notes,
            "mismatches": [c for c in checks if not c["match"]], "stored_e3": stored,
            "stored_m3c_max_abs_diff_pp": max(x["abs_diff_pp"] for x in m3c_stored), "stored_m3c_n": len(m3c_stored),
            "e3_evaluate_consistency_max_abs_diff_pp": max(s["check_e3_evaluate_max_abs_diff_pp"] for s in S.values())}


# ----------------------------------------------------------------------------- report
def print_report(res: dict) -> None:
    S = res["tables"]
    print("table                      timing                          n    floor_us above  CV%cell(all/below/above)   "
          "default mean/p95  best-static  shape  model(LD)  table  headroom_cov_us(max)")
    for nm, s in S.items():
        pol = s["policies"]
        cv = s["within_block_cv_pct"]
        h = s["headroom"].get("all", {})
        mld = pol.get("work_term_launch_default", {}).get("all", {}).get("mean_pct")
        print(f"{nm:26s} {s['timing'][:30]:30s} {s['n_calls']:5d} {s['floor']['floor_us']:8.1f} {s['floor']['n_above']:5d}  "
              f"{cv['all']['cell_median']:5.2f}/{(cv['below_floor']['cell_median'] or float('nan')):5.2f}/"
              f"{(cv['above_floor']['cell_median'] or float('nan')):5.2f}   "
              f"{pol['default']['all']['mean_pct']:6.2f}/{pol['default']['all']['p95_pct']:6.1f}  "
              f"{pol['best_static_insample']['all']['mean_pct']:6.2f}  {pol['e3_shape']['all']['mean_pct']:5.2f}  "
              f"{(mld if mld is not None else float('nan')):6.2f}  {pol['exact_table_heldout']['all']['mean_pct']:5.2f}  "
              f"{(h.get('covered_us') or float('nan')):6.2f} ({(h.get('covered_max_key_us') or float('nan')):.1f})")
    print("\ncomparisons (a vs b): oracle ratio median, oracle same, a-oracle in b tie set, regret under a of b's oracle")
    for k, c in res["comparisons"].items():
        print(f"  {k:45s} n={c['n_joined']:5d} ratio {c['oracle_latency_ratio_a_over_b']['median']:.3f} "
              f"same {c['oracle_same_share']:.2f} tie {c['oracle_a_in_tieset_b_share']:.2f} "
              f"cross-regret {c['regret_under_a_pct']['per_call_oracle_of_b']['mean']:.2f}% "
              f"spearman {c['per_call_spearman_chunk_profiles']['median']:.2f}")
    lb = res["launch_boundary"]
    print("\nlaunch boundary (tokens per call):", {k: (round(v.get("floor_us", 0), 1),
                                                     (v.get("measured_floor_eager_terms") or {}).get("BL_star_tokens"),
                                                     (v.get("refit_all_calls") or {}).get("BL_star_tokens"),
                                                     v.get("empirical_boundary_tokens_bin"))
                                                 for k, v in lb.items() if k != "paper_eager"})
    rp = res["reproduction"]
    print(f"\nreproduction: {rp['n_printed_match']}/{rp['n_printed']} printed values match; stored e3 max diff "
          f"{max(x['max_abs_diff_pp'] for x in rp['stored_e3']):.2e} pp; stored M3c max diff "
          f"{rp['stored_m3c_max_abs_diff_pp']:.2e} pp; e3.evaluate consistency {rp['e3_evaluate_consistency_max_abs_diff_pp']:.2e}")
    for m in rp["mismatches"]:
        print(f"  MISMATCH {m['check']}: recomputed {m['recomputed']} printed {m['printed']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--out", type=Path, default=None, help="output JSON (default: <in-dir>/e22_graph_analysis_vgpu3090.json; "
                   "with --rounds3: e22_graph_analysis_vgpu3090_rounds3.json)")
    p.add_argument("--rounds3", action="store_true",
                   help="use the three-round warm graph sweeps (E25) where present; single-round ones become <name>1")
    args = p.parse_args(argv)
    with np.errstate(invalid="ignore", divide="ignore"):    # +inf marks chunks beyond the SSD 32-bit index limit
        run(args)


def run(args) -> None:
    t0 = time.time()
    in_dir = args.in_dir
    switched = use_three_rounds(in_dir) if getattr(args, "rounds3", False) else []
    if switched:
        print(f"three-round graph tables: {switched}", flush=True)
    T = {nm: Table(nm, in_dir) for nm in TABLES if (in_dir / TABLES[nm][0]).exists()}
    for nm, t in T.items():
        if t.floor_from is None:
            t.set_floor()
    for nm, t in T.items():
        if t.floor_from is not None:
            t.set_floor(T[t.floor_from].floor_ms)
    S = {}
    for nm, t in T.items():
        S[nm] = summarize(t, in_dir)
        print(f"summarized {nm} ({time.time() - t0:.0f} s)", flush=True)
    comps = {f"{a}__vs__{b}": compare(T[a], T[b]) for a, b in COMPARISONS if a in T and b in T}
    res = {"experiment": "E22-analysis (review M3b: CUDA-graph timing, RTX 3090)", "meta": env_metadata(),
           "settings": {"sync_us": 1000 * SYNC_MS, "floor": "2 x median oracle latency of the batch-1 calls with "
                        "L <= 256 (Table S-largewidth-calls); above floor: oracle latency > floor",
                        "regret": "per-call relative regret T(c) / T(c*) - 1 in percent unless stated; us_per_call "
                                  "absolute; ms_per_forward_pass = us_per_call x scan layers / 1000",
                        "tied": "chosen chunk in the call's statistical tie set (95% CI overlap, tied_chunks column)",
                        "cv": "within-block coefficient of variation std / mean of the 50 repeats of one (call, chunk) "
                              "block; round-median view: between-round SD over the median",
                        "predictors": "e3_* trained on all calls with grouped 5-fold CV (prompt folds) and scored on "
                                      "the subset; best_static_insample chosen on all calls"},
           "tables": S, "comparisons": comps, "claims": claims(T, S), "launch_boundary": launch_boundary(T, S, in_dir)}
    res["reproduction"] = reproduction(S, in_dir)
    res["three_round_tables"] = switched
    res["runtime_s"] = round(time.time() - t0, 1)
    out = args.out or (in_dir / ("e22_graph_analysis_vgpu3090_rounds3.json" if switched
                                 else "e22_graph_analysis_vgpu3090.json"))
    write_json(out, _clean(res))
    print_report(_clean(res))
    print(f"\nwrote {out} ({res['runtime_s']} s)")


if __name__ == "__main__":
    main()
