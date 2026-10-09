"""T1 baselines, round 4 - clean form selection, calibration-only baselines and the top-k hybrid
(review items M5a, M5b and M6d). CPU only; reads existing result files, no GPU.

Data (per result tag; ``--tags``, default pro6000, vgpu3090, h800 and the device-time tags
vgpu3090_r3spin, h800_r3spin):
  calibration     the E4 launch-space sweep (12 shape keys, L in {128, 512, 2048, 8192} x batch
                  in {1, 4, 16}, d = 2048, 184 launch configurations each), the E4 static
                  re-measurements (default launch configuration of every chunk) and the E1 medians
                  (batch 1, d = 2048), exactly as ``cost_model.load_points`` loads them.
  captured calls  the 2360 E2 Mamba-1 calls (``e2_mamba1_<tag>.csv``); scored exactly as
                  ``cost_model.e2_eval`` / ``cost_model_baselines.evaluate`` score them: relative
                  regret lat(choice) / lat(per-call oracle) - 1 (mean, p95, p99, max) and the tied
                  share (chosen chunk inside the call's statistically tied set).
  shard widths    E12 (``e12_tp_widths_<tag>.jsonl``, Triton kernel, 168 shapes: widths 512..5120,
                  L 128..8192, batch 1/4/16), scored exactly as ``e12_analysis.regret`` (regret
                  against the round-mean oracle of the shape, tied = 95% CIs of round means overlap),
                  pooled over the 126 shard shapes (t > 1) and per (checkpoint, t).
No selector below sees a captured-call or E12 timing unless stated (the captured-call selection
of protocol M5a-ii is used only on a GPU other than the one it is tested on).

M5a - clean selection of the terms of Eq. (5)
  Candidate forms: the twelve forms of the ablation table (full model with eight parameters,
  seven drop-one forms, four add-one forms; ``cost_model.ABLATIONS``) and, as a 13th candidate,
  the three-parameter restriction ``work_only`` of ``cost_model_baselines``. Every form is fitted
  on the calibration data of the tag (the full form uses the stored parameters; a refit
  reproduces them) and scored on the captured calls.
  (i)  calibration only: leave one calibration shape out. For each of the 12 E4 shapes (L, B),
       every calibration point of that shape (sweep, static re-measurements and, for batch 1, the
       E1 medians of the same shape) is removed, the form is refitted on the rest (start: the
       form's all-data fit), and the chunk with the lowest predicted latency at the default
       launch configuration is compared with the measured static re-measurements of the held-out
       shape (chunk regret). Also the held-out log-latency RMSE over all held-out points.
       Rules: ``strict`` = lowest mean chunk regret, exact ties -> fewer free parameters -> lower
       held-out RMSE; ``one_se`` = the form with the fewest free parameters whose mean chunk
       regret is within one standard error (over the 12 shapes) of the minimum (ties -> lower
       mean -> lower held-out RMSE); ``rmse`` = lowest held-out log-latency RMSE. Forms the rule
       cannot tell apart (same criterion, same parameter count) are listed with their
       captured-call regrets (``exact_tie_alternatives``). Candidate sets: the 12 ablation forms
       (``ablation12``) and those plus the three-parameter restriction (``ablation12+work_only``).
  (ii) cross-GPU: the form with the lowest captured-call mean regret on a source GPU (``strict``:
       exact ties -> fewer parameters -> the source's held-out RMSE; ``one_se``: SE over calls) is
       refitted on the calibration data of a target GPU and scored on the target's captured calls.
       Sources are the three primary tags with a device different from the target's. ``leave_one_gpu_out`` selects on
       the mean of the per-GPU mean regrets of the two other primary GPUs.

M5b - selectors fitted on calibration data only (scored on captured calls, shard widths and
  cross-GPU: fitted on GPU A, applied with the hardware constants and compile facts of GPU B to
  B's captured calls)
  cost_model           Eq. (5), stored eight-parameter fit (``cost_model_<tag>.json``).
  work_only            three-parameter restriction of Eq. (5) (``cost_model_baselines``).
  work_term_model      (F^p + W^p)^(1/p), W = B L (a + b C) + B ceil(L / C) g (``cost_model_baselines``).
  nearest_calibrated_key   best static chunk of the nearest E4 key (``cost_model_baselines``).
  interp_log_length_same_batch   existing variant: normalized curves of the two bracketing
                       calibrated lengths at the same batch, linear in log2 L (captured calls only).
  loglinear_interp_logL_logB   new: log latency of every chunk interpolated bilinearly in
                       (log2 L, log2 B) over the 4 x 3 grid of E4 static re-measurement means,
                       clamped at the grid edges; the width is ignored (the grid has one width).
  length_tree_calibration  regret-labelled decision tree (as E3's L_only: depth 3, leaves labelled
                       with the chunk of lowest summed normalized latency) on log2 L, trained on the
                       12 calibration shapes (E4 static means, min_samples_leaf = 1).
  shape_tree_calibration   the same on (log2 L, log2 B), depth 4.
  gbr_basic / gbr_structural   histogram gradient-boosted regressor on log latency, trained on all
                       calibration timings (2298 points, all launch configurations). ``basic``
                       features: log2 L, B, d, C, BLOCK_D, num_warps, num_stages, alignment
                       (L mod 16 = 0), log2 ceil(L / C), log2 programs (ceil(d / BLOCK_D) B) and
                       the compile facts of the configuration (registers, spill store/load bytes,
                       shared bytes, the distributed layout: size per thread, threads per warp and
                       warps per CTA along both tile axes; ``kernel_stats_sm<arch>.json``).
                       ``structural`` adds the derived quantities of Eq. (5) (elements per thread,
                       rows per thread, lanes and warps along the scan axis, loads, spill words, L1
                       miss share, resident programs per SM, waves, resident programs, L2 miss
                       share, log bytes / bandwidth). Columns constant on the calibration data
                       (e.g. log2 d) are dropped. Hyperparameters (max_iter x max_leaf_nodes) are
                       chosen by leave-one-calibration-shape-out MSE of log latency (12 folds,
                       calibration data only). The chunk with the lowest predicted latency at the
                       default launch configuration is chosen per call.
  mlp_basic / mlp_structural   standardized inputs, two hidden layers of 64 ReLU units, Adam
                       (mini-batches of 128, learning rate 1e-3, 300 epochs, no early stopping),
                       L2 penalty chosen by the same 12-fold calibration CV; random_state 0.
  tfla_runtime         TFLA's theoretical runtime (Beck et al., "Tiled Flash Linear Attention",
                       NeurIPS 2025, arXiv 2503.14376, App. G.1, Eq. "theoretical runtime
                       mLSTMsig"): no overlap of compute and memory traffic,
                           tau = N_batch N_head N_chunk (Bytes(C) / beta + FLOPs(C) / alpha),
                       with their per-head, per-chunk counts of the chunkwise-parallel form
                       (App. F, simplified FLOP counts and memory-operation counts, mLSTMsig):
                           Bytes(C) = 4 C bytes_if + 3 C (d_hv + d_qk) bytes_qkv + 2 d_hv d_qk bytes_Cmn
                           FLOPs(C) = C^2 F_causal (2 d_qk + 2 d_hv + 6) + C^2
                                      + C (4 d_hv d_qk + 2 d_qk + d_hv + 11) + 2 d_hv d_qk + 5.
                       Instantiation for the Mamba-1 scan: every channel is a head (N_head = d,
                       d_hv = 1) with an N = 16 dimensional state (d_qk = 16), N_batch = B,
                       sequence length T = L, chunk C, N_chunk = ceil(L / C) (TFLA writes T / L for
                       divisible lengths; the kernel pads the last tile), bytes_qkv = bytes_if = 2
                       (fp16 activations), bytes_Cmn = 4 (fp32 state), F_causal = 0.5 and all
                       F_OP = 1 (TFLA's simplified counts). The free constants alpha (FLOP/s) and
                       beta (byte/s) are fitted (robust least squares on log latency, as the other
                       fits) on the calibration timings at the default launch configuration (E4
                       static re-measurements and E1 medians). The chunk choice depends only on
                       alpha / beta (TFLA's accelerator intensity I_acc), whose fitted value and
                       TFLA's closed-form optimum L_opt,Runtime are reported. TFLA's scalar-decay
                       gate matrix is kept (Mamba-1 decays per state, which only rescales the C^2
                       coefficient). Variants: ``tfla_runtime_overlap`` (their lower bound
                       max(FLOPs / alpha, Bytes / beta)), ``tfla_runtime_T_over_C`` (literal
                       N_chunk = L / C, which makes the choice independent of L: one chunk for
                       every call) and ``tfla_runtime_launch_floor`` (an added constant t0; not in
                       TFLA, it changes the fit but not the choice). Besides the two-constant fit, the
                       boundary models alpha -> infinity (memory term only) and beta -> infinity
                       (FLOP term only) are fitted; when a boundary model fits at least as well, the
                       least-squares optimum lies on that boundary and the boundary model is used
                       (``boundary`` in the output), e.g. argmin_C ceil(L / C) Bytes(C) when the FLOP
                       term is rejected.
  tfla_runtime_intensity_<I>   TFLA's expression with datasheet-style constants instead of a fit:
                       only I = alpha / beta matters for the choice; I in {64, 128, 256, 512}
                       FLOP/byte spans TFLA's accelerator table (133..295) and the dense-BF16
                       ratios of the RTX 3090 (about 76) and H800 PCIe (about 378).
  E12 only, for reference (they use E12 timings): width2048_table, unsharded_table, best_static
                       (``e12_analysis``).

M6d - "rank by the cost model, profile the top-k, keep the faster": exactly as
  ``cost_model.topk_refinement`` (per exact shape (L, batch) the k chunks with the lowest predicted
  latency are profiled; the measured best, by the mean latency over the shape's calls, is served
  to every call of the shape), k = 1, 2, 3 and all six chunks (the exact-shape-table floor); mean,
  p95, p99, max regret and tied share. The same on E12 shapes (per shape, round-mean latencies),
  for the cross-GPU transfer (ranking with the source GPU's parameters), and with the other
  rankers (predicted or tabulated latencies of every selector above).

Outputs: ``cost_model_baselines_r4_<tag>.json`` per tag and ``cost_model_baselines_r4_summary.json``
(cross-tag tables: clean selection, baseline table, cross-GPU transfer, top-k, shard widths).
Deterministic: fixed random_state everywhere, single-threaded BLAS, no sampling.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # deterministic BLAS; parallelism is across tags

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from src.experiments.tpds import cost_model as cm  # noqa: E402
from src.experiments.tpds import cost_model_baselines as cmb  # noqa: E402
from src.experiments.tpds import e12_analysis as e12  # noqa: E402
from src.experiments.tpds.analysis import chunk_columns, latency_matrix, load_e2  # noqa: E402
from src.experiments.tpds.common import RESULTS_DIR, write_json  # noqa: E402

PRIMARY = ("pro6000", "vgpu3090", "h800")
DEFAULT_TAGS = "pro6000,vgpu3090,h800,vgpu3090_r3spin,h800_r3spin,pro6000_r4spin"
PUBLISHED_MEAN_PCT = {"pro6000": 0.9, "vgpu3090": 3.2, "h800": 3.5}
SEED = 0

_I = {k: i for i, k in enumerate(cm.PARAMS)}
TABLE_FORMS = ["full", *cm.ABLATIONS]                 # the forms of the ablation table
ALL_FORMS = [*TABLE_FORMS, "work_only"]
CANDIDATE_SETS = {"ablation12": TABLE_FORMS, "ablation12+work_only": ALL_FORMS}


def device_of(tag: str) -> str:
    return tag.split("_")[0]


# --------------------------------------------------------------------------- data
class Ctx:
    """Calibration data, captured calls, compile facts and stored fit of one result tag."""

    def __init__(self, tag: str, in_dir: Path):
        self.tag, self.in_dir, self.device = tag, in_dir, device_of(tag)
        self.hw = cm.hardware(tag, in_dir)
        self.stats = cm.load_kernel_stats(in_dir, self.hw["arch"])
        pts = cm.load_points(tag, in_dir)
        self.f = cm.features(pts, self.stats, self.hw)
        self.points = pts.loc[self.f.index]
        self.y = self.points["lat"].to_numpy(float)
        self.df = load_e2([in_dir / f"e2_mamba1_{tag}.csv"])
        self.chunks = chunk_columns(self.df)
        self.lat = latency_matrix(self.df, self.chunks)
        self.stored = json.loads((in_dir / f"cost_model_{tag}.json").read_text())
        self.theta = np.array([self.stored["params"][k] for k in cm.PARAMS])
        self.keys = cmb.calibration_keys(tag, in_dir, self.chunks)
        self.calls = Target(pd.DataFrame({"L": self.df["L"].to_numpy(int), "B": self.df["batch"].to_numpy(int),
                                          "d": self.df["width"].to_numpy(int)}), self.chunks, self.hw, self.stats)


class Target:
    """Shapes (L, B, d) to choose a chunk for, with the hardware constants and compile facts of
    the GPU they run on; candidates at the default launch configuration of every chunk."""

    def __init__(self, shapes: pd.DataFrame, chunks: list[int], hw: dict, stats: dict):
        self.shapes = shapes.reset_index(drop=True)
        self.chunks, self.hw, self.stats = list(chunks), hw, stats
        rows = []
        for c in self.chunks:
            bd, nw = cm.DEFAULT_LAUNCH[c]
            rows.append(pd.DataFrame({"L": self.shapes["L"].to_numpy(), "B": self.shapes["B"].to_numpy(),
                                      "d": self.shapes["d"].to_numpy(), "C": c, "BD": bd, "nw": nw, "ns": 1}))
        self.cand = pd.concat(rows, ignore_index=True)          # same order as cost_model.e2_candidates
        self.fc = cm.features(self.cand, stats, hw)
        self._learn = None

    def matrix(self, values: np.ndarray) -> np.ndarray:
        out = np.full(len(self.cand), np.inf)
        out[self.fc.index.to_numpy()] = values
        return out.reshape(len(self.chunks), len(self.shapes)).T

    def learn_frame(self) -> pd.DataFrame:
        if self._learn is None:
            self._learn = learn_features(self.cand, self.fc, self.stats)
        return self._learn


# --------------------------------------------------------------------------- features of the learned regressors
BASIC_COLS = ["log2_L", "log2_B", "log2_d", "log2_C", "log2_BD", "nw", "ns", "aligned", "log2_tiles", "log2_programs",
              "regs", "spill_store_bytes", "spill_load_bytes", "shared_bytes", "spt_d", "spt_c", "tpw_d", "tpw_c",
              "wpc_d", "wpc_c"]
STRUCT_EXTRA = ["e", "e_d", "lanes_c", "warps_c", "loads", "spill_words", "l1_miss", "k", "waves", "m", "l2_miss",
                "log_t_mem"]


def learn_features(rows: pd.DataFrame, f: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Shape, launch and compile-fact columns for rows that have compile facts (index of ``f``)."""
    r = rows.loc[f.index]
    L, B, d, C = (r[k].to_numpy(float) for k in ("L", "B", "d", "C"))
    BD = r["BD"].to_numpy(float)
    out = pd.DataFrame(index=f.index)
    out["log2_L"], out["log2_B"], out["log2_d"], out["log2_C"] = np.log2(L), np.log2(B), np.log2(d), np.log2(C)
    out["log2_BD"], out["nw"], out["ns"] = np.log2(BD), r["nw"].to_numpy(float), r["ns"].to_numpy(float)
    out["aligned"] = (r["L"].to_numpy(int) % 16 == 0).astype(float)
    out["log2_tiles"] = np.log2(np.ceil(L / C))
    out["log2_programs"] = np.log2(np.ceil(d / BD) * B)
    facts = []
    for Lr, Cr, BDr, nwr, nsr in r[["L", "C", "BD", "nw", "ns"]].itertuples(index=False):
        s = cm._stats_row(stats, int(Lr) % 16 == 0, int(Cr), int(BDr), int(nwr), int(nsr))
        lay = s["layout"]
        facts.append([s["regs"], s.get("spill_store_bytes", 0), s.get("spill_load_bytes", 0), s.get("shared_bytes", 0),
                      *lay["size_per_thread"], *lay["threads_per_warp"], *lay["warps_per_cta"]])
    out[["regs", "spill_store_bytes", "spill_load_bytes", "shared_bytes", "spt_d", "spt_c", "tpw_d", "tpw_c",
         "wpc_d", "wpc_c"]] = np.array(facts, float)
    for k in STRUCT_EXTRA[:-1]:
        out[k] = f[k].to_numpy(float)
    out["log_t_mem"] = np.log(f["t_mem_ms"].to_numpy(float))
    return out


# --------------------------------------------------------------------------- selectors (score = lower is better)
class Eq5:
    def __init__(self, theta):
        self.theta = np.asarray(theta, float)

    def scores(self, t: Target) -> np.ndarray:
        return t.matrix(cm.predict(self.theta, t.fc))


class WorkTerm:
    def __init__(self, th):
        self.th = np.asarray(th, float)

    def scores(self, t: Target) -> np.ndarray:
        L, B = t.shapes["L"].to_numpy(float), t.shapes["B"].to_numpy(float)
        return np.stack([cm.pipe_predict(self.th, L, B, float(c)) for c in t.chunks], 1)


class NearestKey:
    def __init__(self, keys: list[dict]):
        self.keys = keys

    def scores(self, t: Target) -> np.ndarray:
        idx = [cmb.nearest_key(int(L), int(B), int(d), self.keys) for L, B, d in t.shapes[["L", "B", "d"]].itertuples(index=False)]
        return np.stack([self.keys[i]["mean_ms"] for i in idx])


class LogLinearLB:
    """Bilinear interpolation of log latency per chunk in (log2 L, log2 B) over the calibration grid."""

    def __init__(self, keys: list[dict]):
        self.xs = sorted({math.log2(k["L"]) for k in keys})
        self.ys = sorted({math.log2(k["B"]) for k in keys})
        self.Z = np.full((len(self.xs), len(self.ys), len(keys[0]["mean_ms"])), np.nan)
        for k in keys:
            self.Z[self.xs.index(math.log2(k["L"])), self.ys.index(math.log2(k["B"]))] = np.log(k["mean_ms"])
        if np.isnan(self.Z).any():
            raise ValueError("calibration keys do not form a full (L, B) grid")

    @staticmethod
    def _bracket(grid, v):
        g = np.asarray(grid)
        v = np.clip(v, g[0], g[-1])
        j = np.clip(np.searchsorted(g, v, side="right") - 1, 0, len(g) - 2)
        return j, (v - g[j]) / (g[j + 1] - g[j])

    def scores(self, t: Target) -> np.ndarray:
        jx, wx = self._bracket(self.xs, np.log2(t.shapes["L"].to_numpy(float)))
        jy, wy = self._bracket(self.ys, np.log2(t.shapes["B"].to_numpy(float)))
        Z, wx, wy = self.Z, wx[:, None], wy[:, None]
        return ((1 - wx) * (1 - wy) * Z[jx, jy] + wx * (1 - wy) * Z[jx + 1, jy]
                + (1 - wx) * wy * Z[jx, jy + 1] + wx * wy * Z[jx + 1, jy + 1])


class CalibTree(cmb.ScoredLeafTree):
    """E3's regret-labelled tree, trained on the calibration shapes (min_samples_leaf = 1)."""

    def __init__(self, depth: int, cols: tuple[str, ...]):
        super().__init__(depth)
        self.tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=1, random_state=SEED)
        self.cols = cols

    def _x(self, L, B):
        m = {"log2_L": np.log2(np.asarray(L, float)), "log2_B": np.log2(np.asarray(B, float))}
        return np.stack([m[c] for c in self.cols], 1)

    def fit_keys(self, keys: list[dict]):
        lat = np.stack([k["mean_ms"] for k in keys])
        X = self._x([k["L"] for k in keys], [k["B"] for k in keys])
        return self.fit(X, lat.argmin(1), lat / lat.min(1, keepdims=True))

    def scores(self, t: Target) -> np.ndarray:
        return self.predict_scores(self._x(t.shapes["L"], t.shapes["B"]))


class Learned:
    def __init__(self, model, cols: list[str]):
        self.model, self.cols = model, cols

    def scores(self, t: Target) -> np.ndarray:
        return t.matrix(self.model.predict(t.learn_frame()[self.cols].to_numpy(float)))


# --------------------------------------------------------------------------- TFLA runtime expression
TFLA_MAP = {"N_head": "d (channels)", "d_hv": 1, "d_qk": cm.D_STATE, "bytes_qkv": 2, "bytes_if": 2, "bytes_cmn": 4,
            "f_causal": 0.5}
TFLA_MODES = {"tfla_runtime": "sum", "tfla_runtime_overlap": "max", "tfla_runtime_T_over_C": "sum_literal",
              "tfla_runtime_launch_floor": "sum_floor"}
# accelerator intensities (FLOP/byte) for the unfitted variant; TFLA's Table of accelerators lists
# 133 (V100) to 295 (H100); dense BF16 tensor peak / memory bandwidth is about 76 for the RTX 3090
# (71 TFLOP/s, 936 GB/s) and about 378 for the H800 PCIe (756 TFLOP/s, 2.0 TB/s)
TFLA_FIXED_INTENSITY = (64, 128, 256, 512)


def tfla_counts(C):
    """TFLA per-head, per-chunk byte and FLOP counts of the chunkwise-parallel mLSTMsig (all F_OP = 1)."""
    m = TFLA_MAP
    dh, dq, C = m["d_hv"], m["d_qk"], np.asarray(C, float)
    nbytes = 4 * C * m["bytes_if"] + 3 * C * (dh + dq) * m["bytes_qkv"] + 2 * dh * dq * m["bytes_cmn"]
    flops = (C ** 2 * m["f_causal"] * (2 * dq + 2 * dh + 6) + C ** 2 + C * (4 * dh * dq + 2 * dq + dh + 11)
             + 2 * dh * dq + 5)
    return nbytes, flops


def tfla_lopt(i_acc: float) -> float:
    """TFLA's closed-form runtime-optimal chunk size (App. G.4) for the instantiation above."""
    m = TFLA_MAP
    dh, p = m["d_hv"], m["d_qk"] / m["d_hv"]
    return float(math.sqrt((2 * dh ** 2 * p + 5 + 2 * i_acc * dh ** 2 * p * m["bytes_cmn"])
                           / (2 * m["f_causal"] * (dh * (1 + p) + 3) + 1)))


class TFLA:
    """th = (log alpha [FLOP/ms], log beta [byte/ms][, t0 ms]); ``limit`` = "no_flops" (alpha -> infinity)
    or "no_bytes" (beta -> infinity) drops that term exactly."""

    def __init__(self, th, mode: str, limit: str | None = None):
        self.th, self.mode, self.limit = np.asarray(th, float), mode, limit

    def parts(self, L, B, d, C):
        L, B, d, C = (np.asarray(v, float) for v in (L, B, d, C))
        nbytes, flops = tfla_counts(C)
        n = L / C if self.mode == "sum_literal" else np.ceil(L / C)
        tb = np.zeros_like(nbytes) if self.limit == "no_bytes" else nbytes / math.exp(self.th[1])
        tf = np.zeros_like(flops) if self.limit == "no_flops" else flops / math.exp(self.th[0])
        return B * d * n * tb, B * d * n * tf

    def predict(self, L, B, d, C):
        tb, tf = self.parts(L, B, d, C)
        t = np.maximum(tb, tf) if self.mode == "max" else tb + tf
        return t + self.th[2] if self.mode == "sum_floor" else t

    def scores(self, t: Target) -> np.ndarray:
        L, B, d = (t.shapes[k].to_numpy(float) for k in ("L", "B", "d"))
        return np.stack([self.predict(L, B, d, np.full(len(L), float(c))) for c in t.chunks], 1)


def fit_tfla(pts: pd.DataFrame, mode: str, chunks: list[int]) -> tuple[TFLA, dict]:
    """Robust least squares on log latency. Besides the two-constant fit, the two boundary models
    (alpha -> infinity: memory term only; beta -> infinity: FLOP term only) are fitted; when a boundary
    model fits at least as well, the optimum lies on that boundary and the boundary model is used, so that
    the selector does not depend on where an optimizer stops along a flat direction."""
    L, B, d, C = (pts[k].to_numpy(float) for k in ("L", "B", "d", "C"))
    y = np.log(pts["lat"].to_numpy(float))
    nfloor = mode == "sum_floor"
    lo = [math.log(1e3), math.log(1e3)] + ([0.0] if nfloor else [])
    hi = [math.log(1e30), math.log(1e30)] + ([1.0] if nfloor else [])
    fits = {}
    for limit in (None, "no_flops", "no_bytes"):
        best = None
        for la in (math.log(1e8), math.log(1e10), math.log(1e12), math.log(1e16)):   # FLOP/ms
            for lb in (math.log(1e7), math.log(1e9), math.log(1e11)):                # byte/ms
                x0 = [la, lb] + ([float(np.exp(y.min())) / 2] if nfloor else [])

                def res(th, limit=limit):
                    return np.log(TFLA(th, mode, limit).predict(L, B, d, C)) - y
                sol = least_squares(res, x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.1, x_scale="jac",
                                    max_nfev=20000)
                if best is None or sol.cost < best.cost - 1e-12:
                    best = sol
        fits[limit] = best
    c2 = fits[None].cost
    chosen = None
    for limit in ("no_flops", "no_bytes"):
        if fits[limit].cost <= c2 * (1 + 1e-6) + 1e-12 and (chosen is None or fits[limit].cost < fits[chosen].cost):
            chosen = limit
    best = fits[chosen]
    sel = TFLA(best.x, mode, chosen)
    tb, tf = sel.parts(L, B, d, C)
    Cg = np.array([float(c) for c in chunks])
    tbg, tfg = sel.parts(np.repeat(L, len(Cg)), np.repeat(B, len(Cg)), np.repeat(d, len(Cg)), np.tile(Cg, len(L)))
    alpha = None if chosen == "no_flops" else math.exp(best.x[0])
    beta = None if chosen == "no_bytes" else math.exp(best.x[1])
    i_acc = alpha / beta if alpha is not None and beta is not None else (float("inf") if alpha is None else 0.0)
    info = {"mode": mode, "n_fit_points": int(len(y)), "boundary": chosen,
            "boundary_note": {None: "interior optimum (both terms)",
                              "no_flops": "least-squares optimum at alpha -> infinity: the FLOP term (quadratic in C) "
                                          "is rejected by the data; memory term only",
                              "no_bytes": "least-squares optimum at beta -> infinity: FLOP term only"}[chosen],
            "fit_cost": {str(k): float(v.cost) for k, v in fits.items()},
            "alpha_tflops": None if alpha is None else alpha * 1e3 / 1e12,
            "beta_gbs": None if beta is None else beta * 1e3 / 1e9,
            "accelerator_intensity_flop_per_byte": i_acc if math.isfinite(i_acc) else "inf",
            "closed_form_runtime_optimal_chunk": (tfla_lopt(i_acc) if mode in ("sum", "sum_literal", "sum_floor")
                                                  and math.isfinite(i_acc) else None),
            "flop_share_of_predicted_time_max_over_calibration_and_chunks": float(np.max(tfg / (tbg + tfg))),
            "flop_share_of_predicted_time_median_fit_points": float(np.median(tf / (tb + tf))),
            "calibration_fit": cm.fit_quality(sel.predict(L, B, d, C), np.exp(y))}
    if nfloor:
        info["t0_ms"] = float(best.x[2])
    return sel, info


# --------------------------------------------------------------------------- learned regressors
GBR_GRID = [{"max_iter": it, "max_leaf_nodes": lf} for it in (300, 1000) for lf in (7, 31)]
MLP_GRID = [{"alpha": a} for a in (1e-4, 1e-2, 1.0)]


def _make_learner(kind: str, hp: dict):
    if kind == "gbr":
        return HistGradientBoostingRegressor(learning_rate=0.05, early_stopping=False, random_state=SEED, **hp)
    # Adam on mini-batches of 128 for a fixed 300 epochs (no early stopping: deterministic run length)
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), solver="adam", batch_size=128,
                                                        learning_rate_init=1e-3, max_iter=300, tol=0.0,
                                                        n_iter_no_change=10 ** 9, random_state=SEED, **hp))


def train_learner(kind: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, featset: list[str]):
    cols = [c for c in featset if X[c].nunique() > 1]
    Xa = X[cols].to_numpy(float)
    grid = GBR_GRID if kind == "gbr" else MLP_GRID
    cv = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for hp in grid:
            err = np.empty(len(y))
            for g in np.unique(groups):
                te = groups == g
                m = _make_learner(kind, hp).fit(Xa[~te], y[~te])
                err[te] = m.predict(Xa[te]) - y[te]
            cv.append({"hyperparameters": hp, "cv_mse_log": float(np.mean(err ** 2))})
        best = min(range(len(grid)), key=lambda i: cv[i]["cv_mse_log"])
        model = _make_learner(kind, grid[best]).fit(Xa, y)
    pred = model.predict(Xa)
    info = {"columns": cols, "dropped_constant_columns": [c for c in featset if c not in cols],
            "hyperparameter_cv": cv, "chosen": grid[best], "n_train": int(len(y)),
            "calibration_fit": cm.fit_quality(np.exp(pred), np.exp(y)),
            "leave_one_shape_out_rmse_log": float(math.sqrt(cv[best]["cv_mse_log"]))}
    return Learned(model, cols), info


# --------------------------------------------------------------------------- metrics
def call_regret(lat: np.ndarray, choice: np.ndarray) -> np.ndarray:
    return lat[np.arange(len(lat)), choice] / lat.min(1) - 1


def score_calls(ctx_df: pd.DataFrame, chunks: list[int], choice: np.ndarray, S: np.ndarray | None = None) -> dict:
    """``cost_model_baselines.evaluate`` (hence ``cost_model.e2_eval``) plus p99, compacted."""
    if S is not None and not np.array_equal(np.argmin(S, 1), choice):
        S = None
    st = cmb.evaluate(ctx_df, chunks, choice, S)
    reg = call_regret(latency_matrix(ctx_df, chunks), choice)
    out = {k: st[k] for k in ("n", "mean_regret_pct", "p95_regret_pct", "max_regret_pct", "tie_aware_accuracy",
                              "accuracy", "ratio_of_means_headroom_pct")}
    out["p99_regret_pct"] = float(100 * np.percentile(reg, 99))
    out["chosen_distribution"] = st["chosen_distribution"]
    out["by_batch"] = {int(b): {"mean_regret_pct": v["mean_regret_pct"], "p95_regret_pct": v["p95_regret_pct"],
                                "tie_aware_accuracy": v["tie_aware_accuracy"]} for b, v in st["by_batch"].items()}
    if "kendall_tau_per_call" in st:
        out["kendall_tau_per_call_mean"] = st["kendall_tau_per_call"]["mean"]
        out["kendall_tau_per_shape_mean"] = st["kendall_tau_per_shape"]["mean"]
    return out


def tied_sets(df: pd.DataFrame) -> list[set]:
    return [set(int(c) for c in str(t).split(";") if c) for t in df["tied_chunks"].fillna("")]


def topk_picks(df: pd.DataFrame, chunks: list[int], P: np.ndarray, k: int) -> np.ndarray:
    """Same rule as ``cost_model.topk_refinement``: per (L, batch) profile the k best-ranked chunks and
    keep the one with the lowest mean latency over the shape's calls."""
    lat = latency_matrix(df, chunks)
    rank = np.argsort(P, 1)
    pick = np.empty(len(df), dtype=int)
    for _, g in df.groupby(["L", "batch"]):
        i = g.index.to_numpy()
        cand = rank[i[0], :k]
        pick[i] = cand[np.argmin(lat[i][:, cand].mean(0))]
    return pick


def topk_table(df: pd.DataFrame, chunks: list[int], P: np.ndarray, ks=(1, 2, 3)) -> dict:
    lat = latency_matrix(df, chunks)
    tied = tied_sets(df)
    out = {}
    for k in (*ks, len(chunks)):
        pick = topk_picks(df, chunks, P, k)
        reg = call_regret(lat, pick)
        out[f"top{k}" if k < len(chunks) else "all_chunks_profiled"] = {
            "profiled_chunks_per_shape": int(k), "mean_regret_pct": float(100 * reg.mean()),
            "p95_regret_pct": float(100 * np.percentile(reg, 95)), "p99_regret_pct": float(100 * np.percentile(reg, 99)),
            "max_regret_pct": float(100 * reg.max()),
            "tie_aware_accuracy": float(np.mean([chunks[c] in t for c, t in zip(pick, tied)]))}
    return out


def brief(st: dict) -> dict:
    keep = ("mean_regret_pct", "p95_regret_pct", "p99_regret_pct", "max_regret_pct", "tie_aware_accuracy",
            "tied_share", "n")
    return {k: st[k] for k in keep if k in st}


# --------------------------------------------------------------------------- M5a: candidate forms
def form_specs(theta_full: np.ndarray) -> dict:
    """name -> (fixed parameters, start vector); same starts as ``cost_model.ablation`` and
    ``cost_model_baselines.model_selectors``."""
    specs = {"full": (dict(cm.DEFAULT_FIXED), theta_full.copy())}
    for name, change in cm.ABLATIONS.items():
        x0 = theta_full.copy()
        for i, v in change.items():
            if v is None:
                x0[i] = cm.X0[i]
        specs[name] = (cm._fixed_for(change), x0)
    x0 = cm.X0.copy()
    for i, v in cmb.WORK_ONLY_FIXED.items():
        x0[i] = v
    x0[[_I["t0"], _I["tau_m"], _I["tau_e"]]] = theta_full[[_I["t0"], _I["tau_m"], _I["tau_e"]]]
    specs["work_only"] = (dict(cmb.WORK_ONLY_FIXED), x0)
    return specs


def loso(ctx: Ctx, fixed: dict, x0: np.ndarray) -> dict:
    """Leave one calibration shape out (all sources of the shape held out)."""
    pts, rows = ctx.points, []
    keys = sorted(pts.loc[pts["src"] == "e4_static", "key"].unique())
    for key in keys:
        k0 = pts[pts["key"] == key].iloc[0]
        hold = ((pts["L"] == k0["L"]) & (pts["B"] == k0["B"]) & (pts["d"] == k0["d"])).to_numpy()
        tr = pts.index[~hold]
        th, _ = cm.fit(ctx.f.loc[tr], pts.loc[tr, "lat"].to_numpy(), x0=x0, fixed=fixed)
        st = pts[(pts["key"] == key) & (pts["src"] == "e4_static")]
        pst = cm.predict(th, ctx.f.loc[st.index])
        meas = dict(zip(st["C"].astype(int), st["lat"]))
        c_pred = int(st["C"].to_numpy()[int(np.argmin(pst))])
        c_best = min(meas, key=meas.get)
        ph = cm.predict(th, ctx.f.loc[pts.index[hold]])
        rmse = float(np.sqrt(np.mean((np.log(ph) - np.log(pts.loc[pts.index[hold], "lat"].to_numpy())) ** 2)))
        rows.append({"key": key, "n_heldout_points": int(hold.sum()), "chunk_pred": c_pred, "chunk_best": int(c_best),
                     "chunk_regret_pct": float(100 * (meas[c_pred] / meas[c_best] - 1)), "heldout_rmse_log": rmse})
    r = np.array([x["chunk_regret_pct"] for x in rows])
    return {"mean_chunk_regret_pct": float(r.mean()), "max_chunk_regret_pct": float(r.max()),
            "se_chunk_regret_pct": float(r.std(ddof=1) / math.sqrt(len(r))), "n_shapes": len(rows),
            "heldout_rmse_log_mean": float(np.mean([x["heldout_rmse_log"] for x in rows])), "per_shape": rows}


def run_forms(ctx: Ctx, do_loso: bool) -> tuple[dict, dict]:
    specs = form_specs(ctx.theta)
    lat = ctx.lat
    out, per_call = {}, {}
    for name in ALL_FORMS:
        fixed, x0 = specs[name]
        if name == "full":
            th = ctx.theta.copy()
        else:
            th, _ = cm.fit(ctx.f, ctx.y, x0=x0, fixed=fixed)
        P = Eq5(th).scores(ctx.calls)
        choice = P.argmin(1)
        per_call[name] = call_regret(lat, choice)
        rec = {"n_free_params": len(cm.PARAMS) - len(fixed), "in_ablation_table": name in TABLE_FORMS,
               "params": {k: float(v) for k, v in zip(cm.PARAMS, th)},
               "calibration_fit": cm.fit_quality(cm.predict(th, ctx.f), ctx.y),
               "captured_calls": brief(score_calls(ctx.df, ctx.chunks, choice, P))}
        if do_loso:
            rec["leave_one_shape_out"] = loso(ctx, fixed, th)
        out[name] = rec
    return out, per_call


def select_strict(cands: list[str], value: dict, nfree: dict, second: dict) -> str:
    """Lowest criterion; exact ties -> fewer free parameters -> lower ``second`` (held-out RMSE)."""
    return min(cands, key=lambda n: (value[n], nfree[n], second[n], n))


def select_one_se(cands: list[str], value: dict, se: dict, nfree: dict, second: dict) -> str:
    """Fewest free parameters within one standard error of the best criterion value; ties -> lower
    criterion -> lower ``second``."""
    best = select_strict(cands, value, nfree, second)
    ok = [n for n in cands if value[n] <= value[best] + se[best]]
    return min(ok, key=lambda n: (nfree[n], value[n], second[n], n))


def tied_alternatives(sel: str, cands: list[str], value: dict, nfree: dict) -> list[str]:
    """Forms the rule could not tell apart from the selected one (same criterion, same parameter count)."""
    return [n for n in cands if n != sel and value[n] == value[sel] and nfree[n] == nfree[sel]]


def calibration_selection(forms: dict) -> dict:
    if not all("leave_one_shape_out" in r for r in forms.values()):
        return {}
    nfree = {n: r["n_free_params"] for n, r in forms.items()}
    val = {n: r["leave_one_shape_out"]["mean_chunk_regret_pct"] for n, r in forms.items()}
    se = {n: r["leave_one_shape_out"]["se_chunk_regret_pct"] for n, r in forms.items()}
    rmse = {n: r["leave_one_shape_out"]["heldout_rmse_log_mean"] for n, r in forms.items()}
    out = {}
    for set_name, cands in CANDIDATE_SETS.items():
        sel = {"strict": select_strict(cands, val, nfree, rmse), "one_se": select_one_se(cands, val, se, nfree, rmse),
               "rmse": min(cands, key=lambda n: (rmse[n], nfree[n], n))}
        out[set_name] = {}
        for rule, n in sel.items():
            crit = rmse if rule == "rmse" else val
            alts = tied_alternatives(n, cands, crit, nfree)
            out[set_name][rule] = {"selected_form": n, "n_free_params": nfree[n], "loso_mean_chunk_regret_pct": val[n],
                                   "loso_heldout_rmse_log": rmse[n], "captured_calls": forms[n]["captured_calls"],
                                   "exact_tie_alternatives": {a: brief(forms[a]["captured_calls"]) for a in alts}}
    return out


# --------------------------------------------------------------------------- M5b: baselines on one tag
def fit_selectors(ctx: Ctx, do_learned: bool) -> tuple[dict, dict, dict]:
    """Selectors fitted on the tag's calibration data; also the existing baseline rows and checks."""
    rows_m, checks = cmb.model_selectors(ctx.tag, ctx.in_dir, ctx.df, ctx.chunks)
    nk_rows, nk_var, nk_checks = cmb.nearest_key_selectors(ctx.tag, ctx.in_dir, ctx.df, ctx.chunks)
    checks.update(nk_checks)
    sels, info = {}, {}
    sels["cost_model"] = Eq5(ctx.theta)
    sels["work_only"] = Eq5([rows_m["work_only"]["params"][k] for k in cm.PARAMS])
    info["work_only"] = {"params": rows_m["work_only"]["params"], "calibration_fit": rows_m["work_only"]["calibration_fit"]}
    sels["work_term_model"] = WorkTerm([rows_m["work_term_model"]["params"][k] for k in cm.PIPE_PARAMS])
    info["work_term_model"] = {"params": rows_m["work_term_model"]["params"],
                               "calibration_fit": rows_m["work_term_model"]["calibration_fit"]}
    sels["nearest_calibrated_key"] = NearestKey(ctx.keys)
    sels["loglinear_interp_logL_logB"] = LogLinearLB(ctx.keys)
    sels["length_tree_calibration"] = CalibTree(3, ("log2_L",)).fit_keys(ctx.keys)
    sels["shape_tree_calibration"] = CalibTree(4, ("log2_L", "log2_B")).fit_keys(ctx.keys)
    if do_learned:
        X = learn_features(ctx.points, ctx.f, ctx.stats)
        ylog = np.log(ctx.y)
        groups = (ctx.points["L"].astype(str) + "|" + ctx.points["B"].astype(str) + "|" + ctx.points["d"].astype(str)).to_numpy()
        for kind in ("gbr", "mlp"):
            for fs_name, fs in (("basic", BASIC_COLS), ("structural", BASIC_COLS + STRUCT_EXTRA)):
                t = time.time()
                sels[f"{kind}_{fs_name}"], info[f"{kind}_{fs_name}"] = train_learner(kind, X, ylog, groups, fs)
                info[f"{kind}_{fs_name}"]["train_seconds"] = round(time.time() - t, 1)
    default = ctx.points[ctx.points["src"].isin(["e4_static", "e1"])]
    for name, mode in TFLA_MODES.items():
        sels[name], info[name] = fit_tfla(default, mode, ctx.chunks)
    for i_acc in TFLA_FIXED_INTENSITY:   # no fit: the choice depends only on alpha / beta
        name = f"tfla_runtime_intensity_{i_acc}"
        sels[name] = TFLA([math.log(i_acc), 0.0], "sum")
        info[name] = {"mode": "sum", "accelerator_intensity_flop_per_byte": i_acc,
                      "closed_form_runtime_optimal_chunk": tfla_lopt(i_acc), "fitted": False}
    info["_existing_rows"] = {"model_selectors": {k: brief(v) for k, v in rows_m.items()},
                              "nearest_key": {k: brief(v) for k, v in nk_rows.items()},
                              "nearest_key_variants": {k: brief(v) for k, v in nk_var.items()}}
    extra_rows = {"interp_log_length_same_batch": nk_var["interpolated_log_length"]}
    return sels, info, {"checks": checks, "extra_rows": extra_rows}


def evaluate_on_calls(sels: dict, target_ctx: Ctx, target: Target | None = None) -> tuple[dict, dict]:
    target = target or target_ctx.calls
    rows, mats = {}, {}
    for name, s in sels.items():
        S = s.scores(target)
        mats[name] = S
        rows[name] = score_calls(target_ctx.df, target_ctx.chunks, S.argmin(1), S)
    return rows, mats


# --------------------------------------------------------------------------- E12 shard widths
def e12_stats(tab: pd.DataFrame, chunks: list[int], choice: dict) -> dict:
    st = e12.regret(tab, chunks, choice)
    if not st.get("n"):
        return st
    reg = [r.lat[chunks.index(choice[(r.d, r.L, r.batch)])] / r.lat.min() - 1 for r in tab.itertuples()
           if (r.d, r.L, r.batch) in choice]
    st["p99_regret_pct"] = float(100 * np.percentile(reg, 99))
    return st


def run_e12(ctx: Ctx, sels: dict) -> dict | None:
    path = ctx.in_dir / f"e12_tp_widths_{ctx.tag}.jsonl"
    if not path.exists():
        return None
    raw = e12.load(path)
    chunks = list(e12.MAMBA1_CHUNKS)
    assert chunks == ctx.chunks
    tab = e12.shape_table(raw[raw["family"] == "triton"], chunks).reset_index(drop=True)
    shapes = pd.DataFrame({"L": tab["L"].to_numpy(int), "B": tab["batch"].to_numpy(int), "d": tab["d"].to_numpy(int)})
    target = Target(shapes, chunks, ctx.hw, ctx.stats)
    keys = [(int(r.d), int(r.L), int(r.batch)) for r in tab.itertuples()]
    choices, mats = {}, {}
    for name, s in sels.items():
        S = s.scores(target)
        mats[name] = S
        choices[name] = {k: chunks[int(j)] for k, j in zip(keys, S.argmin(1))}
    # reference rows of e12_analysis (they use E12 timings)
    choices["width2048_table"] = e12.table_choice(tab, lambda d: 2048)
    regs = {c: np.mean([r.lat[chunks.index(c)] / r.lat.min() - 1 for r in tab.itertuples()]) for c in chunks}
    best_static = min(regs, key=regs.get)
    choices["best_static"] = {k: best_static for k in keys}
    widths = set(tab["d"])
    unsh = {}
    for model, full in e12.MODEL_WIDTHS.items():
        for t in e12.TP_DEGREES[1:]:
            d = full // t
            if full % t == 0 and d in widths and full in widths:
                unsh.update(e12.table_choice(tab[tab["d"].isin([d, full])], lambda _d, full=full, d=d: full if _d == d else None))
    # top-k hybrids ranked by the cost model (and by the other rankers): measured best among the k best-ranked
    lat = np.stack(tab["lat"].to_numpy())
    topk = {}
    for name in sels:
        rank = np.argsort(mats[name], 1)
        for k in (2, 3):
            pick = np.array([rank[i, :k][np.argmin(lat[i, rank[i, :k]])] for i in range(len(tab))])
            topk[f"{name}__top{k}"] = {kk: chunks[int(j)] for kk, j in zip(keys, pick)}
    shards = {d for d, lst in e12.shard_map(widths).items() if any(t > 1 for _, t in lst)}
    pooled_tab = tab[tab["d"].isin(shards)]
    res = {"n_shapes": int(len(tab)), "n_shard_shapes": int(len(pooled_tab)), "shard_widths": sorted(int(w) for w in shards),
           "best_static_chunk": int(best_static),
           "pooled_shards": {n: e12_stats(pooled_tab, chunks, ch) for n, ch in choices.items()},
           "all_shapes": {n: e12_stats(tab, chunks, ch) for n, ch in choices.items()},
           "pooled_shards_topk": {n: e12_stats(pooled_tab, chunks, ch) for n, ch in topk.items()},
           "all_shapes_topk": {n: e12_stats(tab, chunks, ch) for n, ch in topk.items()}}
    res["pooled_shards"]["unsharded_table"] = e12_stats(pooled_tab, chunks, unsh)
    by_model = {}
    for model, full in e12.MODEL_WIDTHS.items():
        for t in e12.TP_DEGREES:
            d = full // t
            if full % t or d not in widths:
                continue
            sub = tab[tab["d"] == d]
            row = {"d": int(d)}
            for n, ch in choices.items():
                row[n] = brief(e12_stats(sub, chunks, ch))
            for n in ("cost_model__top2", "cost_model__top3"):
                row[n] = brief(e12_stats(sub, chunks, topk[n]))
            if t > 1 and full in widths:
                row["unsharded_table"] = brief(e12_stats(sub, chunks, e12.table_choice(
                    tab[tab["d"].isin([d, full])], lambda _d, full=full, d=d: full if _d == d else None)))
            by_model[f"{model}_t{t}"] = row
    res["by_model_and_tp"] = by_model
    # reproduction of the stored E12 analysis (same selectors, same scoring)
    stored_p = ctx.in_dir / f"e12_analysis_{ctx.tag}.json"
    if stored_p.exists():
        sp = json.loads(stored_p.read_text())["families"]["triton"]["pooled_shards"]
        res["check_vs_stored_e12_analysis"] = {
            n: [sp[n]["mean_regret_pct"], res["pooled_shards"][m]["mean_regret_pct"]]
            for n, m in (("cost_model", "cost_model"), ("nearest_key", "nearest_calibrated_key"),
                         ("width2048_table", "width2048_table"), ("best_static", "best_static"),
                         ("unsharded_table", "unsharded_table")) if n in sp}
    return res


# --------------------------------------------------------------------------- per-tag driver
def analyze_tag(tag: str, in_dir: Path, do_loso: bool = True, do_learned: bool = True) -> dict:
    t0 = time.time()
    ctx = Ctx(tag, in_dir)
    forms, per_call = run_forms(ctx, do_loso)
    sels, sel_info, extra = fit_selectors(ctx, do_learned)
    rows, mats = evaluate_on_calls(sels, ctx)
    for name, r in extra["extra_rows"].items():
        rows[name] = brief(r) | {"note": "existing variant of cost_model_baselines (captured calls only)"}
    checks = {}
    sv = ctx.stored["V2_e2_calls"]
    checks["cost_model_vs_stored"] = {k: [sv[k], rows["cost_model"][k]] for k in ("mean_regret_pct", "p95_regret_pct", "tie_aware_accuracy")}
    checks["cost_model_choice_equals_stored"] = bool([int(ctx.chunks[i]) for i in mats["cost_model"].argmin(1)]
                                                     == ctx.stored["per_call_choice"]["mamba1"]["chunk"])
    checks["full_form_vs_stored"] = [sv["mean_regret_pct"], forms["full"]["captured_calls"]["mean_regret_pct"]]
    if "ablation" in ctx.stored:
        checks["ablation_forms_vs_stored_mean_regret_pct"] = {
            n: [ctx.stored["ablation"][n]["e2_mean_regret_pct"], forms[n]["captured_calls"]["mean_regret_pct"]]
            for n in cm.ABLATIONS}
    old_p = in_dir / f"cost_model_baselines_{tag}.json"
    if old_p.exists():
        old = json.loads(old_p.read_text())["summary"]
        checks["baselines_vs_stored_cost_model_baselines"] = {
            n: [old[n]["mean_regret_pct"], rows[n]["mean_regret_pct"]]
            for n in ("cost_model", "work_only", "work_term_model", "nearest_calibrated_key") if n in old}
        checks["work_only_form_vs_stored"] = [old["work_only"]["mean_regret_pct"], forms["work_only"]["captured_calls"]["mean_regret_pct"]]
    checks.update({f"cmb_{k}": v for k, v in extra["checks"].items()
                   if k in ("cost_model_refit_max_rel_param_diff", "cost_model_choice_equals_stored",
                            "nearest_key_batch_first_equals_length_first")})
    # top-k (M6d)
    topk = {"cost_model": topk_table(ctx.df, ctx.chunks, mats["cost_model"])}
    st = ctx.stored["V2_topk_profiling"]
    checks["topk_vs_stored"] = {k: [[st[k]["mean_regret_pct"], topk["cost_model"][k]["mean_regret_pct"]],
                                    [st[k]["p95_regret_pct"], topk["cost_model"][k]["p95_regret_pct"]]] for k in st}
    for name in sels:
        if name != "cost_model":
            topk[name] = {k: v for k, v in topk_table(ctx.df, ctx.chunks, mats[name]).items() if k in ("top2", "top3")}
    # leave-one-key-out of the stored protocol (E1 points of the held-out shape stay in training)
    if do_loso:
        lk = cm.loko(ctx.points, ctx.f, ctx.stats, ctx.hw, ctx.theta)
        sl = ctx.stored.get("V2_leave_one_key_out", {})
        checks["loko_stored_protocol_chunk_mean_regret_pct"] = [sl.get("chunk_mean_regret_pct"), lk["chunk_mean_regret_pct"]]
    res = {"experiment": "T1-baselines-r4", "tag": tag, "device": ctx.device, "hardware": ctx.hw,
           "n_calls": int(len(ctx.df)), "chunks": ctx.chunks, "n_calibration_points": int(len(ctx.points)),
           "published_cost_model": {k: sv[k] for k in ("mean_regret_pct", "p95_regret_pct", "tie_aware_accuracy")},
           "checks": checks, "forms": forms, "clean_selection_calibration": calibration_selection(forms),
           "baselines": rows, "selector_info": sel_info, "topk": topk,
           "e12_shard_widths": run_e12(ctx, sels), "seconds": round(time.time() - t0, 1)}
    return {"result": res, "selectors": sels, "per_call_forms": per_call}


# --------------------------------------------------------------------------- cross-GPU
def cross_gpu_form_selection(results: dict, per_call: dict) -> dict:
    """Form chosen on the captured calls of source GPU(s), refitted on the target's calibration data and
    scored on the target's captured calls. Ties: fewer parameters, then the source's held-out RMSE."""
    out = {}
    primary = [t for t in PRIMARY if t in results]
    nfree = {n: r["n_free_params"] for n, r in next(iter(results.values()))["forms"].items()}

    def chosen_on(srcs: list[str], cands: list[str], rule: str) -> tuple[str, list[str]]:
        val = {n: float(np.mean([results[s]["forms"][n]["captured_calls"]["mean_regret_pct"] for s in srcs])) for n in cands}
        second = {n: float(np.mean([results[s]["forms"][n].get("leave_one_shape_out", {}).get("heldout_rmse_log_mean", 0.0)
                                    for s in srcs])) for n in cands}
        if rule == "strict":
            n = select_strict(cands, val, nfree, second)
        else:
            regs = {m: np.concatenate([per_call[s][m] for s in srcs]) * 100 for m in cands}
            se = {m: float(regs[m].std(ddof=1) / math.sqrt(len(regs[m]))) for m in cands}
            n = select_one_se(cands, val, se, nfree, second)
        return n, tied_alternatives(n, cands, val, nfree)

    for tgt in results:
        rows = {}
        srcs = [s for s in primary if device_of(s) != device_of(tgt)]
        groups = [([s], f"from_{s}") for s in srcs]
        if tgt in PRIMARY and len(srcs) >= 2:
            groups.append((srcs, "leave_one_gpu_out"))
        for set_name, cands in CANDIDATE_SETS.items():
            for rule in ("strict", "one_se"):
                for src_list, label in groups:
                    n, alts = chosen_on(src_list, cands, rule)
                    rows[f"{set_name}|{rule}|{label}"] = {
                        "selected_form": n, "n_free_params": nfree[n], "sources": src_list,
                        **brief(results[tgt]["forms"][n]["captured_calls"]),
                        "exact_tie_alternatives": {a: brief(results[tgt]["forms"][a]["captured_calls"]) for a in alts}}
        out[tgt] = rows
    return out


def cross_gpu_baselines(results: dict, selectors: dict, ctxs: dict, in_dir: Path) -> dict:
    """Selectors fitted on the calibration data of GPU A, applied with the hardware constants and compile
    facts of GPU B to B's captured calls (B a different device); the cost model's top-k ranks with A's
    parameters and profiles on B."""
    out, checks = {}, {}
    for src in results:
        for tgt in results:
            if device_of(src) == device_of(tgt):
                continue
            ctx = ctxs[tgt]
            rows, mats = evaluate_on_calls(selectors[src], ctx)
            tk = topk_table(ctx.df, ctx.chunks, mats["cost_model"])
            rows["cost_model__top2"], rows["cost_model__top3"] = tk["top2"], tk["top3"]
            out[f"{src}->{tgt}"] = {n: brief(r) for n, r in rows.items()}
            stored = results[src].get("_stored_v3", {}).get(tgt)
            if stored is not None:
                checks[f"{src}->{tgt}"] = [stored, rows["cost_model"]["mean_regret_pct"]]
    p = in_dir / "cost_model_transfer_h800.json"
    if p.exists():
        for k, v in json.loads(p.read_text())["transfers"].items():
            if k in out:
                checks[k] = [v["mean_regret_pct"], out[k]["cost_model"]["mean_regret_pct"]]
    return {"pairs": out, "checks_cost_model_transfer_vs_stored": checks}


def _worker(args):
    tag, in_dir, do_loso, do_learned = args
    return tag, analyze_tag(tag, in_dir, do_loso, do_learned)


# --------------------------------------------------------------------------- summary
def summarize(results: dict, xsel: dict, xbase: dict) -> dict:
    s = {"experiment": "T1-baselines-r4-summary", "definitions": __doc__, "published_mean_regret_pct": PUBLISHED_MEAN_PCT}
    s["captured_calls"] = {t: {n: brief(r) for n, r in res["baselines"].items()} for t, res in results.items()}
    s["forms_captured_calls"] = {t: {n: {"n_free_params": r["n_free_params"], **brief(r["captured_calls"]),
                                         **({"loso_mean_chunk_regret_pct": r["leave_one_shape_out"]["mean_chunk_regret_pct"],
                                             "loso_se_pct": r["leave_one_shape_out"]["se_chunk_regret_pct"],
                                             "loso_heldout_rmse_log": r["leave_one_shape_out"]["heldout_rmse_log_mean"]}
                                            if "leave_one_shape_out" in r else {})}
                                     for n, r in res["forms"].items()} for t, res in results.items()}
    s["clean_selection_calibration"] = {t: res["clean_selection_calibration"] for t, res in results.items()}
    s["clean_selection_cross_gpu"] = xsel
    s["topk_cost_model"] = {t: res["topk"]["cost_model"] for t, res in results.items()}
    s["topk_other_rankers"] = {t: {n: {k: brief(v) for k, v in r.items()} for n, r in res["topk"].items() if n != "cost_model"}
                               for t, res in results.items()}
    s["shard_widths_pooled"] = {t: {n: brief(v) for n, v in res["e12_shard_widths"]["pooled_shards"].items()}
                                for t, res in results.items() if res["e12_shard_widths"]}
    s["shard_widths_pooled_topk"] = {t: {n: brief(v) for n, v in res["e12_shard_widths"]["pooled_shards_topk"].items()}
                                     for t, res in results.items() if res["e12_shard_widths"]}
    s["cross_gpu_baselines"] = xbase
    s["tfla"] = {t: {n: v for n, v in res["selector_info"].items() if n.startswith("tfla")} for t, res in results.items()}
    s["checks"] = {t: res["checks"] for t, res in results.items()}
    return s


ROW_ORDER = ["cost_model", "work_only", "work_term_model", "nearest_calibrated_key", "interp_log_length_same_batch",
             "loglinear_interp_logL_logB", "length_tree_calibration", "shape_tree_calibration", "gbr_basic",
             "gbr_structural", "mlp_basic", "mlp_structural", *TFLA_MODES,
             *(f"tfla_runtime_intensity_{i}" for i in TFLA_FIXED_INTENSITY)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--tags", default=DEFAULT_TAGS)
    p.add_argument("--jobs", type=int, default=5, help="worker processes (one tag per process)")
    p.add_argument("--no-loso", action="store_true", help="skip the leave-one-shape-out refits (smoke test)")
    p.add_argument("--no-learned", action="store_true", help="skip the gradient-boosted and MLP regressors (smoke test)")
    args = p.parse_args(argv)
    out_dir = args.out_dir or args.in_dir
    tags = [t for t in args.tags.split(",") if t]
    tags = [t for t in tags if (args.in_dir / f"e2_mamba1_{t}.csv").exists() and (args.in_dir / f"e4_{t}.jsonl").exists()]
    jobs = [(t, args.in_dir, not args.no_loso, not args.no_learned) for t in tags]
    out = {}
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as ex:
            for tag, r in ex.map(_worker, jobs):
                out[tag] = r
                print(f"{tag}: done in {r['result']['seconds']} s", flush=True)
    else:
        for j in jobs:
            tag, r = _worker(j)
            out[tag] = r
            print(f"{tag}: done in {r['result']['seconds']} s", flush=True)
    results = {t: out[t]["result"] for t in tags}
    selectors = {t: out[t]["selectors"] for t in tags}
    per_call = {t: out[t]["per_call_forms"] for t in tags}
    ctxs = {t: Ctx(t, args.in_dir) for t in tags}
    for t in tags:   # stored V3 transfer of the cost model, for the reproduction check
        results[t]["_stored_v3"] = {k[len("V3_transfer_to_"):]: v["e2"]["mean_regret_pct"]
                                    for k, v in ctxs[t].stored.items() if k.startswith("V3_transfer_to_")}
    xsel = cross_gpu_form_selection(results, per_call)
    xbase = cross_gpu_baselines(results, selectors, ctxs, args.in_dir)
    for t in tags:
        results[t].pop("_stored_v3")
        r = dict(results[t])
        r["clean_selection_cross_gpu_as_target"] = xsel.get(t, {})
        r["cross_gpu_baselines"] = {k: v for k, v in xbase["pairs"].items() if k.startswith(f"{t}->") or k.endswith(f"->{t}")}
        write_json(out_dir / f"cost_model_baselines_r4_{t}.json", r)
    summary = summarize(results, xsel, xbase)
    write_json(out_dir / "cost_model_baselines_r4_summary.json", summary)

    # ---- console tables
    for t in tags:
        res = results[t]
        print(f"\n== {t}: captured calls (n={res['n_calls']}), regret mean / p95 / p99 %, tied share")
        for n in ROW_ORDER:
            if n in res["baselines"]:
                r = res["baselines"][n]
                print(f"  {n:30s} {r['mean_regret_pct']:7.2f} {r['p95_regret_pct']:7.2f} {r.get('p99_regret_pct', float('nan')):7.2f}"
                      f"  tied {r['tie_aware_accuracy']:.3f}")
        tk = res["topk"]["cost_model"]
        print("  top-k (cost model): " + "; ".join(f"{k} {v['mean_regret_pct']:.2f}/{v['p95_regret_pct']:.2f}/{v['p99_regret_pct']:.2f}"
                                                  for k, v in tk.items()))
        for set_name, rr in res["clean_selection_calibration"].items():
            print(f"  clean selection (calibration LOSO, {set_name}): " + "; ".join(
                f"{rule} -> {v['selected_form']} ({v['captured_calls']['mean_regret_pct']:.2f}/"
                f"{v['captured_calls']['p95_regret_pct']:.2f}, tied {v['captured_calls']['tie_aware_accuracy']:.3f})"
                for rule, v in rr.items()))
        for k, v in xsel.get(t, {}).items():
            print(f"  cross-GPU selection {k:45s} -> {v['selected_form']:22s} {v['mean_regret_pct']:6.2f} / {v['p95_regret_pct']:6.2f}"
                  f" tied {v['tie_aware_accuracy']:.3f}")
        if res["e12_shard_widths"]:
            print("  shard widths, pooled (n=%d):" % res["e12_shard_widths"]["n_shard_shapes"])
            for n, v in res["e12_shard_widths"]["pooled_shards"].items():
                print(f"    {n:30s} {v['mean_regret_pct']:7.2f} {v['p95_regret_pct']:7.2f} {v['p99_regret_pct']:7.2f} tied {v['tied_share']:.3f}")
            for n in ("cost_model__top2", "cost_model__top3"):
                v = res["e12_shard_widths"]["pooled_shards_topk"][n]
                print(f"    {n:30s} {v['mean_regret_pct']:7.2f} {v['p95_regret_pct']:7.2f} {v['p99_regret_pct']:7.2f} tied {v['tied_share']:.3f}")
        print("  checks:", json.dumps(res["checks"], default=float)[:2000])
    print("\ncross-GPU cost-model transfer checks:", json.dumps(xbase["checks_cost_model_transfer_vs_stored"]))


if __name__ == "__main__":
    main()
