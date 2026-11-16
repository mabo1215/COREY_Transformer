"""JPDC review items M2 (identifiability), M1 (evidence) and E3 (cost figures) of the Triton Mamba-1 cost model
(Eq. (3)). CPU only; reads stored result files and times nothing.

Fits use the stored protocol (``taco_model_baselines.fit_fast`` / ``fit_single``: soft-l1 on log latency, f_scale 0.1,
bounds of ``cost_model``). The device-time tags of Table 5 are analysed; "own table" is the per-call table of the
calibration tag (the stored analyses), "primary table" the round-median table of the other Table-5 rows.

Reproduction  the stored parameters and a three-restart refit from X0 must give the stored regret; otherwise the
     script stops.
Task 1 (M2)
  multistart        random starts per GPU (same stream as the stored 50-start analysis, seed 0, longer) on the 2298
                    timings and on the 90 full-protocol timings: objective, mean per-call regret of every solution, the
                    distribution among solutions within 0.5% and 2% of the best objective, the distinct optima.
  w0_w1_grid        for fixed (W0, W1) Eq. (3) is linear in the six time constants, and the objective changes form only
                    where W0 or W1 crosses a resident-warp level of the data. A grid over those levels and the midpoints
                    between them, refined by a further midpoint around the cells within 10% of the lowest objective, with
                    the time constants refitted per cell and the best cells polished with all eight parameters free, gives
                    a deterministic search for the optimum that random starts can miss, the number of separate basins
                    below each objective tolerance and the regret across their cells. Cell counts depend on this mesh.
  cluster_bootstrap resampling of the twelve calibration keys (seed 1, the stored stream; the first 200 replicates are the
                    stored ones): stored protocol, and a stronger optimizer (protocol plus warm starts from the known
                    optima, lowest objective kept) that separates optimizer noise from sampling noise. Pearson and rank
                    correlations, pairs with |r| > 0.9, share on bounds and on kinks, spread of parameter combinations.
  jacobian          singular values of d log T / d log theta at the reported and at the best fit (parameter scale
                    max(theta, 1e-3 X0)), also on the interior parameters and with unit columns; sloppiest directions.
  profile           profile of the objective for each parameter (refit of the others from several starts). 95% limits
                    with the robust likelihood-ratio calibration: delta cost = 0.5 q lambda. iid: q = chi2(1), lambda =
                    E[psi^2] / E[psi'] (all timings independent; optimistic). cluster: q = F(1, G - 1), lambda per
                    parameter = (H^-1 V H^-1)_ii / (H^-1)_ii with V the score covariance summed over the G = 12 keys.
Task 2 (physical reading)  fitted terms against the Nsight Compute counters of the calibration shapes (stock RTX 3090; no
     other GPU has counters), the compile-time spill sizes and the launch-table intervention at chunk 128. The profiler's
     occupancy limit differs from the resident programs of Eq. (3) in one register class, so the fit protocol, the multi-start
     and the leave-one-key-out are also repeated with registers divided per scheduler (validated on the RTX 3090 cells only).
Task 3 (M1)  manuscript values against JSON key paths of the stored result files, the leave-one-key-out regret with cold
     starts next to the stored warm start, and multi-starts of the target-calibration forms.
Task 4 (E3)  calibration and profiling costs recomputed from the E4 logs and the per-call tables for both key sets.

Output: ``jpdc_model_identifiability.json``.

Usage:  PYTHONPATH=. python -X utf8 -m src.experiments.tpds.jpdc_model_identifiability [--jobs 8] [--n-starts 300]
        [--n-boot 1000]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.stats import chi2, f as f_dist, spearmanr  # noqa: E402


def _triton_stand_in() -> None:
    """The stored analyses reach the kernel module, which needs Triton only at import time. Nothing here compiles or
    runs a kernel, so a CPU-only host gets an empty stand-in."""
    if importlib.util.find_spec("triton") is None:
        mod = types.ModuleType("triton")
        mod.jit = lambda fn: fn
        mod.language = types.ModuleType("triton.language")
        sys.modules["triton"], sys.modules["triton.language"] = mod, mod.language


_triton_stand_in()

from src.experiments.tpds import cost_model as cm  # noqa: E402
from src.experiments.tpds import cost_model_baselines_r4 as r4  # noqa: E402
from src.experiments.tpds import taco_model_baselines as TMB  # noqa: E402
from src.experiments.tpds import taco_r2_baselines_common as C  # noqa: E402
from src.experiments.tpds.analysis import latency_matrix, load_e2  # noqa: E402
from src.experiments.tpds.common import RESULTS_DIR, read_jsonl, utc_now, write_json  # noqa: E402

GPUS = ("pro6000", "vgpu3090", "h800")
TAGS = TMB.TAGS["device_time"]
LI_TAGS = TMB.TAGS["launch_inclusive"]
OWN_TABLE = {"pro6000": "pro6000_device_pooled", "vgpu3090": "vgpu3090_device_one_block", "h800": "h800_device"}
PRIMARY_TABLE = {"pro6000": "pro6000_device", "vgpu3090": "vgpu3090_device", "h800": "h800_device"}
TARGET_FORM = {"pro6000": "drop_scan_work", "vgpu3090": "work_only", "h800": "drop_spill"}
FREE = [i for i in range(len(cm.PARAMS)) if i not in cm.DEFAULT_FIXED]
NAMES = [cm.PARAMS[i] for i in FREE]
I_W0, I_W1 = cm.PARAMS.index("W0"), cm.PARAMS.index("W1")
START_SEED, BOOT_SEED = TMB.SEED, TMB.SEED + 1      # the streams of the stored multi-start and bootstrap
LOKO_SEED, FORM_SEED = 1000, 2000
F_SCALE = 0.1
NEAR = {"within_0p5pct": 0.005, "within_2pct": 0.02, "within_5pct": 0.05, "within_10pct": 0.10, "within_20pct": 0.20}
SCALE_FLOOR = 1e-3 * cm.X0                          # parameter scale of a time constant at its lower bound
N_POLISH = 20
REFINE_TOL = 0.10
SM_CLOCK_GHZ = {"pro6000": (2.332, 2.422), "vgpu3090": (1.695, 1.950), "h800": (1.515, 1.755)}   # read around each cell
N_ANCHORS = 4
MAX_REANCHOR = 2
BOOT_STORED = 200
STORED = "taco_r2_baselines_m4.json"
STORED_R1 = "taco_model_baselines.json"

_DATA: dict[str, "TagData"] = {}
_IN_DIR = RESULTS_DIR


# =========================================================================== data
class TagData:
    """Calibration timings, captured calls and the two per-call tables of one GPU's device-time tag."""

    def __init__(self, gpu: str, in_dir: Path):
        self.gpu, self.tag = gpu, TAGS[gpu]
        self.ctx = r4.Ctx(self.tag, Path(in_dir))
        self.F, self.Fc = TMB.farr(self.ctx.f), TMB.farr(self.ctx.calls.fc)
        self.y = self.ctx.y
        pts = self.ctx.points
        src = pts["src"].to_numpy()
        self.src = src
        self.masks = {"all": np.ones(len(src), bool), "full_protocol_only": np.isin(src, ["e4_static", "e1"]),
                      "static72_only": src == "e4_static"}
        self.primary = C.triton_table(PRIMARY_TABLE[gpu], Path(in_dir))
        pos = {str(k): i for i, k in enumerate(self.ctx.df["key"])}
        self.rows = np.array([pos[k] for k in self.primary.keys()])
        self.shapes = sorted(set(zip(pts["L"], pts["B"])))
        self.cluster = np.array([self.shapes.index(s) for s in zip(pts["L"], pts["B"])])
        self.cl_idx = [np.flatnonzero(self.cluster == j) for j in range(len(self.shapes))]

    def subset(self, mask: str) -> tuple[dict, np.ndarray]:
        m = self.masks[mask]
        return TMB.sub(self.F, m), self.y[m]

    def levels(self, mask: str) -> list[float]:
        F, _ = self.subset(mask)
        return sorted({float(v) for v in F["m"] * F["nw"]})

    def score(self, theta: np.ndarray, Fc: dict | None = None) -> dict:
        """Per-call regret and latency-weighted share of the chunk selected by ``theta`` on both tables."""
        S = self.ctx.calls.matrix(TMB.predict_np(theta, self.Fc if Fc is None else Fc))
        ch = S.argmin(1)
        lat = self.ctx.lat
        got, orc = lat[np.arange(len(lat)), ch], lat.min(1)
        T = self.primary
        pick = S[self.rows].argmin(1)
        pl = T.lat[np.arange(T.N), pick]
        return {"own_mean_regret_pct": float(100 * (got / orc - 1).mean()),
                "own_lw_share_of_oracle_pct": float(100 * (got - orc).sum() / orc.sum()),
                "primary_mean_regret_pct": float(100 * (pl / T.orc - 1).mean()),
                "primary_lw_share_of_oracle_pct": float(100 * (pl - T.orc).sum() / T.orc.sum()),
                "choice": ch.astype(np.int8)}


def _init(in_dir: str) -> None:
    global _IN_DIR
    _IN_DIR = Path(in_dir)


def _data(gpu: str) -> TagData:
    if gpu not in _DATA:
        _DATA[gpu] = TagData(gpu, _IN_DIR)
    return _DATA[gpu]


def random_starts(n: int, seed: int, fixed: dict | None = None) -> list[np.ndarray]:
    """``taco_model_baselines.m3c_tag`` start distribution: same draws, in the same order, for the same seed."""
    fixed = cm.DEFAULT_FIXED if fixed is None else fixed
    free = [i for i in range(len(cm.PARAMS)) if i not in fixed]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = cm.X0.copy()
        for i in free:
            if i < cm.N_SCALE:
                x[i] = cm.X0[i] * 10.0 ** rng.uniform(-2, 2)
            elif cm.PARAMS[i] == "W0":
                x[i] = rng.uniform(1.0, 64.0)
            else:
                x[i] = rng.uniform(1.0, 256.0)
        out.append(np.clip(x, cm.LO, cm.HI))
    return out


def params_of(theta: np.ndarray) -> dict:
    return {cm.PARAMS[i]: float(theta[i]) for i in FREE}


def bound_flags(theta: np.ndarray, levels: list[float]) -> dict:
    """Which parameters sit on a bound or on a kink. A time constant is at its lower bound below 1e-3 X0; W0 and W1 at a
    kink within 1e-4 (relative) of a resident-warp level, and beyond the data above the highest level."""
    lv = np.array(levels)
    at_lo, at_hi, kink, beyond = [], [], [], []
    for i, n in zip(FREE, NAMES):
        v = float(theta[i])
        if n in ("W0", "W1"):
            if v <= 1.001:
                at_lo.append(n)
            if abs(v / lv - 1).min() <= 1e-4:
                kink.append(n)
            if v >= lv.max():
                beyond.append(n)
        elif v <= cm.LO[i] + SCALE_FLOOR[i]:
            at_lo.append(n)
        if v >= 0.99 * cm.HI[i]:
            at_hi.append(n)
    return {"at_lower_bound": at_lo, "at_upper_bound": at_hi, "on_kink": kink, "beyond_highest_W_level": beyond}


def dist(v) -> dict:
    v = np.asarray(v, float)
    return {"n": int(len(v)), "min": float(v.min()), "p25": float(np.percentile(v, 25)), "median": float(np.median(v)),
            "p75": float(np.percentile(v, 75)), "max": float(v.max()), "mean": float(v.mean())}


# =========================================================================== jobs (worker processes)
def _start_job(job) -> tuple:
    gpu, mask, x, fixed = job
    d = _data(gpu)
    Fm, ym = d.subset(mask)
    th, cost = TMB.fit_single(Fm, ym, x, fixed=fixed)
    s = d.score(th)
    return float(cost), th, s


def _grid_job(job) -> tuple:
    gpu, mask, v0, v1, anchors = job
    d = _data(gpu)
    Fm, ym = d.subset(mask)
    fixed = {**cm.DEFAULT_FIXED, I_W0: v0, I_W1: v1}
    best = None
    for a in anchors:
        x = a.copy()
        x[I_W0], x[I_W1] = v0, v1
        th, cost = TMB.fit_single(Fm, ym, x, fixed=fixed)
        if best is None or cost < best[0]:
            best = (float(cost), th)
    s = d.score(best[1])
    return best[0], best[1], s


def _polish_job(job) -> tuple:
    gpu, mask, x = job
    d = _data(gpu)
    Fm, ym = d.subset(mask)
    th, cost = TMB.fit_single(Fm, ym, x)
    return float(cost), th, d.score(th)


def _profile_job(job) -> dict:
    """One parameter: sweep outward from the optimum in both directions, warm-started from the previous point and from
    the known optima with the parameter replaced; the lowest objective per point is kept."""
    gpu, mask, i, below, above, anchors = job
    d = _data(gpu)
    Fm, ym = d.subset(mask)
    rows = []
    for sweep in (below, above):
        prev = None
        for v in sweep:
            fixed = {**cm.DEFAULT_FIXED, i: float(v)}
            best = None
            for a in ([] if prev is None else [prev]) + list(anchors):
                x = a.copy()
                x[i] = v
                th, cost = TMB.fit_single(Fm, ym, x, fixed=fixed)
                if best is None or cost < best[0]:
                    best = (float(cost), th)
            prev = best[1]
            s = d.score(best[1])
            rows.append({"value": float(v), "cost": best[0], "theta": best[1], "own_mean_regret_pct": s["own_mean_regret_pct"],
                         "primary_mean_regret_pct": s["primary_mean_regret_pct"], "choice": s["choice"]})
    return {"param": cm.PARAMS[i], "rows": sorted(rows, key=lambda r: r["value"])}


def _boot_job(job) -> dict:
    gpu, draw, anchors = job
    d = _data(gpu)
    idx = np.concatenate([d.cl_idx[j] for j in draw])
    Fb, yb = TMB.sub(d.F, idx), d.y[idx]
    th, info = TMB.fit_fast(Fb, yb)
    s = d.score(th)
    best_th, best_cost = th, info["cost"]
    for a in anchors:
        t, c = TMB.fit_single(Fb, yb, a)
        if c < best_cost:
            best_th, best_cost = t, float(c)
    s2 = d.score(best_th)
    return {"theta": th[FREE], "mean": s["own_mean_regret_pct"], "choice": s["choice"], "theta_strong": best_th[FREE],
            "mean_strong": s2["own_mean_regret_pct"], "choice_strong": s2["choice"], "n_distinct": len(set(draw.tolist()))}


def _loko_job(job) -> dict:
    """One calibration shape held out (all its sources), as ``taco_model_baselines.loso_fast``: stored warm start from the
    fit on all keys, cold start from X0, and the best of random starts plus both."""
    gpu, key, n_starts, seed = job
    d = _data(gpu)
    pts = d.ctx.points
    L, B, dd = pts["L"].to_numpy(), pts["B"].to_numpy(), pts["d"].to_numpy()
    kk, Cc = pts["key"].to_numpy(), pts["C"].to_numpy(int)
    i0 = int(np.flatnonzero(kk == key)[0])
    hold = (L == L[i0]) & (B == B[i0]) & (dd == dd[i0])
    st = (kk == key) & (d.src == "e4_static")
    Ftr, ytr = TMB.sub(d.F, ~hold), d.y[~hold]
    warm, ci_warm = TMB.fit_fast(Ftr, ytr, x0=d.ctx.theta)
    cold, ci_cold = TMB.fit_fast(Ftr, ytr)
    best, best_c = (warm, ci_warm["cost"]) if ci_warm["cost"] <= ci_cold["cost"] else (cold, ci_cold["cost"])
    for x in random_starts(n_starts, seed):
        th, c = TMB.fit_single(Ftr, ytr, x)
        if c < best_c:
            best, best_c = th, float(c)
    meas = dict(zip(Cc[st], d.y[st]))
    c_best = int(min(meas, key=meas.get))

    def rec(th, cost):
        pst = TMB.predict_np(th, TMB.sub(d.F, st))
        c_pred = int(Cc[st][int(np.argmin(pst))])
        ph = TMB.predict_np(th, TMB.sub(d.F, hold))
        return {"cost": float(cost), "chunk_pred": c_pred, "chunk_regret_pct": float(100 * (meas[c_pred] / meas[c_best] - 1)),
                "heldout_rmse_log": float(np.sqrt(np.mean((np.log(ph) - np.log(d.y[hold])) ** 2))),
                "captured_own_mean_regret_pct": d.score(th)["own_mean_regret_pct"]}
    return {"key": str(key), "chunk_best": c_best, "warm_from_full_fit": rec(warm, ci_warm["cost"]),
            "cold_from_X0": rec(cold, ci_cold["cost"]), "best_of_starts": rec(best, best_c)}


def pmap(ex, fn, jobs: list) -> list:
    return list(ex.map(fn, jobs, chunksize=max(1, len(jobs) // 200)))


# =========================================================================== multi-start
def near_equal(costs, own, prim, choices, ref_choice, best_cost: float) -> dict:
    out = {}
    for lab, tol in NEAR.items():
        s = costs <= best_cost * (1 + tol)
        rec = {"tolerance": tol, "n_solutions": int(s.sum())}
        if s.any():
            rec["own_mean_regret_pct"] = dist(own[s])
            rec["primary_mean_regret_pct"] = dist(prim[s])
            rec["n_distinct_choice_vectors"] = len({c.tobytes() for c in choices[s]})
            rec["min_share_same_choice_as_best"] = float((choices[s] == ref_choice).mean(1).min())
        out[lab] = rec
    return out


def distinct_optima(costs, thetas, own, prim, choices, levels, top: int = 12) -> list[dict]:
    groups = []
    for i in np.argsort(costs):
        for g in groups:
            if abs(costs[i] / g["cost"] - 1) <= 1e-4 and g["_c"] == choices[i].tobytes():
                g["n_starts"] += 1
                break
        else:
            groups.append({"cost": float(costs[i]), "n_starts": 1, "own_mean_regret_pct": float(own[i]),
                           "primary_mean_regret_pct": float(prim[i]), "params": params_of(thetas[i]),
                           "flags": bound_flags(thetas[i], levels), "_c": choices[i].tobytes()})
    return [{k: v for k, v in g.items() if k != "_c"} for g in groups[:top]]


def fit_record(d: TagData, theta: np.ndarray, cost: float, mask: str, ref_choice=None) -> dict:
    s = d.score(theta)
    rec = {"cost": float(cost), "params": params_of(theta), "flags": bound_flags(theta, d.levels(mask)),
           **{k: v for k, v in s.items() if k != "choice"}}
    if ref_choice is not None:
        rec["share_calls_same_choice_as_reported_fit"] = float((s["choice"] == ref_choice).mean())
    return rec


def run_multistart(ex, gpu: str, d: TagData, mask: str, n: int, reported: dict, best_extra=None) -> dict:
    res = pmap(ex, _start_job, [(gpu, mask, x, None) for x in random_starts(n, START_SEED)])
    costs = np.array([r[0] for r in res])
    thetas = np.array([r[1] for r in res])
    own = np.array([r[2]["own_mean_regret_pct"] for r in res])
    prim = np.array([r[2]["primary_mean_regret_pct"] for r in res])
    choices = np.array([r[2]["choice"] for r in res])
    b = int(np.argmin(costs))
    out = {"n_starts": n, "start_distribution": "time constants X0 x 10^U(-2, 2), W0 ~ U(1, 64), W1 ~ U(1, 256), one "
                                                "soft-l1 run each (seed 0; the first 50 draws are the stored ones)",
           "best": fit_record(d, thetas[b], costs[b], mask, reported["choice"]),
           "best_cost_rel_improvement_over_three_restart": float(1 - costs[b] / reported["cost"]),
           "n_starts_within_1e-6_of_best": int((costs <= costs[b] * (1 + 1e-6)).sum()),
           "first_50_starts": {"best_cost": float(costs[:50].min()), "own_regret_of_best": float(own[int(np.argmin(costs[:50]))]),
                               "rel_improvement_over_three_restart": float(1 - costs[:50].min() / reported["cost"])},
           "near_equal": near_equal(costs, own, prim, choices, choices[b], float(costs[b])),
           "all_starts": {"cost": dist(costs), "own_mean_regret_pct": dist(own), "primary_mean_regret_pct": dist(prim)},
           "distinct_optima": distinct_optima(costs, thetas, own, prim, choices, d.levels(mask)),
           "per_start_cost_and_own_regret": [[round(float(c), 6), round(float(g), 4)] for c, g in zip(costs, own)],
           "_arrays": (costs, own, prim, choices)}
    return out, thetas[b], float(costs[b]), thetas, costs


def anchors_from(thetas, costs, stored: np.ndarray, k: int = N_ANCHORS) -> list[np.ndarray]:
    """The lowest optima that differ by more than 0.1% of the objective, then the stored fit and X0."""
    out, seen = [], []
    for i in np.argsort(costs):
        if all(abs(costs[i] / c - 1) > 1e-3 for c in seen):
            out.append(thetas[i].copy())
            seen.append(float(costs[i]))
        if len(out) >= k:
            break
    return out + [stored.copy(), cm.X0.copy()]


# =========================================================================== W0 x W1 grid
def w_values(levels: list[float], hi: float, extra=()) -> list[float]:
    vals = {1.0, float(hi)}
    prev = 1.0
    for v in levels:
        vals.add(float(v))
        if v > prev:
            vals.add(0.5 * (prev + float(v)))
        prev = float(v)
    if prev < hi:
        vals.add(0.5 * (prev + float(hi)))
    return sorted(vals | {float(x) for x in extra if 1.0 <= x <= hi})


def run_grid(ex, gpu: str, d: TagData, mask: str, anchors: list[np.ndarray], reported: dict) -> tuple[dict, np.ndarray, float]:
    lv = d.levels(mask)
    c0, c1 = w_values(lv, cm.HI[I_W0]), w_values(lv, cm.HI[I_W1])
    coarse = pmap(ex, _grid_job, [(gpu, mask, a, b, anchors[:3]) for a in c0 for b in c1])
    cost_c = np.array([r[0] for r in coarse]).reshape(len(c0), len(c1))
    near = ndimage.binary_dilation(cost_c <= cost_c.min() * (1 + REFINE_TOL), structure=np.ones((3, 3)))
    # the fine mesh adds the midpoint of every coarse interval; only cells touching a good coarse cell are evaluated
    v0s = sorted(set(c0) | {0.5 * (a + b) for a, b in zip(c0[:-1], c0[1:])})
    v1s = sorted(set(c1) | {0.5 * (a + b) for a, b in zip(c1[:-1], c1[1:])})
    i0 = {v: min(np.searchsorted(c0, v, side="right") - 1, len(c0) - 1) for v in v0s}
    i1 = {v: min(np.searchsorted(c1, v, side="right") - 1, len(c1) - 1) for v in v1s}
    best_coarse = coarse[int(np.argmin(cost_c))][1]
    done = {(a, b): coarse[j * len(c1) + k] for j, a in enumerate(c0) for k, b in enumerate(c1)}
    todo = []
    for a in v0s:
        for b in v1s:
            ia, ib = i0[a], i1[b]
            if (a, b) not in done and near[ia:ia + 2, ib:ib + 2].any():
                todo.append((a, b, [coarse[ia * len(c1) + ib][1], best_coarse]))
    fine = pmap(ex, _grid_job, [(gpu, mask, a, b, anc) for a, b, anc in todo])
    done.update({(a, b): r for (a, b, _), r in zip(todo, fine)})
    keys = [(a, b) for a in v0s for b in v1s]
    cost = np.array([done[k][0] if k in done else np.inf for k in keys]).reshape(len(v0s), len(v1s))
    own = np.array([done[k][2]["own_mean_regret_pct"] if k in done else np.nan for k in keys]).reshape(cost.shape)
    prim = np.array([done[k][2]["primary_mean_regret_pct"] if k in done else np.nan for k in keys]).reshape(cost.shape)
    order = np.argsort(cost.ravel())[:N_POLISH]
    pol = pmap(ex, _polish_job, [(gpu, mask, done[keys[i]][1]) for i in order])
    jb = int(np.argmin([x[0] for x in pol]))
    b_cost, b_th = pol[jb][0], pol[jb][1]
    if cost.min() < b_cost:
        k = int(np.argmin(cost.ravel()))
        b_cost, b_th = float(cost.min()), done[keys[k]][1]
    ref = d.score(b_th)["choice"]
    out = {"n_W0_values_fine_mesh": len(v0s), "n_W1_values_fine_mesh": len(v1s), "n_cells_evaluated": len(done),
           "W_levels_in_data": lv, "best_cost_after_polish": b_cost, "best_after_polish": fit_record(d, b_th, b_cost, mask, reported["choice"]),
           "rel_improvement_over_three_restart": float(1 - b_cost / reported["cost"]), "sublevel_sets": {}}
    for lab, tol in NEAR.items():
        sel = cost <= b_cost * (1 + tol)
        rec = {"tolerance": tol, "n_cells": int(sel.sum()), "n_basins_in_W0_W1": int(ndimage.label(sel)[1])}
        if sel.any():
            ij = np.argwhere(sel)
            cells = [keys[int(a) * len(v1s) + int(b)] for a, b in ij]
            th_s = np.array([done[k][1] for k in cells])
            ch_s = np.array([done[k][2]["choice"] for k in cells])
            rec.update({"own_mean_regret_pct": dist(own[sel]), "primary_mean_regret_pct": dist(prim[sel]),
                        "W0_range": [v0s[int(ij[:, 0].min())], v0s[int(ij[:, 0].max())]],
                        "W1_range": [v1s[int(ij[:, 1].min())], v1s[int(ij[:, 1].max())]],
                        "param_range": {n: [float(th_s[:, i].min()), float(th_s[:, i].max())] for i, n in zip(FREE, NAMES)},
                        "n_distinct_choice_vectors": len({c.tobytes() for c in ch_s}),
                        "min_share_same_choice_as_best": float((ch_s == ref).mean(1).min())})
            lab_arr, nb = ndimage.label(sel)
            bas = []
            for b in range(1, nb + 1):
                m = lab_arr == b
                k = np.argwhere(m)[int(np.argmin(cost[m]))]
                bas.append({"best_cost": float(cost[k[0], k[1]]), "W0": v0s[int(k[0])], "W1": v1s[int(k[1])],
                            "own_mean_regret_pct_at_best": float(own[k[0], k[1]]),
                            "own_mean_regret_pct_range": [float(own[m].min()), float(own[m].max())], "n_cells": int(m.sum())})
            rec["basins"] = sorted(bas, key=lambda x: x["best_cost"])[:8]
        out["sublevel_sets"][lab] = rec
    return out, b_th, b_cost


# =========================================================================== Jacobian and inference
def log_jacobian(theta: np.ndarray, F: dict, kink_active: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """d log T / d theta for the eight free parameters and T (bandwidth floor included: floored rows have zero rows)."""
    t0, tf, td, tm, tl, te, ts, tx, tq, tp, W0, W1 = theta
    if tf or td or tl or tx:
        raise ValueError("the Jacobian covers the eight-parameter form only")
    N = cm.D_STATE
    W = F["m"] * F["nw"]
    wn = F["waves"] * F["n_tiles"]
    lg = np.log2(np.clip(F["lanes_c"], 1, None))
    fe, fs, fq = N * F["e"], N * F["e_d"] * lg, N * F["spill_words"]
    fp = fq * F["l1_miss"]
    A = N * tm
    I = te * fe + ts * fs + tq * fq + tp * fp
    hide, share = np.minimum(1.0, W1 / W), np.maximum(1.0, W / W0)
    T = t0 + wn * (A * hide + I * share)
    tol = 1e-4 if kink_active else 0.0
    act0 = W > W0 * (1 - tol)
    act1 = W > W1 * (1 - tol)
    J = np.column_stack([np.ones_like(T), wn * N * hide, wn * fe * share, wn * fs * share, wn * fq * share, wn * fp * share,
                         -wn * I * (W / W0 ** 2) * act0, wn * (A / W) * act1])
    floored = T < F["t_mem_ms"]
    T = np.maximum(T, F["t_mem_ms"])
    J = J / T[:, None]
    J[floored] = 0.0
    return J, T


def param_scale(theta: np.ndarray) -> np.ndarray:
    return np.array([max(float(theta[i]), float(SCALE_FLOOR[i])) if cm.PARAMS[i] not in ("W0", "W1") else float(theta[i])
                     for i in FREE])


def describe_direction(vec: np.ndarray, names: list[str], top: int = 4) -> str:
    order = np.argsort(-np.abs(vec))
    sgn = np.sign(vec[order[0]]) or 1.0
    return ", ".join(f"{sgn * vec[j]:+.2f} log {names[j]}" for j in order[:top] if abs(vec[j]) >= 0.1)


def jacobian_report(theta: np.ndarray, F: dict, y: np.ndarray, levels: list[float]) -> dict:
    flags = bound_flags(theta, levels)
    names = list(NAMES)
    J, T = log_jacobian(theta, F)
    Js = J * param_scale(theta)
    n = len(T)
    r = np.log(T) - np.log(y)
    w = (1 + (r / F_SCALE) ** 2) ** -0.25

    def svd_block(M, nm):
        U, sv, Vt = np.linalg.svd(M, full_matrices=False)
        col = np.linalg.norm(M, axis=0)
        Mu = M / np.where(col > 0, col, 1.0)
        svu = np.linalg.svd(Mu, compute_uv=False)
        Mw = M * w[:, None]
        svw = np.linalg.svd(Mw, compute_uv=False)
        rec = {"params": nm, "singular_values": [float(x) for x in sv], "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else None,
               "rms_change_of_log_latency_per_factor_2_along_each_direction": [float(x * math.log(2) / math.sqrt(n)) for x in sv],
               "condition_number_unit_columns": float(svu[0] / svu[-1]) if svu[-1] > 0 else None,
               "condition_number_robust_weights": float(svw[0] / svw[-1]) if svw[-1] > 0 else None,
               "n_directions_moving_log_latency_1pct_rms_per_factor_2": int((sv * math.log(2) / math.sqrt(n) >= 0.01).sum()),
               "column_rms_per_unit_log_parameter": {m: float(c / math.sqrt(n)) for m, c in zip(nm, col)},
               "directions_from_sloppiest": []}
        for j in range(len(sv) - 1, -1, -1):
            rec["directions_from_sloppiest"].append({
                "singular_value": float(sv[j]), "rms_log_latency_per_factor_2": float(sv[j] * math.log(2) / math.sqrt(n)),
                "loadings": {m: float(v) for m, v in zip(nm, Vt[j])}, "reading": describe_direction(Vt[j], nm)})
        return rec

    out = {"flags": flags, "parameter_scale": {m: float(s) for m, s in zip(NAMES, param_scale(theta))},
           "all_eight": svd_block(Js, names)}
    excl = set(flags["at_lower_bound"]) | set(flags["at_upper_bound"]) | set(flags["beyond_highest_W_level"])
    keep = [j for j, m in enumerate(names) if m not in excl]
    if len(keep) < len(names):
        out["interior_parameters"] = svd_block(Js[:, keep], [names[j] for j in keep])
    if flags["on_kink"]:
        Jk, _ = log_jacobian(theta, F, kink_active=True)
        Jks = Jk * param_scale(theta)
        out["all_eight_other_side_of_kink"] = svd_block(Jks, names)
    return out


def inference(d: TagData, mask: str, theta: np.ndarray) -> dict:
    """Robust likelihood-ratio calibration of the soft-l1 objective at ``theta``: 2 delta cost ~ lambda q. Parameters on a
    bound are left out of H and V; they get the median lambda of the others."""
    F, y = d.subset(mask)
    cl = d.cluster[d.masks[mask]]
    groups = [np.flatnonzero(cl == g) for g in np.unique(cl)]
    G = len(groups)
    J, T = log_jacobian(theta, F)
    Js = J * param_scale(theta)
    r = np.log(T) - np.log(y)
    rt = np.sqrt(1 + (r / F_SCALE) ** 2)
    psi, dpsi = r / rt, rt ** -3
    flags = bound_flags(theta, d.levels(mask))
    excl = set(flags["at_lower_bound"]) | set(flags["at_upper_bound"]) | set(flags["beyond_highest_W_level"])
    no_effect = [m for j, m in enumerate(NAMES) if not np.any(Js[:, j])]
    keep = [j for j, m in enumerate(NAMES) if m not in excl and m not in no_effect]
    Ji = Js[:, keep]
    H = (Ji.T * dpsi) @ Ji
    V = sum(np.outer(Ji[g].T @ psi[g], Ji[g].T @ psi[g]) for g in groups) * G / (G - 1)
    Hi = np.linalg.pinv(H, rcond=1e-12)
    S = Hi @ V @ Hi
    lam = np.divide(np.diag(S), np.diag(Hi), out=np.full(len(keep), np.nan), where=np.diag(Hi) > 0)
    lam_iid = float((psi ** 2).sum() / (len(y) - len(NAMES)) / dpsi.mean())
    q_iid, q_cl = float(chi2.ppf(0.95, 1)), float(f_dist.ppf(0.95, 1, G - 1))
    med = float(np.nanmedian(lam))
    lam = np.where(np.isfinite(lam) & (lam > 0), lam, med)
    lam_all = {m: med for m in NAMES}
    lam_all.update({NAMES[j]: float(l) for j, l in zip(keep, lam)})
    D = np.sqrt(np.diag(S))
    corr = np.divide(S, np.outer(D, D), out=np.full_like(S, np.nan), where=np.outer(D, D) > 0)
    pairs = [[NAMES[keep[a]], NAMES[keep[b]], float(corr[a, b])] for a in range(len(keep)) for b in range(a + 1, len(keep))
             if abs(corr[a, b]) > 0.9]
    return {"lambda_iid": lam_iid, "q_iid": q_iid, "q_cluster": q_cl, "n_clusters": G, "lambda_cluster": lam_all,
            "delta_cost_iid": 0.5 * q_iid * lam_iid, "delta_cost_cluster": {m: 0.5 * q_cl * lam_all[m] for m in NAMES},
            "params_in_H": [NAMES[j] for j in keep], "params_without_effect_on_these_timings": no_effect,
            "local_sd_log_param_iid": {NAMES[keep[j]]: float(math.sqrt(lam_iid * Hi[j, j])) for j in range(len(keep))},
            "local_sd_log_param_cluster": {NAMES[keep[j]]: float(D[j]) for j in range(len(keep))},
            "local_correlation_cluster_params": [NAMES[j] for j in keep],
            "local_correlation_cluster": [[None if not np.isfinite(x) else round(float(x), 3) for x in row] for row in corr],
            "local_pairs_abs_r_above_0p9": pairs}


# =========================================================================== profile
OFFSETS = (0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)


def profile_grid(i: int, hat: float, levels: list[float]) -> tuple[list[float], list[float]]:
    lo, hi = float(cm.LO[i]), float(cm.HI[i])
    if cm.PARAMS[i] in ("W0", "W1"):
        near = [hat * (1 + s * x) for x in (0.02, 0.05, 0.1, 0.2) for s in (-1, 1)]
        vals = w_values(levels, hi, near)
        return sorted([v for v in vals if v < hat], reverse=True), [v for v in vals if v > hat]
    base = max(hat, float(SCALE_FLOOR[i]))
    below = [base * 10 ** (-x) for x in OFFSETS if base * 10 ** (-x) > max(lo, 1e-3 * float(SCALE_FLOOR[i]))]
    above = [base * 10 ** x for x in OFFSETS if base * 10 ** x < hi]
    return below + [lo], above + [hi]


def interval(rows: list[dict], hat: float, c_min: float, delta: float, lo: float, hi: float) -> dict:
    """Outer limits of {value: profile cost <= c_min + delta}; a limit that reaches the edge of the parameter range is not
    set by the data. Crossing points are interpolated linearly in log value (linearly when the lower point is zero)."""
    v = np.array([r["value"] for r in rows] + [hat])
    c = np.array([r["cost"] for r in rows] + [c_min])
    o = np.argsort(v)
    v, c = v[o], c[o]
    t = c_min + delta
    under = c <= t

    def cross(a, b):
        ca, cb = c[a], c[b]
        if cb == ca:
            return float(v[a])
        w = (t - ca) / (cb - ca)
        if v[a] > 0 and v[b] > 0:
            return float(math.exp(math.log(v[a]) + w * (math.log(v[b]) - math.log(v[a]))))
        return float(v[a] + w * (v[b] - v[a]))
    ii = np.flatnonzero(under)
    if not len(ii):
        return {"delta_cost": float(delta), "kind": "no_point_below_threshold"}
    a, b = int(ii[0]), int(ii[-1])
    lower_by_data, upper_by_data = a > 0, b < len(v) - 1
    lower = cross(a, a - 1) if lower_by_data else float(v[0])
    upper = cross(b, b + 1) if upper_by_data else float(v[-1])
    pieces = int(ndimage.label(under)[1])
    kind = {(True, True): "two_sided", (False, True): "upper_only_includes_lower_edge",
            (True, False): "lower_only_reaches_upper_edge", (False, False): "unbounded"}[(lower_by_data, upper_by_data)]
    return {"delta_cost": float(delta), "lower": lower, "upper": upper, "lower_set_by_data": bool(lower_by_data),
            "upper_set_by_data": bool(upper_by_data), "kind": kind, "n_disjoint_pieces": pieces}


def run_profile(ex, gpu: str, d: TagData, mask: str, theta: np.ndarray, c_min: float, anchors: list[np.ndarray]) -> tuple[dict, np.ndarray, float]:
    """A profile that finds a lower objective than its anchor moves the anchor there (after a free polish) and restarts,
    so that every interval is measured from the lowest objective any search reached."""
    lv = d.levels(mask)
    moved = []
    for _ in range(MAX_REANCHOR + 1):
        jobs = []
        for i in FREE:
            below, above = profile_grid(i, float(theta[i]), lv)
            jobs.append((gpu, mask, i, below, above, [theta.copy(), *anchors[:3]]))
        res = pmap(ex, _profile_job, jobs)
        low = min((x for r in res for x in r["rows"]), key=lambda x: x["cost"])
        if low["cost"] >= c_min * (1 - 1e-6):
            break
        cand = _polish_job((gpu, mask, low["theta"]))
        moved.append({"from_cost": c_min, "to_cost": float(min(cand[0], low["cost"]))})
        theta, c_min = (cand[1], float(cand[0])) if cand[0] <= low["cost"] else (low["theta"], float(low["cost"]))
    inf = inference(d, mask, theta)
    ref = d.score(theta)["choice"]
    out = {"anchor_cost": c_min, "anchor_params": params_of(theta), "anchor_moved_by_profile": moved, "calibration": inf,
           "per_parameter": {}, "regret_along_profiles": {}}
    near_c, near_own, near_prim, near_ch = [], [], [], []
    for r in res:
        n = r["param"]
        i = cm.PARAMS.index(n)
        rows = r["rows"]
        hat = float(theta[i])
        lo, hi = float(cm.LO[i]), float(cm.HI[i])
        rec = {"estimate": hat}
        rec["iid_95"] = interval(rows, hat, c_min, inf["delta_cost_iid"], lo, hi)
        rec["cluster_95"] = interval(rows, hat, c_min, inf["delta_cost_cluster"][n], lo, hi)
        rec["objective_within_0p5pct"] = interval(rows, hat, c_min, 0.005 * c_min, lo, hi)
        rec["objective_within_2pct"] = interval(rows, hat, c_min, 0.02 * c_min, lo, hi)
        lowest = min(rows, key=lambda x: x["cost"])
        rec["lowest_profile_point"] = {"value": lowest["value"], "cost": lowest["cost"]}
        rec["profile"] = [[float(f"{x['value']:.6g}"), round(x["cost"], 5), round(x["own_mean_regret_pct"], 4)] for x in rows]
        out["per_parameter"][n] = rec
        for x in rows:
            near_c.append(x["cost"])
            near_own.append(x["own_mean_regret_pct"])
            near_prim.append(x["primary_mean_regret_pct"])
            near_ch.append(x["choice"])
    near_c, near_own, near_prim, near_ch = map(np.array, (near_c, near_own, near_prim, near_ch))
    out["regret_along_profiles"] = near_equal(near_c, near_own, near_prim, near_ch, ref, c_min)
    out["summary_columns"] = "iid_95, cluster_95: kind of the 95% interval; two_sided = both limits set by the data"
    out["summary"] = {n: {"iid_95": out["per_parameter"][n]["iid_95"]["kind"], "cluster_95": out["per_parameter"][n]["cluster_95"]["kind"]}
                      for n in NAMES}
    return out, theta, c_min


# =========================================================================== bootstrap
COMBOS = {"tau_m_x_W1": ("tau_m", "W1", 1), "tau_e_over_W0": ("tau_e", "W0", -1), "tau_s_over_W0": ("tau_s", "W0", -1),
          "tau_q_over_W0": ("tau_q", "W0", -1), "tau_e_over_tau_s": ("tau_e", "tau_s", -1), "tau_m_over_tau_e": ("tau_m", "tau_e", -1)}


def corr_block(TH: np.ndarray) -> dict:
    sd = TH.std(0)
    usable = [j for j in range(TH.shape[1]) if sd[j] > 0]
    P = np.full((TH.shape[1], TH.shape[1]), np.nan)
    P[np.ix_(usable, usable)] = np.corrcoef(TH[:, usable], rowvar=False)
    Sp = np.full_like(P, np.nan)
    Sp[np.ix_(usable, usable)] = spearmanr(TH[:, usable]).statistic
    floor = np.maximum(SCALE_FLOOR[FREE], 1e-30)
    LP = np.corrcoef(np.log(np.maximum(TH[:, usable], floor[usable])), rowvar=False)
    L = np.full_like(P, np.nan)
    L[np.ix_(usable, usable)] = LP

    def pairs(M):
        return [[NAMES[a], NAMES[b], round(float(M[a, b]), 3)] for a in range(len(NAMES)) for b in range(a + 1, len(NAMES))
                if np.isfinite(M[a, b]) and abs(M[a, b]) > 0.9]

    def tab(M):
        return [[None if not np.isfinite(x) else round(float(x), 3) for x in row] for row in M]
    return {"params": NAMES, "pearson": tab(P), "spearman": tab(Sp), "pearson_log_params": tab(L),
            "pairs_abs_pearson_above_0p9": pairs(P), "pairs_abs_spearman_above_0p9": pairs(Sp),
            "pairs_abs_pearson_log_above_0p9": pairs(L)}


def boot_summary(TH: np.ndarray, reg: np.ndarray, ch: np.ndarray, stored_ch: np.ndarray, best_ch: np.ndarray, levels) -> dict:
    n = len(TH)
    flags = [bound_flags(np.array([*_expand(t)]), levels) for t in TH]
    share = {}
    for j, m in enumerate(NAMES):
        share[m] = {"at_lower_bound": float(np.mean([m in f["at_lower_bound"] for f in flags])),
                    "at_upper_bound": float(np.mean([m in f["at_upper_bound"] for f in flags])),
                    "on_kink": float(np.mean([m in f["on_kink"] for f in flags])),
                    "beyond_highest_W_level": float(np.mean([m in f["beyond_highest_W_level"] for f in flags]))}
    any_bound = float(np.mean([bool(f["at_lower_bound"] or f["at_upper_bound"]) for f in flags]))
    floor = np.maximum(SCALE_FLOOR[FREE], 1e-30)
    logs = np.log10(np.maximum(TH, floor))
    spread = {m: float(np.percentile(logs[:, j], 97.5) - np.percentile(logs[:, j], 2.5)) for j, m in enumerate(NAMES)}
    comb = {}
    for k, (a, b, sgn) in COMBOS.items():
        v = logs[:, NAMES.index(a)] + sgn * logs[:, NAMES.index(b)]
        comb[k] = {"median": float(10 ** np.median(v)), "width_95_decades": float(np.percentile(v, 97.5) - np.percentile(v, 2.5))}
    return {"n_replicates": n, "correlation": corr_block(TH),
            "params_median_ci95": {m: {"median": float(np.median(TH[:, j])), "ci95": [float(np.percentile(TH[:, j], 2.5)),
                                                                                       float(np.percentile(TH[:, j], 97.5))]}
                                   for j, m in enumerate(NAMES)},
            "share_on_bounds": share, "share_replicates_any_parameter_on_a_bound": any_bound,
            "width_95_decades_single_parameter": spread, "combinations": comb,
            "own_mean_regret_pct": {**dist(reg), "ci95": [float(np.percentile(reg, 2.5)), float(np.percentile(reg, 97.5))]},
            "share_same_choice_as_reported_fit_median": float(np.median((ch == stored_ch).mean(1))),
            "share_same_choice_as_best_fit_median": float(np.median((ch == best_ch).mean(1))),
            "n_distinct_choice_vectors": len({c.tobytes() for c in ch})}


def _expand(free_values: np.ndarray) -> np.ndarray:
    th = np.array([cm.DEFAULT_FIXED.get(i, 0.0) for i in range(len(cm.PARAMS))], float)
    th[FREE] = free_values
    return th


def run_bootstrap(ex, gpu: str, d: TagData, n_boot: int, anchors: list[np.ndarray], reported_ch, best_ch, in_dir: Path) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    draws = [rng.integers(len(d.shapes), size=len(d.shapes)) for _ in range(n_boot)]
    res = pmap(ex, _boot_job, [(gpu, dr, anchors[:3]) for dr in draws])
    TH = np.array([r["theta"] for r in res])
    TS = np.array([r["theta_strong"] for r in res])
    reg, regs = np.array([r["mean"] for r in res]), np.array([r["mean_strong"] for r in res])
    ch, chs = np.array([r["choice"] for r in res]), np.array([r["choice_strong"] for r in res])
    lv = d.levels("all")
    out = {"clusters": [f"L{a}|b{b}" for a, b in d.shapes], "seed": BOOT_SEED,
           "n_distinct_keys_per_replicate": dist([r["n_distinct"] for r in res]),
           "stored_protocol": boot_summary(TH, reg, ch, reported_ch, best_ch, lv),
           "stronger_optimizer": boot_summary(TS, regs, chs, reported_ch, best_ch, lv)}
    # the first 200 replicates against the stored intervals of Table S (taco_model_baselines.json, 200 replicates)
    st = json.loads((in_dir / STORED_R1).read_text())["per_tag"][d.tag]["M3c"]["cluster_bootstrap"]
    n0 = min(BOOT_STORED, n_boot)
    chk = {}
    for j, m in enumerate(NAMES):
        ci = [float(np.percentile(TH[:n0, j], 2.5)), float(np.percentile(TH[:n0, j], 97.5))]
        chk[m] = {"stored_ci95": st["params"][m]["ci95"], "recomputed_ci95": ci}
    out["first_200_replicates_against_stored"] = {
        "params": chk, "own_mean_regret_stored_ci95": st["captured_mean_regret_pct"]["ci95"],
        "own_mean_regret_recomputed_ci95": [float(np.percentile(reg[:n0], 2.5)), float(np.percentile(reg[:n0], 97.5))]}
    return out


# =========================================================================== leave one key out
def run_loko(ex, gpu: str, d: TagData, n_starts: int, in_dir: Path) -> dict:
    keys = sorted(d.ctx.points.loc[d.ctx.points["src"] == "e4_static", "key"].unique())
    rows = pmap(ex, _loko_job, [(gpu, k, n_starts, LOKO_SEED + j) for j, k in enumerate(keys)])
    out = {"n_keys": len(keys), "per_key": rows, "n_random_starts_per_fold": n_starts}
    stored = json.loads((in_dir / STORED).read_text())["M4c"]["device_time"][gpu]["leave_one_key_out_full_form"]
    out["stored_mean_chunk_regret_pct"] = stored["mean_chunk_regret_pct"]
    for v in ("warm_from_full_fit", "cold_from_X0", "best_of_starts"):
        r = np.array([x[v]["chunk_regret_pct"] for x in rows])
        cap = np.array([x[v]["captured_own_mean_regret_pct"] for x in rows])
        out[v] = {"mean_chunk_regret_pct": float(r.mean()), "max_chunk_regret_pct": float(r.max()),
                  "se_chunk_regret_pct": float(r.std(ddof=1) / math.sqrt(len(r))),
                  "n_keys_regret_above_1pct": int((r > 1).sum()),
                  "heldout_rmse_log_mean": float(np.mean([x[v]["heldout_rmse_log"] for x in rows])),
                  "captured_own_mean_regret_pct_over_folds": dist(cap)}
    return out


# =========================================================================== target-calibration forms
def run_forms(ex, gpu: str, d: TagData, n_starts: int, seed: int) -> dict:
    form = TARGET_FORM[gpu]
    fixed, x0 = r4.form_specs(d.ctx.theta)[form]
    th3, info = TMB.fit_fast(d.F, d.y, x0=x0, fixed=fixed)
    s3 = d.score(th3)
    res = pmap(ex, _start_job, [(gpu, "all", x, fixed) for x in random_starts(n_starts, seed, fixed)])
    costs = np.array([r[0] for r in res])
    own = np.array([r[2]["own_mean_regret_pct"] for r in res])
    prim = np.array([r[2]["primary_mean_regret_pct"] for r in res])
    ch = np.array([r[2]["choice"] for r in res])
    b = int(np.argmin(costs))
    best_cost = min(float(costs[b]), info["cost"])
    return {"form": form, "n_free_params": len(cm.PARAMS) - len(fixed), "three_restart": {"cost": info["cost"], **{k: v for k, v in s3.items() if k != "choice"}},
            "n_starts": n_starts, "best_of_starts": {"cost": float(costs[b]), "own_mean_regret_pct": float(own[b]),
                                                     "primary_mean_regret_pct": float(prim[b]),
                                                     "share_same_choice_as_three_restart": float((ch[b] == s3["choice"]).mean())},
            "near_equal": near_equal(costs, own, prim, ch, ch[b], best_cost)}


# =========================================================================== Task 2: physical reading
def partitioned_k(src: pd.DataFrame, f: pd.DataFrame, hw: dict) -> tuple[dict, np.ndarray]:
    """Eq. (3) inputs when the registers of an SM are divided among its four schedulers, as the profiler's occupancy limit
    implies on the RTX 3090; the features module divides them per SM. L1-miss shares are left unchanged."""
    alloc = np.ceil(f["regs"].to_numpy(float) / 8.0) * 8.0 * 32.0
    nw = f["nw"].to_numpy(float)
    k = np.maximum(1.0, np.minimum.reduce([4.0 * ((hw["regs_per_sm"] // 4) // alloc) // nw, hw["max_warps_per_sm"] // nw,
                                           np.full_like(nw, float(hw["max_blocks_per_sm"]))]))
    P = np.ceil(src["d"].to_numpy(float) / src["BD"].to_numpy(float)) * src["B"].to_numpy(float)
    F = dict(TMB.farr(f))
    F["waves"], F["m"] = np.ceil(P / (hw["n_sm"] * k)), np.minimum(k, np.ceil(P / hw["n_sm"]))
    return F, k


def term_parts(th: np.ndarray, F: dict) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """The terms of one tile before hiding and sharing (ms), with the hiding and sharing factors and the resident warps."""
    t0, tm, te, ts, tq, tp, W0, W1 = th[FREE]
    N = cm.D_STATE
    lg = np.log2(np.clip(F["lanes_c"], 1, None))
    W = F["m"] * F["nw"]
    parts = {"tau_m": N * tm * np.ones_like(W), "tau_e": N * te * F["e"], "tau_s": N * ts * F["e_d"] * lg, "tau_q": N * tq * F["spill_words"],
             "tau_p": N * tp * F["spill_words"] * F["l1_miss"]}
    return parts, np.minimum(1.0, W1 / W), np.maximum(1.0, W / W0), W


def counter_table(d: TagData, in_dir: Path, fits: dict) -> tuple[pd.DataFrame, dict]:
    """One row per Triton-scan counter cell (fresh placement) with the model features and the fitted parts."""
    src = json.loads((in_dir / "e35_counters_4c_e35.json").read_text())
    rows = []
    for r in src["rows"]:
        if r["family"] != "m1" or r["placement"] != "fresh":
            continue
        bd, nw = cm.DEFAULT_LAUNCH[r["chunk"]] if r["launch"] == "default" else (4, 2)
        rows.append({"L": r["L"], "B": r["batch"], "d": 2048, "C": r["chunk"], "BD": bd, "nw": nw, "ns": 1, "launch": r["launch"],
                     "dur_us": r["gpu__time_duration.sum"] / 1e3, "f_ghz": r["sm__cycles_elapsed.avg.per_second"] / 1e9,
                     "warps_per_sm": r["sm__warps_active.avg.pct_of_peak_sustained_active"] / 100 * 48,
                     "warps_per_scheduler": r["smsp__warps_active.avg.per_cycle_active"],
                     "warps_limit_ncu": r["sm__maximum_warps_avg_per_active_cycle"],
                     "k_ncu": r["launch__occupancy_limit_registers"], "waves_ncu": r["launch__waves_per_multiprocessor"],
                     "regs_ncu": r["launch__registers_per_thread"], "grid": r["launch__grid_size"], "block": r["launch__block_size"],
                     "issue_pct": r["smsp__issue_active.avg.pct_of_peak_sustained_active"], "inst": r["smsp__inst_executed.sum"],
                     "local_ld": r["l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"], "local_st": r["l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"],
                     "long_sb": r["smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"]})
    df = pd.DataFrame(rows)
    f = cm.features(df, d.ctx.stats, d.ctx.hw)
    if len(f) != len(df):
        raise RuntimeError("counter cells without compile facts")
    for k in ("n_tiles", "e", "e_d", "lanes_c", "spill_words", "l1_miss", "k", "waves", "m", "regs"):
        df[k] = f[k].to_numpy(float)
    df["W_model"] = df["m"] * df["nw"]
    df["inst_per_warp_tile"] = df["inst"] / (df["grid"] * df["nw"] * df["n_tiles"])
    df["spill_bytes_per_thread_tile"] = (df["local_ld"] + df["local_st"]) * 32.0 / (df["grid"] * df["block"] * df["n_tiles"])
    df["compile_spill_bytes"] = df["spill_words"] * 4.0
    # share of resident warp cycles stalled on the long scoreboard: stalls per issue-active cycle x issue-active share / warps per scheduler
    df["long_wait_share"] = df["long_sb"] * df["issue_pct"] / 100.0 / df["warps_per_scheduler"]
    Fa = TMB.farr(f)
    Fk, k_part = partitioned_k(df, f, d.ctx.hw)
    df["k_partitioned"] = k_part
    for name, th in fits.items():
        parts, hide, share, W = term_parts(th, Fa)
        A, I = parts["tau_m"], parts["tau_e"] + parts["tau_s"] + parts["tau_q"] + parts["tau_p"]
        df[f"Ah_ms_{name}"], df[f"Is_ms_{name}"], df[f"I_ms_{name}"] = A * hide, I * share, I
        df[f"pred_us_{name}"] = 1e3 * TMB.predict_np(th, Fa)
        df[f"pred_k_us_{name}"] = 1e3 * TMB.predict_np(th, Fk)
    return df, Fk


def spearman(a, b) -> float:
    return float(spearmanr(a, b).statistic)


def gpu_term_scale(d: TagData, name_to_theta: dict) -> dict:
    """Per GPU, without counters: where W0 and W1 sit among the resident-warp levels, the cycle scale of the time constants
    and each term's share of the predicted tile time over the calibration timings."""
    lo_f, hi_f = SM_CLOCK_GHZ[d.gpu]
    spill = d.F["spill_words"] > 0
    Wc = d.Fc["m"] * d.Fc["nw"]
    out = {"sm_clock_range_ghz_manuscript": [lo_f, hi_f], "W_levels_calibration": d.levels("all"),
           "W_levels_captured_candidates": sorted({float(x) for x in Wc}),
           "compile_spill_bytes_store_plus_load_calibration_range": [float(4.0 * d.F["spill_words"][spill].min()), float(4.0 * d.F["spill_words"][spill].max())],
           "compile_spill_bytes_store_plus_load_captured_range": [float(4.0 * d.Fc["spill_words"][d.Fc["spill_words"] > 0].min()), float(4.0 * d.Fc["spill_words"].max())],
           "n_calibration_timings_with_spill": int(spill.sum()), "n_captured_candidates_with_spill": int((d.Fc["spill_words"] > 0).sum()),
           "n_captured_candidates": int(len(Wc)), "n_calibration_timings_with_l1_miss": int((d.F["l1_miss"] > 0).sum()),
           "n_captured_candidates_with_l1_miss": int((d.Fc["l1_miss"] > 0).sum()), "fits": {}}
    for n, th in name_to_theta.items():
        parts, hide, share, W = term_parts(th, d.F)
        tile = parts["tau_m"] * hide + (parts["tau_e"] + parts["tau_s"] + parts["tau_q"] + parts["tau_p"]) * share
        ix = {m: cm.PARAMS.index(m) for m in NAMES}
        shares = {"tau_m_after_hiding": parts["tau_m"] * hide / tile}
        for m in ("tau_e", "tau_s", "tau_q", "tau_p"):
            shares[m] = parts[m] * share / tile
        out["fits"][n] = {"W0": float(th[I_W0]), "W1": float(th[I_W1]), "share_calibration_W_above_W0": float((W > th[I_W0]).mean()),
                          "share_calibration_W_above_W1": float((W > th[I_W1]).mean()),
                          "share_captured_candidates_W_above_W0": float((Wc > th[I_W0]).mean()),
                          "share_captured_candidates_W_above_W1": float((Wc > th[I_W1]).mean()),
                          "tau_m_cycles_range": [1e6 * th[ix["tau_m"]] * lo_f, 1e6 * th[ix["tau_m"]] * hi_f],
                          "tau_e_cycles_range_per_feature_element": [1e6 * th[ix["tau_e"]] * lo_f, 1e6 * th[ix["tau_e"]] * hi_f],
                          "tau_s_cycles_range_per_feature_element": [1e6 * th[ix["tau_s"]] * lo_f, 1e6 * th[ix["tau_s"]] * hi_f],
                          "tau_q_cycles_range_per_compile_word": [1e6 * th[ix["tau_q"]] * lo_f, 1e6 * th[ix["tau_q"]] * hi_f],
                          "median_share_of_predicted_tile_time": {m: float(np.median(v)) for m, v in shares.items()},
                          "median_share_where_spilling": {m: float(np.median(v[spill])) for m, v in shares.items()} if spill.any() else None}
    return out


def physical_reading(d: TagData, in_dir: Path, fits: dict, n_starts: int = 50) -> dict:
    df, _ = counter_table(d, in_dir, fits)
    dflt = df[df["launch"] == "default"].reset_index(drop=True)
    pts = d.ctx.points
    st = pts[pts["src"] == "e4_static"]
    timed = {(int(r.L), int(r.B), int(r.C)): float(r.lat) for r in st.itertuples()}
    ratio = np.array([1e-3 * r.dur_us / timed[(int(r.L), int(r.B), int(r.C))] for r in dflt.itertuples()])
    out = {"counter_source": {"file": "e35_counters_4c_e35.json", "board": "stock RTX 3090, kernel replay at natural clocks, Nsight "
                                                                       "Compute; the calibration tag was timed on another RTX 3090 host",
                              "n_default_table_cells": int(len(dflt)), "n_shapes": 12,
                              "SM_clock_ghz_in_cells": [float(df["f_ghz"].min()), float(df["f_ghz"].max())],
                              "counter_duration_over_calibration_static_timing": dist(ratio)}}
    # occupancy: resident programs, warps and waves
    k_mis = dflt[dflt["k"] != dflt["k_ncu"]]
    per = []
    for (b, c), g in dflt.groupby(["B", "C"]):
        gg = g[g["L"] > 128]
        per.append({"batch": int(b), "chunk": int(c), "regs": float(g["regs"].iloc[0]), "k_model": float(g["k"].iloc[0]), "k_ncu": float(g["k_ncu"].iloc[0]),
                    "W_model": float(g["W_model"].iloc[0]), "warps_limit_ncu": float(g["warps_limit_ncu"].iloc[0]),
                    "warps_per_sm_measured": [float(gg["warps_per_sm"].min()), float(gg["warps_per_sm"].max())],
                    "issue_active_pct": [float(gg["issue_pct"].min()), float(gg["issue_pct"].max())],
                    "long_scoreboard_per_issue": [float(gg["long_sb"].min()), float(gg["long_sb"].max())],
                    "long_wait_share": [float(gg["long_wait_share"].min()), float(gg["long_wait_share"].max())],
                    "waves_model": float(g["waves"].iloc[0]), "waves_ncu": float(g["waves_ncu"].iloc[0])})
    fall, fcall = d.ctx.f, d.ctx.calls.fc
    _, k_c = partitioned_k(d.ctx.points, fall, d.ctx.hw)
    Fck, k_cc = partitioned_k(d.ctx.calls.cand, fcall, d.ctx.hw)
    out["occupancy"] = {"per_batch_and_chunk_excluding_L128": per,
                        "n_cells_k_model_differs_from_ncu_limit": int(len(k_mis)), "regs_of_those_cells": sorted({float(x) for x in k_mis["regs"]}),
                        "k_model_vs_ncu_of_those_cells": sorted({(float(a), float(b)) for a, b in zip(k_mis["k"], k_mis["k_ncu"])}),
                        "n_cells_k_partitioned_per_scheduler_equals_ncu_limit": int((df["k_partitioned"] == df["k_ncu"]).sum()), "n_cells_all_launches": int(len(df)),
                        "share_cells_W_model_above_ncu_warp_limit": float((dflt["W_model"] > dflt["warps_limit_ncu"]).mean()),
                        "W_model_over_measured_warps_per_sm": dist(dflt["W_model"] / dflt["warps_per_sm"]),
                        "spearman_W_model_measured_warps": spearman(dflt["W_model"], dflt["warps_per_sm"]),
                        "spearman_W_model_measured_warps_with_partitioned_k": spearman(np.minimum(dflt["k_partitioned"], np.ceil(dflt["grid"] / d.ctx.hw["n_sm"])) * dflt["nw"], dflt["warps_per_sm"]),
                        "waves_model_over_ncu": dist(dflt["waves"] / np.maximum(dflt["waves_ncu"], 1.0)),
                        "n_calibration_timings_k_differs_from_partitioned": int((fall["k"].to_numpy(float) != k_c).sum()),
                        "n_captured_candidates_k_differs_from_partitioned": int((fcall["k"].to_numpy(float) != k_cc).sum()),
                        "note": "measured warps per SM = achieved occupancy x 48; the profiler reports fractional waves, a single partial wave as below 1"}
    # issue-active level against measured warps (issue saturation, W0)
    gg = dflt[dflt["L"] > 128]
    tab = []
    for a, b in [(0, 7), (7, 9), (12, 17), (21, 24), (26, 30)]:
        s = gg[(gg["warps_per_sm"] >= a) & (gg["warps_per_sm"] < b)]
        if len(s):
            tab.append({"warps_per_sm_range": [a, b], "n_cells": int(len(s)), "issue_active_pct": dist(s["issue_pct"])})
    out["issue_active_by_measured_warps"] = tab
    out["peak_issue_active_pct"] = float(dflt["issue_pct"].max())
    # model-free latency part: chunks 32, 64, 128 at batch 1 share 256 programs, two warps each and one partial wave
    sel = dflt[(dflt["B"] == 1) & (dflt["C"] <= 128) & (dflt["L"] > 128)]

    def regress(values, g):
        slope, icpt = np.polyfit(g["inst_per_warp_tile"].to_numpy(float), values, 1)
        return float(icpt), float(1e3 * slope), float(1e3 * slope * g["f_ghz"].mean())
    reg = []
    for L, g in sel.groupby("L"):
        ic, sl, cy = regress((g["dur_us"] / g["n_tiles"]).to_numpy(float), g)
        reg.append({"L": int(L), "intercept_us_per_tile": ic, "slope_ns_per_warp_instruction": sl, "slope_cycles_per_warp_instruction": cy, "n_points": int(len(g))})
    out["counter_only_latency_regression"] = {"definition": "time per tile against warp instructions per warp and tile over chunks 32, 64, 128 at batch 1 "
                                                            "(256 programs, measured 6.2 warps per SM, one partial wave); three points per length",
                                              "per_length": reg, "intercept_us_per_tile_mean": float(np.mean([x["intercept_us_per_tile"] for x in reg])),
                                              "intercept_us_per_tile_range": [min(x["intercept_us_per_tile"] for x in reg), max(x["intercept_us_per_tile"] for x in reg)],
                                              "slope_cycles_per_warp_instruction_mean": float(np.mean([x["slope_cycles_per_warp_instruction"] for x in reg])),
                                              "slope_cycles_range": [min(x["slope_cycles_per_warp_instruction"] for x in reg), max(x["slope_cycles_per_warp_instruction"] for x in reg)]}
    # checks per fit
    out["fits"] = {}
    for name, th in fits.items():
        Ah, Is, I = dflt[f"Ah_ms_{name}"], dflt[f"Is_ms_{name}"], dflt[f"I_ms_{name}"]
        tile = Ah + Is
        W0, W1 = th[I_W0], th[I_W1]
        share_A = Ah / tile
        pred_issue = 100 * (Is / tile) * np.minimum(1, dflt["W_model"] / W0)
        # a warp issues at most one instruction per cycle and an SM's four schedulers at most four
        cyc = I * 1e6 * dflt["f_ghz"] / dflt["inst_per_warp_tile"]
        thru = cyc * 4 / W0
        hi_w = dflt["warps_per_sm"] > 20
        at_or_above = gg["warps_per_sm"] >= W0
        spill_share = cm.D_STATE * dflt["spill_words"] * th[cm.PARAMS.index("tau_q")] / I
        ix = cm.PARAMS.index
        mreg = []
        for L, g in sel.groupby("L"):
            ic, sl, cy = regress(1e3 * (g[f"Ah_ms_{name}"] + g[f"Is_ms_{name}"]).to_numpy(float), g)
            mreg.append({"L": int(L), "intercept_us_per_tile": ic, "slope_cycles_per_warp_instruction": cy})
        out["fits"][name] = {
            "params": {"t0_us": 1e3 * th[0], "tau_m_us": 1e3 * th[ix("tau_m")], "tau_e_ns": 1e6 * th[ix("tau_e")], "tau_s_ns": 1e6 * th[ix("tau_s")],
                       "tau_q_ns": 1e6 * th[ix("tau_q")], "tau_p_ns": 1e6 * th[ix("tau_p")], "W0": float(W0), "W1": float(W1)},
            "share_W_model_above_W0": float((dflt["W_model"] > W0).mean()), "share_W_model_above_W1": float((dflt["W_model"] > W1).mean()),
            "pred_over_counter_duration": dist(dflt[f"pred_us_{name}"] / dflt["dur_us"]),
            "latency_part_share": dist(share_A), "measured_long_wait_share": dist(dflt["long_wait_share"]),
            "spearman_latency_share_vs_measured_long_wait_share": spearman(share_A, dflt["long_wait_share"]),
            "spearman_latency_share_vs_measured_long_wait_share_batch_1": spearman(share_A[dflt["B"] == 1], dflt.loc[dflt["B"] == 1, "long_wait_share"]),
            "spearman_latency_share_vs_one_minus_issue_active": spearman(share_A, 100 - dflt["issue_pct"]),
            "implied_issue_active_pct_minus_measured": dist(pred_issue - dflt["issue_pct"]),
            "spearman_implied_vs_measured_issue_active": spearman(pred_issue, dflt["issue_pct"]),
            "issue_cycles_per_warp_instruction_without_sharing": dist(cyc),
            "n_cells_issue_cycles_per_warp_instruction_below_1": int((cyc < 1).sum()),
            "I_over_W0_x_4f_over_instructions": dist(thru),
            "I_over_W0_x_4f_over_instructions_above_20_warps": dist(thru[hi_w]),
            "n_cells_I_over_W0_x_4f_below_1": int((thru < 1).sum()),
            "issue_active_pct_where_measured_warps_at_least_W0": dist(gg.loc[at_or_above, "issue_pct"]),
            "regression_of_fitted_tile_time_on_same_cells": {"per_length": mreg, "intercept_us_per_tile_mean": float(np.mean([x["intercept_us_per_tile"] for x in mreg])),
                                                             "slope_cycles_per_warp_instruction_mean": float(np.mean([x["slope_cycles_per_warp_instruction"] for x in mreg]))},
            "tile_latency_part_at_W8_us": float(1e3 * cm.D_STATE * th[ix("tau_m")] * min(1.0, W1 / 8.0)),
            "tau_e_cycles_per_feature_element": dist(1e6 * th[ix("tau_e")] * dflt["f_ghz"]),
            "tau_m_cycles": dist(1e6 * th[ix("tau_m")] * dflt["f_ghz"]),
            "spill_share_of_issue_part_chunk_1024": dist(spill_share[dflt["C"] == 1024]),
            "spill_time_per_tile_us_chunk_1024": float(1e3 * cm.D_STATE * dflt.loc[dflt["C"] == 1024, "spill_words"].iloc[0] * th[ix("tau_q")]),
            "tau_q_ns_per_word_if_traffic_is_the_measured_one": float(1e6 * cm.D_STATE * th[ix("tau_q")])}
    # intervention: chunk 128 at the default entry (BD 8) and at (BD 4, nw 2), which doubles the grid and the resident warps
    a = df[(df["C"] == 128) & (df["launch"] == "default")].set_index(["L", "B"])
    b = df[(df["C"] == 128) & (df["launch"] == "d4_w2")].set_index(["L", "B"])
    iv = []
    for key in a.index:
        rec = {"L": int(key[0]), "batch": int(key[1]), "measured_ratio_d4w2_over_default": float(b.loc[key, "dur_us"] / a.loc[key, "dur_us"]),
               "W_model": [float(a.loc[key, "W_model"]), float(b.loc[key, "W_model"])], "waves_model": [float(a.loc[key, "waves"]), float(b.loc[key, "waves"])],
               "waves_ncu": [float(a.loc[key, "waves_ncu"]), float(b.loc[key, "waves_ncu"])],
               "warps_per_sm_measured": [float(a.loc[key, "warps_per_sm"]), float(b.loc[key, "warps_per_sm"])],
               "inst_ratio_d4w2_over_default": float(b.loc[key, "inst"] / a.loc[key, "inst"])}
        for n in fits:
            rec[f"pred_ratio_{n}"] = float(b.loc[key, f"pred_us_{n}"] / a.loc[key, f"pred_us_{n}"])
            rec[f"pred_ratio_partitioned_k_{n}"] = float(b.loc[key, f"pred_k_us_{n}"] / a.loc[key, f"pred_k_us_{n}"])
        iv.append(rec)
    meas = np.array([x["measured_ratio_d4w2_over_default"] for x in iv])
    bat = np.array([x["batch"] for x in iv])
    blk = {"cells": iv, "measured_ratio": dist(meas), "by_batch": {}}
    for bv in sorted(set(bat.tolist())):
        blk["by_batch"][str(bv)] = {"measured_ratio_range": [float(meas[bat == bv].min()), float(meas[bat == bv].max())]}
    for n in fits:
        for pre in ("pred_ratio_", "pred_ratio_partitioned_k_"):
            pr = np.array([x[f"{pre}{n}"] for x in iv])
            blk[f"{pre}{n}"] = dist(pr)
            blk[f"n_cells_direction_correct_{pre}{n}"] = int(((pr < 1) == (meas < 1)).sum())
            blk[f"mean_abs_error_points_{pre}{n}"] = float(100 * np.abs(pr - meas).mean())
            for bv in sorted(set(bat.tolist())):
                blk["by_batch"][str(bv)][f"{pre}{n}_range"] = [float(pr[bat == bv].min()), float(pr[bat == bv].max())]
    out["launch_intervention_chunk_128"] = blk
    # spill traffic against the compile facts
    sp = dflt[dflt["compile_spill_bytes"] > 0]
    ns = dflt[dflt["compile_spill_bytes"] == 0]
    out["spill"] = {"n_cells_compile_reports_spill": int(len(sp)), "chunks_with_spill": sorted({int(c) for c in sp["C"]}),
                    "n_nonspilling_cells_with_local_traffic": int((ns["local_ld"] + ns["local_st"] > 0).sum()),
                    "compile_spill_bytes_per_thread_store_plus_load": sorted({float(x) for x in sp["compile_spill_bytes"]}),
                    "measured_local_bytes_per_thread_and_tile_load_plus_store": dist(sp["spill_bytes_per_thread_tile"]),
                    "model_traffic_bytes_per_thread_and_tile": float(cm.D_STATE * sp["compile_spill_bytes"].iloc[0]),
                    "model_over_measured_traffic": dist(cm.D_STATE * sp["compile_spill_bytes"] / sp["spill_bytes_per_thread_tile"]),
                    "local_warp_instructions_share_of_all_instructions": dist((sp["local_ld"] + sp["local_st"]) / 4.0 / sp["inst"]),
                    "note": "bytes = sectors x 32; a 4-byte access by 32 threads touches 4 sectors, so sectors / 4 approximates warp instructions"}
    F = d.F
    out["tau_p_exposure"] = {"n_calibration_timings_with_l1_miss": int((F["l1_miss"] > 0).sum()), "n_calibration_timings": int(len(d.y)),
                             "n_full_protocol_timings_with_l1_miss": int((TMB.sub(F, d.masks["full_protocol_only"])["l1_miss"] > 0).sum()),
                             "n_captured_candidates_with_l1_miss": int((d.Fc["l1_miss"] > 0).sum()), "n_captured_candidates": int(len(d.Fc["l1_miss"])),
                             "n_counter_cells_with_l1_miss": int((df["l1_miss"] > 0).sum())}
    return out


def partitioned_k_refit(d: TagData, n_starts: int = 50) -> dict:
    """The fit protocol, the multi-start and the leave-one-key-out on inputs with registers divided per scheduler. The
    division is validated by the counter cells of the RTX 3090 alone; for the other GPUs it is an untested assumption."""
    Fk, k_new = partitioned_k(d.ctx.points, d.ctx.f, d.ctx.hw)
    Fck, k_cand = partitioned_k(d.ctx.calls.cand, d.ctx.calls.fc, d.ctx.hw)
    th3, i3 = TMB.fit_fast(Fk, d.y)
    s3 = d.score(th3, Fck)
    res = []
    for x in random_starts(n_starts, START_SEED):
        t, c = TMB.fit_single(Fk, d.y, x)
        res.append((float(c), t, d.score(t, Fck)))
    costs = np.array([r[0] for r in res])
    own = np.array([r[2]["own_mean_regret_pct"] for r in res])
    prim = np.array([r[2]["primary_mean_regret_pct"] for r in res])
    ch = np.array([r[2]["choice"] for r in res])
    b = int(np.argmin(costs))
    th_b, c_b = (res[b][1], float(costs[b])) if costs[b] < i3["cost"] else (th3, float(i3["cost"]))
    sb = d.score(th_b, Fck)
    pts, kk, Cc = d.ctx.points, d.ctx.points["key"].to_numpy(), d.ctx.points["C"].to_numpy(int)
    L, B, dd = pts["L"].to_numpy(), pts["B"].to_numpy(), pts["d"].to_numpy()
    fold = []
    for key in sorted(pts.loc[pts["src"] == "e4_static", "key"].unique()):
        i0 = int(np.flatnonzero(kk == key)[0])
        hold = (L == L[i0]) & (B == B[i0]) & (dd == dd[i0])
        st = (kk == key) & (d.src == "e4_static")
        th, _ = TMB.fit_fast(TMB.sub(Fk, ~hold), d.y[~hold])
        meas = dict(zip(Cc[st], d.y[st]))
        c_pred = int(Cc[st][int(np.argmin(TMB.predict_np(th, TMB.sub(Fk, st))))])
        fold.append(100 * (meas[c_pred] / min(meas.values()) - 1))
    return {"n_calibration_timings_k_changes": int((d.ctx.f["k"].to_numpy(float) != k_new).sum()), "n_captured_candidates_k_changes": int((d.ctx.calls.fc["k"].to_numpy(float) != k_cand).sum()),
            "three_restart": {"cost": float(i3["cost"]), "params": params_of(th3), **{k: v for k, v in s3.items() if k != "choice"}},
            "lowest_objective": {"cost": c_b, "params": params_of(th_b), **{k: v for k, v in sb.items() if k != "choice"}},
            "three_restart_rel_excess_over_lowest_objective": float(i3["cost"] / c_b - 1), "n_starts": n_starts,
            "near_equal_among_starts": near_equal(costs, own, prim, ch, sb["choice"], c_b),
            "leave_one_key_out_cold_from_X0": {"mean_chunk_regret_pct": float(np.mean(fold)), "max_chunk_regret_pct": float(np.max(fold)), "se_chunk_regret_pct": float(np.std(fold, ddof=1) / math.sqrt(len(fold))),
                                               "per_key_regret_pct": [float(x) for x in fold]},
            "validated_by_counters": d.gpu == "vgpu3090"}


def _within(a: float, b: float, factor: float) -> bool:
    return a > 0 and b > 0 and max(a / b, b / a) <= factor


def term_verdicts(ph: dict, per_gpu: dict, t0_ci_ms: list[float]) -> dict:
    """Per term and GPU: consistent, inconsistent or cannot be checked, with the numbers; every verdict for the RTX 3090 follows
    from the comparison stated next to it. The other two GPUs have no counters, so nothing there can be checked term by term."""
    occ, sp, reg = ph["occupancy"], ph["spill"], ph["counter_only_latency_regression"]
    fr, fb = ph["fits"]["three_restart"], ph["fits"]["best_objective"]
    both = (fr, fb)
    v = {}
    v["t0"] = {"verdict": "cannot be checked", "numbers": {"fitted_us": [x["params"]["t0_us"] for x in both], "stored_bootstrap_ci95_us": [1e3 * x for x in t0_ci_ms]},
               "reason": "no counter isolates a constant term of device time"}
    icpt = reg["intercept_us_per_tile_mean"]
    a_w8 = [x["tile_latency_part_at_W8_us"] for x in both]
    a_reg = [x["regression_of_fitted_tile_time_on_same_cells"]["intercept_us_per_tile_mean"] for x in both]
    rank = [x["spearman_latency_share_vs_measured_long_wait_share"] for x in both]
    rank1 = [x["spearman_latency_share_vs_measured_long_wait_share_batch_1"] for x in both]
    level_ok = all(_within(a, icpt, 2.0) for a in a_w8 + a_reg)
    v["tau_m"] = {"verdict": f"{'consistent' if level_ok else 'inconsistent'} in level for the latency part of a tile at eight resident warps; "
                             f"{'consistent' if min(rank1) >= 0.7 else 'inconsistent'} in rank across chunks at batch 1; "
                             f"{'consistent' if min(rank) >= 0.7 else 'inconsistent'} in rank across all cells; tau_m alone is not determined",
                  "numbers": {"tau_m_us": [x["params"]["tau_m_us"] for x in both], "latency_part_at_W8_us": a_w8, "intercept_of_fitted_tile_time_on_same_cells_us": a_reg,
                              "counter_only_intercept_us_per_tile": icpt, "counter_only_intercept_range": reg["intercept_us_per_tile_range"],
                              "spearman_latency_share_vs_measured_long_wait_share_all_cells": rank, "same_at_batch_1": rank1,
                              "fitted_latency_share_range": [[x["latency_part_share"]["min"], x["latency_part_share"]["max"]] for x in both],
                              "measured_long_wait_share_range": [fr["measured_long_wait_share"]["min"], fr["measured_long_wait_share"]["max"]],
                              "tau_m_cycles_median": [x["tau_m_cycles"]["median"] for x in both]},
                  "rule": "level: counter intercept within a factor 2 of both fitted values, for both fits; rank: rank correlation at least 0.7, for both fits",
                  "limit": "the repository has no memory-latency measurement, so the absolute value per state index cannot be compared with one"}
    cy = [x["regression_of_fitted_tile_time_on_same_cells"]["slope_cycles_per_warp_instruction_mean"] for x in both]
    cm_ = reg["slope_cycles_per_warp_instruction_mean"]
    below = [x["n_cells_issue_cycles_per_warp_instruction_below_1"] for x in both]
    ok = all(_within(c, cm_, 1.25) for c in cy) and max(below) == 0
    v["tau_e_tau_s"] = {"verdict": ("consistent" if ok else "inconsistent") + " for the issue part at eight resident warps, I max(1, W/W0); tau_e and tau_s are not separately determined",
                        "numbers": {"tau_e_ns": [x["params"]["tau_e_ns"] for x in both], "tau_s_ns": [x["params"]["tau_s_ns"] for x in both],
                                    "fitted_slope_cycles_per_warp_instruction": cy, "counter_slope_cycles_per_warp_instruction": cm_, "counter_slope_range": reg["slope_cycles_range"],
                                    "unshared_cycles_per_warp_instruction_median": [x["issue_cycles_per_warp_instruction_without_sharing"]["median"] for x in both],
                                    "n_cells_below_one_cycle_per_warp_instruction": below,
                                    "tau_e_cycles_per_feature_element_median": [x["tau_e_cycles_per_feature_element"]["median"] for x in both]},
                        "rule": "fitted slope within 25% of the counter slope for both fits, and no cell below one cycle per warp instruction",
                        "limit": "tau_e is per element of the feature e, not per instruction, so its cycle count is not an instruction latency"}
    ratio = sp["model_over_measured_traffic"]["median"]
    v["tau_q"] = {"verdict": "inconsistent as a cost per spilled word" if ratio > 1.5 else "consistent",
                  "numbers": {"compile_spill_bytes_store_plus_load": sp["compile_spill_bytes_per_thread_store_plus_load"],
                              "measured_bytes_per_thread_and_tile": sp["measured_local_bytes_per_thread_and_tile_load_plus_store"]["median"],
                              "model_bytes_per_thread_and_tile": sp["model_traffic_bytes_per_thread_and_tile"], "model_over_measured": ratio,
                              "local_warp_instruction_share": sp["local_warp_instructions_share_of_all_instructions"]["median"],
                              "tau_q_ns": [x["params"]["tau_q_ns"] for x in both], "tau_q_ns_per_measured_word": [x["tau_q_ns_per_word_if_traffic_is_the_measured_one"] for x in both],
                              "spill_share_of_issue_part_chunk_1024": [x["spill_share_of_issue_part_chunk_1024"]["median"] for x in both]},
                  "rule": "the term charges the compile-time spill size for each of the 16 state indices; the counters on the only spilling cells (chunk 1024) show it once per tile",
                  "limit": "one spill size on one GPU; no spilling non-default configuration has counters"}
    v["tau_p"] = {"verdict": "cannot be checked", "numbers": {**ph["tau_p_exposure"], "tau_p_ns": [x["params"]["tau_p_ns"] for x in both]},
                  "reason": "no counter cell and no captured candidate has an L1-miss share above zero, so the term never acts on a chunk choice on the captured calls"}
    above = [x["issue_active_pct_where_measured_warps_at_least_W0"]["median"] for x in both]
    pk = ph["peak_issue_active_pct"]
    th_hi = [x["I_over_W0_x_4f_over_instructions_above_20_warps"]["median"] for x in both]
    v["W0"] = {"verdict": ("inconsistent" if max(above) < 90 else "consistent") + " as the point where the issue slots saturate; "
                          + ("consistent" if all(_within(t, 100.0 / pk, 1.15) for t in th_hi) else "inconsistent") + " as the ratio I / W0 at the highest occupancy",
               "numbers": {"W0": [x["params"]["W0"] for x in both], "median_issue_active_pct_at_measured_warps_at_least_W0": above, "peak_issue_active_pct": pk,
                           "one_over_peak_issue_active": 100.0 / pk, "I_over_W0_x_4f_over_instructions_above_20_warps": th_hi,
                           "issue_active_by_measured_warps": [{"warps": x["warps_per_sm_range"], "median_pct": x["issue_active_pct"]["median"]} for x in ph["issue_active_by_measured_warps"]]},
               "rule": "issue-active share at or above W0 at least 90% for saturation; I / W0 x 4f / instructions within 15% of 100 / peak issue-active for the ratio"}
    iv = ph["launch_intervention_chunk_128"]
    v["W1"] = {"verdict": "cannot be checked directly; the one intervention on resident warps is not reproduced",
               "numbers": {"W1": [x["params"]["W1"] for x in both], "W1_on_kink": [ph["three_restart_flags"]["on_kink"], ph["best_flags"]["on_kink"]], "by_batch": iv["by_batch"],
                           "mean_abs_error_points": {k: iv[k] for k in iv if k.startswith("mean_abs_error_points")},
                           "n_cells_direction_correct_of_12": {k: iv[k] for k in iv if k.startswith("n_cells_direction_correct")}},
               "reason": "no counter measures hidden latency; the chunk-128 intervention also changes the instruction count, so it is not a clean test of hiding either"}
    v["resident_warps_W_input"] = {"verdict": "inconsistent for one register class (a compile-derived input, not a fitted value)",
                                   "numbers": {"n_default_cells": occ["n_cells_k_model_differs_from_ncu_limit"], "regs": occ["regs_of_those_cells"], "k_model_vs_ncu": occ["k_model_vs_ncu_of_those_cells"],
                                               "share_cells_W_model_above_ncu_warp_limit": occ["share_cells_W_model_above_ncu_warp_limit"], "W_model_over_measured": occ["W_model_over_measured_warps_per_sm"],
                                               "n_cells_partitioned_k_equals_ncu": occ["n_cells_k_partitioned_per_scheduler_equals_ncu_limit"], "n_cells": occ["n_cells_all_launches"],
                                               "n_calibration_timings_affected": occ["n_calibration_timings_k_differs_from_partitioned"], "n_captured_candidates_affected": occ["n_captured_candidates_k_differs_from_partitioned"],
                                               "refit_three_restart": {k: ph["refit_with_partitioned_k"]["vgpu3090"]["three_restart"][k] for k in ("cost", "own_mean_regret_pct")},
                                               "refit_leave_one_key_out_mean_chunk_regret_pct": ph["refit_with_partitioned_k"]["vgpu3090"]["leave_one_key_out_cold_from_X0"]["mean_chunk_regret_pct"]},
                                   "reason": "at 194 registers the features allow 5 resident programs per SM; the profiler reports a limit of 4 in all those cells"}
    out = {"rtx3090": v}
    for g, label in (("pro6000", "rtx_pro_6000"), ("h800", "h800")):
        r = per_gpu[g]["fits"]
        out[label] = {t: {"verdict": "cannot be checked", "reason": "the repository has no counters for this GPU"} for t in ("t0", "tau_m", "tau_e_tau_s", "tau_q", "tau_p", "W0", "W1", "resident_warps_W_input")}
        out[label]["tau_p"]["numbers"] = {k: per_gpu[g][k] for k in ("n_calibration_timings_with_l1_miss", "n_captured_candidates_with_l1_miss")}
        out[label]["W0"]["numbers"] = {n: {"W0": x["W0"], "share_captured_candidates_W_above_W0": x["share_captured_candidates_W_above_W0"]} for n, x in r.items()}
        out[label]["W1"]["numbers"] = {n: {"W1": x["W1"], "share_captured_candidates_W_above_W1": x["share_captured_candidates_W_above_W1"]} for n, x in r.items()}
        out[label]["tau_m"]["numbers"] = {n: {"cycles": x["tau_m_cycles_range"]} for n, x in r.items()}
        out[label]["tau_e_tau_s"]["numbers"] = {n: {"tau_e_cycles_per_feature_element": x["tau_e_cycles_range_per_feature_element"]} for n, x in r.items()}
    return out


# =========================================================================== Task 3: M1 evidence
def _at(obj, path: str):
    cur = obj
    for p in path.split("/"):
        cur = cur[p]
    return cur


def m1_evidence(in_dir: Path, per_gpu: dict) -> tuple[list[dict], list[dict]]:
    m4 = json.loads((in_dir / STORED).read_text())
    r1 = json.loads((in_dir / STORED_R1).read_text())
    rows, mism = [], []

    def add(statement, value_in_paper, file, path, tol):
        obj = m4 if file == STORED else r1
        v = float(_at(obj, path))
        ok = abs(v - value_in_paper) <= tol
        rec = {"statement": statement, "manuscript_value": value_in_paper, "file": file, "json_path": path, "file_value": v, "match": bool(ok)}
        rows.append(rec)
        if not ok:
            mism.append(rec)
    dev = "M4c/device_time"
    for g, q in zip(GPUS, (0.8, 7.4, 3.7)):
        add(f"leave-one-key-out chunk regret, {g}", q, STORED, f"{dev}/{g}/leave_one_key_out_full_form/mean_chunk_regret_pct", 0.05)
    for g, q in zip(GPUS, (0.73, 4.28, 6.05)):
        own = OWN_TABLE[g]
        add(f"full-protocol-only refit regret, {g} (taco_r2 rerun)", q, STORED, f"{dev}/{g}/variants/full_protocol_only/eval_{own}/mean_regret_pct", 0.005)
        add(f"full-protocol-only refit regret, {g} (taco_r1 run)", q, STORED_R1, f"per_tag/{TAGS[g]}/M3c/full_protocol_only/captured_calls/mean_regret_pct", 0.005)
    for g, q in zip(GPUS, (0.56, 1.8, 6.1)):
        add(f"static-72 refit regret, {g} (taco_r2 rerun)", q, STORED, f"{dev}/{g}/variants/static72_only/eval_{OWN_TABLE[g]}/mean_regret_pct", 0.05)
        add(f"static-72 refit regret, {g} (taco_r1 run)", q, STORED_R1, f"per_tag/{TAGS[g]}/M5b/static72_only_fit/captured_calls/mean_regret_pct", 0.05)
    add("RTX 3090 full-protocol optimum selecting with 1.6 (taco_r2 rerun)", 1.6, STORED, f"{dev}/vgpu3090/variants/full_protocol_only/eval_vgpu3090_device_one_block/mean_regret_pct", 0.05)
    add("RTX 3090 full-protocol optimum selecting with 4.3 (taco_r1 run)", 4.3, STORED_R1, "per_tag/vgpu3090_r3spin/M3c/full_protocol_only/captured_calls/mean_regret_pct", 0.05)
    for g in GPUS:
        add(f"search timings of 2298, {g}", 2208, STORED, f"{dev}/{g}/n_points_by_source/e4_search", 0)
    add("50 random starts reach a 16% lower objective, RTX 3090", 0.16, STORED, f"{dev}/vgpu3090/multistart/best_cost_rel_improvement_over_stored_protocol", 0.005)
    for g, q in zip(("pro6000", "h800"), (0.0, 0.0)):
        add(f"50 random starts reach the stored optimum, {g} (relative improvement)", q, STORED, f"{dev}/{g}/multistart/best_cost_rel_improvement_over_stored_protocol", 1e-6)
    tc = "ablation12+work_only|target_calibration_loso|rmse_tiebreak"
    for g, q, q2 in zip(GPUS, (0.58, 0.85, 2.14), (2.32, 0.86, 3.14)):
        add(f"target-calibration per-call regret, {g} (primary table)", q, STORED, f"M4a/device_time/summary/{g}/rows/{tc}/per_call_mean_pct", 0.005)
        add(f"target-calibration latency-weighted share, {g} (primary table)", q2, STORED, f"M4a/device_time/summary/{g}/rows/{tc}/lw_share_of_oracle_pct", 0.005)
    meta = {g: {"exclusive_guard": _at(m4, f"{dev}/{g}/calibration_sweep_metadata/exclusive_guard"),
                "shuffle_or_seed_argument_present": _at(m4, f"{dev}/{g}/calibration_sweep_metadata/shuffle_or_seed_argument_present"),
                "identical_configuration_order_for_every_key": _at(m4, f"{dev}/{g}/calibration_sweep_metadata/identical_configuration_order_for_every_key"),
                "key_order_in_file": _at(m4, f"{dev}/{g}/calibration_sweep_metadata/key_order_in_file"),
                "search_warmup_repeats": [_at(m4, f"{dev}/{g}/calibration_sweep_metadata/search_warmup"), _at(m4, f"{dev}/{g}/calibration_sweep_metadata/search_repeats")],
                "json_path": f"{STORED}: {dev}/{g}/calibration_sweep_metadata"} for g in GPUS}
    return rows, mism, meta


# =========================================================================== Task 4: E3 cost figures
def _per_key(tag: str, in_dir: Path) -> dict:
    rows = read_jsonl(in_dir / f"e4_{tag}.jsonl")
    m1 = [r for r in rows if r.get("kind") == "m1tune"]
    scale = TMB.e5b._tuning_scale(rows)
    c_v = m1[0]["tuning_first_call_s"] / max(m1[0].get("n_configs", 1), 1)
    warm = float(sum(r["tuning_bench_s"] for r in m1))
    cold = float(sum(r["tuning_wall_s"] for r in m1))
    comp = float(sum(r["tuning_first_call_s"] for r in m1))
    sum_static = float(sum(sum(v["mean_ms"] for k, v in r["final"].items() if k.startswith("static_")) for r in m1))
    df = load_e2([in_dir / f"e2_mamba1_{tag}.csv"])
    df["aligned"] = (df["L"] % 16 == 0).astype(int)
    lat = latency_matrix(df, TMB.CHUNKS)
    n_spec = int(df["aligned"].nunique())
    out = {"tag": tag, "calibration_warm_s": warm, "calibration_cold_s": cold, "calibration_compile_s": comp,
           "compile_share_of_cold": comp / cold, "timing_share_of_cold": warm / cold,
           "compile_s_per_variant_first_key": float(c_v), "profile_s_per_ms_of_summed_chunk_latency": scale["profile_s_per_ms"],
           "static72_timing_wall_s_estimate_110_runs_per_chunk": 0.110 * sum_static, "key_sets": {}}
    for mode, cols in (("exact_L_batch", ["L", "batch"]), ("bucket_batch_alignment", ["L_bucket", "batch", "aligned"])):
        P = np.array([scale["profile_s_per_ms"] * float(lat[np.asarray(idx)].mean(0).sum()) for idx in df.groupby(cols).indices.values()])
        out["key_sets"][mode] = {
            "n_keys": int(len(P)), "sum_s": float(P.sum()), "with_compile_s": float(P.sum() + len(TMB.CHUNKS) * n_spec * c_v),
            "mean_ms_per_key": float(1e3 * P.mean()), "median_ms_per_key": float(1e3 * np.median(P)),
            "min_ms_per_key": float(1e3 * P.min()), "max_ms_per_key": float(1e3 * P.max()),
            "payback_keys_warm_calibration_over_mean_key": float(warm / P.mean()),
            "payback_keys_warm_calibration_over_median_key": float(warm / np.median(P))}
    return out


def cost_figures(in_dir: Path) -> dict:
    li = {g: _per_key(LI_TAGS[g], in_dir) for g in GPUS}
    dv = {g: _per_key(TAGS[g], in_dir) for g in GPUS}
    st = json.loads((in_dir / STORED_R1).read_text())["M5b"]["captured_workload_costs"]
    chk = {}
    for g in GPUS:
        for mode, nm in (("exact_L_batch", "exact"), ("bucket_batch_alignment", "bucket_aligned")):
            a, b = li[g]["key_sets"][mode], st[g]["modes"][nm]
            chk[f"{g}/{mode}"] = {"sum_s_abs_diff": abs(a["sum_s"] - b["sum_P_profile_s"]),
                                  "payback_abs_diff": abs(a["payback_keys_warm_calibration_over_mean_key"] - b["payback_keys"]["full_calibration_warm"])}
    ex, bu = "exact_L_batch", "bucket_batch_alignment"
    rng = lambda key, mode, table=li: [min(table[g]["key_sets"][mode][key] for g in GPUS), max(table[g]["key_sets"][mode][key] for g in GPUS)]  # noqa: E731
    paths = {g: {"calibration_warm_s": f"{STORED_R1}: M5b/captured_workload_costs/{g}/one_off_costs/full_calibration_warm_s",
                 "calibration_cold_s": f"{STORED_R1}: M5b/captured_workload_costs/{g}/one_off_costs/full_calibration_cold_s",
                 "profile_57_keys_warm_s": f"{STORED_R1}: M5b/captured_workload_costs/{g}/modes/exact/sum_P_profile_s",
                 "profile_57_keys_with_compile_s": f"{STORED_R1}: M5b/captured_workload_costs/{g}/modes/exact/strategies/profile_every_key/cold_s",
                 "mean_key_57_s": f"{STORED_R1}: M5b/captured_workload_costs/{g}/modes/exact/mean_P_profile_s",
                 "per_key_57_s": f"{STORED_R1}: M5b/captured_workload_costs/{g}/modes/exact/P_per_key_s",
                 "per_key_33_s": f"{STORED_R1}: M5b/captured_workload_costs/{g}/modes/bucket_aligned/P_per_key_s",
                 "payback_57": f"{STORED_R1}: M5b/captured_workload_costs/{g}/modes/exact/payback_keys/full_calibration_warm",
                 "payback_33": f"{STORED_R1}: M5b/captured_workload_costs/{g}/modes/bucket_aligned/payback_keys/full_calibration_warm"} for g in GPUS}
    e5b = {}
    for g in ("pro6000", "vgpu3090"):
        k = json.loads((in_dir / f"e5b_{g}.json").read_text())["keys"]
        p, s = [x["P_s"]["profile"] for x in k], [x["P_s"]["search"] for x in k]
        e5b[g] = {"file": f"e5b_{g}.json", "n_keys": len(k), "median_profile_s": float(np.median(p)), "sum_profile_s": float(np.sum(p)),
                  "mean_profile_s": float(np.mean(p)), "median_search_s": float(np.median(s))}
    pb_ex = rng("payback_keys_warm_calibration_over_mean_key", ex)
    pb_bu = rng("payback_keys_warm_calibration_over_mean_key", bu)
    h800_lo = li["h800"]["key_sets"][bu]["payback_keys_warm_calibration_over_mean_key"]
    res = {"launch_inclusive_tags_used_by_the_manuscript": li, "device_time_tags": dv, "check_against_stored": chk, "json_paths": paths,
           "supplement_33_keys_e5b": e5b,
           "per_key_summary_launch_inclusive": {
               "exact_57_keys": {"mean_ms_range": rng("mean_ms_per_key", ex), "median_ms_range": rng("median_ms_per_key", ex),
                                 "min_max_ms_over_keys": [rng("min_ms_per_key", ex)[0], rng("max_ms_per_key", ex)[1]]},
               "bucket_33_keys": {"mean_ms_range": rng("mean_ms_per_key", bu), "median_ms_range": rng("median_ms_per_key", bu),
                                  "min_max_ms_over_keys": [rng("min_ms_per_key", bu)[0], rng("max_ms_per_key", bu)[1]]}},
           "payback": {"definition": "warm calibration seconds / mean per-key profiling seconds of the key set (taco_model_baselines.workload_costs); "
                                     "the profiling seconds are scaled (seconds per ms of the key's summed six-chunk latency, fitted over the 12 E4 keys; "
                                     "3 warm-ups and 10 repeats per chunk, launch-inclusive), not timed per key",
                       "exact_57_keys_range": pb_ex, "bucket_33_keys_range": pb_bu,
                       "union_of_both_key_sets": [min(pb_ex[0], pb_bu[0]), max(pb_ex[1], pb_bu[1])],
                       "lower_bound_1600_source": {"value": h800_lo, "gpu": "h800", "key_set": "33 bucket keys",
                                                   "json_path": paths["h800"]["payback_33"]},
                       "upper_bound_2500_source": {"value": li["vgpu3090"]["key_sets"][ex]["payback_keys_warm_calibration_over_mean_key"], "gpu": "vgpu3090",
                                                   "key_set": "57 exact keys", "json_path": paths["vgpu3090"]["payback_57"]}}}
    return res


# =========================================================================== one GPU
def run_gpu(ex, gpu: str, args, in_dir: Path) -> dict:
    t0 = time.time()
    d = _data(gpu)
    stored = d.ctx.theta.copy()
    stored_cost = TMB.loss_cost(np.log(TMB.predict_np(stored, d.F)) - np.log(d.y))
    stored_s = d.score(stored)
    out = {"tag": d.tag}
    report_all = {"cost": stored_cost, "choice": stored_s["choice"]}
    # --- multi-start, 2298 timings
    ms, th_ms, c_ms, ths, cs = run_multistart(ex, gpu, d, "all", args.n_starts, report_all)
    anchors = anchors_from(ths, cs, stored)
    grid, th_g, c_g = run_grid(ex, gpu, d, "all", anchors, report_all)
    th_best, c_best = (th_g, c_g) if c_g < c_ms else (th_ms, c_ms)
    best_s = d.score(th_best)
    three = {"cost": stored_cost, "params": params_of(stored), "flags": bound_flags(stored, d.levels("all")),
             **{k: v for k, v in stored_s.items() if k != "choice"}}
    best_rec = fit_record(d, th_best, c_best, "all", stored_s["choice"])
    out["fits_2298"] = {"three_restart_reported": three, "best_objective": best_rec,
                        "best_objective_rel_improvement_over_three_restart": float(1 - c_best / stored_cost),
                        "best_objective_found_by": "w0_w1_grid" if c_g < c_ms else "multistart"}
    out["multistart_2298"] = ms
    out["w0_w1_grid_2298"] = grid
    print(f"  {gpu}: multistart and grid (2298) done, {time.time() - t0:.0f} s", flush=True)
    # --- 90 full-protocol timings: the three-restart protocol on this host, multi-start, grid
    Fm, ym = d.subset("full_protocol_only")
    th3, i3 = TMB.fit_fast(Fm, ym)
    report_fp = {"cost": i3["cost"], "choice": d.score(th3)["choice"]}
    ms90, th_m90, c_m90, th90, c90 = run_multistart(ex, gpu, d, "full_protocol_only", args.n_starts, report_fp)
    grid90, th_g90, c_g90 = run_grid(ex, gpu, d, "full_protocol_only", anchors_from(np.vstack([th90, th3[None, :]]), np.r_[c90, i3["cost"]], stored),
                                     report_fp)
    th_b90, c_b90 = (th_g90, c_g90) if c_g90 < c_m90 else (th_m90, c_m90)
    if i3["cost"] < c_b90:
        th_b90, c_b90 = th3, float(i3["cost"])
    Fs, ys = d.subset("static72_only")
    th72, i72 = TMB.fit_fast(Fs, ys)
    out["fits_90_full_protocol"] = {"three_restart_this_host": fit_record(d, th3, i3["cost"], "full_protocol_only"),
                                    "best_objective": fit_record(d, th_b90, c_b90, "full_protocol_only"),
                                    "n_timings_with_l1_miss": int((Fm["l1_miss"] > 0).sum())}
    out["fits_72_static"] = {"three_restart_this_host": fit_record(d, th72, i72["cost"], "static72_only")}
    out["multistart_90"] = ms90
    out["w0_w1_grid_90"] = grid90
    print(f"  {gpu}: multistart and grid (90) done, {time.time() - t0:.0f} s", flush=True)
    # --- profiles (a lower objective found here replaces the best fit), then the Jacobian
    prof, th_p, c_p = run_profile(ex, gpu, d, "all", th_best, c_best, anchors_from(ths, cs, stored))
    out["profile_2298"] = prof
    if c_p < c_best:
        out["fits_2298"]["best_objective_found_by"] = "profile"
        th_best, c_best = th_p, c_p
        best_s = d.score(th_best)
        out["fits_2298"]["best_objective"] = fit_record(d, th_best, c_best, "all", stored_s["choice"])
        out["fits_2298"]["best_objective_rel_improvement_over_three_restart"] = float(1 - c_best / stored_cost)
    prof90, th_p90, c_p90 = run_profile(ex, gpu, d, "full_protocol_only", th_b90, c_b90, anchors_from(th90, c90, stored))
    out["profile_90"] = prof90
    if c_p90 < c_b90:
        th_b90, c_b90 = th_p90, c_p90
        out["fits_90_full_protocol"]["best_objective"] = fit_record(d, th_b90, c_b90, "full_protocol_only")
    for rec, th_x, c_x in ((ms, th_best, c_best), (ms90, th_b90, c_b90)):
        a_c, a_o, a_p, a_ch = rec["_arrays"]
        rec["near_equal_relative_to_lowest_objective_of_any_search"] = near_equal(a_c, a_o, a_p, a_ch, d.score(th_x)["choice"], c_x)
    print(f"  {gpu}: profiles done, {time.time() - t0:.0f} s", flush=True)
    Jn, _ = log_jacobian(th_best, d.F)
    fd = np.empty_like(Jn)
    lb = np.log(TMB.predict_np(th_best, d.F))
    for j, i in enumerate(FREE):
        h = max(abs(th_best[i]), SCALE_FLOOR[i]) * 1e-6
        tp_ = th_best.copy()
        tp_[i] += h
        fd[:, j] = (np.log(TMB.predict_np(tp_, d.F)) - lb) / h
    out["jacobian"] = {"analytic_vs_finite_difference_max_abs_scaled_diff": float(np.abs((Jn - fd) * param_scale(th_best)).max()),
                       "three_restart_reported": jacobian_report(stored, d.F, d.y, d.levels("all")),
                       "best_objective": jacobian_report(th_best, d.F, d.y, d.levels("all"))}
    # --- bootstrap, leave one key out, target-calibration form
    out["cluster_bootstrap"] = run_bootstrap(ex, gpu, d, args.n_boot, anchors_from(ths, cs, stored), stored_s["choice"], best_s["choice"], in_dir)
    print(f"  {gpu}: bootstrap done, {time.time() - t0:.0f} s", flush=True)
    out["leave_one_key_out"] = run_loko(ex, gpu, d, args.loko_starts, in_dir)
    out["target_calibration_form"] = run_forms(ex, gpu, d, args.form_starts, FORM_SEED + GPUS.index(gpu))
    out["seconds"] = round(time.time() - t0, 1)
    out["_best_theta"] = th_best
    return out


# =========================================================================== main
def reproduce(gpu: str, in_dir: Path) -> dict:
    """Stored parameters and a three-restart refit from X0 against the stored regret; stops on a mismatch."""
    d = _data(gpu)
    st = json.loads((in_dir / STORED).read_text())["M4c"]["device_time"][gpu]["variants"]["stored_soft_l1"]
    want_own, want_prim = st[f"eval_{OWN_TABLE[gpu]}"]["mean_regret_pct"], st[f"eval_{PRIMARY_TABLE[gpu]}"]["mean_regret_pct"]
    s = d.score(d.ctx.theta)
    th, info = TMB.fit_fast(d.F, d.y)
    s2 = d.score(th)
    stored_fit = d.ctx.stored["fit"]["cost"]
    rec = {"stored_cost_file": stored_fit, "stored_params_own_mean_regret_pct": s["own_mean_regret_pct"],
           "stored_params_primary_mean_regret_pct": s["primary_mean_regret_pct"], "file_own_mean_regret_pct": want_own,
           "file_primary_mean_regret_pct": want_prim, "refit_from_X0_cost": info["cost"], "refit_from_X0_own_mean_regret_pct": s2["own_mean_regret_pct"],
           "refit_same_choices_as_stored": bool((s["choice"] == s2["choice"]).all()),
           "max_abs_rel_param_diff_refit_vs_stored_interior": float(max(abs(th[i] / d.ctx.theta[i] - 1) for i in FREE if d.ctx.theta[i] > SCALE_FLOOR[i]))}
    ok = abs(s["own_mean_regret_pct"] - want_own) < 1e-9 and abs(s["primary_mean_regret_pct"] - want_prim) < 1e-9 \
        and abs(s2["own_mean_regret_pct"] - want_own) < 1e-9 and abs(info["cost"] / stored_fit - 1) < 1e-6
    rec["reproduced"] = bool(ok)
    if not ok:
        raise SystemExit(f"{gpu}: the stored regret could not be reproduced, stopping: {rec}")
    return rec


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--n-starts", type=int, default=300, help="random starts per GPU and data set (at least 50)")
    p.add_argument("--n-boot", type=int, default=1000, help="key-cluster bootstrap replicates")
    p.add_argument("--loko-starts", type=int, default=50, help="random starts per leave-one-key-out fold")
    p.add_argument("--form-starts", type=int, default=100, help="random starts of each target-calibration form")
    args = p.parse_args(argv)
    t0 = time.time()
    _init(str(args.in_dir))
    repro = {g: reproduce(g, args.in_dir) for g in GPUS}
    for g, r in repro.items():
        print(f"{g}: stored regret reproduced: own table {r['stored_params_own_mean_regret_pct']:.4f}, primary table "
              f"{r['stored_params_primary_mean_regret_pct']:.4f}, refit cost {r['refit_from_X0_cost']:.6f}", flush=True)
    res = {"experiment": "JPDC M2/M1/E3 cost-model identifiability", "definitions": __doc__,
           "settings": {"n_starts": args.n_starts, "n_boot": args.n_boot, "loko_starts": args.loko_starts, "form_starts": args.form_starts,
                        "start_seed": START_SEED, "boot_seed": BOOT_SEED, "loko_seed_base": LOKO_SEED, "form_seed_base": FORM_SEED,
                        "lower_bound_rule": "time constant below 1e-3 X0; W0, W1 at most 1.001", "upper_bound_rule": "at least 0.99 of the upper bound",
                        "kink_rule": "W0 or W1 within 1e-4 (relative) of a resident-warp level of the data",
                        "note_on_reruns": "the small refits stop in different points across hosts and library versions, so three-restart values on "
                                          "the 90 and 72 timings are given for this host next to the stored runs"},
           "reproduction": repro, "per_gpu": {}}
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init, initargs=(str(args.in_dir),)) as ex:
        for g in GPUS:
            print(f"{g} ...", flush=True)
            res["per_gpu"][g] = run_gpu(ex, g, args, args.in_dir)
    # Task 2
    d = _data("vgpu3090")
    best = res["per_gpu"]["vgpu3090"]["_best_theta"]
    ph = physical_reading(d, args.in_dir, {"three_restart": d.ctx.theta.copy(), "best_objective": best})
    ph["three_restart_flags"] = bound_flags(d.ctx.theta, d.levels("all"))
    ph["best_flags"] = bound_flags(best, d.levels("all"))
    ph["per_gpu_term_scale"] = {g: gpu_term_scale(_data(g), {"three_restart": _data(g).ctx.theta.copy(), "best_objective": res["per_gpu"][g]["_best_theta"]}) for g in GPUS}
    ph["refit_with_partitioned_k"] = {g: partitioned_k_refit(_data(g)) for g in GPUS}
    r1 = json.loads((args.in_dir / STORED_R1).read_text())["per_tag"]["vgpu3090_r3spin"]["M3c"]["cluster_bootstrap"]["params"]
    ph["term_verdicts"] = term_verdicts(ph, ph["per_gpu_term_scale"], r1["t0"]["ci95"])
    res["physical_reading"] = ph
    # Task 3
    rows, mism, meta = m1_evidence(args.in_dir, res["per_gpu"])
    m4 = json.loads((args.in_dir / STORED).read_text())
    r1j = json.loads((args.in_dir / STORED_R1).read_text())
    fp = res["per_gpu"]["vgpu3090"]["fits_90_full_protocol"]
    res["m1_evidence"] = {
        "rows": rows, "mismatches": mism, "sweep_metadata": meta,
        "rtx3090_full_protocol_runs": {
            "taco_r1_run": {"cost": r1j["per_tag"]["vgpu3090_r3spin"]["M3c"]["full_protocol_only"]["fit_cost"],
                            "own_mean_regret_pct": r1j["per_tag"]["vgpu3090_r3spin"]["M3c"]["full_protocol_only"]["captured_calls"]["mean_regret_pct"],
                            "json_path": f"{STORED_R1}: per_tag/vgpu3090_r3spin/M3c/full_protocol_only"},
            "taco_r2_rerun": {"cost": m4["M4c"]["device_time"]["vgpu3090"]["small_refit_multistart"]["full_protocol_only"]["three_restart_protocol_cost"],
                              "own_mean_regret_pct": m4["M4c"]["device_time"]["vgpu3090"]["variants"]["full_protocol_only"]["eval_vgpu3090_device_one_block"]["mean_regret_pct"],
                              "json_path": f"{STORED}: M4c/device_time/vgpu3090/small_refit_multistart/full_protocol_only and variants/full_protocol_only"},
            "this_host": {"cost": fp["three_restart_this_host"]["cost"], "own_mean_regret_pct": fp["three_restart_this_host"]["own_mean_regret_pct"],
                          "json_path": "per_gpu/vgpu3090/fits_90_full_protocol/three_restart_this_host"},
            "best_objective_found": {"cost": fp["best_objective"]["cost"], "own_mean_regret_pct": fp["best_objective"]["own_mean_regret_pct"],
                                     "json_path": "per_gpu/vgpu3090/fits_90_full_protocol/best_objective"}}}
    # Task 4
    res["e3_cost_figures"] = cost_figures(args.in_dir)
    res["runtime_s"] = round(time.time() - t0, 1)
    res["created_utc"] = utc_now()
    out = args.out or (args.in_dir / "jpdc_model_identifiability.json")
    write_json(out, jsonable(res))
    print(f"wrote {out} ({res['runtime_s']} s)")
    for g in GPUS:
        r = res["per_gpu"][g]
        f = r["fits_2298"]
        n = r["multistart_2298"]["near_equal"]
        print(f"{g}: reported cost {f['three_restart_reported']['cost']:.4f} regret {f['three_restart_reported']['own_mean_regret_pct']:.3f}; best cost "
              f"{f['best_objective']['cost']:.4f} regret {f['best_objective']['own_mean_regret_pct']:.3f}; within 0.5% n={n['within_0p5pct']['n_solutions']} "
              f"within 2% n={n['within_2pct']['n_solutions']}")
        print("   profile:", {k: v["cluster_95"] for k, v in r["profile_2298"]["summary"].items()})


if __name__ == "__main__":
    main()
