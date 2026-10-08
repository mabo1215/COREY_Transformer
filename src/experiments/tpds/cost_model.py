"""T1 - analytical shape-and-hardware cost model of the chunked Mamba-1 scan.

The model predicts the latency T(L, B, d, C, launch; hw) of the Triton Mamba-1
scan from the call shape, the launch configuration and a handful of per-GPU
parameters, without looking at the input values. Its structure follows the
kernel (``kernels.py``): the grid has P = ceil(d / BLOCK_D) * B programs; each
program walks ceil(L / C) time tiles; inside a tile it loads the [BLOCK_D, C]
inputs, then for each of the N state indices runs an associative scan over
the C steps of the tile and a reduction that extracts the carried state.

Compile-time facts are not fitted: the register count, the spill traffic and
the distributed layout of the tiles (elements per thread, lanes and warps
along the scan axis, load vector width) come from ptxas and the TTGIR of the
compiled kernel for the target architecture (``kernel_stats.py``). Occupancy
follows from the register count and the SM limits in ``HARDWARE``.

Per time tile of one program (time units: ms):

    latency part  A = tau_f + tau_d * miss_L2 + N * tau_m
    issue part    I = tau_l * loads + N * (tau_e * e + tau_s * e_D * log2(lanes_C)
                        + tau_x * e_D * [warps_C > 1] + spill_words * (tau_q + tau_p * miss_L1))

A collects the memory round trips that a tile waits for (the tile loads, of
which the share miss_L2 misses the L2 cache because the call's working set
exceeds it, and the per-state loads of B and C). I collects the instructions
a thread issues: loads and stores (e / vector width for each of the four
[BLOCK_D, C] tensors), the sequential part of the scan and of the state
reduction (e elements per thread), the shuffle levels across the lanes_C
lanes along the scan axis (e_D rows per thread), a cross-warp exchange
through shared memory when the scan axis spans warps_C > 1 warps, and the
spill traffic, which costs more once the spilled state of all resident
warps no longer fits in L1 (miss_L1). With W = m * num_warps resident warps
per SM, other warps hide the latency part and share the issue slots:

    T_tile = A * min(1, W1 / W) + I * max(1, W / W0)
    T      = t0 + waves * ceil(L / C) * T_tile,  bounded below by bytes / bandwidth

with waves = ceil(P / (n_SM * k)), k the resident programs per SM allowed by
registers, warps and blocks, and m = min(k, ceil(P / n_SM)).

Eight parameters per GPU are fitted by robust least squares on log latency:
t0, tau_m, tau_e, tau_s, tau_q, tau_p, W0 (warps that saturate issue) and W1
(warps that hide the tile latency). Four further terms of the general form
(tau_f per-tile fixed cost, tau_d L2 misses, tau_l load issue, tau_x
cross-warp exchange) are fixed at zero by default because adding them
changes neither the fit nor any chunk decision; ``--ablation`` shows this
(add-one) and refits the model without each of the eight mechanisms
(drop-one). The fitted model also narrows profiling: ``top-k`` profiles only
the k chunks with the lowest predicted latency per shape and keeps the
measured best.

Validation (all from existing result files; no GPU):
  V1  fit quality on the E4 launch-space sweep, the E4 static re-measurements
      and the E1 medians (R^2 on log latency, relative errors).
  V2  chunk selection without profiling: the chunk with the lowest predicted
      latency under the default launch table, for every E2 call (regret and
      tied share exactly as in E3), and leave-one-key-out over the E4 keys
      (fit on 11 keys, select the chunk and the full launch configuration of
      the 12th).
  V3  cross-GPU transfer: parameters of one GPU with the hardware constants
      and compile facts of the other.
  V4  the production pipelines. Mamba-2 SSD and Gated DeltaNet are multi-kernel
      pipelines; they are modeled at the level of work terms,

          T = (F^p + W^p)^(1/p),   W = B * L * (a + b * C) + B * ceil(L / C) * g,

      a launch floor F, a per-token cost a, an intra-chunk cost b * C (the
      chunk-local matrix products grow with C) and a per-chunk cost g of the
      inter-chunk state pass, joined by a smooth maximum. The unconstrained
      optimum is C* = sqrt(g / b), independent of the shape once the pipeline
      is work-bound, and the chunk matters only above B * L ~ F / (a + 2 sqrt(b g)).
      Mamba-2 is fitted on the E4 static re-measurements and E1 medians and
      validated on the E2 calls; Gated DeltaNet (no captured calls) on E1.

Output: ``cost_model_<host>.json`` (parameters, validation tables, per-call
choices for the key-churn analysis) and ``cost_model_curves_<host>.csv``
(measured and predicted latency per chunk for the figures).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.tpds.analysis import chunk_columns, latency_matrix, load_e2, regret_stats
from src.experiments.tpds.common import RESULTS_DIR, read_jsonl, write_json

try:
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover
    least_squares = None

# Per-GPU constants: SM count from the run metadata (gpu_sm_count), the rest from the CUDA
# occupancy limits of the compute capability and the vendor memory bandwidth.
HARDWARE = {
    "pro6000": {"gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "arch": 120, "n_sm": 188,
                "regs_per_sm": 65536, "max_warps_per_sm": 48, "max_blocks_per_sm": 32, "mem_bw_gbs": 1792.0,
                "l2_mb": 128.0, "l1_kb": 128.0},
    "vgpu3090": {"gpu": "NVIDIA GeForce RTX 3090", "arch": 86, "n_sm": 82,
                 "regs_per_sm": 65536, "max_warps_per_sm": 48, "max_blocks_per_sm": 16, "mem_bw_gbs": 936.0,
                 "l2_mb": 6.0, "l1_kb": 128.0},
}
# Occupancy limits per compute capability (CUDA occupancy calculator) and L1/shared capacity.
CC_LIMITS = {80: {"regs_per_sm": 65536, "max_warps_per_sm": 64, "max_blocks_per_sm": 32, "l1_kb": 192.0},
             86: {"regs_per_sm": 65536, "max_warps_per_sm": 48, "max_blocks_per_sm": 16, "l1_kb": 128.0},
             89: {"regs_per_sm": 65536, "max_warps_per_sm": 48, "max_blocks_per_sm": 24, "l1_kb": 128.0},
             90: {"regs_per_sm": 65536, "max_warps_per_sm": 64, "max_blocks_per_sm": 32, "l1_kb": 256.0},
             120: {"regs_per_sm": 65536, "max_warps_per_sm": 48, "max_blocks_per_sm": 32, "l1_kb": 128.0}}
# Vendor memory bandwidth (GB/s) and L2 size (MB) of data-centre GPUs, matched on the device name.
DC_MEMORY = [("H20", 4000.0, 60.0), ("H100 80GB HBM3", 3350.0, 50.0), ("H100 PCIe", 2000.0, 50.0),
             ("H800 PCIe", 2000.0, 50.0), ("H800", 3350.0, 50.0), ("A800 80GB PCIe", 1935.0, 40.0),
             ("A800", 2039.0, 40.0), ("A100-SXM4-80GB", 2039.0, 40.0), ("A100 80GB PCIe", 1935.0, 40.0),
             ("A100", 1555.0, 40.0), ("L40S", 864.0, 96.0), ("L20", 864.0, 96.0),
             ("RTX 3090", 936.0, 6.0), ("RTX 3070", 448.0, 4.0), ("RTX 4080", 716.8, 64.0)]


def hardware(host: str, in_dir: Path | None = None) -> dict:
    """Per-GPU constants: the table above, or derived from the run metadata of a new host
    (SM count and compute capability from ``e2_mamba1_<host>.json``, memory from DC_MEMORY)."""
    if host in HARDWARE:
        return HARDWARE[host]
    in_dir = in_dir or RESULTS_DIR
    meta = {}
    for name in (f"e2_mamba1_{host}.json", f"e12_tp_widths_{host}.json", f"e4_{host}.json"):
        pth = in_dir / name
        if pth.exists():
            d = json.loads(pth.read_text())
            meta = d.get("meta", d)
            if meta.get("gpu_sm_count"):
                break
    if not meta.get("gpu_sm_count"):
        raise KeyError(f"no hardware constants for host {host!r} (add it to HARDWARE or run E2/E12 there first)")
    cc = int(round(float(meta["gpu_cc"]) * 10))
    lim = CC_LIMITS.get(cc)
    mem = next(((bw, l2) for key, bw, l2 in DC_MEMORY if key.lower() in str(meta.get("gpu", "")).lower()), None)
    if lim is None or mem is None:
        raise KeyError(f"host {host!r}: unknown compute capability {cc} or memory of {meta.get('gpu')!r}")
    return {"gpu": meta["gpu"], "arch": cc, "n_sm": int(meta["gpu_sm_count"]), **lim,
            "mem_bw_gbs": mem[0], "l2_mb": mem[1], "derived_from_metadata": True}


DEFAULT_LAUNCH = {32: (8, 2), 64: (8, 2), 128: (8, 2), 256: (4, 2), 512: (4, 2), 1024: (4, 4)}
D_STATE = 16
PARAMS = ["t0", "tau_f", "tau_d", "tau_m", "tau_l", "tau_e", "tau_s", "tau_x", "tau_q", "tau_p", "W0", "W1"]
X0 = np.array([0.01, 5e-4, 5e-4, 2e-4, 2e-5, 2e-5, 2e-5, 2e-4, 1e-7, 1e-6, 8.0, 16.0])
LO = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0])
HI = np.array([1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.1, 0.01, 0.01, 64.0, 256.0])
N_SCALE = 10    # the first ten parameters are time constants (restarts rescale them together)
# terms of the general form that the default model leaves out (fixed at zero)
DEFAULT_FIXED = {1: 0.0, 2: 0.0, 4: 0.0, 7: 0.0}   # tau_f, tau_d, tau_l, tau_x
# ablation: drop one mechanism of the default model, or add one of the fixed terms back
ABLATIONS = {"drop_state_latency": {3: 0.0}, "drop_scan_work": {5: 0.0}, "drop_shuffle": {6: 0.0},
             "drop_spill": {8: 0.0, 9: 0.0}, "drop_spill_l1_miss": {9: 0.0}, "drop_issue_sharing": {10: 64.0},
             "drop_latency_hiding": {11: 1.0},
             "add_tile_fixed": {1: None}, "add_l2_miss": {2: None}, "add_load_issue": {4: None},
             "add_cross_warp": {7: None}}


def _fixed_for(change: dict) -> dict:
    """Default fixed set with one ablation applied (None frees a parameter)."""
    fx = dict(DEFAULT_FIXED)
    for i, v in change.items():
        if v is None:
            fx.pop(i, None)
        else:
            fx[i] = v
    return fx


# --------------------------------------------------------------------------- compile facts
def load_kernel_stats(in_dir: Path, arch: int) -> dict:
    p = in_dir / f"kernel_stats_sm{arch}.json"
    if p.exists():
        rows = json.loads(p.read_text())["configs"]
    else:  # partial sweep: the resumable per-configuration log
        rows = read_jsonl(in_dir / f"kernel_stats_sm{arch}.jsonl", kind="config")
    return {(bool(r["aligned"]), r["block_t"], r["block_d"], r["num_warps"], r["num_stages"]): r for r in rows}


def _stats_row(stats, aligned, C, BD, nw, ns):
    r = stats.get((aligned, C, BD, nw, ns)) or stats.get((aligned, C, BD, nw, 1 if ns == 2 else 2))
    if r is None and not aligned:
        r = stats.get((True, C, BD, nw, ns))
    return r


def features(df: pd.DataFrame, stats: dict, hw: dict) -> pd.DataFrame:
    """Shape, compile and occupancy features for rows with L, B, d, C, BD, nw, ns."""
    out = []
    for L, B, d, C, BD, nw, ns in df[["L", "B", "d", "C", "BD", "nw", "ns"]].itertuples(index=False):
        aligned = int(L) % 16 == 0
        r = _stats_row(stats, aligned, int(C), int(BD), int(nw), int(ns))
        if r is None or r.get("regs") is None or r.get("layout") is None:
            out.append(None)
            continue
        lay = r["layout"]
        spt, tpw, wpc = lay["size_per_thread"], lay["threads_per_warp"], lay["warps_per_cta"]
        cov_d, cov_c = spt[0] * tpw[0] * wpc[0], spt[1] * tpw[1] * wpc[1]
        e_d = max(1.0, BD / cov_d * spt[0])            # rows per thread (>= 1; replication otherwise)
        e_c = max(float(spt[1]), C / cov_c * spt[1])  # elements per thread along the scan axis
        e = e_d * e_c
        vec = spt[1]
        regs = int(r["regs"])
        alloc = int(math.ceil(regs / 8.0) * 8) * 32   # registers per warp (allocation granularity 256)
        k = min(hw["regs_per_sm"] // (alloc * nw), hw["max_warps_per_sm"] // nw, hw["max_blocks_per_sm"])
        k = max(int(k), 1)
        P = int(math.ceil(d / BD)) * int(B)
        waves = int(math.ceil(P / (hw["n_sm"] * k)))
        m = min(k, int(math.ceil(P / hw["n_sm"])))
        nbytes = 2.0 * B * d * L * 4 + 2.0 * 2 * B * D_STATE * L   # fp16 u, dt, z, out; B and C
        miss = max(0.0, 1.0 - hw["l2_mb"] * 2 ** 20 / nbytes)       # share of tile loads that miss L2
        spill_b = r.get("spill_store_bytes", 0)
        foot = spill_b * 32.0 * m * nw                                # spilled bytes of all resident warps
        out.append({"aligned": aligned, "n_tiles": int(math.ceil(L / C)), "e": e, "e_d": e_d,
                    "lanes_c": tpw[1], "warps_c": wpc[1], "loads": 4.0 * e / vec,
                    "spill_words": (spill_b + r.get("spill_load_bytes", 0)) / 4.0,
                    "l1_miss": max(0.0, 1.0 - hw["l1_kb"] * 1024 / foot) if foot > 0 else 0.0,
                    "regs": regs, "k": k, "waves": waves, "m": m, "nw": nw, "l2_miss": miss,
                    "t_mem_ms": nbytes / (hw["mem_bw_gbs"] * 1e6)})
    ok = [o is not None for o in out]
    f = pd.DataFrame([o for o in out if o is not None], index=df.index[ok])
    return f


def predict(theta: np.ndarray, f: pd.DataFrame) -> np.ndarray:
    t0, tf, td, tm, tl, te, ts, tx, tq, tp, W0, W1 = theta
    N = D_STATE
    A = tf + td * f["l2_miss"] + N * tm
    I = tl * f["loads"] + N * (te * f["e"] + ts * f["e_d"] * np.log2(f["lanes_c"].clip(lower=1))
                               + tx * f["e_d"] * (f["warps_c"] > 1) + f["spill_words"] * (tq + tp * f["l1_miss"]))
    W = f["m"] * f["nw"]
    T = t0 + f["waves"] * f["n_tiles"] * (A * np.minimum(1.0, W1 / W) + I * np.maximum(1.0, W / W0))
    return np.maximum(T.to_numpy(float), f["t_mem_ms"].to_numpy(float))


def fit(f: pd.DataFrame, y_ms: np.ndarray, x0=None, weights=None, fixed=None) -> tuple[np.ndarray, dict]:
    if least_squares is None:
        raise RuntimeError("scipy is required for the cost-model fit")
    y = np.log(y_ms)
    w = np.ones_like(y) if weights is None else np.sqrt(np.asarray(weights, float))
    x0 = X0 if x0 is None else x0
    fixed = DEFAULT_FIXED if fixed is None else fixed
    free = [i for i in range(len(PARAMS)) if i not in fixed]

    def full(v):
        th = np.empty(len(PARAMS))
        th[free] = v
        for i, val in fixed.items():
            th[i] = val
        return th

    def res(v):
        return w * (np.log(predict(full(v), f)) - y)

    best = None
    for scale in (1.0, 0.3, 3.0):   # a few restarts; the objective is piecewise smooth
        start = np.clip(x0 * np.r_[np.full(N_SCALE, scale), np.ones(len(PARAMS) - N_SCALE)], LO, HI)
        sol = least_squares(res, start[free], bounds=(LO[free], HI[free]), loss="soft_l1", f_scale=0.1,
                            x_scale="jac", max_nfev=5000)
        if best is None or sol.cost < best.cost:
            best = sol
    return full(best.x), {"cost": float(best.cost), "nfev": int(best.nfev), "success": bool(best.success)}


def fit_quality(pred: np.ndarray, meas: np.ndarray) -> dict:
    lp, lm = np.log(pred), np.log(meas)
    rel = np.abs(pred / meas - 1)
    return {"n": int(len(meas)), "r2_log": float(1 - ((lp - lm) ** 2).sum() / ((lm - lm.mean()) ** 2).sum()),
            "median_rel_err": float(np.median(rel)), "p90_rel_err": float(np.percentile(rel, 90))}


# --------------------------------------------------------------------------- data
def load_points(host: str, in_dir: Path) -> pd.DataFrame:
    """E4 launch-space sweep, E4 static re-measurements and E1 medians (batch 1)."""
    pts = []
    for r in read_jsonl(in_dir / f"e4_{host}.jsonl", kind="m1tune"):
        for s in r["search"]:
            if s.get("median_ms"):
                pts.append({"src": "e4_search", "key": r["key"], "L": r["L"], "B": r["batch"], "d": r["width"],
                            "C": s["block_t"], "BD": s["block_d"], "nw": s["num_warps"], "ns": s["num_stages"],
                            "lat": s["median_ms"]})
        for name, v in r["final"].items():
            if name.startswith("static_"):
                c = int(name.split("_")[1])
                bd, nw = DEFAULT_LAUNCH[c]
                pts.append({"src": "e4_static", "key": r["key"], "L": r["L"], "B": r["batch"], "d": r["width"],
                            "C": c, "BD": bd, "nw": nw, "ns": 1, "lat": v["median_ms"]})
    e1 = pd.DataFrame(read_jsonl(in_dir / f"e1_{host}.jsonl", kind="timing"))
    e1 = e1[e1["family"] == "mamba1"]
    for (L, b, c), g in e1.groupby(["L", "batch", "chunk"]):
        bd, nw = DEFAULT_LAUNCH[int(c)]
        pts.append({"src": "e1", "key": f"e1|L{L}|b{b}", "L": int(L), "B": int(b), "d": 2048, "C": int(c),
                    "BD": bd, "nw": nw, "ns": 1, "lat": float(g["median_ms"].median())})
    return pd.DataFrame(pts)


def e2_candidates(df: pd.DataFrame, chunks: list[int]) -> pd.DataFrame:
    """One row per (call, chunk) with the default launch configuration of the chunk."""
    rows = []
    for c in chunks:
        bd, nw = DEFAULT_LAUNCH[c]
        rows.append(pd.DataFrame({"call": np.arange(len(df)), "L": df["L"].to_numpy(), "B": df["batch"].to_numpy(),
                                  "d": df["width"].to_numpy(), "C": c, "BD": bd, "nw": nw, "ns": 1}))
    return pd.concat(rows, ignore_index=True)


def select_e2(theta, df, chunks, stats, hw) -> tuple[np.ndarray, np.ndarray]:
    cand = e2_candidates(df, chunks)
    f = features(cand, stats, hw)
    pred = np.full(len(cand), np.inf)
    pred[f.index] = predict(theta, f)
    P = pred.reshape(len(chunks), len(df)).T
    return P.argmin(1), P


def e2_eval(df: pd.DataFrame, chunks: list[int], choice: np.ndarray, P: np.ndarray | None = None) -> dict:
    lat = latency_matrix(df, chunks)
    reg = lat[np.arange(len(df)), choice] / lat.min(1) - 1
    tied = [set(int(c) for c in str(t).split(";") if c) for t in df["tied_chunks"].fillna("")]
    st = regret_stats(reg)
    chosen = np.array(chunks)[choice]
    st["tie_aware_accuracy"] = float(np.mean([c in t for c, t in zip(chosen, tied)]))
    st["accuracy"] = float((choice == lat.argmin(1)).mean())
    st["chosen_distribution"] = {int(k): int(v) for k, v in pd.Series(chosen).value_counts().items()}
    al = (df["L"] % 16 == 0).to_numpy()
    st["mean_regret_pct_aligned"] = float(100 * reg[al].mean()) if al.any() else None
    st["mean_regret_pct_unaligned"] = float(100 * reg[~al].mean()) if (~al).any() else None
    by_b = {}
    for b in sorted(df["batch"].unique()):
        sel = (df["batch"] == b).to_numpy()
        by_b[int(b)] = {"mean_regret_pct": float(100 * reg[sel].mean()),
                        "p95_regret_pct": float(100 * np.percentile(reg[sel], 95))}
    st["by_batch"] = by_b
    if P is not None:
        st["curve_fit"] = fit_quality(P.ravel(), lat.ravel())
    return st


# --------------------------------------------------------------------------- validation
def loko(points: pd.DataFrame, f: pd.DataFrame, stats, hw, theta_full) -> dict:
    """Leave one E4 key out: fit on the other keys and E1, choose chunk and launch config for the held-out key."""
    keys = sorted(points.loc[points["src"] == "e4_search", "key"].unique())
    rows = []
    for key in keys:
        tr = points.index[(points["key"] != key) & points.index.isin(f.index)]
        te = points[(points["key"] == key)]
        th, _ = fit(f.loc[tr], points.loc[tr, "lat"].to_numpy(), x0=theta_full)
        st = te[te["src"] == "e4_static"]
        pst = predict(th, f.loc[st.index])
        c_pred = int(st["C"].to_numpy()[pst.argmin()])
        meas = dict(zip(st["C"], st["lat"]))
        c_best = min(meas, key=meas.get)
        se = te[(te["src"] == "e4_search") & te.index.isin(f.index)]
        pse = predict(th, f.loc[se.index])
        j = int(pse.argmin())
        cfg_pred_lat = float(se["lat"].to_numpy()[j])
        cfg_best_lat = float(se["lat"].min())
        rows.append({"key": key, "chunk_pred": c_pred, "chunk_best": int(c_best),
                     "chunk_regret_pct": 100 * (meas[c_pred] / meas[c_best] - 1),
                     "config_pred": {k: int(se[k].to_numpy()[j]) for k in ("C", "BD", "nw", "ns")},
                     "config_regret_pct": 100 * (cfg_pred_lat / cfg_best_lat - 1),
                     "config_rank": int((se["lat"].to_numpy() < cfg_pred_lat).sum()) + 1,
                     "n_configs": int(len(se)),
                     "heldout_fit": fit_quality(pse, se["lat"].to_numpy())})
    cr = np.array([r["chunk_regret_pct"] for r in rows])
    fr = np.array([r["config_regret_pct"] for r in rows])
    return {"keys": rows, "chunk_mean_regret_pct": float(cr.mean()), "chunk_max_regret_pct": float(cr.max()),
            "config_mean_regret_pct": float(fr.mean()), "config_max_regret_pct": float(fr.max()),
            "config_median_rank": float(np.median([r["config_rank"] for r in rows]))}


def ablation(f, y, df, chunks, stats, hw, theta) -> dict:
    out = {}
    for name, change in ABLATIONS.items():
        fixed = _fixed_for(change)
        x0 = theta.copy()
        for i in change:
            if change[i] is None:
                x0[i] = X0[i]
        th, _ = fit(f, y, x0=x0, fixed=fixed)
        c, _ = select_e2(th, df, chunks, stats, hw)
        ev = e2_eval(df, chunks, c)
        out[name] = {"n_free_params": len(PARAMS) - len(fixed), "fit": fit_quality(predict(th, f), y),
                     "e2_mean_regret_pct": ev["mean_regret_pct"],
                     "e2_p95_regret_pct": ev["p95_regret_pct"], "e2_tie_aware_accuracy": ev["tie_aware_accuracy"]}
    return out


def topk_refinement(df, chunks, P, ks=(1, 2, 3)) -> dict:
    """Profile only the k chunks with the lowest predicted latency per (L, batch) and keep the measured best."""
    lat = latency_matrix(df, chunks)
    rank = np.argsort(P, 1)
    out = {}
    for k in ks:
        pick = np.empty(len(df), dtype=int)
        for _, g in df.groupby(["L", "batch"]):
            i = g.index.to_numpy()
            cand = rank[i[0], :k]
            pick[i] = cand[np.argmin(lat[i][:, cand].mean(0))]
        reg = lat[np.arange(len(df)), pick] / lat.min(1) - 1
        out[f"top{k}"] = {"profiled_chunks_per_shape": k, "mean_regret_pct": float(100 * reg.mean()),
                          "p95_regret_pct": float(100 * np.percentile(reg, 95))}
    pick = np.empty(len(df), dtype=int)
    for _, g in df.groupby(["L", "batch"]):
        i = g.index.to_numpy()
        pick[i] = int(np.argmin(lat[i].mean(0)))
    reg = lat[np.arange(len(df)), pick] / lat.min(1) - 1
    out["all_chunks_profiled"] = {"profiled_chunks_per_shape": len(chunks), "mean_regret_pct": float(100 * reg.mean()),
                                  "p95_regret_pct": float(100 * np.percentile(reg, 95)),
                                  "note": "exact-shape table chosen on the same calls: the floor of any shape-keyed policy"}
    return out


# --------------------------------------------------------------------------- V4: multi-kernel pipelines
PIPE_PARAMS = ["F", "a", "b", "g", "p"]


def pipe_predict(th, L, B, C):
    F, a, b, g, p = th
    work = B * L * (a + b * C) + B * np.ceil(L / C) * g
    return (F ** p + work ** p) ** (1.0 / p)


def pipe_points(host: str, in_dir: Path, family: str) -> pd.DataFrame:
    pts = []
    if family == "mamba2":
        for r in read_jsonl(in_dir / f"e4_{host}.jsonl", kind="m2tune"):
            for k, v in r["final"].items():
                if k.startswith("static_"):
                    pts.append({"src": "e4_static", "L": r["L"], "B": r["batch"], "C": int(k.split("_")[1]),
                                "lat": v["median_ms"]})
    e1 = pd.DataFrame(read_jsonl(in_dir / f"e1_{host}.jsonl", kind="timing"))
    e1 = e1[e1["family"] == family]
    for (L, b, c), g in e1.groupby(["L", "batch", "chunk"]):
        pts.append({"src": "e1", "L": int(L), "B": int(b), "C": int(c), "lat": float(g["median_ms"].median())})
    return pd.DataFrame(pts)


def pipe_fit(d: pd.DataFrame):
    y = np.log(d["lat"].to_numpy())
    L, B, C = (d[k].to_numpy(float) for k in ("L", "B", "C"))

    def res(th):
        return np.log(pipe_predict(th, L, B, C)) - y

    sol = least_squares(res, [float(np.exp(y.min())), 1e-6, 1e-8, 1e-4, 4.0], bounds=([0, 0, 0, 0, 1], [10, 1, 1, 1, 20]),
                        loss="soft_l1", f_scale=0.1, x_scale="jac", max_nfev=20000)
    return sol.x, float(1 - (res(sol.x) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def pipeline_model(host: str, in_dir: Path, family: str) -> dict | None:
    d = pipe_points(host, in_dir, family)
    if d.empty:
        return None
    th, r2 = pipe_fit(d)
    F, a, b, g, p = th
    c_star = float(np.sqrt(g / b)) if b > 0 else float("inf")
    out = {"family": family, "params": dict(zip(PIPE_PARAMS, map(float, th))), "n_points": int(len(d)),
           "fit_r2_log": r2, "unconstrained_optimal_chunk": c_star,
           "work_bound_above_BL_tokens": float(F / (a + 2 * np.sqrt(b * g))) if b > 0 else None}
    if family == "mamba2":
        df = load_e2([in_dir / f"e2_mamba2_{host}.csv"])
        if not df.empty:
            chunks = chunk_columns(df)
            lat = latency_matrix(df, chunks)
            P = np.stack([pipe_predict(th, df["L"].to_numpy(float), df["batch"].to_numpy(float), c) for c in chunks], 1)
            out["e2"] = e2_eval(df, chunks, P.argmin(1), P)
    return out


def analyze(host: str, in_dir: Path, other: str | None = None, do_loko: bool = True,
            do_ablation: bool = False) -> dict:
    hw = hardware(host, in_dir)
    stats = load_kernel_stats(in_dir, hw["arch"])
    points = load_points(host, in_dir)
    f = features(points, stats, hw)
    fitted = points.loc[f.index]
    theta, info = fit(f, fitted["lat"].to_numpy())
    pred = predict(theta, f)
    v1 = {"all": fit_quality(pred, fitted["lat"].to_numpy())}
    for src in ("e4_search", "e4_static", "e1"):
        sel = (fitted["src"] == src).to_numpy()
        v1[src] = fit_quality(pred[sel], fitted["lat"].to_numpy()[sel])
    df = load_e2([in_dir / f"e2_mamba1_{host}.csv"])
    chunks = chunk_columns(df)
    choice, P = select_e2(theta, df, chunks, stats, hw)
    res = {"experiment": "T1-cost-model", "host": host, "hardware": hw, "params": dict(zip(PARAMS, map(float, theta))),
           "fit": info, "n_points": int(len(fitted)), "n_points_without_stats": int(len(points) - len(fitted)),
           "n_free_params": len(PARAMS) - len(DEFAULT_FIXED), "V1": v1, "V2_e2_calls": e2_eval(df, chunks, choice, P),
           "V2_topk_profiling": topk_refinement(df, chunks, P),
           "per_call_choice": {"mamba1": {"keys": df["key"].tolist(), "chunk": [int(chunks[i]) for i in choice]}}}
    # tiny calibration: the E1 lengths only (18 points, batch 1, default launch configurations)
    sel = (fitted["src"] == "e1").to_numpy()
    th_tiny, _ = fit(f[sel], fitted["lat"].to_numpy()[sel], x0=theta)
    c_tiny, _ = select_e2(th_tiny, df, chunks, stats, hw)
    res["V2_tiny_calibration"] = {"n_points": int(sel.sum()), "params": dict(zip(PARAMS, map(float, th_tiny))),
                                  "e2": e2_eval(df, chunks, c_tiny)}
    if do_loko:
        res["V2_leave_one_key_out"] = loko(fitted, f, stats, hw, theta)
    if do_ablation:
        res["ablation"] = ablation(f, fitted["lat"].to_numpy(), df, chunks, stats, hw, theta)
    res["V4_pipelines"] = {fam: m for fam in ("mamba2", "gdn") if (m := pipeline_model(host, in_dir, fam))}
    if other:
        ohw = hardware(other, in_dir)
        ostats = load_kernel_stats(in_dir, ohw["arch"])
        odf = load_e2([in_dir / f"e2_mamba1_{other}.csv"])
        oc, _ = select_e2(theta, odf, chunks, ostats, ohw)
        res["V3_transfer_to_" + other] = {"note": f"parameters of {host}, hardware constants and compile facts of {other}",
                                          "e2": e2_eval(odf, chunks, oc)}
    # curves for the figure: medians over calls of each (batch, L) with measured and predicted latency
    lat = latency_matrix(df, chunks)
    cur = []
    for (b, L), g in df.groupby(["batch", "L"]):
        i = g.index.to_numpy()
        for j, c in enumerate(chunks):
            cur.append({"host": host, "batch": int(b), "L": int(L), "chunk": c, "n_calls": int(len(i)),
                        "measured_ms": float(np.median(lat[i, j])), "predicted_ms": float(np.median(P[i, j]))})
    res["_curves"] = cur
    return res


# --------------------------------------------------------------------------- reference CUDA scan (E8)
REF_THREADS = {128: 32, 256: 32, 512: 32, 1024: 64, 2048: 128}   # stock (kNThreads, kNItems = C / threads)
REF_REGS = 168      # registers per thread reported by ptxas for most instantiations of the runtime-chunk build (sm_86)
REF_PARAMS = ["t0", "tau_f", "tau_m", "tau_e", "tau_s", "tau_x", "W0", "W1"]


def ref_predict(th, L, B, d, C, hw, threads=None):
    """Same form as Eq. (5) for the reference scan: one block per (batch, channel), kNThreads threads
    walking ceil(L / C) chunks of kNItems elements per thread, a block scan over the threads per state.
    ``threads`` maps the chunk to kNThreads (default: the stock pairing ``REF_THREADS``)."""
    t0, tf, tm, te, ts, tx, W0, W1 = th
    thr = np.vectorize((threads or REF_THREADS).get)(C).astype(float)
    items, warps = C / thr, thr / 32
    alloc = np.ceil(REF_REGS / 8.0) * 8 * thr
    k = np.maximum(1, np.minimum.reduce([np.full(len(C), hw["max_blocks_per_sm"]), np.floor(hw["max_warps_per_sm"] / warps),
                                         np.floor(hw["regs_per_sm"] / alloc)]))
    P = B * d
    waves, m = np.ceil(P / (hw["n_sm"] * k)), np.minimum(k, np.ceil(P / hw["n_sm"]))
    W = m * warps
    A = tf + D_STATE * tm
    I = D_STATE * (te * items + ts * np.log2(np.minimum(thr, 32)) + tx * (warps > 1) * np.log2(warps + 1))
    return t0 + waves * np.ceil(L / C) * (A * np.minimum(1.0, W1 / W) + I * np.maximum(1.0, W / W0))


def reference_scan_model(in_dir: Path, host: str = "vgpu3090") -> dict:
    """Fit on the twelve E4 shape keys of the E8 sweep (5 chunks each), evaluate on the 2360 captured calls."""
    hw = HARDWARE[host]
    df = pd.read_csv(in_dir / f"e8_refscan_{host}.csv")
    chunks = sorted(REF_THREADS)
    keys, calls = df[df["regimes"] == "e4_key"], df[df["regimes"] != "e4_key"].reset_index(drop=True)

    def long(d):
        n = len(d)
        return (np.tile(d["L"].to_numpy(float), len(chunks)), np.tile(d["batch"].to_numpy(float), len(chunks)),
                np.tile(d["width"].to_numpy(float), len(chunks)), np.repeat(np.array(chunks, float), n))

    L, B, d, C = long(keys)
    y = np.log(np.concatenate([keys[f"lat_c{c}_mean"].to_numpy() for c in chunks]))
    lo, hi = [0, 0, 0, 0, 0, 0, 1, 1], [1, 0.1, 0.1, 0.01, 0.01, 0.1, 64, 256]
    best = None
    for x0 in ([0.01, 1e-3, 1e-4, 1e-5, 1e-5, 1e-4, 8, 16], [0.02, 1e-4, 1e-5, 1e-6, 1e-6, 1e-5, 4, 8],
               [0.005, 1e-3, 1e-3, 1e-4, 1e-4, 1e-3, 16, 32]):
        sol = least_squares(lambda th: np.log(ref_predict(th, L, B, d, C, hw)) - y, x0, bounds=(lo, hi),
                            loss="soft_l1", f_scale=0.1, x_scale="jac", max_nfev=20000)
        if best is None or sol.cost < best.cost:
            best = sol
    th = best.x
    fit_q = fit_quality(ref_predict(th, L, B, d, C, hw), np.exp(y))
    Lc, Bc, dc, Cc = long(calls)
    P = ref_predict(th, Lc, Bc, dc, Cc, hw).reshape(len(chunks), len(calls)).T
    ev = e2_eval(calls, chunks, P.argmin(1), P)
    return {"experiment": "T1-reference-scan", "host": host, "params": dict(zip(REF_PARAMS, map(float, th))),
            "fixed": {"regs_per_thread": REF_REGS, "threads_per_chunk": {str(k): v for k, v in REF_THREADS.items()}},
            "n_fit_points": int(len(y)), "fit": fit_q, "e8_calls": ev, "topk": topk_refinement(calls, chunks, P)}


def evaluate_transfer(host: str, in_dir: Path, csv_path: Path, ref_csv: Path | None = None) -> dict:
    """E11: apply the fitted parameters of ``host`` to calls of another checkpoint (other width), no refit.

    Also reports the regret of a chunk table tuned on the reference checkpoint's calls (``ref_csv``,
    keyed on length bucket, batch and alignment) and applied to the new calls.
    """
    res0 = json.loads((in_dir / f"cost_model_{host}.json").read_text())
    theta = np.array([res0["params"][k] for k in PARAMS])
    hw = hardware(host, in_dir)
    stats = load_kernel_stats(in_dir, hw["arch"])
    df = load_e2([csv_path])
    chunks = chunk_columns(df)
    choice, P = select_e2(theta, df, chunks, stats, hw)
    out = {"experiment": "T1-transfer", "host": host, "calls": str(csv_path.name), "n_calls": int(len(df)),
           "widths": sorted(int(w) for w in df["width"].unique()), "model_e2": e2_eval(df, chunks, choice, P),
           "topk": topk_refinement(df, chunks, P)}
    if ref_csv is not None and ref_csv.exists():
        ref = load_e2([ref_csv])
        key = ["L_bucket", "batch", "aligned"]
        for d_ in (ref, df):
            d_["aligned"] = (d_["L"] % 16 == 0).astype(int)
        rlat = latency_matrix(ref, chunks)
        table = {k: int(np.argmin(rlat[g.index.to_numpy()].mean(0))) for k, g in ref.groupby(key)}
        glob = int(np.argmin((rlat / rlat.min(1, keepdims=True)).sum(0)))
        pick = np.array([table.get(tuple(r), glob) for r in df[key].itertuples(index=False)])
        out["reference_table_e2"] = e2_eval(df, chunks, pick)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--hosts", default="pro6000,vgpu3090")
    p.add_argument("--no-loko", action="store_true")
    p.add_argument("--ablation", action="store_true", help="refit without each mechanism (drop-one)")
    p.add_argument("--transfer-calls", type=Path, default=None,
                   help="E11: E2-style CSV of another checkpoint; evaluate the fitted model of --hosts[0] on it")
    p.add_argument("--transfer-ref", type=Path, default=None, help="E11: reference-checkpoint CSV for the tuned table")
    p.add_argument("--reference", action="store_true",
                   help="fit the model to the runtime-chunk reference CUDA scan (E8) and write cost_model_refscan_<host>.json")
    p.add_argument("--pipelines-only", action="store_true",
                   help="only (re)compute V4 for Mamba-2 SSD and Gated DeltaNet into the existing result files")
    args = p.parse_args(argv)
    hosts = args.hosts.split(",")
    if args.reference:
        res = reference_scan_model(args.in_dir, "vgpu3090")
        write_json(args.in_dir / "cost_model_refscan_vgpu3090.json", res)
        e = res["e8_calls"]
        print(f"reference scan: fit R2 {res['fit']['r2_log']:.3f} on {res['n_fit_points']} timings; calls R2 "
              f"{e['curve_fit']['r2_log']:.3f}, regret {e['mean_regret_pct']:.2f}% (p95 {e['p95_regret_pct']:.1f}), "
              f"tied {e['tie_aware_accuracy']:.2f}; top-2 {res['topk']['top2']['mean_regret_pct']:.2f}%")
        return
    if args.transfer_calls is not None:
        res = evaluate_transfer(hosts[0], args.in_dir, args.transfer_calls, args.transfer_ref)
        out = args.in_dir / f"cost_model_transfer_{args.transfer_calls.stem}.json"
        write_json(out, res)
        m = res["model_e2"]
        print(f"{out.name}: widths {res['widths']}, model regret {m['mean_regret_pct']:.2f}% (tied "
              f"{m['tie_aware_accuracy']:.2f})" + (f", reference-tuned table {res['reference_table_e2']['mean_regret_pct']:.2f}%"
                                                   if "reference_table_e2" in res else ""))
        return
    if args.pipelines_only:
        for h in hosts:
            path = args.in_dir / f"cost_model_{h}.json"
            res = json.loads(path.read_text())
            res["V4_pipelines"] = {fam: m for fam in ("mamba2", "gdn") if (m := pipeline_model(h, args.in_dir, fam))}
            write_json(path, res)
            for fam, m in res["V4_pipelines"].items():
                e2 = m.get("e2", {})
                print(f"{h} {fam}: fit R2 {m['fit_r2_log']:.3f}, C* {m['unconstrained_optimal_chunk']:.0f}, work-bound above "
                      f"B*L {m['work_bound_above_BL_tokens']}" + (f", E2 regret {e2['mean_regret_pct']:.2f}% tied "
                                                                 f"{e2['tie_aware_accuracy']:.2f}" if e2 else ""))
        return
    for h in hosts:
        other = next((o for o in hosts if o != h), None)
        res = analyze(h, args.in_dir, other=other, do_loko=not args.no_loko, do_ablation=args.ablation)
        curves = pd.DataFrame(res.pop("_curves"))
        curves.to_csv(args.in_dir / f"cost_model_curves_{h}.csv", index=False)
        write_json(args.in_dir / f"cost_model_{h}.json", res)
        v2 = res["V2_e2_calls"]
        print(f"{h}: fit R2(log)={res['V1']['all']['r2_log']:.3f} median rel err {res['V1']['all']['median_rel_err']:.3f}; "
              f"E2 regret mean {v2['mean_regret_pct']:.2f}% p95 {v2['p95_regret_pct']:.2f}% tied {v2['tie_aware_accuracy']:.2f}; "
              f"tiny-calibration mean {res['V2_tiny_calibration']['e2']['mean_regret_pct']:.2f}%"
              + (f"; LOKO chunk {res['V2_leave_one_key_out']['chunk_mean_regret_pct']:.2f}% "
                 f"config {res['V2_leave_one_key_out']['config_mean_regret_pct']:.2f}%" if "V2_leave_one_key_out" in res else ""))


if __name__ == "__main__":
    main()
