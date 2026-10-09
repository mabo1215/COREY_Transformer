"""Deployment units of the chunk-selection regret (review items M6a, M6b, M10b and N1).

CPU-only, deterministic re-analysis of existing result files: no GPU, no new timing, no
refitting (every model choice is recomputed from stored parameters or read from the stored
per-call choices).

Part 1 - kernel level (M6a). For every kernel, GPU and captured-call set timed at every
chunk, the mean latency above the per-call oracle of
  (i)   the default chunk: 256 for the Triton Mamba-1 scan (the untuned choice of the
        break-even analysis) and for Mamba-2 SSD (its library default), 64 for Gated
        DeltaNet (the FLA default and largest accepted chunk), and the stock length
        dispatch of the reference CUDA scan, timed on the stock build;
  (ii)  the cost model: stored per-call choices (370M Triton scan, three GPUs), or the
        argmin of the stored model evaluated without refitting (reference scan; 1.4B and
        2.8B Triton calls with the device-time RTX 3090 parameters; Mamba-2 SSD and Gated
        DeltaNet with the work-term model of the same GPU, not refitted to other widths);
  (iii) a chunk table keyed on the exact shape (L, batch) and chosen on held-out prompt
        folds (the estimator of ``analysis.cv_exact_headroom``), and its in-sample floor;
  plus the best single chunk (lowest mean relative regret over all calls, in sample).
Per (length bucket, batch) cell the regret per scan call is converted to milliseconds per
forward pass (difference per call x scan layers of the checkpoint: 48 for Mamba-1 370M and
1.4B and Mamba-2 370M, 64 for Mamba-1 2.8B and Mamba-2 2.7B, 12 Gated DeltaNet layers in
the 1.4B hybrid) and per request (/ batch). For the RTX 3090 370M Mamba-1 calls at the
prompt lengths of the routed end-to-end study (256, 976 -> captured 1024, 4096 tokens,
batch 1) the same numbers are given as shares of the measured prefill and total time, with
the sanity check 48 x (chunk C - chunk 512) at L = 4096 against the measured paired
prefill difference. For the production kernels timed in device time at larger widths
(Mamba-2 2.7B, Gated DeltaNet 1.4B) calls below the launch floor (twice the median oracle
latency of the batch-1 calls with L <= 256) mix two timing regimes and are reported apart.

Part 2 - serving replay (M6a serving, M6b, M10b). The replay of ``serving_key_churn.py``
is re-run with that module's own trace loader, window selection, scheduler simulation,
policy entries and nearest-key mapping (``HostCosts``), and every replayed row is checked
against ``serving_key_churn.json`` (number of bucket keys, shares below N*). Each call is
charged, on the nearest measured key of the RTX PRO 6000 or RTX 3090 Triton Mamba-1 data
(scaled beyond the measured range exactly as the replay scales P, h and the model regret),
with the regret to the per-call oracle of the default chunk (256), the single profiled chunk
(lowest mean latency over the captured calls), the cost model and the profiled bucket
table (h_k). GPU time uses the replay's own step-time model (fixed 2 ms per step, 10 us per
prefill token; decode charged at 20 us per generated token after the first, i.e. ideal
batching without per-step fixed cost, so decode is a lower bound and the shares of total
GPU time are upper bounds).
  M6b: per replay key, the break-even of profiling the six chunks against the cost model as
  zero-cost default, N*_model = P_k / (r_model - h_k) (call-weighted key means as in the
  replay; unbounded when the model is not worse than the profiled table); share of calls on
  keys with N_k < N*_model (the model default is cheaper over the window); the startup cost
  of profiling every key of the row (sum of P_k, compilation excluded) and the cost of the
  two strategies in ms per request. The per-key N*_model of the 33 captured keys is listed.
  M10b: per row, the share of calls whose nearest measured key has fewer than 30 captured
  calls, and the share on the (bucket 2048, batch 4, unaligned) key.

Part 3 - N1 reconciliations: Table III vs VII-H regret (5), the decision overhead o (6),
the routed static-128 difference in % and ms with the absolute prefill time (8), and the
requests per break-even (9).

Output: ``deployment_units.json`` in the results directory.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.tpds import cost_model as cm
from src.experiments.tpds import serving_key_churn as skc
from src.experiments.tpds.analysis import chunk_columns, cv_exact_headroom, latency_matrix, load_e2
from src.experiments.tpds.common import RESULTS_DIR, read_jsonl, utc_now, write_json
from src.experiments.tpds.e5b_key_churn import KEY_MODES, LAYERS_PER_REQUEST, _n_star, _overhead

THIN_KEY_CALLS = 30                      # M10b: measured keys with fewer captured calls are "thin"
THIN_KEY_OF_INTEREST = (2048, 4, 0)      # (L_bucket, batch, aligned) behind the RTX 3090 93% figure
ROUTED_LENGTHS = {256: 256, 1024: 976, 4096: 4096}   # captured L -> routed prompt length
REF_CHUNK_ROUTED = 512                   # fastest static chunk of the routed study at 4096 tokens
HOSTS = ("pro6000", "vgpu3090")


# =========================================================================== part 1: kernel level
def kernel_specs() -> list[dict]:
    base = {"convention": "launch-inclusive", "floor": False, "routed": False}
    return [
        {**base, "name": "mamba1_triton_370m_pro6000", "kernel": "Triton Mamba-1 scan", "checkpoint": "Mamba-1 370M",
         "gpu": "RTX PRO 6000", "csv": "e2_mamba1_pro6000.csv", "layers": 48, "default": 256,
         "model": ("stored", "pro6000")},
        {**base, "name": "mamba1_triton_370m_vgpu3090", "kernel": "Triton Mamba-1 scan", "checkpoint": "Mamba-1 370M",
         "gpu": "RTX 3090", "csv": "e2_mamba1_vgpu3090.csv", "layers": 48, "default": 256,
         "model": ("stored", "vgpu3090"), "routed": True},
        {**base, "name": "mamba1_triton_370m_h800", "kernel": "Triton Mamba-1 scan", "checkpoint": "Mamba-1 370M",
         "gpu": "H800 PCIe", "csv": "e2_mamba1_h800.csv", "layers": 48, "default": 256, "model": ("stored", "h800")},
        {**base, "name": "mamba1_reference_cuda_370m_vgpu3090", "kernel": "reference CUDA scan (runtime chunk)",
         "checkpoint": "Mamba-1 370M", "gpu": "RTX 3090", "csv": "e8_refscan_vgpu3090.csv", "layers": 48,
         "default": "stock", "model": ("refscan", "vgpu3090"), "routed": True},
        {**base, "name": "mamba1_triton_1p4b_vgpu3090", "kernel": "Triton Mamba-1 scan", "checkpoint": "Mamba-1 1.4B",
         "gpu": "RTX 3090", "convention": "device time", "csv": "e2_mamba1_vgpu3090_1p4b_r3.csv", "layers": 48,
         "default": 256, "model": ("triton_params", "vgpu3090_r3spin")},
        {**base, "name": "mamba1_triton_2p8b_vgpu3090", "kernel": "Triton Mamba-1 scan", "checkpoint": "Mamba-1 2.8B",
         "gpu": "RTX 3090", "convention": "device time", "csv": "e2_mamba1_vgpu3090_2p8b_r3.csv", "layers": 64,
         "default": 256, "model": ("triton_params", "vgpu3090_r3spin")},
        {**base, "name": "mamba2_ssd_370m_pro6000", "kernel": "Mamba-2 SSD", "checkpoint": "Mamba-2 370M",
         "gpu": "RTX PRO 6000", "csv": "e2_mamba2_pro6000.csv", "layers": 48, "default": 256,
         "model": ("pipeline", "pro6000", "mamba2")},
        {**base, "name": "mamba2_ssd_370m_vgpu3090", "kernel": "Mamba-2 SSD", "checkpoint": "Mamba-2 370M",
         "gpu": "RTX 3090", "csv": "e2_mamba2_vgpu3090.csv", "layers": 48, "default": 256,
         "model": ("pipeline", "vgpu3090", "mamba2")},
        {**base, "name": "mamba2_ssd_2p7b_vgpu3090", "kernel": "Mamba-2 SSD", "checkpoint": "Mamba-2 2.7B",
         "gpu": "RTX 3090", "convention": "device time", "csv": "e2_mamba2_vgpu3090_2p7b_r3.csv", "layers": 64,
         "default": 256, "model": ("pipeline", "vgpu3090", "mamba2"), "floor": True},
        {**base, "name": "gdn_1p4b_vgpu3090", "kernel": "Gated DeltaNet (FLA chunk kernel)",
         "checkpoint": "Gated DeltaNet 1.4B hybrid (12 GDN layers)", "gpu": "RTX 3090", "convention": "device time",
         "csv": "e2_gdn_vgpu3090_1p4b_r3.csv", "layers": 12, "default": 64, "model": ("pipeline", "vgpu3090", "gdn"),
         "floor": True},
    ]


def load_calls(in_dir: Path, spec: dict) -> pd.DataFrame:
    df = load_e2([in_dir / spec["csv"]])
    if "regimes" in df:
        df = df[df["regimes"] != "e4_key"].reset_index(drop=True)      # E8: drop the twelve tuning-key rows
    return df


def lat_inf(df: pd.DataFrame, chunks: list[int]) -> np.ndarray:
    """Latency matrix with unavailable (call, chunk) cells as +inf (SSD 2.7B: chunk 32 beyond 32-bit indexing)."""
    lat = latency_matrix(df, chunks)
    return np.where(np.isfinite(lat), lat, np.inf)


def model_choice(in_dir: Path, spec: dict, df: pd.DataFrame, chunks: list[int], lat: np.ndarray) -> tuple[np.ndarray, dict]:
    """Per-call index of the cost-model chunk and the stored mean regret it must reproduce (if stored)."""
    kind = spec["model"][0]
    ref = None
    if kind == "stored":
        res = json.loads((in_dir / f"cost_model_{spec['model'][1]}.json").read_text())
        pc = res["per_call_choice"]["mamba1"]
        m = dict(zip(pc["keys"], pc["chunk"]))
        idx = np.array([chunks.index(int(m[k])) for k in df["key"]])
        ref = res["V2_e2_calls"]["mean_regret_pct"]
    elif kind == "triton_params":
        src = spec["model"][1]
        res = json.loads((in_dir / f"cost_model_{src}.json").read_text())
        theta = np.array([res["params"][k] for k in cm.PARAMS])
        hw = cm.hardware(src, in_dir)
        idx, _ = cm.select_e2(theta, df, chunks, cm.load_kernel_stats(in_dir, hw["arch"]), hw)
        tr = in_dir / f"cost_model_transfer_{Path(spec['csv']).stem}.json"
        if tr.exists():
            ref = json.loads(tr.read_text())["model_e2"]["mean_regret_pct"]
    elif kind == "refscan":
        res = json.loads((in_dir / "cost_model_refscan_vgpu3090.json").read_text())
        th = [res["params"][k] for k in cm.REF_PARAMS]
        hw = cm.HARDWARE[spec["model"][1]]
        P = np.stack([cm.ref_predict(th, df["L"].to_numpy(float), df["batch"].to_numpy(float),
                                     df["width"].to_numpy(float), np.full(len(df), float(c)), hw) for c in chunks], 1)
        idx = P.argmin(1)
        ref = res["e8_calls"]["mean_regret_pct"]
    elif kind == "pipeline":
        res = json.loads((in_dir / f"cost_model_{spec['model'][1]}.json").read_text())
        pm = res["V4_pipelines"][spec["model"][2]]
        th = [pm["params"][k] for k in cm.PIPE_PARAMS]
        P = np.stack([cm.pipe_predict(th, df["L"].to_numpy(float), df["batch"].to_numpy(float), c) for c in chunks], 1)
        P = np.where(np.isfinite(lat), P, np.inf)
        idx = cm.pipe_select(th, P, chunks)
        if "e2" in pm and Path(spec["csv"]).stem == f"e2_mamba2_{spec['model'][1]}":
            ref = pm["e2"]["mean_regret_pct"]
    else:
        raise ValueError(kind)
    return np.asarray(idx, dtype=int), {"source": list(spec["model"]), "stored_mean_regret_pct": ref}


def cv_exact_pick(df: pd.DataFrame, lat: np.ndarray) -> np.ndarray:
    """Per-call chunk index of the exact-shape table chosen on held-out prompt folds.

    Same estimator as ``analysis.cv_exact_headroom``: exact (L, batch) entry of the training
    folds, else the (length bucket, batch, alignment) entry, else the best single chunk.
    """
    orc = lat.min(1)
    aligned = (df["L"].to_numpy() % 16 == 0).astype(int)
    folds = df["fold"].to_numpy()
    pick = np.full(len(df), -1)
    for f in np.unique(folds):
        te, tr = folds == f, folds != f
        dtr = df[tr]
        exact = {k: int(np.argmin(lat[g.index].mean(0))) for k, g in dtr.groupby(["L", "batch"])}
        bucket = {k: int(np.argmin(lat[g.index].mean(0)))
                  for k, g in dtr.assign(aligned=aligned[tr]).groupby(["L_bucket", "batch", "aligned"])}
        glob = int(np.argmin((lat[tr] / orc[tr, None]).sum(0)))
        for i in np.where(te)[0]:
            L, b, lb = int(df["L"].iat[i]), int(df["batch"].iat[i]), int(df["L_bucket"].iat[i])
            pick[i] = exact.get((L, b), bucket.get((lb, b, int(aligned[i])), glob))
    return pick


def insample_exact_pick(df: pd.DataFrame, lat: np.ndarray) -> np.ndarray:
    pick = np.empty(len(df), dtype=int)
    for _, g in df.groupby(["L", "batch"]):
        i = g.index.to_numpy()
        pick[i] = int(np.argmin(lat[i].mean(0)))
    return pick


def routed_reference(in_dir: Path) -> dict:
    """Absolute prefill and total times of the routed study (RTX 3090, 370M, batch 1)."""
    e10 = json.loads((in_dir / "e10_routed_vgpu3090.json").read_text())
    out = {}
    for L, rec in e10["lengths"].items():
        best = rec["best_static"]["prefill_ms"]
        pc = rec["per_config"]
        statics = {n: {"prefill_ms": pc[n]["prefill_ms"]["mean"], "total_ms": pc[n]["total_ms"]["mean"],
                       "prefill_ci95_half_ms": pc[n]["prefill_ms"]["ci95_half"]}
                   for n in pc if n.startswith("static_chunk_") or n == "passive"}
        paired = {n: {"diff_ms": v["diff_ms"]["mean"], "diff_ms_ci95_half": v["diff_ms"]["ci95_half"],
                      "diff_pct": v["diff_pct"]["mean"], "diff_pct_ci95_half": v["diff_pct"]["ci95_half"]}
                  for n, v in rec["paired_vs_best_static"]["prefill_ms"].items()
                  if v and (n.startswith("static_chunk_") or n == "passive")}
        out[int(L)] = {"best_static_prefill": best, "best_static_prefill_ms": pc[best]["prefill_ms"]["mean"],
                       "best_static_total_ms": pc[rec["best_static"]["total_ms"]]["total_ms"]["mean"],
                       "best_static_total": rec["best_static"]["total_ms"],
                       "passive_prefill_ms": rec["passive_prefill_mean_ms"], "static": statics,
                       "paired_prefill_vs_best_static": paired}
    return out


def _cell(lat, orc, picks, sel, layers, batch):
    n = int(sel.sum())
    rec = {"n_calls": n, "oracle_us": float(1000 * orc[sel].mean())}
    for pol, p in picks.items():
        v = lat[np.where(sel)[0], p[sel]]
        d_us = 1000 * float((v - orc[sel]).mean())
        rec[pol] = {"regret_us_per_call": d_us, "regret_pct_rel": float(100 * (v / orc[sel] - 1).mean()),
                    "ms_per_forward_pass": d_us * layers / 1000.0,
                    "ms_per_request": None if batch is None else d_us * layers / 1000.0 / batch}
    return rec


def kernel_level(in_dir: Path) -> dict:
    routed = routed_reference(in_dir)
    out = {}
    for spec in kernel_specs():
        df = load_calls(in_dir, spec)
        chunks = chunk_columns(df)
        lat = lat_inf(df, chunks)
        orc = lat.min(1)
        if spec["default"] == "stock":
            lat = np.concatenate([lat, df[["lat_stock_mean"]].to_numpy(float)], 1)   # extra column: stock build
            p_def = np.full(len(df), lat.shape[1] - 1)
        else:
            p_def = np.full(len(df), chunks.index(int(spec["default"])))
        p_model, model_info = model_choice(in_dir, spec, df, chunks, lat[:, :len(chunks)])
        p_single = np.full(len(df), int(np.argmin((lat[:, :len(chunks)] / orc[:, None]).mean(0))))
        p_cv = cv_exact_pick(df, lat[:, :len(chunks)])
        p_in = insample_exact_pick(df, lat[:, :len(chunks)])
        picks = {"default": p_def, "best_single_chunk": p_single, "cost_model": p_model,
                 "exact_table_heldout": p_cv, "exact_table_insample": p_in}
        rel_model = float(100 * (lat[np.arange(len(df)), p_model] / orc - 1).mean())
        checks = {"model_mean_regret_pct_recomputed": rel_model,
                  "model_mean_regret_pct_stored": model_info["stored_mean_regret_pct"]}
        if np.isfinite(lat[:, :len(chunks)]).all():
            ref_cv = cv_exact_headroom(df, chunks)["mean_ms"]
            checks["heldout_table_mean_us_recomputed"] = float(1000 * (lat[np.arange(len(df)), p_cv] - orc).mean())
            checks["heldout_table_mean_us_analysis_py"] = 1000 * ref_cv
        floor_ms = None
        above = np.ones(len(df), bool)
        if spec["floor"]:
            small = (df["batch"] == 1).to_numpy() & (df["L"] <= 256).to_numpy()
            floor_ms = float(2 * np.median(orc[small]))
            above = orc > floor_ms
        rows = []
        for (lb, b), g in df.groupby(["L_bucket", "batch"]):
            sel = np.zeros(len(df), bool)
            sel[g.index.to_numpy()] = True
            rec = {"L_bucket": int(lb), "batch": int(b), "L_mean": float(g["L"].mean()),
                   **_cell(lat, orc, picks, sel, spec["layers"], int(b))}
            if spec["floor"]:
                rec["share_calls_above_floor"] = float(above[sel].mean())
            rows.append(rec)
        pooled = {}
        for name, sel in [("all", np.ones(len(df), bool))] + [(f"batch{b}", (df["batch"] == b).to_numpy())
                                                               for b in sorted(df["batch"].unique())]:
            pooled[name] = _cell(lat, orc, picks, sel, spec["layers"], None)
            if spec["floor"]:
                s2 = sel & above
                pooled[name + "_above_floor"] = _cell(lat, orc, picks, s2, spec["layers"], None) if s2.any() else None
        rec = {k: v for k, v in spec.items() if k not in ("csv",)}
        rec.update({"source_csv": spec["csv"], "chunks": chunks, "n_calls": int(len(df)),
                    "default_chunk": spec["default"], "best_single_chunk": int(chunks[p_single[0]]),
                    "model": model_info, "checks": checks, "floor_ms": floor_ms,
                    "n_calls_above_floor": int(above.sum()) if spec["floor"] else None,
                    "by_bucket_batch": rows, "pooled": pooled})
        if spec["routed"]:
            rr = {}
            for Lc, Lr in ROUTED_LENGTHS.items():
                sel = (df["batch"] == 1).to_numpy() & (df["L"] == Lc).to_numpy()
                if not sel.any():
                    continue
                c = _cell(lat, orc, picks, sel, spec["layers"], 1)
                pre, tot = routed[Lr]["best_static_prefill_ms"], routed[Lr]["best_static_total_ms"]
                for pol in picks:
                    c[pol]["share_of_prefill_pct"] = 100 * c[pol]["ms_per_request"] / pre
                    c[pol]["share_of_total_pct"] = 100 * c[pol]["ms_per_request"] / tot
                c.update({"captured_L": Lc, "routed_L": Lr, "routed_prefill_ms": pre, "routed_total_ms": tot,
                          "denominator_note": ("measured" if Lc == Lr else "routed 976-token prompt for captured L=1024")
                          + ("" if spec["default"] == "stock" else "; prefill measured with the reference scan")})
                # sanity check: 48 x kernel difference to chunk 512 versus the measured paired prefill difference
                c["routed_study_kernel"] = spec["default"] == "stock"
                if spec["default"] == "stock":      # the routed study ran the runtime-chunk reference scan
                    j512 = chunks.index(REF_CHUNK_ROUTED)
                    pred = {}
                    for j, ch in enumerate(chunks):
                        meas = routed[Lr]["paired_prefill_vs_best_static"].get(f"static_chunk_{ch}")
                        best = routed[Lr]["best_static_prefill"]
                        p_ms = float(spec["layers"] * (lat[sel, j] - lat[sel, j512]).mean())
                        m_ms = None
                        if meas is not None:
                            # measured paired difference re-referenced to static 512 (the best static chunk at 4096)
                            m512 = routed[Lr]["paired_prefill_vs_best_static"]["static_chunk_512"]
                            m_ms = meas["diff_ms"] - (m512["diff_ms"] if best != "static_chunk_512" else 0.0)
                        pred[str(ch)] = {"predicted_ms_vs_c512": p_ms, "measured_ms_vs_c512": m_ms,
                                         "measured_vs_best_static_ci95_half_ms":
                                             None if meas is None else meas["diff_ms_ci95_half"]}
                    c["sanity_48x_kernel_diff_vs_routed"] = pred
                rr[str(Lc)] = c
            rec["routed_lengths"] = rr
        out[spec["name"]] = rec
    return {"kernels": out, "routed_study": {str(k): v for k, v in routed.items()}}


# =========================================================================== part 2: serving replay
class MeasuredKeys:
    """Per measured bucket_aligned key (E5b order): regrets of each policy, captured calls, tuning cost."""

    def __init__(self, host: str, in_dir: Path):
        e5b = json.loads((in_dir / f"e5b_{host}.json").read_text())
        e5 = json.loads((in_dir / f"e5_{host}.json").read_text())
        df = load_e2([in_dir / f"e2_mamba1_{host}.csv"])
        df["aligned"] = (df["L"] % 16 == 0).astype(int)
        chunks = chunk_columns(df)
        lat = latency_matrix(df, chunks)
        orc = lat.min(1)
        cmj = json.loads((in_dir / f"cost_model_{host}.json").read_text())["per_call_choice"]["mamba1"]
        mch = dict(zip(cmj["keys"], cmj["chunk"]))
        model = np.array([chunks.index(int(mch[k])) for k in df["key"]])
        self.profiled_chunk = int(e5["view_e2_traces"]["mamba1"]["policies"]["profiled_static"]["chunk"])
        j_def, j_prof = chunks.index(256), chunks.index(self.profiled_chunk)
        groups = {k: g.index.to_numpy() for k, g in df.groupby(["L_bucket", "batch", "aligned"])}
        recs, checks = [], []
        for r in e5b["keys"]:
            i = groups[(r["L_bucket"], r["batch"], r["aligned"])]
            tab = int(np.argmin(lat[i].mean(0)))
            rec = {"L_bucket": r["L_bucket"], "batch": r["batch"], "aligned": r["aligned"], "n_calls": r["n_calls"],
                   "P_profile_s": r["P_s"]["profile"], "oracle_ms": float(orc[i].mean()),
                   "table_chunk": int(chunks[tab]),
                   "regret_default_ms": float((lat[i, j_def] - orc[i]).mean()),
                   "regret_profiled_chunk_ms": float((lat[i, j_prof] - orc[i]).mean()),
                   "regret_model_ms": float((lat[i, model[i]] - orc[i]).mean()),
                   "regret_table_ms": float((lat[i, tab] - orc[i]).mean())}
            checks.append(max(abs(rec["regret_model_ms"] - r["model_regret_vs_oracle_ms"]),
                              abs(rec["regret_table_ms"] - r["headroom_ms"])))
            gap = rec["regret_model_ms"] - rec["regret_table_ms"]
            rec["model_minus_table_ms"] = gap
            rec["n_star_model_calls"] = None if gap <= 0 else 1000.0 * rec["P_profile_s"] / gap
            rec["n_star_model_requests"] = None if gap <= 0 else rec["n_star_model_calls"] / LAYERS_PER_REQUEST
            recs.append(rec)
        self.recs = recs
        self.max_abs_diff_vs_e5b_ms = float(max(checks))
        self.sync_ms = float(e5b["host_sync_ms"])
        for name in ("n_calls", "oracle_ms", "regret_default_ms", "regret_profiled_chunk_ms", "regret_model_ms",
                     "regret_table_ms"):
            setattr(self, name, np.array([r[name] for r in recs], dtype=float))
        self.is_key_of_interest = np.array([(r["L_bucket"], r["batch"], r["aligned"]) == THIN_KEY_OF_INTEREST
                                            for r in recs])


def nearest_measured(hc: skc.HostCosts, L: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the nearest measured key and the extrapolation scale, exactly as ``HostCosts.costs``."""
    Lb = np.left_shift(1, skc._bucket_log2(L))
    al = (L % 16 == 0).astype(int)
    d = (np.abs(np.log2(Lb)[:, None] - hc.rLb[None]) + np.abs(np.log2(b)[:, None] - hc.rb[None])
         + (al[:, None] != hc.ral[None]))
    d = d - 1e-9 * hc.rPp[None] / hc.rPp.max()
    nn = np.argmin(d, axis=1)
    scale = np.maximum(1.0, L / hc.L_max) * np.maximum(1.0, b / hc.b_max)
    return nn, scale


POLICIES = ("default_chunk_256", "profiled_single_chunk", "cost_model", "profiled_bucket_table")


def row_analysis(Lk, bk, w, packed, hosts, mkeys, n_requests, prefill_ms, decode_ms) -> dict:
    """Extended per-row statistics (validation fields, M6a serving, M6b, M10b) for every host."""
    code = Lk.astype(np.int64) << skc.SHAPE_BITS | bk.astype(np.int64)
    ue, inv = np.unique(code, return_inverse=True)
    We = np.bincount(inv, weights=w)
    Le, be = ue >> skc.SHAPE_BITS, ue & ((1 << skc.SHAPE_BITS) - 1)
    bcost = np.ones_like(be) if packed else be
    calls_e = LAYERS_PER_REQUEST * We
    total_calls = float(calls_e.sum())
    modes = {"bucket_aligned": KEY_MODES["bucket_aligned"], "exact": KEY_MODES["exact"]}
    if packed:
        modes[skc.PACKED_TOKENS_MODE] = ["L_bucket", "aligned"]
    kinv_by_mode = {}
    for mode, cols in modes.items():
        if mode == "exact":
            kcode = np.arange(len(ue))
        else:
            kc = skc._bucket_log2(Le)
            if "aligned" in cols:
                kc = kc * 2 + (Le % 16 == 0)
            kcode = kc << skc.SHAPE_BITS | (be if "batch" in cols else 0)
        kinv_by_mode[mode] = np.unique(kcode, return_inverse=True)[1]
    total_gpu_ms = prefill_ms + decode_ms
    out = {"n_requests": int(n_requests), "n_calls": total_calls,
           "gpu_time_step_model": {"prefill_ms_per_request": prefill_ms / n_requests,
                                   "decode_ms_per_request": decode_ms / n_requests,
                                   "total_ms_per_request": total_gpu_ms / n_requests},
           "by_host": {}}
    for hc in hosts:
        mk = mkeys[hc.host]
        c = hc.costs(Le, bcost)
        nn, scale = nearest_measured(hc, Le, bcost)
        assert np.allclose(c["P_profile"], hc.rPp[nn] * scale) and np.allclose(c["h"], hc.rh[nn] * scale, equal_nan=True)
        reg = {"default_chunk_256": mk.regret_default_ms[nn] * scale,
               "profiled_single_chunk": mk.regret_profiled_chunk_ms[nn] * scale,
               "cost_model": c["regret"], "profiled_bucket_table": c["h"]}
        orc = mk.oracle_ms[nn] * scale
        hr = {"scan_oracle_ms_per_request": float((calls_e * orc).sum() / n_requests), "policies": {}}
        for pol, v in reg.items():
            tot = float((calls_e * v).sum())
            hr["policies"][pol] = {"ms_per_request": tot / n_requests, "us_per_call": 1000 * tot / total_calls,
                                   "share_of_prefill_pct": 100 * tot / prefill_ms,
                                   "share_of_total_gpu_pct": 100 * tot / total_gpu_ms}
        thin = mk.n_calls[nn] < THIN_KEY_CALLS
        hr["m10b"] = {"share_calls_on_thin_measured_keys": float(calls_e[thin].sum() / total_calls),
                      "share_calls_on_key_2048_b4_unaligned": float(calls_e[mk.is_key_of_interest[nn]].sum()
                                                                    / total_calls),
                      "thin_measured_keys_hit": [f"L{mk.recs[k]['L_bucket']}|b{mk.recs[k]['batch']}|"
                                                 f"al{mk.recs[k]['aligned']}|n{mk.recs[k]['n_calls']}"
                                                 for k in np.unique(nn[thin])],
                      "share_calls_extrapolated": float(calls_e[c["extrapolated"]].sum() / total_calls)}
        per_mode = {}
        for mode, kinv in kinv_by_mode.items():
            Wk = np.bincount(kinv, weights=We)
            calls = LAYERS_PER_REQUEST * Wk
            agg = {k: np.bincount(kinv, weights=We * v) / Wk for k, v in c.items() if k != "extrapolated"}
            rec = {"n_keys": int(len(Wk))}
            if mode == "bucket_aligned":
                val = {}
                for combo in skc.NSTAR_COMBOS:
                    pk, ok = combo.split("|")
                    ns = [_n_star(p, o, hh) for p, o, hh in zip(agg[f"P_{pk}"], agg[f"o_{ok}"], agg["h"])]
                    below = np.array([s is None or n < s for n, s in zip(calls, ns)])
                    val[combo] = float(calls[below].sum() / calls.sum())
                rec["share_calls_below_nstar_online"] = val
            gap = agg["regret"] - agg["h"]
            nstar = np.where(gap > 0, 1000.0 * agg["P_profile"] / np.where(gap > 0, gap, 1.0), np.inf)
            model_cheaper = calls < nstar
            startup_s = float(agg["P_profile"].sum())
            fin = nstar[np.isfinite(nstar)]
            rec["m6b"] = {
                "share_calls_model_default_cheaper_than_profiling": float(calls[model_cheaper].sum() / calls.sum()),
                "share_keys_model_default_cheaper": float(model_cheaper.mean()),
                "n_keys_model_never_worse_than_table": int((~np.isfinite(nstar)).sum()),
                "nstar_model_median_calls": float(np.median(fin)) if len(fin) else None,
                "startup_profile_all_keys_s": startup_s,
                "cost_ms_per_request": {
                    "model_only": float((calls * agg["regret"]).sum() / n_requests),
                    "profile_all_keys_at_start": float((1000.0 * agg["P_profile"] + calls * agg["h"]).sum() / n_requests),
                    "per_key_best_of_both_hindsight": float(np.minimum(calls * agg["regret"],
                                                                        1000.0 * agg["P_profile"] + calls * agg["h"]).sum()
                                                            / n_requests)}}
            per_mode[mode] = rec
        hr["key_modes"] = per_mode
        out["by_host"][hc.host] = hr
    return out


def _gen_decode_ms(gen: np.ndarray) -> float:
    return float(np.maximum(gen - 1, 0).sum() * skc.T_DEC_S * 1000.0)


def _tasks_for(name: str, window: str, policies) -> list[tuple]:
    t = []
    for pol in policies:
        if pol == "per_request":
            t.append((name, window, "per_request", None, None, None))
        elif pol == "fixed_slice":
            t += [(name, window, "fixed_slice", T, None, None) for T in skc.BUDGETS]
        else:
            settings = ([("colocated", skc.RHO_PRIMARY)] + [("colocated", r) for r in skc.RHO_SENSITIVITY]
                        + [("prefill_only", skc.RHO_PRIMARY)])
            t += [(name, window, "chunked_budget", T, s, r) for s, r in settings for T in skc.BUDGETS]
    return t


_G: dict = {}      # trace and cost tables shared with forked workers


def _run_task(task) -> list[dict]:
    name, window, policy, T, sched, rho = task
    tr, hosts, mkeys = _G["trace"], _G["hosts"], _G["mkeys"]
    win, info = skc.select_window(tr, window)
    n_req = len(win["ctx"])
    dec = _gen_decode_ms(win["gen"])
    base = {"trace": name, "window": window, "policy": policy, "T": T, "sched": sched, "rho": rho}
    t0 = time.time()
    if policy == "per_request":
        Lk, bk, w = skc.policy_entries(win, "per_request")["batch1"]
        pre = float(((skc.T_FIX_S + skc.T_PF_S * Lk) * w).sum() * 1000.0)
        rows = [{**base, "kernel": "batch1", **row_analysis(Lk, bk, w, False, hosts, mkeys, n_req, pre, dec)}]
    elif policy == "fixed_slice":
        Lk, bk, w = skc.policy_entries(win, "fixed_slice", T)["batch1"]
        pre = float(((skc.T_FIX_S + skc.T_PF_S * Lk) * w).sum() * 1000.0)
        rows = [{**base, "kernel": "batch1", **row_analysis(Lk, bk, w, False, hosts, mkeys, n_req, pre, dec)}]
    else:
        tok = float(win["ctx"].sum())
        s = tok * skc.T_PF_S / (info["duration_s"] * rho)
        sim = skc.simulate_chunked(win["t"] * s, win["ctx"], win["gen"], T, colocate=(sched == "colocated"))
        w = np.ones(len(sim["nseq"]))
        pre = float((skc.T_FIX_S + skc.T_PF_S * sim["tot"]).sum() * 1000.0)
        rows = []
        for kernel, (Lk, bk) in (("padded", (sim["lmax"], sim["nseq"])), ("packed", (sim["tot"], sim["nseq"]))):
            rows.append({**base, "kernel": kernel, "sim_prefill_steps": int(len(sim["nseq"])),
                         **row_analysis(Lk, bk, w, kernel == "packed", hosts, mkeys, n_req, pre, dec)})
    for r in rows:
        r["seconds"] = round(time.time() - t0, 1)
    return rows


def serving(in_dir: Path, data_dir: Path, workers: int, windows: str, log) -> dict:
    hosts = [skc.HostCosts(h, in_dir) for h in HOSTS]
    mkeys = {h: MeasuredKeys(h, in_dir) for h in HOSTS}
    ref = json.loads((in_dir / "serving_key_churn.json").read_text())
    ref_rows = {(r["trace"], r["window"], r["policy"], r["T"], r["kernel"], r["sched"], r["rho"]): r for r in ref["results"]}
    sel = skc.WINDOWS if windows == "all" else [w for w in skc.WINDOWS if f"{w[0]}/{w[1]}" in windows.split(",")]
    results = []
    _G.update({"hosts": hosts, "mkeys": mkeys})
    for name in dict.fromkeys(w[0] for w in sel):
        t0 = time.time()
        _G["trace"] = skc.load_trace(name, data_dir)
        log(f"loaded {name}: {len(_G['trace']['ctx'])} requests, {time.time() - t0:.1f}s")
        tasks = [t for nm, win, pols in sel if nm == name for t in _tasks_for(nm, win, pols)]
        # longest tasks first: chunked simulations of the largest windows
        tasks.sort(key=lambda t: (t[2] != "chunked_budget", t[1] not in ("full", "24h"), -(t[3] or 0)))
        with get_context("fork").Pool(min(workers, len(tasks))) as pool:
            for rows in pool.imap_unordered(_run_task, tasks):
                for r in rows:
                    results.append(r)
                    log(f"  {r['trace']}/{r['window']} {r['policy']} T={r['T']} {r['kernel']} {r['sched']} "
                        f"rho={r['rho']}: {r['seconds']}s")
        _G.pop("trace", None)
    # validation against serving_key_churn.json
    max_diff, n_checked, missing = 0.0, 0, []
    for r in results:
        k = (r["trace"], r["window"], r["policy"], r["T"], r["kernel"], r["sched"], r["rho"])
        rr = ref_rows.get(k)
        if rr is None:
            missing.append(list(k))
            continue
        for h in HOSTS:
            a = r["by_host"][h]["key_modes"]["bucket_aligned"]
            b = rr["keys"]["bucket_aligned"]
            assert a["n_keys"] == b["n_keys"], (k, a["n_keys"], b["n_keys"])
            for combo, v in a["share_calls_below_nstar_online"].items():
                max_diff = max(max_diff, abs(v - b["by_host"][h]["share_calls_below_nstar"][combo]))
            n_checked += 1
            r["by_host"][h]["paper_fields"] = {
                "share_calls_below_nstar_profile_sync": b["by_host"][h]["share_calls_below_nstar"]["profile|host_sync"],
                "share_calls_below_nstar_profile_hist8":
                    b["by_host"][h]["share_calls_below_nstar"]["profile|sampled_histogram"],
                "cache_tuning_total_profile_s": b["by_host"][h]["cache_tuning_total_s"]["profile"]}
    results.sort(key=lambda r: (r["trace"], r["window"], r["policy"], r["T"] or 0, r["kernel"], r["sched"] or "",
                                r["rho"] or 0))
    measured = {h: {"profiled_single_chunk": mkeys[h].profiled_chunk, "host_sync_ms": mkeys[h].sync_ms,
                    "max_abs_diff_vs_e5b_ms": mkeys[h].max_abs_diff_vs_e5b_ms, "keys": mkeys[h].recs}
                for h in HOSTS}
    return {"validation": {"rows_checked_host_pairs": n_checked, "rows_missing_in_reference": missing,
                           "max_abs_diff_share_below_nstar": max_diff},
            "measured_keys": measured, "rows": results}


# =========================================================================== M6b summary on the 33 captured keys
def m6b_captured(in_dir: Path, measured: dict) -> dict:
    out = {}
    for h in HOSTS:
        e5 = json.loads((in_dir / f"e5_{h}.json").read_text())
        e5b = json.loads((in_dir / f"e5b_{h}.json").read_text())
        recs = measured[h]["keys"]
        w = np.array([r["n_calls"] for r in recs], float)
        ns = [r["n_star_model_calls"] for r in recs]
        fin = np.array([x for x in ns if x is not None])
        sync = measured[h]["host_sync_ms"]
        reg_m = np.array([r["regret_model_ms"] for r in recs])
        reg_t = np.array([r["regret_table_ms"] for r in recs])
        above = reg_m > sync
        v4 = e5["view_e4_keys"]["mamba1"]["policies"]
        ex = e5b["exact_cv_headroom"]
        out[h] = {
            "n_keys": len(recs), "P_profile_s_median": float(np.median([r["P_profile_s"] for r in recs])),
            "P_profile_s_sum_33_keys": float(sum(r["P_profile_s"] for r in recs)),
            "n_star_model_calls": {"median": float(np.median(fin)), "min": float(fin.min()), "max": float(fin.max()),
                                   "n_unbounded": int(sum(x is None for x in ns))},
            "n_star_model_requests_median": float(np.median(fin)) / LAYERS_PER_REQUEST,
            "model_regret_vs_oracle_us_callweighted": float(1000 * np.average(reg_m, weights=w)),
            "table_headroom_vs_oracle_us_callweighted": float(1000 * np.average(reg_t, weights=w)),
            "model_regret_vs_table_us_callweighted": float(1000 * np.average(reg_m - reg_t, weights=w)),
            "n_keys_model_regret_above_sync": int(above.sum()),
            "keys_model_regret_above_sync": [
                {k: r[k] for k in ("L_bucket", "batch", "aligned", "n_calls", "P_profile_s", "regret_model_ms",
                                   "regret_table_ms", "n_star_model_calls", "n_star_model_requests")}
                for r, a in zip(recs, above) if a],
            "headroom_used_against_dispatch_us": {
                "exact_shape_heldout_covered_mean": 1000 * ex["covered"]["mean_ms"],
                "exact_shape_heldout_all_calls_mean": 1000 * ex["mean_ms"],
                "bucket_batch_table_heldout_mean": 1000 * e5["view_e2_traces"]["mamba1"]["oracle_headroom_over_shape_table_ms"]},
            "calibration_wall_s_e4_full_search": v4["autotuned"]["T_ms"] / 1000.0,
            "profiling_12_keys_s": v4["profiled_static"]["T_ms"] / 1000.0,
            "compile_per_chunk_variant_s": v4["profiled_static"]["compile_per_config_ms"] / 1000.0,
            "host_sync_us": 1000 * sync}
        o = out[h]
        hd = o["headroom_used_against_dispatch_us"]
        o["model_regret_over_headroom_ratio"] = {k: o["model_regret_vs_oracle_us_callweighted"] / v for k, v in hd.items()}
    return out


# =========================================================================== part 3: N1
def n1(in_dir: Path) -> dict:
    out = {}
    # N1.5: Table III/IV best single chunk 23.4% vs VII-H 34 us (10.6%)
    df = load_e2([in_dir / "e2_mamba1_pro6000.csv"])
    chunks = chunk_columns(df)
    lat = latency_matrix(df, chunks)
    orc = lat.min(1)
    per = {}
    for j, c in enumerate(chunks):
        per[str(c)] = {"relative_mean_regret_pct": float(100 * (lat[:, j] / orc - 1).mean()),
                       "absolute_regret_us_per_call": float(1000 * (lat[:, j] - orc).mean()),
                       "latency_weighted_pct_of_chunk_latency": float(100 * (lat[:, j].mean() - orc.mean()) / lat[:, j].mean()),
                       "latency_weighted_pct_of_oracle_latency": float(100 * (lat[:, j].mean() - orc.mean()) / orc.mean()),
                       "mean_latency_ms": float(lat[:, j].mean())}
    e3 = json.loads((in_dir / "e3_mamba1_pro6000.json").read_text())["predictors_all_batches"]["global_static"]
    e5 = json.loads((in_dir / "e5_pro6000.json").read_text())["view_e2_traces"]["mamba1"]
    out["5_best_single_chunk"] = {
        "table_IV_global_static_heldout": {"mean_regret_pct": e3["mean_regret_pct"],
                                           "chosen_distribution": e3["chosen_distribution"],
                                           "source": "e3_mamba1_pro6000.json predictors_all_batches.global_static"},
        "vii_h_profiled_static": {"chunk": e5["policies"]["profiled_static"]["chunk"],
                                  "oracle_headroom_ms": e5["oracle_headroom_ms"],
                                  "oracle_headroom_pct_of_static": e5["oracle_headroom_pct_of_static"],
                                  "source": "e5_pro6000.json view_e2_traces.mamba1"},
        "per_chunk": per,
        "note": "Table IV's best single chunk minimizes the summed per-call normalized latency on the training folds "
                "(chunk 256 in every fold) and is reported as the mean of per-call relative regrets; VII-H's single "
                "profiled chunk minimizes the mean absolute latency (chunk 512) and is reported as total latency above "
                "the oracle divided by that chunk's total latency. Different chunk and different estimator."}
    # N1.6: o = 4.43 ms
    e4 = read_jsonl(in_dir / "e4_pro6000.jsonl")
    numel = (df["L"] * df["batch"] * df["width"]).to_numpy()
    ov = np.array([_overhead(e4, int(x), 1) for x in numel])
    by_b = {str(int(b)): {"n_calls": int((df["batch"] == b).sum()), "mean_full_hist_ms": float(ov[(df["batch"] == b).to_numpy()].mean())}
            for b in sorted(df["batch"].unique())}
    e10 = json.loads((in_dir / "e10_routed_vgpu3090.json").read_text())
    hist = e10["lengths"]["4096"]["decision_cost"]["hist"]
    out["6_decision_overhead_o"] = {
        "vii_h_o_ms": e5["overhead_ms"], "recomputed_ms": float(ov.mean()), "by_batch": by_b,
        "definition": "mean over the 2360 captured Mamba-1 calls (RTX PRO 6000) of the isolated full-histogram decision "
                      "time (histogram K=256 + host sync + rule) at the nearest of the twelve measured tensor sizes",
        "source": "e5_pro6000.json view_e2_traces.mamba1.overhead_ms; e4_pro6000.jsonl kind=overhead stride=1",
        "table_V_full_histogram_ms_per_generation": hist["decision_cost_ms"],
        "table_V_per_scan_ms": hist["decision_cost_ms"] / 48.0,
        "table_V_context": "RTX 3090, reference CUDA scan inside the 370M model, 4096 tokens, batch 1, all 48 layers",
        "table_S_overhead_batch1_ms": {str(r["L"]): r["decision_wall"]["mean_ms"] for r in e4
                                       if r.get("kind") == "overhead" and r["stride"] == 1 and r["batch"] == 1},
        "nearest_size_note": "the nearest-size lookup matches tensor sizes only (L x batch x width), so a batch-1 "
                             "call of L = 2048 takes the first measured row of equal size (L = 128, batch 16)"}
    # N1.8: routed static 128
    L4 = e10["lengths"]["4096"]
    d = L4["paired_vs_best_static"]["prefill_ms"]["static_chunk_128"]
    out["8_routed_static128"] = {
        "best_static": L4["best_static"]["prefill_ms"],
        "static512_prefill_ms": L4["per_config"]["static_chunk_512"]["prefill_ms"]["mean"],
        "static512_prefill_ci95_half_ms": L4["per_config"]["static_chunk_512"]["prefill_ms"]["ci95_half"],
        "static128_prefill_ms": L4["per_config"]["static_chunk_128"]["prefill_ms"]["mean"],
        "passive_prefill_ms": L4["passive_prefill_mean_ms"],
        "paired_diff_ms": d["diff_ms"]["mean"], "paired_diff_ms_ci95_half": d["diff_ms"]["ci95_half"],
        "paired_diff_pct": d["diff_pct"]["mean"], "paired_diff_pct_ci95_half": d["diff_pct"]["ci95_half"],
        "source": "e10_routed_vgpu3090.json lengths.4096.paired_vs_best_static.prefill_ms.static_chunk_128"}
    # N1.9: requests per break-even
    s9 = {}
    for h in HOSTS:
        s = json.loads((in_dir / f"e5b_{h}.json").read_text())["summary_n_star"]["profile|host_sync"]
        s9[h] = {"median_calls": s["median_calls"], "median_requests": s["median_requests"],
                 "rounded_calls_over_48": round(s["median_calls"], -2) / 48.0}
    out["9_requests_per_breakeven"] = {**s9, "source": "e5b_<host>.json summary_n_star['profile|host_sync']"}
    return out


# =========================================================================== main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--data-dir", type=Path, default=skc.DATA_DIR)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "deployment_units.json")
    p.add_argument("--workers", type=int, default=6, help="processes for the serving replay (fork)")
    p.add_argument("--windows", default="all", help="comma list of trace/window for the replay, or 'all'")
    p.add_argument("--skip-serving", action="store_true", help="kernel level and N1 only")
    args = p.parse_args(argv)
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    t0 = time.time()
    res = {"experiment": "deployment_units", "items": ["M6a", "M6b", "M10b", "N1"], "created_utc": utc_now(),
           "conventions": {
               "regret": "latency above the per-call oracle (fastest timed chunk of each call)",
               "ms_per_forward_pass": "regret per scan call x scan layers of the checkpoint",
               "ms_per_request": "ms_per_forward_pass / batch (kernel level); total regret / requests (replay)",
               "replay_gpu_time": f"step model of the replay: {skc.T_FIX_S * 1e3:g} ms per step + "
                                  f"{skc.T_PF_S * 1e6:g} us per prefill token; decode {skc.T_DEC_S * 1e6:g} us per "
                                  "generated token after the first (ideal batching; decode lower bound)",
               "thin_key_calls": THIN_KEY_CALLS}}
    log("part 1: kernel level")
    res.update(kernel_level(args.in_dir))
    log("part 3: N1")
    res["n1"] = n1(args.in_dir)
    if not args.skip_serving:
        log("part 2: serving replay")
        sv = serving(args.in_dir, args.data_dir, args.workers, args.windows, log)
        res["serving"] = sv
        res["m6b_captured_keys"] = m6b_captured(args.in_dir, sv["measured_keys"])
    res["runtime_s"] = round(time.time() - t0, 1)
    write_json(args.out, res)
    log(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB) in {res['runtime_s']} s")


if __name__ == "__main__":
    main()
