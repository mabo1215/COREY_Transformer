"""Offline re-analysis for review cycle tpds-r4: items M8a, N1.1-N1.4, N1.7 and M3c.

CPU only, deterministic (fixed seed), reads only stored result files.

M8a  Oracle headroom over an exact-shape chunk table for every Mamba-1 checkpoint, GPU and timing
     convention with captured calls (370M on the RTX PRO 6000, RTX 3090 and H800 PCIe, launch-inclusive
     and, where re-timed with a GPU-side pre-spin, device time; 1.4B and 2.8B re-timed on the RTX 3090
     in both conventions; the reference CUDA scan), plus the Mamba-2 / Gated DeltaNet rows and the
     superseded batch-1 sweep of the larger checkpoints for the reproduction check. Every estimate is
     produced by ``headroom_noise.analyze_dataset`` with the seed and dataset index that
     ``headroom_noise.py`` uses, so the stored rows are reproduced exactly (checked leaf by leaf against
     ``headroom_noise*.json`` and against the rows printed in the supplement). Added per row:
       * the held-out per-shape maximum over the covered (tuned) shapes, next to the in-sample maximum,
         and the shapes at which both maxima occur;
       * the split-half per-shape maximum (per-shape mean over the 200 draws);
       * the number of tuned shapes above one host synchronization (the value measured on that GPU and a
         nominal 20 us) and above the device-side decision cost (one reduction kernel before the scan,
         RTX 3090: 7.3 us at L = 2048 and 42 us at L = 8192, batch 1, width 2048), for the in-sample,
         held-out and split-half estimates; and a per-shape device cost interpolated linearly in the
         number of scan-input elements (batch x length x width) between the two measured sizes, held at
         the smaller value below it (an assumption, reported as such).
     The split-half per-shape headroom is recomputed with the same random stream as
     ``analyze_dataset`` and checked against its mean.

N1.1 R^2 of the Mamba-2 work-term model on its fitting data (75 timings per GPU) and on the captured
     calls (2360 calls x 5 chunks), refitted with ``cost_model.pipe_fit``.
N1.2 95th percentile of the cost-model regret pooled over the tensor-parallel shard widths, recomputed
     from the per-shape rows of ``e12_analysis_<host>.json``.
N1.7 Leave-one-key-out regret of the cost model used as a full auto-tuner of the 184 launch
     configurations, refitted with ``cost_model.loko`` (12 fits per GPU).

M3c  The Mamba-2 work-term selector picks argmin_C (F^p + W^p)^(1/p). Where the pipeline is
     launch-bound, (W/F)^p underflows and every chunk is predicted at exactly F; argmin then returns the
     first (smallest) chunk, 32. Re-scored selectors: ``original`` (ties to chunk 32), ``launch_default``
     (exact ties to the work-bound optimum argmin_C (b C + g / C) over the available chunks, 256 on both
     GPUs), ``launch_default_near`` (ties within a relative 1e-6 to the same chunk), ``work_argmin``
     (argmin of W, the selection in the limit of a vanishing floor), and the static chunks 32 and 256, on
     every Mamba-2 data set: the 370M captured calls (RTX PRO 6000 and RTX 3090, launch-inclusive; RTX
     3090 device time, which the paper does not use because the timings mix regimes), the 2.7B captured
     calls (RTX 3090, device time; split at the floor of twice the median batch-1 latency of calls with
     L <= 256, as in the supplement), and the synthetic sweep at the 370M, 1.3B and 2.7B widths (RTX 3090,
     launch-inclusive, three rounds). The work-bound boundary B*L = F / (a + 2 sqrt(b g)) is restated for a
     320 us floor with the fitted a, b, g (and at the discrete chunk 256: F / (a + 256 b + g / 256)).

Output: ``headroom_r4.json`` in the results directory.

Usage:  python3 -m src.experiments.tpds.headroom_r4 [--draws 200] [--seed 20260930] [--skip-loko]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.tpds import cost_model as CM
from src.experiments.tpds import e12_analysis as E12
from src.experiments.tpds import e17_analysis as E17
from src.experiments.tpds import headroom_noise as HN
from src.experiments.tpds.analysis import chunk_columns, latency_matrix, load_e2
from src.experiments.tpds.common import RESULTS_DIR, read_jsonl, write_json

# ----------------------------------------------------------------------------- M8a data sets
# name, csv, gpu, kernel, checkpoint, convention, stored headroom file (None: new in this cycle), label
HEADROOM_SETS = [
    ("mamba1_pro6000", "e2_mamba1_pro6000.csv", "pro6000", "M1", "370M", "launch-inclusive", "headroom_noise.json",
     "Mamba-1 370M, Triton scan, RTX PRO 6000, launch-inclusive"),
    ("mamba1_vgpu3090", "e2_mamba1_vgpu3090.csv", "vgpu3090", "M1", "370M", "launch-inclusive", "headroom_noise.json",
     "Mamba-1 370M, Triton scan, RTX 3090, launch-inclusive"),
    ("mamba1_r3spin_vgpu3090", "e2_mamba1_vgpu3090_r3spin.csv", "vgpu3090", "M1", "370M", "device", "headroom_noise_r3.json",
     "Mamba-1 370M, Triton scan, RTX 3090, device time"),
    ("mamba1_h800", "e2_mamba1_h800.csv", "h800", "M1", "370M", "launch-inclusive", None,
     "Mamba-1 370M, Triton scan, H800 PCIe, launch-inclusive"),
    ("mamba1_r3spin_h800", "e2_mamba1_h800_r3spin.csv", "h800", "M1", "370M", "device", None,
     "Mamba-1 370M, Triton scan, H800 PCIe, device time"),
    ("refscan_vgpu3090", "e8_refscan_vgpu3090.csv", "vgpu3090", "Ref", "370M", "launch-inclusive", "headroom_noise.json",
     "Mamba-1 370M, reference CUDA scan, RTX 3090, launch-inclusive"),
    ("mamba1_1p4b_r3plain_vgpu3090", "e2_mamba1_vgpu3090_1p4b_r3plain.csv", "vgpu3090", "M1", "1.4B", "launch-inclusive",
     "headroom_noise_r3.json", "Mamba-1 1.4B, Triton scan, RTX 3090, launch-inclusive"),
    ("mamba1_1p4b_r3_vgpu3090", "e2_mamba1_vgpu3090_1p4b_r3.csv", "vgpu3090", "M1", "1.4B", "device",
     "headroom_noise_r3.json", "Mamba-1 1.4B, Triton scan, RTX 3090, device time"),
    ("mamba1_2p8b_r3plain_vgpu3090", "e2_mamba1_vgpu3090_2p8b_r3plain.csv", "vgpu3090", "M1", "2.8B", "launch-inclusive",
     "headroom_noise_r3.json", "Mamba-1 2.8B, Triton scan, RTX 3090, launch-inclusive"),
    ("mamba1_2p8b_r3_vgpu3090", "e2_mamba1_vgpu3090_2p8b_r3.csv", "vgpu3090", "M1", "2.8B", "device",
     "headroom_noise_r3.json", "Mamba-1 2.8B, Triton scan, RTX 3090, device time"),
    # superseded batch-1 sweep of the larger checkpoints (kept only to reproduce the published rows)
    ("mamba1_1p4b_vgpu3090", "e2_mamba1_vgpu3090_1p4b.csv", "vgpu3090", "M1", "1.4B", "launch-inclusive (superseded)",
     "headroom_noise.json", "Mamba-1 1.4B, Triton scan, RTX 3090, batch 1 (superseded)"),
    ("mamba1_2p8b_vgpu3090", "e2_mamba1_vgpu3090_2p8b.csv", "vgpu3090", "M1", "2.8B", "launch-inclusive (superseded)",
     "headroom_noise.json", "Mamba-1 2.8B, Triton scan, RTX 3090, batch 1 (superseded)"),
    # production kernels (same table; the new held-out maximum column)
    ("mamba2_pro6000", "e2_mamba2_pro6000.csv", "pro6000", "M2", "370M", "launch-inclusive", "headroom_noise.json",
     "Mamba-2 370M, SSD scan, RTX PRO 6000, launch-inclusive"),
    ("mamba2_vgpu3090", "e2_mamba2_vgpu3090.csv", "vgpu3090", "M2", "370M", "launch-inclusive", "headroom_noise.json",
     "Mamba-2 370M, SSD scan, RTX 3090, launch-inclusive"),
    ("mamba2_2p7b_r3_vgpu3090", "e2_mamba2_vgpu3090_2p7b_r3.csv", "vgpu3090", "M2", "2.7B", "device",
     "headroom_noise_e17.json", "Mamba-2 2.7B, SSD scan, RTX 3090, device time"),
    ("gdn_1p4b_r3_vgpu3090", "e2_gdn_vgpu3090_1p4b_r3.csv", "vgpu3090", "GDN", "1.4B", "device",
     "headroom_noise_e17.json", "Gated DeltaNet 1.4B, FLA chunk kernel, RTX 3090, device time"),
]
NEW_INDEX_BASE = 100        # RNG stream index of data sets that are not in headroom_noise.DATASETS

# Rows of Table S-headroom-noise as printed in the supplement (tpds-r3 state): in-sample mean, its largest
# shape, held-out all / covered, split-half covered mean [95% interval], above sync naive / split-half.
PUBLISHED_ROWS = {
    "mamba1_pro6000": ("0.34", "4.4", "0.65", "0.45", "0.13", "0.09", "0.17", 0, 0),
    "mamba1_vgpu3090": ("1.86", "6.7", "4.89", "2.16", "1.78", "1.72", "1.83", 0, 0),
    "mamba2_pro6000": ("4.75", "7.8", "5.89", "6.06", "4.48", "4.29", "4.66", 0, 0),
    "mamba2_vgpu3090": ("15.2", "43.0", "18.6", "17.2", "14.6", "14.2", "15.0", 5, 4),
    "refscan_vgpu3090": ("2.07", "14.5", "6.21", "2.16", "2.05", "1.99", "2.10", 0, 0),
    "mamba1_1p4b_vgpu3090": ("10.7", "220", "15.8", "7.05", "-2.4", "-3.8", "-1.2", 1, 1),
    "mamba1_2p8b_vgpu3090": ("86.4", "789", "120.6", "55.7", "7.7", "4.9", "10.6", 7, 5),
}
# Headroom column of Table S-conventions (held-out, covered calls, us per call)
PUBLISHED_CONVENTIONS = {"mamba1_vgpu3090": "2.16", "mamba1_r3spin_vgpu3090": "0.03",
                         "mamba1_1p4b_r3plain_vgpu3090": "2.59", "mamba1_1p4b_r3_vgpu3090": "0.18",
                         "mamba1_2p8b_r3plain_vgpu3090": "4.80", "mamba1_2p8b_r3_vgpu3090": "0.32",
                         "mamba2_vgpu3090": "17.20"}

NOMINAL_SYNC_MS = 0.020


# ----------------------------------------------------------------------------- helpers
def _us(v):
    return None if v is None or not np.isfinite(v) else float(1000.0 * v)


def _matches_printed(value: float, printed: str) -> bool:
    nd = len(printed.split(".")[1]) if "." in printed else 0
    return bool(abs(round(value, nd) - float(printed)) < 1e-9)


def _fmt_mean(x_us: float) -> str:
    return f"{x_us:.2f}" if abs(x_us) < 10 else f"{x_us:.1f}"


def _fmt_max(x_us: float) -> str:
    return f"{x_us:.1f}" if abs(x_us) < 100 else f"{x_us:.0f}"


def deep_compare(stored, new, path="", tol=1e-12, out=None):
    """Compare every leaf of ``stored`` with the same path in ``new``; returns (n_leaves, mismatches)."""
    if out is None:
        out = {"n": 0, "mismatch": []}
    if isinstance(stored, dict):
        for k, v in stored.items():
            if not isinstance(new, dict) or k not in new:
                out["n"] += 1
                out["mismatch"].append(f"{path}.{k}: missing")
                continue
            deep_compare(v, new[k], f"{path}.{k}", tol, out)
    elif isinstance(stored, list):
        if not isinstance(new, list) or len(new) != len(stored):
            out["n"] += 1
            out["mismatch"].append(f"{path}: list length")
            return out
        for i, (a, b) in enumerate(zip(stored, new)):
            deep_compare(a, b, f"{path}[{i}]", tol, out)
    else:
        out["n"] += 1
        if isinstance(stored, float) or isinstance(new, float):
            a = float("nan") if stored is None else float(stored)
            b = float("nan") if new is None else float(new)
            same = (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol * max(1.0, abs(a))
        else:
            same = stored == new
        if not same:
            out["mismatch"].append(f"{path}: {stored!r} != {new!r}")
    return out


def _jsonable(obj):
    """Round-trip through JSON so that stored and recomputed records are compared in the same types."""
    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    return json.loads(json.dumps(obj, default=default))


# ----------------------------------------------------------------------------- M8a
def device_cost_points(in_dir: Path) -> list[tuple[float, float]]:
    """(scan-input elements, marginal cost in ms) of one reduction kernel before the scan (RTX 3090)."""
    rows = json.loads((in_dir / "e9_placement_vgpu3090.json").read_text())["decision_cost"]
    return sorted((float(r["numel"]), float(r["marginal_ms"])) for r in rows)


def interp_device_cost(numel: np.ndarray, pts: list[tuple[float, float]]) -> np.ndarray:
    (n1, c1), (n2, c2) = pts[0], pts[-1]
    return np.maximum(c1, c1 + (c2 - c1) * (numel - n1) / (n2 - n1))


def split_half_shapes(st: HN.TableStructure, m: np.ndarray, se: np.ndarray, draws: int, rng) -> dict:
    """Per-shape split-half headroom with the random stream of ``headroom_noise.analyze_dataset``."""
    acc_in, acc_cv, cov_means = np.zeros(st.n_exact), np.zeros(st.n_exact), []
    for _ in range(draws):
        rng.standard_normal(m.shape)                      # E (bootstrap / null draws of analyze_dataset)
        D = rng.standard_normal(m.shape) * se
        hd = st.cross_fit(m + D, m - D)
        acc_in += st.per_shape(hd["in"])
        acc_cv += st.per_shape(hd["cv"])
        cov_means.append(float(hd["cv"][st.covered].mean()) if st.covered.any() else np.nan)
    return {"in": acc_in / draws, "cv": acc_cv / draws, "cv_covered_mean_ms": float(np.mean(cov_means))}


def headroom_row(name: str, csv: str, gpu: str, kernel: str, ckpt: str, conv: str, stored_file: str | None,
                 label: str, in_dir: Path, thresholds: dict, dev_pts, draws: int, seed: int,
                 new_idx: int) -> tuple[dict, dict]:
    hn_names = [d[0] for d in HN.DATASETS]
    idx = hn_names.index(name) if name in hn_names else new_idx
    path = in_dir / csv
    hn_label = HN.DATASETS[idx][1] if name in hn_names else label     # stored label, for the leaf comparison
    full = HN.analyze_dataset(hn_label, path, thresholds, draws, np.random.default_rng([seed, idx]))
    full["gpu"] = gpu
    full = _jsonable(full)

    # recompute the structure for the per-shape views
    df = HN.load_dataset(path)
    chunks = chunk_columns(df)
    m = latency_matrix(df, chunks)
    s = df[[f"lat_c{c}_std" for c in chunks]].to_numpy(dtype=float)
    se = s / np.sqrt(df["repeats"].to_numpy(dtype=float))[:, None]
    st = HN.TableStructure(df)
    hd = st.headroom(m, m)
    ps_in, ps_cv = st.per_shape(hd["in"]), st.per_shape(hd["cv"])
    sh = split_half_shapes(st, m, se, draws, np.random.default_rng([seed, idx]))
    split_check = abs(sh["cv_covered_mean_ms"] - full["split_half"]["cv_covered"]["mean_ms"]) if st.covered.any() else 0.0

    cov = st.shape_covered
    width = int(df["width"].iloc[0]) if "width" in df else None
    keys = st.exact_keys
    numel = np.array([L * b * (width or 0) for L, b in keys], dtype=float)
    dev_interp = interp_device_cost(numel, dev_pts)
    sync = thresholds.get("host_sync")

    def counts(ps: np.ndarray, mask: np.ndarray) -> dict:
        v = ps[mask]
        out = {"n_shapes": int(mask.sum()),
               "above_sync_nominal_20us": int((v > NOMINAL_SYNC_MS).sum()),
               "above_device_7p3us": int((v > dev_pts[0][1]).sum()),
               "above_device_42us": int((v > dev_pts[-1][1]).sum()),
               "above_device_interpolated": int((v > dev_interp[mask]).sum())}
        if sync is not None:
            out["above_host_sync_measured"] = int((v > sync).sum())
        return out

    def argmax_shape(ps: np.ndarray, mask: np.ndarray) -> dict:
        k = np.where(mask)[0][int(np.argmax(ps[mask]))]
        return {"L": keys[k][0], "batch": keys[k][1], "n_calls": int(st.shape_counts[k]), "headroom_us": _us(ps[k])}

    allm = np.ones(st.n_exact, dtype=bool)
    nv, sp = full["naive"], full["split_half"]
    ho_cov = nv["held_out"]["covered"]
    # does the largest held-out (covered) shape keep its in-sample table entry in every fold?
    k_in = int(np.argmax(ps_in))
    k_cv = int(np.where(cov)[0][int(np.argmax(ps_cv[cov]))])
    j_in, j_cv, _ = st.choose(m)
    sel_k = st.exact_id == k_cv
    same_entry = bool(np.all(j_in[sel_k] == j_cv[sel_k]))

    above_list = []
    for k in np.where(cov & (ps_cv > dev_pts[0][1]))[0]:
        above_list.append({"L": keys[k][0], "batch": keys[k][1], "n_calls": int(st.shape_counts[k]),
                           "in_sample_us": _us(ps_in[k]), "held_out_us": _us(ps_cv[k]),
                           "split_half_held_out_us": _us(sh["cv"][k]),
                           "device_cost_interpolated_us": _us(dev_interp[k])})
    above_list.sort(key=lambda r: -r["held_out_us"])

    row = {
        "label": label, "kernel": kernel, "checkpoint": ckpt, "gpu": gpu, "convention": conv, "source_table": csv,
        "stored_in": stored_file, "rng_stream_index": idx, "n_calls": full["n_calls"], "n_shapes": full["n_shapes"],
        "n_shapes_covered": full["n_shapes_covered"], "n_calls_covered": int(st.covered.sum()),
        "median_within_block_cv_pct": float(100 * np.median(s / m)),
        "host_sync_measured_us": _us(sync) if sync is not None else None,
        "in_sample": {"mean_us": nv["in_sample"]["all"]["mean_us"], "max_shape_us": nv["in_sample"]["all"]["max_shape_us"],
                      "argmax_shape": argmax_shape(ps_in, allm)},
        "held_out_all": {"mean_us": nv["held_out"]["all"]["mean_us"], "max_shape_us": nv["held_out"]["all"]["max_shape_us"]},
        "held_out_covered": {"mean_us": ho_cov.get("mean_us"), "max_shape_us": ho_cov.get("max_shape_us"),
                             "argmax_shape": argmax_shape(ps_cv, cov),
                             "argmax_equals_in_sample_argmax": bool(k_in == k_cv),
                             "argmax_shape_has_in_sample_entry_in_every_fold": same_entry},
        "split_half_covered": {"mean_us": sp["cv_covered"].get("mean_us"), "ci95_lo_us": sp["cv_covered"].get("ci95_lo_us"),
                               "ci95_hi_us": sp["cv_covered"].get("ci95_hi_us"),
                               "max_shape_of_draw_mean_us": sp["cv_covered"].get("max_shape_of_draw_mean_us"),
                               "replicated_mean_abs_diff_ms": float(split_check)},
        "shapes_above": {"in_sample_all_shapes": counts(ps_in, allm), "held_out_covered": counts(ps_cv, cov),
                         "held_out_fallback": counts(ps_cv, ~cov),
                         "split_half_held_out_covered": counts(sh["cv"], cov)},
        "covered_shapes_above_device_7p3us": above_list,
        "implementation_check_passed": full["implementation_check"]["passed"],
    }
    a = row["shapes_above"]
    sync_key = "above_host_sync_measured" if sync is not None else "above_sync_nominal_20us"
    row["table_row"] = (f"{_fmt_mean(row['in_sample']['mean_us'])} ({_fmt_max(row['in_sample']['max_shape_us'])}) & "
                        f"{_fmt_mean(row['held_out_all']['mean_us'])} / {_fmt_mean(row['held_out_covered']['mean_us'])} & "
                        f"{_fmt_max(row['held_out_covered']['max_shape_us'])} & "
                        f"{_fmt_mean(row['split_half_covered']['mean_us'])} [{_fmt_mean(row['split_half_covered']['ci95_lo_us'])}, "
                        f"{_fmt_mean(row['split_half_covered']['ci95_hi_us'])}] & "
                        f"{a['held_out_covered'][sync_key]} / {a['split_half_held_out_covered'][sync_key]} & "
                        f"{a['held_out_covered']['above_device_7p3us']} / {a['split_half_held_out_covered']['above_device_7p3us']}")
    return row, full


def check_published(rows: dict) -> list[dict]:
    out = []
    for name, pub in PUBLISHED_ROWS.items():
        r = rows.get(name)
        if r is None:
            continue
        a = r["shapes_above"]
        sync_key = "above_host_sync_measured"
        vals = [r["in_sample"]["mean_us"], r["in_sample"]["max_shape_us"], r["held_out_all"]["mean_us"],
                r["held_out_covered"]["mean_us"], r["split_half_covered"]["mean_us"],
                r["split_half_covered"]["ci95_lo_us"], r["split_half_covered"]["ci95_hi_us"]]
        ok = [_matches_printed(v, p) for v, p in zip(vals, pub[:7])]
        ok += [a["held_out_covered"][sync_key] == pub[7], a["split_half_held_out_covered"][sync_key] == pub[8]]
        out.append({"dataset": name, "printed": list(pub), "recomputed": [round(v, 4) for v in vals]
                    + [a["held_out_covered"][sync_key], a["split_half_held_out_covered"][sync_key]],
                    "match": bool(all(ok))})
    for name, pub in PUBLISHED_CONVENTIONS.items():
        r = rows.get(name)
        if r is None:
            continue
        v = r["held_out_covered"]["mean_us"]
        out.append({"dataset": name, "printed_conventions_headroom": pub, "recomputed": round(v, 4),
                    "match": _matches_printed(v, pub)})
    return out


# ----------------------------------------------------------------------------- N1.1, N1.2, N1.7
def n1_1_pipeline_r2(in_dir: Path) -> dict:
    out = {}
    for host in ("pro6000", "vgpu3090"):
        stored = json.loads((in_dir / f"cost_model_{host}.json").read_text())["V4_pipelines"]
        rec = {}
        for fam in ("mamba2", "gdn"):
            d = CM.pipe_points(host, in_dir, fam)
            th, r2 = CM.pipe_fit(d)
            r = {"fit_data": f"{len(d)} timings: " + ", ".join(f"{k} {v}" for k, v in d["src"].value_counts().sort_index().items()),
                 "r2_log_fit_data": r2, "params": dict(zip(CM.PIPE_PARAMS, map(float, th))),
                 "stored_r2_log_fit_data": stored[fam]["fit_r2_log"],
                 "params_match_stored": bool(all(abs(stored[fam]["params"][k] - v) <= 1e-12 * max(1, abs(v))
                                                 for k, v in zip(CM.PIPE_PARAMS, th)))}
            if fam == "mamba2":
                df = load_e2([in_dir / f"e2_mamba2_{host}.csv"])
                chunks = chunk_columns(df)
                lat = latency_matrix(df, chunks)
                P = np.stack([CM.pipe_predict(th, df["L"].to_numpy(float), df["batch"].to_numpy(float), c) for c in chunks], 1)
                q = CM.fit_quality(P.ravel(), lat.ravel())
                r.update({"captured_calls": f"{len(df)} calls x {len(chunks)} chunks = {q['n']} latencies",
                          "r2_log_captured_calls": q["r2_log"], "median_rel_err_captured_calls": q["median_rel_err"],
                          "stored_r2_log_captured_calls": stored[fam]["e2"]["curve_fit"]["r2_log"]})
            rec[fam] = r
        out[host] = rec
    return out


def n1_2_tp_p95(in_dir: Path) -> dict:
    out = {}
    for host in ("vgpu3090", "h800"):
        p = in_dir / f"e12_analysis_{host}.json"
        if not p.exists():
            continue
        res = json.loads(p.read_text())
        rec = {}
        for fam, fr in res["families"].items():
            if not fr or "per_shape" not in fr:
                continue
            chunks = fr["chunks"]
            widths = {r["d"] for r in fr["per_shape"]}
            sm = E12.shard_map(widths)
            shards = {d for d, lst in sm.items() if any(t > 1 for _, t in lst)}
            reg, tied = [], []
            for r in fr["per_shape"]:
                c = r.get("choice_cost_model")
                if r["d"] not in shards or c is None:
                    continue
                lat = np.asarray(r["lat_ms"], dtype=float)
                reg.append(lat[chunks.index(c)] / lat.min() - 1)
                tied.append(c in r["tied"])
            if not reg:
                continue
            reg = np.asarray(reg)
            stored = fr["pooled_shards"].get("cost_model") or {}
            rec[fam] = {"shard_widths": sorted(shards), "n_shapes": int(len(reg)),
                        "mean_regret_pct": float(100 * reg.mean()), "p95_regret_pct": float(100 * np.percentile(reg, 95)),
                        "max_regret_pct": float(100 * reg.max()), "tied_share": float(np.mean(tied)),
                        "stored_p95_regret_pct": stored.get("p95_regret_pct"),
                        "stored_mean_regret_pct": stored.get("mean_regret_pct")}
        out[host] = rec
    return out


def n1_7_loko(in_dir: Path, refit: bool) -> dict:
    out = {}
    for host in ("pro6000", "vgpu3090"):
        stored = json.loads((in_dir / f"cost_model_{host}.json").read_text())
        s = stored["V2_leave_one_key_out"]
        rec = {"stored": {k: s[k] for k in ("chunk_mean_regret_pct", "chunk_max_regret_pct", "config_mean_regret_pct",
                                            "config_max_regret_pct", "config_median_rank")},
               "n_keys": len(s["keys"]), "n_configs_per_key": sorted({int(k["n_configs"]) for k in s["keys"]})}
        if refit:
            hw = CM.hardware(host, in_dir)
            stats = CM.load_kernel_stats(in_dir, hw["arch"])
            points = CM.load_points(host, in_dir)
            f = CM.features(points, stats, hw)
            theta = np.array([stored["params"][k] for k in CM.PARAMS])
            lk = CM.loko(points.loc[f.index], f, stats, hw, theta)
            rec["refit"] = {k: lk[k] for k in rec["stored"]}
            rec["refit"]["config_regret_pct_per_key"] = [round(k["config_regret_pct"], 3) for k in lk["keys"]]
            rec["refit_matches_stored"] = bool(all(abs(lk[k] - s[k]) <= 1e-9 * max(1, abs(s[k])) for k in rec["stored"]))
        out[host] = rec
    return out


# ----------------------------------------------------------------------------- M3c
def work_terms(th, L, B, C, scale=1.0):
    F, a, b, g, p = th
    return scale * (B * L * (a + b * C) + B * np.ceil(L / C) * g)


def select_pipeline(th, L, B, chunks, avail, rule: str) -> np.ndarray:
    """Chunk index per call under one selection rule; ``avail`` masks chunks the kernel can run."""
    F, a, b, g, p = th
    W = np.stack([work_terms(th, L, B, c) for c in chunks], 1)
    P = (F ** p + W ** p) ** (1.0 / p)
    P = np.where(avail, P, np.inf)
    Wm = np.where(avail, W, np.inf)
    if rule == "work_argmin":
        return Wm.argmin(1)
    if rule.startswith("static_"):
        # a chunk the kernel cannot run at a shape is replaced by the next larger available chunk
        j = chunks.index(int(rule.split("_")[1]))
        ok = avail[:, j:]
        return j + np.where(ok.any(1), ok.argmax(1), 0)
    choice = P.argmin(1)
    if rule == "original":
        return choice
    pf = np.where(avail, P, np.nan)
    spread = np.nanmax(pf, 1) / np.nanmin(pf, 1) - 1.0
    tied = spread == 0.0 if rule == "launch_default" else spread <= 1e-6
    wb = np.array([b * c + g / c for c in chunks])
    j_wb = int(np.argmin(wb))
    alt = np.where(avail[:, j_wb], j_wb, choice)
    return np.where(tied, alt, choice)


def regret_summary(lat: np.ndarray, choice: np.ndarray, chunks: list[int], tied_sets=None, mask=None) -> dict:
    if mask is None:
        mask = np.ones(len(lat), dtype=bool)
    if not mask.any():
        return {"n": 0}
    lat, choice = lat[mask], choice[mask]
    orc = lat.min(1)
    picked = lat[np.arange(len(lat)), choice]
    reg = picked / orc - 1.0
    out = {"n": int(mask.sum()), "mean_regret_pct": float(100 * reg.mean()),
           "p95_regret_pct": float(100 * np.percentile(reg, 95)), "max_regret_pct": float(100 * reg.max()),
           "latency_weighted_regret_pct": float(100 * (picked - orc).sum() / orc.sum()),
           "mean_regret_us_per_call": float(1000 * (picked - orc).mean()),
           "chosen_distribution": {str(chunks[k]): int(v) for k, v in zip(*np.unique(choice, return_counts=True))}}
    if tied_sets is not None:
        ts = [t for t, mm in zip(tied_sets, mask) if mm]
        out["tied_with_oracle_share"] = float(np.mean([chunks[c] in t for c, t in zip(choice, ts)]))
    return out


RULES = ("original", "launch_default", "launch_default_near", "work_argmin", "static_32", "static_256")


def m3c_calls(th, csv: Path) -> dict:
    df = load_e2([csv])
    chunks = chunk_columns(df)
    lat = latency_matrix(df, chunks)
    avail = np.isfinite(lat)
    lat = np.where(avail, lat, np.inf)
    L, B = df["L"].to_numpy(float), df["batch"].to_numpy(float)
    tied_sets = [set(int(c) for c in str(t).split(";") if c) for t in df["tied_chunks"].fillna("")]
    orc = lat.min(1)
    small = (df["batch"] == 1).to_numpy() & (df["L"] <= 256).to_numpy()
    floor = 2.0 * float(np.median(orc[small]))
    above = orc > floor
    P_orig = select_pipeline(th, L, B, chunks, avail, "original")
    F, a, b, g, p = th
    W = np.stack([work_terms(th, L, B, c) for c in chunks], 1)
    Pm = np.where(avail, (F ** p + W ** p) ** (1.0 / p), np.nan)
    exact_tie = (np.nanmax(Pm, 1) == np.nanmin(Pm, 1))
    subsets = {"all": None, "model_launch_bound": exact_tie, "above_floor": above, "below_floor": ~above}
    res = {"source": csv.name, "n_calls": int(len(df)), "chunks": chunks,
           "unavailable_cells": int((~avail).sum()),
           "floor_ms_2x_median_batch1_L_le_256": floor, "median_batch1_L_le_256_ms": floor / 2,
           "n_above_floor": int(above.sum()), "n_model_launch_bound_exact_tie": int(exact_tie.sum()),
           "n_model_launch_bound_exact_tie_above_floor": int((exact_tie & above).sum()),
           "max_tokens_BL_in_exact_tie": float((L * B)[exact_tie].max()) if exact_tie.any() else None,
           "min_tokens_BL_not_tied": float((L * B)[~exact_tie].min()) if (~exact_tie).any() else None,
           "original_choice_in_exact_ties": {str(chunks[k]): int(v) for k, v in
                                             zip(*np.unique(P_orig[exact_tie], return_counts=True))},
           "rules": {}}
    for rule in RULES:
        ch = select_pipeline(th, L, B, chunks, avail, rule)
        res["rules"][rule] = {s: regret_summary(lat, ch, chunks, tied_sets, m) for s, m in subsets.items()}
    return res


def m3c_synthetic(th, path: Path, tie_pct: float = 2.0) -> dict | None:
    if not path.exists():
        return None
    rows = [r for r in read_jsonl(path) if r.get("kind") == "timing"]
    syn = E17.synthetic(rows, None, tie_pct, 2.0)
    cells = [c for c in syn["cells"] if c["family"] == "ssd"]
    chunks = sorted({ch for c in cells for ch in c["chunks"]})
    lat = np.array([[c["mean_ms"].get(str(ch), np.inf) for ch in chunks] for c in cells], dtype=float)
    avail = np.isfinite(lat)
    L = np.array([c["L"] for c in cells], dtype=float)
    B = np.array([c["batch"] for c in cells], dtype=float)
    tied_sets = [set(c["tied_margin"]) for c in cells]
    widths = np.array([int(str(c["shape"]).lstrip("d")) for c in cells])
    F, a, b, g, p = th
    W = np.stack([work_terms(th, L, B, c) for c in chunks], 1)
    Pm = np.where(avail, (F ** p + W ** p) ** (1.0 / p), np.nan)
    exact_tie = np.nanmax(Pm, 1) == np.nanmin(Pm, 1)
    out = {"source": path.name, "n_cells": len(cells), "chunks": chunks, "rounds": sorted({c["rounds"] for c in cells}),
           "tie_margin_pct": tie_pct, "n_model_launch_bound_exact_tie": int(exact_tie.sum()),
           "max_tokens_BL_in_exact_tie": float((L * B)[exact_tie].max()) if exact_tie.any() else None,
           "rules": {}}
    for rule in RULES:
        ch = select_pipeline(th, L, B, chunks, avail, rule)
        rec = {"all": regret_summary(lat, ch, chunks, tied_sets),
               "model_launch_bound": regret_summary(lat, ch, chunks, tied_sets, exact_tie)}
        for w in sorted(set(widths)):
            rec[f"width_{w}"] = regret_summary(lat, ch, chunks, tied_sets, widths == w)
        out["rules"][rule] = rec
    return out


def m3c(in_dir: Path, floor_ms: float) -> dict:
    out = {"note": "selection rules: original = argmin of (F^p + W^p)^(1/p) (exact ties at F go to the first chunk, "
                   "32); launch_default = exact ties go to the work-bound optimum argmin_C (b C + g / C); "
                   "launch_default_near = ties within a relative 1e-6; work_argmin = argmin of W (no floor); "
                   "regrets are per-call relative means unless stated; latency_weighted = sum of excess latency / "
                   "sum of oracle latency.", "hosts": {}}
    for host, csvs in (("pro6000", ["e2_mamba2_pro6000.csv"]),
                       ("vgpu3090", ["e2_mamba2_vgpu3090.csv", "e2_mamba2_vgpu3090_r3spin.csv",
                                     "e2_mamba2_vgpu3090_2p7b_r3.csv"])):
        th, r2 = CM.pipe_fit(CM.pipe_points(host, in_dir, "mamba2"))
        F, a, b, g, p = th
        chunks_all = [32, 64, 128, 256, 512]
        wb = {c: b * c + g / c for c in chunks_all}
        c_wb = min(wb, key=wb.get)
        den_star = a + 2 * math.sqrt(b * g)
        den_256 = a + 256 * b + g / 256
        boundary = {"formula": "launch-bound while B*L < F / (a + 2 sqrt(b g)) (continuous optimum C* = sqrt(g/b)); "
                               "at chunk 256: F / (a + 256 b + g / 256); a, b, g held at the fitted values",
                    "fitted_F_ms": F, "C_star": math.sqrt(g / b), "work_bound_optimum_chunk": c_wb,
                    "BL_star_fitted_F": F / den_star, "BL_256_fitted_F": F / den_256,
                    "F_ms_lowered": floor_ms, "BL_star_lowered_F": floor_ms / den_star,
                    "BL_256_lowered_F": floor_ms / den_256,
                    "L_star_lowered_F_by_batch": {str(bb): floor_ms / den_star / bb for bb in (1, 4, 16)}}
        rec = {"params": dict(zip(CM.PIPE_PARAMS, map(float, th))), "fit_r2_log": r2, "boundary": boundary, "datasets": {}}
        for csv in csvs:
            pth = in_dir / csv
            if pth.exists():
                rec["datasets"][csv] = m3c_calls(th, pth)
        if host == "vgpu3090":
            syn = m3c_synthetic(th, in_dir / "e17_workbound_vgpu3090full.jsonl")
            if syn is not None:
                rec["datasets"]["e17_workbound_vgpu3090full.jsonl"] = syn
        out["hosts"][host] = rec
    return out


# ----------------------------------------------------------------------------- main
def print_report(res: dict) -> None:
    print("M8a  headroom rows (us/call): in-sample mean (max) & held-out all / covered & held-out max (covered) & "
          "split-half covered [95%] & above sync naive / split & above 7.3 us naive / split")
    for name, r in res["M8a"]["rows"].items():
        print(f"  {name:30s} {r['convention'][:18]:18s} {r['table_row']}   "
              f"[argmax in {r['in_sample']['argmax_shape']['L']}x{r['in_sample']['argmax_shape']['batch']}, "
              f"held-out {r['held_out_covered']['argmax_shape']['L']}x{r['held_out_covered']['argmax_shape']['batch']}]")
    chk = res["M8a"]["checks"]
    print(f"  stored-json leaf comparison: {chk['stored_leaves_compared']} leaves, {chk['stored_leaf_mismatches']} mismatches;"
          f" published rows: {sum(c['match'] for c in chk['published'])}/{len(chk['published'])} match;"
          f" split-half replication max diff {chk['split_half_replication_max_abs_diff_ms']:.2e} ms;"
          f" headroom_noise reference checks {chk['headroom_noise_reference_checks']}")
    for k, v in res["claims"].items():
        print(f"  {k}: {v}")
    for host, r in res["N1_1"].items():
        m = r["mamba2"]
        print(f"N1.1 {host}: R2 fit data {m['r2_log_fit_data']:.4f} ({m['fit_data']}); captured calls "
              f"{m['r2_log_captured_calls']:.4f} ({m['captured_calls']})")
    for host, r in res["N1_2"].items():
        for fam, v in r.items():
            print(f"N1.2 {host} {fam}: pooled shards mean {v['mean_regret_pct']:.2f}% p95 {v['p95_regret_pct']:.2f}% "
                  f"(stored {v['stored_p95_regret_pct']})")
    for host, r in res["N1_7"].items():
        v = r.get("refit", r["stored"])
        print(f"N1.7 {host}: LOKO config regret mean {v['config_mean_regret_pct']:.2f}% max {v['config_max_regret_pct']:.1f}% "
              f"median rank {v['config_median_rank']}; chunk {v['chunk_mean_regret_pct']:.2f}%"
              + (f"; refit matches stored: {r['refit_matches_stored']}" if "refit" in r else ""))
    for host, r in res["M3c"]["hosts"].items():
        bd = r["boundary"]
        print(f"M3c {host}: C*={bd['C_star']:.0f}, work-bound optimum chunk {bd['work_bound_optimum_chunk']}; BL* "
              f"{bd['BL_star_fitted_F']:.0f} at F={1000 * bd['fitted_F_ms']:.0f} us -> {bd['BL_star_lowered_F']:.0f} at "
              f"F={1000 * bd['F_ms_lowered']:.0f} us (chunk 256: {bd['BL_256_lowered_F']:.0f})")
        for ds, d in r["datasets"].items():
            parts = []
            for rule in RULES:
                a = d["rules"][rule]["all"]
                parts.append(f"{rule} {a['mean_regret_pct']:.2f}")
            extra = ""
            if "above_floor" in d["rules"]["original"]:
                extra = (f" | above floor ({d['n_above_floor']}): " + ", ".join(
                    f"{rule} {d['rules'][rule]['above_floor']['mean_regret_pct']:.2f}" for rule in RULES))
            print(f"   {ds}: " + ", ".join(parts) + extra)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--out", type=Path, default=None, help="output JSON (default: <in-dir>/headroom_r4.json)")
    p.add_argument("--draws", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--floor-us", type=float, default=320.0, help="lowered Mamba-2 launch floor for the M3c boundary")
    p.add_argument("--skip-loko", action="store_true", help="report the stored leave-one-key-out values without refitting")
    args = p.parse_args(argv)
    t0 = time.time()
    in_dir = args.in_dir

    thr_gpu = HN.host_thresholds(in_dir)
    thr_gpu["h800"] = {"host_sync": None, "device_reduction_L2048": None}     # no host-sync measurement on the H800
    dev_pts = device_cost_points(in_dir)
    extra_thr = {"sync_nominal_20us": NOMINAL_SYNC_MS, "device_reduction_7p3us": dev_pts[0][1],
                 "device_reduction_42us": dev_pts[-1][1]}

    rows, all_full, new_full, leaves, mism, split_diff = {}, {}, {}, 0, [], 0.0
    stored_cache = {}
    for k, (name, csv, gpu, kernel, ckpt, conv, stored_file, label) in enumerate(HEADROOM_SETS):
        if not (in_dir / csv).exists():
            continue
        thresholds = {**thr_gpu[gpu], **extra_thr}
        row, full = headroom_row(name, csv, gpu, kernel, ckpt, conv, stored_file, label, in_dir, thresholds, dev_pts,
                                 args.draws, args.seed, NEW_INDEX_BASE + k)
        rows[name] = row
        all_full[name] = full
        split_diff = max(split_diff, row["split_half_covered"]["replicated_mean_abs_diff_ms"])
        if stored_file is None:
            new_full[name] = full
            continue
        if stored_file not in stored_cache:
            stored_cache[stored_file] = json.loads((in_dir / stored_file).read_text())
        st = stored_cache[stored_file]["datasets"].get(name)
        if st is None:
            mism.append(f"{name}: not in {stored_file}")
            continue
        cmp = deep_compare(st, full)
        leaves += cmp["n"]
        mism += [f"{name}{x}" for x in cmp["mismatch"]]
        row["stored_leaf_mismatches"] = len(cmp["mismatch"])
    ref = HN.reference_checks(in_dir, all_full)      # e5b / e8 values against the recomputed 370M rows

    def r(name):
        return rows[name]

    dev_m = [r(n)["held_out_covered"] for n in ("mamba1_1p4b_r3_vgpu3090", "mamba1_2p8b_r3_vgpu3090")]
    li_m = [r(n)["held_out_covered"] for n in ("mamba1_1p4b_r3plain_vgpu3090", "mamba1_2p8b_r3plain_vgpu3090")]
    claims = {
        "VII-E 'at most 4.7 us at any tuned shape' (1.4B / 2.8B, device time)": {
            "held_out_covered_max_us": [d["max_shape_us"] for d in dev_m],
            "shapes": [d["argmax_shape"] for d in dev_m],
            "launch_inclusive_held_out_covered_max_us": [d["max_shape_us"] for d in li_m],
            "source": "headroom_noise_r3.json datasets mamba1_{1p4b,2p8b}_r3_vgpu3090 naive.held_out.covered.max_shape_us"},
        "VII-H 'at most 4.4, 6.7 and 14.5 us' (M1 PRO 6000, M1 RTX 3090, reference scan)": {
            n: {"in_sample_max_us": r(n)["in_sample"]["max_shape_us"],
                "held_out_covered_max_us": r(n)["held_out_covered"]["max_shape_us"],
                "held_out_all_max_us_incl_fallback": r(n)["held_out_all"]["max_shape_us"],
                "same_shape": r(n)["held_out_covered"]["argmax_equals_in_sample_argmax"],
                "shape_keeps_in_sample_entry_in_every_fold": r(n)["held_out_covered"]["argmax_shape_has_in_sample_entry_in_every_fold"]}
            for n in ("mamba1_pro6000", "mamba1_vgpu3090", "refscan_vgpu3090")},
    }
    res = {
        "experiment": "headroom-r4 (review tpds-r4 M8a, N1.1-N1.4, N1.7, M3c)", "seed": args.seed, "draws": args.draws,
        "units": "headroom in us per call; regrets in percent",
        "thresholds": {"host_sync_measured_us": {g: _us(t["host_sync"]) if t["host_sync"] is not None else None
                                                 for g, t in thr_gpu.items()},
                       "sync_nominal_us": 1000 * NOMINAL_SYNC_MS,
                       "device_reduction_points": [{"numel": n, "marginal_us": 1000 * c} for n, c in dev_pts],
                       "device_reduction_interpolation": "linear in batch x length x width between the two measured "
                                                         "sizes (RTX 3090, batch 1, width 2048, L = 2048 and 8192), "
                                                         "held at the smaller value below it (assumption)"},
        "M8a": {"columns": "in-sample mean (largest shape) & held-out all / covered & held-out per-shape maximum "
                           "(covered shapes) & split-half covered mean [95% interval] & covered shapes above one host "
                           "synchronization (measured on the GPU; nominal 20 us on the H800), naive / split-half & "
                           "covered shapes above the device-side reduction cost of 7.3 us, naive / split-half",
                "rows": rows,
                "checks": {"stored_leaves_compared": leaves, "stored_leaf_mismatches": len(mism),
                           "stored_mismatch_examples": mism[:20], "published": check_published(rows),
                           "split_half_replication_max_abs_diff_ms": split_diff,
                           "headroom_noise_reference_checks": f"{sum(c['match'] for c in ref)}/{len(ref)} match"},
                "full_analysis_new_datasets": new_full},
        "claims": claims,
        "N1_1": n1_1_pipeline_r2(in_dir),
        "N1_2": n1_2_tp_p95(in_dir),
        "N1_7": n1_7_loko(in_dir, refit=not args.skip_loko),
        "M3c": m3c(in_dir, args.floor_us / 1000.0),
    }
    res["runtime_s"] = round(time.time() - t0, 1)
    out = args.out or (in_dir / "headroom_r4.json")
    write_json(out, res)
    print_report(res)
    print(f"\nwrote {out} ({res['runtime_s']} s)")


if __name__ == "__main__":
    main()
