"""Work-term model of the SSD and Gated DeltaNet pipelines calibrated on graph-timed keys (taco-r2 M4f; CPU only).

The work-term model of the multi-kernel pipelines (``cost_model.pipe_predict``: a launch floor F combined by a
smooth maximum with the work B L (a + b C) + B ceil(L/C) g) was calibrated on eager timings and loses 9 to 22% per
call under CUDA graphs. Here it is calibrated on the twelve tuning keys (L in {128, 512, 2048, 8192} x batch 1, 4, 16,
every chunk) timed as CUDA-graph replays on a stock RTX 3090 (E37, three rounds, round medians of the 10% trimmed
means), in two forms:

  plain    the model as in the paper;
  padded   the intra-chunk work counts the padded tokens, B ceil(L/C) C (a + b C), so that a call shorter than the
           chunk pays for a whole chunk (the padding term the paper's Limitations named).

Each fitted model selects the chunk of every captured call of the RTX 3090 CUDA-graph sweeps (the per-call tables of
Table 7, round medians), and the per-call mean and latency-weighted regret against the per-call oracle are reported,
with the library default for reference.

Output: ``taco_r2_graph_calibration.json`` in the results directory.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares

from src.experiments.tpds.analysis import chunk_columns, latency_matrix, load_e2, round_median_view
from src.experiments.tpds.common import RESULTS_DIR, read_jsonl

FAMILIES = {"ssd370m": ("e2_mamba2_vgpu3090_r4graph3.csv", "mamba2|", 256),
            "ssd2p7b": ("e2_mamba2_vgpu3090_2p7b_r4graph3.csv", "mamba2|", 256),
            "gdn1p4b": ("e2_gdn_vgpu3090_1p4b_r4graph3.csv", "gdn|", 64)}


def predict(th, L, B, C, padded: bool):
    F, a, b, g, p = th
    n = np.ceil(L / C)
    tokens = n * C if padded else L
    work = B * tokens * (a + b * C) + B * n * g
    return (F ** p + work ** p) ** (1.0 / p)


def fit(L, B, C, lat, padded: bool):
    y = np.log(lat)

    def res(th):
        return np.log(predict(th, L, B, C, padded)) - y

    sol = least_squares(res, [float(np.exp(y.min())), 1e-6, 1e-8, 1e-4, 4.0],
                        bounds=([0, 0, 0, 0, 1], [10, 1, 1, 1, 20]), loss="soft_l1", f_scale=0.1, x_scale="jac",
                        max_nfev=20000)
    return sol.x, float(1 - (res(sol.x) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def points(fam: str):
    rows = [r for r in read_jsonl(RESULTS_DIR / "e33_value_graph_4c_e37calib.jsonl")
            if r.get("kind") == "timing" and r["family"] == fam and r["design"] == "A"]
    by = defaultdict(list)
    for r in rows:
        by[(r["L"], r["batch"], r["chunk"])].append(r.get("tmean_ms", r["median_ms"]))
    keys = sorted(by)
    L = np.array([k[0] for k in keys], float)
    B = np.array([k[1] for k in keys], float)
    C = np.array([k[2] for k in keys], float)
    lat = np.array([statistics.median(by[k]) for k in keys])
    return L, B, C, lat


def evaluate(th, padded, csv_name, prefix, default):
    df = load_e2([RESULTS_DIR / csv_name])
    if "oracle_chunk_med" in df.columns:
        df = round_median_view(df)
    df = df[df["key"].astype(str).str.startswith(prefix)].reset_index(drop=True)
    chunks = chunk_columns(df)
    lat = latency_matrix(df, chunks)
    ok = np.isfinite(lat)
    P = np.stack([predict(th, df["L"].to_numpy(float), df["batch"].to_numpy(float), float(c), padded)
                  for c in chunks], 1)
    P = np.where(ok, P, np.inf)
    pick = P.argmin(1)
    orc = np.nanmin(np.where(ok, lat, np.nan), 1)
    sel = lat[np.arange(len(df)), pick]
    d = chunks.index(default)
    dl = lat[:, d]
    return {"n_calls": int(len(df)),
            "model_mean_regret_pct": float(100 * np.nanmean(sel / orc - 1)),
            "model_lw_pct": float(100 * (np.nansum(sel) / np.nansum(orc) - 1)),
            "default_mean_regret_pct": float(100 * np.nanmean(dl / orc - 1)),
            "default_lw_pct": float(100 * (np.nansum(dl) / np.nansum(orc) - 1)),
            "chosen_chunk_counts": {str(c): int((pick == i).sum()) for i, c in enumerate(chunks)}}


def main():
    res = {"source": "e33_value_graph_4c_e37calib.jsonl", "families": {}}
    for fam, (csv_name, prefix, default) in FAMILIES.items():
        L, B, C, lat = points(fam)
        if len(lat) == 0:
            continue
        out = {"n_points": int(len(lat))}
        for name, padded in (("plain", False), ("padded", True)):
            th, r2 = fit(L, B, C, lat, padded)
            ev = evaluate(th, padded, csv_name, prefix, default)
            out[name] = {"params": dict(zip(["F", "a", "b", "g", "p"], map(float, th))), "fit_r2_log": r2, **ev}
            print(f"{fam} {name}: R2 {r2:.3f}; model {ev['model_mean_regret_pct']:.2f}% per call "
                  f"(LW {ev['model_lw_pct']:.2f}%), default {ev['default_mean_regret_pct']:.2f}% "
                  f"(LW {ev['default_lw_pct']:.2f}%); choices {ev['chosen_chunk_counts']}")
        res["families"][fam] = out
    (RESULTS_DIR / "taco_r2_graph_calibration.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
