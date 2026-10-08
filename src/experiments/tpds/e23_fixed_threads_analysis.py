"""E23 analysis (review item M9a): the reference CUDA scan at a fixed thread count next to the stock pairing.

CPU only, deterministic; reads ``e8_refscan_<tag>.csv`` written by ``e8_refscan.py --thread-families 32``
(default tag ``vgpu3090_r4t32``: the 2360 captured 370M Mamba-1 calls, batch 1, 4 and 16, device time).

The stock dispatch of the reference scan pairs every chunk with a thread count (32 threads up to chunk 512,
64 at 1024, 128 at 2048), so a sweep over its chunks also varies the intra-block parallelism. The same
session timed the 32-thread family: chunks 128, 256 and 512 are the stock instantiations (already 32 threads),
1024 and 2048 are 32 threads x 32 / 64 items. Two views of the same calls are evaluated identically:

  stock     chunks {128, ..., 2048} with the stock thread pairing (``lat_c<C>``, ``oracle_chunk``,
            ``tied_chunks``);
  t32       the 32-thread family (``lat_c128..c512``, ``lat_t32c1024``, ``lat_t32c2048``;
            ``oracle_chunk_t32``, ``tied_chunks_t32``).

Per view: spread (slowest / fastest chunk per call), oracle distribution, regret of every single chunk and
of the best one, of the stock length rule (the chunk the stock code picks for the call's length, with the
view's thread count), the E3 predictors (grouped five-fold cross-validation over the prompt folds), the
exact-shape floor and the held-out exact-shape headroom (``analysis.cv_exact_headroom``, RTX 3090 host
synchronization as threshold), and the Eq. (5)-form reference-scan model (``cost_model.ref_predict`` with the
view's thread map). The published model of the stock pairing was fitted on the twelve tuning keys; this
session timed no tuning keys, so both views are fitted here the same way, on the captured calls of the
training folds (log latency of every chunk), and scored on the held-out fold. Also reported: the share of
the stock-pairing spread that remains at a fixed thread count.

Output: ``e23_fixed_threads_<tag>.json``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.experiments.tpds.analysis import cv_exact_headroom, latency_matrix, regret_stats
from src.experiments.tpds.common import RESULTS_DIR, utc_now, write_json
from src.experiments.tpds.cost_model import HARDWARE, REF_PARAMS, REF_THREADS, e2_eval, fit_quality, ref_predict
from src.experiments.tpds.e3_predictors import evaluate

CHUNKS = (128, 256, 512, 1024, 2048)
T32 = {c: 32 for c in CHUNKS}
HOST_SYNC_MS = 0.0214  # RTX 3090 host synchronization (e5b_vgpu3090.json)


def stock_chunk(L: int) -> int:
    for c in (128, 256, 512, 1024):
        if L <= c:
            return c
    return 2048


def view(calls: pd.DataFrame, name: str) -> pd.DataFrame:
    """Copy of the calls in the single-family E2 layout (``lat_c<C>_mean``, oracle and tie set of the view)."""
    v = calls.copy()
    if name == "t32":
        for c in (1024, 2048):
            for s in ("mean", "std", "ci_lo", "ci_hi"):
                v[f"lat_c{c}_{s}"] = calls[f"lat_t32c{c}_{s}"]
        v["oracle_chunk"] = calls["oracle_chunk_t32"]
        v["oracle_mean_ms"] = calls["oracle_mean_ms_t32"]
        v["tied_chunks"] = calls["tied_chunks_t32"]
    return v


def fit_ref(train: pd.DataFrame, hw: dict, threads: dict) -> np.ndarray:
    n = len(train)
    L = np.tile(train["L"].to_numpy(float), len(CHUNKS))
    B = np.tile(train["batch"].to_numpy(float), len(CHUNKS))
    d = np.tile(train["width"].to_numpy(float), len(CHUNKS))
    C = np.repeat(np.array(CHUNKS, float), n)
    y = np.log(np.concatenate([train[f"lat_c{c}_mean"].to_numpy() for c in CHUNKS]))
    lo, hi = [0, 0, 0, 0, 0, 0, 1, 1], [1, 0.1, 0.1, 0.01, 0.01, 0.1, 64, 256]
    best = None
    for x0 in ([0.01, 1e-3, 1e-4, 1e-5, 1e-5, 1e-4, 8, 16], [0.02, 1e-4, 1e-5, 1e-6, 1e-6, 1e-5, 4, 8],
               [0.005, 1e-3, 1e-3, 1e-4, 1e-4, 1e-3, 16, 32]):
        sol = least_squares(lambda th: np.log(ref_predict(th, L, B, d, C, hw, threads)) - y, x0, bounds=(lo, hi),
                            loss="soft_l1", f_scale=0.1, x_scale="jac", max_nfev=20000)
        if best is None or sol.cost < best.cost:
            best = sol
    return best.x


def predict_matrix(th, df: pd.DataFrame, hw: dict, threads: dict) -> np.ndarray:
    n = len(df)
    L = np.tile(df["L"].to_numpy(float), len(CHUNKS))
    B = np.tile(df["batch"].to_numpy(float), len(CHUNKS))
    d = np.tile(df["width"].to_numpy(float), len(CHUNKS))
    C = np.repeat(np.array(CHUNKS, float), n)
    return ref_predict(th, L, B, d, C, hw, threads).reshape(len(CHUNKS), n).T


def cost_model_cv(v: pd.DataFrame, hw: dict, threads: dict) -> dict:
    folds = v["fold"].to_numpy()
    P = np.full((len(v), len(CHUNKS)), np.nan)
    params = {}
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        th = fit_ref(v[tr], hw, threads)
        P[te] = predict_matrix(th, v[te], hw, threads)
        params[str(f)] = dict(zip(REF_PARAMS, map(float, th)))
    st = e2_eval(v, list(CHUNKS), np.nanargmin(P, 1), P)
    st["fold_params"] = params
    return st


def analyze_view(v: pd.DataFrame, name: str, hw: dict) -> dict:
    chunks = list(CHUNKS)
    lat = latency_matrix(v, chunks)
    orc = lat.min(1)
    norm = lat / orc[:, None]
    spread = lat.max(1) / orc
    tied = [set(int(c) for c in str(t).split(";") if c) for t in v["tied_chunks"].fillna("")]
    out = {"view": name, "n_calls": int(len(v)),
           "threads": {str(c): (32 if name == "t32" else REF_THREADS[c]) for c in chunks},
           "spread": {"median": float(np.median(spread)), "p95": float(np.percentile(spread, 95)),
                      "max": float(spread.max())},
           "oracle_distribution": {int(k): int(n) for k, n in v["oracle_chunk"].value_counts().sort_index().items()},
           "mean_tie_set_size": float(np.mean([len(t) for t in tied])),
           "oracle_latency_ms_median": float(np.median(orc))}
    out["static"] = {int(c): {**regret_stats(norm[:, j] - 1),
                              "tied_share": float(np.mean([c in t for t in tied]))} for j, c in enumerate(chunks)}
    best = min(out["static"], key=lambda c: out["static"][c]["mean_regret_pct"])
    out["best_single_chunk"] = {"chunk": best, **out["static"][best]}
    rule = np.array([chunks.index(stock_chunk(int(L))) for L in v["L"]])
    out["stock_length_rule"] = {**regret_stats(norm[np.arange(len(v)), rule] - 1),
                                "tied_share": float(np.mean([chunks[i] in t for i, t in zip(rule, tied)]))}
    out["predictors"] = evaluate(v, chunks)
    exact = np.zeros(len(v))
    for _, g in v.groupby(["L", "batch"]):
        i = g.index.to_numpy()
        exact[i] = norm[i, int(np.argmin(lat[i].mean(0)))] - 1
    out["exact_shape_floor"] = regret_stats(exact)
    out["headroom_exact_cv"] = cv_exact_headroom(v, chunks, threshold_ms=HOST_SYNC_MS)
    out["cost_model_cv"] = cost_model_cv(v, hw, T32 if name == "t32" else REF_THREADS)
    out["by_batch"] = {}
    for b, g in v.groupby("batch"):
        i = g.index.to_numpy()
        out["by_batch"][int(b)] = {
            "n": int(len(i)), "spread_median": float(np.median(spread[i])),
            "best_single_chunk_regret_pct": float(min(100 * (norm[i, j] - 1).mean() for j in range(len(chunks)))),
            "oracle_distribution": {int(k): int(n) for k, n in g["oracle_chunk"].value_counts().sort_index().items()}}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--tag", default="vgpu3090_r4t32")
    p.add_argument("--host", default="vgpu3090")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    df = pd.read_csv(args.in_dir / f"e8_refscan_{args.tag}.csv")
    calls = df[df["regimes"] != "e4_key"].reset_index(drop=True)
    hw = HARDWARE[args.host]
    res = {"experiment": "E23-analysis", "tag": args.tag, "created_utc": utc_now(),
           "timing": "device time (GPU pre-spin), one 50-repeat block per (call, chunk)",
           "views": {name: analyze_view(view(calls, name), name, hw) for name in ("stock", "t32")}}
    s, t = res["views"]["stock"], res["views"]["t32"]
    ls, lt = np.log(latency_matrix(view(calls, "stock"), list(CHUNKS))), np.log(latency_matrix(view(calls, "t32"),
                                                                                                 list(CHUNKS)))
    span_s, span_t = ls.max(1) - ls.min(1), lt.max(1) - lt.min(1)
    res["spread_share_remaining"] = {
        "median_ratio_log_spread": float(np.median(span_t / span_s)),
        "ratio_of_median_log_spread": float(np.median(span_t) / np.median(span_s)),
        "oracle_same_share": float((calls["oracle_chunk"] == calls["oracle_chunk_t32"]).mean()),
        "t32_oracle_faster_than_stock_oracle_share": float((calls["oracle_mean_ms_t32"] < calls["oracle_mean_ms"]).mean()),
        "t32_vs_stock_oracle_latency_ratio_median": float(np.median(calls["oracle_mean_ms_t32"] / calls["oracle_mean_ms"]))}
    out = args.out or args.in_dir / f"e23_fixed_threads_{args.tag}.json"
    write_json(out, res)
    for name, r in res["views"].items():
        pr = r["predictors"]
        print(f"{name:6s} spread median {r['spread']['median']:.2f}x p95 {r['spread']['p95']:.1f}x | oracle "
              f"{r['oracle_distribution']} | best single c{r['best_single_chunk']['chunk']} "
              f"{r['best_single_chunk']['mean_regret_pct']:.1f}% | stock rule {r['stock_length_rule']['mean_regret_pct']:.1f}% "
              f"(tied {r['stock_length_rule']['tied_share']:.0%}) | shape {pr['shape']['mean_regret_pct']:.2f}% "
              f"(tied {pr['shape']['tie_aware_accuracy']:.0%}) | entropy only {pr['entropy_only']['mean_regret_pct']:.1f}% "
              f"| shape+ent {pr['shape+entropy']['mean_regret_pct']:.2f}% | model {r['cost_model_cv']['mean_regret_pct']:.2f}% "
              f"(tied {r['cost_model_cv']['tie_aware_accuracy']:.0%}, R2 {r['cost_model_cv']['curve_fit'].get('r2_log', float('nan')):.3f}) "
              f"| floor {r['exact_shape_floor']['mean_regret_pct']:.2f}% | headroom {r['headroom_exact_cv']}")
    print("spread share remaining:", res["spread_share_remaining"])
    print("wrote", out)


if __name__ == "__main__":
    main()
