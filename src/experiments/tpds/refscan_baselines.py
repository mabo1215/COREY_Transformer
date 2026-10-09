"""Calibration-only chunk selectors on the reference Mamba-1 CUDA scan (review item M5b).

CPU only, deterministic; reads stored result files and times nothing.

Data: ``e8_refscan_<host>.csv`` (RTX 3090): the twelve E4 tuning keys (L in {128, 512, 2048, 8192} x
batch in {1, 4, 16}, d = 2048) timed at the five chunks of the runtime-chunk reference scan
(``regimes == "e4_key"``; 60 calibration timings) and the 2360 captured 370M calls timed at the same
chunks. No selector sees a captured call. Selectors, as in ``cost_model_baselines_r4`` where the kernel
allows it:
  cost_model        the reference-scan form of Eq. (5) (``cost_model.ref_predict``) with the parameters
                    stored in ``cost_model_refscan_<host>.json`` (fitted on the 60 calibration timings);
  nearest_key       best chunk of the nearest calibration key (``cost_model_baselines.nearest_key``);
  loglinear_interp  log latency of every chunk interpolated bilinearly in (log2 L, log2 B) over the
                    4 x 3 calibration grid (``cost_model_baselines_r4.LogLinearLB``);
  length_tree       regret-labelled tree on log2 L, depth 3, trained on the 12 calibration shapes
                    (``cost_model_baselines_r4.CalibTree``);
  gbr_basic / mlp_basic   gradient-boosted regressor / MLP of log latency on shape and chunk features
                    (log2 L, log2 B, log2 C, alignment, log2 ceil(L / C), log2 threads per block,
                    log2 items per thread); hyperparameters by leave-one-calibration-shape-out CV
                    (``cost_model_baselines_r4.train_learner``). The Triton kernel's compile facts and
                    structural features have no counterpart for this prebuilt CUDA kernel;
  tfla_runtime      TFLA's runtime expression with fitted constants (``cost_model_baselines_r4.fit_tfla``);
  cost_model_top2 / exact_shape_table   rank by the cost model and profile the best two chunks per exact
                    shape / profile all five (``cost_model.topk_refinement``);
  stock_dispatch    the library's length-based chunk rule, timed separately (reference only).
The three-parameter restriction of the Triton model uses compile facts and is not defined here.

Output: ``refscan_baselines_<host>.json``.

Usage:  python3 -m src.experiments.tpds.refscan_baselines [--host vgpu3090]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # deterministic BLAS

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.tpds import cost_model as cm  # noqa: E402
from src.experiments.tpds import cost_model_baselines as cmb  # noqa: E402
from src.experiments.tpds import cost_model_baselines_r4 as r4  # noqa: E402
from src.experiments.tpds.analysis import chunk_columns, latency_matrix  # noqa: E402
from src.experiments.tpds.common import RESULTS_DIR, utc_now, write_json  # noqa: E402

FEATURES = ["log2_L", "log2_B", "log2_C", "aligned", "log2_tiles", "log2_threads", "log2_items"]


class Shapes:
    """Minimal stand-in for ``cost_model_baselines_r4.Target``: shapes and chunks only."""

    def __init__(self, L, B, d, chunks):
        self.shapes = pd.DataFrame({"L": np.asarray(L, int), "B": np.asarray(B, int), "d": np.asarray(d, int)})
        self.chunks = list(chunks)


def features(L, B, C) -> pd.DataFrame:
    L, B, C = (np.asarray(v, float) for v in (L, B, C))
    thr = np.array([cm.REF_THREADS[int(c)] for c in C], float)
    return pd.DataFrame({"log2_L": np.log2(L), "log2_B": np.log2(B), "log2_C": np.log2(C),
                         "aligned": (L.astype(int) % 16 == 0).astype(float), "log2_tiles": np.log2(np.ceil(L / C)),
                         "log2_threads": np.log2(thr), "log2_items": np.log2(C / thr)})


def learned_scores(model, cols, t: Shapes) -> np.ndarray:
    n = len(t.shapes)
    L = np.tile(t.shapes["L"].to_numpy(), len(t.chunks))
    B = np.tile(t.shapes["B"].to_numpy(), len(t.chunks))
    C = np.repeat(np.array(t.chunks, float), n)
    return model.model.predict(features(L, B, C)[cols].to_numpy(float)).reshape(len(t.chunks), n).T


def analyze(in_dir: Path, host: str) -> dict:
    df = pd.read_csv(in_dir / f"e8_refscan_{host}.csv")
    keys_df = df[df["regimes"] == "e4_key"].reset_index(drop=True)
    calls = df[df["regimes"] != "e4_key"].reset_index(drop=True)
    chunks = chunk_columns(calls)
    assert chunks == sorted(cm.REF_THREADS), chunks
    hw = cm.HARDWARE[host]
    stored = json.loads((in_dir / f"cost_model_refscan_{host}.json").read_text())
    th = np.array([stored["params"][k] for k in cm.REF_PARAMS])
    lat_calls = latency_matrix(calls, chunks)

    # calibration keys (one row per E4 shape) and the long calibration table (shape x chunk)
    keys = [{"L": int(r.L), "B": int(r.batch), "d": int(r.width),
             "mean_ms": np.array([getattr(r, f"lat_c{c}_mean") for c in chunks], float)}
            for r in keys_df.itertuples(index=False)]
    pts = pd.DataFrame([{"L": k["L"], "B": k["B"], "d": k["d"], "C": c, "lat": k["mean_ms"][j], "shape": i}
                        for i, k in enumerate(keys) for j, c in enumerate(chunks)])
    t = Shapes(calls["L"], calls["batch"], calls["width"], chunks)
    n = len(calls)

    out, info = {}, {"n_calibration_shapes": len(keys), "n_calibration_timings": int(len(pts)), "n_calls": n,
                     "chunks": chunks, "threads_per_block": {str(c): cm.REF_THREADS[c] for c in chunks}}

    # cost model (stored parameters)
    Lc = np.tile(calls["L"].to_numpy(float), len(chunks))
    Bc = np.tile(calls["batch"].to_numpy(float), len(chunks))
    dc = np.tile(calls["width"].to_numpy(float), len(chunks))
    Cc = np.repeat(np.array(chunks, float), n)
    P = cm.ref_predict(th, Lc, Bc, dc, Cc, hw).reshape(len(chunks), n).T
    sel = {"cost_model": P}
    sel["nearest_key"] = np.stack([keys[cmb.nearest_key(int(L), int(B), int(d), keys)]["mean_ms"]
                                   for L, B, d in zip(calls["L"], calls["batch"], calls["width"])])
    sel["loglinear_interp"] = r4.LogLinearLB(keys).scores(t)
    tree = r4.CalibTree(3, ("log2_L",))
    tree.fit_keys(keys)
    sel["length_tree"] = tree.scores(t)
    X = features(pts["L"], pts["B"], pts["C"])
    y = np.log(pts["lat"].to_numpy(float))
    for kind in ("gbr", "mlp"):
        model, linfo = r4.train_learner(kind, X, y, pts["shape"].to_numpy(), FEATURES)
        sel[f"{kind}_basic"] = learned_scores(model, model.cols, t)
        info[f"{kind}_basic"] = linfo
    tf, tinfo = r4.fit_tfla(pts, "sum", chunks)
    sel["tfla_runtime"] = tf.scores(t)
    info["tfla_runtime"] = tinfo

    for name, S in sel.items():
        out[name] = r4.score_calls(calls, chunks, np.argmin(S, 1), S)
    topk = cm.topk_refinement(calls, chunks, P)
    for k_name, k_key in (("cost_model_top2", "top2"), ("exact_shape_table", "all_chunks_profiled")):
        if k_key in topk:
            out[k_name] = {kk: topk[k_key][kk] for kk in topk[k_key] if not isinstance(topk[k_key][kk], (dict, list))}
    # library default, timed separately in the same sweep
    if "lat_stock_mean" in calls:
        reg = calls["lat_stock_mean"].to_numpy(float) / lat_calls.min(1) - 1
        out["stock_dispatch"] = {"n": n, "mean_regret_pct": float(100 * reg.mean()),
                                 "p95_regret_pct": float(100 * np.percentile(reg, 95))}
    checks = {"cost_model_mean_regret_vs_stored_pp": abs(out["cost_model"]["mean_regret_pct"]
                                                          - stored["e8_calls"]["mean_regret_pct"])}
    return {"selectors": out, "info": info, "checks": checks}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--host", default="vgpu3090")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    t0 = time.time()
    res = {"experiment": "refscan_baselines", "item": "M5b", "host": args.host, "created_utc": utc_now()}
    res.update(analyze(args.in_dir, args.host))
    res["runtime_s"] = round(time.time() - t0, 1)
    out = args.out or args.in_dir / f"refscan_baselines_{args.host}.json"
    write_json(out, res)
    for name, s in res["selectors"].items():
        print(f"{name:20s} mean {s['mean_regret_pct']:6.2f}%  p95 {s.get('p95_regret_pct', float('nan')):6.1f}%"
              + (f"  tied {s['tie_aware_accuracy']:.2f}" if "tie_aware_accuracy" in s else ""))
    print("checks:", res["checks"], f"({res['runtime_s']} s)")


if __name__ == "__main__":
    main()
