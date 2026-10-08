"""E20 analysis (review item M3a): cold-L2 re-timing on the RTX PRO 6000.

CPU only, deterministic, reads only stored result files.

Inputs (``--warm`` / ``--cold`` tags, defaults ``pro6000_r4spin`` / ``pro6000_r4cold``; both device time with a
GPU-side pre-spin, three rounds read through ``analysis.round_median_view``):
  warm  E18: the E4 Mamba-1 launch sweep with the twelve tuning keys (``e4_<warm>.jsonl``), the E1 Mamba-1
        medians (``e1_<warm>.jsonl``), the 2360 captured 370M Mamba-1 calls (``e2_mamba1_<warm>.csv``) and the
        cost model fitted on that calibration (``cost_model_<warm>.json``, ``cost_model.py --hosts <warm>``);
  cold  E20: the same E4 keys, E1 medians and the captured calls with L >= 2048, with the L2 overwritten
        before every timed repeat (``--flush-l2``, as Triton's ``do_bench``).

Reported (review M3a: "does the best chunk / the model's selection change with a cold L2?"):
  keys        per E4 tuning key: best static chunk warm and cold, cold / warm latency ratio per chunk, the
              regret under cold of the warm-best chunk, and whether the tuned launch configuration changes;
  calls       the E22 table summary (spread, oracle distribution, Table II per batch, E3 predictors with
              grouped 5-fold CV, held-out exact-shape table and headroom, Table I analog) of the cold calls and
              of the same calls warm, and the joined comparison (latency ratio by length bucket and batch,
              oracle agreement exact and tie-aware, regret under cold of the warm per-call oracle and of the
              warm-tuned exact-shape table, rank agreement of the per-call chunk profiles);
  cost_model  on the cold calls: the warm-calibrated frozen model, the model refitted on the cold
              calibration (same frozen form, ``cost_model.fit``), their fit quality on the cold calibration
              points, parameter ratios, and how often the two choose different chunks.
The launch floor of the cold table is taken from the full warm table (the cold sweep has no short calls), and
the headroom threshold is one host synchronization measured on the RTX PRO 6000 (``e5b_<gpu>.json``).

Output: ``e20_cold_analysis_<gpu>.json`` (or ``--out``).

Usage:  PYTHONPATH=. python3 -m src.experiments.tpds.e20_cold_analysis [--in-dir DIR] [--warm TAG] [--cold TAG]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.tpds import cost_model as cm
from src.experiments.tpds import e22_graph_analysis as E22
from src.experiments.tpds.analysis import chunk_columns, load_e2, round_median_view
from src.experiments.tpds.common import RESULTS_DIR, env_metadata, read_jsonl, write_json

CHECKPOINT = "Mamba-1 370M (Triton scan)"
SCAN_LAYERS = 48


def _gpu(tag: str) -> str:
    return tag.split("_")[0]


def _pct(x):
    return None if x is None or not np.isfinite(x) else float(100.0 * x)


# ----------------------------------------------------------------------------- tuning keys (E4)
def e4_keys(tag: str, in_dir: Path) -> dict:
    out = {}
    for r in read_jsonl(in_dir / f"e4_{tag}.jsonl", kind="m1tune"):
        static = {int(k.split("_")[1]): float(v["mean_ms"]) for k, v in r["final"].items()
                  if k.startswith("static_") and v.get("mean_ms")}
        out[r["key"]] = {"L": int(r["L"]), "batch": int(r["batch"]), "static_ms": static,
                         "tuned_ms": float(r["final"]["tuned"]["mean_ms"]) if r["final"].get("tuned") else None,
                         "best_config": r.get("best_config")}
    return out


def compare_keys(warm: dict, cold: dict) -> dict:
    rows = []
    for k in sorted(set(warm) & set(cold), key=lambda x: (warm[x]["L"], warm[x]["batch"])):
        w, c = warm[k], cold[k]
        chunks = sorted(set(w["static_ms"]) & set(c["static_ms"]))
        if not chunks:
            continue
        cw = min(chunks, key=lambda x: w["static_ms"][x])
        cc = min(chunks, key=lambda x: c["static_ms"][x])
        bw = {x: w["static_ms"][x] for x in chunks}
        bc = {x: c["static_ms"][x] for x in chunks}
        cfg_w = {kk: (w["best_config"] or {}).get(kk) for kk in ("block_t", "block_d", "num_warps", "num_stages")}
        cfg_c = {kk: (c["best_config"] or {}).get(kk) for kk in ("block_t", "block_d", "num_warps", "num_stages")}
        rows.append({"key": k, "L": w["L"], "batch": w["batch"], "best_chunk_warm": cw, "best_chunk_cold": cc,
                     "same_best_chunk": cw == cc,
                     "regret_cold_of_warm_best_pct": _pct(bc[cw] / bc[cc] - 1.0),
                     "cold_over_warm_ratio_by_chunk": {str(x): bc[x] / bw[x] for x in chunks},
                     "cold_over_warm_ratio_at_cold_best": bc[cc] / bw[cw],
                     "tuned_config_warm": cfg_w, "tuned_config_cold": cfg_c, "same_tuned_config": cfg_w == cfg_c,
                     "tuned_cold_over_warm": (c["tuned_ms"] / w["tuned_ms"]) if w["tuned_ms"] and c["tuned_ms"] else None})
    reg = [r["regret_cold_of_warm_best_pct"] for r in rows]
    return {"n_keys": len(rows), "n_same_best_chunk": int(sum(r["same_best_chunk"] for r in rows)),
            "n_same_tuned_config": int(sum(r["same_tuned_config"] for r in rows)),
            "regret_cold_of_warm_best_pct": {"mean": float(np.mean(reg)) if reg else None,
                                             "max": float(np.max(reg)) if reg else None},
            "keys": rows}


# ----------------------------------------------------------------------------- cost model on the cold calls
def cost_model_block(warm: str, cold: str, in_dir: Path, cold_df: pd.DataFrame, chunks: list[int]) -> dict:
    hw = cm.hardware(cold, in_dir)
    stats = cm.load_kernel_stats(in_dir, hw["arch"])
    stored = json.loads((in_dir / f"cost_model_{warm}.json").read_text())
    th_w = np.array([stored["params"][k] for k in cm.PARAMS])
    pts = cm.load_points(cold, in_dir)
    f = cm.features(pts, stats, hw)
    y = pts.loc[f.index, "lat"].to_numpy(float)
    th_c, info = cm.fit(f, y)
    out = {"warm_model_tag": warm, "hardware": hw, "n_cold_calibration_points": int(len(y)), "cold_fit": info,
           "calibration_fit_quality": {"warm_params_on_cold_points": cm.fit_quality(cm.predict(th_w, f), y),
                                       "cold_params_on_cold_points": cm.fit_quality(cm.predict(th_c, f), y)},
           "params_warm": dict(zip(cm.PARAMS, map(float, th_w))), "params_cold": dict(zip(cm.PARAMS, map(float, th_c))),
           "param_ratio_cold_over_warm": {k: (float(c / w) if w else None)
                                          for k, w, c in zip(cm.PARAMS, th_w, th_c)}}
    ch_w, P_w = cm.select_e2(th_w, cold_df, chunks, stats, hw)
    ch_c, P_c = cm.select_e2(th_c, cold_df, chunks, stats, hw)
    out["on_cold_calls"] = {"warm_model": cm.e2_eval(cold_df, chunks, ch_w, P_w),
                            "cold_refit_model": cm.e2_eval(cold_df, chunks, ch_c, P_c),
                            "share_calls_choice_differs": float((ch_w != ch_c).mean()),
                            "topk_warm_model": cm.topk_refinement(cold_df, chunks, P_w),
                            "topk_cold_refit_model": cm.topk_refinement(cold_df, chunks, P_c)}
    return out


# ----------------------------------------------------------------------------- run
def register_tables(warm: str, cold: str) -> tuple[str, str]:
    g = _gpu(warm)
    nw, nc = f"m1_{g}_warm", f"m1_{g}_cold"
    E22.TABLES[nw] = (f"e2_mamba1_{warm}.csv", "mamba1", CHECKPOINT, "device time, warm L2, three rounds",
                      g, SCAN_LAYERS, None, None, True, False)
    E22.TABLES[nc] = (f"e2_mamba1_{cold}.csv", "mamba1", CHECKPOINT,
                      "device time, L2 flushed before every repeat, three rounds (calls with L >= 2048)",
                      g, SCAN_LAYERS, None, nw, True, False)
    return nw, nc


def run(args) -> dict:
    t0 = time.time()
    in_dir, warm, cold = args.in_dir, args.warm, args.cold
    gpu = _gpu(cold)
    e5b = in_dir / f"e5b_{gpu}.json"
    if e5b.exists():
        E22.SYNC_MS = float(json.loads(e5b.read_text())["host_sync_ms"])
    nw, nc = register_tables(warm, cold)
    T_full = E22.Table(nw, in_dir)
    T_full.set_floor()
    T_cold = E22.Table(nc, in_dir)
    T_cold.set_floor(T_full.floor_ms)
    T_warm = E22.Table(nw, in_dir, keys=T_cold.df["key"])
    T_warm.set_floor(T_full.floor_ms)
    T_warm.name = f"{nw}_same_calls"
    res = {"experiment": "E20-analysis (review M3a: cold L2, RTX PRO 6000)", "meta": env_metadata(),
           "settings": {"warm": warm, "cold": cold, "sync_us": 1000 * E22.SYNC_MS, "floor_us": 1000 * T_full.floor_ms,
                        "floor": "2 x median oracle latency of the warm batch-1 calls with L <= 256",
                        "regret": "per-call relative regret T(c) / T(c*) - 1 in percent on round medians",
                        "tied": "chosen chunk in the call's between-round 95% CI tie set (tied_chunks_med)"},
           "n_calls": {"warm_all": int(len(T_full.df)), "cold": int(len(T_cold.df)),
                       "warm_same_calls": int(len(T_warm.df))}}
    res["keys"] = compare_keys(e4_keys(warm, in_dir), e4_keys(cold, in_dir))
    res["calls"] = {"cold": E22.summarize(T_cold, in_dir), "warm_same_calls": E22.summarize(T_warm, in_dir),
                    "cold_vs_warm": E22.compare(T_cold, T_warm)}
    cold_df = round_median_view(load_e2([in_dir / f"e2_mamba1_{cold}.csv"]))
    res["cost_model"] = cost_model_block(warm, cold, in_dir, cold_df, chunk_columns(cold_df))
    res["runtime_s"] = round(time.time() - t0, 1)
    return E22._clean(res)


def print_report(res: dict) -> None:
    k = res["keys"]
    print(f"E4 keys: best static chunk unchanged in {k['n_same_best_chunk']}/{k['n_keys']}, tuned config unchanged in "
          f"{k['n_same_tuned_config']}/{k['n_keys']}; regret under cold of the warm-best chunk mean "
          f"{k['regret_cold_of_warm_best_pct']['mean']} % max {k['regret_cold_of_warm_best_pct']['max']} %")
    c = res["calls"]["cold_vs_warm"]
    print(f"calls ({c['n_joined']} joined): oracle cold/warm latency median {c['oracle_latency_ratio_a_over_b']['median']:.3f}; "
          f"same oracle {c['oracle_same_share']:.2f}, cold oracle in warm tie set {c['oracle_a_in_tieset_b_share']:.2f}; "
          f"regret under cold of warm oracle {c['regret_under_a_pct']['per_call_oracle_of_b']['mean']:.2f} %, "
          f"of warm-tuned exact table {c['regret_under_a_pct']['exact_table_tuned_on_b_insample']['mean']:.2f} %")
    for name in ("cold", "warm_same_calls"):
        p = res["calls"][name]["policies"]
        h = res["calls"][name]["headroom"].get("all", {})
        print(f"  {name:16s} shape tree {p['e3_shape']['all']['mean_pct']:.2f} %, held-out exact table "
              f"{p['exact_table_heldout']['all']['mean_pct']:.2f} %, held-out headroom covered "
              f"{h.get('covered_us')} us (max shape {h.get('covered_max_key_us')} us)")
    m = res["cost_model"]["on_cold_calls"]
    print(f"cost model on cold calls: warm-calibrated {m['warm_model']['mean_regret_pct']:.2f} % "
          f"(p95 {m['warm_model']['p95_regret_pct']:.2f}), cold refit {m['cold_refit_model']['mean_regret_pct']:.2f} % "
          f"(p95 {m['cold_refit_model']['p95_regret_pct']:.2f}); choices differ on {m['share_calls_choice_differs']:.2f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--warm", default="pro6000_r4spin")
    p.add_argument("--cold", default="pro6000_r4cold")
    p.add_argument("--out", type=Path, default=None, help="output JSON (default: <in-dir>/e20_cold_analysis_<gpu>.json)")
    args = p.parse_args(argv)
    res = run(args)
    out = args.out or (args.in_dir / f"e20_cold_analysis_{_gpu(args.cold)}.json")
    write_json(out, res)
    print_report(res)
    print(f"\nwrote {out} ({res['runtime_s']} s)")


if __name__ == "__main__":
    main()
