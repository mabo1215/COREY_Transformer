"""E5b - key-churn break-even of any online chunk policy against a shape-keyed tuning cache (T3).

Generalizes the E5 break-even from "entropy versus profiling" to "any online
policy versus a cache keyed on the call shape". Shape keys (length bucket x
batch) arrive in a stream. The cache tunes a key once, at cost P_k, and then
serves it by table lookup with per-call regret r_c; an online policy tunes
nothing but pays a decision overhead o on every call and cannot beat the
per-call oracle, so its regret is r_on >= r_c - h_k, where h_k is the oracle
headroom over the cache on that key (the no-gain bound of Theorem 1). Over the
N_k calls that a key receives in the lifetime of the workload, the online
policy is cheaper iff

    N_k * (o + r_on - r_c) < P_k,   hence necessarily   N_k < N_k* = P_k / (o - h_k).

N_k* is reported per key for three decision overheads (the host
synchronization that any host-side decision pays, the every-8th-element
histogram, the full histogram; all measured in E4 at the nearest tensor size)
and two tuning costs (profiling the six chunks, and the full launch-space
search; both without JIT compilation, scaled per key from the E4 wall clocks
by the key's summed chunk latency). With a zero-profiling model default for
unseen keys (the T1 cost model), P_k = 0 and the cache is cheaper at every
N_k as soon as the model's per-call regret is below o - h_k; this is reported
when ``cost_model_<host>.json`` is present.

Workload view: the number of distinct keys in the 2360 captured calls, the
calls per key, and the equivalent number of 48-layer requests.

Output: ``e5b_<host>.json``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.tpds.analysis import chunk_columns, latency_matrix, load_e2
from src.experiments.tpds.common import RESULTS_DIR, read_jsonl, write_json

LAYERS_PER_REQUEST = 48
OVERHEAD_GRID_MS = [round(float(x), 6) for x in np.logspace(-3, 2, 61)]


def _tuning_scale(e4_rows: list[dict]) -> dict:
    """Seconds of profiling / search per ms of summed static-chunk latency, fitted over the E4 keys."""
    rs = [r for r in e4_rows if r.get("kind") == "m1tune"]
    tot = np.array([sum(v["mean_ms"] for k, v in r["final"].items() if k.startswith("static_")) for r in rs])
    prof = np.array([r["profile_static_wall_s"] for r in rs])
    bench = np.array([r["tuning_bench_s"] for r in rs])
    # least squares through the origin
    a = float((tot * prof).sum() / (tot * tot).sum())
    b = float((tot * bench).sum() / (tot * tot).sum())
    return {"profile_s_per_ms": a, "search_s_per_ms": b,
            "profile_total_s": float(prof.sum()), "search_total_s": float(bench.sum()),
            "compile_total_s": float(sum(r["tuning_first_call_s"] for r in rs)),
            "n_keys": len(rs)}


def _overhead(e4_rows: list[dict], numel: int, stride: int) -> float | None:
    ov = [r for r in e4_rows if r.get("kind") == "overhead" and r["stride"] == stride]
    if not ov:
        return None
    best = min(ov, key=lambda r: abs(math.log(r["numel"]) - math.log(max(numel, 1))))
    return float(best["decision_wall"]["mean_ms"])


def _n_star(P_s: float, o_ms: float, h_ms: float) -> float | None:
    """Largest number of calls on a key for which an online policy can still be cheaper; None = unbounded."""
    gap = o_ms - h_ms
    if gap <= 0:
        return None
    return 1000.0 * P_s / gap


def analyze(host: str, in_dir: Path) -> dict:
    e4_rows = read_jsonl(in_dir / f"e4_{host}.jsonl")
    sync_ms = next(r["sync_wall"]["mean_ms"] for r in e4_rows if r.get("kind") == "sync")
    scale = _tuning_scale(e4_rows)
    df = load_e2([in_dir / f"e2_mamba1_{host}.csv"])
    chunks = chunk_columns(df)
    lat = latency_matrix(df, chunks)
    oracle = lat.min(1)
    model = None
    cm_path = in_dir / f"cost_model_{host}.json"
    if cm_path.exists():
        cm = json.loads(cm_path.read_text())
        choice = cm.get("per_call_choice", {}).get("mamba1")
        if choice and len(choice.get("keys", [])) == len(df):
            m = dict(zip(choice["keys"], choice["chunk"]))
            model = np.array([chunks.index(int(m[k])) for k in df["key"]])
    keys = []
    for (lb, b), g in df.groupby(["L_bucket", "batch"]):
        idx = g.index.to_numpy()
        table = int(np.argmin(lat[idx].mean(0)))     # in-sample shape table (E5: CV and in-sample agree)
        h = float((lat[idx, table] - oracle[idx]).mean())
        sum_lat = float(lat[idx].mean(0).sum())
        numel = int(b * int(g["width"].iloc[0]) * int(g["L"].median()))
        P = {"profile": scale["profile_s_per_ms"] * sum_lat, "search": scale["search_s_per_ms"] * sum_lat}
        o = {"host_sync": sync_ms, "sampled_histogram": _overhead(e4_rows, numel, 8),
             "full_histogram": _overhead(e4_rows, numel, 1)}
        rec = {"L_bucket": int(lb), "batch": int(b), "n_calls": int(len(idx)),
               "table_chunk": int(chunks[table]), "table_ms": float(lat[idx, table].mean()),
               "oracle_ms": float(oracle[idx].mean()), "headroom_ms": h,
               "headroom_pct": 100 * h / float(lat[idx, table].mean()),
               "P_s": P, "overhead_ms": o, "n_star_calls": {}, "n_star_requests": {}}
        for pk, pv in P.items():
            for ok, ov in o.items():
                ns = _n_star(pv, ov, h) if ov is not None else None
                rec["n_star_calls"][f"{pk}|{ok}"] = ns
                rec["n_star_requests"][f"{pk}|{ok}"] = None if ns is None else ns / LAYERS_PER_REQUEST
        if model is not None:
            reg_ms = float((lat[idx, model[idx]] - lat[idx, table]).mean())
            rec["model_regret_vs_table_ms"] = reg_ms
            rec["model_regret_vs_oracle_ms"] = float((lat[idx, model[idx]] - oracle[idx]).mean())
            # with the model as zero-cost default, online wins on this key only if o < model regret vs oracle
            rec["online_beats_model_default"] = {ok: (ov is not None and ov < rec["model_regret_vs_oracle_ms"])
                                                 for ok, ov in o.items()}
        keys.append(rec)

    # pooled view for the figure: N* versus o at the call-weighted mean headroom and mean per-key tuning cost
    w = np.array([k["n_calls"] for k in keys], dtype=float)
    h_pool = float(np.average([k["headroom_ms"] for k in keys], weights=w))
    P_pool = {pk: float(np.average([k["P_s"][pk] for k in keys], weights=w)) for pk in ("profile", "search")}
    curve = {pk: [_n_star(P_pool[pk], o, h_pool) for o in OVERHEAD_GRID_MS] for pk in P_pool}

    def _mins(name):
        vals = [k["n_star_calls"][name] for k in keys if k["n_star_calls"][name] is not None]
        return {"min_calls": float(min(vals)), "median_calls": float(np.median(vals)), "max_calls": float(max(vals)),
                "median_requests": float(np.median(vals)) / LAYERS_PER_REQUEST} if vals else None

    summary = {name: _mins(name) for name in keys[0]["n_star_calls"]}
    workload = {"n_calls": int(len(df)), "n_keys": len(keys),
                "calls_per_key_median": float(np.median(w)), "calls_per_key_min": int(w.min()),
                "calls_per_key_max": int(w.max()),
                "note": "captured calls cover 8 of 48 layers per prompt; a request contributes 48 calls to its key"}
    out = {"experiment": "E5b", "host": host, "layers_per_request": LAYERS_PER_REQUEST,
           "host_sync_ms": sync_ms, "tuning_scale": scale, "headroom_pooled_ms": h_pool,
           "P_pooled_s": P_pool, "summary_n_star": summary, "workload": workload, "keys": keys,
           "curve": {"overhead_ms": OVERHEAD_GRID_MS, "n_star_calls": curve}}
    if model is not None:
        out["model_default"] = {
            "mean_regret_vs_oracle_ms": float(np.average([k["model_regret_vs_oracle_ms"] for k in keys], weights=w)),
            "max_regret_vs_oracle_ms": float(max(k["model_regret_vs_oracle_ms"] for k in keys)),
            "keys_where_host_sync_online_beats_model": int(sum(k["online_beats_model_default"]["host_sync"] for k in keys)),
        }
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--hosts", default="pro6000,vgpu3090")
    args = p.parse_args(argv)
    for host in args.hosts.split(","):
        res = analyze(host, args.in_dir)
        out = args.in_dir / f"e5b_{host}.json"
        write_json(out, res)
        s = res["summary_n_star"]
        print(f"wrote {out}: {res['workload']['n_keys']} keys, pooled headroom {res['headroom_pooled_ms']*1000:.2f} us; "
              f"N* (profile, host sync) median {s['profile|host_sync']['median_calls']:.0f} calls; "
              f"(search, host sync) median {s['search|host_sync']['median_calls']:.0f} calls")


if __name__ == "__main__":
    main()
