"""Cross-session and cross-board reproducibility of the production kernels' graph replays (taco-r2 M5f; CPU only).

Inputs (stored results):
  * the two E30 sessions on the stock 24 GB RTX 3090 of the lab host (``e30_autotune_state_<tag1|tag2>_<model>``;
    CUDA-graph replay, three rounds, round medians, the "pinned" autotuner state, which is the state of the sweeps),
  * the three-round CUDA-graph sweep of the same calls on the non-stock 48 GB RTX 3090 cloud board
    (``e2_<kind>_vgpu3090_<...>_r4graph3.csv``, round medians).
Calls are joined on the sweep's call key. Per model and comparison the script reports the share of calls with the
same oracle chunk, the regret of a table "tuned" in one place (its per-call oracle, and a held-out
(length bucket, batch) table) when it is served in the other (per-call mean and latency-weighted), the regret of
the library default in each place, and the latency ratio between the boards per (call, chunk).

Output: ``taco_r2_sessions.json`` in the results directory (and a printed summary).
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from src.experiments.tpds.common import RESULTS_DIR, read_jsonl

MODELS = {"ssd370m": ("e2_mamba2_vgpu3090_r4graph3.csv", 256),
          "ssd2p7b": ("e2_mamba2_vgpu3090_2p7b_r4graph3.csv", 256),
          "gdn1p4b": ("e2_gdn_vgpu3090_1p4b_r4graph3.csv", 64)}


def _parse(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def load_e30(tag: str, model: str) -> dict[str, dict]:
    rows = [r for r in read_jsonl(RESULTS_DIR / f"e30_autotune_state_{tag}_{model}.jsonl") if r.get("kind") == "call"]
    out = {}
    for r in rows:
        med = _parse(r["med_ms"])["pinned"]
        out[r["key"]] = {"lat": {int(c): float(v) for c, v in med.items() if v is not None and not math.isnan(float(v))},
                         "L": int(r["L"]), "batch": int(r["batch"]), "role": r.get("role")}
    return out


def load_sweep(csv_name: str, keys: set[str]) -> dict[str, dict]:
    from src.experiments.tpds.analysis import chunk_columns, load_e2, round_median_view

    df = load_e2([RESULTS_DIR / csv_name])
    if "oracle_chunk_med" in df.columns:
        df = round_median_view(df)
    df = df[df["key"].astype(str).isin(keys)]
    chunks = chunk_columns(df)
    out = {}
    for _, r in df.iterrows():
        out[str(r["key"])] = {"lat": {c: float(r[f"lat_c{c}_mean"]) for c in chunks if not math.isnan(r[f"lat_c{c}_mean"])},
                              "L": int(r["L"]), "batch": int(r["batch"])}
    return out


def _bucket(L: int) -> int:
    return 1 << max(0, int(math.floor(math.log2(max(L, 1)))))


def serve(tuned: dict, served: dict, chunk_of) -> dict:
    """Regret on ``served`` of the chunk ``chunk_of(key, tuned)`` (per-call mean and latency-weighted, %)."""
    reg, num, den, n = [], 0.0, 0.0, 0
    for k, s in served.items():
        c = chunk_of(k, tuned)
        if c is None or c not in s["lat"]:
            continue
        o = min(s["lat"].values())
        reg.append(s["lat"][c] / o - 1)
        num += s["lat"][c] - o
        den += o
        n += 1
    if not n:
        return {"n": 0}
    return {"n": n, "mean_pct": round(100 * statistics.fmean(reg), 3), "lw_pct": round(100 * num / den, 3)}


def oracle(k, src):
    lat = src.get(k, {}).get("lat")
    return min(lat, key=lambda c: (lat[c], c)) if lat else None


def bucket_table(src: dict):
    agg = defaultdict(lambda: defaultdict(float))
    for k, s in src.items():
        o = min(s["lat"].values())
        for c, v in s["lat"].items():
            agg[(_bucket(s["L"]), s["batch"])][c] += v / o
    table = {bk: min(v, key=lambda c: (v[c], c)) for bk, v in agg.items()}
    return lambda k, _src, table=table, ref=src: table.get((_bucket(ref[k]["L"]), ref[k]["batch"])) if k in ref else None


def compare(a: dict, b: dict, default: int) -> dict:
    common = sorted(set(a) & set(b))
    a = {k: a[k] for k in common}
    b = {k: b[k] for k in common}
    same = sum(1 for k in common if oracle(k, a) == oracle(k, b))
    ratios = [b[k]["lat"][c] / a[k]["lat"][c] for k in common for c in a[k]["lat"] if c in b[k]["lat"]]
    ratios.sort()
    q = lambda p: ratios[min(len(ratios) - 1, int(p * len(ratios)))]  # noqa: E731
    return {"n_calls": len(common), "same_oracle_share": round(same / max(len(common), 1), 3),
            "default_in_a": serve(a, a, lambda k, t: default), "default_in_b": serve(b, b, lambda k, t: default),
            "oracle_of_a_served_in_b": serve(a, b, oracle), "oracle_of_b_served_in_a": serve(b, a, oracle),
            "bucket_table_of_a_served_in_b": serve(a, b, bucket_table(a)),
            "latency_ratio_b_over_a": {"median": round(q(0.5), 4), "p5": round(q(0.05), 4), "p95": round(q(0.95), 4)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s1", default="4c_e30")
    ap.add_argument("--s2", default="4c_e30s2")
    a = ap.parse_args(argv)
    res = {"sessions": [a.s1, a.s2], "models": {}}
    for m, (csv_name, default) in MODELS.items():
        s1 = load_e30(a.s1, m)
        s2 = load_e30(a.s2, m) if (RESULTS_DIR / f"e30_autotune_state_{a.s2}_{m}.jsonl").exists() else {}
        sweep = load_sweep(csv_name, set(s1))
        out = {"stock_s1_vs_cloud_board": compare(sweep, s1, default)}
        if s2:
            out["stock_s1_vs_stock_s2"] = compare(s1, s2, default)
            out["cloud_board_vs_stock_s2"] = compare(sweep, s2, default)
        res["models"][m] = out
        print(m, json.dumps(out)[:900])
    (RESULTS_DIR / "taco_r2_sessions.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
