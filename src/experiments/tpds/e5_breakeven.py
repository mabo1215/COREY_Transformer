"""E5 - break-even amortization of offline tuning versus online entropy scheduling (item C5).

Cost model for policy P over N scan calls:
    cost_P(N) = T_P + N * (l_P + o_P)
T_P: one-time tuning/profiling cost (E4 wall clock); l_P: mean per-call scan
latency; o_P: per-call decision overhead (E4 isolated measurement; zero for
static and offline-tuned policies, which are table lookups).

Policies:
  default_static  - fixed vendor chunk (256), T = 0
  profiled_static - one chunk chosen by profiling chunks only (T = chunk-only profiling cost)
  autotuned       - per-shape (L bucket, batch) best launch configuration (T = full search cost)
  entropy_online  - Eq. (2) per call, T = 0, o = measured decision overhead
  oracle          - per-call best chunk, no cost (unattainable bound)

Two views are reported:
  A. E4 keys (uniform mix over the tuned shapes): l from E4 re-measurements.
  B. E2 real traces: l from the per-call tables; the attainable headroom of any
     online chunk scheduler is l(best static) - l(oracle), which must exceed its
     per-call overhead for online adaptation to ever pay off.
Break-even N between policies A and B solves cost_A(N) = cost_B(N). Requests
are converted from calls with the number of scan layers per prefill (48).

Output: ``e5_<host>.json`` (break-even table and cost curves).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from src.experiments.tpds.analysis import chunk_columns, latency_matrix, load_e2
from src.experiments.tpds.common import RESULTS_DIR, write_json

LAYERS_PER_REQUEST = 48


def breakeven(T_a, per_a, T_b, per_b):
    """Smallest N >= 0 with cost_a(N) <= cost_b(N); None if a never catches up, 0 if a always wins."""
    dT = T_a - T_b          # extra up-front cost of a
    dper = per_b - per_a    # per-call saving of a
    if dT <= 0 and dper >= 0:
        return 0.0
    if dper <= 0:
        return None
    return dT / dper


def overhead_lookup(e4_rows, numel, stride):
    ov = [r for r in e4_rows if r.get("kind") == "overhead" and r["stride"] == stride]
    if not ov:
        return None
    best = min(ov, key=lambda r: abs(math.log(r["numel"]) - math.log(max(numel, 1))))
    return best["decision_wall"]["mean_ms"]


def view_e4(e4):
    rows = e4.get("rows", [])
    out = {}
    for fam, kind in (("mamba1", "m1tune"), ("mamba2", "m2tune")):
        rs = [r for r in rows if r.get("kind") == kind]
        if not rs:
            continue
        chunks = sorted(int(k.split("_")[1]) for k in rs[0]["final"] if k.startswith("static_"))
        # profiled static: one chunk minimizing total latency over the keys
        tot = {c: sum(r["final"][f"static_{c}"]["mean_ms"] for r in rs) for c in chunks}
        g = min(tot, key=tot.get)
        n = len(rs)
        l_default = sum(r["final"]["static_256"]["mean_ms"] for r in rs) / n if 256 in chunks else None
        l_prof = tot[g] / n
        l_tuned = sum(r["final"]["tuned"]["mean_ms"] for r in rs) / n
        l_rule = sum(r["final"]["entropy_rule"]["mean_ms"] for r in rs) / n
        numels = [r["L"] * r["batch"] * 2048 for r in rs]
        o_full = [overhead_lookup(rows, ne, 1) for ne in numels]
        o_samp = [overhead_lookup(rows, ne, 8) for ne in numels]
        o_full = sum(o_full) / n if all(v is not None for v in o_full) else None
        o_samp = sum(o_samp) / n if all(v is not None for v in o_samp) else None
        T_tune = sum(r["tuning_wall_s"] for r in rs) * 1e3
        T_prof = sum(r.get("profile_static_wall_s", r["tuning_wall_s"]) for r in rs) * 1e3
        pol = {
            "default_static": {"T_ms": 0.0, "per_call_ms": l_default},
            "profiled_static": {"T_ms": T_prof, "per_call_ms": l_prof, "chunk": g},
            "autotuned": {"T_ms": T_tune, "per_call_ms": l_tuned},
            "entropy_online_full": {"T_ms": 0.0, "per_call_ms": None if o_full is None else l_rule + o_full,
                                    "scan_ms": l_rule, "overhead_ms": o_full},
            "entropy_online_sampled": {"T_ms": 0.0, "per_call_ms": None if o_samp is None else l_rule + o_samp,
                                       "scan_ms": l_rule, "overhead_ms": o_samp},
        }
        out[fam] = {"n_keys": n, "policies": pol, "breakeven_calls": _pairwise(pol)}
    return out


def _pairwise(pol):
    res = {}
    names = [k for k, v in pol.items() if v.get("per_call_ms") is not None]
    for a in names:
        for b in names:
            if a == b:
                continue
            n = breakeven(pol[a]["T_ms"], pol[a]["per_call_ms"], pol[b]["T_ms"], pol[b]["per_call_ms"])
            res[f"{a}_vs_{b}"] = {"calls": n, "requests": None if n is None else n / LAYERS_PER_REQUEST}
    return res


def view_e2(csv_paths, e4_rows, T_tune_ms, T_prof_ms):
    df = load_e2(csv_paths)
    if df.empty:
        return None
    chunks = chunk_columns(df)
    lat = latency_matrix(df, chunks)
    l_oracle = float(lat.min(1).mean())
    per_chunk = lat.mean(0)
    g = int(np.argmin(per_chunk))
    l_static = float(per_chunk[g])
    rule = df["entropy_rule_chunk"].clip(min(chunks), max(chunks)).to_numpy()
    idx = np.array([chunks.index(int(c)) for c in rule])
    l_rule = float(lat[np.arange(len(df)), idx].mean())
    # per-shape table lookup (in-sample upper bound of any shape-keyed tuner)
    shape_best = 0.0
    for _, sub in df.groupby(["L_bucket", "batch", "layer"]):
        m = latency_matrix(sub, chunks)
        shape_best += float(m.mean(0).min()) * len(sub)
    l_shape = shape_best / len(df)
    numel = (df["L"] * df["batch"] * df["width"]).to_numpy()
    o_full = [overhead_lookup(e4_rows, int(ne), 1) for ne in numel]
    o = float(np.mean(o_full)) if o_full and all(v is not None for v in o_full) else None
    headroom = l_static - l_oracle
    pol = {"profiled_static": {"T_ms": T_prof_ms, "per_call_ms": l_static, "chunk": chunks[g]},
           "shape_keyed_table": {"T_ms": T_tune_ms, "per_call_ms": l_shape},
           "entropy_online_full": {"T_ms": 0.0, "per_call_ms": None if o is None else l_rule + o,
                                   "scan_ms": l_rule, "overhead_ms": o},
           "oracle": {"T_ms": 0.0, "per_call_ms": l_oracle}}
    return {"n_calls": int(len(df)), "policies": pol, "breakeven_calls": _pairwise(pol),
            "oracle_headroom_ms": headroom, "oracle_headroom_pct_of_static": 100 * headroom / l_static,
            "overhead_ms": o,
            "online_can_ever_win": None if o is None else bool(headroom > o),
            "min_per_call_gain_needed_ms": o,
            "note": "Any online chunk scheduler gains at most the oracle headroom per call; it pays off only if "
                    "that gain exceeds its own per-call decision overhead."}


def curves(pol, n_max=1e9):
    ns = np.unique(np.logspace(0, math.log10(n_max), 200).astype(np.int64))
    return {"N_calls": ns.tolist(),
            **{k: [(v["T_ms"] + n * v["per_call_ms"]) / 1e3 for n in ns] for k, v in pol.items()
               if v.get("per_call_ms") is not None}}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--host", required=True, help="host tag of the E2/E4 files (e.g. pro6000)")
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = args.out_dir or args.in_dir
    e4_path = args.in_dir / f"e4_{args.host}.json"
    e4 = json.loads(e4_path.read_text()) if e4_path.exists() else {"rows": []}
    res = {"experiment": "E5", "host": args.host, "layers_per_request": LAYERS_PER_REQUEST,
           "view_e4_keys": view_e4(e4), "view_e2_traces": {}}
    for fam in ("mamba1", "mamba2"):
        v4 = res["view_e4_keys"].get(fam)
        T_tune = v4["policies"]["autotuned"]["T_ms"] if v4 else float("nan")
        T_prof = v4["policies"]["profiled_static"]["T_ms"] if v4 else float("nan")
        v2 = view_e2([args.in_dir / f"e2_{fam}_{args.host}.csv"], e4.get("rows", []), T_tune, T_prof)
        if v2:
            res["view_e2_traces"][fam] = v2
            v2["cost_curves_s"] = curves(v2["policies"])
        if v4:
            v4["cost_curves_s"] = curves(v4["policies"])
    write_json(out_dir / f"e5_{args.host}.json", res)
    for fam, v in res["view_e2_traces"].items():
        print(f"{fam}: headroom {v['oracle_headroom_ms']:.4f} ms/call ({v['oracle_headroom_pct_of_static']:.1f}%), "
              f"entropy overhead {v['overhead_ms']} ms/call, online can win: {v['online_can_ever_win']}")
    for fam, v in res["view_e4_keys"].items():
        print(fam, {k: (None if x["calls"] is None else round(x["calls"])) for k, x in v["breakeven_calls"].items()
                    if k.startswith(("autotuned_vs_profiled", "entropy_online_full_vs_profiled",
                                     "autotuned_vs_default"))})


if __name__ == "__main__":
    main()
