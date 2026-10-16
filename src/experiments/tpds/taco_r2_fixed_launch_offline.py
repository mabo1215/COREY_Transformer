"""taco-r2 item M2d (offline part): the Triton Mamba-1 chunk effect at a fixed launch configuration, from the
stored 184-configuration launch-space sweep at the twelve tuning keys (L in {128, 512, 2048, 8192} x batch
{1, 4, 16}, width 2048). CPU only; reads ``e4_<tag>.jsonl`` and times nothing.

Timings: the sweep's search timings (3 warm-up calls, the median of 10 repeats per configuration, taken once,
key after key, in a fixed configuration order), i.e. short autotuner-style timings, not the full protocol; the
six default-table entries were also re-measured under the full protocol (10 warm-ups, 100 repeats), which gives
the noise level of the search timings at those entries. A GPU re-timing of the whole space under the full
protocol is the counterpart (E34); ``e34_analysis_on_search_timings`` below applies E34's own analysis code
(``e34_fixed_launch.analyze``) to the search timings so that both can be compared one to one.

Per GPU and timing convention (device time primary: RTX PRO 6000 r4spin, RTX 3090 r3spin, H800 r3spin;
launch-inclusive secondary; RTX PRO 6000 with a flushed L2 as a control):

  fixed_launch        for every fixed (BLOCK_D, num_warps, num_stages): the best chunk per key and its counts
                      (movement of the optimum), the number of distinct optima, the spread over chunks
                      (slowest / fastest - 1; median and maximum over keys), the best single chunk across the keys
                      and its mean and maximum regret against the per-key best chunk at that launch, and how far
                      the launch's per-key best lies above the key's overall optimum
  default_table       the same along the default launch table (chunks 32/64/128 at (8, 2), 256/512 at (4, 2),
                      1024 at (4, 4), one stage), with search and with full-protocol timings
  best_launch_per_chunk  the chunk curve when every chunk gets its best launch configuration at each key
  retuned_table       a launch table retuned on this GPU (per chunk the launch with the lowest geometric-mean ratio
                      to that chunk's per-key best launch over the twelve keys) and the chunk effect along it
  shapley             two-player Shapley split of a gap between a 'from' and a 'to' configuration into a chunk part
                      and a launch part: chunk = 1/2 [(T(C_b, l_a) - T(C_a, l_a)) + (T(C_b, l_b) - T(C_a, l_b))],
                      launch = 1/2 [(T(C_a, l_b) - T(C_a, l_a)) + (T(C_b, l_b) - T(C_b, l_a))], chunk + launch =
                      T(b) - T(a); interaction = the chunk gain at the 'to' launch minus the chunk gain at the
                      'from' launch (the difference between the two orders; same sign as taco_model_baselines). Reported
                      (i) in % of the
                      'to' time (as 'x% slower than'; the manuscript's '33% = 29 + 3' convention) and (ii) in % of
                      the 'from' time (E34's convention), mean over the twelve keys. Pairs: chunk 256 at its
                      default launch -> best default-table entry (the manuscript's split); chunk 256 -> the overall
                      optimum of the 184 configurations; the free length rule clip(nextpow2(L), 32, 1024) at its
                      default launch -> the overall optimum; best default-table entry -> overall optimum.

Output: ``taco_r2_baselines_m2d.json``.

Usage:  PYTHONPATH=. python3 -m src.experiments.tpds.taco_r2_fixed_launch_offline [--scratch DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.experiments.tpds import cost_model as cm
from src.experiments.tpds import taco_r2_baselines_common as C
from src.experiments.tpds.common import RESULTS_DIR, read_jsonl

TAGS = {"device_time": {"pro6000": "pro6000_r4spin", "vgpu3090": "vgpu3090_r3spin", "h800": "h800_r3spin"},
        "launch_inclusive": {"pro6000": "pro6000", "vgpu3090": "vgpu3090", "h800": "h800"},
        "device_time_cold_l2": {"pro6000": "pro6000_r4cold"}}
CHUNKS = sorted(cm.DEFAULT_LAUNCH)
MAX_TILE = 16384
QUOTED = {  # reviewer R2's offline check and the manuscript's split
    "vgpu3090_fixed_4_2": {"best_chunk": 128, "n_keys_best": 11, "single_chunk_regret_pct": 0.0},
    "vgpu3090_default_table_single_chunk_regret_pct": 17.0,
    "h800_fixed_8_2": {"best_chunk": 128, "n_keys_best": 11, "single_chunk_regret_pct": 1.2},
    "h800_default_table_single_chunk_regret_pct": 21.0,
    "shapley_launch_inclusive_pro6000": (32.7, 29.3, 3.4),
    "shapley_device_time_pro6000": (44.46, 50.6, -6.2),   # review: '50.6 points chunk, -6.2 launch' (sum 44.46)
}


def dl(c: int) -> tuple:
    return (c, *cm.DEFAULT_LAUNCH[c], 1)


def load(tag: str, in_dir: Path) -> tuple[dict, dict, list]:
    rows = read_jsonl(in_dir / f"e4_{tag}.jsonl", kind="m1tune")
    T, full = {}, {}
    for r in rows:
        k = (int(r["L"]), int(r["batch"]))
        T[k] = {(s["block_t"], s["block_d"], s["num_warps"], s["num_stages"]): float(s["median_ms"])
                for s in r["search"] if s.get("median_ms")}
        full[k] = {c: float(r["final"][f"static_{c}"]["median_ms"]) for c in CHUNKS}
    return T, full, rows


def chunk_view(T: dict, cfg_of_chunk) -> dict:
    """Per key: chunk -> latency under a per-chunk configuration map ``cfg_of_chunk(key, chunk)`` (None: skip)."""
    out = {}
    for k, t in T.items():
        v = {}
        for c in CHUNKS:
            cfg = cfg_of_chunk(k, c)
            if cfg is not None and cfg in t:
                v[c] = t[cfg]
        out[k] = v
    return out


def chunk_stats(view: dict) -> dict:
    """Optimum movement, spread and single-chunk regret of a per-key chunk -> latency view."""
    keys = sorted(k for k in view if len(view[k]) >= 2)
    cs = sorted(set.intersection(*[set(view[k]) for k in keys])) if keys else []
    if len(cs) < 2:
        return {"n_keys": len(keys), "chunks": cs}
    best = {k: min(cs, key=lambda c: (view[k][c], c)) for k in keys}
    reg = {c: [view[k][c] / view[k][best[k]] - 1 for k in keys] for c in cs}
    c_star = min(cs, key=lambda c: (float(np.mean(reg[c])), c))
    spread = [max(view[k][c] for c in cs) / min(view[k][c] for c in cs) - 1 for k in keys]
    counts = {str(c): sum(1 for k in keys if best[k] == c) for c in cs}
    return {"n_keys": len(keys), "chunks": cs,
            "best_chunk_per_key": {f"L{k[0]}|B{k[1]}": best[k] for k in keys},
            "best_chunk_counts": counts, "n_distinct_optima": len({best[k] for k in keys}),
            "modal_best_chunk": int(max(cs, key=lambda c: (counts[str(c)], -c))),
            "n_keys_at_modal_best": max(counts.values()),
            "best_single_chunk": c_star, "best_single_chunk_regret_mean_pct": 100 * float(np.mean(reg[c_star])),
            "best_single_chunk_regret_max_pct": 100 * float(np.max(reg[c_star])),
            "single_chunk_regret_mean_pct": {str(c): 100 * float(np.mean(reg[c])) for c in cs},
            "spread_pct_median": 100 * float(np.median(spread)), "spread_pct_max": 100 * float(np.max(spread)),
            "spread_pct_per_key": {f"L{k[0]}|B{k[1]}": 100 * s for k, s in zip(keys, spread)}}


def shapley(S: dict, a: tuple, b: tuple) -> dict | None:
    ab, ba = (b[0],) + a[1:], (a[0],) + b[1:]
    if any(x not in S for x in (a, b, ab, ba)):
        return None
    fa, fb, fx, fy = S[a], S[b], S[ab], S[ba]      # fx = (C_b, launch_a), fy = (C_a, launch_b)
    chunk = 0.5 * ((fx - fa) + (fb - fy))            # change of latency due to the chunk (negative = faster)
    launch = 0.5 * ((fy - fa) + (fb - fx))
    inter = (fy - fb) - (fa - fx)                     # chunk gain at the 'to' launch minus that at the 'from' launch
    return {"from": list(a), "to": list(b), "from_ms": fa, "to_ms": fb,
            "gap_pct_of_to": 100 * (fa / fb - 1), "chunk_pct_of_to": -100 * chunk / fb, "launch_pct_of_to": -100 * launch / fb,
            "interaction_pct_of_to": 100 * inter / fb,
            "gap_pct_of_from": 100 * (1 - fb / fa), "chunk_pct_of_from": -100 * chunk / fa, "launch_pct_of_from": -100 * launch / fa,
            "chunk_first_pct_of_to": 100 * (fa - fx) / fb, "launch_first_pct_of_to": 100 * (fa - fy) / fb}


def shapley_all(T: dict) -> dict:
    out = {}
    for name in ("default256_to_best_default_table", "default256_to_optimum", "free_rule_to_optimum",
                 "free_rule_to_best_default_table", "best_default_table_to_optimum"):
        per = []
        for k, t in sorted(T.items()):
            bdt = dl(min(CHUNKS, key=lambda c: (t[dl(c)], c)))
            opt = min(t, key=lambda x: (t[x], x))
            fr = dl(int(min(max(2 ** math.ceil(math.log2(max(k[0], 1))), 32), 1024)))
            a = {"default256": dl(256), "free_rule": fr, "best_default_table": bdt}[name.split("_to_")[0]]
            b = {"best_default_table": bdt, "optimum": opt}[name.split("_to_")[1]]
            s = shapley(t, a, b)
            per.append({"key": f"L{k[0]}|B{k[1]}", **(s if s else {"missing_corner": True})})
        ok = [x for x in per if "missing_corner" not in x]
        agg = {"n_keys": len(ok), "n_keys_missing_corner": len(per) - len(ok)}
        for f in ("gap_pct_of_to", "chunk_pct_of_to", "launch_pct_of_to", "interaction_pct_of_to", "gap_pct_of_from",
                  "chunk_pct_of_from", "launch_pct_of_from", "chunk_first_pct_of_to", "launch_first_pct_of_to"):
            v = np.array([x[f] for x in ok])
            agg[f"mean_{f}"] = float(v.mean())
            agg[f"median_{f}"] = float(np.median(v))
        g = np.array([x["from_ms"] - x["to_ms"] for x in ok])
        agg["n_keys_from_equals_to"] = int(sum(x["from"] == x["to"] for x in ok))
        agg["chunk_share_of_summed_gap_ms"] = float(sum(x["chunk_pct_of_to"] * x["to_ms"] for x in ok) / 100 / g.sum()) \
            if g.sum() > 0 else None
        agg["per_key"] = [{kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in x.items()} for x in per]
        out[name] = agg
    return out


def retuned_table(T: dict) -> dict:
    """Per chunk, the launch with the lowest geometric mean over keys of T(c, l) / min_l T(c, l)."""
    keys = sorted(T)
    launches = sorted({cfg[1:] for t in T.values() for cfg in t})
    table = {}
    for c in CHUNKS:
        best_per_key = {k: min(v for cfg, v in T[k].items() if cfg[0] == c) for k in keys}
        score = {}
        for la in launches:
            if all((c, *la) in T[k] for k in keys):
                score[la] = float(np.exp(np.mean([math.log(T[k][(c, *la)] / best_per_key[k]) for k in keys])))
        la = min(score, key=lambda x: (score[x], x))
        table[c] = {"launch": list(la), "geomean_ratio_to_per_key_best_launch": score[la],
                    "default_launch": [*cm.DEFAULT_LAUNCH[c], 1],
                    "default_launch_geomean_ratio": score.get((*cm.DEFAULT_LAUNCH[c], 1))}
    return table


def noise_of_search(T: dict, full: dict) -> dict:
    """Search timing (median of 10) vs the full-protocol re-measurement (median of 100) at the 72 default entries."""
    r = np.array([T[k][dl(c)] / full[k][c] - 1 for k in T for c in CHUNKS])
    flips = sum(1 for k in T if min(CHUNKS, key=lambda c: T[k][dl(c)]) != min(CHUNKS, key=lambda c: full[k][c]))
    return {"n_pairs": int(len(r)), "median_abs_pct": 100 * float(np.median(np.abs(r))),
            "p90_abs_pct": 100 * float(np.percentile(np.abs(r), 90)), "max_abs_pct": 100 * float(np.max(np.abs(r))),
            "n_keys_best_default_table_chunk_differs": int(flips)}


def analyze_tag(tag: str, in_dir: Path, scratch: Path) -> dict:
    T, full, rows = load(tag, in_dir)
    launches = sorted({cfg[1:] for t in T.values() for cfg in t})
    fixed = {}
    for la in launches:
        st = chunk_stats(chunk_view(T, lambda k, c, la=la: (c, *la)))
        if st.get("n_keys", 0) and len(st.get("chunks", [])) >= 2:
            st["fixed_launch_best_vs_key_optimum_pct_mean"] = 100 * float(np.mean(
                [T[k][(st["best_chunk_per_key"][f"L{k[0]}|B{k[1]}"], *la)] / min(T[k].values()) - 1 for k in sorted(T)]))
            fixed[f"BD{la[0]}_nw{la[1]}_ns{la[2]}"] = st
    dtab_search = chunk_stats(chunk_view(T, lambda k, c: dl(c)))
    dtab_full = chunk_stats({k: dict(v) for k, v in full.items()})
    blpc = chunk_stats({k: {c: min(v for cfg, v in t.items() if cfg[0] == c) for c in CHUNKS} for k, t in T.items()})
    rt = retuned_table(T)
    rtab = chunk_stats(chunk_view(T, lambda k, c: (c, *rt[c]["launch"])))
    # summary over the fixed launches
    regs = np.array([v["best_single_chunk_regret_mean_pct"] for v in fixed.values()])
    nopt = np.array([v["n_distinct_optima"] for v in fixed.values()])
    spr = np.array([v["spread_pct_median"] for v in fixed.values()])
    at_default = {la: fixed.get(la) for la in ("BD8_nw2_ns1", "BD4_nw2_ns1", "BD4_nw4_ns1")}
    # the launch configurations that are fastest overall (where a practitioner would fix the launch)
    good = sorted(fixed, key=lambda la: fixed[la]["fixed_launch_best_vs_key_optimum_pct_mean"])[:5]
    # E34's own analysis on the search timings (written to the scratch directory only)
    jl = scratch / f"e34_fixed_launch_{tag}_search.jsonl"
    with jl.open("w") as f:
        for (L, b), t in sorted(T.items()):
            for cfg, v in t.items():
                f.write(json.dumps({"kind": "timing", "key": f"e34|L{L}|B{b}|r0|t{cfg[0]}_d{cfg[1]}_w{cfg[2]}_s{cfg[3]}",
                                    "L": L, "batch": b, "round": 0, "block_t": cfg[0], "block_d": cfg[1],
                                    "num_warps": cfg[2], "num_stages": cfg[3], "median_ms": v}) + "\n")
    from src.experiments.tpds import e34_fixed_launch as E34
    e34 = E34.analyze(SimpleNamespace(out_dir=scratch, tag=f"{tag}_search"))
    return {"tag": tag, "n_keys": len(T), "n_configurations_per_key": sorted({len(t) for t in T.values()}),
            "timing": "search medians (3 warm-ups, 10 repeats) for every configuration; default-table entries also "
                      "under the full protocol (10 warm-ups, 100 repeats)",
            "search_noise_at_default_entries": noise_of_search(T, full),
            "default_table_search": dtab_search, "default_table_full_protocol": dtab_full,
            "fixed_launch_at_default_table_launches": at_default,
            "fixed_launch_summary": {
                "n_launch_configurations": len(fixed),
                "best_single_chunk_regret_mean_pct": {"median": float(np.median(regs)), "min": float(regs.min()),
                                                      "max": float(regs.max())},
                "n_launches_single_chunk_regret_below_2pct": int((regs < 2).sum()),
                "n_launches_single_chunk_regret_below_5pct": int((regs < 5).sum()),
                "n_distinct_optima": {"median": float(np.median(nopt)), "min": int(nopt.min()), "max": int(nopt.max())},
                "n_launches_with_one_optimum": int((nopt == 1).sum()),
                "spread_pct_median_over_launches": float(np.median(spr)),
                "five_fastest_launches_by_mean_gap_to_key_optimum": {la: {k: fixed[la][k] for k in (
                    "fixed_launch_best_vs_key_optimum_pct_mean", "best_chunk_counts", "best_single_chunk",
                    "best_single_chunk_regret_mean_pct", "best_single_chunk_regret_max_pct", "spread_pct_median")}
                    for la in good}},
            "fixed_launch": fixed, "best_launch_per_chunk": blpc,
            "retuned_table": {"table": {str(c): v for c, v in rt.items()}, "chunk_effect": rtab},
            "shapley": shapley_all(T), "e34_analysis_on_search_timings": e34}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--scratch", type=Path, default=None, help="directory for the E34-format copies of the search timings")
    args = p.parse_args(argv)
    scratch = args.scratch or Path(tempfile.mkdtemp(prefix="taco_r2_m2d_"))
    scratch.mkdir(parents=True, exist_ok=True)
    checks = C.Checks()
    res = {"experiment": "taco-r2 M2d offline: chunk effect at fixed launch configuration (search timings)",
           "definitions": __doc__, "per_tag": {}}
    for conv, tags in TAGS.items():
        for gpu, tag in tags.items():
            if not (args.in_dir / f"e4_{tag}.jsonl").exists():
                continue
            r = analyze_tag(tag, args.in_dir, scratch)
            r["convention"], r["gpu"] = conv, gpu
            res["per_tag"][tag] = r
    # reproduction of the quoted values
    st = res["per_tag"]
    v = st["vgpu3090_r3spin"]["fixed_launch"]
    for ns in (1, 2):
        f = v[f"BD4_nw2_ns{ns}"]
        checks.add(f"R2: RTX 3090 fixed (4,2,{ns}) keys at chunk 128", f["best_chunk_counts"].get("128"), 11, "R2 M1.1", 0)
        checks.add(f"R2: RTX 3090 fixed (4,2,{ns}) single-chunk regret", f["best_single_chunk_regret_mean_pct"], 0.0, "R2 M1.1", 0.5)
    checks.add("R2: RTX 3090 default table single-chunk regret", st["vgpu3090_r3spin"]["default_table_search"]["best_single_chunk_regret_mean_pct"],
               17.0, "R2 M1.1", 0.5)
    h = st["h800_r3spin"]["fixed_launch"]
    for ns in (1, 2):
        f = h[f"BD8_nw2_ns{ns}"]
        checks.add(f"R2: H800 fixed (8,2,{ns}) keys at chunk 128", f["best_chunk_counts"].get("128"), 11, "R2 M1.1", 0)
        checks.add(f"R2: H800 fixed (8,2,{ns}) single-chunk regret", f["best_single_chunk_regret_mean_pct"], 1.2, "R2 M1.1", 0.05)
    checks.add("R2: H800 default table single-chunk regret", st["h800_r3spin"]["default_table_search"]["best_single_chunk_regret_mean_pct"],
               21.0, "R2 M1.1", 0.5)
    for tag, q in (("pro6000", QUOTED["shapley_launch_inclusive_pro6000"]), ("pro6000_r4spin", QUOTED["shapley_device_time_pro6000"])):
        s = st[tag]["shapley"]["default256_to_best_default_table"]
        checks.add(f"Shapley {tag} gap", s["mean_gap_pct_of_to"], q[0], "main text / review", 0.06)
        checks.add(f"Shapley {tag} chunk part", s["mean_chunk_pct_of_to"], q[1], "main text / review", 0.06)
        checks.add(f"Shapley {tag} launch part", s["mean_launch_pct_of_to"], q[2], "main text / review", 0.06)
    stored = json.loads((args.in_dir / "taco_model_baselines.json").read_text())["M3a"]["per_tag_summary"]
    for tag in st:
        if tag in stored:
            for mine, theirs in (("default256_to_best_default_table", "default256_to_best_default_table"),
                                 ("default256_to_optimum", "default256_to_best_overall"),
                                 ("best_default_table_to_optimum", "best_default_table_to_best_overall")):
                a, b = st[tag]["shapley"][mine], stored[tag]["shapley"][theirs]
                checks.add(f"{tag} {mine} chunk part vs taco_model_baselines", a["mean_chunk_pct_of_to"], b["mean_chunk_part_pct"],
                           "taco_model_baselines.json M3a", 1e-6)
                checks.add(f"{tag} {mine} launch part vs taco_model_baselines", a["mean_launch_pct_of_to"], b["mean_launch_part_pct"],
                           "taco_model_baselines.json M3a", 1e-6)
                checks.add(f"{tag} {mine} interaction vs taco_model_baselines", a["mean_interaction_pct_of_to"],
                           b["mean_interaction_pct"], "taco_model_baselines.json M3a", 1e-6)
    res["checks"] = checks.summary()
    res["scratch_dir_for_e34_format_copies"] = "temporary; not part of the results"
    path = C.save("m2d", res, args.in_dir)
    print(f"wrote {path}; checks {res['checks']['n_match']}/{res['checks']['n']}")
    for m in res["checks"]["mismatches"]:
        print("  MISMATCH", m["check"], m["computed"], m["reference"])
    for tag, r in st.items():
        d, fs = r["default_table_search"], r["fixed_launch_summary"]
        s1, s2 = r["shapley"]["default256_to_best_default_table"], r["shapley"]["default256_to_optimum"]
        s3 = r["shapley"]["free_rule_to_optimum"]
        print(f"{tag:16s} default table: counts {d['best_chunk_counts']} single {d['best_single_chunk']} "
              f"{d['best_single_chunk_regret_mean_pct']:.1f}% | fixed launches: single-chunk regret median "
              f"{fs['best_single_chunk_regret_mean_pct']['median']:.1f}% ({fs['n_launches_single_chunk_regret_below_2pct']}/"
              f"{fs['n_launch_configurations']} < 2%), distinct optima median {fs['n_distinct_optima']['median']}")
        print(f"{'':16s} Shapley 256->best table {s1['mean_gap_pct_of_to']:.1f} = {s1['mean_chunk_pct_of_to']:.1f} + "
              f"{s1['mean_launch_pct_of_to']:.1f}; 256->optimum {s2['mean_gap_pct_of_to']:.1f} = {s2['mean_chunk_pct_of_to']:.1f} + "
              f"{s2['mean_launch_pct_of_to']:.1f}; free rule->optimum {s3['mean_gap_pct_of_to']:.1f} = "
              f"{s3['mean_chunk_pct_of_to']:.1f} + {s3['mean_launch_pct_of_to']:.1f}")


if __name__ == "__main__":
    main()
