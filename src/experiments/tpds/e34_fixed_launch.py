"""E34 - chunk effect of the Triton Mamba-1 scan at fixed launch configurations, full protocol (taco-r2 M2d).

The chunk sweeps vary the chunk along a default launch table that also changes the channel block and the
warp count (chunks 32/64/128 -> (BLOCK_D, num_warps) = (8, 2), 256/512 -> (4, 2), 1024 -> (4, 4)), so the
grid doubles between chunks 128 and 256. The launch-space search (E4) timed the joint space only briefly
(as an autotuner does). This experiment re-times the whole joint space (chunk x BLOCK_D x num_warps x
num_stages, BLOCK_T * BLOCK_D <= 16384; 184 configurations) under the full protocol at the twelve tuning
keys (L in {128, 512, 2048, 8192} x batch {1, 4, 16}, width 2048, state 16), so that the chunk effect can
be read at a fixed launch configuration and separated from the launch configuration.

Protocol: ``--rounds`` rounds; per round and key the configurations run in a seeded random order; every
configuration is ``--warmup`` calls and ``--repeats`` calls timed with CUDA events after a GPU-side
pre-spin (device time). Inputs: Gaussian synthetic inputs of the 370M shapes (latency of this kernel does
not depend on the input values at fixed placement; one buffer set per key). Exclusive-GPU guard and GPU
state as in the other experiments.

Resumable: one fsynced JSONL row per (key, round, configuration). ``--analyze`` writes
``e34_fixed_launch_<tag>.json``: per key the round-median latency of every configuration; per fixed
(BLOCK_D, num_warps, num_stages) the best chunk, the spread over chunks and the regret of each single
chunk across the keys; the same along the default table; the full-search optimum against the best
default-table entry; and a two-factor Shapley split of the default (chunk 256, default launch) to
optimum gap into a chunk part and a launch part.
"""
from __future__ import annotations

import argparse
import itertools
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.experiments.tpds.common import (MAMBA1_CHUNKS, LOG_DIR, RESULTS_DIR, JsonlWriter, Logger, check_disk,
                                         cuda_time, dedupe_by_key, env_metadata, foreign_gpu_pids, gpu_state,
                                         host_name, read_jsonl, utc_now, wait_for_exclusive_gpu, write_json)

BLOCK_DS = (4, 8, 16, 32)
WARPS = (1, 2, 4, 8)
STAGES = (1, 2)
MAX_TILE = 16384
DEFAULT_CHUNK = 256


def _g(x, digits=6):
    return None if x is None else float(f"{float(x):.{digits}g}")


def space(chunks):
    for bt, bd, nw, ns in itertools.product(chunks, BLOCK_DS, WARPS, STAGES):
        if bt * bd <= MAX_TILE:
            yield (bt, bd, nw, ns)


def cfg_name(c) -> str:
    return f"t{c[0]}_d{c[1]}_w{c[2]}_s{c[3]}"


def run(args, log) -> None:
    import torch

    from src.experiments.tpds import kernels as K

    jsonl = args.out_dir / f"e34_fixed_launch_{args.tag}.jsonl"
    check_disk(args.out_dir, need_gb=0.2, need_inodes=100, log=log)
    lengths = [int(x) for x in args.lengths.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    cfgs = list(space(MAMBA1_CHUNKS))
    meta = env_metadata()
    meta.update({"experiment": "E34", "n_configs": len(cfgs), "lengths": lengths, "batches": batches,
                 "rounds": args.rounds, "warmup": args.warmup, "repeats": args.repeats,
                 "pre_spin_ms": args.pre_spin_ms, "default_launch": {str(k): v for k, v in K.DEFAULT_MAMBA1_LAUNCH.items()},
                 "guard_gpu": args.guard_gpu, "args": {k: str(v) for k, v in vars(args).items()}})
    w = JsonlWriter(jsonl, meta=meta)
    log(f"E34 on {meta.get('gpu')}: {len(cfgs)} configurations x {len(lengths) * len(batches)} keys x "
        f"{args.rounds} rounds; {len(w.done)} rows done")
    for b, L in itertools.product(batches, lengths):
        inp = K.make_mamba1_inputs(b, 2048, 16, L, dtype=torch.float16, seed=args.seed)
        out = torch.empty_like(inp["u"])
        fns, failed = {}, {}
        for c in cfgs:
            cfg = K.LaunchConfig(*c)
            fn = (lambda cfg=cfg: K.mamba1_scan(inp["u"], inp["delta"], inp["A"], inp["B"], inp["C"], inp["D"],
                                                inp.get("z"), None, True, config=cfg, out=out))
            try:
                fn()
                torch.cuda.synchronize()
                fns[c] = fn
            except Exception as exc:  # resources exceeded for this configuration
                failed[c] = repr(exc)[:160]
        if failed:
            log(f"L={L} B={b}: {len(failed)} configurations failed to launch")
        for rnd in range(args.rounds):
            order = list(fns)
            random.Random(f"{args.seed}|{L}|{b}|{rnd}").shuffle(order)
            n_new = 0
            for c in order:
                key = f"e34|L{L}|B{b}|r{rnd}|{cfg_name(c)}"
                if key in w.done:
                    continue
                row: dict[str, Any] = {}
                if args.exclusive_guard:
                    waited = wait_for_exclusive_gpu(args.guard_until, None, index=args.guard_gpu)
                    if waited is None:
                        raise SystemExit("guard deadline passed while another process held the GPU")
                    row["guard_wait_s"] = _g(waited, 4)
                if args.record_gpu_state:
                    row["gpu_before"] = gpu_state(args.guard_gpu)
                ts = cuda_time(fns[c], warmup=args.warmup, repeats=args.repeats, pre_spin_ms=args.pre_spin_ms)
                if args.exclusive_guard and foreign_gpu_pids(args.guard_gpu):
                    wait_for_exclusive_gpu(args.guard_until, None, index=args.guard_gpu)
                    ts = cuda_time(fns[c], warmup=args.warmup, repeats=args.repeats, pre_spin_ms=args.pre_spin_ms)
                    row["retimed"] = True
                if args.record_gpu_state:
                    row["gpu_after"] = gpu_state(args.guard_gpu)
                row.update({"key": key, "kind": "timing", "L": L, "batch": b, "round": rnd, "block_t": c[0],
                            "block_d": c[1], "num_warps": c[2], "num_stages": c[3],
                            "median_ms": _g(statistics.median(ts)), "mean_ms": _g(statistics.fmean(ts)),
                            "min_ms": _g(min(ts)), "std_ms": _g(statistics.pstdev(ts)), "utc": utc_now()})
                w.write(row)
                n_new += 1
            log(f"L={L} B={b} round {rnd}: {n_new} configurations timed")
        for c, err in failed.items():
            key = f"e34|L{L}|B{b}|failed|{cfg_name(c)}"
            if key not in w.done:
                w.write({"key": key, "kind": "failed", "L": L, "batch": b, "block_t": c[0], "block_d": c[1],
                         "num_warps": c[2], "num_stages": c[3], "error": err})
        del inp, out, fns
        torch.cuda.empty_cache()
    w.close()
    try:
        analyze(args)
    except Exception as exc:
        log(f"analysis failed: {exc!r}")


# --------------------------------------------------------------------------- analysis
def analyze(args) -> dict[str, Any]:
    from src.experiments.tpds.kernels import DEFAULT_MAMBA1_LAUNCH

    rows = dedupe_by_key([r for r in read_jsonl(args.out_dir / f"e34_fixed_launch_{args.tag}.jsonl")
                          if r.get("kind") == "timing"])
    per: dict[tuple, dict[tuple, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[(r["L"], r["batch"])][(r["block_t"], r["block_d"], r["num_warps"], r["num_stages"])].append(r["median_ms"])
    T: dict[tuple, dict[tuple, float]] = {k: {c: statistics.median(v) for c, v in d.items()} for k, d in per.items()}
    keys = sorted(T)
    chunks = sorted({c[0] for k in keys for c in T[k]})
    launches = sorted({c[1:] for k in keys for c in T[k]})
    res: dict[str, Any] = {"experiment": "E34", "created_utc": utc_now(), "keys": [f"L{L}|B{b}" for L, b in keys],
                           "per_key": {}, "fixed_launch": {}, "default_table": {}, "summary": {}}
    dflt = {c: (c, *DEFAULT_MAMBA1_LAUNCH[c], 1) for c in chunks if c in DEFAULT_MAMBA1_LAUNCH}
    shap_chunk, shap_launch, totals = [], [], []
    for k in keys:
        t = T[k]
        opt = min(t, key=lambda c: (t[c], c))
        dtab = {c: t[cfg] for c, cfg in dflt.items() if cfg in t}
        best_dtab = min(dtab, key=lambda c: (dtab[c], c)) if dtab else None
        base = dflt.get(DEFAULT_CHUNK)
        entry: dict[str, Any] = {"n_configs": len(t), "optimum": cfg_name(opt), "optimum_ms": _g(t[opt], 5),
                                 "default_table_ms": {str(c): _g(v, 5) for c, v in sorted(dtab.items())},
                                 "default_table_best_chunk": best_dtab,
                                 "optimum_vs_best_default_table_pct": _g(100 * (dtab[best_dtab] / t[opt] - 1), 4)
                                 if best_dtab else None,
                                 "n_rounds_min": min(len(v) for v in per[k].values())}
        if base in t:
            l0, lstar, cstar = base[1:], opt[1:], opt[0]
            T00, Tss = t[base], t[opt]
            T_c0_ls = t.get((DEFAULT_CHUNK, *lstar))
            T_cs_l0 = t.get((cstar, *l0))
            if T_c0_ls is not None and T_cs_l0 is not None:
                chunk_first = (T00 - T_cs_l0)
                launch_first = (T00 - T_c0_ls)
                sc = 0.5 * (chunk_first + (T_c0_ls - Tss))
                sl = 0.5 * (launch_first + (T_cs_l0 - Tss))
                entry["shapley_pct_of_default"] = {"chunk": _g(100 * sc / T00, 4), "launch": _g(100 * sl / T00, 4),
                                                   "total": _g(100 * (T00 - Tss) / T00, 4)}
                shap_chunk.append(sc / T00)
                shap_launch.append(sl / T00)
                totals.append((T00 - Tss) / T00)
        res["per_key"][f"L{k[0]}|B{k[1]}"] = entry
    # fixed launch configurations: chunk optimum and single-chunk regret across keys
    for lc in launches:
        ok_keys = [k for k in keys if all((c, *lc) in T[k] for c in chunks if c * lc[0] <= MAX_TILE)]
        cs = [c for c in chunks if c * lc[0] <= MAX_TILE]
        if not ok_keys or len(cs) < 2:
            continue
        best = {k: min(cs, key=lambda c: (T[k][(c, *lc)], c)) for k in ok_keys}
        reg = {}
        for c in cs:
            r = [T[k][(c, *lc)] / T[k][(best[k], *lc)] - 1 for k in ok_keys]
            reg[c] = (statistics.fmean(r), max(r))
        c_star = min(reg, key=lambda c: (reg[c][0], c))
        spread = [max(T[k][(c, *lc)] for c in cs) / min(T[k][(c, *lc)] for c in cs) - 1 for k in ok_keys]
        res["fixed_launch"][f"d{lc[0]}_w{lc[1]}_s{lc[2]}"] = {
            "chunks": cs, "n_keys": len(ok_keys),
            "best_chunk_counts": {str(c): sum(1 for k in ok_keys if best[k] == c) for c in cs},
            "best_single_chunk": c_star, "best_single_chunk_regret_mean_pct": _g(100 * reg[c_star][0], 4),
            "best_single_chunk_regret_max_pct": _g(100 * reg[c_star][1], 4),
            "chunk_spread_pct_median": _g(100 * statistics.median(spread), 4),
            "chunk_spread_pct_max": _g(100 * max(spread), 4),
            "fixed_launch_best_vs_key_optimum_pct_mean": _g(100 * statistics.fmean(
                [T[k][(best[k], *lc)] / min(T[k].values()) - 1 for k in ok_keys]), 4)}
    # along the default table
    ok_keys = [k for k in keys if all(cfg in T[k] for cfg in dflt.values())]
    if ok_keys:
        best = {k: min(dflt, key=lambda c: (T[k][dflt[c]], c)) for k in ok_keys}
        reg = {c: [T[k][dflt[c]] / T[k][dflt[best[k]]] - 1 for k in ok_keys] for c in dflt}
        c_star = min(reg, key=lambda c: (statistics.fmean(reg[c]), c))
        res["default_table"] = {"n_keys": len(ok_keys),
                                "best_chunk_counts": {str(c): sum(1 for k in ok_keys if best[k] == c) for c in dflt},
                                "regret_mean_pct": {str(c): _g(100 * statistics.fmean(v), 4) for c, v in reg.items()},
                                "best_single_chunk": c_star,
                                "best_single_chunk_regret_mean_pct": _g(100 * statistics.fmean(reg[c_star]), 4),
                                "chunk_spread_pct_median": _g(100 * statistics.median(
                                    [max(T[k][cfg] for cfg in dflt.values()) / min(T[k][cfg] for cfg in dflt.values()) - 1
                                     for k in ok_keys]), 4)}
    if totals:
        res["summary"]["shapley_mean_pct_of_default"] = {"chunk": _g(100 * statistics.fmean(shap_chunk), 4),
                                                         "launch": _g(100 * statistics.fmean(shap_launch), 4),
                                                         "total": _g(100 * statistics.fmean(totals), 4)}
    res["summary"]["optimum_vs_best_default_table_pct"] = {
        "mean": _g(statistics.fmean([e["optimum_vs_best_default_table_pct"] for e in res["per_key"].values()
                                     if e["optimum_vs_best_default_table_pct"] is not None]), 4),
        "max": _g(max(e["optimum_vs_best_default_table_pct"] for e in res["per_key"].values()
                      if e["optimum_vs_best_default_table_pct"] is not None), 4)} if res["per_key"] else None
    path = args.out_dir / f"e34_fixed_launch_{args.tag}.json"
    write_json(path, res)
    print(f"wrote {path}")
    print("summary", res["summary"])
    print("default table", {k: v for k, v in res["default_table"].items() if k != "regret_mean_pct"})
    for lc, v in sorted(res["fixed_launch"].items(), key=lambda x: x[1]["best_single_chunk_regret_mean_pct"])[:6]:
        print(lc, v["best_chunk_counts"], v["best_single_chunk"], v["best_single_chunk_regret_mean_pct"])
    return res


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lengths", default="128,512,2048,8192")
    p.add_argument("--batches", default="1,4,16")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--pre-spin-ms", type=float, default=50.0)
    p.add_argument("--exclusive-guard", action="store_true")
    p.add_argument("--guard-until", type=float, default=None)
    p.add_argument("--guard-gpu", type=int, default=None)
    p.add_argument("--record-gpu-state", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--log-dir", type=Path, default=LOG_DIR)
    p.add_argument("--tag", default=None)
    p.add_argument("--analyze", action="store_true")
    a = p.parse_args(argv)
    a.tag = a.tag or f"{host_name()}_e34"
    if a.guard_gpu is None:
        import os

        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        a.guard_gpu = int(vis) if vis.isdigit() else 0
    return a


def main(argv=None):
    args = parse_args(argv)
    if args.analyze:
        analyze(args)
        return
    run(args, Logger(f"e34_fixed_launch_{args.tag}", args.log_dir))


if __name__ == "__main__":
    main()
