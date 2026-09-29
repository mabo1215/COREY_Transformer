"""E8 - chunk sweep of the reference CUDA Mamba-1 selective scan on the captured calls (reviewer point 2).

Is the chunk sensitivity of Section VI a property of chunked scans or of the Triton kernel
written for this study? The reference CUDA scan of mamba_ssm fixes its chunk at compile time;
the runtime-chunk build in ``src/kernels/mamba1_runtime_chunk`` (module
``selective_scan_chunk_cuda``) runs the same kernel with the chunk as an argument. Every
captured call of E2 (same grouping, same keys, batch 1, 4 and 16) is timed at every supported
chunk {128, 256, 512, 1024, 2048}, with the installed stock ``selective_scan_cuda.fwd`` and with
the stock dispatch of the rebuilt module (same-binary baseline). Every chunk's output is
compared with the installed stock output on every call, which also validates the build at
every chunk. With ``--e4-keys`` the twelve E4 shape keys are timed as well (input to the cost
model refit).

The table has the E2 layout (features, oracle chunk, tied chunks, per-chunk latency columns),
so the E2/E3 analysis applies unchanged. The entropy rule is Eq. (2) over the routed range
[128, 2048].

``--validate-only`` compares every chunk with the stock forward on synthetic inputs whose
lengths are not a multiple of any chunk, and exits. ``--analyze`` (CPU) writes
``e8_analysis_<tag>.json``: chunk spread, static and stock-dispatch regret per batch size, the E3
predictors (grouped cross-validation), conditional mutual information, and the oracle headroom
over bucket, bucket-and-alignment and exact-shape tables (Theorem 1).

Output: ``e8_refscan_<tag>.csv`` (resumable, one fsynced row per call), ``e8_refscan_<tag>.json``
(metadata, validation, summary).
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch

from src.experiments.tpds import kernels as K
from src.experiments.tpds.capture_traces import load_manifest, load_params
from src.experiments.tpds.common import (DEFAULT_TRACE_DIR, RESULTS_DIR, CsvWriter, Logger, all_gpu_pids, check_disk,
                                         cuda_time, entropy_rule_chunk, env_metadata, foreign_gpu_pids, host_name,
                                         peak_vram_gb, register_own_gpu_pids, summarize, wait_for_exclusive_gpu,
                                         write_json)
from src.experiments.tpds.e2_oracle_sweep import FEATURES, activation_features, build_groups, length_bucket, stack_inputs

REF_CHUNKS = (128, 256, 512, 1024, 2048)
BASELINES = ("stock", "stock_rebuilt")


def columns(chunks):
    cols = ["key", "model", "host", "group_id", "prompt_ids", "regimes", "sources", "fold", "layer", "batch",
            "n_unique_prompts", "L", "L_bucket", "width", "d_state"] + FEATURES + [
        "entropy_rule_chunk", "oracle_chunk", "oracle_mean_ms", "tied_chunks", "max_out_diff_across_chunks",
        "max_out_diff_vs_stock", "stock_chunk", "feature_ms", "warmup", "repeats", "peak_vram_gb"]
    for name in [f"c{c}" for c in chunks] + list(BASELINES):
        cols += [f"lat_{name}_mean", f"lat_{name}_std", f"lat_{name}_ci_lo", f"lat_{name}_ci_hi"]
    return cols


def stock_chunk(L: int) -> int:
    """Chunk the stock dispatch uses for a sequence of length L."""
    for c in (128, 256, 512, 1024):
        if L <= c:
            return c
    return 2048


def _modules():
    import selective_scan_chunk_cuda  # type: ignore
    import selective_scan_cuda  # type: ignore

    return selective_scan_cuda, selective_scan_chunk_cuda


def make_variants(inp, chunks, mods):
    stock, patched = mods
    u, dt, A, D, z = inp["u"], inp["delta"], inp["A"], inp["D"], inp.get("z")
    B = inp["B"].unsqueeze(1).contiguous() if inp["B"].dim() == 3 else inp["B"]
    C = inp["C"].unsqueeze(1).contiguous() if inp["C"].dim() == 3 else inp["C"]

    def pick(res):
        return res[2] if z is not None else res[0]   # the gated output when z is given

    v = {"stock": lambda: pick(stock.fwd(u, dt, A, B, C, D, z, None, True)),
         "stock_rebuilt": lambda: pick(patched.fwd(u, dt, A, B, C, D, z, None, True))}
    for c in chunks:
        v[f"c{c}"] = (lambda c=c: pick(patched.fwd_with_chunk_size(u, dt, A, B, C, D, z, None, True, c)))
    return v


def time_variants(variants, order, warmup, repeats, guard, until, log):
    while True:
        if guard and wait_for_exclusive_gpu(until, log) is None:
            return None
        before = foreign_gpu_pids() if guard else None
        res = {name: summarize(cuda_time(variants[name], warmup, repeats)) for name in order}
        after = foreign_gpu_pids() if guard else None
        if not (before or after):
            return res
        log(f"foreign GPU processes {before or after} during the call; re-timing")


@torch.no_grad()
def validate(chunks, mods, log, lengths=(300, 1000, 3000, 5000), dim=64, bsz=2) -> dict:
    out = {"lengths": list(lengths), "dim": dim, "batch": bsz, "chunks": {}}
    ok = True
    for L in lengths:
        inp = K.make_mamba1_inputs(bsz, dim, 16, L, dtype=torch.float16, seed=L)
        v = make_variants(inp, chunks, mods)
        ref = v["stock"]().float()
        for name in ["stock_rebuilt"] + [f"c{c}" for c in chunks]:
            y = v[name]().float()
            d = float((y - ref).abs().max())
            rel = d / max(float(ref.abs().max()), 1e-12)
            rec = out["chunks"].setdefault(name, {"max_abs": 0.0, "max_rel": 0.0})
            rec["max_abs"], rec["max_rel"] = max(rec["max_abs"], d), max(rec["max_rel"], rel)
            ok &= rel < 1e-2
    out["passed"] = bool(ok)
    log(f"validation (fp16, L={list(lengths)}): " + ", ".join(f"{k} max|dy|={r['max_abs']:.2e} rel={r['max_rel']:.1e}"
                                                             for k, r in out["chunks"].items())
        + f" -> {'PASS' if ok else 'FAIL'}")
    return out


def analyze(csv_path: Path) -> dict:
    import numpy as np
    import pandas as pd

    from src.experiments.tpds.analysis import chunk_columns, conditional_mutual_information, latency_matrix, regret_stats
    from src.experiments.tpds.e3_predictors import evaluate

    df = pd.read_csv(csv_path)
    calls = df[df["regimes"] != "e4_key"].reset_index(drop=True)
    chunks = chunk_columns(calls)
    lat = latency_matrix(calls, chunks)
    orc = lat.min(1)
    stock = calls["lat_stock_mean"].to_numpy()
    tied = [set(int(c) for c in str(t).split(";") if c) for t in calls["tied_chunks"].fillna("")]
    spread = lat.max(1) / orc
    out = {"experiment": "E8-analysis", "n_calls": int(len(calls)), "chunks": chunks,
           "max_out_diff_vs_stock": float(df["max_out_diff_vs_stock"].max()),
           "spread": {"median": float(np.median(spread)), "p95": float(np.percentile(spread, 95)),
                      "max": float(spread.max())},
           "stock_dispatch": {**regret_stats(stock / orc - 1),
                              "tie_aware_accuracy": float(np.mean([c in t for c, t in zip(calls["stock_chunk"], tied)]))},
           "by_batch": {}, "predictors": evaluate(calls, chunks)}
    for b, g in calls.groupby("batch"):
        i = g.index.to_numpy()
        out["by_batch"][int(b)] = {
            "n": int(len(i)),
            "static_mean_regret_pct": {int(c): float(100 * (lat[i, j] / orc[i] - 1).mean()) for j, c in enumerate(chunks)},
            "stock_mean_regret_pct": float(100 * (stock[i] / orc[i] - 1).mean()),
            "oracle_distribution": {int(k): int(v) for k, v in g["oracle_chunk"].value_counts().items()}}
    Lb = np.log2(calls["L_bucket"].to_numpy()).astype(int)
    out["mi_entropy_oracle_given_Lbucket_batch"] = conditional_mutual_information(
        calls["entropy_bits"].to_numpy(), calls["oracle_chunk"].to_numpy(), Lb * 100 + calls["batch"].to_numpy())
    out["mi_logL_oracle_given_batch"] = conditional_mutual_information(
        np.log2(calls["L"].to_numpy()), calls["oracle_chunk"].to_numpy(), calls["batch"].to_numpy())
    calls["aligned"] = (calls["L"] % 16 == 0).astype(int)
    out["headroom"] = {}
    for name, keys in (("bucket", ["L_bucket", "batch"]), ("bucket_aligned", ["L_bucket", "batch", "aligned"]),
                       ("exact", ["L", "batch"])):
        h = np.zeros(len(calls))
        for _, g in calls.groupby(keys):
            i = g.index.to_numpy()
            h[i] = lat[i, int(np.argmin(lat[i].mean(0)))] - orc[i]
        per_key = pd.Series(h).groupby([calls[k] for k in keys]).mean()
        out["headroom"][name] = {"n_keys": int(len(per_key)), "mean_ms": float(h.mean()),
                                 "mean_pct": float(100 * (h / orc).mean()), "max_key_ms": float(per_key.max())}
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    p.add_argument("--batches", default="1,4,16")
    p.add_argument("--layers", default=None)
    p.add_argument("--chunks", default=",".join(map(str, REF_CHUNKS)))
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--e4-keys", action="store_true", help="also time the twelve E4 shape keys")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--analyze", action="store_true", help="only write the analysis of an existing table (CPU)")
    p.add_argument("--exclusive-guard", action="store_true")
    p.add_argument("--guard-until", type=float, default=None)
    p.add_argument("--max-rows", type=int, default=None, help="stop after this many new rows (smoke)")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--tag", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    tag = args.tag or host_name()
    if args.analyze:
        res = analyze(args.out_dir / f"e8_refscan_{tag}.csv")
        write_json(args.out_dir / f"e8_analysis_{tag}.json", res)
        pr = res["predictors"]
        print(f"E8 analysis: spread median {res['spread']['median']:.2f}x; stock dispatch "
              f"{res['stock_dispatch']['mean_regret_pct']:.2f}%; shape {pr['shape']['mean_regret_pct']:.2f}%, "
              f"entropy only {pr['entropy_only']['mean_regret_pct']:.2f}%, shape+entropy "
              f"{pr['shape+entropy']['mean_regret_pct']:.2f}%; exact-shape headroom "
              f"{1000 * res['headroom']['exact']['mean_ms']:.2f} us (max {1000 * res['headroom']['exact']['max_key_ms']:.1f} us)")
        return
    log = Logger(f"e8_refscan_{tag}")
    gpu_baseline = all_gpu_pids()
    check_disk(args.out_dir, need_gb=0.5, log=log)
    chunks = [int(c) for c in args.chunks.split(",")]
    mods = _modules()
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    own = register_own_gpu_pids(gpu_baseline, log=log) if args.exclusive_guard else set()
    meta = env_metadata()
    meta.update({"experiment": "E8", "chunks": chunks, "own_gpu_pids": sorted(own),
                 "args": {k: str(v) for k, v in vars(args).items()},
                 "stock_dispatch": {"<=128": 128, "<=256": 256, "<=512": 512, "<=1024": 1024, ">1024": 2048}})
    summary_path = args.out_dir / f"e8_refscan_{tag}.json"
    val = validate(chunks, mods, log)
    if args.validate_only or not val["passed"]:
        write_json(summary_path, {"meta": meta, "validation": val})
        raise SystemExit(0 if val["passed"] else 4)

    csv_path = args.out_dir / f"e8_refscan_{tag}.csv"
    w = CsvWriter(csv_path, columns(chunks))
    manifest = load_manifest(args.trace_dir, "mamba1")
    if not manifest:
        raise SystemExit(f"no captured Mamba-1 traces under {args.trace_dir}; run capture_traces first")
    layers = sorted({l for r in manifest for l in r["layers"]})
    if args.layers:
        keep = {int(x) for x in args.layers.split(",")}
        layers = [l for l in layers if l in keep]
    params = {l: load_params(args.trace_dir, "mamba1", l) for l in layers}
    plan = []
    for b in [int(x) for x in args.batches.split(",")]:
        plan += [("call", b, gid, layer, members) for gid, layer, members in build_groups(manifest, layers, b, args.seed)]
    if args.e4_keys:
        plan += [("e4key", b, None, None, L) for L in (128, 512, 2048, 8192) for b in (1, 4, 16)]
    rng = random.Random(args.seed)
    log(f"E8: {len(plan)} rows planned, {len(w.done)} already done; chunks {chunks}")
    t_start, new_rows = time.time(), 0
    for i, (kind, b, gid, layer, members) in enumerate(plan):
        key = f"mamba1|b{b}|{gid}|layer{layer}" if kind == "call" else f"e8key|L{members}|b{b}"
        if key in w.done:
            continue
        if args.max_rows is not None and new_rows >= args.max_rows:
            break
        if kind == "call":
            inp, act, dlt, L = stack_inputs("mamba1", args.trace_dir, members, layer, params[layer])
            info = {"group_id": gid, "prompt_ids": ";".join(m["prompt_id"] for m in members),
                    "regimes": ";".join(sorted({m["regime"] for m in members})),
                    "sources": ";".join(sorted({m["source"] for m in members})), "fold": members[0]["fold"],
                    "layer": layer, "n_unique_prompts": len({m["prompt_id"] for m in members})}
        else:
            from src.experiments.tpds.e4_autotune import mamba1_key_inputs

            L = members
            inp, source = mamba1_key_inputs(L, b, args.trace_dir)
            act, dlt = inp["u"], inp["delta"]
            info = {"group_id": "", "prompt_ids": source, "regimes": "e4_key", "sources": source, "fold": -1,
                    "layer": -1, "n_unique_prompts": 1}
        inp = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inp.items()}
        t0 = time.perf_counter()
        feats = activation_features(act.cuda(), dlt.cuda())
        torch.cuda.synchronize()
        feat_ms = (time.perf_counter() - t0) * 1e3
        variants = make_variants(inp, chunks, mods)
        ref = variants["stock"]().float()
        diffs = {}
        for name in variants:
            if name != "stock":
                diffs[name] = float((variants[name]().float() - ref).abs().max())
        del ref
        order = list(variants)
        rng.shuffle(order)
        stats = time_variants(variants, order, args.warmup, args.repeats, args.exclusive_guard, args.guard_until, log)
        if stats is None:
            w.close()
            raise SystemExit(3)
        oracle = min(chunks, key=lambda c: stats[f"c{c}"]["mean_ms"])
        tied = [c for c in chunks if stats[f"c{c}"]["ci95_lo_ms"] <= stats[f"c{oracle}"]["ci95_hi_ms"]]
        row = {"key": key, "model": "mamba1", "host": tag, **info, "batch": b, "L": L, "L_bucket": length_bucket(L),
               "width": inp["u"].shape[1], "d_state": inp["A"].shape[1], **{k: round(v, 6) for k, v in feats.items()},
               "entropy_rule_chunk": entropy_rule_chunk(feats["entropy_norm"], 128, 2048), "oracle_chunk": oracle,
               "oracle_mean_ms": stats[f"c{oracle}"]["mean_ms"], "tied_chunks": ";".join(map(str, tied)),
               "max_out_diff_across_chunks": max(diffs[f"c{c}"] for c in chunks),
               "max_out_diff_vs_stock": max(diffs.values()), "stock_chunk": stock_chunk(L),
               "feature_ms": round(feat_ms, 4), "warmup": args.warmup, "repeats": args.repeats,
               "peak_vram_gb": round(peak_vram_gb(), 3)}
        for name, s in stats.items():
            row.update({f"lat_{name}_mean": s["mean_ms"], f"lat_{name}_std": s["std_ms"],
                        f"lat_{name}_ci_lo": s["ci95_lo_ms"], f"lat_{name}_ci_hi": s["ci95_hi_ms"]})
        w.write(row)
        new_rows += 1
        del inp, act, dlt, variants
        if new_rows % 25 == 0:
            torch.cuda.empty_cache()
        el = time.time() - t_start
        log(f"[{i + 1}/{len(plan)}] {key} L={L} oracle={oracle} {stats[f'c{oracle}']['mean_ms']:.4f} ms "
            f"stock={stats['stock']['mean_ms']:.4f} ms tied={row['tied_chunks']} max|dy|={row['max_out_diff_vs_stock']:.1e} "
            f"({el / 60:.1f} min)")
    w.close()
    write_json(summary_path, {"meta": meta, "validation": val, "rows_done": len(w.done)})
    log(f"E8 done ({new_rows} new rows)")


if __name__ == "__main__":
    main()
