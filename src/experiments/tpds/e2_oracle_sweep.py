"""E2 - per-call oracle chunk on real captured traces (items C2, C6, C7).

Every captured scan call (prompt x layer) of a real checkpoint is replayed
through the recurrence-preserving kernel at every candidate chunk, at batch
sizes {1, 4, 16}. Batched calls stack captured calls of the same layer, the
same cross-validation fold and the same power-of-two length bucket, cropped
to the shortest member (a prefix of a causal scan input is itself a valid
scan input). If a bucket has fewer prompts than the batch size, members are
reused and ``n_unique_prompts`` records it.

Per row: activation features of the timed tensor (entropy K=256, normalized
entropy, variance, kurtosis, sparsity, mean |x|, delta entropy), shape
(L, batch, width, layer), latency statistics at each chunk (mean, std, 95%
CI), the oracle chunk, the set of chunks statistically tied with it, the
chunk chosen by the entropy rule (Eq. 2, C in [32, 512]) and the maximum
output difference across chunks (all chunks compute the same recurrence).

Output: ``e2_<model>_<host>.csv`` (resumable, one fsynced row per call) and
``e2_<model>_<host>.json`` (metadata and summary).
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
import time
from pathlib import Path

import torch

from src.algorithms.torch_entropy import activation_entropy
from src.experiments.tpds import kernels as K
from src.experiments.tpds.capture_traces import load_call, load_manifest, load_params
from src.experiments.tpds.common import (DEFAULT_TRACE_DIR, GDN_CHUNKS, MAMBA1_CHUNKS, MAMBA2_CHUNKS, RESULTS_DIR,
                                         CsvWriter, Logger, all_gpu_pids, check_disk, cuda_time, entropy_rule_chunk,
                                         env_metadata, foreign_gpu_pids, host_name, peak_vram_gb,
                                         register_own_gpu_pids, summarize, wait_for_exclusive_gpu, write_json)

FEATURES = ["entropy_bits", "entropy_norm", "variance", "kurtosis", "sparsity", "mean_abs", "delta_entropy_bits"]


def chunks_for(kind):
    return {"mamba1": MAMBA1_CHUNKS, "mamba2": MAMBA2_CHUNKS, "gdn": GDN_CHUNKS}[kind]


def columns_for(kind):
    cols = ["key", "model", "host", "group_id", "prompt_ids", "regimes", "sources", "fold", "layer", "batch",
            "n_unique_prompts", "L", "L_bucket", "width", "d_state"] + FEATURES + [
        "entropy_rule_chunk", "oracle_chunk", "oracle_mean_ms", "tied_chunks", "max_out_diff_across_chunks",
        "feature_ms", "warmup", "repeats", "flush_l2", "peak_vram_gb"]
    for c in chunks_for(kind):
        cols += [f"lat_c{c}_mean", f"lat_c{c}_std", f"lat_c{c}_ci_lo", f"lat_c{c}_ci_hi"]
    return cols


# --------------------------------------------------------------------------- grouping
def length_bucket(L: int) -> int:
    return 1 << max(0, int(math.floor(math.log2(max(L, 1)))))


def build_groups(manifest, layers, batch, seed=0):
    """Return a list of (group_id, [manifest rows]) for one batch size."""
    groups = []
    rng = random.Random(seed + batch)
    for layer in layers:
        if batch == 1:
            for r in sorted(manifest, key=lambda r: r["prompt_id"]):
                groups.append((layer, [r]))
            continue
        cells: dict[tuple, list] = {}
        for r in manifest:
            cells.setdefault((r["fold"], length_bucket(r["L"])), []).append(r)
        for (_, _), rs in sorted(cells.items()):
            rs = sorted(rs, key=lambda r: (r["L"], r["prompt_id"]))
            for i in range(0, len(rs), batch):
                members = rs[i:i + batch]
                if len(members) < batch:
                    pool = list(rs)
                    rng.shuffle(pool)
                    j = 0
                    while len(members) < batch:
                        members.append(pool[j % len(pool)])
                        j += 1
                groups.append((layer, members))
    out = []
    for layer, members in groups:
        ids = [m["prompt_id"] for m in members]
        gid = hashlib.sha1("|".join(ids).encode()).hexdigest()[:12]
        out.append((gid, layer, members))
    return out


# --------------------------------------------------------------------------- features
@torch.no_grad()
def activation_features(x: torch.Tensor, delta: torch.Tensor) -> dict[str, float]:
    xf = x.float().reshape(-1)
    h = float(activation_entropy(xf, num_bins=256))
    mean = xf.mean()
    var = xf.var()
    std = var.clamp(min=1e-12).sqrt()
    kurt = (((xf - mean) / std) ** 4).mean() - 3.0
    amax = xf.abs().max().clamp(min=1e-12)
    sparsity = (xf.abs() <= 1e-3 * amax).float().mean()
    dh = float(activation_entropy(delta.float().reshape(-1), num_bins=256))
    return {"entropy_bits": h, "entropy_norm": h / 8.0, "variance": float(var), "kurtosis": float(kurt),
            "sparsity": float(sparsity), "mean_abs": float(xf.abs().mean()), "delta_entropy_bits": dh}


def stack_inputs(kind, trace_dir, members, layer, params):
    Lmin = min(m["L"] for m in members)
    calls = [load_call(trace_dir, kind, m["prompt_id"], layer) for m in members]
    if kind == "mamba1":
        inp = {k: torch.cat([c[k][..., :Lmin] for c in calls], 0).contiguous() for k in ("u", "delta", "B", "C")}
        if all("z" in c for c in calls):
            inp["z"] = torch.cat([c["z"][..., :Lmin] for c in calls], 0).contiguous()
        else:
            inp["z"] = None
        inp["A"] = params["A"].contiguous()
        inp["D"] = params["D"].float()
        act, dlt = inp["u"], inp["delta"]
    elif kind == "gdn":
        # production kernel path: raw q, k, v, a_proj output g and b_proj output beta, all bf16 as in the checkpoint
        inp = {k: torch.cat([c[k][:, :Lmin] for c in calls], 0).contiguous().to(torch.bfloat16)
               for k in ("q", "k", "v", "g", "beta")}
        inp.update({"A_log": params["A_log"], "dt_bias": params["dt_bias"],
                    "allow_neg_eigval": bool(params["allow_neg_eigval"].item()) if "allow_neg_eigval" in params
                    else False})
        act, dlt = inp["v"], inp["g"]
    else:
        inp = {k: torch.cat([c[k][:, :Lmin] for c in calls], 0).contiguous() for k in ("x", "dt", "B", "C")}
        inp.update({"A": params["A"], "D": params["D"].float(), "dt_bias": params["dt_bias"].float()})
        act, dlt = inp["x"], inp["dt"]
    return inp, act, dlt, Lmin


def make_fn(kind, inp, chunk):
    if kind == "mamba1":
        out = torch.empty_like(inp["u"])
        return lambda: K.mamba1_scan(inp["u"], inp["delta"], inp["A"], inp["B"], inp["C"], inp["D"], inp["z"],
                                     None, True, chunk=chunk, out=out)
    if kind == "gdn":
        return lambda: K.gdn_chunk_prod(inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], inp["A_log"],
                                        inp["dt_bias"], chunk_size=chunk, allow_neg_eigval=inp["allow_neg_eigval"])
    return lambda: K.mamba2_ssd(inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"], chunk, D=inp["D"],
                                dt_bias=inp["dt_bias"])


# --------------------------------------------------------------------------- summary
def summarize_csv(csv_path: Path, kind: str) -> dict:
    from src.experiments.tpds.analysis import load_e2, static_regret_table

    df = load_e2([csv_path])
    if df.empty:
        return {}
    out = {"rows": int(len(df)), "by_batch": {}}
    for b, sub in df.groupby("batch"):
        out["by_batch"][int(b)] = {
            "rows": int(len(sub)),
            "oracle_distribution": {int(k): int(v) for k, v in sub["oracle_chunk"].value_counts().items()},
            "static_regret": static_regret_table(sub, chunks_for(kind)),
            "entropy_bits_range": [float(sub["entropy_bits"].min()), float(sub["entropy_bits"].max())],
        }
    b1 = df[df["batch"] == 1]
    if not b1.empty:
        out["entropy_by_regime_b1"] = {
            reg: {"mean": float(g["entropy_bits"].mean()), "std": float(g["entropy_bits"].std(ddof=0)),
                  "min": float(g["entropy_bits"].min()), "max": float(g["entropy_bits"].max()), "n": int(len(g))}
            for reg, g in b1.groupby("regimes")}
    out["max_out_diff_across_chunks"] = float(df["max_out_diff_across_chunks"].max())
    return out


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("mamba1", "mamba2", "gdn"), required=True)
    p.add_argument("--exclusive-guard", action="store_true",
                   help="pause while other processes use the GPU and re-time calls they overlapped")
    p.add_argument("--guard-until", type=float, default=None, help="epoch seconds after which the guard gives up")
    p.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    p.add_argument("--batches", default="1,4,16")
    p.add_argument("--layers", default=None, help="subset of captured layers (default: all captured)")
    p.add_argument("--chunks", default=None)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--flush-l2", action="store_true")
    p.add_argument("--pre-spin-ms", type=float, default=0.0,
                   help="GPU-side spin before each timed block so host load cannot insert gaps (0 = off)")
    p.add_argument("--max-rows", type=int, default=None, help="stop after this many new rows (smoke)")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--tag", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    kind = args.model
    tag = args.tag or host_name()
    log = Logger(f"e2_{kind}_{tag}")
    check_disk(args.out_dir, need_gb=0.5, log=log)
    gpu_baseline = all_gpu_pids()
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    own = register_own_gpu_pids(gpu_baseline, log=log) if args.exclusive_guard else set()
    manifest = load_manifest(args.trace_dir, kind)
    if not manifest:
        raise SystemExit(f"no captured traces for {kind} under {args.trace_dir}; run capture_traces first")
    layers = sorted({l for r in manifest for l in r["layers"]})
    if args.layers:
        keep = {int(x) for x in args.layers.split(",")}
        layers = [l for l in layers if l in keep]
    chunks = [int(c) for c in args.chunks.split(",")] if args.chunks else list(chunks_for(kind))
    if set(chunks) != set(chunks_for(kind)):
        log(f"WARNING: chunk subset {chunks}; CSV columns still list {list(chunks_for(kind))}")
    csv_path = args.out_dir / f"e2_{kind}_{tag}.csv"
    w = CsvWriter(csv_path, columns_for(kind))
    meta = env_metadata()
    meta.update({"experiment": "E2", "model": kind, "args": vars(args), "layers": layers,
                 "n_prompts": len(manifest), "chunks": chunks, "own_gpu_pids": sorted(own),
                 "mamba1_default_launch": {str(c): K.default_mamba1_config(c).as_dict() for c in chunks}
                 if kind == "mamba1" else None})
    params = {l: load_params(args.trace_dir, kind, l) for l in layers}
    rng = random.Random(args.seed)
    new_rows = 0
    batches = [int(b) for b in args.batches.split(",")]
    plan = []
    for b in batches:
        plan += [(b, gid, layer, members) for gid, layer, members in build_groups(manifest, layers, b, args.seed)]
    log(f"{kind}: {len(manifest)} prompts, layers {layers}, {len(plan)} calls planned, {len(w.done)} already done")
    t_start = time.time()
    for i, (b, gid, layer, members) in enumerate(plan):
        key = f"{kind}|b{b}|{gid}|layer{layer}"
        if key in w.done:
            continue
        if args.max_rows is not None and new_rows >= args.max_rows:
            break
        inp, act, dlt, Lmin = stack_inputs(kind, args.trace_dir, members, layer, params[layer])
        t0 = time.perf_counter()
        feats = activation_features(act, dlt)
        torch.cuda.synchronize()
        feat_ms = (time.perf_counter() - t0) * 1e3
        retimed = 0
        while True:
            if args.exclusive_guard:
                waited = wait_for_exclusive_gpu(args.guard_until, log)
                if waited is None:
                    log("GPU still busy at the guard deadline; stopping (restart resumes)")
                    w.close()
                    raise SystemExit(3)
            before = foreign_gpu_pids() if args.exclusive_guard else None
            order = list(chunks)
            rng.shuffle(order)
            stats, ref, max_diff = {}, None, 0.0
            for c in order:
                fn = make_fn(kind, inp, c)
                y = fn().float()
                if ref is None:
                    ref = y.clone()
                else:
                    max_diff = max(max_diff, float((y - ref).abs().max()))
                del y
                stats[c] = summarize(cuda_time(fn, args.warmup, args.repeats, args.flush_l2, args.pre_spin_ms))
            del ref
            after = foreign_gpu_pids() if args.exclusive_guard else None
            if not (before or after):
                break
            retimed += 1
            log(f"foreign GPU processes {before or after} during {key}; re-timing")
        oracle = min(stats, key=lambda c: stats[c]["mean_ms"])
        tied = [c for c in chunks if stats[c]["ci95_lo_ms"] <= stats[oracle]["ci95_hi_ms"]]
        width = act.shape[1] if kind == "mamba1" else act.shape[2] * act.shape[3]
        row = {"key": key, "model": kind, "host": tag, "group_id": gid,
               "prompt_ids": ";".join(m["prompt_id"] for m in members),
               "regimes": ";".join(sorted({m["regime"] for m in members})),
               "sources": ";".join(sorted({m["source"] for m in members})),
               "fold": members[0]["fold"], "layer": layer, "batch": b,
               "n_unique_prompts": len({m["prompt_id"] for m in members}), "L": Lmin,
               "L_bucket": length_bucket(Lmin), "width": width,
               "d_state": (inp["A"].shape[1] if kind == "mamba1" else inp["k"].shape[-1] if kind == "gdn"
                           else inp["B"].shape[-1]),
               **{k: round(v, 6) for k, v in feats.items()},
               "entropy_rule_chunk": entropy_rule_chunk(feats["entropy_norm"]),
               "oracle_chunk": oracle, "oracle_mean_ms": stats[oracle]["mean_ms"],
               "tied_chunks": ";".join(map(str, tied)), "max_out_diff_across_chunks": max_diff,
               "feature_ms": round(feat_ms, 4), "warmup": args.warmup, "repeats": args.repeats,
               "flush_l2": args.flush_l2, "peak_vram_gb": round(peak_vram_gb(), 3)}
        for c in chunks:
            s = stats[c]
            row.update({f"lat_c{c}_mean": s["mean_ms"], f"lat_c{c}_std": s["std_ms"],
                        f"lat_c{c}_ci_lo": s["ci95_lo_ms"], f"lat_c{c}_ci_hi": s["ci95_hi_ms"]})
        w.write(row)
        new_rows += 1
        del inp, act, dlt
        if new_rows % 25 == 0:
            torch.cuda.empty_cache()
            write_json(args.out_dir / f"e2_{kind}_{tag}.json", {"meta": meta, "summary": summarize_csv(csv_path, kind)})
        el = time.time() - t_start
        log(f"[{i + 1}/{len(plan)}] {key} L={Lmin} H={feats['entropy_bits']:.2f} oracle={oracle} "
            f"rule={row['entropy_rule_chunk']} {stats[oracle]['mean_ms']:.4f} ms tied={row['tied_chunks']} "
            f"({el / 60:.1f} min, peak {row['peak_vram_gb']} GB)")
    w.close()
    write_json(args.out_dir / f"e2_{kind}_{tag}.json", {"meta": meta, "summary": summarize_csv(csv_path, kind)})
    log(f"E2 {kind} done ({new_rows} new rows)")


if __name__ == "__main__":
    main()
