"""E4 - offline auto-tuning baseline and per-call entropy decision overhead (item C4; input to E5).

Part A (Mamba-1 Triton scan): Triton-autotune-style exhaustive search over
chunk (BLOCK_T) x BLOCK_D x num_warps x num_stages, keyed on
(L bucket, batch, width). The search times each configuration briefly (as
``triton.autotune`` does), then the winner and the baselines are re-measured
with the full protocol under identical conditions:
  * tuned        - the search winner;
  * static_<c>   - every chunk c with the default launch configuration;
  * entropy_rule - Eq. (2) chunk from the input's measured entropy, default launch.
Wall-clock tuning cost per key is recorded (total, JIT-compile share, bench share)
starting from an empty Triton cache.

Part B (Mamba-2 SSD): search over chunk_size per key (SSD's own internal
block-size autotuning is included in the tuning wall clock).

Part C (decision overhead): isolated cost of the entropy scheduler's
per-call decision on real captured activations: K=256 histogram entropy on the
full tensor (and on every 8th element), Eq. (2) mapping and the host
synchronization (.item()) needed to use the result for a launch decision.
Measured as host wall clock per decision; the GPU-only histogram time and the
bare synchronization cost are reported separately.

Output: ``e4_<host>.jsonl`` (resumable, one row per key) and ``e4_<host>.json``.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import tempfile
import time
from pathlib import Path

# A fresh Triton cache makes the measured tuning cost include JIT compilation.
if "--keep-triton-cache" not in sys.argv and "TPDS_E4_CACHE_SET" not in os.environ:
    os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="tpds_e4_triton_cache_")
    os.environ["TPDS_E4_CACHE_SET"] = "1"

import statistics  # noqa: E402

import torch  # noqa: E402

from src.algorithms.torch_entropy import activation_entropy  # noqa: E402
from src.experiments.tpds import kernels as K  # noqa: E402
from src.experiments.tpds.common import (DEFAULT_TRACE_DIR, MAMBA1_CHUNKS, MAMBA2_CHUNKS, RESULTS_DIR,  # noqa: E402
                                         JsonlWriter, Logger, check_disk, cuda_time, dedupe_by_key,
                                         entropy_rule_chunk, env_metadata, host_name, peak_vram_gb, read_jsonl,
                                         summarize, write_json)
from src.experiments.tpds.e1_value_invariance import _fit_time, _longest_call  # noqa: E402

BLOCK_DS = (4, 8, 16, 32)
WARPS = (1, 2, 4, 8)
STAGES = (1, 2)
MAX_TILE = 16384  # BLOCK_T * BLOCK_D limit (larger tiles spill registers heavily)


# --------------------------------------------------------------------------- inputs
def mamba1_key_inputs(L, bsz, trace_dir, seed=0):
    inp = K.make_mamba1_inputs(bsz, 2048, 16, L, dtype=torch.float16, seed=seed)
    source = "synthetic_gaussian"
    found = _longest_call(trace_dir, "mamba1") if trace_dir else None
    if found is not None:
        from src.experiments.tpds.capture_traces import load_call, load_params

        call = load_call(trace_dir, "mamba1", *found)
        prm = load_params(trace_dir, "mamba1", found[1])
        for k in ("u", "delta", "B", "C", "z"):
            if k in call:
                inp[k] = _fit_time(call[k], L, 2).expand(bsz, -1, -1).contiguous()
        inp["A"] = prm["A"].contiguous()
        inp["D"] = prm["D"].float()
        source = f"captured:{found[0]}:layer{found[1]}"
    return inp, source


def mamba2_key_inputs(L, bsz, trace_dir, seed=0):
    inp = K.make_ssd_inputs(bsz, L, dtype=torch.float16, seed=seed)
    source = "synthetic_gaussian"
    found = _longest_call(trace_dir, "mamba2") if trace_dir else None
    if found is not None:
        from src.experiments.tpds.capture_traces import load_call, load_params

        call = load_call(trace_dir, "mamba2", *found)
        prm = load_params(trace_dir, "mamba2", found[1])
        for k in ("x", "dt", "B", "C"):
            t = _fit_time(call[k], L, 1)
            inp[k] = t.expand(bsz, *t.shape[1:]).contiguous()
        inp.update({"A": prm["A"], "D": prm["D"].float(), "dt_bias": prm["dt_bias"].float()})
        source = f"captured:{found[0]}:layer{found[1]}"
    return inp, source


def m1_fn(inp, cfg, out):
    return lambda: K.mamba1_scan(inp["u"], inp["delta"], inp["A"], inp["B"], inp["C"], inp["D"], inp.get("z"),
                                 None, True, config=cfg, out=out)


def m2_fn(inp, chunk):
    return lambda: K.mamba2_ssd(inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"], chunk, D=inp["D"],
                                dt_bias=inp["dt_bias"])


def mamba1_space(chunks):
    for bt, bd, nw, ns in itertools.product(chunks, BLOCK_DS, WARPS, STAGES):
        if bt * bd <= MAX_TILE:
            yield K.LaunchConfig(bt, bd, nw, ns)


# --------------------------------------------------------------------------- decision overhead
def time_decision(x: torch.Tensor, stride: int, repeats: int, warmup: int) -> dict:
    """Host wall clock of one full entropy decision (histogram, Eq. 2, .item() sync)."""
    def decide():
        v = x.reshape(-1)[::stride] if stride > 1 else x.reshape(-1)
        h = activation_entropy(v.float(), num_bins=256)
        return entropy_rule_chunk(h.item() / 8.0)

    for _ in range(warmup):
        decide()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        decide()
        ts.append((time.perf_counter() - t0) * 1e3)
    gpu = cuda_time(lambda: activation_entropy((x.reshape(-1)[::stride] if stride > 1 else x.reshape(-1)).float(),
                                               num_bins=256), warmup, repeats)
    return {"decision_wall": summarize(ts), "histogram_gpu": summarize(gpu)}


def time_sync(repeats: int) -> dict:
    ts = []
    t = torch.zeros(1, device="cuda")
    for _ in range(repeats):
        t0 = time.perf_counter()
        (t + 1).item()
        ts.append((time.perf_counter() - t0) * 1e3)
    return summarize(ts)


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parts", default="m1,m2,overhead")
    p.add_argument("--lengths", default="128,512,2048,8192")
    p.add_argument("--batches", default="1,4,16")
    p.add_argument("--search-warmup", type=int, default=3)
    p.add_argument("--search-repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--overhead-repeats", type=int, default=200)
    p.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--tag", default=None)
    p.add_argument("--keep-triton-cache", action="store_true", help="do not start from an empty Triton cache")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tag = args.tag or host_name()
    log = Logger(f"e4_{tag}")
    check_disk(args.out_dir, need_gb=0.5, log=log)
    meta = env_metadata()
    meta.update({"experiment": "E4", "args": vars(args), "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
                 "search_space_mamba1": {"block_t": list(MAMBA1_CHUNKS), "block_d": list(BLOCK_DS),
                                         "num_warps": list(WARPS), "num_stages": list(STAGES),
                                         "max_tile": MAX_TILE}})
    w = JsonlWriter(args.out_dir / f"e4_{tag}.jsonl", meta=meta)
    parts = set(args.parts.split(","))
    lengths = [int(x) for x in args.lengths.split(",")]
    batches = [int(x) for x in args.batches.split(",")]

    if "m1" in parts:
        for L, b in itertools.product(lengths, batches):
            key = f"m1tune|L{L}|b{b}"
            if key in w.done:
                continue
            inp, source = mamba1_key_inputs(L, b, args.trace_dir)
            ent = float(activation_entropy(inp["u"].float(), num_bins=256))
            out = torch.empty_like(inp["u"])
            search, compile_s, bench_s = [], 0.0, 0.0
            t_key = time.perf_counter()
            for cfg in mamba1_space(MAMBA1_CHUNKS):
                fn = m1_fn(inp, cfg, out)
                t0 = time.perf_counter()
                try:
                    fn()
                    torch.cuda.synchronize()
                except Exception as exc:  # e.g. out of shared memory / registers for this config
                    search.append({**cfg.as_dict(), "error": repr(exc)[:200]})
                    continue
                compile_s += time.perf_counter() - t0
                t1 = time.perf_counter()
                ms = statistics.median(cuda_time(fn, args.search_warmup, args.search_repeats))
                bench_s += time.perf_counter() - t1
                search.append({**cfg.as_dict(), "median_ms": ms})
            tune_s = time.perf_counter() - t_key
            ok = [s for s in search if "median_ms" in s]
            best = min(ok, key=lambda s: s["median_ms"])
            best_cfg = K.LaunchConfig(best["block_t"], best["block_d"], best["num_warps"], best["num_stages"])
            # Cost of the cheaper offline alternative: profiling only the chunk (default launch).
            t_prof = time.perf_counter()
            for c in MAMBA1_CHUNKS:
                cuda_time(m1_fn(inp, K.default_mamba1_config(c), out), args.search_warmup, args.search_repeats)
            profile_static_s = time.perf_counter() - t_prof
            final = {"tuned": summarize(cuda_time(m1_fn(inp, best_cfg, out), args.warmup, args.repeats))}
            for c in MAMBA1_CHUNKS:
                final[f"static_{c}"] = summarize(cuda_time(m1_fn(inp, K.default_mamba1_config(c), out),
                                                           args.warmup, args.repeats))
            rule_c = entropy_rule_chunk(ent / 8.0)
            final["entropy_rule"] = dict(final[f"static_{rule_c}"])
            row = {"kind": "m1tune", "key": key, "family": "mamba1", "L": L, "batch": b, "width": 2048,
                   "input_source": source, "entropy_bits": ent, "entropy_rule_chunk": rule_c,
                   "tuning_wall_s": tune_s, "tuning_first_call_s": compile_s, "tuning_bench_s": bench_s,
                   "profile_static_wall_s": profile_static_s, "n_configs": len(search), "n_failed": len(search) - len(ok), "best_config": best,
                   "final": final, "search": search, "peak_vram_gb": peak_vram_gb()}
            w.write(row)
            best_static = min((k for k in final if k.startswith("static_")), key=lambda k: final[k]["mean_ms"])
            log(f"mamba1 L={L} b={b}: tuned {final['tuned']['mean_ms']:.4f} ms {best} | best static "
                f"{best_static} {final[best_static]['mean_ms']:.4f} ms | rule c={rule_c} "
                f"{final['entropy_rule']['mean_ms']:.4f} ms | tuning {tune_s:.1f} s ({len(search)} configs)")
            del inp, out
            torch.cuda.empty_cache()

    if "m2" in parts and K.ssd_available():
        for L, b in itertools.product(lengths, batches):
            key = f"m2tune|L{L}|b{b}"
            if key in w.done:
                continue
            inp, source = mamba2_key_inputs(L, b, args.trace_dir)
            ent = float(activation_entropy(inp["x"].float(), num_bins=256))
            search, compile_s = [], 0.0
            t_key = time.perf_counter()
            for c in MAMBA2_CHUNKS:
                fn = m2_fn(inp, c)
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                compile_s += time.perf_counter() - t0
                search.append({"chunk_size": c, "median_ms": statistics.median(
                    cuda_time(fn, args.search_warmup, args.search_repeats))})
            tune_s = time.perf_counter() - t_key
            best = min(search, key=lambda s: s["median_ms"])
            final = {f"static_{c}": summarize(cuda_time(m2_fn(inp, c), args.warmup, args.repeats))
                     for c in MAMBA2_CHUNKS}
            final["tuned"] = dict(final[f"static_{best['chunk_size']}"])
            rule_c = entropy_rule_chunk(ent / 8.0)
            final["entropy_rule"] = dict(final[f"static_{rule_c}"])
            w.write({"kind": "m2tune", "key": key, "family": "mamba2", "L": L, "batch": b, "width": 2048,
                     "input_source": source, "entropy_bits": ent, "entropy_rule_chunk": rule_c,
                     "tuning_wall_s": tune_s, "tuning_first_call_s": compile_s, "best_config": best,
                     "final": final, "search": search, "peak_vram_gb": peak_vram_gb()})
            log(f"mamba2 L={L} b={b}: tuned chunk {best['chunk_size']} {final['tuned']['mean_ms']:.4f} ms | "
                f"rule c={rule_c} {final['entropy_rule']['mean_ms']:.4f} ms | tuning {tune_s:.1f} s")
            del inp
            torch.cuda.empty_cache()

    if "overhead" in parts:
        key = "sync_baseline"
        if key not in w.done:
            w.write({"kind": "sync", "key": key, "sync_wall": time_sync(args.overhead_repeats)})
        for L, b in itertools.product(lengths, batches):
            inp, source = mamba1_key_inputs(L, b, args.trace_dir)
            for stride in (1, 8):
                key = f"overhead|mamba1|L{L}|b{b}|s{stride}"
                if key in w.done:
                    continue
                res = time_decision(inp["u"], stride, args.overhead_repeats, args.warmup)
                scan = summarize(cuda_time(m1_fn(inp, K.default_mamba1_config(256), torch.empty_like(inp["u"])),
                                           args.warmup, args.repeats))
                w.write({"kind": "overhead", "key": key, "family": "mamba1", "L": L, "batch": b, "stride": stride,
                         "numel": inp["u"].numel(), "input_source": source, **res, "scan_c256": scan})
                log(f"overhead L={L} b={b} stride={stride}: decision {res['decision_wall']['mean_ms']:.4f} ms "
                    f"(GPU hist {res['histogram_gpu']['mean_ms']:.4f} ms) vs scan {scan['mean_ms']:.4f} ms")
            del inp
            torch.cuda.empty_cache()

    rows = dedupe_by_key(read_jsonl(args.out_dir / f"e4_{tag}.jsonl"))
    metas = read_jsonl(args.out_dir / f"e4_{tag}.jsonl", kind="meta")
    write_json(args.out_dir / f"e4_{tag}.json", {"experiment": "E4", "meta": metas[-1] if metas else meta,
                                                "summary": summarize_e4(rows), "rows": rows})
    w.close()
    log(f"E4 done; peak VRAM {peak_vram_gb():.2f} GB")


def summarize_e4(rows):
    out = {}
    for fam, kind in (("mamba1", "m1tune"), ("mamba2", "m2tune")):
        rs = [r for r in rows if r.get("kind") == kind]
        if not rs:
            continue
        chunks = MAMBA1_CHUNKS if fam == "mamba1" else MAMBA2_CHUNKS
        # global static: one chunk (default launch) minimizing the summed latency ratio to the tuned result
        ratio = {c: statistics.fmean(r["final"][f"static_{c}"]["mean_ms"] / r["final"]["tuned"]["mean_ms"] for r in rs)
                 for c in chunks}
        g = min(ratio, key=ratio.get)
        per_key = []
        for r in rs:
            f = r["final"]
            bs = min((k for k in f if k.startswith("static_")), key=lambda k: f[k]["mean_ms"])
            per_key.append({"L": r["L"], "batch": r["batch"], "tuned_ms": f["tuned"]["mean_ms"],
                            "best_static_key": bs, "best_static_ms": f[bs]["mean_ms"],
                            "global_static_ms": f[f"static_{g}"]["mean_ms"],
                            "entropy_rule_ms": f["entropy_rule"]["mean_ms"],
                            "entropy_rule_chunk": r["entropy_rule_chunk"], "tuning_wall_s": r["tuning_wall_s"]})
        out[fam] = {"global_static_chunk": g, "global_static_mean_ratio_to_tuned": ratio[g],
                    "total_tuning_wall_s": sum(r["tuning_wall_s"] for r in rs), "n_keys": len(rs),
                    "total_profile_static_wall_s": sum(r.get("profile_static_wall_s", r["tuning_wall_s"]) for r in rs),
                    "per_key": per_key}
    ov = [r for r in rows if r.get("kind") == "overhead"]
    if ov:
        out["overhead"] = [{"L": r["L"], "batch": r["batch"], "stride": r["stride"],
                            "decision_ms": r["decision_wall"]["mean_ms"],
                            "histogram_gpu_ms": r["histogram_gpu"]["mean_ms"],
                            "scan_ms": r["scan_c256"]["mean_ms"]} for r in ov]
    sy = [r for r in rows if r.get("kind") == "sync"]
    if sy:
        out["sync_ms"] = sy[-1]["sync_wall"]["mean_ms"]
    return out


if __name__ == "__main__":
    main()
