"""E7 - Triton Mamba-1 scan vs the reference CUDA selective scan (limitation of item C1/C4).

The kernel-level Mamba-1 results use a Triton scan written for this study, in which the
chunk is the time tile. This experiment times the reference CUDA ``selective_scan_fn`` of
mamba_ssm on the same inputs (the twelve E4 shape keys, captured activations) and compares
it with the Triton kernel at every chunk (default launch configuration) and, when the E4
result file of the same host is present, at the auto-tuned configuration of the key. It
also checks that both produce the same output.

Every timed key waits until no other process holds the GPU (``--exclusive-guard``) and is
re-timed if one appeared during it. Resumable: completed keys in the JSONL are skipped.

Output: ``e7_<tag>.jsonl`` (one row per key) and ``e7_<tag>.json`` (rows + summary).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path

import torch

from src.experiments.tpds import kernels as K
from src.experiments.tpds.common import (DEFAULT_TRACE_DIR, MAMBA1_CHUNKS, RESULTS_DIR, JsonlWriter, Logger,
                                         all_gpu_pids, check_disk, cuda_time, dedupe_by_key, env_metadata, foreign_gpu_pids,
                                         host_name, peak_vram_gb, read_jsonl, register_own_gpu_pids, summarize,
                                         wait_for_exclusive_gpu,
                                         write_json)
from src.experiments.tpds.e4_autotune import m1_fn, mamba1_key_inputs


def reference_fn(inp):
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    return lambda: selective_scan_fn(inp["u"], inp["delta"], inp["A"], inp["B"], inp["C"], inp["D"], inp.get("z"),
                                     None, True)


def tuned_configs(e4_path: Path) -> dict:
    """E4 auto-tuned launch configuration per (L, batch), if the E4 file of this host exists."""
    if not e4_path.exists():
        return {}
    rows = json.loads(e4_path.read_text()).get("rows", [])
    out = {}
    for r in rows:
        if r.get("kind") == "m1tune" and r.get("best_config"):
            b = r["best_config"]
            out[(r["L"], r["batch"])] = K.LaunchConfig(b["block_t"], b["block_d"], b["num_warps"], b["num_stages"])
    return out


def time_key(inp, variants, warmup, repeats, guard, until, log):
    """Time every variant of one key back to back; re-time the key if another GPU process appeared."""
    while True:
        if guard and wait_for_exclusive_gpu(until, log) is None:
            return None
        before = foreign_gpu_pids() if guard else None
        res = {name: summarize(cuda_time(fn, warmup, repeats)) for name, fn in variants.items()}
        after = foreign_gpu_pids() if guard else None
        if not (before or after):
            return res
        log(f"foreign GPU processes {before or after} during the key; re-timing")


def summarize_rows(rows):
    ok = [r for r in rows if r.get("ref")]
    if not ok:
        return {}
    ratio_best = [r["best_static"]["mean_ms"] / r["ref"]["mean_ms"] for r in ok]
    out = {"n_keys": len(ok),
           "best_static_over_ref_geomean": math.exp(statistics.fmean(math.log(x) for x in ratio_best)),
           "best_static_over_ref_min": min(ratio_best), "best_static_over_ref_max": max(ratio_best),
           "max_abs_out_diff": max(r["max_abs_out_diff"] for r in ok)}
    tuned = [r["tuned"]["mean_ms"] / r["ref"]["mean_ms"] for r in ok if r.get("tuned")]
    if tuned:
        out["tuned_over_ref_geomean"] = math.exp(statistics.fmean(math.log(x) for x in tuned))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lengths", default="128,512,2048,8192")
    p.add_argument("--batches", default="1,4,16")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--tag", default=None)
    p.add_argument("--e4-tag", default=None, help="tag of the E4 file with tuned configurations (default: --tag)")
    p.add_argument("--exclusive-guard", action="store_true")
    p.add_argument("--guard-until", type=float, default=None)
    p.add_argument("--allow-missing-ref", action="store_true", help="smoke tests without mamba_ssm")
    args = p.parse_args(argv)
    tag = args.tag or host_name()
    log = Logger(f"e7_{tag}")
    gpu_baseline = all_gpu_pids()  # before this process creates its CUDA context
    check_disk(args.out_dir, need_gb=0.2, log=log)
    try:
        import mamba_ssm  # noqa: F401
        have_ref = True
    except ImportError:
        have_ref = False
        if not args.allow_missing_ref:
            raise SystemExit("mamba_ssm is not installed; the reference CUDA scan is unavailable")
        log("mamba_ssm missing: timing the Triton kernel only (smoke test)")
    meta = env_metadata()
    torch.zeros(1, device="cuda")  # make sure the CUDA context exists before identifying this process
    torch.cuda.synchronize()
    own = register_own_gpu_pids(gpu_baseline, log=log) if args.exclusive_guard else set()
    meta.update({"experiment": "E7", "own_gpu_pids": sorted(own), "args": {k: str(v) for k, v in vars(args).items()}})
    jsonl = args.out_dir / f"e7_{tag}.jsonl"
    w = JsonlWriter(jsonl, meta=meta)
    tuned = tuned_configs(args.out_dir / f"e4_{args.e4_tag or tag}.json")
    for L, b in itertools.product([int(x) for x in args.lengths.split(",")],
                                  [int(x) for x in args.batches.split(",")]):
        key = f"e7|L{L}|b{b}"
        if key in w.done:
            continue
        inp, source = mamba1_key_inputs(L, b, args.trace_dir)
        out = torch.empty_like(inp["u"])
        variants = {f"triton_c{c}": m1_fn(inp, K.default_mamba1_config(c), out) for c in MAMBA1_CHUNKS}
        if (L, b) in tuned:
            variants["triton_tuned"] = m1_fn(inp, tuned[(L, b)], out)
        diff = None
        if have_ref:
            ref = reference_fn(inp)
            variants["ref"] = ref
            y_ref = ref().float()
            y_tri = m1_fn(inp, K.default_mamba1_config(256), torch.empty_like(inp["u"]))().float()
            diff = float((y_ref - y_tri).abs().max())
        res = time_key(inp, variants, args.warmup, args.repeats, args.exclusive_guard, args.guard_until, log)
        if res is None:
            w.close()
            raise SystemExit(3)
        static = {k: v for k, v in res.items() if k.startswith("triton_c")}
        best = min(static, key=lambda k: static[k]["mean_ms"])
        row = {"kind": "key", "key": key, "L": L, "batch": b, "input_source": source,
               "triton": static, "best_static_chunk": int(best[len("triton_c"):]), "best_static": static[best],
               "tuned": res.get("triton_tuned"),
               "tuned_config": vars(tuned[(L, b)]) if (L, b) in tuned else None,
               "ref": res.get("ref"), "max_abs_out_diff": diff, "peak_vram_gb": peak_vram_gb()}
        w.write(row)
        msg = f"L={L} b={b}: best Triton chunk {row['best_static_chunk']} {static[best]['mean_ms']:.4f} ms"
        if row["ref"]:
            msg += (f" | reference CUDA {row['ref']['mean_ms']:.4f} ms (ratio "
                    f"{static[best]['mean_ms'] / row['ref']['mean_ms']:.2f}) | max |dy| {diff:.2e}")
        log(msg)
        del inp, out
        torch.cuda.empty_cache()
    rows = [r for r in dedupe_by_key(read_jsonl(jsonl)) if r.get("kind") == "key"]
    write_json(args.out_dir / f"e7_{tag}.json", {"experiment": "E7", "meta": meta, "summary": summarize_rows(rows),
                                                 "rows": rows})
    w.close()
    log(f"E7 done: {summarize_rows(rows)}")


if __name__ == "__main__":
    main()
