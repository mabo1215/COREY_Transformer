"""E12: chunk sweep at the per-device widths of tensor parallelism (review item M1c).

Tensor parallelism over t devices splits the d_inner channels of every Mamba layer, so each
device runs the selective scan at width d/t. This experiment times every chunk of the Mamba-1
scan at the full and sharded widths of the 1.4B (d = 4096) and 2.8B (d = 5120) checkpoints,
and of the 370M checkpoint (d = 2048), for a grid of lengths and batch sizes, so that the
analysis (``e12_analysis.py``) can test whether the frozen cost model, calibrated once at
d = 2048, predicts the chunk at widths it has never seen, and how much a chunk table tuned at
the unsharded width loses on the shards.

Inputs are synthetic (``kernels.make_mamba1_inputs``): the kernels are data-oblivious, and the
value-invariance experiments of the paper (E1, E9) bound the effect of the input values at the
selected chunk to a few percent, so the timing of a shape does not need captured activations.
One Gaussian draw per shape is reused for every chunk and round.

Every (width, length, batch) shape is timed in ``--rounds`` independent rounds; the shapes are
shuffled within a round and the chunks within a shape, so that drift between timing blocks
shows up as variation between rounds. One JSONL row per (family, shape, round) is written and
fsynced; a restart skips completed rows.

Families: ``triton`` (the study's Triton kernel, chunks 32..1024, default launch
configuration of each chunk) and, with ``--ref``, ``ref`` (the reference CUDA scan rebuilt with
a runtime chunk, module ``selective_scan_chunk_cuda``, chunks 128..2048).

Output: ``e12_tp_widths_<tag>.jsonl`` (rows) and ``e12_tp_widths_<tag>.json`` (meta + counts).
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch

from src.experiments.tpds import kernels as K
from src.experiments.tpds.common import (MAMBA1_CHUNKS, RESULTS_DIR, JsonlWriter, Logger, all_gpu_pids, check_disk,
                                         cuda_time, env_metadata, foreign_gpu_pids, gpu_state, host_name,
                                         peak_vram_gb, register_own_gpu_pids, summarize, wait_for_exclusive_gpu,
                                         write_json)

REF_CHUNKS = (128, 256, 512, 1024, 2048)
# Full d_inner of the checkpoints and their tensor-parallel shards (t = 2, 4, 8).
DEFAULT_WIDTHS = "5120,4096,2560,2048,1280,1024,640,512"
MODEL_WIDTHS = {"370m": 2048, "1.4b": 4096, "2.8b": 5120}
D_STATE = 16


def triton_runner(inp, out, chunk):
    return lambda: K.mamba1_scan(inp["u"], inp["delta"], inp["A"], inp["B"], inp["C"], inp["D"], inp["z"], None,
                                 True, chunk=chunk, out=out)


def ref_runner(inp, chunk):
    import selective_scan_chunk_cuda  # type: ignore

    B = inp["B"].unsqueeze(1)
    C = inp["C"].unsqueeze(1)
    return lambda: selective_scan_chunk_cuda.fwd_with_chunk_size(inp["u"], inp["delta"], inp["A"], B, C, inp["D"],
                                                                 inp["z"], None, True, chunk)[2]


def ref_available() -> bool:
    try:
        import selective_scan_chunk_cuda  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def time_cell(fn, args, log):
    """Time one (shape, chunk) cell; with the guard, re-time if another process used the GPU."""
    retimed = 0
    while True:
        if args.exclusive_guard:
            waited = wait_for_exclusive_gpu(args.guard_until, log)
            if waited is None:
                return None
            if waited > 0:
                cuda_time(fn, 5 * args.warmup, 1)
        before = foreign_gpu_pids() if args.exclusive_guard else None
        samples = cuda_time(fn, args.warmup, args.repeats)
        after = foreign_gpu_pids() if args.exclusive_guard else None
        if not (before or after):
            return {**summarize(samples), "retimed": retimed}
        retimed += 1
        log(f"foreign GPU processes {before or after}; re-timing")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--widths", default=DEFAULT_WIDTHS)
    p.add_argument("--lengths", default="128,256,512,1024,2048,4096,8192")
    p.add_argument("--batches", default="1,4,16")
    p.add_argument("--chunks", default=",".join(map(str, MAMBA1_CHUNKS)))
    p.add_argument("--ref", action="store_true", help="also time the runtime-chunk reference CUDA scan")
    p.add_argument("--ref-chunks", default=",".join(map(str, REF_CHUNKS)))
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--exclusive-guard", action="store_true",
                   help="suspend timing while another process is resident on the GPU")
    p.add_argument("--guard-until", type=float, default=None, help="epoch seconds after which the guard gives up")
    p.add_argument("--record-gpu-state", action="store_true", help="log clock/power/temperature per shape")
    p.add_argument("--max-rows", type=int, default=None, help="stop after this many new rows (smoke)")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--tag", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    tag = args.tag or host_name()
    log = Logger(f"e12_tp_widths_{tag}")
    gpu_baseline = all_gpu_pids()
    check_disk(args.out_dir, need_gb=0.2, log=log)
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    own = register_own_gpu_pids(gpu_baseline, log=log) if args.exclusive_guard else set()
    widths = [int(x) for x in args.widths.split(",") if x]
    lengths = [int(x) for x in args.lengths.split(",") if x]
    batches = [int(x) for x in args.batches.split(",") if x]
    fams = {"triton": [int(c) for c in args.chunks.split(",") if c]}
    if args.ref:
        if ref_available():
            fams["ref"] = [int(c) for c in args.ref_chunks.split(",") if c]
        else:
            log("reference CUDA scan (selective_scan_chunk_cuda) not importable; timing the Triton kernel only")
    meta = env_metadata()
    meta.update({"experiment": "E12", "own_gpu_pids": sorted(own), "widths": widths, "lengths": lengths,
                 "batches": batches, "families": fams, "d_state": D_STATE, "model_widths": MODEL_WIDTHS,
                 "mamba1_default_launch": {str(c): K.default_mamba1_config(c).as_dict() for c in fams["triton"]},
                 "args": {k: str(v) for k, v in vars(args).items()}})
    w = JsonlWriter(args.out_dir / f"e12_tp_widths_{tag}.jsonl", meta=meta)
    shapes = [(d, L, b) for d in widths for L in lengths for b in batches]
    rng = random.Random(args.seed)
    new_rows, t_start = 0, time.time()
    total = len(shapes) * args.rounds * len(fams)
    log(f"{len(shapes)} shapes x {args.rounds} rounds x {len(fams)} families = {total} rows, {len(w.done)} done")
    for r in range(args.rounds):
        order = list(shapes)
        rng.shuffle(order)
        for d, L, b in order:
            keys = {fam: f"e12|{fam}|d{d}|L{L}|b{b}|r{r}" for fam in fams}
            todo = [fam for fam in fams if keys[fam] not in w.done]
            if not todo:
                continue
            if args.max_rows is not None and new_rows >= args.max_rows:
                break
            inp = K.make_mamba1_inputs(b, d, D_STATE, L, dtype=torch.float16, seed=args.seed + d + 7 * L + 131 * b)
            out = torch.empty_like(inp["u"])
            s0 = gpu_state() if args.record_gpu_state else None
            for fam in todo:
                chunks = list(fams[fam])
                rng.shuffle(chunks)
                cells, ref, max_diff = {}, None, 0.0
                for c in chunks:
                    fn = triton_runner(inp, out, c) if fam == "triton" else ref_runner(inp, c)
                    y = fn().float()
                    if ref is None:
                        ref = y.clone()
                    else:
                        max_diff = max(max_diff, float((y - ref).abs().max()))
                    del y
                    cell = time_cell(fn, args, log)
                    if cell is None:
                        log("guard gave up; stopping")
                        w.close()
                        return
                    cells[str(c)] = cell
                del ref
                best = min(cells, key=lambda c: cells[c]["mean_ms"])
                tied = sorted(int(c) for c in cells if cells[c]["ci95_lo_ms"] <= cells[best]["ci95_hi_ms"])
                w.write({"kind": "timing", "key": keys[fam], "family": fam, "d": d, "L": L, "batch": b, "round": r,
                         "cells": cells, "best_chunk": int(best), "tied_chunks": tied,
                         "max_out_diff_across_chunks": max_diff, "gpu_state_before": s0,
                         "gpu_state_after": gpu_state() if args.record_gpu_state else None,
                         "peak_vram_gb": peak_vram_gb()})
                new_rows += 1
            del inp, out
            torch.cuda.empty_cache()
            if new_rows and new_rows % 20 == 0:
                el = time.time() - t_start
                log(f"{len(w.done)}/{total} rows ({new_rows} new, {el / 60:.1f} min)")
    w.close()
    write_json(args.out_dir / f"e12_tp_widths_{tag}.json",
               {**meta, "rows_done": len(w.done), "rows_planned": total, "finished_utc": time.strftime(
                   "%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    log(f"done: {len(w.done)}/{total} rows")


if __name__ == "__main__":
    main()
