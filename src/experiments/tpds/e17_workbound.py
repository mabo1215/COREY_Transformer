"""E17c: production kernels pushed out of the launch floor (review item M2b).

In the paper's measurements the Mamba-2 SSD pipeline is bound by launch overhead up to L = 2048 at the
370M width, and the Gated DeltaNet kernel always prefers its largest chunk (64) at the 370M-class head
layout. This experiment times both kernels at larger widths, longer sequences and larger batches, so
that the analysis can say at which shapes the chunk starts to matter and whether the per-shape optimum
moves:

  ssd   Mamba-2 SSD (``mamba_chunk_scan_combined``) at d_inner in ``--ssd-widths`` (default: the 370M,
        1.3B and 2.7B checkpoints, head dim 64, one group, state 128), every chunk in 32..512.
  gdn   Gated DeltaNet (``chunk_gated_delta_rule``) at the head layouts H x K x V in ``--gdn-shapes``
        (default: the 340M/1.3B public checkpoints 4x256x256, the paper's synthetic 4x256x512, and the
        16/32/64-head x 128 layouts of production hybrids), chunks 16, 32 and 64 (the kernel accepts no
        larger chunk).

Inputs are synthetic (the kernels are data-oblivious; E1/E9 bound value effects at the selected chunk
to a few percent), one Gaussian draw per shape reused for every chunk and round. Every (family, shape,
L, batch) is timed in ``--rounds`` rounds with shapes and chunks shuffled; one fsynced JSONL row per
(family, shape, round); a restart skips completed rows. The launch floor of an empty kernel is timed
once and stored in the meta row. With ``--exclusive-guard`` a cell is re-timed when another process
used the GPU during it.

Output: ``e17_workbound_<tag>.jsonl`` (rows) and ``e17_workbound_<tag>.json`` (meta + counts).
"""
from __future__ import annotations

import argparse
import random
import time
import zlib
from pathlib import Path

import torch

from src.experiments.tpds import kernels as K
from src.experiments.tpds.common import (GDN_CHUNKS, MAMBA2_CHUNKS, RESULTS_DIR, JsonlWriter, Logger, all_gpu_pids,
                                         check_disk, cuda_time, env_metadata, foreign_gpu_pids, gpu_state, host_name,
                                         peak_vram_gb, register_own_gpu_pids, summarize, wait_for_exclusive_gpu,
                                         write_json)

SSD_HEADDIM, SSD_NGROUPS, SSD_DSTATE = 64, 1, 128
DEFAULT_SSD_WIDTHS = "2048,4096,5120"            # d_inner of mamba2-370m / 1.3b / 2.7b (32 / 64 / 80 heads)
DEFAULT_GDN_SHAPES = "4x256x256,4x256x512,16x128x128,32x128x128,64x128x128"   # H x K x V


def ssd_runner(inp, chunk):
    return lambda: K.mamba2_ssd(inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"], chunk, D=inp["D"],
                                dt_bias=inp["dt_bias"])


def gdn_runner(inp, chunk):
    return lambda: K.gdn_chunk(inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], chunk_size=chunk)


def time_cell(fn, args, log):
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


def parse_shape(s: str) -> tuple[int, int, int]:
    h, k, v = (int(x) for x in s.lower().split("x"))
    return h, k, v


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--families", default="ssd,gdn")
    p.add_argument("--ssd-widths", default=DEFAULT_SSD_WIDTHS, help="d_inner values; heads = d_inner / 64")
    p.add_argument("--ssd-chunks", default=",".join(map(str, MAMBA2_CHUNKS)))
    p.add_argument("--gdn-shapes", default=DEFAULT_GDN_SHAPES, help="HxKxV layouts")
    p.add_argument("--gdn-chunks", default=",".join(map(str, GDN_CHUNKS)))
    p.add_argument("--lengths", default="512,1024,2048,4096,8192")
    p.add_argument("--batches", default="1,4,16")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--exclusive-guard", action="store_true")
    p.add_argument("--guard-until", type=float, default=None, help="epoch seconds after which the guard gives up")
    p.add_argument("--record-gpu-state", action="store_true")
    p.add_argument("--max-rows", type=int, default=None, help="stop after this many new rows (smoke)")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--tag", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def make_inputs(fam, shape, L, b, seed):
    if fam == "ssd":
        return K.make_ssd_inputs(b, L, shape // SSD_HEADDIM, SSD_HEADDIM, SSD_NGROUPS, SSD_DSTATE,
                                 dtype=torch.float16, seed=seed)
    h, k, v = shape
    return K.make_gdn_inputs(b, L, h, k, v, dtype=torch.bfloat16, seed=seed)


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    tag = args.tag or host_name()
    log = Logger(f"e17_workbound_{tag}")
    gpu_baseline = all_gpu_pids()
    check_disk(args.out_dir, need_gb=0.2, log=log)
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    own = register_own_gpu_pids(gpu_baseline, log=log) if args.exclusive_guard else set()
    fams = {}
    want = [f for f in args.families.split(",") if f]
    if "ssd" in want:
        if K.ssd_available():
            fams["ssd"] = {"shapes": [int(x) for x in args.ssd_widths.split(",") if x],
                           "chunks": [int(c) for c in args.ssd_chunks.split(",") if c]}
        else:
            log("Mamba-2 SSD (mamba_ssm Triton ops) not importable; skipping the ssd family")
    if "gdn" in want:
        if K.gdn_available():
            fams["gdn"] = {"shapes": [parse_shape(s) for s in args.gdn_shapes.split(",") if s],
                           "chunks": [int(c) for c in args.gdn_chunks.split(",") if c]}
        else:
            log("Gated DeltaNet (fla) not importable; skipping the gdn family")
    if not fams:
        raise SystemExit("no kernel family available")
    lengths = [int(x) for x in args.lengths.split(",") if x]
    batches = [int(x) for x in args.batches.split(",") if x]
    tiny = torch.zeros(1, device="cuda")
    floor = summarize(cuda_time(lambda: tiny.add_(0.0), 50, 500))
    log(f"empty-kernel launch floor: {floor['mean_ms'] * 1e3:.1f} us")
    meta = env_metadata()
    meta.update({"experiment": "E17c", "own_gpu_pids": sorted(own), "lengths": lengths, "batches": batches,
                 "families": {f: {"shapes": [list(s) if isinstance(s, tuple) else s for s in d["shapes"]],
                                  "chunks": d["chunks"]} for f, d in fams.items()},
                 "ssd_layout": {"headdim": SSD_HEADDIM, "ngroups": SSD_NGROUPS, "dstate": SSD_DSTATE},
                 "gdn_path": "plain (pre-normalized k, log-gate g, sigmoid beta; as the paper's synthetic study)",
                 "empty_kernel": floor, "args": {k: str(v) for k, v in vars(args).items()}})
    w = JsonlWriter(args.out_dir / f"e17_workbound_{tag}.jsonl", meta=meta)
    cells_plan = [(f, s, L, b) for f, d in fams.items() for s in d["shapes"] for L in lengths for b in batches]
    rng = random.Random(args.seed)
    new_rows, t_start = 0, time.time()
    total = len(cells_plan) * args.rounds
    log(f"{len(cells_plan)} (family, shape, L, batch) cells x {args.rounds} rounds = {total} rows, {len(w.done)} done")

    def shape_tag(f, s):
        return f"d{s}" if f == "ssd" else "x".join(map(str, s))

    for r in range(args.rounds):
        order = list(cells_plan)
        rng.shuffle(order)
        for f, s, L, b in order:
            key = f"e17|{f}|{shape_tag(f, s)}|L{L}|b{b}|r{r}"
            if key in w.done:
                continue
            if args.max_rows is not None and new_rows >= args.max_rows:
                break
            seed = args.seed + zlib.crc32(f"{f}|{shape_tag(f, s)}|{L}|{b}".encode()) % 100000
            inp = make_inputs(f, s, L, b, seed)
            s0 = gpu_state() if args.record_gpu_state else None
            chunks = list(fams[f]["chunks"])
            rng.shuffle(chunks)
            cells, ref, max_diff = {}, None, 0.0
            for c in chunks:
                fn = ssd_runner(inp, c) if f == "ssd" else gdn_runner(inp, c)
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
            row = {"kind": "timing", "key": key, "family": f, "shape": shape_tag(f, s), "L": L, "batch": b,
                   "round": r, "cells": cells, "best_chunk": int(best), "tied_chunks": tied,
                   "spread": cells[max(cells, key=lambda c: cells[c]["mean_ms"])]["mean_ms"] / cells[best]["mean_ms"],
                   "max_out_diff_across_chunks": max_diff, "gpu_state_before": s0,
                   "gpu_state_after": gpu_state() if args.record_gpu_state else None, "peak_vram_gb": peak_vram_gb()}
            if f == "ssd":
                row.update({"d_inner": s, "nheads": s // SSD_HEADDIM, "headdim": SSD_HEADDIM, "dstate": SSD_DSTATE,
                            "tokens": L * b, "width": s})
            else:
                row.update({"nheads": s[0], "dk": s[1], "dv": s[2], "tokens": L * b, "width": s[0] * s[2]})
            w.write(row)
            new_rows += 1
            del inp
            torch.cuda.empty_cache()
            log(f"{f} {shape_tag(f, s)} L={L} b={b} r{r}: best {best} (tied {tied}), spread {row['spread']:.2f}x, "
                f"{cells[best]['mean_ms']:.3f} ms")
            if new_rows % 20 == 0:
                log(f"{len(w.done)}/{total} rows ({new_rows} new, {(time.time() - t_start) / 60:.1f} min)")
    w.close()
    write_json(args.out_dir / f"e17_workbound_{tag}.json",
               {**meta, "rows_done": len(w.done), "rows_planned": total,
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    log(f"done: {len(w.done)}/{total} rows")


if __name__ == "__main__":
    main()
