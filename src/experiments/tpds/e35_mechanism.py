"""E35 - mechanism of the chunk effect: per-kernel breakdown and hardware counters (review taco-r2 M5e).

Cells (one call each, synthetic Gaussian inputs of the checkpoint shapes; the latency of these kernels does
not depend on the values at fixed placement):
  m1     Triton Mamba-1 scan (in-house), width 2048, state 16, the twelve tuning keys (L in {128, 512, 2048,
         8192} x batch {1, 4, 16}), every chunk along the default launch table, plus chunk 128 and 256 at the
         fixed launch configuration (BLOCK_D, num_warps) = (4, 2) (one variable at a time); and the same key
         (L = 8192, batch 1, chunk 64) in two buffer placements (fresh vs. 256-byte offset) for the L2 hit rate.
  ssd    Mamba-2 SSD (``mamba_chunk_scan_combined``; 370M shapes: 32 heads x 64, state 128, fp16) at
         L in {256, 512, 4096} (batch 1) and L = 2048 (batch 4), chunks 32..512. Five kernels per call.
  gdn    Gated DeltaNet 1.4B on the production path (shapes read from a captured call, or the 1.4B defaults),
         bf16, the same lengths, chunks 16/32/64.

Parts:
  --part breakdown   (no special rights) every cell is warmed up, then profiled with the PyTorch profiler
                     (CUPTI activity tracing) over ``--repeats`` calls; per kernel name the mean device time,
                     launch grid and block, registers per thread, static shared memory, estimated achieved
                     occupancy, warps per SM and blocks per SM as the profiler reports them. Output
                     ``e35_breakdown_<tag>.json``. This gives the per-kernel time breakdown versus chunk and the
                     number of programs (empty or masked programs appear as grid sizes when C > L).
  --part target      the process that Nsight Compute profiles: every cell runs ``--warmup`` unprofiled calls,
                     then one call inside the NVTX range ``e35cell``; a sidecar JSONL records the cell order.
  --part ncu         runs Nsight Compute on ``--part target`` (needs the right to read GPU performance counters,
                     i.e. root where the driver restricts profiling to administrators) and then parses.
  --part parse       re-parses an existing Nsight Compute CSV and writes ``e35_counters_<tag>.json``.

Counters: achieved and theoretical occupancy, warps per scheduler, issue-slot utilization, the main warp-stall
reasons per issued instruction, local-memory (spill) sectors loaded and stored, L1 and L2 hit rates, DRAM
bytes, duration and SM cycles; collected with ``--replay-mode kernel`` at natural clocks and without cache
control (``--clock-control none --cache-control none``), as the latency effects were measured that way.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.experiments.tpds.common import LOG_DIR, RESULTS_DIR, JsonlWriter, Logger, env_metadata, host_name, read_jsonl, \
    utc_now, write_json

NVTX_RANGE = "e35cell"
STALLS = ("long_scoreboard", "short_scoreboard", "wait", "barrier", "math_pipe_throttle", "mio_throttle",
          "lg_throttle", "not_selected", "selected", "no_instruction", "dispatch_stall", "branch_resolving",
          "imc_miss", "tex_throttle", "drain", "membar", "sleeping")
METRICS = (
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.max",
    "sm__cycles_elapsed.avg.per_second",
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "launch__waves_per_multiprocessor",
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
) + tuple(f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio" for s in STALLS)
SSD_LENGTHS = ((256, 1), (512, 1), (4096, 1), (2048, 4))
M1_KEYS = tuple((L, b) for L in (128, 512, 2048, 8192) for b in (1, 4, 16))


# --------------------------------------------------------------------------- cells
def build_cells(args, log) -> list[dict[str, Any]]:
    """Cell descriptions (no tensors)."""
    from src.experiments.tpds.common import GDN_CHUNKS, MAMBA1_CHUNKS, MAMBA2_CHUNKS

    fams = [f for f in args.families.split(",") if f]
    cells = []
    if "m1" in fams:
        for L, b in M1_KEYS:
            for c in MAMBA1_CHUNKS:
                cells.append({"family": "m1", "L": L, "batch": b, "chunk": c, "launch": "default", "placement": "fresh"})
            for c in (128, 256):
                cells.append({"family": "m1", "L": L, "batch": b, "chunk": c, "launch": "d4_w2", "placement": "fresh"})
        cells.append({"family": "m1", "L": 8192, "batch": 1, "chunk": 64, "launch": "default", "placement": "offset"})
    for fam, chunks in (("ssd", MAMBA2_CHUNKS), ("gdn", GDN_CHUNKS)):
        if fam in fams:
            for L, b in SSD_LENGTHS:
                for c in chunks:
                    cells.append({"family": fam, "L": L, "batch": b, "chunk": c, "launch": "library", "placement": "fresh"})
    for i, c in enumerate(cells):
        c["cell"] = i
    return cells


def gdn_dims(args) -> dict[str, int]:
    """(Hq, K, Hv, V) of the Gated DeltaNet 1.4B layers, from a captured call when available."""
    dims = {"Hq": 16, "K": 128, "Hv": 16, "V": 256}
    if args.gdn_trace_dir:
        try:
            from src.experiments.tpds.e33_value_invariance_graph import load_real

            real = load_real(Path(args.gdn_trace_dir), "gdn", print)
            if real is not None:
                call = real[0]
                dims = {"Hq": int(call["q"].shape[2]), "K": int(call["q"].shape[3]), "Hv": int(call["v"].shape[2]),
                        "V": int(call["v"].shape[3])}
        except Exception as exc:  # pragma: no cover
            print(f"could not read GDN dims from traces: {exc!r}")
    return dims


def make_fn(cell: dict[str, Any], dims: dict[str, int]):
    import torch

    from src.experiments.tpds import kernels as K

    L, b, c = cell["L"], cell["batch"], cell["chunk"]
    if cell["family"] == "m1":
        inp = K.make_mamba1_inputs(b, 2048, 16, L, dtype=torch.float16, seed=0)
        if cell["placement"] == "offset":
            for k in ("u", "delta"):
                base = torch.empty(inp[k].numel() + 128, dtype=inp[k].dtype, device="cuda")
                v = base[128:].view(inp[k].shape)
                v.copy_(inp[k])
                inp[k] = v
                inp[f"_{k}_base"] = base
        out = torch.empty_like(inp["u"])
        cfg = K.default_mamba1_config(c) if cell["launch"] == "default" else K.LaunchConfig(c, 4, 2, 1)
        return lambda: K.mamba1_scan(inp["u"], inp["delta"], inp["A"], inp["B"], inp["C"], inp["D"], inp.get("z"),
                                     None, True, config=cfg, out=out)
    if cell["family"] == "ssd":
        inp = K.make_ssd_inputs(b, L, 32, 64, 1, 128, dtype=torch.float16, seed=0)
        return lambda: K.mamba2_ssd(inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"], c, D=inp["D"],
                                    dt_bias=inp["dt_bias"])
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(b, L, dims["Hq"], dims["K"], device="cuda", generator=g).bfloat16()
    k = torch.randn(b, L, dims["Hq"], dims["K"], device="cuda", generator=g).bfloat16()
    v = torch.randn(b, L, dims["Hv"], dims["V"], device="cuda", generator=g).bfloat16()
    gg = torch.randn(b, L, dims["Hv"], device="cuda", generator=g).bfloat16()
    beta = torch.randn(b, L, dims["Hv"], device="cuda", generator=g).bfloat16()
    A_log = torch.log(torch.rand(dims["Hv"], device="cuda", generator=g) * 15 + 1)
    dt_bias = torch.randn(dims["Hv"], device="cuda", generator=g) * 0.1
    return lambda: K.gdn_chunk_prod(q, k, v, gg, beta, A_log, dt_bias, chunk_size=c)


# --------------------------------------------------------------------------- part: breakdown
def breakdown(args, log) -> None:
    import torch
    from torch.profiler import ProfilerActivity, profile

    cells = build_cells(args, log)
    dims = gdn_dims(args)
    out_rows = []
    trace_dir = Path(args.log_dir) / f"e35_traces_{args.tag}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        fn = make_fn(cell, dims)
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.repeats):
                fn()
            torch.cuda.synchronize()
        path = trace_dir / f"cell{cell['cell']}.json"
        prof.export_chrome_trace(str(path))
        kernels = parse_trace(path, args.repeats)
        path.unlink(missing_ok=True)
        out_rows.append({**cell, "kernels": kernels,
                         "total_us_per_call": round(sum(k["dur_us_per_call"] for k in kernels), 3)})
        log(f"cell {cell['cell']} {cell['family']} L={cell['L']} B={cell['batch']} c={cell['chunk']} "
            f"{cell['launch']}/{cell['placement']}: {len(kernels)} kernels, "
            f"{out_rows[-1]['total_us_per_call']:.1f} us/call")
        del fn
        torch.cuda.empty_cache()
    meta = env_metadata()
    meta.update({"experiment": "E35-breakdown", "repeats": args.repeats, "warmup": args.warmup, "gdn_dims": dims})
    path = args.out_dir / f"e35_breakdown_{args.tag}.json"
    write_json(path, {"meta": meta, "cells": out_rows})
    log(f"wrote {path}")


def parse_trace(path: Path, repeats: int) -> list[dict[str, Any]]:
    """Per kernel name: device time per call and launch attributes from a Chrome trace of the profiler."""
    d = json.load(open(path))
    by: dict[str, list[dict]] = defaultdict(list)
    for ev in d.get("traceEvents", []):
        if ev.get("cat") == "kernel" and ev.get("ph") == "X":
            by[ev["name"]].append(ev)
    out = []
    for name, evs in by.items():
        a = evs[0].get("args", {})
        short = name.split("(")[0][:80]
        out.append({"kernel": short, "n_launches": len(evs), "launches_per_call": round(len(evs) / repeats, 3),
                    "dur_us_per_call": round(sum(e["dur"] for e in evs) / repeats, 3),
                    "grid": a.get("grid"), "block": a.get("block"),
                    "registers_per_thread": a.get("registers per thread"),
                    "shared_memory": a.get("shared memory"),
                    "blocks_per_sm": a.get("blocks per SM"), "warps_per_sm": a.get("warps per SM"),
                    "est_achieved_occupancy_pct": a.get("est. achieved occupancy %")})
    out.sort(key=lambda k: -k["dur_us_per_call"])
    return out


# --------------------------------------------------------------------------- part: target / ncu / parse
def target(args, log) -> None:
    import torch

    cells = build_cells(args, log)
    dims = gdn_dims(args)
    side = JsonlWriter(Path(args.log_dir) / f"e35_cells_{args.tag}.jsonl")
    for cell in cells:
        fn = make_fn(cell, dims)
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(NVTX_RANGE)
        fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        side.write({"kind": "cell", **cell, "utc": utc_now()}, track=False)
        del fn
        torch.cuda.empty_cache()
    side.close()


def run_ncu(args, log) -> None:
    from src.experiments.tpds.e16_counters import available_metrics, find_ncu

    ncu = find_ncu(args.ncu)
    if ncu is None:
        raise SystemExit("Nsight Compute (ncu) not found")
    metrics, dropped = available_metrics(ncu, list(METRICS))
    if dropped:
        log(f"metrics unavailable on this ncu/GPU: {dropped}")
    csv_path = Path(args.log_dir) / f"e35_ncu_{args.tag}.csv"
    side = Path(args.log_dir) / f"e35_cells_{args.tag}.jsonl"
    side.unlink(missing_ok=True)
    cmd = [ncu, "--nvtx", "--nvtx-include", f"{NVTX_RANGE}/", "--replay-mode", "kernel",
           "--clock-control", "none", "--cache-control", "none", "--csv", "--page", "raw", "--print-units", "base",
           "--metrics", ",".join(metrics), "--log-file", str(csv_path), "--target-processes", "all",
           sys.executable, "-m", "src.experiments.tpds.e35_mechanism", "--part", "target", "--tag", args.tag,
           "--families", args.families, "--warmup", str(args.warmup), "--log-dir", str(args.log_dir)]
    if args.gdn_trace_dir:
        cmd += ["--gdn-trace-dir", args.gdn_trace_dir]
    t0 = time.time()
    log("running: " + " ".join(cmd[:12]) + " ...")
    rc = subprocess.run(cmd).returncode
    log(f"ncu exit code {rc} after {time.time() - t0:.0f} s")
    parse(args, log, ncu_meta={"ncu": ncu, "metrics": metrics, "metrics_dropped": dropped, "exit_code": rc})


def parse(args, log, ncu_meta: dict[str, Any] | None = None) -> None:
    from src.experiments.tpds.e16_counters import parse_ncu_csv

    csv_path = Path(args.log_dir) / f"e35_ncu_{args.tag}.csv"
    prof = parse_ncu_csv(csv_path)
    cells = [r for r in read_jsonl(Path(args.log_dir) / f"e35_cells_{args.tag}.jsonl") if r.get("kind") == "cell"]
    # kernels are profiled in launch order; assign them to cells by the cell's kernel count, which the
    # breakdown file gives per cell; without it, attach by order assuming one cell after another
    bd_path = args.out_dir / f"e35_breakdown_{args.breakdown_tag or args.tag}.json"
    counts = None
    if bd_path.exists():
        bd = json.load(open(bd_path))
        counts = {c["cell"]: int(round(sum(k["launches_per_call"] for k in c["kernels"]))) for c in bd["cells"]}
    rows, i = [], 0
    for cell in cells:
        n = counts.get(cell["cell"], 1) if counts else (1 if cell["family"] == "m1" else None)
        if n is None:
            raise SystemExit("per-cell kernel counts are needed for multi-kernel cells: run --part breakdown first")
        for r in prof[i: i + n]:
            rows.append({**{k: cell[k] for k in ("cell", "family", "L", "batch", "chunk", "launch", "placement")}, **r})
        i += n
    res = {"experiment": "E35-counters", "created_utc": utc_now(), "ncu": ncu_meta, "n_kernels": len(prof),
           "n_assigned": i, "rows": rows}
    path = args.out_dir / f"e35_counters_{args.tag}.json"
    write_json(path, res)
    log(f"wrote {path}: {len(rows)} kernel rows ({len(prof)} profiled)")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", choices=("breakdown", "target", "ncu", "parse"), default="breakdown")
    p.add_argument("--families", default="m1,ssd,gdn")
    p.add_argument("--gdn-trace-dir", default=None)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--ncu", default=None)
    p.add_argument("--breakdown-tag", default=None)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--log-dir", type=Path, default=LOG_DIR)
    p.add_argument("--tag", default=None)
    a = p.parse_args(argv)
    a.tag = a.tag or f"{host_name()}_e35"
    return a


def main(argv=None):
    args = parse_args(argv)
    log = Logger(f"e35_mechanism_{args.tag}", args.log_dir)
    {"breakdown": breakdown, "target": target, "ncu": run_ncu, "parse": parse}[args.part](args, log)


if __name__ == "__main__":
    main()
