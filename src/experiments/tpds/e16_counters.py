"""E16b - hardware counters behind the fixed-chunk input effects on the RTX PRO 6000 (review M6b).

E1/E1c found input effects of up to 43% at fixed shape and chunk on the RTX PRO 6000 (Triton
Mamba-1 scan, L=8192, chunks 32-256), as large between two draws of one distribution as between
distributions. E9 separated value from placement by timing only, and only on the RTX 3090. This
experiment profiles the same cells with Nsight Compute so that the paper can say *which* hardware
quantity moves when the latency moves:

  instructions   smsp__inst_executed.sum             must be identical across inputs (data-oblivious)
  global loads   l1tex__t_{requests,sectors}_pipe_lsu_mem_global_op_ld.sum   alignment / coalescing
  L2 / DRAM      lts__t_sector_hit_rate.pct, dram__bytes_{read,write}.sum    placement in L2 / DRAM
  clock          sm__cycles_elapsed.avg.per_second    DVFS (value-dependent power would show here)
  cycles         sm__cycles_elapsed.max               latency in clock-independent units
  occupancy      sm__warps_active.avg.pct_of_peak_sustained_active

Cells are the E9 designs (``e9_placement.py``): A same buffers with the values copied in place,
B fresh buffers per input, C one Gaussian draw in ten placements. Every cell runs ``--warmup``
unprofiled launches, then ``--launches`` launches inside the NVTX range ``e16cell`` that Nsight
Compute profiles. Profiling keeps the natural clocks and a warm L2 (``--clock-control none
--cache-control none``) because the latency effects were measured that way.

Modes:
  (default)   run Nsight Compute on this script with ``--target``, then parse and analyse.
  --target    the profiled process: builds the cells and writes the cell order (sidecar JSONL).
  --analyze   only re-parse the Nsight Compute CSV and rewrite the analysis.

Output: ``e16_counters_<tag>.jsonl`` (one row per profiled launch, cell description joined in),
``e16_counters_<tag>.json`` (analysis), and under ``src/logs/tpds/``: the raw CSV and the sidecar.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from src.experiments.tpds.common import (DEFAULT_TRACE_DIR, LOG_DIR, MAMBA1_CHUNKS, RESULTS_DIR, JsonlWriter, Logger,
                                         host_name, read_jsonl, utc_now, write_json)

KERNEL_REGEX = "regex:_mamba1_chunked_scan_kernel"
NVTX_RANGE = "e16cell"
METRICS = (
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.max",
    "sm__cycles_elapsed.avg.per_second",
    "dram__cycles_elapsed.avg.per_second",
    "smsp__inst_executed.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sector_hit_rate.pct",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
)
DURATION = "gpu__time_duration.sum"
CYCLES = "sm__cycles_elapsed.max"
CLOCK = "sm__cycles_elapsed.avg.per_second"
# counts that a data-oblivious kernel must reproduce exactly when only the values change
INVARIANT_COUNTS = ("smsp__inst_executed.sum", "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")


# --------------------------------------------------------------------------- profiled process
def target(args):
    """Build the E9 cells and run each one inside the profiled NVTX range; write the cell order."""
    import torch

    from src.experiments.tpds import e9_placement as E9
    from src.experiments.tpds.common import gpu_state
    from src.experiments.tpds.e1_value_invariance import input_names

    log = Logger(f"e16_target_{args.tag}")
    torch.zeros(1, device="cuda")
    rng = random.Random(args.seed)
    dists = input_names([d for d in args.inputs.split(",") if d], args.draws - 1)
    chunks = [int(c) for c in args.chunks.split(",")]
    designs = [d for d in args.designs.split(",") if d]
    side = open(args.sidecar, "w", encoding="utf-8")
    seq = 0
    with torch.no_grad():
        for L in [int(x) for x in args.lengths.split(",")]:
            for mode in [m for m in args.modes.split(",") if m]:
                fresh = E9.build_inputs(dists, mode, L, args.batch, args.seed, args.trace_dir, log)
                if not fresh:
                    continue
                base = next(iter(fresh.values()))
                shared = E9.clone_set(base)
                gauss = fresh.get("gaussian", base)
                n_off = min(len(E9.OFFSETS), args.placements // 2)
                offsets = [0] * (args.placements - n_off) + list(E9.OFFSETS[:n_off])
                placements = [E9.clone_set(gauss) if o == 0 else E9.offset_set(gauss, o) for o in offsets]
                out_a = torch.empty_like(shared["u"])
                out_c = [torch.empty_like(p["u"]) for p in placements]
                fresh_outs = {d: torch.empty_like(v["u"]) for d, v in fresh.items()}
                ptrs = {f"p{i}": {"offset_bytes": o, **E9.pointer_info(p)}
                        for i, (o, p) in enumerate(zip(offsets, placements))}
                for r in range(args.rounds):
                    order = list(chunks)
                    rng.shuffle(order)
                    for c in order:
                        cells = [(dsg, name) for dsg in designs
                                 for name in (list(fresh) if dsg in "AB" else [f"p{i}" for i in range(len(placements))])]
                        rng.shuffle(cells)
                        for dsg, name in cells:
                            if dsg == "A":
                                for k in E9.TENSORS:
                                    if torch.is_tensor(shared.get(k)):
                                        shared[k].copy_(fresh[name][k])
                                inp, out = shared, out_a
                            elif dsg == "B":
                                inp, out = fresh[name], fresh_outs[name]
                            else:
                                inp, out = placements[int(name[1:])], out_c[int(name[1:])]
                            fn = E9.triton_runner(inp, out, c)
                            for _ in range(args.warmup):
                                fn()
                            torch.cuda.synchronize()
                            s0 = gpu_state()
                            torch.cuda.nvtx.range_push(NVTX_RANGE)
                            for _ in range(args.launches):
                                fn()
                            torch.cuda.synchronize()
                            torch.cuda.nvtx.range_pop()
                            row = {"seq": seq, "design": dsg, "L": L, "batch": args.batch, "chunk": c, "input": name,
                                   "mode": mode, "round": r, "launches": args.launches, "gpu_state_before": s0,
                                   "pointers": ptrs[name] if dsg == "C" else E9.pointer_info(inp)}
                            side.write(json.dumps(row) + "\n")
                            side.flush()
                            seq += 1
                        log(f"L={L} {mode} r{r} c={c}: {seq} cells so far")
                del fresh, shared, placements, out_c, fresh_outs
                torch.cuda.empty_cache()
    side.close()
    log(f"target done: {seq} cells, {seq * args.launches} profiled launches expected")


# --------------------------------------------------------------------------- Nsight Compute
def find_ncu(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit if Path(explicit).exists() or shutil.which(explicit) else None
    found = shutil.which("ncu")
    if found:
        return found
    cands = [Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / "ncu"]
    cands += sorted(Path("/opt/nvidia").glob("nsight-compute/*/ncu"), reverse=True)
    cands += sorted(Path("/usr/local").glob("cuda*/bin/ncu"), reverse=True)
    for c in cands:
        if c.exists():
            return str(c)
    return None


def available_metrics(ncu: str, metrics, chip: str | None = None) -> tuple[list[str], list[str]]:
    """Split ``metrics`` into those whose base name Nsight Compute lists for this device (or chip) and the rest."""
    cmd = [ncu, "--query-metrics"] + (["--chip", chip] if chip else [])
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout
    except (OSError, subprocess.SubprocessError):
        return list(metrics), []
    names = {line.split()[0] for line in out.splitlines() if line.strip() and "__" in line.split()[0]}
    if not names:  # listing failed (no device and no chip); let the profiler decide
        return list(metrics), []
    keep, drop = [], []
    for m in metrics:
        (keep if m.split(".")[0] in names else drop).append(m)
    return keep, drop


def _num(s: str) -> float | None:
    s = (s or "").strip().replace(",", "")
    if s in ("", "n/a", "N/A", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_ncu_csv(path: Path) -> list[dict]:
    """Rows of ``ncu --csv --page raw --print-units base`` (one per profiled kernel), metrics as floats."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [ln for ln in fh if not ln.startswith("==")]
    start = next((i for i, ln in enumerate(lines) if ln.startswith('"ID"')), None)
    if start is None:
        return []
    reader = csv.reader(lines[start:])
    header = next(reader)
    rows = []
    for rec in reader:
        if not rec or len(rec) != len(header):
            continue
        d = dict(zip(header, rec))
        if not (d.get("ID") or "").strip().isdigit():
            continue  # units row
        row = {"ncu_id": int(d["ID"]), "kernel_name": d.get("Kernel Name", ""),
               "block_size": d.get("Block Size"), "grid_size": d.get("Grid Size")}
        for k, v in d.items():
            if "__" in k:
                row[k] = _num(v)
        rows.append(row)
    rows.sort(key=lambda r: r["ncu_id"])
    return rows


def join_rows(cells: list[dict], prof: list[dict]) -> tuple[list[dict], str | None]:
    """Assign the profiled launches to the cells in order (``launches`` per cell)."""
    need = sum(c["launches"] for c in cells)
    err = None if need == len(prof) else f"expected {need} profiled launches from {len(cells)} cells, got {len(prof)}"
    out, i = [], 0
    for c in cells:
        for j in range(c["launches"]):
            if i >= len(prof):
                return out, err
            out.append({"kind": "counters",
                        "key": f"e16|{c['design']}|L{c['L']}|b{c['batch']}|c{c['chunk']}|{c['input']}|{c['mode']}"
                               f"|r{c['round']}|l{j}",
                        **{k: v for k, v in c.items() if k != "seq"}, "cell_seq": c["seq"], "launch": j, **prof[i]})
            i += 1
    return out, err


def run(args):
    tag = args.tag or host_name()
    log = Logger(f"e16_counters_{tag}")
    ncu = find_ncu(args.ncu)
    if ncu is None:
        log("Nsight Compute (ncu) not found; set --ncu or install nsight-compute")
        raise SystemExit(4)
    ver = subprocess.run([ncu, "--version"], capture_output=True, text=True).stdout.strip().splitlines()
    log(f"ncu: {ncu} ({ver[-1] if ver else '?'})")
    metrics, dropped = available_metrics(ncu, [m for m in args.metrics.split(",") if m])
    if dropped:
        log(f"metrics not available on this device, dropped: {dropped}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LOG_DIR / f"e16_counters_{tag}.csv"
    sidecar = LOG_DIR / f"e16_cells_{tag}.jsonl"
    me = [sys.executable, "-m", "src.experiments.tpds.e16_counters", "--target", "--sidecar", str(sidecar),
          "--tag", tag, "--lengths", args.lengths, "--batch", str(args.batch), "--inputs", args.inputs,
          "--draws", str(args.draws), "--placements", str(args.placements), "--modes", args.modes,
          "--chunks", args.chunks, "--designs", args.designs, "--rounds", str(args.rounds),
          "--warmup", str(args.warmup), "--launches", str(args.launches), "--trace-dir", str(args.trace_dir),
          "--seed", str(args.seed)]
    cmd = [ncu, "--nvtx", "--nvtx-include", f"{NVTX_RANGE}/", "--kernel-name", KERNEL_REGEX,
           "--metrics", ",".join(metrics), "--clock-control", "none", "--cache-control", "none",
           "--replay-mode", "kernel", "--csv", "--page", "raw", "--print-units", "base",
           "--log-file", str(csv_path), *me]
    log("profiling: " + " ".join(cmd))
    t0 = utc_now()
    rc = subprocess.run(cmd).returncode
    log(f"ncu exit code {rc}")
    text = csv_path.read_text(errors="replace") if csv_path.exists() else ""
    if "ERR_NVGPUCTRPERM" in text:
        log("performance counters are not accessible (ERR_NVGPUCTRPERM): the container needs profiling "
            "permission (NVreg_RestrictProfilingToAdminUsers=0 on the host, or CAP_SYS_ADMIN)")
        raise SystemExit(5)
    meta = {"experiment": "E16b", "tag": tag, "ncu": ncu, "ncu_version": ver[-1] if ver else None,
            "metrics": metrics, "metrics_dropped": dropped, "started_utc": t0, "ncu_exit_code": rc,
            "command": cmd, "args": {k: str(v) for k, v in vars(args).items()}}
    try:
        from src.experiments.tpds.common import env_metadata
        meta.update(env_metadata())
        meta["experiment"] = "E16b"
    except Exception:  # noqa: BLE001 - metadata is best effort
        pass
    write_rows(args, tag, meta, csv_path, sidecar, log)
    if rc != 0:
        raise SystemExit(rc)


def write_rows(args, tag, meta, csv_path, sidecar, log):
    cells = read_jsonl(sidecar)
    prof = parse_ncu_csv(csv_path)
    rows, err = join_rows(cells, prof)
    if err:
        log("WARNING: " + err)
    meta = {**meta, "n_cells": len(cells), "n_profiled": len(prof), "join_error": err}
    out = args.out_dir / f"e16_counters_{tag}.jsonl"
    if out.exists():
        out.unlink()
    w = JsonlWriter(out, meta=meta)
    for r in rows:
        w.write(r)
    w.close()
    log(f"{len(rows)} profiled launches written to {out}")
    analyze(args)
    if err and not args.allow_mismatch:
        raise SystemExit(6)


# --------------------------------------------------------------------------- analysis
def _range_pct(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = statistics.fmean(xs)
    return None if m == 0 else 100.0 * (max(xs) - min(xs)) / abs(m)


def _pearson(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)


def _median(xs):
    xs = [x for x in xs if x is not None]
    return float(statistics.median(xs)) if xs else None


def group_analysis(rows: list[dict], metrics: list[str]) -> list[dict]:
    """Per (design, L, mode, chunk): range over inputs/placements of every metric, and what tracks the duration."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["design"], r["L"], r["mode"], r["chunk"])].append(r)
    out = []
    for (dsg, L, mode, c), rs in sorted(groups.items()):
        by_in = defaultdict(list)
        for r in rs:
            by_in[r["input"]].append(r)
        names = sorted(by_in)
        per_in = {m: [statistics.fmean([r[m] for r in by_in[n] if r.get(m) is not None])
                      if any(r.get(m) is not None for r in by_in[n]) else None for n in names] for m in metrics}
        rec = {"design": dsg, "L": L, "mode": mode, "chunk": c, "n_inputs": len(names), "n_launches": len(rs),
               "range_pct": {m: _range_pct(per_in[m]) for m in metrics},
               "invariant": {m: (len({r.get(m) for r in rs}) == 1) for m in INVARIANT_COUNTS if m in metrics}}
        dur = [r.get(DURATION) for r in rs]
        rec["corr_with_duration"] = {m: _pearson([r.get(m) for r in rs], dur) for m in metrics if m != DURATION}
        # decompose the duration range into cycles and clock: duration = cycles / clock
        if CYCLES in metrics and CLOCK in metrics:
            rec["cycles_vs_clock"] = {"range_cycles_pct": rec["range_pct"].get(CYCLES),
                                      "range_clock_pct": rec["range_pct"].get(CLOCK)}
        cands = {m: v for m, v in rec["corr_with_duration"].items()
                 if v is not None and m not in (CYCLES, "smsp__inst_executed.sum")}
        rec["best_tracking_metric"] = max(cands, key=lambda m: abs(cands[m])) if cands else None
        out.append(rec)
    return out


def summarize_design(groups: list[dict], metrics: list[str]) -> dict:
    res = {}
    for dsg in sorted({g["design"] for g in groups}):
        gs = [g for g in groups if g["design"] == dsg]
        d = {"n_groups": len(gs),
             "median_range_pct": {m: _median([g["range_pct"].get(m) for g in gs]) for m in metrics},
             "max_range_pct": {m: max((g["range_pct"][m] for g in gs if g["range_pct"].get(m) is not None),
                                      default=None) for m in metrics},
             "share_invariant": {m: sum(1 for g in gs if g["invariant"].get(m)) / len(gs)
                                 for m in INVARIANT_COUNTS if m in metrics},
             "best_tracking_metric_counts": dict(sorted(
                 ((m, sum(1 for g in gs if g["best_tracking_metric"] == m))
                  for m in {g["best_tracking_metric"] for g in gs if g["best_tracking_metric"]}),
                 key=lambda kv: -kv[1]))}
        res[dsg] = d
    return res


def analyze(args):
    tag = args.tag or host_name()
    path = args.out_dir / f"e16_counters_{tag}.jsonl"
    meta = next(iter(read_jsonl(path, kind="meta")), {})
    rows = [r for r in read_jsonl(path) if r.get("kind") == "counters"]
    metrics = meta.get("metrics") or [m for m in METRICS if any(m in r for r in rows)]
    groups = group_analysis(rows, metrics)
    res = {"experiment": "E16b-analysis", "tag": tag, "n_launches": len(rows), "metrics": metrics,
           "join_error": meta.get("join_error"), "designs": summarize_design(groups, metrics), "groups": groups}
    write_json(args.out_dir / f"e16_counters_{tag}.json", res)
    for dsg, d in res["designs"].items():
        mr = d["median_range_pct"]
        fmt = lambda v: "n/a" if v is None else f"{v:.1f}%"  # noqa: E731
        print(f"design {dsg}: {d['n_groups']} groups; median range duration {fmt(mr.get(DURATION))}, "
              f"cycles {fmt(mr.get(CYCLES))}, clock {fmt(mr.get(CLOCK))}, "
              f"L2 hit {fmt(mr.get('lts__t_sector_hit_rate.pct'))}; instruction count invariant in "
              f"{d['share_invariant'].get('smsp__inst_executed.sum', float('nan')):.0%} of groups; "
              f"tracking metrics {d['best_tracking_metric_counts']}")
    return res


# --------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lengths", default="8192,2048")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--inputs", default="uniform,gaussian,laplace,sparse50,sparse10,sparse02,constant,real")
    p.add_argument("--draws", type=int, default=2, help="draws per synthetic distribution")
    p.add_argument("--placements", type=int, default=10)
    p.add_argument("--modes", default="act,all")
    p.add_argument("--chunks", default=",".join(str(c) for c in MAMBA1_CHUNKS))
    p.add_argument("--designs", default="A,B,C")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--launches", type=int, default=2, help="profiled launches per cell")
    p.add_argument("--metrics", default=",".join(METRICS))
    p.add_argument("--ncu", default=None, help="path of the ncu binary (default: PATH, CUDA_HOME, /opt/nvidia)")
    p.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    p.add_argument("--target", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--sidecar", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--analyze", action="store_true", help="only re-parse the CSV and rewrite the analysis")
    p.add_argument("--allow-mismatch", action="store_true", help="do not fail when cells and profiles disagree")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--tag", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.tag = args.tag or host_name()
    if args.target:
        target(args)
    elif args.analyze:
        tag = args.tag
        csv_path, sidecar = LOG_DIR / f"e16_counters_{tag}.csv", LOG_DIR / f"e16_cells_{tag}.jsonl"
        if csv_path.exists() and sidecar.exists():
            old = next(iter(read_jsonl(args.out_dir / f"e16_counters_{tag}.jsonl", kind="meta")), {}) \
                if (args.out_dir / f"e16_counters_{tag}.jsonl").exists() else {}
            write_rows(args, tag, {k: v for k, v in old.items() if k != "kind"}, csv_path, sidecar,
                       Logger(f"e16_counters_{tag}"))
        else:
            analyze(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
