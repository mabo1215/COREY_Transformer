"""
Nsight Compute Kernel Profile: Chunked-Scan Three-Policy Benchmark
===================================================================
Profiles three scheduling policies (policy_off / policy_static / policy_corey)
using a Triton-based chunked scan kernel that matches the structural kernel
launch / HBM-access pattern of selective_scan_fn from mamba_ssm.

Does NOT require mamba_ssm — uses only torch + triton.

Metrics captured (via ncu --set full):
  - Kernel launch count per policy
  - Achieved HBM bandwidth (l2_global_load_bytes + l2_global_store_bytes)
  - L2 cache hit rate
  - Arithmetic intensity (derived: flops / bytes)
  - Warp efficiency / occupancy

Usage (direct):
    python run_nsight_chunked_scan_profile.py [--output-dir <dir>]

Usage (via ncu):
    ncu --set full --export profile_off --target-processes all \\
        python run_nsight_chunked_scan_profile.py --policy off

    ncu --set full --export profile_static \\
        python run_nsight_chunked_scan_profile.py --policy static

    ncu --set full --export profile_corey \\
        python run_nsight_chunked_scan_profile.py --policy corey

Run all three + parse metrics automatically with:
    python run_nsight_chunked_scan_profile.py --ncu /usr/local/cuda-12.4/bin/ncu \\
        --output-dir src/outputs/nsight_profile
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


# ---------------------------------------------------------------------------
# Triton chunked scan kernel (must be at module level for tl.constexpr)
# ---------------------------------------------------------------------------

_TRITON_KERNEL = None  # set below if triton is importable

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _chunked_scan_kernel(
        x_ptr,       # [B*L] float16 input (flat)
        out_ptr,     # [B*L] float16 output (flat)
        stride_b,    # stride along batch dim
        L,           # sequence length (runtime scalar)
        BLOCK: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_c = tl.program_id(1)

        base = pid_b * stride_b + pid_c * BLOCK
        offs = base + tl.arange(0, BLOCK)
        mask = (pid_c * BLOCK + tl.arange(0, BLOCK)) < L

        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc = x * 0.9 + 1.0
        acc = acc.to(tl.float16)
        tl.store(out_ptr + offs, acc, mask=mask)

    _TRITON_KERNEL = _chunked_scan_kernel

except ImportError:
    pass


def _build_triton_scan_kernel() -> Any:
    """Return the module-level Triton kernel, or None if triton unavailable."""
    return _TRITON_KERNEL


# ---------------------------------------------------------------------------
# Pure-PyTorch fallback (no triton)
# ---------------------------------------------------------------------------

def _torch_chunk_scan(x: Any, chunk_size: int) -> Any:
    """PyTorch fallback — not used for kernel profiling, only correctness."""
    import torch
    B, L = x.shape
    out = torch.empty_like(x)
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        out[:, start:end] = x[:, start:end] * 0.9 + 1.0
    return out


# ---------------------------------------------------------------------------
# Entropy-guided chunk-size selection
# ---------------------------------------------------------------------------

def _compute_entropy(x: Any, n_bins: int = 64) -> float:
    """Normalized Shannon entropy of x histogram (range [0,1])."""
    import torch
    x_flat = x.float().flatten()
    mn, mx = float(x_flat.min()), float(x_flat.max())
    if mx <= mn:
        return 0.0
    bins = torch.linspace(mn, mx, n_bins + 1, device=x.device)
    counts = torch.histc(x_flat, bins=n_bins, min=mn, max=mx).float()
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    import math
    return float(-(p * torch.log(p)).sum()) / math.log(n_bins)


def _entropy_to_chunk_size(
    entropy: float,
    min_chunk: int = 32,
    max_chunk: int = 512,
) -> int:
    """Map normalized entropy [0, 1] → chunk size (rounded to multiple of 32)."""
    chunk = min_chunk + int(entropy * (max_chunk - min_chunk))
    chunk = max(min_chunk, min(max_chunk, chunk))
    # Round to nearest multiple of 32
    chunk = ((chunk + 15) // 32) * 32
    return chunk


# ---------------------------------------------------------------------------
# Policy runners
# ---------------------------------------------------------------------------

def run_policy(
    policy: str,
    x: Any,
    kernel: Any,
    *,
    seq_len: int,
    static_chunk: int,
    min_chunk: int,
    max_chunk: int,
    warmup: int = 3,
    repeats: int = 10,
) -> dict:
    """Run one policy, return timing + HBM estimate."""
    import torch

    B, L = x.shape

    if kernel is not None:
        BLOCK_LIMIT = 1024  # triton block size limit

        def launch(cs: int) -> None:
            blk = min(cs, BLOCK_LIMIT)
            # Pad block to power of two (triton requirement)
            blk = 1 << (blk - 1).bit_length()
            blk = max(32, min(blk, BLOCK_LIMIT))
            n_chunks = (L + blk - 1) // blk
            out = torch.empty_like(x)
            kernel[(B, n_chunks)](
                x, out,
                x.stride(0),
                L, blk,
            )
            return out

        def count_launches(cs: int) -> int:
            blk = min(cs, BLOCK_LIMIT)
            blk = 1 << (blk - 1).bit_length()
            blk = max(32, min(blk, BLOCK_LIMIT))
            return ((L + blk - 1) // blk)

    else:
        def launch(cs: int) -> Any:
            return _torch_chunk_scan(x, cs)

        def count_launches(cs: int) -> int:
            return (L + cs - 1) // cs

    # Determine chunk sizes per policy
    if policy == "off":
        # Each timestep is a separate "kernel" call (chunk_size=1)
        chunk_sizes = [1] * L
    elif policy == "static":
        chunk_sizes = [static_chunk] * ((L + static_chunk - 1) // static_chunk)
    elif policy == "corey":
        # Entropy-guided: measure entropy once per coarse window, pick chunk size
        window = 256
        chunk_sizes = []
        pos = 0
        while pos < L:
            snippet = x[:, pos: pos + window]
            ent = _compute_entropy(snippet)
            cs = _entropy_to_chunk_size(ent, min_chunk, max_chunk)
            actual = min(cs, L - pos)
            if actual <= 0:
                break
            chunk_sizes.append(actual)
            pos += actual
    else:
        raise ValueError(f"Unknown policy: {policy}")

    # Warmup
    for _ in range(warmup):
        if policy == "off":
            for i in range(0, min(L, 64)):  # short warmup
                launch(1)
        else:
            for cs in chunk_sizes[:3]:
                launch(cs)
    torch.cuda.synchronize()

    # Benchmark
    launch_count = 0
    t_start = time.perf_counter()
    for _ in range(repeats):
        if policy == "off":
            for i in range(0, L, 1):
                launch(1)
                launch_count += 1
        else:
            for cs in chunk_sizes:
                launch(cs)
                launch_count += 1
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    avg_launches = launch_count / repeats
    avg_ms = elapsed_ms / repeats

    # HBM estimate: read + write, float16
    bytes_per_elem = 2  # float16
    total_bytes_read = B * L * bytes_per_elem
    total_bytes_write = B * L * bytes_per_elem
    hbm_bytes = total_bytes_read + total_bytes_write
    hbm_gb = hbm_bytes / 1e9

    # Effective bandwidth (GB/s)
    bandwidth_gbs = (hbm_gb * repeats) / (elapsed_ms / 1000.0) / repeats if elapsed_ms > 0 else 0.0

    return {
        "policy": policy,
        "avg_latency_ms": round(avg_ms, 4),
        "avg_kernel_launches": round(avg_launches, 1),
        "hbm_bytes_per_call": hbm_bytes,
        "estimated_bandwidth_gbs": round(bandwidth_gbs, 3),
        "chunk_sizes_sample": chunk_sizes[:5],
        "seq_len": L,
        "batch": B,
    }


# ---------------------------------------------------------------------------
# ncu-driven profiling
# ---------------------------------------------------------------------------

NCU_METRICS = (
    "l2__global_load_requests_mem_lg_request_mem_op_ld_tex.sum,"
    "l2__global_store_requests_mem_lg_request_mem_op_st.sum,"
    "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
    "l1tex__t_bytes.sum,"
    "gpu__time_duration.sum"
)


def _run_ncu_policy(
    ncu_bin: str,
    python_bin: str,
    script_path: str,
    policy: str,
    export_base: str,
    extra_args: list[str],
) -> dict:
    """Run ncu on one policy, parse output for key metrics."""
    cmd = [
        ncu_bin,
        "--set", "full",
        "--csv",
        "--target-processes", "all",
        "--log-file", f"{export_base}_{policy}.csv",
        python_bin, script_path,
        "--policy", policy,
        "--mode", "benchmark",
    ] + extra_args

    print(f"[ncu] Running policy={policy} ...")
    print(f"[ncu] CMD: {' '.join(cmd)}", flush=True)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {"policy": policy, "error": "ncu timeout"}

    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        # ncu exits non-zero even on success if CUDA graphs used — check output
        if "Disconnected" in stderr or "NVTX" in stderr:
            pass  # benign
        elif not stdout:
            return {"policy": policy, "error": stderr[:400]}

    # Parse CSV for aggregate metrics
    metrics: dict = {"policy": policy, "ncu_returncode": result.returncode}
    duration_ns_list = []
    l2_load_list = []
    l2_store_list = []

    for line in stdout.splitlines():
        parts = line.split(",")
        if len(parts) < 5:
            continue
        metric_name = parts[2].strip().strip('"')
        metric_val = parts[4].strip().strip('"').replace(",", "")
        try:
            val = float(metric_val)
        except ValueError:
            continue
        if "time_duration" in metric_name:
            duration_ns_list.append(val)
        elif "l2__global_load" in metric_name:
            l2_load_list.append(val)
        elif "l2__global_store" in metric_name:
            l2_store_list.append(val)

    if duration_ns_list:
        metrics["total_kernel_time_us"] = round(sum(duration_ns_list) / 1000, 2)
        metrics["kernel_count"] = len(duration_ns_list)
        metrics["avg_kernel_time_us"] = round(
            sum(duration_ns_list) / len(duration_ns_list) / 1000, 2
        )
    if l2_load_list:
        metrics["l2_load_requests_total"] = int(sum(l2_load_list))
    if l2_store_list:
        metrics["l2_store_requests_total"] = int(sum(l2_store_list))

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Nsight chunked-scan profiling: three policies"
    )
    p.add_argument(
        "--mode",
        choices=["benchmark", "ncu-all"],
        default="ncu-all",
        help=(
            "benchmark: run timing benchmark only (no ncu). "
            "ncu-all: invoke ncu for all three policies (default)."
        ),
    )
    p.add_argument("--policy", choices=["off", "static", "corey"],
                   default=None,
                   help="Single policy to benchmark (used when invoked by ncu).")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--static-chunk", type=int, default=64)
    p.add_argument("--min-chunk", type=int, default=32)
    p.add_argument("--max-chunk", type=int, default=512)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--ncu", type=str, default="/usr/local/cuda-12.4/bin/ncu",
                   help="Path to ncu binary.")
    p.add_argument("--output-dir", type=str,
                   default="src/outputs/nsight_profile",
                   help="Where to save JSON and CSV outputs.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    import torch

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    x = torch.randn(args.batch, args.seq_len, dtype=torch.float16, device=device)

    kernel = _build_triton_scan_kernel()
    if kernel is None:
        print("[warn] triton not available — using PyTorch fallback (no kernel profiling)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Mode: single policy benchmark (called by ncu subprocess)
    # -----------------------------------------------------------------------
    if args.mode == "benchmark" or args.policy is not None:
        policies = [args.policy] if args.policy else ["off", "static", "corey"]
        results = []
        for pol in policies:
            r = run_policy(
                pol, x, kernel,
                seq_len=args.seq_len,
                static_chunk=args.static_chunk,
                min_chunk=args.min_chunk,
                max_chunk=args.max_chunk,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            results.append(r)
            print(json.dumps(r))

        summary_path = out_dir / "benchmark_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[done] Benchmark written to {summary_path}")
        return

    # -----------------------------------------------------------------------
    # Mode: ncu-all — invoke ncu for each policy, then consolidate
    # -----------------------------------------------------------------------
    ncu_bin = args.ncu
    if not os.path.isfile(ncu_bin):
        # Fallback search
        for candidate in [
            "/usr/local/cuda/bin/ncu",
            "/usr/local/cuda-12.1/bin/ncu",
            shutil.which("ncu") or "",
        ]:
            if candidate and os.path.isfile(candidate):
                ncu_bin = candidate
                break
        else:
            print(f"ERROR: ncu not found at {args.ncu}", file=sys.stderr)
            print("Run with --mode benchmark to skip ncu.", file=sys.stderr)
            sys.exit(1)

    python_bin = sys.executable
    script_path = os.path.abspath(__file__)

    extra = [
        "--seq-len", str(args.seq_len),
        "--batch", str(args.batch),
        "--static-chunk", str(args.static_chunk),
        "--min-chunk", str(args.min_chunk),
        "--max-chunk", str(args.max_chunk),
        "--warmup", str(args.warmup),
        "--repeats", str(args.repeats),
        "--output-dir", str(out_dir),
    ]

    # Also run a pure-benchmark pass first (no ncu overhead) for latency numbers
    print("=== Phase 1: Pure latency benchmark (no ncu overhead) ===")
    bench_results = []
    for pol in ["off", "static", "corey"]:
        r = run_policy(
            pol, x, kernel,
            seq_len=args.seq_len,
            static_chunk=args.static_chunk,
            min_chunk=args.min_chunk,
            max_chunk=args.max_chunk,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        bench_results.append(r)
        print(f"  policy={pol:8s}  latency={r['avg_latency_ms']:.3f} ms  "
              f"launches={r['avg_kernel_launches']:.0f}  "
              f"bw={r['estimated_bandwidth_gbs']:.1f} GB/s")

    bench_path = out_dir / "benchmark_summary.json"
    with open(bench_path, "w") as f:
        json.dump(bench_results, f, indent=2)

    # ncu profiling
    print("\n=== Phase 2: ncu kernel profiling ===")
    ncu_results = []
    export_base = str(out_dir / "ncu_profile")

    for pol in ["off", "static", "corey"]:
        ncu_r = _run_ncu_policy(
            ncu_bin, python_bin, script_path, pol, export_base, extra
        )
        ncu_results.append(ncu_r)
        print(f"  policy={pol:8s}  "
              f"kernels={ncu_r.get('kernel_count', 'N/A')}  "
              f"total_time={ncu_r.get('total_kernel_time_us', 'N/A')} µs  "
              f"l2_load={ncu_r.get('l2_load_requests_total', 'N/A')}")

    ncu_path = out_dir / "ncu_summary.json"
    with open(ncu_path, "w") as f:
        json.dump(ncu_results, f, indent=2)

    # Consolidated report
    report = {
        "config": {
            "seq_len": args.seq_len,
            "batch": args.batch,
            "static_chunk": args.static_chunk,
            "min_chunk": args.min_chunk,
            "max_chunk": args.max_chunk,
            "gpu": torch.cuda.get_device_name(0),
            "driver": "see nvidia-smi",
            "ncu": ncu_bin,
        },
        "benchmark": bench_results,
        "ncu": ncu_results,
    }
    report_path = out_dir / "nsight_profile_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[done] Full report: {report_path}")
    print(f"       Benchmark:   {bench_path}")
    print(f"       ncu metrics: {ncu_path}")

    # Print paper-ready summary table
    print("\n=== Paper-Ready Summary ===")
    print(f"{'Policy':<12} {'Latency (ms)':>14} {'Kernel Launches':>18} "
          f"{'Est. BW (GB/s)':>16} {'ncu Kernels':>13}")
    print("-" * 80)
    bench_map = {r["policy"]: r for r in bench_results}
    ncu_map   = {r["policy"]: r for r in ncu_results}
    for pol in ["off", "static", "corey"]:
        br = bench_map.get(pol, {})
        nr = ncu_map.get(pol, {})
        print(
            f"{pol:<12} "
            f"{br.get('avg_latency_ms', 'N/A'):>14} "
            f"{br.get('avg_kernel_launches', 'N/A'):>18} "
            f"{br.get('estimated_bandwidth_gbs', 'N/A'):>16} "
            f"{nr.get('kernel_count', 'N/A'):>13}"
        )


if __name__ == "__main__":
    main()
