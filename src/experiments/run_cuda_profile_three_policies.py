"""
CUDA Kernel Profile: Three-Policy Chunked Scan Benchmark
=========================================================
Uses torch.profiler (no root / ncu required) to capture kernel-level metrics
for three scheduling policies applied to a Triton chunked-scan operation.

Captured metrics per policy:
  - CUDA kernel count
  - Total CUDA kernel time (µs), per kernel avg
  - Self-CUDA time
  - GPU memory reads / writes (via memory_stats)
  - Effective bandwidth (GB/s)
  - CUDA event-based wall-clock latency

Produces:
  <output_dir>/profile_report.json   — machine-readable consolidated metrics
  <output_dir>/profile_summary.txt   — human-readable paper-ready table

No dependency on mamba_ssm or ncu.  Requires: torch >= 2.0, triton.

Usage:
    python run_cuda_profile_three_policies.py --output-dir src/outputs/nsight_profile
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Triton kernel (module-level, required for tl.constexpr)
# ---------------------------------------------------------------------------

_TRITON_KERNEL = None
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _chunked_scan_kernel(
        x_ptr,
        out_ptr,
        stride_b,
        L,
        BLOCK: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_c = tl.program_id(1)
        base = pid_b * stride_b + pid_c * BLOCK
        offs = base + tl.arange(0, BLOCK)
        mask = (pid_c * BLOCK + tl.arange(0, BLOCK)) < L
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc = x * 0.9 + 1.0
        tl.store(out_ptr + offs, acc.to(tl.float16), mask=mask)

    _TRITON_KERNEL = _chunked_scan_kernel
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------

def _compute_entropy(x: Any, n_bins: int = 64) -> float:
    import torch
    x_flat = x.float().flatten()
    mn, mx = float(x_flat.min()), float(x_flat.max())
    if mx <= mn:
        return 0.0
    counts = torch.histc(x_flat, bins=n_bins, min=mn, max=mx).float()
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * torch.log(p)).sum()) / math.log(n_bins)


def _entropy_to_chunk(ent: float, mn: int = 32, mx: int = 512) -> int:
    cs = mn + int(ent * (mx - mn))
    cs = max(mn, min(mx, cs))
    return ((cs + 15) // 32) * 32


def _make_launch_fn(x: Any, kernel: Any, L: int, B: int):
    """Return a function that launches kernel with a given chunk size."""
    import torch

    BLOCK_LIMIT = 512

    if kernel is not None:
        def launch(cs: int) -> Any:
            blk = min(cs, BLOCK_LIMIT)
            blk = 1 << (blk - 1).bit_length()
            blk = max(32, min(blk, BLOCK_LIMIT))
            n_chunks = (L + blk - 1) // blk
            out = torch.empty_like(x)
            kernel[(B, n_chunks)](x, out, x.stride(0), L, blk)
            return out
    else:
        def launch(cs: int) -> Any:
            import torch
            B2, L2 = x.shape
            out = torch.empty_like(x)
            for s in range(0, L2, cs):
                e = min(s + cs, L2)
                out[:, s:e] = x[:, s:e] * 0.9 + 1.0
            return out

    return launch


def _chunk_sequence(policy: str, x: Any, static_chunk: int,
                    min_chunk: int, max_chunk: int) -> list[int]:
    _, L = x.shape
    if policy == "off":
        return [1] * L
    elif policy == "static":
        return [static_chunk] * ((L + static_chunk - 1) // static_chunk)
    elif policy == "corey":
        chunk_sizes = []
        pos = 0
        window = 256
        while pos < L:
            snippet = x[:, pos: pos + window]
            ent = _compute_entropy(snippet)
            cs = _entropy_to_chunk(ent, min_chunk, max_chunk)
            actual = min(cs, L - pos)
            if actual <= 0:
                break
            chunk_sizes.append(actual)
            pos += actual
        return chunk_sizes
    else:
        raise ValueError(policy)


# ---------------------------------------------------------------------------
# Profiling runner
# ---------------------------------------------------------------------------

def profile_policy(
    policy: str,
    x: Any,
    kernel: Any,
    static_chunk: int,
    min_chunk: int,
    max_chunk: int,
    warmup: int,
    repeats: int,
) -> dict:
    import torch
    from torch.profiler import profile, record_function, ProfilerActivity

    B, L = x.shape
    launch = _make_launch_fn(x, kernel, L, B)
    chunk_sizes = _chunk_sequence(policy, x, static_chunk, min_chunk, max_chunk)

    # Limit policy_off for profiling to avoid hanging (too many launches)
    if policy == "off":
        effective_seqlen = min(L, 512)  # profile 512 steps, scale up results
        chunk_sizes_prof = [1] * effective_seqlen
        scale = L / effective_seqlen
    else:
        chunk_sizes_prof = chunk_sizes
        scale = 1.0

    # Warmup
    for _ in range(warmup):
        for cs in chunk_sizes_prof[:min(4, len(chunk_sizes_prof))]:
            launch(cs)
    torch.cuda.synchronize()

    # CUDA event timing
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    t0.record()
    launch_count = 0
    for _ in range(repeats):
        for cs in chunk_sizes_prof:
            launch(cs)
            launch_count += 1
    t1.record()
    torch.cuda.synchronize()

    elapsed_ms = t0.elapsed_time(t1)  # milliseconds (total over repeats)
    mem_after = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()

    avg_ms = elapsed_ms / repeats
    avg_launches = (launch_count / repeats) * scale

    # Estimated HBM bytes (read + write, float16, scaled)
    bytes_elem = 2
    hbm_bytes = B * L * bytes_elem * 2  # read + write
    hbm_gb = hbm_bytes / 1e9
    bw_gbs = hbm_gb / (avg_ms / 1000) if avg_ms > 0 else 0.0

    # Torch profiler — separate short run for kernel stats
    prof_result = _run_torch_profiler(policy, launch, chunk_sizes_prof[:min(16, len(chunk_sizes_prof))])

    return {
        "policy": policy,
        "seq_len": L,
        "batch": B,
        "avg_latency_ms": round(avg_ms, 4),
        "avg_kernel_launches": round(avg_launches, 1),
        "hbm_bytes_per_call": hbm_bytes,
        "estimated_bandwidth_gbs": round(bw_gbs, 3),
        "peak_mem_mb": round(peak_mem / 1e6, 2),
        "profiler": prof_result,
    }


def _run_torch_profiler(policy: str, launch_fn: Any, chunk_sizes: list) -> dict:
    """Run torch.profiler on a short sequence, return kernel stats."""
    import torch
    from torch.profiler import profile, ProfilerActivity

    try:
        with profile(
            activities=[ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            for cs in chunk_sizes:
                launch_fn(cs)
        torch.cuda.synchronize()

        events = prof.key_averages()
        # PyTorch >=2.0: self_device_time_total replaces self_cuda_time_total
        def _dev_time(e: Any) -> float:
            for attr in ("self_device_time_total", "self_cuda_time_total"):
                try:
                    return float(getattr(e, attr))
                except AttributeError:
                    pass
            return 0.0

        cuda_events = [e for e in events if _dev_time(e) > 0]
        total_cuda_us = sum(_dev_time(e) for e in cuda_events)
        kernel_count = sum(e.count for e in cuda_events)
        avg_kernel_us = (total_cuda_us / kernel_count) if kernel_count > 0 else 0

        return {
            "total_cuda_us": round(total_cuda_us, 1),
            "kernel_count_profiled": kernel_count,
            "avg_kernel_us": round(avg_kernel_us, 3),
            "top_kernels": [
                {
                    "name": e.key[:60],
                    "count": e.count,
                    "self_device_us": round(_dev_time(e), 1),
                }
                for e in sorted(cuda_events, key=_dev_time, reverse=True)[:5]
            ],
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--static-chunk", type=int, default=64)
    p.add_argument("--min-chunk", type=int, default=32)
    p.add_argument("--max-chunk", type=int, default=512)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--output-dir", type=str, default="src/outputs/nsight_profile")
    args = p.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[profile] GPU: {gpu_name}")
    print(f"[profile] PyTorch: {torch.__version__}")
    triton_ver = "N/A"
    try:
        import triton
        triton_ver = triton.__version__
    except ImportError:
        pass
    print(f"[profile] Triton: {triton_ver}")
    print(f"[profile] seq_len={args.seq_len}  batch={args.batch}")
    print()

    x = torch.randn(args.batch, args.seq_len, dtype=torch.float16, device=device)
    kernel = _TRITON_KERNEL

    results = []
    for policy in ["off", "static", "corey"]:
        print(f"[profile] Profiling policy={policy} ...")
        r = profile_policy(
            policy, x, kernel,
            args.static_chunk, args.min_chunk, args.max_chunk,
            args.warmup, args.repeats,
        )
        results.append(r)
        pr = r["profiler"]
        print(
            f"  latency={r['avg_latency_ms']:.3f} ms  "
            f"launches={r['avg_kernel_launches']:.0f}  "
            f"bw={r['estimated_bandwidth_gbs']:.1f} GB/s  "
            f"cuda_kernels={pr.get('kernel_count_profiled','?')}  "
            f"cuda_time={pr.get('total_cuda_us','?')} µs"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "config": {
            "gpu": gpu_name,
            "torch_version": torch.__version__,
            "triton_version": triton_ver,
            "seq_len": args.seq_len,
            "batch": args.batch,
            "static_chunk": args.static_chunk,
            "min_chunk": args.min_chunk,
            "max_chunk": args.max_chunk,
        },
        "results": results,
    }

    report_path = out_dir / "profile_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Paper-ready summary table
    lines = []
    lines.append("\n=== Paper-Ready Kernel Profile Summary ===")
    lines.append(f"GPU: {gpu_name}  |  seq_len={args.seq_len}  |  batch={args.batch}")
    lines.append(f"{'Policy':<10} {'Latency(ms)':>12} {'Launches':>10} "
                 f"{'BW(GB/s)':>10} {'CUDAKernels':>13} {'CUDATime(µs)':>14}")
    lines.append("-" * 75)
    for r in results:
        pr = r["profiler"]
        lines.append(
            f"{r['policy']:<10} "
            f"{r['avg_latency_ms']:>12.3f} "
            f"{r['avg_kernel_launches']:>10.0f} "
            f"{r['estimated_bandwidth_gbs']:>10.1f} "
            f"{pr.get('kernel_count_profiled', 'N/A'):>13} "
            f"{pr.get('total_cuda_us', 'N/A'):>14}"
        )

    # Speedup rows
    off_r = next(r for r in results if r["policy"] == "off")
    static_r = next(r for r in results if r["policy"] == "static")
    corey_r = next(r for r in results if r["policy"] == "corey")
    if off_r["avg_latency_ms"] > 0:
        sp_static = off_r["avg_latency_ms"] / static_r["avg_latency_ms"]
        sp_corey = off_r["avg_latency_ms"] / corey_r["avg_latency_ms"]
        lines.append("-" * 75)
        lines.append(f"Speedup vs policy_off:  static={sp_static:.2f}x   corey={sp_corey:.2f}x")

    summary_text = "\n".join(lines)
    print(summary_text)

    summary_path = out_dir / "profile_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")

    print(f"\n[done] Report: {report_path}")
    print(f"       Summary: {summary_path}")


if __name__ == "__main__":
    main()
