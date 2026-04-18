"""
Calibrated Chunk-512 Latency Benchmark for RTX 3090.

Fills the TODO in Table 3 (main text):
  - COREY (default, H_ref=log K), chunk=512 on RTX 3090

Method: same Triton benchmark kernel used in run_cuda_profile_three_policies.py.
Runs static_chunk=[64, 256, 512] and COREY (H_ref=log K) with 30 repeats each.
Reports mean ± std latency per policy.

Usage (adama-cuda128 environment with Triton):
    python -m src.experiments.run_calibrated_chunk512_3090 \\
        --seq-len 4096 --repeats 30 \\
        --output-dir src/outputs/calibrated_chunk512_3090
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


# ---------------------------------------------------------------------------
# Triton kernel (same as run_cuda_profile_three_policies.py)
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
# Entropy + COREY chunk selection (H_ref = log K, default calibration)
# ---------------------------------------------------------------------------

def _hist_entropy(x: Any, num_bins: int = 256) -> float:
    import torch
    flat = x.detach().float().reshape(-1)
    vmin, vmax = float(flat.min()), float(flat.max())
    if vmax - vmin < 1e-8:
        return 0.0
    normalized = (flat - vmin) / (vmax - vmin + 1e-8)
    indices = (normalized * num_bins).long().clamp(0, num_bins - 1)
    counts = torch.zeros(num_bins, device=flat.device, dtype=torch.float32)
    counts.scatter_add_(0, indices, torch.ones_like(flat, dtype=torch.float32))
    prob = counts / (counts.sum() + 1e-10)
    log_prob = torch.where(prob > 1e-10, torch.log(prob + 1e-10), torch.zeros_like(prob))
    return float(-(prob * log_prob).sum().item())


def _entropy_to_chunk_logk(H: float, num_bins: int = 256,
                            c_min: int = 32, c_max: int = 512) -> int:
    h_ref = math.log(num_bins)
    ratio = min(H / h_ref, 1.0) if h_ref > 0 else 0.0
    raw = c_min + ratio * (c_max - c_min)
    rounded = int(2 ** round(math.log2(max(raw, 1.0))))
    return max(c_min, min(c_max, rounded))


# ---------------------------------------------------------------------------
# Launch helper
# ---------------------------------------------------------------------------

def _make_launch(x: Any, kernel: Any, L: int, B: int):
    BLOCK_LIMIT = 512
    if kernel is not None:
        def launch(cs: int) -> Any:
            blk = min(cs, BLOCK_LIMIT)
            blk = 1 << (blk - 1).bit_length()
            blk = max(32, min(blk, BLOCK_LIMIT))
            n_chunks = (L + blk - 1) // blk
            out = x.new_empty(x.shape)
            kernel[(B, n_chunks)](x, out, x.stride(0), L, blk)
            return out
    else:
        def launch(cs: int) -> Any:
            out = x.new_empty(x.shape)
            for s in range(0, L, cs):
                e = min(s + cs, L)
                out[:, s:e] = x[:, s:e] * 0.9 + 1.0
            return out
    return launch


# ---------------------------------------------------------------------------
# Single-policy timed run
# ---------------------------------------------------------------------------

def _time_policy(
    x: Any,
    kernel: Any,
    chunk_size: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    import torch

    B, L = x.shape
    launch = _make_launch(x, kernel, L, B)
    n_calls = (L + chunk_size - 1) // chunk_size

    for _ in range(warmup):
        launch(chunk_size)
    torch.cuda.synchronize()

    latencies: list[float] = []
    for _ in range(repeats):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        launch(chunk_size)
        t1.record()
        torch.cuda.synchronize()
        latencies.append(t0.elapsed_time(t1))

    return {
        "chunk": chunk_size,
        "n_calls": n_calls,
        "lat_mean_ms": round(mean(latencies), 4),
        "lat_std_ms":  round(pstdev(latencies), 4) if len(latencies) > 1 else 0.0,
        "lat_min_ms":  round(min(latencies), 4),
        "lat_max_ms":  round(max(latencies), 4),
        "repeats": repeats,
    }


# ---------------------------------------------------------------------------
# COREY (H_ref=log K) calibrated run
# ---------------------------------------------------------------------------

def _time_corey_logk(
    x: Any,
    kernel: Any,
    warmup: int,
    repeats: int,
    num_bins: int = 256,
) -> dict[str, Any]:
    import torch

    H = _hist_entropy(x, num_bins=num_bins)
    chunk = _entropy_to_chunk_logk(H, num_bins=num_bins)
    print(f"  [corey_logk] H={H:.4f} nats  H_ref=log({num_bins})={math.log(num_bins):.4f}  "
          f"r={H/math.log(num_bins):.3f}  chunk_selected={chunk}")

    result = _time_policy(x, kernel, chunk, warmup, repeats)
    result["policy"] = "corey_logk"
    result["entropy_nats"] = round(H, 6)
    result["h_ref"] = round(math.log(num_bins), 6)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-len",   type=int,   default=4096)
    p.add_argument("--batch",     type=int,   default=1)
    p.add_argument("--warmup",    type=int,   default=5)
    p.add_argument("--repeats",   type=int,   default=30)
    p.add_argument("--num-bins",  type=int,   default=256)
    p.add_argument("--output-dir", type=Path, default=Path("src/outputs/calibrated_chunk512_3090"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[calibrated_chunk512] GPU: {gpu_name}")
    print(f"[calibrated_chunk512] PyTorch: {torch.__version__}")
    if _TRITON_KERNEL is None:
        print("[calibrated_chunk512] WARNING: Triton not available, using Python fallback.")
    print(f"[calibrated_chunk512] seq_len={args.seq_len}  batch={args.batch}  repeats={args.repeats}")
    print()

    x = torch.randn(args.batch, args.seq_len, dtype=torch.float16, device=device)
    kernel = _TRITON_KERNEL
    results = []

    # Static policies: chunk = 64, 256, 512
    for cs in [64, 256, 512]:
        policy_label = f"static_{cs}"
        print(f"[calibrated_chunk512] Timing {policy_label} ({args.repeats} repeats) …")
        r = _time_policy(x, kernel, cs, args.warmup, args.repeats)
        r["policy"] = policy_label
        print(f"  {policy_label}: {r['lat_mean_ms']:.4f} ± {r['lat_std_ms']:.4f} ms  "
              f"({r['n_calls']} kernel calls)")
        results.append(r)

    # COREY (H_ref = log K) calibrated run
    print("[calibrated_chunk512] Timing COREY (H_ref=log K, default) …")
    r_corey = _time_corey_logk(x, kernel, args.warmup, args.repeats, num_bins=args.num_bins)
    print(f"  corey_logk: {r_corey['lat_mean_ms']:.4f} ± {r_corey['lat_std_ms']:.4f} ms  "
          f"({r_corey['n_calls']} kernel calls)")
    results.append(r_corey)

    # Speedup table
    static64_lat = next(r["lat_mean_ms"] for r in results if r["policy"] == "static_64")
    static512_lat = next(r["lat_mean_ms"] for r in results if r["policy"] == "static_512")
    print()
    print("[calibrated_chunk512] === Summary ===")
    print(f"{'Policy':<30} {'Chunk':>6} {'Calls':>6} {'Lat(ms)':>10} {'Std(ms)':>8} {'Spdup-A':>10} {'Spdup-B':>10}")
    for r in results:
        spdup_a = static64_lat / r["lat_mean_ms"] if r["lat_mean_ms"] > 0 else float("nan")
        spdup_b = static512_lat / r["lat_mean_ms"] if r["lat_mean_ms"] > 0 else float("nan")
        print(f"{r['policy']:<30} {r['chunk']:>6} {r['n_calls']:>6} "
              f"{r['lat_mean_ms']:>10.4f} {r['lat_std_ms']:>8.4f} "
              f"{spdup_a:>10.2f}x {spdup_b:>10.2f}x")

    output = {
        "gpu": gpu_name,
        "torch": torch.__version__,
        "triton_available": _TRITON_KERNEL is not None,
        "seq_len": args.seq_len,
        "batch": args.batch,
        "repeats": args.repeats,
        "platform": platform.platform(),
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "summary.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[calibrated_chunk512] Results saved to {out_path}")


if __name__ == "__main__":
    main()
