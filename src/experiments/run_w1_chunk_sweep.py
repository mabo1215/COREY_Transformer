"""
W1 Chunk-Size Sweep: Static-Oracle Latency Curve + COREY Entropy-Selected Point.

This benchmark sweeps chunk_size in {32, 64, 128, 256, 512} under the
policy_static scheduling policy and overlays the COREY entropy-guided chunk
selection to show where COREY lands relative to the static oracle.

Key questions answered:
  (a) How does kernel latency vary across chunk sizes for the same workload?
  (b) Does COREY's entropy-selected chunk_size fall at or near the minimum?
  (c) How does COREY compare to the fixed oracle (best static chunk_size)?

Usage (WSL2 / Linux with CUDA):
  python -m src.experiments.run_w1_chunk_sweep \\
      --seq-len 4096 --dim 1024 --d-state 16 \\
      --warmup-runs 5 --benchmark-repeats 30 \\
      --output-dir src/outputs/w1_chunk_sweep

The sweep runs in the adama-cuda128 WSL2 environment (Python 3.11,
torch 2.11.0+cu128, RTX 3070) using mamba_ssm selective_scan_fn.
Results are written as CSV and JSON for table/figure insertion into the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


# ---------------------------------------------------------------------------
# HBM traffic estimator (replicated from run_w1_triton_triplet for standalone use)
# ---------------------------------------------------------------------------

def _tensor_nbytes(tensor: Any) -> int:
    return int(tensor.numel() * tensor.element_size())


def _estimate_policy_hbm_bytes(
    *,
    batch_size: int,
    dim: int,
    seq_len: int,
    d_state: int,
    input_bytes: int,
    state_bytes: int,
    param_bytes: int,
    chunk_size: int,
    n_chunks: int,
) -> dict[str, float]:
    estimated_bytes = n_chunks * (
        2 * batch_size * dim * chunk_size * input_bytes
        + 2 * state_bytes
        + param_bytes
    ) + batch_size * dim * seq_len * input_bytes
    estimated_gb = estimated_bytes / 1e9
    estimated_gib = estimated_bytes / float(1 << 30)
    return {
        "estimated_hbm_bytes": float(estimated_bytes),
        "estimated_hbm_gb": round(estimated_gb, 6),
        "estimated_hbm_gib": round(estimated_gib, 6),
    }


# ---------------------------------------------------------------------------
# Entropy helpers (replicated inline for standalone use)
# ---------------------------------------------------------------------------

def _hist_entropy(values: Any, num_bins: int = 256) -> float:
    """Shannon entropy of a 1-D float tensor via fixed-width histogram (nats)."""
    import torch

    flat = values.float().reshape(-1)
    vmin, vmax = float(flat.min()), float(flat.max())
    if vmax - vmin < 1e-8:
        return 0.0
    normalized = (flat - vmin) / (vmax - vmin + 1e-8)
    indices = (normalized * num_bins).long().clamp(0, num_bins - 1)
    counts = torch.zeros(num_bins, device=flat.device, dtype=torch.float32)
    counts.scatter_add_(0, indices, torch.ones_like(flat, dtype=torch.float32))
    prob = counts / (counts.sum() + 1e-10)
    log_prob = torch.where(
        prob > 1e-10,
        torch.log(prob + 1e-10),
        torch.zeros_like(prob),
    )
    return float(-(prob * log_prob).sum().item())


def _entropy_to_chunk_size(
    entropy_nats: float,
    min_chunk: int = 32,
    max_chunk: int = 512,
    ref_max_entropy: float = 8.0,
) -> int:
    """Map activation entropy (nats) to chunk_size for COREY policy."""
    ratio = min(entropy_nats / ref_max_entropy, 1.0)
    chunk = min_chunk + ratio * (max_chunk - min_chunk)
    log2 = math.log2(max(chunk, 1.0))
    rounded = 2 ** round(log2)
    return max(min_chunk, min(max_chunk, int(rounded)))


# ---------------------------------------------------------------------------
# Core benchmark: run one (policy_name, chunk_size) configuration
# ---------------------------------------------------------------------------

def _run_chunked_benchmark(
    u: Any,
    delta: Any,
    A: Any,
    B: Any,
    C: Any,
    D: Any,
    *,
    chunk_size: int,
    warmup_runs: int = 5,
    benchmark_repeats: int = 30,
    policy_name: str = "static",
    entropy_nats: float | None = None,
    entropy_overhead_ms: float | None = None,
) -> dict[str, Any]:
    """
    Run selective_scan_fn on fixed-size chunks and collect timing statistics.

    This is the core microbenchmark used for both the static sweep points and
    the COREY entropy-selected point.
    """
    import torch
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    seq_len = u.shape[2]
    batch_size, d_inner, _ = u.shape
    d_state = A.shape[-1]
    input_bytes = u.element_size()
    state_bytes = batch_size * d_inner * d_state * input_bytes
    param_bytes = sum(_tensor_nbytes(t) for t in (A, B, C, D))
    starts = list(range(0, seq_len, chunk_size))
    n_chunks = len(starts)

    def _one_pass() -> Any:
        outs: list[Any] = []
        for start in starts:
            end = min(start + chunk_size, seq_len)
            out_c = selective_scan_fn(
                u[:, :, start:end],
                delta[:, :, start:end],
                A,
                B,
                C,
                D=D,
                delta_softplus=True,
            )
            outs.append(out_c)
        return torch.cat(outs, dim=2)

    for _ in range(warmup_runs):
        _ = _one_pass()
    torch.cuda.synchronize()

    latencies: list[float] = []
    for _ in range(benchmark_repeats):
        t0 = time.perf_counter()
        out = _one_pass()
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    output_mean = float(out.float().nanmean().item()) if out.isfinite().any() else float("nan")
    traffic = _estimate_policy_hbm_bytes(
        batch_size=batch_size,
        dim=d_inner,
        seq_len=seq_len,
        d_state=d_state,
        input_bytes=input_bytes,
        state_bytes=state_bytes,
        param_bytes=param_bytes,
        chunk_size=chunk_size,
        n_chunks=n_chunks,
    )
    tokens_per_second = (batch_size * seq_len) / (mean(latencies) / 1000.0)
    estimated_bandwidth_gbps = traffic["estimated_hbm_gb"] / (mean(latencies) / 1000.0)

    return {
        "policy": policy_name,
        "chunk_size": chunk_size,
        "n_chunks": n_chunks,
        "entropy_nats": round(entropy_nats, 6) if entropy_nats is not None else None,
        "entropy_overhead_ms": round(entropy_overhead_ms, 4) if entropy_overhead_ms is not None else None,
        "latency_mean_ms": round(mean(latencies), 4),
        "latency_std_ms": round(pstdev(latencies), 4) if len(latencies) > 1 else 0.0,
        "latency_min_ms": round(min(latencies), 4),
        "latency_max_ms": round(max(latencies), 4),
        "latency_p50_ms": round(sorted(latencies)[len(latencies) // 2], 4),
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 4),
        "output_checksum": round(output_mean, 8) if math.isfinite(output_mean) else None,
        "finite_ratio": round(float(out.isfinite().float().mean().item()), 6),
        "kernel_launches": n_chunks,
        "tokens_per_second": round(tokens_per_second, 4),
        "estimated_hbm_bytes": int(traffic["estimated_hbm_bytes"]),
        "estimated_hbm_gb": traffic["estimated_hbm_gb"],
        "estimated_hbm_gib": traffic["estimated_hbm_gib"],
        "estimated_hbm_bandwidth_gbps": round(estimated_bandwidth_gbps, 4),
    }


# ---------------------------------------------------------------------------
# Main sweep runner
# ---------------------------------------------------------------------------

SWEEP_CHUNK_SIZES: list[int] = [32, 64, 128, 256, 512]


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for the W1 chunk sweep.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run this script in WSL2 with GPU access.")

    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "mamba_ssm is required. Install it in the WSL2 CUDA environment."
        ) from exc

    device = torch.device("cuda")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    # Synthetic activations matching the W1 triplet benchmark shape.
    u = torch.randn(args.batch_size, args.dim, args.seq_len, device=device, dtype=dtype, generator=rng)
    delta = torch.rand(args.batch_size, args.dim, args.seq_len, device=device, dtype=dtype, generator=rng)
    A = torch.randn(args.dim, args.d_state, device=device, dtype=torch.float32)
    B = torch.randn(args.dim, args.d_state, device=device, dtype=torch.float32)
    C = torch.randn(args.dim, args.d_state, device=device, dtype=torch.float32)
    D = torch.randn(args.dim, device=device, dtype=torch.float32)

    device_name = torch.cuda.get_device_name(device)
    print(f"[ChunkSweep] Device: {device_name}")
    print(f"[ChunkSweep] Config: batch={args.batch_size}, dim={args.dim}, "
          f"seq_len={args.seq_len}, d_state={args.d_state}, dtype={args.dtype}")

    # --- Compute activation entropy for COREY ---
    print("[ChunkSweep] Computing activation entropy for COREY chunk selection...")
    t_entropy_start = time.perf_counter()
    entropy_nats = _hist_entropy(u, num_bins=args.entropy_bins)
    entropy_overhead_ms = (time.perf_counter() - t_entropy_start) * 1000.0
    corey_chunk_size = _entropy_to_chunk_size(
        entropy_nats,
        min_chunk=args.corey_min_chunk,
        max_chunk=args.corey_max_chunk,
    )
    print(f"     entropy = {entropy_nats:.4f} nats -> COREY chunk_size = {corey_chunk_size}  "
          f"(overhead = {entropy_overhead_ms:.2f} ms)")

    # --- Static sweep ---
    sweep_results: list[dict[str, Any]] = []
    chunk_sizes = sorted(set(args.sweep_chunks + [corey_chunk_size]))

    for cs in chunk_sizes:
        is_corey = cs == corey_chunk_size
        label = f"static-{cs}" + (" [COREY selection]" if is_corey else "")
        print(f"[ChunkSweep] Benchmarking {label}...")
        result = _run_chunked_benchmark(
            u, delta, A, B, C, D,
            chunk_size=cs,
            warmup_runs=args.warmup_runs,
            benchmark_repeats=args.benchmark_repeats,
            policy_name="corey" if is_corey else "static",
            entropy_nats=entropy_nats if is_corey else None,
            entropy_overhead_ms=entropy_overhead_ms if is_corey else None,
        )
        sweep_results.append(result)
        print(f"     latency = {result['latency_mean_ms']:.3f} ± {result['latency_std_ms']:.3f} ms  "
              f"(n_chunks={result['n_chunks']}, {result['tokens_per_second']:.1f} tok/s)")

    # --- Oracle (best static chunk) ---
    static_only = [r for r in sweep_results if r["policy"] == "static"]
    oracle_result = min(static_only, key=lambda r: r["latency_mean_ms"]) if static_only else None
    oracle_chunk = oracle_result["chunk_size"] if oracle_result else None
    oracle_latency = oracle_result["latency_mean_ms"] if oracle_result else None

    corey_result = next((r for r in sweep_results if r["policy"] == "corey"), None)
    corey_latency = corey_result["latency_mean_ms"] if corey_result else None

    # Gap between COREY and oracle
    corey_vs_oracle_pct: float | None = None
    if corey_latency is not None and oracle_latency is not None and oracle_latency > 0:
        corey_vs_oracle_pct = round(100.0 * (corey_latency - oracle_latency) / oracle_latency, 2)

    print(f"\n[ChunkSweep] Oracle (best static): chunk_size={oracle_chunk}, "
          f"latency={oracle_latency:.3f} ms")
    print(f"[ChunkSweep] COREY entropy-selected: chunk_size={corey_chunk_size}, "
          f"latency={corey_latency:.3f} ms")
    if corey_vs_oracle_pct is not None:
        direction = "slower" if corey_vs_oracle_pct > 0 else "faster"
        print(f"[ChunkSweep] COREY vs oracle: {abs(corey_vs_oracle_pct):.1f}% {direction}")

    # --- Write outputs ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sweep CSV (one row per chunk_size point)
    sweep_fields = [
        "policy", "chunk_size", "n_chunks",
        "latency_mean_ms", "latency_std_ms", "latency_min_ms", "latency_p95_ms",
        "tokens_per_second", "kernel_launches",
        "estimated_hbm_gb", "estimated_hbm_bandwidth_gbps",
        "entropy_nats", "entropy_overhead_ms",
        "finite_ratio",
    ]
    with (output_dir / "chunk_sweep.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=sweep_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sweep_results)

    # Annotated comparison: mark oracle and corey
    annotated: list[dict[str, Any]] = []
    for r in sweep_results:
        row = {k: r.get(k) for k in sweep_fields}
        row["is_oracle"] = (r["chunk_size"] == oracle_chunk and r["policy"] == "static")
        row["is_corey"] = (r["policy"] == "corey")
        if oracle_latency and oracle_latency > 0:
            row["pct_above_oracle"] = round(
                100.0 * (r["latency_mean_ms"] - oracle_latency) / oracle_latency, 2
            )
        else:
            row["pct_above_oracle"] = None
        annotated.append(row)

    annotated_fields = sweep_fields + ["is_oracle", "is_corey", "pct_above_oracle"]
    with (output_dir / "chunk_sweep_annotated.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=annotated_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(annotated)

    summary = {
        "device_name": device_name,
        "cuda_version": torch.version.cuda,
        "batch_size": args.batch_size,
        "dim": args.dim,
        "seq_len": args.seq_len,
        "d_state": args.d_state,
        "dtype": args.dtype,
        "sweep_chunk_sizes": sorted(set(args.sweep_chunks)),
        "corey_chunk_size": corey_chunk_size,
        "corey_entropy_nats": round(entropy_nats, 6),
        "corey_entropy_overhead_ms": round(entropy_overhead_ms, 4),
        "oracle_chunk_size": oracle_chunk,
        "oracle_latency_ms": oracle_latency,
        "corey_latency_ms": corey_latency,
        "corey_vs_oracle_pct": corey_vs_oracle_pct,
        "warmup_runs": args.warmup_runs,
        "benchmark_repeats": args.benchmark_repeats,
        "hbm_traffic_note": (
            "estimated_hbm_* fields are analytic traffic proxies derived from "
            "tensor-volume movement, not Nsight DRAM counter measurements."
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sweep_results": sweep_results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n[ChunkSweep] Full latency curve (chunk_size -> latency_mean_ms):")
    for r in sweep_results:
        marker = ""
        if r["policy"] == "corey":
            marker = " <-- COREY (entropy-guided)"
        elif r["chunk_size"] == oracle_chunk:
            marker = " <-- oracle (fastest static)"
        print(f"  chunk={r['chunk_size']:>4}  "
              f"{r['latency_mean_ms']:>8.3f} ± {r['latency_std_ms']:.3f} ms  "
              f"({r['n_chunks']} kernel calls){marker}")

    print(f"\n[ChunkSweep] Outputs written to: {output_dir}")
    return {"output_dir": str(output_dir), "summary": summary}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "W1 chunk-size sweep: static-oracle latency curve + COREY entropy-selected point. "
            "Sweeps chunk_size in a configurable set, then measures COREY's entropy-selected "
            "chunk to show how it compares to the static oracle."
        )
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--benchmark-repeats", type=int, default=30)
    parser.add_argument(
        "--sweep-chunks",
        type=int,
        nargs="+",
        default=SWEEP_CHUNK_SIZES,
        help="List of chunk sizes to sweep for the static oracle curve.",
    )
    parser.add_argument("--corey-min-chunk", type=int, default=32)
    parser.add_argument("--corey-max-chunk", type=int, default=512)
    parser.add_argument("--entropy-bins", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/outputs/w1_chunk_sweep"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_sweep(args)
    print(json.dumps({"status": "ok", "output_dir": result["output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
