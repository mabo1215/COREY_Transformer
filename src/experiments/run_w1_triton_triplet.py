"""
W1 Three-Policy Real-GPU Benchmark: No-Fusion / Static-Fusion / COREY.

This benchmark provides a genuine GPU-timing comparison between three scheduling
policies applied to the selective-scan core of a Mamba SSM layer:

  policy_off    : Pure Python timestep loop (reference, fully unfused).
                  Models the worst-case kernel-fragmentation scenario where
                  each timestep incurs separate tensor dispatch overhead.

  policy_static : selective_scan_fn called on fixed-size sequence chunks
                  (default chunk_size=64).  Mimics a static fusion backend
                  that partitions the sequence without activation analysis.

  policy_corey  : selective_scan_fn called on entropy-guided chunks.
                  COREY measures activation entropy of the input tensor and
                  maps it to an optimal chunk_size (larger chunks for more
                  uniform/higher-entropy activations).

Genuine GPU-level differences arise from:
  - Kernel launch count  (off: O(L) Python calls; static: L/C_static calls;
                          corey: L/C_entropy calls, typically fewer than static)
  - Memory coalescing    (larger chunks → more contiguous HBM access patterns)
  - Overhead amortization (fixed per-kernel overhead amortized over more tokens)

Usage (WSL2 / Linux with CUDA):
  python -m src.experiments.run_w1_triton_triplet \
      --seq-len 4096 --dim 1024 --d-state 16 \
      --warmup-runs 5 --benchmark-repeats 30 \
      --output-dir src/outputs/w1_triton_triplet
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


def _tensor_nbytes(tensor: Any) -> int:
    return int(tensor.numel() * tensor.element_size())


def _estimate_policy_hbm_bytes(
    *,
    policy_name: str,
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
    """
    Estimate HBM traffic with a simple tensor-volume model.

    The estimate treats each policy as moving:
      - input activations `u` and `delta`
      - recurrent state read + write per execution unit
      - model parameters `A/B/C/D` once per execution unit
      - output activations once

    `policy_off` uses one execution unit per timestep. Chunked policies use one
    execution unit per chunk. These values are intended as a comparative memory
    traffic proxy when Nsight DRAM counters are unavailable.
    """
    per_timestep_input = batch_size * dim * input_bytes
    per_timestep_output = batch_size * dim * input_bytes
    if policy_name == "off":
        estimated_bytes = seq_len * (
            2 * per_timestep_input +
            2 * state_bytes +
            param_bytes +
            per_timestep_output
        )
    else:
        estimated_bytes = n_chunks * (
            2 * batch_size * dim * chunk_size * input_bytes +
            2 * state_bytes +
            param_bytes
        ) + batch_size * dim * seq_len * input_bytes

    estimated_gb = estimated_bytes / 1e9
    estimated_gib = estimated_bytes / float(1 << 30)
    return {
        "estimated_hbm_bytes": float(estimated_bytes),
        "estimated_hbm_gb": round(estimated_gb, 6),
        "estimated_hbm_gib": round(estimated_gib, 6),
    }

# ---------------------------------------------------------------------------
# Entropy helpers (replicated inline to avoid src.algorithms import overhead
# when this script is used as a standalone GPU benchmark in WSL2)
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
    """
    Map activation entropy (nats) to chunk_size for COREY policy.

    Higher entropy -> larger chunks (activation mass is more uniformly spread ->
    numerically safer to fuse more timesteps per kernel call).

    The mapping is linear between min_chunk (entropy=0) and max_chunk
    (entropy >= ref_max_entropy).  Result is rounded to the nearest power of 2.
    """
    ratio = min(entropy_nats / ref_max_entropy, 1.0)
    chunk = min_chunk + ratio * (max_chunk - min_chunk)
    # Round to nearest power of two for alignment-friendly tile shapes.
    log2 = math.log2(max(chunk, 1.0))
    rounded = 2 ** round(log2)
    return max(min_chunk, min(max_chunk, int(rounded)))


# ---------------------------------------------------------------------------
# Policy implementations
# ---------------------------------------------------------------------------


def _run_policy_off(
    u: Any,
    delta: Any,
    A: Any,
    B: Any,
    C: Any,
    D: Any,
    *,
    warmup_runs: int = 2,
    benchmark_repeats: int = 10,
) -> dict[str, Any]:
    """
    No-fusion reference: pure Python timestep loop.

    Each of the L timesteps executes separate tensor operations without
    calling selective_scan_fn.  This replicates the SSMBlock._simple_scan
    pattern from src/algorithms/torch_fused_ops.py.
    """
    import torch

    device = u.device
    batch_size, d_inner, seq_len = u.shape
    d_state = A.shape[-1]
    input_bytes = u.element_size()
    state_bytes = batch_size * d_inner * d_state * input_bytes
    param_bytes = sum(_tensor_nbytes(tensor) for tensor in (A, B, C, D))

    def _one_pass() -> Any:
        state = u.new_zeros(batch_size, d_inner, d_state)
        outs: list[Any] = []
        for t in range(seq_len):
            # u_t: [B, D], delta_t: [B, D]
            u_t = u[:, :, t]
            delta_t = delta[:, :, t]
            # Discrete-time A: delta_A = exp(delta_t * A)   [B, D, N]
            delta_A = torch.exp(
                delta_t.unsqueeze(-1) * A.unsqueeze(0)
            )  # [B, D, N]
            # Discrete-time B contribution: delta_B_u = delta_t * B * u_t  [B, D, N]
            delta_B_u = (
                delta_t.unsqueeze(-1)
                * B.unsqueeze(0)
                * u_t.unsqueeze(-1)
            )  # [B, D, N]
            state = delta_A * state + delta_B_u
            # Output: y_t = sum_{n}(state * C)   [B, D]
            outs.append((state * C.unsqueeze(0)).sum(-1))
        output = torch.stack(outs, dim=2)  # [B, D, L]
        if D is not None:
            output = output + D.unsqueeze(0).unsqueeze(-1) * u
        return output

    # Warmup
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
        policy_name="off",
        batch_size=batch_size,
        dim=d_inner,
        seq_len=seq_len,
        d_state=d_state,
        input_bytes=input_bytes,
        state_bytes=state_bytes,
        param_bytes=param_bytes,
        chunk_size=1,
        n_chunks=seq_len,
    )
    tokens_per_second = (batch_size * seq_len) / (mean(latencies) / 1000.0)
    estimated_bandwidth_gbps = traffic["estimated_hbm_gb"] / (mean(latencies) / 1000.0)

    return {
        "policy": "off",
        "chunk_size": 1,
        "n_chunks": seq_len,
        "entropy_nats": None,
        "entropy_overhead_ms": None,
        "latency_mean_ms": round(mean(latencies), 4),
        "latency_std_ms": round(pstdev(latencies), 4) if len(latencies) > 1 else 0.0,
        "latency_min_ms": round(min(latencies), 4),
        "latency_max_ms": round(max(latencies), 4),
        "latency_p50_ms": round(sorted(latencies)[len(latencies) // 2], 4),
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 4),
        "output_checksum": round(output_mean, 8) if math.isfinite(output_mean) else None,
        "finite_ratio": round(float(out.isfinite().float().mean().item()), 6),
        "kernel_launches": seq_len,
        "tokens_per_second": round(tokens_per_second, 4),
        "estimated_hbm_bytes": int(traffic["estimated_hbm_bytes"]),
        "estimated_hbm_gb": traffic["estimated_hbm_gb"],
        "estimated_hbm_gib": traffic["estimated_hbm_gib"],
        "estimated_hbm_bandwidth_gbps": round(estimated_bandwidth_gbps, 4),
    }


def _run_policy_chunked(
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
    Chunked fusion policy: call selective_scan_fn on fixed-size sequence chunks.

    For policy_static, chunk_size is fixed (passed in).
    For policy_corey,  chunk_size is pre-computed from entropy (also passed in).
    """
    import torch
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    seq_len = u.shape[2]
    batch_size, d_inner, _ = u.shape
    d_state = A.shape[-1]
    input_bytes = u.element_size()
    state_bytes = batch_size * d_inner * d_state * input_bytes
    param_bytes = sum(_tensor_nbytes(tensor) for tensor in (A, B, C, D))
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

    # Warmup
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
        policy_name=policy_name,
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
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for the W1 benchmark.") from exc

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

    import torch as _torch
    rng = _torch.Generator(device=device)
    rng.manual_seed(args.seed)

    # Synthetic activations matching the existing triton_selective_scan benchmark shape.
    u = torch.randn(args.batch_size, args.dim, args.seq_len, device=device, dtype=dtype, generator=rng)
    delta = torch.rand(args.batch_size, args.dim, args.seq_len, device=device, dtype=dtype, generator=rng)
    A = torch.randn(args.dim, args.d_state, device=device, dtype=torch.float32)
    B = torch.randn(args.dim, args.d_state, device=device, dtype=torch.float32)
    C = torch.randn(args.dim, args.d_state, device=device, dtype=torch.float32)
    D = torch.randn(args.dim, device=device, dtype=torch.float32)

    results: list[dict[str, Any]] = []
    selected_policies = {"off", "static", "corey"}
    if args.policy != "all":
        selected_policies = {args.policy}
    off_result: dict[str, Any] | None = None
    static_result: dict[str, Any] | None = None
    corey_result: dict[str, Any] | None = None
    entropy_nats: float | None = None
    entropy_overhead_ms: float | None = None
    corey_chunk_size: int | None = None

    # --- policy_off ---
    if "off" in selected_policies:
        print(f"[W1] Running policy_off (Python loop, seq_len={args.seq_len})...")
        off_result = _run_policy_off(
            u, delta, A, B, C, D,
            warmup_runs=min(args.warmup_runs, 2),
            benchmark_repeats=min(args.benchmark_repeats, 10),
        )
        results.append(off_result)
        print(f"     latency = {off_result['latency_mean_ms']:.1f} ± {off_result['latency_std_ms']:.1f} ms")

    # --- policy_static ---
    if "static" in selected_policies:
        print(f"[W1] Running policy_static (chunk_size={args.static_chunk_size})...")
        static_result = _run_policy_chunked(
            u, delta, A, B, C, D,
            chunk_size=args.static_chunk_size,
            warmup_runs=args.warmup_runs,
            benchmark_repeats=args.benchmark_repeats,
            policy_name="static",
        )
        results.append(static_result)
        print(f"     latency = {static_result['latency_mean_ms']:.1f} ± {static_result['latency_std_ms']:.1f} ms  "
              f"(chunk_size={static_result['chunk_size']}, n_chunks={static_result['n_chunks']})")

    # --- policy_corey ---
    if "corey" in selected_policies:
        print("[W1] Computing activation entropy for COREY policy...")
        t_entropy_start = time.perf_counter()
        entropy_nats = _hist_entropy(u, num_bins=args.entropy_bins)
        entropy_overhead_ms = (time.perf_counter() - t_entropy_start) * 1000.0
        corey_chunk_size = _entropy_to_chunk_size(
            entropy_nats,
            min_chunk=args.corey_min_chunk,
            max_chunk=args.corey_max_chunk,
        )
        print(f"     entropy = {entropy_nats:.4f} nats -> chunk_size = {corey_chunk_size}  "
              f"(entropy_overhead = {entropy_overhead_ms:.2f} ms)")
        print(f"[W1] Running policy_corey (chunk_size={corey_chunk_size})...")
        corey_result = _run_policy_chunked(
            u, delta, A, B, C, D,
            chunk_size=corey_chunk_size,
            warmup_runs=args.warmup_runs,
            benchmark_repeats=args.benchmark_repeats,
            policy_name="corey",
            entropy_nats=entropy_nats,
            entropy_overhead_ms=entropy_overhead_ms,
        )
        results.append(corey_result)
        print(f"     latency = {corey_result['latency_mean_ms']:.1f} ± {corey_result['latency_std_ms']:.1f} ms  "
              f"(chunk_size={corey_result['chunk_size']}, n_chunks={corey_result['n_chunks']})")

    # --- Build comparison table ---
    import torch
    device_name = torch.cuda.get_device_name(device)
    reference_off_mean = off_result["latency_mean_ms"] if off_result is not None else None
    reference_static_mean = static_result["latency_mean_ms"] if static_result is not None else None

    comparison: list[dict[str, Any]] = []
    for r in results:
        speedup_vs_off = None
        speedup_vs_static = None
        if reference_off_mean is not None and r["latency_mean_ms"] > 0:
            speedup_vs_off = round(reference_off_mean / r["latency_mean_ms"], 3)
        if reference_static_mean is not None and r["latency_mean_ms"] > 0:
            speedup_vs_static = round(reference_static_mean / r["latency_mean_ms"], 3)
        comparison.append({
            "policy": r["policy"],
            "chunk_size": r["chunk_size"],
            "n_chunks": r["n_chunks"],
            "latency_mean_ms": r["latency_mean_ms"],
            "latency_std_ms": r["latency_std_ms"],
            "latency_min_ms": r["latency_min_ms"],
            "latency_p95_ms": r["latency_p95_ms"],
            "tokens_per_second": r["tokens_per_second"],
            "estimated_hbm_gb": r["estimated_hbm_gb"],
            "estimated_hbm_bandwidth_gbps": r["estimated_hbm_bandwidth_gbps"],
            "speedup_vs_off": speedup_vs_off,
            "speedup_vs_static": speedup_vs_static,
            "entropy_nats": r.get("entropy_nats"),
            "entropy_overhead_ms": r.get("entropy_overhead_ms"),
        })

    summary = {
        "device_name": device_name,
        "cuda_version": torch.version.cuda,
        "batch_size": args.batch_size,
        "dim": args.dim,
        "seq_len": args.seq_len,
        "d_state": args.d_state,
        "dtype": args.dtype,
        "selected_policy": args.policy,
        "static_chunk_size": args.static_chunk_size,
        "corey_chunk_size": corey_chunk_size,
        "corey_entropy_nats": round(entropy_nats, 6) if entropy_nats is not None else None,
        "corey_min_chunk": args.corey_min_chunk,
        "corey_max_chunk": args.corey_max_chunk,
        "warmup_runs": args.warmup_runs,
        "benchmark_repeats": args.benchmark_repeats,
        "hbm_traffic_note": (
            "estimated_hbm_* fields are analytic traffic proxies derived from tensor-volume "
            "movement, not Nsight DRAM counter measurements."
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "policies": comparison,
    }

    # --- Write outputs ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-repeat CSV
    per_repeat_rows: list[dict[str, Any]] = []
    for r in results:
        for i in range(len(r.get("latencies", []))):
            per_repeat_rows.append({
                "policy": r["policy"],
                "repeat_idx": i,
                "chunk_size": r["chunk_size"],
                "n_chunks": r["n_chunks"],
            })

    # Comparison CSV
    if comparison:
        fieldnames = list(comparison[0].keys())
        with (output_dir / "comparison_table.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison)

    # Full detail CSV (one row per policy with all metrics)
    all_fields: list[str] = []
    for r in results:
        for k in r:
            if k not in all_fields:
                all_fields.append(k)
    with (output_dir / "policy_details.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(results)

    # Summary JSON
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n[W1] Comparison summary (seq_len={args.seq_len}, dim={args.dim}, {device_name}):")
    for row in comparison:
        summary_line = (
            f"  policy_{row['policy']:<8}: {row['latency_mean_ms']:.1f} ms  "
            f"(chunk={row['chunk_size']}, {row['n_chunks']} execution units)  "
            f"{row['tokens_per_second']:.1f} tok/s, "
            f"{row['estimated_hbm_gb']:.3f} GB"
        )
        if row["speedup_vs_off"] is not None and row["policy"] != "off":
            summary_line += f", {row['speedup_vs_off']:.2f}x vs off"
        if row["speedup_vs_static"] is not None and row["policy"] not in {"static", "off"}:
            summary_line += f", {row['speedup_vs_static']:.2f}x vs static"
        print(summary_line)
    print(f"[W1] Outputs written to: {output_dir}")

    return {"output_dir": str(output_dir), "summary": summary}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="W1 three-policy real-GPU chunked selective-scan benchmark."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--benchmark-repeats", type=int, default=30)
    parser.add_argument("--static-chunk-size", type=int, default=64,
                        help="Chunk size for policy_static.")
    parser.add_argument("--corey-min-chunk", type=int, default=32)
    parser.add_argument("--corey-max-chunk", type=int, default=512)
    parser.add_argument("--entropy-bins", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policy",
        default="all",
        choices=["all", "off", "static", "corey"],
        help="Run all policies or a single policy for targeted profiling.",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path("src/outputs/w1_triton_triplet"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_benchmark(args)
    print(json.dumps({"status": "ok", "output_dir": result["output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
