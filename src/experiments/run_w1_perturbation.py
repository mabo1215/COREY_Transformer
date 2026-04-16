"""
C2 Item 2 — Perturbation Experiment: Entropy-Adaptive Chunk Selection.

Sweeps five synthetic activation distributions with deliberately different
entropy levels.  For each distribution we:
  1. Generate activations of the same shape (batch=1, dim=1024, seq_len=4096).
  2. Compute histogram entropy.
  3. COREY maps entropy -> chunk_size.
  4. Benchmark policy_corey (entropy-selected chunk) vs policy_static (chunk=64).

Key claim under test (C2 reviewer requirement):
  "chunk recommendation changes with entropy; COREY improves over fixed oracle."

Expected behavior:
  - High-entropy (uniform/normal) activations -> large chunks -> faster than static-64.
  - Low-entropy (sparse/constant) activations -> small chunks -> conservative,
    comparable to or slower than static-64 on latency but avoids large-chunk
    numerical errors in real deployments.
  - The monotone entropy->chunk mapping is verified by printing the selected
    chunk sizes across the distribution sweep.

Distributions:
  uniform     : torch.rand()       -- maximum entropy for bounded support
  normal      : torch.randn()      -- standard normal, high entropy
  laplace     : Laplace(0,1)       -- heavier tails, medium entropy
  sparse10    : 10% non-zero normal entries  -- low entropy
  sparse2     : 2% non-zero large values    -- very low entropy

Usage (WSL2 with mamba_ssm available):
  python -m src.experiments.run_w1_perturbation \\
      --seq-len 4096 --dim 1024 --d-state 16 \\
      --warmup-runs 5 --benchmark-repeats 30 \\
      --output-dir src/outputs/w1_perturbation
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
# Entropy and chunk helpers (standalone copies)
# ---------------------------------------------------------------------------

def _hist_entropy(values: Any, num_bins: int = 256) -> float:
    """Shannon entropy of a float tensor via fixed-width histogram (nats)."""
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
    """Map activation entropy -> chunk_size (same mapping as COREY policy)."""
    ratio = min(entropy_nats / ref_max_entropy, 1.0)
    chunk = min_chunk + ratio * (max_chunk - min_chunk)
    log2 = math.log2(max(chunk, 1.0))
    rounded = 2 ** round(log2)
    return max(min_chunk, min(max_chunk, int(rounded)))


# ---------------------------------------------------------------------------
# Activation generators
# ---------------------------------------------------------------------------

DISTRIBUTIONS: list[tuple[str, str]] = [
    ("uniform",  "Uniform[0,1] — maximum entropy for bounded support"),
    ("normal",   "Standard Normal — high entropy, typical random activations"),
    ("laplace",  "Laplace(0,1) — heavy tails, medium entropy"),
    ("sparse10", "Sparse 10% — 10% non-zero normal entries, low entropy"),
    ("sparse2",  "Sparse 2%  — 2% non-zero large entries, very low entropy"),
]


def _make_activations(
    dist_name: str,
    batch: int,
    dim: int,
    seq_len: int,
    dtype: Any,
    device: Any,
    rng: Any,
) -> Any:
    """Generate a synthetic activation tensor for the given distribution."""
    import torch

    shape = (batch, dim, seq_len)
    if dist_name == "uniform":
        u = torch.rand(shape, device=device, dtype=dtype, generator=rng)
    elif dist_name == "normal":
        u = torch.randn(shape, device=device, dtype=dtype, generator=rng)
    elif dist_name == "laplace":
        # Laplace via inverse CDF from uniform
        uniform = torch.rand(shape, device=device, dtype=torch.float32, generator=rng)
        u = -torch.sign(uniform - 0.5) * torch.log(1.0 - 2.0 * (uniform - 0.5).abs() + 1e-8)
        u = u.to(dtype)
    elif dist_name == "sparse10":
        mask = (torch.rand(shape, device=device, generator=rng) < 0.10).to(dtype)
        vals = torch.randn(shape, device=device, dtype=dtype, generator=rng)
        u = mask * vals
    elif dist_name == "sparse2":
        mask = (torch.rand(shape, device=device, generator=rng) < 0.02).to(dtype)
        vals = torch.randn(shape, device=device, dtype=dtype, generator=rng) * 5.0
        u = mask * vals
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")
    return u


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------

def _tensor_nbytes(tensor: Any) -> int:
    return int(tensor.numel() * tensor.element_size())


def _run_chunked(
    u: Any,
    delta: Any,
    A: Any,
    B: Any,
    C: Any,
    D: Any,
    *,
    chunk_size: int,
    warmup_runs: int,
    benchmark_repeats: int,
    policy_name: str,
) -> dict[str, Any]:
    import torch
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    seq_len = u.shape[2]
    starts = list(range(0, seq_len, chunk_size))
    n_chunks = len(starts)

    def _one_pass() -> Any:
        outs = []
        for start in starts:
            end = min(start + chunk_size, seq_len)
            outs.append(selective_scan_fn(
                u[:, :, start:end], delta[:, :, start:end],
                A, B, C, D=D, delta_softplus=True,
            ))
        return torch.cat(outs, dim=2)

    for _ in range(warmup_runs):
        _ = _one_pass()
    torch.cuda.synchronize()

    latencies: list[float] = []
    for _ in range(benchmark_repeats):
        t0 = time.perf_counter()
        _one_pass()
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "policy": policy_name,
        "chunk_size": chunk_size,
        "n_chunks": n_chunks,
        "latency_mean_ms": round(mean(latencies), 4),
        "latency_std_ms": round(pstdev(latencies), 4) if len(latencies) > 1 else 0.0,
        "latency_min_ms": round(min(latencies), 4),
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 4),
        "kernel_launches": n_chunks,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_perturbation_sweep(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch required.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required. Run in WSL2 with GPU access.")

    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401
    except ImportError as exc:
        raise ImportError("mamba_ssm required.") from exc

    device = torch.device("cuda")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    device_name = torch.cuda.get_device_name(device)
    print(f"[Perturb] Device: {device_name}")
    print(f"[Perturb] Config: batch={args.batch_size}, dim={args.dim}, "
          f"seq_len={args.seq_len}, d_state={args.d_state}, dtype={args.dtype}")

    # Fixed scan parameters (same across all distributions for fair comparison)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)
    A = torch.randn(args.dim, args.d_state, device=device, dtype=torch.float32)
    B = torch.randn(args.dim, args.d_state, device=device, dtype=torch.float32)
    C = torch.randn(args.dim, args.d_state, device=device, dtype=torch.float32)
    D = torch.randn(args.dim, device=device, dtype=torch.float32)
    delta_template = torch.rand(args.batch_size, args.dim, args.seq_len,
                                device=device, dtype=dtype, generator=rng)

    results: list[dict[str, Any]] = []
    static64_latencies: dict[str, float] = {}

    for dist_name, dist_desc in DISTRIBUTIONS:
        print(f"\n[Perturb] Distribution: {dist_name}")
        # Re-seed for each distribution to ensure reproducible activations
        act_rng = torch.Generator(device=device)
        act_rng.manual_seed(args.seed + hash(dist_name) % 10000)

        u = _make_activations(dist_name, args.batch_size, args.dim, args.seq_len,
                               dtype, device, act_rng)

        # Measure entropy
        t0 = time.perf_counter()
        entropy = _hist_entropy(u, num_bins=args.entropy_bins)
        entropy_overhead_ms = (time.perf_counter() - t0) * 1000.0
        corey_chunk = _entropy_to_chunk_size(entropy, args.corey_min_chunk, args.corey_max_chunk)
        print(f"  entropy = {entropy:.4f} nats -> COREY chunk = {corey_chunk}  "
              f"(entropy overhead = {entropy_overhead_ms:.2f} ms)")

        # Benchmark static-64
        r_static = _run_chunked(
            u, delta_template, A, B, C, D,
            chunk_size=64,
            warmup_runs=args.warmup_runs,
            benchmark_repeats=args.benchmark_repeats,
            policy_name="static-64",
        )
        static64_latencies[dist_name] = r_static["latency_mean_ms"]
        print(f"  static-64:   {r_static['latency_mean_ms']:.3f} ± {r_static['latency_std_ms']:.3f} ms  "
              f"({r_static['n_chunks']} kernel calls)")

        # Benchmark COREY chunk
        r_corey = _run_chunked(
            u, delta_template, A, B, C, D,
            chunk_size=corey_chunk,
            warmup_runs=args.warmup_runs,
            benchmark_repeats=args.benchmark_repeats,
            policy_name="corey",
        )
        speedup = r_static["latency_mean_ms"] / r_corey["latency_mean_ms"] if r_corey["latency_mean_ms"] > 0 else None
        print(f"  corey-{corey_chunk:<4}: {r_corey['latency_mean_ms']:.3f} ± {r_corey['latency_std_ms']:.3f} ms  "
              f"({r_corey['n_chunks']} kernel calls)  "
              f"speedup vs static-64: {speedup:.2f}x" if speedup else "  speedup: N/A")

        row = {
            "distribution": dist_name,
            "description": dist_desc,
            "entropy_nats": round(entropy, 4),
            "entropy_overhead_ms": round(entropy_overhead_ms, 4),
            "corey_chunk_size": corey_chunk,
            "static64_latency_ms": r_static["latency_mean_ms"],
            "static64_latency_std": r_static["latency_std_ms"],
            "corey_latency_ms": r_corey["latency_mean_ms"],
            "corey_latency_std": r_corey["latency_std_ms"],
            "corey_n_chunks": r_corey["n_chunks"],
            "speedup_vs_static64": round(speedup, 3) if speedup else None,
        }
        results.append(row)

    # Summary table
    print(f"\n[Perturb] Summary (Device: {device_name}, seq_len={args.seq_len}, dim={args.dim}):")
    print(f"  {'Dist':<12} {'Entropy':>8} {'COREY chunk':>12} {'COREY lat':>12} {'static-64 lat':>14} {'Speedup':>9}")
    print(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*12} {'-'*14} {'-'*9}")
    for r in results:
        sp = f"{r['speedup_vs_static64']:.2f}x" if r['speedup_vs_static64'] else "N/A"
        print(f"  {r['distribution']:<12} {r['entropy_nats']:>8.3f} {r['corey_chunk_size']:>12} "
              f"{r['corey_latency_ms']:>10.3f} ms {r['static64_latency_ms']:>12.3f} ms {sp:>9}")

    # Write outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fields = list(results[0].keys())
    with (output_dir / "perturbation_sweep.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "device_name": device_name,
        "cuda_version": torch.version.cuda,
        "batch_size": args.batch_size,
        "dim": args.dim,
        "seq_len": args.seq_len,
        "d_state": args.d_state,
        "dtype": args.dtype,
        "warmup_runs": args.warmup_runs,
        "benchmark_repeats": args.benchmark_repeats,
        "corey_min_chunk": args.corey_min_chunk,
        "corey_max_chunk": args.corey_max_chunk,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n[Perturb] Outputs written to: {output_dir}")
    return {"output_dir": str(output_dir), "summary": summary}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "C2 perturbation experiment: sweep input distributions to verify "
            "COREY chunk selection tracks entropy."
        )
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--benchmark-repeats", type=int, default=30)
    parser.add_argument("--corey-min-chunk", type=int, default=32)
    parser.add_argument("--corey-max-chunk", type=int, default=512)
    parser.add_argument("--entropy-bins", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("src/outputs/w1_perturbation"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_perturbation_sweep(args)
    print(json.dumps({"status": "ok", "output_dir": result["output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
