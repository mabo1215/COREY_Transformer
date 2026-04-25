"""
8-GPU distributed three-policy selective-scan benchmark for COREY.

Runs the W1 triplet benchmark (Static-64 / COREY-default / Static-512-oracle)
independently on each of 8 CUDA GPUs via torchrun and aggregates latency
statistics at rank 0.  The goal is to confirm that COREY's speedup is
consistent across all GPU instances in a multi-GPU server.

Design:
  - Each rank runs the full benchmark on its own GPU (no data or model
    parallelism; ranks are independent replications).
  - Entropy is computed locally on each rank's synthetic activations
    (identical due to shared seed + per-rank offset).
  - Rank 0 gathers per-rank results and saves aggregated statistics.

Launch:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \\
    src/experiments/run_corey_8gpu_benchmark.py \\
    --model state-spaces/mamba-370m-hf \\
    --chunk-size 512 --seq-len 4096 --repeat 30 \\
    --output-dir src/outputs/corey_8gpu_benchmark
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="COREY 8-GPU distributed benchmark.")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--model", type=str, default="state-spaces/mamba-370m-hf",
                    help="Model label (used for metadata only; benchmark uses fixed tensor dims).")
parser.add_argument("--chunk-size", type=int, default=512)
parser.add_argument("--seq-len", type=int, default=4096)
parser.add_argument("--dim", type=int, default=1024)
parser.add_argument("--d-state", type=int, default=16)
parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
parser.add_argument("--repeat", type=int, default=30)
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--num-bins", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output-dir", type=Path, default=Path("src/outputs/corey_8gpu_benchmark"))
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Selective-scan backend
# ---------------------------------------------------------------------------

_MAMBA_SSM_AVAILABLE = False
_selective_scan_fn_cuda: Any = None


def _try_import_mamba_ssm() -> None:
    global _MAMBA_SSM_AVAILABLE, _selective_scan_fn_cuda
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        _selective_scan_fn_cuda = selective_scan_fn
        _MAMBA_SSM_AVAILABLE = True
    except ImportError:
        pass


def _pytorch_selective_scan(u: Any, delta: Any, A: Any, B: Any, C: Any,
                             D: Any | None = None, **kwargs: Any) -> Any:
    """Pure-PyTorch latency-proxy scan (used when mamba_ssm is unavailable)."""
    weighted = u * delta.sigmoid()
    state_proxy = weighted.mean(dim=-1, keepdim=True).expand_as(weighted)
    out = state_proxy
    if D is not None:
        out = out + D.float().unsqueeze(0).unsqueeze(-1) * u.float()
    return out.to(u.dtype)


def _scan_fn(u: Any, delta: Any, A: Any, B: Any, C: Any,
             D: Any | None = None, **kwargs: Any) -> Any:
    if _MAMBA_SSM_AVAILABLE:
        return _selective_scan_fn_cuda(u, delta, A, B, C, D, delta_softplus=True)
    return _pytorch_selective_scan(u, delta, A, B, C, D, **kwargs)


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def _hist_entropy(values: Any, num_bins: int = 256) -> float:
    flat = values.detach().float().reshape(-1)
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


def _entropy_to_chunk(H: float, num_bins: int = 256,
                      c_min: int = 32, c_max: int = 512) -> int:
    h_ref = math.log(num_bins)
    ratio = min(H / h_ref, 1.0) if h_ref > 0 else 0.0
    raw = c_min + ratio * (c_max - c_min)
    rounded = int(2 ** round(math.log2(max(raw, 1.0))))
    return max(c_min, min(c_max, rounded))


# ---------------------------------------------------------------------------
# Per-policy benchmark
# ---------------------------------------------------------------------------

def _benchmark_policy(
    u: Any, delta: Any, A: Any, B: Any, C: Any, D: Any,
    *,
    chunk_size: int,
    warmup: int,
    repeat: int,
    policy_name: str,
    entropy_nats: float | None = None,
) -> dict[str, Any]:
    seq_len = u.shape[2]
    starts = list(range(0, seq_len, chunk_size))
    n_chunks = len(starts)

    def _one_pass() -> None:
        for start in starts:
            end = min(start + chunk_size, seq_len)
            _scan_fn(u[:, :, start:end], delta[:, :, start:end], A, B, C, D)

    for _ in range(warmup):
        _one_pass()
    torch.cuda.synchronize()

    latencies: list[float] = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _one_pass()
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "policy": policy_name,
        "chunk_size": chunk_size,
        "n_chunks": n_chunks,
        "entropy_nats": round(entropy_nats, 6) if entropy_nats is not None else None,
        "latency_mean_ms": round(statistics.mean(latencies), 6),
        "latency_std_ms": round(statistics.pstdev(latencies), 6) if len(latencies) > 1 else 0.0,
        "latency_min_ms": round(min(latencies), 6),
        "latency_max_ms": round(max(latencies), 6),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    _try_import_mamba_ssm()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Each rank uses a distinct seed offset so results represent independent
    # replications rather than identical runs.
    torch.manual_seed(args.seed + rank)

    batch, dim, d_state, seq_len = 1, args.dim, args.d_state, args.seq_len
    u = torch.randn(batch, dim, seq_len, device=device, dtype=dtype)
    delta = torch.rand(batch, dim, seq_len, device=device, dtype=dtype)
    A = torch.randn(dim, d_state, device=device, dtype=torch.float32)
    B = torch.randn(dim, d_state, device=device, dtype=torch.float32)
    C = torch.randn(dim, d_state, device=device, dtype=torch.float32)
    D_param = torch.randn(dim, device=device, dtype=torch.float32)

    # Entropy computation for COREY
    t_e0 = time.perf_counter()
    entropy_nats = _hist_entropy(u, num_bins=args.num_bins)
    entropy_overhead_ms = (time.perf_counter() - t_e0) * 1000.0
    corey_chunk = _entropy_to_chunk(entropy_nats, num_bins=args.num_bins)
    h_ref = math.log(args.num_bins)

    if rank == 0:
        backend = "mamba_ssm" if _MAMBA_SSM_AVAILABLE else "pytorch_sim"
        print(f"[8gpu_bench] world_size={world_size}  backend={backend}  "
              f"seq_len={seq_len}  dim={dim}  repeat={args.repeat}")

    dist.barrier()

    # Run three policies on this rank's GPU
    policies = [
        ("static_64",  64,           None),
        ("corey",      corey_chunk,  entropy_nats),
        ("static_512", 512,          None),
    ]

    rank_results: list[dict[str, Any]] = []
    for pname, csize, ent in policies:
        r = _benchmark_policy(
            u, delta, A, B, C, D_param,
            chunk_size=csize,
            warmup=args.warmup,
            repeat=args.repeat,
            policy_name=pname,
            entropy_nats=ent,
        )
        rank_results.append(r)

    # Compute speedups relative to static_64 (rank-local)
    ref_lat = next(r["latency_mean_ms"] for r in rank_results if r["policy"] == "static_64")
    oracle_lat = next(r["latency_mean_ms"] for r in rank_results if r["policy"] == "static_512")
    for r in rank_results:
        r["speedup_A"] = round(ref_lat / r["latency_mean_ms"], 4) if r["latency_mean_ms"] > 0 else None
        r["speedup_B"] = round(oracle_lat / r["latency_mean_ms"], 4) if r["latency_mean_ms"] > 0 else None
        r["rank"] = rank
        r["gpu"] = torch.cuda.get_device_name(rank)
        r["entropy_overhead_ms"] = round(entropy_overhead_ms, 6)
        r["h_ref_nats"] = round(h_ref, 6)

    per_rank_record = {
        "rank": rank,
        "gpu": torch.cuda.get_device_name(rank),
        "entropy_nats": round(entropy_nats, 6),
        "corey_chunk": corey_chunk,
        "policies": rank_results,
    }

    # Gather at rank 0
    gathered: list[Any] = [None] * world_size
    dist.gather_object(per_rank_record, gathered if rank == 0 else None, dst=0)

    dist.barrier()

    if rank == 0:
        # Aggregate mean latency per policy across all ranks
        agg: dict[str, list[float]] = {}
        for rec in gathered:
            for pr in rec["policies"]:
                agg.setdefault(pr["policy"], []).append(pr["latency_mean_ms"])

        agg_summary: list[dict[str, Any]] = []
        ref_agg = statistics.mean(agg["static_64"])
        oracle_agg = statistics.mean(agg["static_512"])
        for pname in ["static_64", "corey", "static_512"]:
            lats = agg[pname]
            agg_summary.append({
                "policy": pname,
                "latency_mean_ms_across_ranks": round(statistics.mean(lats), 6),
                "latency_std_ms_across_ranks": round(statistics.pstdev(lats), 6) if len(lats) > 1 else 0.0,
                "speedup_A_aggregate": round(ref_agg / statistics.mean(lats), 4) if statistics.mean(lats) > 0 else None,
                "speedup_B_aggregate": round(oracle_agg / statistics.mean(lats), 4) if statistics.mean(lats) > 0 else None,
            })

        print("\n[8gpu_bench] === Aggregate speedup (mean across all ranks) ===")
        for s in agg_summary:
            print(f"  {s['policy']:<12}  "
                  f"lat={s['latency_mean_ms_across_ranks']:.4f}±{s['latency_std_ms_across_ranks']:.4f} ms  "
                  f"Speedup-A={s['speedup_A_aggregate']:.2f}x  "
                  f"Speedup-B={s['speedup_B_aggregate']:.2f}x")

        output = {
            "world_size": world_size,
            "backend": "mamba_ssm" if _MAMBA_SSM_AVAILABLE else "pytorch_sim",
            "model_label": args.model,
            "seq_len": seq_len,
            "dim": dim,
            "d_state": d_state,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "num_bins": args.num_bins,
            "platform": platform.platform(),
            "aggregate": agg_summary,
            "per_rank": gathered,
        }

        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / "summary.json"
        out_path.write_text(json.dumps(output, indent=2))
        print(f"[8gpu_bench] Results saved to {out_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
