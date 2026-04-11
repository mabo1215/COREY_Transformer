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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the real selective-scan kernel exposed by mamba_ssm in a CUDA/Triton-enabled environment."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--benchmark-repeats", type=int, default=30)
    parser.add_argument("--delta-softplus", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("src/outputs/triton_selective_scan_wsl"))
    return parser.parse_args()


def _dtype_from_name(torch_module: Any, dtype_name: str) -> Any:
    return getattr(torch_module, dtype_name)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except ImportError as exc:
        raise ImportError(
            "This benchmark requires torch and mamba_ssm with the selective_scan interface available in WSL2."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton selective-scan benchmark.")

    device = torch.device("cuda")
    dtype = _dtype_from_name(torch, args.dtype)

    u = torch.randn(args.batch_size, args.dim, args.seq_len, device=device, dtype=dtype)
    delta = torch.rand(args.batch_size, args.dim, args.seq_len, device=device, dtype=dtype)
    A = torch.randn(args.dim, args.d_state, device=device, dtype=torch.float32)
    B = torch.randn(args.dim, args.d_state, device=device, dtype=torch.float32)
    C = torch.randn(args.dim, args.d_state, device=device, dtype=torch.float32)
    D = torch.randn(args.dim, device=device, dtype=torch.float32)

    for _ in range(max(0, args.warmup_runs)):
        _ = selective_scan_fn(u, delta, A, B, C, D=D, delta_softplus=args.delta_softplus)
    torch.cuda.synchronize()

    repeats: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    checksums: list[float] = []
    finite_output_ratios: list[float] = []
    for repeat_index in range(max(1, args.benchmark_repeats)):
        start_time = time.perf_counter()
        output = selective_scan_fn(u, delta, A, B, C, D=D, delta_softplus=args.delta_softplus)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        output_float = output.float()
        finite_mask = torch.isfinite(output_float)
        finite_ratio = float(finite_mask.float().mean().item())
        finite_output_ratios.append(finite_ratio)
        checksum = float(torch.nanmean(output_float).item()) if finite_ratio > 0.0 else math.nan
        latencies_ms.append(latency_ms)
        checksums.append(checksum)
        repeats.append(
            {
                "repeat_index": repeat_index,
                "latency_ms": round(latency_ms, 4),
                "output_mean": round(checksum, 8) if math.isfinite(checksum) else None,
                "finite_output_ratio": round(finite_ratio, 6),
                "batch_size": args.batch_size,
                "dim": args.dim,
                "seq_len": args.seq_len,
                "d_state": args.d_state,
                "dtype": args.dtype,
                "delta_softplus": args.delta_softplus,
            }
        )

    finite_checksums = [value for value in checksums if math.isfinite(value)]
    summary = {
        "kernel": "mamba_ssm.ops.selective_scan_interface.selective_scan_fn",
        "device": torch.cuda.get_device_name(device),
        "cuda_available": True,
        "batch_size": args.batch_size,
        "dim": args.dim,
        "seq_len": args.seq_len,
        "d_state": args.d_state,
        "dtype": args.dtype,
        "delta_softplus": args.delta_softplus,
        "warmup_runs": args.warmup_runs,
        "benchmark_repeats": args.benchmark_repeats,
        "latency_mean_ms": round(mean(latencies_ms), 4),
        "latency_std_ms": round(pstdev(latencies_ms), 4) if len(latencies_ms) > 1 else 0.0,
        "latency_min_ms": round(min(latencies_ms), 4),
        "latency_max_ms": round(max(latencies_ms), 4),
        "output_mean_mean": round(mean(finite_checksums), 8) if finite_checksums else None,
        "finite_output_ratio_mean": round(mean(finite_output_ratios), 6),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "repeats.csv", repeats)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"output_dir": str(output_dir), "summary": summary}


def main() -> None:
    args = _parse_args()
    result = run_benchmark(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()