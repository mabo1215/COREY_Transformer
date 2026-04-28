#!/usr/bin/env python
"""Matched FlashAttention-3 benchmark for Hopper/H800.

This script is a kernel-level external baseline for the COREY paper.  It
benchmarks the official FlashAttention-3 Hopper interface on matched tensor
shapes and records enough metadata to make unsupported/fallback runs explicit.

Expected FA3 install:
    git clone https://github.com/Dao-AILab/flash-attention
    cd flash-attention/hopper
    python setup.py install

FA3 import used here:
    import flash_attn_interface
    flash_attn_interface.flash_attn_func(q, k, v, ...)
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


def _run_text(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def _env_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import torch

        meta.update(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
            }
        )
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            meta.update(
                {
                    "gpu_name": torch.cuda.get_device_name(idx),
                    "gpu_capability": list(torch.cuda.get_device_capability(idx)),
                    "gpu_count_visible": torch.cuda.device_count(),
                }
            )
    except Exception as exc:
        meta["torch_error"] = repr(exc)

    smi = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,cuda_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    if smi:
        meta["nvidia_smi"] = smi
    return meta


def _load_fa3() -> tuple[Callable[..., Any] | None, str | None, str | None]:
    """Return (flash_attn_func, module_name, error)."""
    candidates = [
        "flash_attn_interface",  # official hopper install path
        "hopper.flash_attn_interface",  # source-tree PYTHONPATH fallback
    ]
    errors: list[str] = []
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            fn = getattr(mod, "flash_attn_func")
            return fn, name, None
        except Exception as exc:
            errors.append(f"{name}: {exc!r}")
    return None, None, "; ".join(errors)


def _dtype_from_name(name: str):
    import torch

    table = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    key = name.lower()
    if key not in table:
        raise ValueError(f"Unsupported dtype {name!r}; use fp16 or bf16")
    return table[key]


def _call_fa3(
    flash_attn_func: Callable[..., Any],
    q: Any,
    k: Any,
    v: Any,
    *,
    causal: bool,
    softmax_scale: float | None,
) -> Any:
    try:
        out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    except TypeError as exc:
        if "dropout_p" not in str(exc):
            raise
        # FlashAttention-3 hopper interface has no dropout_p parameter; dropout
        # is implicitly disabled for this eval-only benchmark.
        out = flash_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    return out[0] if isinstance(out, tuple) else out


def _benchmark_one(
    flash_attn_func: Callable[..., Any],
    *,
    batch_size: int,
    seq_len: int,
    n_heads: int,
    head_dim: int,
    dtype_name: str,
    causal: bool,
    warmup: int,
    repeats: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed + seq_len + n_heads + head_dim)
    dtype = _dtype_from_name(dtype_name)
    shape = (batch_size, seq_len, n_heads, head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Warmup compiles/selects kernels and fills caches.
    for _ in range(warmup):
        out = _call_fa3(
            flash_attn_func, q, k, v, causal=causal, softmax_scale=softmax_scale
        )
    torch.cuda.synchronize()

    latencies_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = _call_fa3(
            flash_attn_func, q, k, v, causal=causal, softmax_scale=softmax_scale
        )
        end.record()
        torch.cuda.synchronize()
        latencies_ms.append(float(start.elapsed_time(end)))

    # Keep a tiny observable value after timing to catch NaNs without moving the
    # whole tensor to host.
    checksum = float(out.float().mean().detach().cpu())
    mean_ms = statistics.mean(latencies_ms)
    std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)

    # Standard fwd attention FLOPs estimate: QK^T + P@V, multiply-add counted as
    # two FLOPs. Causal uses roughly half of the SxS attention region.
    dense_flops = 4.0 * batch_size * n_heads * seq_len * seq_len * head_dim
    effective_flops = dense_flops * (0.5 if causal else 1.0)
    tflops = effective_flops / (mean_ms * 1.0e-3) / 1.0e12
    tokens_per_s = batch_size * seq_len / (mean_ms * 1.0e-3)

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "hidden_dim": n_heads * head_dim,
        "dtype": dtype_name,
        "causal": causal,
        "warmup": warmup,
        "repeats": repeats,
        "latency_mean_ms": mean_ms,
        "latency_std_ms": std_ms,
        "latency_min_ms": min_ms,
        "latency_max_ms": max_ms,
        "tokens_per_s": tokens_per_s,
        "nominal_forward_tflops": tflops,
        "checksum_mean": checksum,
        "latencies_ms": latencies_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Matched FlashAttention-3 H800 benchmark.")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-heads", type=int, nargs="+", default=[16])
    parser.add_argument("--head-dims", type=int, nargs="+", default=[64])
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--require-sm90",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require Hopper SM90/H800/H100. Disable only for diagnostics.",
    )
    parser.add_argument(
        "--allow-unsupported",
        action="store_true",
        help="Write unsupported metadata instead of exiting non-zero.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/outputs/flashattention3_matched"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "summary.json"

    meta = _env_metadata()
    output: dict[str, Any] = {
        "benchmark": "flashattention3_matched",
        "status": "initializing",
        "metadata": meta,
        "args": vars(args) | {"output_dir": str(args.output_dir)},
        "results": [],
    }

    def unsupported(reason: str, exit_code: int = 2) -> int:
        output["status"] = "unsupported"
        output["reason"] = reason
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"[FA3] unsupported: {reason}")
        print(f"[FA3] wrote {out_path}")
        return 0 if args.allow_unsupported else exit_code

    try:
        import torch
    except Exception as exc:
        return unsupported(f"PyTorch import failed: {exc!r}")

    if not torch.cuda.is_available():
        return unsupported("CUDA is not available.")

    capability = torch.cuda.get_device_capability()
    if args.require_sm90 and capability[0] < 9:
        return unsupported(
            f"FlashAttention-3 requires Hopper SM90; detected capability {capability}."
        )

    flash_attn_func, module_name, import_error = _load_fa3()
    if flash_attn_func is None:
        return unsupported(
            "FlashAttention-3 import failed. Install Dao-AILab/flash-attention "
            f"from the hopper/ directory. Details: {import_error}"
        )
    output["metadata"]["fa3_module"] = module_name

    print("[FA3] starting matched benchmark")
    print(f"[FA3] GPU: {torch.cuda.get_device_name(0)} capability={capability}")
    print(f"[FA3] module: {module_name}")

    try:
        for n_heads in args.n_heads:
            for head_dim in args.head_dims:
                for seq_len in args.seq_lens:
                    print(
                        "[FA3] "
                        f"B={args.batch_size} S={seq_len} H={n_heads} D={head_dim} "
                        f"dtype={args.dtype} causal={args.causal}"
                    )
                    result = _benchmark_one(
                        flash_attn_func,
                        batch_size=args.batch_size,
                        seq_len=seq_len,
                        n_heads=n_heads,
                        head_dim=head_dim,
                        dtype_name=args.dtype,
                        causal=args.causal,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        seed=args.seed,
                        device=args.device,
                    )
                    output["results"].append(result)
                    print(
                        "      "
                        f"{result['latency_mean_ms']:.3f} +/- "
                        f"{result['latency_std_ms']:.3f} ms, "
                        f"{result['nominal_forward_tflops']:.1f} TFLOP/s"
                    )
    except RuntimeError as exc:
        output["status"] = "failed"
        output["reason"] = repr(exc)
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        raise

    output["status"] = "ok"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[FA3] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
