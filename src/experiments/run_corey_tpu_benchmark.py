"""
run_corey_tpu_benchmark.py

Three-policy selective-scan benchmark for Google Cloud TPU (PyTorch XLA) and
CUDA GPU.  Reproduces the W1 triplet structure (Static-64 / COREY-default /
Static-512-oracle) so TPU latency is directly comparable to the GPU figures in
Table 1 of the main paper (tab:real-gpu-three-policy).

On CUDA, this script tries to import mamba_ssm.ops.selective_scan_interface
and falls back to a pure-PyTorch chunked scan simulation if mamba_ssm is
unavailable.  On TPU (torch_xla), only the pure-PyTorch simulation is used
because mamba_ssm CUDA kernels cannot run on XLA devices; latency therefore
reflects XLA kernel dispatch and memory bandwidth rather than Triton kernel
launch overhead.

Usage (TPU VM):
  python run_corey_tpu_benchmark.py --device tpu --model mamba-370m \
    --chunk-size 512 --seq-len 4096 --dtype float16 --repeat 30 \
    --output-dir src/outputs/corey_tpu_benchmark

Usage (CUDA GPU):
  python run_corey_tpu_benchmark.py --device cuda --model mamba-370m \
    --chunk-size 512 --seq-len 4096 --dtype float16 --repeat 30 \
    --output-dir src/outputs/corey_gpu_benchmark

Google Cloud setup:
  On TPU VM, install torch_xla (for PyTorch) as needed.
  Use service account token for GCS upload if desired.
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

# ---------------------------------------------------------------------------
# Argument parsing (module-level so it works when called directly or via torchrun)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="COREY selective_scan_fn three-policy benchmark on TPU/GPU.")
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "tpu"])
parser.add_argument("--model", type=str, default="mamba-370m")
parser.add_argument("--chunk-size", type=int, default=512, help="Max chunk size (used for COREY and Static-512 oracle).")
parser.add_argument("--seq-len", type=int, default=4096)
parser.add_argument("--dim", type=int, default=1024)
parser.add_argument("--d-state", type=int, default=16)
parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
parser.add_argument("--repeat", type=int, default=30)
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--num-bins", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output-dir", type=Path, default=Path("src/outputs/corey_tpu_benchmark"))
args = parser.parse_args()


# ---------------------------------------------------------------------------
# HF token helper
# ---------------------------------------------------------------------------

def _set_hf_token_from_envfile(env_path: str = "/home/amabo1215/source/.env") -> None:
    import os
    try:
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if line.strip().startswith("Huggingface_model_token:="):
                        token = line.strip().split("Huggingface_model_token:=", 1)[-1].strip()
                        if token:
                            os.environ["HF_TOKEN"] = token
                            print(f"[corey_bench] Set HF_TOKEN from {env_path}")
                        break
    except Exception as e:
        print(f"[corey_bench] Failed to set HF_TOKEN from {env_path}: {e}")


_set_hf_token_from_envfile()


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

import torch

_xm = None
if args.device == "tpu":
    try:
        import torch_xla.core.xla_model as _xm
    except ImportError:
        raise RuntimeError("torch_xla is required for TPU execution. See https://pytorch.org/xla/")
    dev = _xm.xla_device()
elif args.device == "cuda":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device found.")
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")

dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
dtype = dtype_map[args.dtype]


# ---------------------------------------------------------------------------
# Selective-scan backends
# ---------------------------------------------------------------------------

_MAMBA_SSM_AVAILABLE = False
_selective_scan_fn_cuda: Any = None

if args.device == "cuda":
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _ssm_fn
        _selective_scan_fn_cuda = _ssm_fn
        _MAMBA_SSM_AVAILABLE = True
        print("[corey_bench] mamba_ssm backend: available (using CUDA Triton kernel)")
    except ImportError:
        print("[corey_bench] mamba_ssm not found; using pure-PyTorch scan simulation")
else:
    print(f"[corey_bench] Device={args.device}; using pure-PyTorch scan simulation")


def _pytorch_selective_scan(u: Any, delta: Any, A: Any, B: Any, C: Any,
                             D: Any | None = None, **kwargs: Any) -> Any:
    """
    Pure-PyTorch chunked selective-scan simulation.

    Each call processes one chunk of the sequence.  The computation is a
    simplified SSM step with correct memory footprint (O(seq_len * dim)),
    suitable for benchmarking XLA/CPU dispatch and memory bandwidth.
    This is NOT numerically equivalent to mamba_ssm; it is a latency proxy.
    """
    # u:     [batch, dim, seq_len_chunk]
    # delta: [batch, dim, seq_len_chunk]
    # A:     [dim, d_state]
    # D:     [dim] or None
    weighted = u * delta.sigmoid()
    state_proxy = weighted.mean(dim=-1, keepdim=True).expand_as(weighted)
    out = state_proxy
    if D is not None:
        out = out + D.float().unsqueeze(0).unsqueeze(-1) * u.float()
    return out.to(u.dtype)


def _scan_fn(u: Any, delta: Any, A: Any, B: Any, C: Any,
             D: Any | None = None, **kwargs: Any) -> Any:
    """Dispatch to mamba_ssm or pure-PyTorch fallback."""
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
# Device sync helper
# ---------------------------------------------------------------------------

def _sync() -> None:
    if _xm is not None:
        _xm.mark_step()
    elif args.device == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Per-policy benchmark function
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
    _sync()

    latencies: list[float] = []
    for _ in range(repeat):
        _sync()
        t0 = time.perf_counter()
        _one_pass()
        _sync()
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
        "latencies_ms": [round(x, 6) for x in latencies],
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

torch.manual_seed(args.seed)

batch = 1
seq_len = args.seq_len
dim = args.dim
d_state = args.d_state

u = torch.randn(batch, dim, seq_len, device=dev, dtype=dtype)
delta = torch.rand(batch, dim, seq_len, device=dev, dtype=dtype)
A = torch.randn(dim, d_state, device=dev, dtype=torch.float32)
B = torch.randn(dim, d_state, device=dev, dtype=torch.float32)
C = torch.randn(dim, d_state, device=dev, dtype=torch.float32)
D_param = torch.randn(dim, device=dev, dtype=torch.float32)

# Compute entropy for COREY (H_ref = log K)
print("[corey_bench] Computing entropy for COREY policy ...")
t_e0 = time.perf_counter()
entropy_nats = _hist_entropy(u, num_bins=args.num_bins)
entropy_overhead_ms = (time.perf_counter() - t_e0) * 1000.0
corey_chunk = _entropy_to_chunk(entropy_nats, num_bins=args.num_bins)
h_ref = math.log(args.num_bins)
print(f"[corey_bench]   H={entropy_nats:.4f} nats  H_ref=log({args.num_bins})={h_ref:.4f}"
      f"  r={min(entropy_nats/h_ref, 1.0):.3f}  → chunk={corey_chunk}")

policies = [
    ("static_64",  64),
    ("corey",      corey_chunk),
    ("static_512", 512),
]

print(f"\n[corey_bench] Running three-policy benchmark on {args.device.upper()}"
      f"  seq_len={seq_len}  warmup={args.warmup}  repeat={args.repeat}")
print(f"[corey_bench] Backend: {'mamba_ssm Triton kernel' if _MAMBA_SSM_AVAILABLE else 'pure-PyTorch simulation'}\n")

policy_results: list[dict[str, Any]] = []
for pname, csize in policies:
    ent = entropy_nats if pname == "corey" else None
    print(f"[corey_bench]   policy={pname:<12}  chunk={csize} ...", end=" ", flush=True)
    r = _benchmark_policy(
        u, delta, A, B, C, D_param,
        chunk_size=csize,
        warmup=args.warmup,
        repeat=args.repeat,
        policy_name=pname,
        entropy_nats=ent,
    )
    policy_results.append(r)
    print(f"lat={r['latency_mean_ms']:.4f}±{r['latency_std_ms']:.4f} ms  "
          f"n_chunks={r['n_chunks']}")

# Compute speedups relative to static_64
ref_lat = next(r["latency_mean_ms"] for r in policy_results if r["policy"] == "static_64")
oracle_lat = next(r["latency_mean_ms"] for r in policy_results if r["policy"] == "static_512")
for r in policy_results:
    r["speedup_A"] = round(ref_lat / r["latency_mean_ms"], 4) if r["latency_mean_ms"] > 0 else None
    r["speedup_B"] = round(oracle_lat / r["latency_mean_ms"], 4) if r["latency_mean_ms"] > 0 else None

print(f"\n[corey_bench] === Speedup summary (Speedup-A vs static-64) ===")
for r in policy_results:
    print(f"  {r['policy']:<12}  chunk={r['chunk_size']:>3}  "
          f"lat={r['latency_mean_ms']:.4f} ms  "
          f"Speedup-A={r['speedup_A']:.2f}x  Speedup-B={r['speedup_B']:.2f}x")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

device_label = (
    str(_xm.xla_device()) if _xm is not None
    else (torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu")
)

output = {
    "device": args.device,
    "device_label": device_label,
    "backend": "mamba_ssm" if _MAMBA_SSM_AVAILABLE else "pytorch_sim",
    "model": args.model,
    "seq_len": seq_len,
    "dim": dim,
    "d_state": d_state,
    "dtype": args.dtype,
    "warmup": args.warmup,
    "repeat": args.repeat,
    "num_bins": args.num_bins,
    "h_ref_nats": round(h_ref, 6),
    "entropy_nats": round(entropy_nats, 6),
    "entropy_overhead_ms": round(entropy_overhead_ms, 6),
    "corey_chunk": corey_chunk,
    "platform": platform.platform(),
    "policies": policy_results,
}

args.output_dir.mkdir(parents=True, exist_ok=True)
out_path = args.output_dir / "summary.json"
out_path.write_text(json.dumps(output, indent=2))
print(f"\n[corey_bench] Results saved to {out_path}")
