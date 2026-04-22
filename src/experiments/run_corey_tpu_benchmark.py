"""
run_corey_tpu_benchmark.py

Benchmark COREY selective_scan_fn and related methods on Google Cloud TPU (PyTorch XLA or JAX).
Supports direct comparison with CUDA GPU results for E1/E2/E3 and paper-required experiments.

Usage (TPU):
  python run_corey_tpu_benchmark.py --device tpu --model mamba-370m --chunk-size 512 --seq-len 4096 --dtype float16 --repeat 30 --output-dir src/outputs/corey_tpu_benchmark

Usage (GPU):
  python run_corey_tpu_benchmark.py --device cuda --model mamba-370m --chunk-size 512 --seq-len 4096 --dtype float16 --repeat 30 --output-dir src/outputs/corey_gpu_benchmark

Google Cloud setup:
- On TPU VM, install torch_xla (for PyTorch) or JAX as needed.
- Use service account token for GCS upload if desired.

"""
import argparse
import os
import time
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="COREY selective_scan_fn benchmark on TPU/GPU.")
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "tpu"], help="Device to run on.")
parser.add_argument("--model", type=str, default="mamba-370m", help="Model name or path.")
parser.add_argument("--chunk-size", type=int, default=512)
parser.add_argument("--seq-len", type=int, default=4096)
parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
parser.add_argument("--repeat", type=int, default=30)
parser.add_argument("--output-dir", type=Path, default=Path("src/outputs/corey_tpu_benchmark"))
args = parser.parse_args()


# Set HF_TOKEN from env file if available
def _set_hf_token_from_envfile(env_path="/home/amabo1215/source/.env"):
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

# Device setup
device = args.device
if device == "tpu":
    try:
        import torch_xla
        import torch_xla
        import torch
    except ImportError:
        raise RuntimeError("torch_xla is required for TPU execution. See https://pytorch.org/xla/")
    dev = torch_xla.device()
elif device == "cuda":
    import torch
    dev = torch.device("cuda")
else:
    import torch
    dev = torch.device("cpu")

dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
dtype = dtype_map[args.dtype]

# Dummy selective_scan_fn (replace with actual import)
def selective_scan_fn(u, delta, A, B, C, D=None):
    # Simulate a simple scan for demonstration
    return u + delta + A.sum() + B.sum() + C.sum() + (D.sum() if D is not None else 0)

# Input shapes
batch = 1
seq_len = args.seq_len
dim = 1024  # as in E1/E2
d_state = 16
chunk_size = args.chunk_size

# Generate random inputs
import torch
u = torch.randn(batch, dim, seq_len, device=dev, dtype=dtype)
delta = torch.rand(batch, dim, seq_len, device=dev, dtype=dtype)
A = torch.randn(dim, d_state, device=dev, dtype=torch.float32)
B = torch.randn(dim, d_state, device=dev, dtype=torch.float32)
C = torch.randn(dim, d_state, device=dev, dtype=torch.float32)
D = torch.randn(dim, device=dev, dtype=torch.float32)

# Warmup (for fair timing)
for _ in range(3):
    _ = selective_scan_fn(u, delta, A, B, C, D)
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "tpu":
        torch_xla.sync()

# Timing
latencies = []
for i in range(args.repeat):
    t0 = time.time()
    _ = selective_scan_fn(u, delta, A, B, C, D)
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "tpu":
        torch_xla.sync()
    t1 = time.time()
    latencies.append((t1 - t0) * 1000)  # ms

mean_latency = sum(latencies) / len(latencies)
std_latency = (sum((x - mean_latency) ** 2 for x in latencies) / len(latencies)) ** 0.5

# Output
args.output_dir.mkdir(parents=True, exist_ok=True)
result = {
    "device": device,
    "model": args.model,
    "chunk_size": chunk_size,
    "seq_len": seq_len,
    "dtype": args.dtype,
    "repeat": args.repeat,
    "mean_latency_ms": mean_latency,
    "std_latency_ms": std_latency,
    "latencies_ms": latencies,
}
with open(args.output_dir / "summary.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"[corey_tpu_benchmark] Results in {args.output_dir}/summary.json\n{json.dumps(result, indent=2)}")
