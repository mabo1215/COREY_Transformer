# WSL2 CUDA 12.8 Migration Checklist

This document captures the remaining deployment-grade benchmark path on WSL2.

## Target State

- WSL distro: Ubuntu 24.04 on WSL2
- Repository mount: `/mnt/c/source/ADAMA_Transformer`
- User-local package manager: micromamba under `$HOME/.adama-micromamba`
- CUDA compiler inside WSL env: `cuda-nvcc=12.8`
- PyTorch wheel inside WSL env: `torch` from the official `cu128` index
- Benchmark env name: `adama-cuda128`

## Current Facts

- `nvidia-smi` works inside WSL, so the GPU is visible to Linux.
- The Windows-side benchmark path can run on the RTX 3070, but it is not deployment-grade because the official Mamba fast path is still unavailable.
- WSL does not require root for the micromamba-based workflow.
- The local Ubuntu image does not provide `pip` or `python3-venv` out of the box, so the WSL workflow uses micromamba instead of system Python tooling.
- The micromamba root is kept on the Linux filesystem rather than `/mnt/c/...` to avoid package cache corruption on DrvFS.

## One-Time Setup

Run the setup script from Windows PowerShell:

```powershell
wsl bash /mnt/c/source/ADAMA_Transformer/scripts/wsl_setup_cuda128_env.sh
```

What the script does:

1. Installs a local micromamba binary under `$HOME/.adama-wsl-tools/` if missing.
2. Creates or updates the `adama-cuda128` environment with Python 3.11, pip, ninja, GCC/G++, and `cuda-nvcc=12.8`.
3. Installs the official PyTorch `cu128` wheel.
4. Installs runtime benchmark dependencies used by the repository.
5. Attempts to install `triton`, `causal-conv1d`, and `mamba-ssm`. The script exports `TORCH_CUDA_ARCH_LIST=8.6` for tools that honor it, but `causal-conv1d` currently hardcodes a broader architecture list in its upstream `setup.py`, so its WSL build can still take noticeably longer.
6. Prints a compact validation report covering `torch.cuda`, `nvcc`, Triton, and the official fast-path symbols.

## One-Click Benchmark

After setup completes, run:

```powershell
wsl bash /mnt/c/source/ADAMA_Transformer/scripts/wsl_run_official_benchmark.sh
```

This launches the official Hugging Face benchmark on GPU and writes outputs under:

- `src/outputs/official_hf_benchmark_wsl/`

## Manual Activation

If you want an interactive shell in the aligned environment:

```bash
export MAMBA_ROOT_PREFIX=$HOME/.adama-micromamba
eval "$(/mnt/c/source/ADAMA_Transformer/.wsl-tools/bin/micromamba shell hook -s bash)"
micromamba activate adama-cuda128
```

## Validation Checklist

Run these checks inside WSL after activation:

```bash
python - <<'PY'
import json, torch
from src.algorithms.mamba_integration import official_mamba_fast_path_status
print(json.dumps({
    'torch_version': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'torch_cuda_version': torch.version.cuda,
    'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'fast_path': official_mamba_fast_path_status(),
}, indent=2))
PY

nvcc --version
```

Deployment-grade status is reached only when all of the following are true:

- `torch.cuda.is_available()` is `True`
- `official_mamba_fast_path_status()` reports all required symbols as `True`
- the benchmark metadata records `fast_path_available=true`
- the benchmark metadata records `deployment_grade=true`

## Known Failure Modes

- If `triton` has no matching wheel, the environment can still run the official HF benchmark, but Triton kernel benchmarking remains blocked.
- If `mamba-ssm` or `causal-conv1d` fail to build, inspect `nvcc --version`, `torch.version.cuda`, and the compiler toolchain first.
- `causal-conv1d` v1.6.1 does not currently honor `TORCH_CUDA_ARCH_LIST`; on CUDA 12.8 it hardcodes a multi-architecture `nvcc` build that includes newer targets beyond RTX 3070, so long compile times are expected even after the environment is otherwise healthy.
- If the benchmark runs on GPU but `deployment_grade=false`, then the naive Mamba path is still active and the run should stay in the appendix rather than the main paper.