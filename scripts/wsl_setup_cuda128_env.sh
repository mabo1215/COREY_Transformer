#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/c/source/ADAMA_Transformer"
TOOLS_DIR="${HOME}/.adama-wsl-tools"
MAMBA_ROOT_PREFIX="${HOME}/.adama-micromamba"
ENV_NAME="adama-cuda128"
ENV_PREFIX="$MAMBA_ROOT_PREFIX/envs/$ENV_NAME"
REPO_MICROMAMBA_BIN="$REPO_ROOT/.wsl-tools/bin/micromamba"
MICROMAMBA_BIN="$TOOLS_DIR/bin/micromamba"

mkdir -p "$TOOLS_DIR" "$MAMBA_ROOT_PREFIX"

if [[ -x "$REPO_MICROMAMBA_BIN" ]]; then
  MICROMAMBA_BIN="$REPO_MICROMAMBA_BIN"
elif [[ ! -x "$MICROMAMBA_BIN" ]]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$TOOLS_DIR" bin/micromamba
fi

export MAMBA_ROOT_PREFIX
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-4}"

if [[ -d "$ENV_PREFIX" ]]; then
  if ! "$MICROMAMBA_BIN" run -n "$ENV_NAME" python -m pip --version >/dev/null 2>&1; then
    "$MICROMAMBA_BIN" remove -y -n "$ENV_NAME" --all
  fi
fi

if [[ ! -d "$ENV_PREFIX" ]]; then
  "$MICROMAMBA_BIN" create -y -n "$ENV_NAME" -c nvidia -c conda-forge python=3.11 pip setuptools wheel ninja cuda-nvcc=12.8
fi

"$MICROMAMBA_BIN" run -n "$ENV_NAME" python -m pip install --upgrade pip
if ! "$MICROMAMBA_BIN" run -n "$ENV_NAME" python - <<'PY'
import sys
import torch

sys.exit(0 if torch.__version__ == '2.11.0+cu128' else 1)
PY
then
  "$MICROMAMBA_BIN" run -n "$ENV_NAME" python -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
fi
"$MICROMAMBA_BIN" run -n "$ENV_NAME" python -m pip install --upgrade transformers datasets sentencepiece accelerate psutil huggingface_hub

TRITON_STATUS="ok"
if ! "$MICROMAMBA_BIN" run -n "$ENV_NAME" python -m pip install triton; then
  TRITON_STATUS="failed"
fi

CAUSAL_STATUS="ok"
if ! "$MICROMAMBA_BIN" run -n "$ENV_NAME" bash -lc "export TORCH_CUDA_ARCH_LIST='$TORCH_CUDA_ARCH_LIST'; export MAX_JOBS='$MAX_JOBS'; python -m pip install --no-build-isolation causal-conv1d"; then
  CAUSAL_STATUS="failed"
fi

MAMBA_STATUS="ok"
if ! "$MICROMAMBA_BIN" run -n "$ENV_NAME" bash -lc "export TORCH_CUDA_ARCH_LIST='$TORCH_CUDA_ARCH_LIST'; export MAX_JOBS='$MAX_JOBS'; python -m pip install --no-build-isolation mamba-ssm"; then
  MAMBA_STATUS="failed"
fi

"$MICROMAMBA_BIN" run -n "$ENV_NAME" bash -lc "export PYTHONPATH='$REPO_ROOT'; python - <<'PY'
import json
import shutil
import subprocess

import torch

from src.algorithms.mamba_integration import official_mamba_fast_path_status

report = {
    'torch_version': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'torch_cuda_version': torch.version.cuda,
    'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'nvcc_path': shutil.which('nvcc'),
    'nvcc_version': subprocess.check_output(['nvcc', '--version'], text=True).strip().splitlines()[-1] if shutil.which('nvcc') else None,
    'fast_path_status': official_mamba_fast_path_status(),
}
print(json.dumps(report, indent=2))
PY"

printf '\n[setup-status] triton=%s causal-conv1d=%s mamba-ssm=%s\n' "$TRITON_STATUS" "$CAUSAL_STATUS" "$MAMBA_STATUS"
