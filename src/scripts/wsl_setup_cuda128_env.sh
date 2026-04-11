#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/c/source/COREY_Transformer}"
TOOLS_DIR="${TOOLS_DIR:-${HOME}/.corey-wsl-tools}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/.corey-micromamba}"
ENV_NAME="${ENV_NAME:-corey-cuda128}"
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
export CUDAARCHS="${CUDAARCHS:-86}"
export MAX_JOBS="${MAX_JOBS:-4}"
WINDOWS_USER="${WINDOWS_USER:-$(cmd.exe /c \"echo %USERNAME%\" 2>/dev/null | tr -d '\r')}"
export HF_HOME="${HF_HOME:-/mnt/c/Users/${WINDOWS_USER:-$USER}/.cache/huggingface}"
export MAMBA_SRC_REF="${MAMBA_SRC_REF:-v2.3.1}"
export MAMBA_GENCODE_FLAGS="${MAMBA_GENCODE_FLAGS:--gencode arch=compute_86,code=sm_86}"

if [[ -d "$ENV_PREFIX" ]]; then
  if ! "$MICROMAMBA_BIN" run -n "$ENV_NAME" python -m pip --version >/dev/null 2>&1; then
    "$MICROMAMBA_BIN" remove -y -n "$ENV_NAME" --all
  fi
fi

if [[ ! -d "$ENV_PREFIX" ]]; then
  "$MICROMAMBA_BIN" create -y -n "$ENV_NAME" -c nvidia -c conda-forge python=3.11 pip setuptools wheel ninja cuda-nvcc=12.8
fi

if ! "$MICROMAMBA_BIN" run -n "$ENV_NAME" python - <<'PY'
import sys
import torch

sys.exit(0 if torch.__version__ == '2.11.0+cu128' else 1)
PY
then
  "$MICROMAMBA_BIN" run -n "$ENV_NAME" python -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
fi

if ! "$MICROMAMBA_BIN" run -n "$ENV_NAME" python - <<'PY'
import importlib

for module_name in [
    "transformers",
    "datasets",
    "sentencepiece",
    "accelerate",
    "psutil",
    "huggingface_hub",
    "triton",
    "einops",
]:
    importlib.import_module(module_name)
PY
then
  "$MICROMAMBA_BIN" run -n "$ENV_NAME" python -m pip install transformers datasets sentencepiece accelerate psutil huggingface_hub triton einops
fi

install_cuda_extension() {
  local package_name="$1"
  local primary_command="$2"
  local fallback_command="$3"

  if "$MICROMAMBA_BIN" run -n "$ENV_NAME" bash -lc "export TORCH_CUDA_ARCH_LIST='$TORCH_CUDA_ARCH_LIST'; export CUDAARCHS='$CUDAARCHS'; export MAX_JOBS='$MAX_JOBS'; export HF_HOME='$HF_HOME'; ${primary_command}"; then
    return 0
  fi

  if [[ -n "$fallback_command" ]] && "$MICROMAMBA_BIN" run -n "$ENV_NAME" bash -lc "export TORCH_CUDA_ARCH_LIST='$TORCH_CUDA_ARCH_LIST'; export CUDAARCHS='$CUDAARCHS'; export MAX_JOBS='$MAX_JOBS'; export HF_HOME='$HF_HOME'; ${fallback_command}"; then
    return 0
  fi

  printf '[warn] %s installation failed after primary and fallback attempts\n' "$package_name" >&2
  return 1
}

python_module_importable() {
  local module_name="$1"
  "$MICROMAMBA_BIN" run -n "$ENV_NAME" python -c "import importlib; importlib.import_module('$module_name')" >/dev/null 2>&1
}

install_patched_mamba_source() {
  local temp_dir
  temp_dir="$(mktemp -d)"
  trap 'rm -rf "$temp_dir"' RETURN

  git clone --depth 1 --branch "$MAMBA_SRC_REF" https://github.com/state-spaces/mamba.git "$temp_dir"
  (
    cd "$temp_dir"
    python - <<'PY'
import os
from pathlib import Path

path = Path("setup.py")
text = path.read_text()
start = '        if bare_metal_version <= Version("12.9"):\n            cc_flag.append("-gencode")\n            cc_flag.append("arch=compute_53,code=sm_53")'
end = '    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as'
start_index = text.index(start)
end_index = text.index(end)
replacement = '''        custom_cc_flags = os.environ.get("MAMBA_CUDA_GENCODE", "").strip().split()
        if custom_cc_flags:
            cc_flag.extend(custom_cc_flags)
        else:
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_75,code=sm_75")
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_80,code=sm_80")
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_87,code=sm_87")
            if bare_metal_version >= Version("11.8"):
                cc_flag.append("-gencode")
                cc_flag.append("arch=compute_90,code=sm_90")
            if bare_metal_version >= Version("12.8"):
                cc_flag.append("-gencode")
                cc_flag.append("arch=compute_100,code=sm_100")
                cc_flag.append("-gencode")
                cc_flag.append("arch=compute_120,code=sm_120")
            if bare_metal_version >= Version("13.0"):
                cc_flag.append("-gencode")
                cc_flag.append("arch=compute_103,code=sm_103")
                cc_flag.append("-gencode")
                cc_flag.append("arch=compute_110,code=sm_110")
                cc_flag.append("-gencode")
                cc_flag.append("arch=compute_121,code=sm_121")

'''
path.write_text(text[:start_index] + replacement + text[end_index:])
PY
    MAMBA_FORCE_BUILD=TRUE MAMBA_CUDA_GENCODE="$MAMBA_GENCODE_FLAGS" python -m pip install --no-build-isolation --no-deps --no-cache-dir --force-reinstall .
  )
}

TRITON_STATUS="ok"
if ! "$MICROMAMBA_BIN" run -n "$ENV_NAME" python -m pip install triton; then
  TRITON_STATUS="failed"
fi

CAUSAL_STATUS="ok"
if python_module_importable "causal_conv1d"; then
  CAUSAL_STATUS="present"
elif ! install_cuda_extension \
  "causal-conv1d" \
  "export CAUSAL_CONV1D_FORCE_BUILD=TRUE; python -m pip install --no-build-isolation --no-deps --no-cache-dir --force-reinstall git+https://github.com/Dao-AILab/causal-conv1d.git@0d2252d" \
  "python -m pip install --no-build-isolation --no-deps causal-conv1d"; then
  CAUSAL_STATUS="failed"
fi

MAMBA_STATUS="ok"
if python_module_importable "mamba_ssm"; then
  MAMBA_STATUS="present"
elif ! "$MICROMAMBA_BIN" run -n "$ENV_NAME" bash -lc "$(declare -f install_patched_mamba_source); export MAMBA_SRC_REF='$MAMBA_SRC_REF'; export MAMBA_GENCODE_FLAGS='$MAMBA_GENCODE_FLAGS'; install_patched_mamba_source" \
  && ! install_cuda_extension \
    "mamba-ssm" \
    "export MAMBA_FORCE_BUILD=TRUE; python -m pip install --no-build-isolation --no-deps --no-cache-dir --force-reinstall git+https://github.com/state-spaces/mamba.git@$MAMBA_SRC_REF" \
    "python -m pip install --no-build-isolation --no-deps mamba-ssm"; then
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
