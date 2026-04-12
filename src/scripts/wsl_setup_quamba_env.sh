#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/c/source/COREY_Transformer}"
QUAMBA_DIR="${QUAMBA_DIR:-$REPO_ROOT/Quamba}"
TOOLS_DIR="${TOOLS_DIR:-${HOME}/.corey-wsl-tools}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/.adama-micromamba}"
ENV_NAME="${ENV_NAME:-quamba-py310}"
ENV_PREFIX="$MAMBA_ROOT_PREFIX/envs/$ENV_NAME"
REPO_MICROMAMBA_BIN="$REPO_ROOT/.wsl-tools/bin/micromamba"
MICROMAMBA_BIN="$TOOLS_DIR/bin/micromamba"
INIT_SUBMODULES="${INIT_SUBMODULES:-1}"
INSTALL_CORE_RUNTIME="${INSTALL_CORE_RUNTIME:-1}"
BUILD_THIRD_PARTY="${BUILD_THIRD_PARTY:-0}"
BUILD_QUAMBA_PACKAGE="${BUILD_QUAMBA_PACKAGE:-0}"

if [[ -n "${WINDOWS_USER:-}" ]]; then
  DETECTED_WINDOWS_USER="$WINDOWS_USER"
elif command -v cmd.exe >/dev/null 2>&1; then
  DETECTED_WINDOWS_USER="$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r')"
else
  DETECTED_WINDOWS_USER=""
fi

if [[ -n "${HF_HOME:-}" ]]; then
  EFFECTIVE_HF_HOME="$HF_HOME"
elif [[ -n "$DETECTED_WINDOWS_USER" && -d "/mnt/c/Users/$DETECTED_WINDOWS_USER" ]]; then
  EFFECTIVE_HF_HOME="/mnt/c/Users/$DETECTED_WINDOWS_USER/.cache/huggingface"
else
  EFFECTIVE_HF_HOME="${HOME}/.cache/huggingface"
fi

WINDOWS_USER="$DETECTED_WINDOWS_USER"
HF_HOME="$EFFECTIVE_HF_HOME"

mkdir -p "$TOOLS_DIR" "$MAMBA_ROOT_PREFIX"

if [[ -x "$REPO_MICROMAMBA_BIN" ]]; then
  MICROMAMBA_BIN="$REPO_MICROMAMBA_BIN"
elif [[ ! -x "$MICROMAMBA_BIN" ]]; then
  if command -v curl >/dev/null 2>&1; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$TOOLS_DIR" bin/micromamba
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$TOOLS_DIR" bin/micromamba
  else
    printf '[error] neither curl nor wget is available to download micromamba\n' >&2
    exit 1
  fi
fi

export MAMBA_ROOT_PREFIX
export HF_HOME

if [[ ! -d "$QUAMBA_DIR" ]]; then
  printf '[error] Quamba checkout not found at %s\n' "$QUAMBA_DIR" >&2
  exit 1
fi

run_in_env() {
  "$MICROMAMBA_BIN" run -n "$ENV_NAME" "$@"
}

if [[ ! -d "$ENV_PREFIX" ]]; then
  "$MICROMAMBA_BIN" create -y -n "$ENV_NAME" -c conda-forge python=3.10 pip cmake ninja
fi

if [[ "$INIT_SUBMODULES" == "1" ]]; then
  git config --global url."https://github.com/".insteadOf git@github.com:
  (
    cd "$QUAMBA_DIR"
    git submodule sync --recursive
    git submodule update --init --recursive --depth 1 --jobs 4
  )
fi

if [[ "$INSTALL_CORE_RUNTIME" == "1" ]]; then
  if ! run_in_env python - <<'PY'
import importlib
import sys

expected = {
    'torch': '2.4.0+cu121',
    'torchvision': '0.19.0+cu121',
    'torchaudio': '2.4.0+cu121',
    'transformers': '4.41.2',
    'peft': '0.10.0',
    'huggingface_hub': '0.27.0',
    'datasets': '2.19.0',
    'numpy': '1.26.4',
    'sentencepiece': '0.2.0',
    'google.protobuf': '4.25.2',
    'sklearn': '1.6.1',
    'scipy': '1.15.2',
    'zstandard': '0.23.0',
}

for module_name, version in expected.items():
    module = importlib.import_module(module_name)
    actual = getattr(module, '__version__', None)
    if actual != version:
        sys.exit(1)
PY
  then
    run_in_env python -m pip install --upgrade pip
    run_in_env python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    run_in_env python -m pip install \
      transformers==4.41.2 \
      peft==0.10.0 \
      huggingface_hub==0.27.0 \
      datasets==2.19.0 \
      zstandard==0.23.0 \
      numpy==1.26.4 \
      sentencepiece==0.2.0 \
      protobuf==4.25.2 \
      scikit-learn==1.6.1 \
      scipy==1.15.2
  fi
fi

if [[ "$BUILD_THIRD_PARTY" == "1" ]]; then
  (
    cd "$QUAMBA_DIR"
    run_in_env env FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE python -m pip install 3rdparty/fast-hadamard-transform
    run_in_env python -m pip install 3rdparty/lm-evaluation-harness
    run_in_env env MAMBA_FORCE_BUILD=TRUE python -m pip install 3rdparty/mamba
    run_in_env bash build_cutlass.sh
    run_in_env python -m pip install --no-deps -e 3rdparty/Megatron-LM
  )
fi

if [[ "$BUILD_QUAMBA_PACKAGE" == "1" ]]; then
  (
    cd "$QUAMBA_DIR"
    run_in_env python -m pip install .
  )
fi

run_in_env python - <<'PY'
import sys

import datasets
import numpy
import scipy
import sentencepiece
import sklearn
import torch
import transformers

print('python', sys.version.split()[0])
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('datasets', datasets.__version__)
print('numpy', numpy.__version__)
print('sentencepiece', sentencepiece.__version__)
print('sklearn', sklearn.__version__)
print('scipy', scipy.__version__)
print('cuda_available', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
PY

printf '\n[quamba-setup] env=%s init_submodules=%s install_core=%s build_third_party=%s build_quamba_package=%s\n' \
  "$ENV_NAME" "$INIT_SUBMODULES" "$INSTALL_CORE_RUNTIME" "$BUILD_THIRD_PARTY" "$BUILD_QUAMBA_PACKAGE"