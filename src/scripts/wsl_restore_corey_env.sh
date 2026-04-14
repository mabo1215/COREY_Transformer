#!/usr/bin/env bash
# Restore corey-cuda128 environment after Quamba build corruption.
# Quamba build v6 downgraded torch to 2.4+cu121; this script restores torch 2.11+cu128
# and rebuilds mamba-ssm against it.
set -euo pipefail

MM=/home/bobma-resideo/.corey-wsl-tools/bin/micromamba
ROOT=/home/bobma-resideo/.corey-micromamba
ENV=corey-cuda128

echo "[restore] Reinstalling torch 2.11+cu128..."
$MM run -r $ROOT -n $ENV python -m pip install \
    --force-reinstall --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cu128

echo "[restore] Verifying torch version..."
$MM run -r $ROOT -n $ENV python -c "import torch; assert '2.11' in torch.__version__, f'Expected 2.11 got {torch.__version__}'; print('[ok] torch:', torch.__version__)"

echo "[restore] Rebuilding mamba-ssm for torch 2.11+cu128 / sm_89..."
# Remove 'kernels' package installed by lm-evaluation-harness: it requires
# huggingface_hub.dataclasses which does not exist in the current pinned version,
# and its import failure blocks pip's metadata subprocess when building mamba-ssm.
$MM run -r $ROOT -n $ENV pip uninstall -y kernels 2>/dev/null || true
export MM ROOT ENV
export MAMBA_ROOT_PREFIX=$ROOT
export ENV_NAME=$ENV
export MAX_JOBS=2
bash /mnt/c/source/Corey_Transformer/src/scripts/wsl_build_patched_mamba_sm89.sh

echo "[restore] Verifying mamba_ssm..."
$MM run -r $ROOT -n $ENV python -c "import mamba_ssm; print('[ok] mamba_ssm:', mamba_ssm.__version__)"

echo "[restore] corey-cuda128 env RESTORED"
