#!/bin/bash
# Fix Quamba build by properly initializing submodules and following official build order
# This script implements the corrected build sequence from Quamba README

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
REPO_ROOT="${REPO_ROOT:-.}"
MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-.mamba}"
ENV_NAME="${ENV_NAME:-quamba-py310}"
MAX_JOBS="${MAX_JOBS:-4}"
TORCH_CUDA_ARCH="${TORCH_CUDA_ARCH_LIST:-sm_86}"

echo -e "${GREEN}[Quamba Fix] Starting corrected build sequence${NC}"
echo "REPO_ROOT=$REPO_ROOT"
echo "ENV=$ENV_NAME"
echo "TORCH_CUDA_ARCH=$TORCH_CUDA_ARCH"
echo ""

# Detect micromamba
MM=""
if command -v micromamba &> /dev/null; then
    MM="$(command -v micromamba)"
    echo -e "${GREEN}✓${NC} Using system micromamba: $MM"
fi

# Try repository/local fallback paths if system micromamba is not available.
if [ -z "$MM" ]; then
    for mm_path in \
        "$MAMBA_ROOT/../bin/micromamba" \
        "$REPO_ROOT/.wsl-tools/bin/micromamba" \
        "$HOME/.corey-wsl-tools/bin/micromamba" \
        "$HOME/.adama-wsl-tools/bin/micromamba" \
        "/home/bobma-resideo/.corey-wsl-tools/bin/micromamba" \
        "/home/bobma-resideo/.adama-wsl-tools/bin/micromamba"; do
        if [ -x "$mm_path" ]; then
            MM="$mm_path"
            echo -e "${GREEN}✓${NC} Using micromamba from $MM"
            break
        fi
    done
fi

if [ -z "$MM" ] || [ ! -x "$MM" ]; then
    echo -e "${RED}✗${NC} micromamba executable not found in PATH or fallback paths"
    exit 1
fi

# Step 0: Ensure submodules are fully initialized
echo ""
echo -e "${YELLOW}[Step 0]${NC} Initializing git submodules..."
cd "$REPO_ROOT/Quamba"
# Normalize submodule URL to avoid SSH key dependency during automated setup.
git config --file=.gitmodules submodule.3rdparty/mamba.url https://github.com/enyac-group/mamba.git || true
git submodule sync --recursive

# Clean stale non-git folders from interrupted submodule clones.
for sm in \
    3rdparty/cutlass \
    3rdparty/fast-hadamard-transform \
    3rdparty/lm-evaluation-harness \
    3rdparty/mamba \
    3rdparty/Megatron-LM; do
    if [ -d "$sm" ] && [ ! -e "$sm/.git" ]; then
        echo -e "${YELLOW}⚠${NC} Removing stale submodule folder: $sm"
        rm -rf "$sm"
    fi
done

git submodule update --init --recursive --force
echo -e "${GREEN}✓${NC} Submodules initialized"

# Verify fast-hadamard-transform has content
if [ ! -f "3rdparty/fast-hadamard-transform/setup.py" ]; then
    echo -e "${RED}✗${NC} fast-hadamard-transform still empty after git init"
    echo "Attempting SSH→HTTPS fallback for mamba submodule..."
    git config --file=.gitmodules --get-regexp url | sed -E 's|- url = git@github.com:|.url = https://github.com/|' | git config --file=.gitmodules --set-multivar-regexp url
    git submodule sync --recursive
    git submodule update --init --recursive --force
fi

if [ ! -f "3rdparty/fast-hadamard-transform/setup.py" ]; then
    echo -e "${RED}✗${NC} fast-hadamard-transform setup.py not found"
    ls -la 3rdparty/fast-hadamard-transform/
    exit 1
fi
echo -e "${GREEN}✓${NC} fast-hadamard-transform ready"

# Step 1: Install fast-hadamard-transform with force build
echo ""
echo -e "${YELLOW}[Step 1]${NC} Installing fast-hadamard-transform..."
# Must set FORCE_BUILD to trigger compilation for all variants (12N, 40N)
export FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE
# Keep setuptools compatible with torch and avoid isolated build env without torch.
$MM run -r "$MAMBA_ROOT" -n "$ENV_NAME" python -m pip install --no-cache-dir "setuptools<82"
$MM run -r "$MAMBA_ROOT" -n "$ENV_NAME" python -m pip install \
    --no-cache-dir \
    --no-build-isolation \
    --no-deps \
    --force-reinstall \
    -v \
    "3rdparty/fast-hadamard-transform"
echo -e "${GREEN}✓${NC} fast-hadamard-transform installed"

# Step 2: Install lm-evaluation-harness
echo ""
echo -e "${YELLOW}[Step 2]${NC} Installing lm-evaluation-harness..."
$MM run -r "$MAMBA_ROOT" -n "$ENV_NAME" python -m pip install \
    --no-cache-dir \
    -v \
    "3rdparty/lm-evaluation-harness"
echo -e "${GREEN}✓${NC} lm-evaluation-harness installed"

# Step 3: Install mamba with force build
echo ""
echo -e "${YELLOW}[Step 3]${NC} Installing mamba-ssm..."
export MAMBA_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH"
export MAX_JOBS="$MAX_JOBS"
$MM run -r "$MAMBA_ROOT" -n "$ENV_NAME" python -m pip install \
    --no-cache-dir \
    --force-reinstall \
    -v \
    --no-build-isolation \
    "3rdparty/mamba"
echo -e "${GREEN}✓${NC} mamba-ssm installed"

# Step 4: Build CUTLASS
echo ""
echo -e "${YELLOW}[Step 4]${NC} Building CUTLASS (this may take 10-20 minutes)..."
cd "$REPO_ROOT/Quamba"
if [ -f "build_cutlass.sh" ]; then
    bash build_cutlass.sh
else
    echo -e "${YELLOW}⚠${NC} build_cutlass.sh not found, skipping CUTLASS build"
fi
echo -e "${GREEN}✓${NC} CUTLASS built/skipped"

# Step 5: Install Megatron-LM
echo ""
echo -e "${YELLOW}[Step 5]${NC} Installing Megatron-LM..."
$MM run -r "$MAMBA_ROOT" -n "$ENV_NAME" python -m pip install \
    --no-cache-dir \
    -e "3rdparty/Megatron-LM"
# Re-apply requirements.txt due to Megatron forcing pytorch 2.6.0+cu124
echo -e "${YELLOW}⚠${NC} Re-applying requirements.txt after Megatron install..."
$MM run -r "$MAMBA_ROOT" -n "$ENV_NAME" python -m pip install \
    --no-cache-dir \
    -r requirements.txt
echo -e "${GREEN}✓${NC} Megatron-LM installed and requirements re-applied"

# Step 6: Install Quamba package
echo ""
echo -e "${YELLOW}[Step 6]${NC} Installing Quamba package itself..."
cd "$REPO_ROOT/Quamba"
export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH"
export MAX_JOBS="$MAX_JOBS"
$MM run -r "$MAMBA_ROOT" -n "$ENV_NAME" python -m pip install \
    --no-cache-dir \
    --force-reinstall \
    -v \
    .
echo -e "${GREEN}✓${NC} Quamba package installed"

# Step 7: Smoke test
echo ""
echo -e "${YELLOW}[Step 7]${NC} Running smoke test..."
$MM run -r "$MAMBA_ROOT" -n "$ENV_NAME" python -c "
import torch
import quamba
print('✓ torch.cuda.is_available():', torch.cuda.is_available())
print('✓ quamba imported successfully')
if torch.cuda.is_available():
    print('✓ CUDA device:', torch.cuda.get_device_name(0))
" 2>&1 | tee -a "$REPO_ROOT/src/outputs/quamba_smoke_test.log"

echo ""
echo -e "${GREEN}✓ Quamba build completed successfully!${NC}"
echo "You can now use: $MM run -r $MAMBA_ROOT -n $ENV_NAME python ..."
