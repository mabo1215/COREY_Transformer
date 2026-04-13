#!/bin/bash
set -e
cd /mnt/c/source/Corey_Transformer

echo "[quamba-verify] Starting Quamba build chain verification..."
echo ""

if [[ ! -d "Quamba" ]]; then
    echo "[error] Quamba directory not found at $(pwd)/Quamba"
    exit 1
fi

export INIT_SUBMODULES=0
export INSTALL_CORE_RUNTIME=0
export BUILD_THIRD_PARTY=1
export BUILD_QUAMBA_PACKAGE=1
export MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST=8.6

echo "[verify] Environment settings:"
echo "  BUILD_THIRD_PARTY=$BUILD_THIRD_PARTY"
echo "  BUILD_QUAMBA_PACKAGE=$BUILD_QUAMBA_PACKAGE"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo ""

echo "[exec] Running wsl_run_quamba_phase2.sh..."
bash src/scripts/wsl_run_quamba_phase2.sh

echo ""
echo "[quamba-verify-complete] Quamba build chain verification finished"
