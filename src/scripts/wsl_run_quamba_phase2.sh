#!/usr/bin/env bash
# Quamba phase-2 build: third-party extensions + package install.
# Prerequisite: wsl_setup_quamba_env.sh phase-1 (core runtime) already succeeded
# from task 37/44.  This script skips submodule re-init and core-runtime
# reinstall and jumps straight to the BUILD_THIRD_PARTY / BUILD_QUAMBA_PACKAGE
# stages that were unblocked in task 44.
#
# Sequence:
#   1. fast-hadamard-transform  (FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE)
#   2. lm-evaluation-harness
#   3. mamba                    (MAMBA_FORCE_BUILD=TRUE)
#   4. build_cutlass.sh
#   5. Megatron-LM
#   6. pip install .            (Quamba package)
#
# RTX 3070 is Ampere sm_86; restrict TORCH_CUDA_ARCH_LIST to avoid wasted
# multi-arch nvcc passes that inflated build time in earlier attempts.
set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/mnt/c/source/COREY_Transformer}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mabo1215/.adama-micromamba}"
export QUAMBA_DIR="${QUAMBA_DIR:-$REPO_ROOT/Quamba}"

# Skip already-completed phases.
export INIT_SUBMODULES=0
export INSTALL_CORE_RUNTIME=0

# Run the two remaining phases.
export BUILD_THIRD_PARTY=1
export BUILD_QUAMBA_PACKAGE=1

# Constrain nvcc to sm_86 (RTX 3070 / Ampere) to minimise compile time.
export TORCH_CUDA_ARCH_LIST="8.6"
# Limit parallel compile jobs to avoid OOM during CUTLASS build.
export MAX_JOBS="${MAX_JOBS:-4}"

cd "$REPO_ROOT"
echo "[quamba-phase2] starting third-party build + pip install"
bash src/scripts/wsl_setup_quamba_env.sh
echo "[quamba-phase2] done"
