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

DEFAULT_REPO_ROOT="/mnt/c/source/Corey_Transformer"
if [[ ! -d "$DEFAULT_REPO_ROOT" && -d "/mnt/c/source/COREY_Transformer" ]]; then
	DEFAULT_REPO_ROOT="/mnt/c/source/COREY_Transformer"
fi

export REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
if [[ -z "${MAMBA_ROOT_PREFIX:-}" ]]; then
	if [[ -d "${HOME}/.adama-micromamba" ]]; then
		export MAMBA_ROOT_PREFIX="${HOME}/.adama-micromamba"
	elif [[ -d "${HOME}/.corey-micromamba" ]]; then
		export MAMBA_ROOT_PREFIX="${HOME}/.corey-micromamba"
	else
		export MAMBA_ROOT_PREFIX="${HOME}/.adama-micromamba"
	fi
else
	export MAMBA_ROOT_PREFIX
fi
export QUAMBA_DIR="${QUAMBA_DIR:-$REPO_ROOT/Quamba}"

# Skip already-completed phases.
export INIT_SUBMODULES=0
export INSTALL_CORE_RUNTIME=0

# Run the two remaining phases.
export BUILD_THIRD_PARTY=1
export BUILD_QUAMBA_PACKAGE=1

# Constrain nvcc to a single architecture to minimise compile time.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
# Limit parallel compile jobs to avoid OOM during CUTLASS build.
export MAX_JOBS="${MAX_JOBS:-4}"

cd "$REPO_ROOT"
echo "[quamba-phase2] starting third-party build + pip install"
bash src/scripts/wsl_setup_quamba_env.sh
echo "[quamba-phase2] done"
