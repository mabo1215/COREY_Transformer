#!/bin/bash
# Combined fix script for the missing einops dependency and policy_corey reruns

set -euo pipefail

REPO_ROOT="${1:-.}"
cd "$REPO_ROOT"

export MAMBA_ROOT_PREFIX="/home/bobma-resideo/.corey-micromamba"
export ENV_NAME="corey-cuda128"
export HF_HOME="/mnt/c/Users/295461/.cache/huggingface"
export TORCH_CUDA_ARCH_LIST="8.9"
export CUDAARCHS="89"
export MAMBA_GENCODE_FLAGS="-gencode arch=compute_89,code=sm_89"

MM="/home/bobma-resideo/.corey-wsl-tools/bin/micromamba"

if [[ ! -x "$MM" ]]; then
    echo "[error] micromamba not found at $MM"
    exit 1
fi

echo "[step 1] Installing einops in $ENV_NAME..."
$MM run -r "$MAMBA_ROOT_PREFIX" -n "$ENV_NAME" pip install -q einops 2>&1 | grep -v "already satisfied" || true

echo "[step 2] Verifying einops installation..."
$MM run -r "$MAMBA_ROOT_PREFIX" -n "$ENV_NAME" python -c "import einops; print(f'einops {einops.__version__} OK')" || exit 1

echo "[step 3] Re-running policy_corey matrix with fixed environment..."
export SCHEDULER_POLICY=corey
export MAX_SAMPLES=5
export OUTPUT_DIR="src/outputs/revision_matrix_4task5_policy_corey_fixed"
export EVAL_PERPLEXITY=1

for model in mamba-370m mamba-1.4b mamba-2.8b; do
    export MODELS="$model"
    if [[ "$model" == "mamba-2.8b" ]]; then
        export EVAL_PERPLEXITY=0
    fi
    echo "[exec] Running model=$model..."
    bash src/scripts/wsl_run_checkpoint_matrix.sh || echo "[warning] model=$model encountered errors"
done

echo "[success] Policy_corey execution complete. Results in: $OUTPUT_DIR"
