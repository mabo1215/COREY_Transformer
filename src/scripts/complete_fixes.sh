#!/bin/bash
# One-click fix and verification script for Policy_Corey and Quamba
# Usage: bash complete_fixes.sh [task]
# Task options: all (default) | corey-fix | corey-rerun | quamba-verify

set -euo pipefail

REPO_ROOT="${1:-.}"
TASK="${2:-all}"

# Environment configuration
export REPO_ROOT="/mnt/c/source/Corey_Transformer"
export MAMBA_ROOT_PREFIX="/home/bobma-resideo/.corey-micromamba"
export ENV_NAME="corey-cuda128"
export HF_HOME="/mnt/c/Users/295461/.cache/huggingface"
export TORCH_CUDA_ARCH_LIST="8.9"
export CUDAARCHS="89"

MM="/home/bobma-resideo/.corey-wsl-tools/bin/micromamba"

if [[ ! -x "$MM" ]]; then
    echo "[error] micromamba not found at $MM"
    exit 1
fi

print_header() {
    echo ""
    echo "======================================"
    echo "  $1"
    echo "======================================"
    echo ""
}

step_install_einops() {
    print_header "[STEP] Installing einops in $ENV_NAME"
    
    echo "[check] Current mamba_ssm status..."
    $MM run -r "$MAMBA_ROOT_PREFIX" -n "$ENV_NAME" python -c \
        "try:
    import mamba_ssm; print(f'mamba_ssm version: {mamba_ssm.__version__}')
except ImportError as e: print(f'mamba_ssm import failed: {e}')" 2>&1 || true
    
    echo "[install] Installing einops..."
    $MM run -r "$MAMBA_ROOT_PREFIX" -n "$ENV_NAME" pip install -q einops 2>&1 | grep -v "already satisfied" || true
    
    echo "[verify] Verifying einops..."
    $MM run -r "$MAMBA_ROOT_PREFIX" -n "$ENV_NAME" python -c "import einops; print(f'✓ einops {einops.__version__} OK')"
}

step_rerun_corey() {
    print_header "[STEP] Re-running policy_corey matrix"
    
    cd "$REPO_ROOT"
    
    export SCHEDULER_POLICY=corey
    export MAX_SAMPLES=5
    export OUTPUT_DIR="src/outputs/revision_matrix_4task5_policy_corey_fixed"
    export EVAL_PERPLEXITY=1
    
    for model in mamba-370m mamba-1.4b mamba-2.8b; do
        export MODELS="$model"
        if [[ "$model" == "mamba-2.8b" ]]; then
            export EVAL_PERPLEXITY=0
        fi
        echo "[exec] Executing model=$model with policy_corey..."
        bash src/scripts/wsl_run_checkpoint_matrix.sh || echo "[warn] model=$model encountered issues"
    done
    
    echo "[summary] Policy_corey execution complete. Results in: $OUTPUT_DIR"
    if [[ -f "$OUTPUT_DIR/aggregate_summary.csv" ]]; then
        echo "[info] Aggregate summary:"
        cat "$OUTPUT_DIR/aggregate_summary.csv" | head -5
    fi
}

step_quamba_verify() {
    print_header "[STEP] Verifying Quamba build chain"
    
    cd "$REPO_ROOT"
    
    # Check if Quamba directory exists
    if [[ ! -d "Quamba" ]]; then
        echo "[error] Quamba directory not found. Clone with: git clone https://github.com/enyac-group/Quamba.git"
        return 1
    fi
    
    echo "[check] Verifying Quamba environment structure..."
    echo "[info] Running wsl_run_quamba_phase2.sh..."
    export INIT_SUBMODULES=0
    export INSTALL_CORE_RUNTIME=0
    export BUILD_THIRD_PARTY=1
    export BUILD_QUAMBA_PACKAGE=1
    export MAX_JOBS=2
    
    bash src/scripts/wsl_run_quamba_phase2.sh 2>&1 | tee "src/outputs/quamba_complete_verification.log"
    
    if [[ $? -eq 0 ]]; then
        echo "[success] ✓ Quamba build verification succeeded"
    else
        echo "[warn] ⚠ Quamba build had issues. Check log: src/outputs/quamba_complete_verification.log"
    fi
}

# Main execution
case "$TASK" in
    all)
        step_install_einops
        step_rerun_corey
        step_quamba_verify
        print_header "[COMPLETE] All tasks finished"
        echo "Next: Review output results and backfill paper/appendix.tex"
        ;;
    corey-fix)
        step_install_einops
        ;;
    corey-rerun)
        step_rerun_corey
        ;;
    quamba-verify)
        step_quamba_verify
        ;;
    *)
        echo "Usage: $0 [task]"
        echo "Tasks: all (default) | corey-fix | corey-rerun | quamba-verify"
        exit 1
        ;;
esac
