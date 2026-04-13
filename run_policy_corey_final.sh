#!/bin/bash
set -e
cd /mnt/c/source/Corey_Transformer

export SCHEDULER_POLICY=corey
export MAX_SAMPLES=5
export OUTPUT_DIR="src/outputs/revision_matrix_4task5_policy_corey_final"
export EVAL_PERPLEXITY=1

echo "[step 1] Running policy_corey for mamba-370m..."
export MODELS="mamba-370m"
bash src/scripts/wsl_run_checkpoint_matrix.sh
echo "[checkpoint] mamba-370m completed"

echo ""
echo "[step 2] Running policy_corey for mamba-1.4b..."
export MODELS="mamba-1.4b"
bash src/scripts/wsl_run_checkpoint_matrix.sh
echo "[checkpoint] mamba-1.4b completed"

echo ""
echo "[step 3] Running policy_corey for mamba-2.8b (no perplexity eval)..."
export MODELS="mamba-2.8b"
export EVAL_PERPLEXITY=0
bash src/scripts/wsl_run_checkpoint_matrix.sh
echo "[checkpoint] mamba-2.8b completed"

echo ""
echo "[complete] All policy_corey executions finished. Results in: $OUTPUT_DIR"
