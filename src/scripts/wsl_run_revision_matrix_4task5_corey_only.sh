#!/usr/bin/env bash
# Run only the missing policy_corey leg of the 4task5 revision matrix.
# policy_off and policy_static already have outputs; this fills the gap.
set -euo pipefail

DEFAULT_REPO_ROOT="/mnt/c/source/Corey_Transformer"
if [[ ! -d "$DEFAULT_REPO_ROOT" && -d "/mnt/c/source/COREY_Transformer" ]]; then
  DEFAULT_REPO_ROOT="/mnt/c/source/COREY_Transformer"
fi

export REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
cd "$REPO_ROOT"

export MODES="longbench benchmark"
export TASKS="narrativeqa qasper multifieldqa_en gov_report"
export LM_DATASETS="wikitext103 pg19"
export DATASET_SOURCE=local
export DATASET_ROOT="$REPO_ROOT/src/data/longbench_subset"
export MAX_SAMPLES=5
export LM_MAX_SAMPLES=5
export PPL_MAX_SAMPLES=5
export MAX_LENGTH=4096
export BENCHMARK_TASK=narrativeqa
export BENCHMARK_MAX_SAMPLES=1
export BENCHMARK_MAX_NEW_TOKENS=32
export WARMUP_RUNS=1
export BENCHMARK_REPEATS=1
export DEVICE=cuda
export DTYPE=float16
export COLLECT_ENERGY=1
export ENERGY_GPU_INDEX=0

export SCHEDULER_POLICY=corey
export DISABLE_ENTROPY_HOOK=0
export STATIC_TILE_SIZE=256
export OUTPUT_DIR="src/outputs/revision_matrix_4task5_policy_corey"

for model in mamba-370m mamba-1.4b mamba-2.8b; do
  export MODELS="$model"
  if [[ "$model" == "mamba-2.8b" ]]; then
    # Disable per-sample PPL for 2.8b to keep wall-clock manageable.
    export EVAL_PERPLEXITY=0
  else
    export EVAL_PERPLEXITY=1
  fi
  echo "[corey-only] policy=corey model=$model eval_ppl=$EVAL_PERPLEXITY output=$OUTPUT_DIR"
  bash src/scripts/wsl_run_checkpoint_matrix.sh
  echo "[corey-only] done model=$model"
done

# Final consolidation: regenerate aggregate_summary.csv across all three models.
export MODELS="mamba-370m mamba-1.4b mamba-2.8b"
export EVAL_PERPLEXITY=1
echo "[corey-only] consolidating aggregate across all models"
bash src/scripts/wsl_run_checkpoint_matrix.sh

echo "[corey-only] revision_matrix_4task5_policy_corey complete"
