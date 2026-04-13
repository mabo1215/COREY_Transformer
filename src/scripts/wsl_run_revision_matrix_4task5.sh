#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/source/COREY_Transformer

export REPO_ROOT=/mnt/c/source/COREY_Transformer
export MAMBA_ROOT_PREFIX=/home/mabo1215/.adama-micromamba
export ENV_NAME=adama-cuda128
export MODES="longbench benchmark"
export MODELS="mamba-370m mamba-1.4b mamba-2.8b"
export TASKS="narrativeqa qasper multifieldqa_en gov_report"
export LM_DATASETS="wikitext103 pg19"
export DATASET_SOURCE=local
export DATASET_ROOT=/mnt/c/source/COREY_Transformer/src/data/longbench_subset
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

for policy in off static corey; do
  export SCHEDULER_POLICY="$policy"
  export STATIC_TILE_SIZE=256
  if [[ "$policy" == "off" ]]; then
    export DISABLE_ENTROPY_HOOK=1
  else
    export DISABLE_ENTROPY_HOOK=0
  fi
  export OUTPUT_DIR="src/outputs/revision_matrix_4task5_policy_${policy}"

  for model in mamba-370m mamba-1.4b mamba-2.8b; do
    export MODELS="$model"
    if [[ "$model" == "mamba-2.8b" ]]; then
      export EVAL_PERPLEXITY=0
    else
      export EVAL_PERPLEXITY=1
    fi
    echo "[revision-matrix] running policy=$policy model=$model eval_ppl=$EVAL_PERPLEXITY output=$OUTPUT_DIR"
    bash src/scripts/wsl_run_checkpoint_matrix.sh
    echo "[revision-matrix] completed policy=$policy model=$model"
  done

  # Rebuild a full aggregate summary across all models for this policy.
  export MODELS="mamba-370m mamba-1.4b mamba-2.8b"
  export EVAL_PERPLEXITY=1
  echo "[revision-matrix] consolidating policy=$policy aggregate across all models"
  bash src/scripts/wsl_run_checkpoint_matrix.sh

  echo "[revision-matrix] completed policy=$policy"
done

echo "[revision-matrix] all policies completed"
