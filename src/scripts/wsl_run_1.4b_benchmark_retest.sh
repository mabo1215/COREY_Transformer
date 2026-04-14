#!/usr/bin/env bash
# Minimum retest for mamba-1.4b policy_corey benchmark latency.
# Runs benchmark-only (no LongBench tasks) with repeats>=3 to get a stable value.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/c/source/Corey_Transformer}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/bobma-resideo/.corey-micromamba}"
ENV_NAME="${ENV_NAME:-corey-cuda128}"
HF_HOME="${HF_HOME:-/mnt/c/Users/295461/.cache/huggingface}"

cd "$REPO_ROOT"

export REPO_ROOT
export MAMBA_ROOT_PREFIX
export ENV_NAME
export HF_HOME

export MODES="benchmark"
export MODELS="mamba-1.4b"
export TASKS="narrativeqa"
export LM_DATASETS=""
export DATASET_SOURCE="local"
export DATASET_ROOT="$REPO_ROOT/src/data/longbench_subset"
export MAX_SAMPLES=1
export LM_MAX_SAMPLES=0
export PPL_MAX_SAMPLES=0
export MAX_LENGTH=4096
export BENCHMARK_TASK="narrativeqa"
export BENCHMARK_MAX_SAMPLES=1
export BENCHMARK_MAX_NEW_TOKENS=32
export WARMUP_RUNS=2
export BENCHMARK_REPEATS=3
export DEVICE="cuda"
export DTYPE="float16"
export COLLECT_ENERGY=1
export ENERGY_GPU_INDEX=0
export SCHEDULER_POLICY="corey"
export STATIC_TILE_SIZE=256
export DISABLE_ENTROPY_HOOK=0
export EVAL_PERPLEXITY=0
export OUTPUT_DIR="src/outputs/revision_matrix_1.4b_benchmark_retest"

echo "[1.4b_benchmark_retest] start output=$OUTPUT_DIR warmup=$WARMUP_RUNS repeats=$BENCHMARK_REPEATS"
bash src/scripts/wsl_run_checkpoint_matrix.sh
echo "[1.4b_benchmark_retest] done"
