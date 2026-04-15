#!/usr/bin/env bash
set -euo pipefail

# W1 triplet runner: executes the same matrix config for off/static/corey policies
# and then builds a compact comparison table.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-src/outputs}"
RUN_TAG="${RUN_TAG:-revision_matrix_w1_triplet}"
POLICIES=(off static corey)

# Common matrix settings (override via env vars if needed)
export MODES="${MODES:-longbench benchmark}"
export MODELS="${MODELS:-mamba-370m mamba-1.4b}"
export PRECISIONS="${PRECISIONS:-fp16}"
export TASKS="${TASKS:-narrativeqa qasper multifieldqa_en gov_report}"
export MAX_SAMPLES="${MAX_SAMPLES:-5}"
export BENCHMARK_MAX_SAMPLES="${BENCHMARK_MAX_SAMPLES:-1}"
export BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-3}"
export WARMUP_RUNS="${WARMUP_RUNS:-1}"
export LM_DATASETS="${LM_DATASETS:-wikitext103}"
export LM_MAX_SAMPLES="${LM_MAX_SAMPLES:-5}"
export PPL_MAX_SAMPLES="${PPL_MAX_SAMPLES:-5}"
export EVAL_PERPLEXITY="${EVAL_PERPLEXITY:-1}"
export DATASET_SOURCE="${DATASET_SOURCE:-local}"
export DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/src/data/longbench_subset}"
export DEVICE="${DEVICE:-cuda}"
export DTYPE="${DTYPE:-float16}"

cd "$REPO_ROOT"

for policy in "${POLICIES[@]}"; do
  echo "[W1] Running policy: ${policy}"
  export SCHEDULER_POLICY="$policy"
  export OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_TAG}_${policy}"
  bash "$REPO_ROOT/src/scripts/wsl_run_checkpoint_matrix.sh"
done

python -m src.experiments.build_w1_policy_comparison \
  --off-root "${BASE_OUTPUT_DIR}/${RUN_TAG}_off" \
  --static-root "${BASE_OUTPUT_DIR}/${RUN_TAG}_static" \
  --corey-root "${BASE_OUTPUT_DIR}/${RUN_TAG}_corey" \
  --output-csv "${BASE_OUTPUT_DIR}/${RUN_TAG}_comparison.csv" \
  --output-json "${BASE_OUTPUT_DIR}/${RUN_TAG}_comparison.json"

echo "[W1] Completed. Comparison written to ${BASE_OUTPUT_DIR}/${RUN_TAG}_comparison.csv"
