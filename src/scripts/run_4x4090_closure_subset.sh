#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_BASE="${OUTPUT_BASE:-src/outputs/rtx4090_closure_subset}"
DATA_BASE="${DATA_BASE:-src/data/longbench_subset}"
TASKS_ARRAY=(${TASKS:-narrativeqa qasper gov_report multifieldqa_en})
MAX_SAMPLES="${MAX_SAMPLES:-20}"
MAMBA2_MODEL_ID="${MAMBA2_MODEL_ID:-benchang1110/mamba2-2.7b-hf}"
MODEL="${MODEL:-mamba-370m}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
LOG_DIR="${REPO_ROOT}/${OUTPUT_BASE}/logs"
mkdir -p "$LOG_DIR"

echo "Launching 4x4090-compatible closure subset."
echo "Note: FlashAttention-3 full Transformer is skipped on RTX 4090 because it is sm_89, not Hopper/sm_90."

pids=()

for index in "${!TASKS_ARRAY[@]}"; do
  task="${TASKS_ARRAY[$index]}"
  gpu="$index"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON_BIN" "$REPO_ROOT/src/experiments/run_hf_longbench_baseline.py" \
      --model-name "mamba2-ssd-2.7b-${task}" \
      --model-id "$MAMBA2_MODEL_ID" \
      --dataset-root "$REPO_ROOT/$DATA_BASE" \
      --tasks "$task" \
      --max-samples "$MAX_SAMPLES" \
      --device cuda \
      --dtype float16 \
      --trust-remote-code \
      --output-dir "$REPO_ROOT/$OUTPUT_BASE/mamba2_ssd"
  ) > "$LOG_DIR/mamba2_${task}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

(
  export CUDA_VISIBLE_DEVICES=0
  "$PYTHON_BIN" "$REPO_ROOT/src/experiments/run_real_workload_diversity_h800.py" \
    --model "$MODEL" \
    --dataset-root "$REPO_ROOT/$DATA_BASE" \
    --tasks "${TASKS_ARRAY[@]}" \
    --samples-per-task "$MAX_SAMPLES" \
    --output-dir "$REPO_ROOT/$OUTPUT_BASE/real_workload_diversity"
) > "$LOG_DIR/real_workload_diversity.log" 2>&1 || status=1

exit "$status"
