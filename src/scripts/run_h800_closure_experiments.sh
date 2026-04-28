#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_BASE="${OUTPUT_BASE:-src/outputs/h800_closure}"
DATA_BASE="${DATA_BASE:-src/data/longbench_subset}"
TASKS="${TASKS:-narrativeqa qasper gov_report multifieldqa_en}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
MODEL="${MODEL:-mamba-370m}"
MAMBA2_MODEL_ID="${MAMBA2_MODEL_ID:-benchang1110/mamba2-2.7b-hf}"
TRANSFORMER_MODEL_ID="${TRANSFORMER_MODEL_ID:-EleutherAI/pythia-1.4b}"
TRANSFORMER_MODEL_NAME="${TRANSFORMER_MODEL_NAME:-pythia-1.4b-fa3}"
DISPATCH_MODULE="${DISPATCH_MODULE:-}"
RUN_MULTIBLOCK_PROBE="${RUN_MULTIBLOCK_PROBE:-1}"
RUN_FA3_BASELINE="${RUN_FA3_BASELINE:-1}"
RUN_MAMBA2_BASELINE="${RUN_MAMBA2_BASELINE:-1}"
RUN_DIVERSITY="${RUN_DIVERSITY:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
LOG_DIR="${REPO_ROOT}/${OUTPUT_BASE}/logs"
mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

stage_done() {
  [[ "$FORCE_RERUN" != "1" && -f "$1/.stage_done" ]]
}

mark_done() {
  mkdir -p "$1"
  touch "$1/.stage_done"
}

run_logged() {
  local log_file="$1"
  shift
  log "Running: $*"
  "$@" >> "$log_file" 2>&1
}

if [[ "$RUN_MULTIBLOCK_PROBE" == "1" ]]; then
  out_dir="${REPO_ROOT}/${OUTPUT_BASE}/multiblock_dispatch"
  if ! stage_done "$out_dir"; then
    args=()
    if [[ -n "$DISPATCH_MODULE" ]]; then
      args+=(--dispatch-module "$DISPATCH_MODULE")
    fi
    run_logged "$LOG_DIR/01_multiblock_dispatch.log" "$PYTHON_BIN" \
      "$REPO_ROOT/src/experiments/run_integrated_multiblock_dispatch.py" \
      --model "$MODEL" --output-dir "$out_dir" "${args[@]}"
    mark_done "$out_dir"
  fi
  if [[ -n "$DISPATCH_MODULE" ]]; then
    out_dir="${REPO_ROOT}/${OUTPUT_BASE}/integrated_multiblock_end_to_end"
    if ! stage_done "$out_dir"; then
      run_logged "$LOG_DIR/01b_integrated_multiblock_end_to_end.log" "$PYTHON_BIN" \
        "$REPO_ROOT/src/experiments/run_integrated_end_to_end.py" \
        --model "$MODEL" \
        --new-tokens 32 \
        --warmup 2 \
        --repeats 10 \
        --num-bins 256 \
        --selective-scan-dispatch-module "$DISPATCH_MODULE" \
        --output-dir "$out_dir"
      mark_done "$out_dir"
    fi
  fi
fi

if [[ "$RUN_DIVERSITY" == "1" ]]; then
  out_dir="${REPO_ROOT}/${OUTPUT_BASE}/real_workload_diversity"
  if ! stage_done "$out_dir"; then
    run_logged "$LOG_DIR/02_real_workload_diversity.log" "$PYTHON_BIN" \
      "$REPO_ROOT/src/experiments/run_real_workload_diversity_h800.py" \
      --model "$MODEL" \
      --dataset-root "$REPO_ROOT/$DATA_BASE" \
      --tasks $TASKS \
      --samples-per-task "$MAX_SAMPLES" \
      --output-dir "$out_dir"
    mark_done "$out_dir"
  fi
fi

if [[ "$RUN_FA3_BASELINE" == "1" ]]; then
  out_dir="${REPO_ROOT}/${OUTPUT_BASE}/transformer_fa3_longbench"
  if ! stage_done "$out_dir"; then
    run_logged "$LOG_DIR/03_transformer_fa3_longbench.log" "$PYTHON_BIN" \
      "$REPO_ROOT/src/experiments/run_hf_longbench_baseline.py" \
      --model-name "$TRANSFORMER_MODEL_NAME" \
      --model-id "$TRANSFORMER_MODEL_ID" \
      --dataset-root "$REPO_ROOT/$DATA_BASE" \
      --tasks $TASKS \
      --max-samples "$MAX_SAMPLES" \
      --device cuda \
      --dtype bfloat16 \
      --attn-implementation flash_attention_3 \
      --output-dir "$out_dir"
    mark_done "$out_dir"
  fi
fi

if [[ "$RUN_MAMBA2_BASELINE" == "1" ]]; then
  out_dir="${REPO_ROOT}/${OUTPUT_BASE}/mamba2_ssd_longbench"
  if ! stage_done "$out_dir"; then
    run_logged "$LOG_DIR/04_mamba2_ssd_longbench.log" "$PYTHON_BIN" \
      "$REPO_ROOT/src/experiments/run_hf_longbench_baseline.py" \
      --model-name mamba2-ssd-2.7b \
      --model-id "$MAMBA2_MODEL_ID" \
      --dataset-root "$REPO_ROOT/$DATA_BASE" \
      --tasks $TASKS \
      --max-samples "$MAX_SAMPLES" \
      --device cuda \
      --dtype bfloat16 \
      --trust-remote-code \
      --output-dir "$out_dir"
    mark_done "$out_dir"
  fi
fi

log "H800 closure experiments submitted/completed under ${REPO_ROOT}/${OUTPUT_BASE}"
