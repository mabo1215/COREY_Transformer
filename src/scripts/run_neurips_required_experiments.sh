#!/usr/bin/env bash
# ==============================================================================
# Run the three NeurIPS-required follow-up experiments:
#
#   1. Tier-2a -> Tier-2b integrated end-to-end benchmark
#   2. Real-workload entropy diversity / dynamic chunk switching benchmark
#   3. External modern-baseline harness (Mamba2 / FlashAttention reference)
#
# Usage:
#   bash src/scripts/run_neurips_required_experiments.sh
#
# Common overrides:
#   MODEL=mamba-370m REPEATS=5 bash src/scripts/run_neurips_required_experiments.sh
#   USE_8GPU=1 NPROC_PER_NODE=8 bash src/scripts/run_neurips_required_experiments.sh
#   BASELINE_MODE=real BASELINE_MODELS=mamba2 bash src/scripts/run_neurips_required_experiments.sh
#   RUN_FA3_MATCHED=1 bash src/scripts/run_neurips_required_experiments.sh
#
# Notes:
#   - The existing external-baseline harness currently has real Mamba2 support
#     and FlashAttention literature/mock support, not a native FlashAttention-3
#     matched kernel benchmark.
#   - Run from any directory; paths are resolved relative to the repository root.
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ------------------------------------------------------------------------------
# Configuration. Override via environment variables.
# ------------------------------------------------------------------------------
MODEL="${MODEL:-mamba-370m}"
NEW_TOKENS="${NEW_TOKENS:-32}"
WARMUP="${WARMUP:-2}"
REPEATS="${REPEATS:-5}"
HETERO_WARMUP="${HETERO_WARMUP:-1}"
HETERO_REPEATS="${HETERO_REPEATS:-3}"
NUM_BINS="${NUM_BINS:-256}"

USE_8GPU="${USE_8GPU:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

DEVICE="${DEVICE:-cuda}"
TASKS="${TASKS:-narrativeqa qasper gov_report multifieldqa_en}"
MAX_PROMPTS="${MAX_PROMPTS:-20}"
BASELINE_MODE="${BASELINE_MODE:-auto}"
BASELINE_MODELS="${BASELINE_MODELS:-rwkv flashattention mamba2}"
MAMBA2_MODEL_ID="${MAMBA2_MODEL_ID:-benchang1110/mamba2-2.7b-hf}"
RUN_FA3_MATCHED="${RUN_FA3_MATCHED:-0}"
FA3_SEQ_LENS="${FA3_SEQ_LENS:-1024 2048 4096 8192}"
FA3_N_HEADS="${FA3_N_HEADS:-16}"
FA3_HEAD_DIMS="${FA3_HEAD_DIMS:-64}"
FA3_DTYPE="${FA3_DTYPE:-bf16}"
FA3_WARMUP="${FA3_WARMUP:-10}"
FA3_REPEATS="${FA3_REPEATS:-100}"
FA3_CAUSAL="${FA3_CAUSAL:-1}"

OUTPUT_BASE="${OUTPUT_BASE:-src/outputs/neurips_required}"
DATA_BASE="${DATA_BASE:-src/data/longbench_subset}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FORCE_RERUN="${FORCE_RERUN:-0}"

LOG_DIR="${REPO_ROOT}/${OUTPUT_BASE}/logs"
mkdir -p "$LOG_DIR"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_cmd() {
  local log_file="$1"
  shift
  log "Running: $*"
  "$@" >> "$log_file" 2>&1
}

mark_done() {
  local output_dir="$1"
  mkdir -p "$output_dir"
  touch "${output_dir}/.stage_done"
}

stage_done() {
  local output_dir="$1"
  [[ "$FORCE_RERUN" != "1" && -f "${output_dir}/.stage_done" ]]
}

# ------------------------------------------------------------------------------
# Experiment 1: integrated end-to-end benchmark.
# ------------------------------------------------------------------------------
run_integrated() {
  local out_dir log_file
  if [[ "$USE_8GPU" == "1" ]]; then
    out_dir="${REPO_ROOT}/${OUTPUT_BASE}/integrated_end_to_end_8gpu"
    log_file="${LOG_DIR}/01_integrated_end_to_end_8gpu.log"
    if stage_done "$out_dir"; then
      log "Experiment 1/3 already complete; skipping -> $out_dir"
      return 0
    fi
    mkdir -p "$out_dir"
    log "Experiment 1/3: integrated end-to-end benchmark (8GPU)"
    run_cmd "$log_file" torchrun --nproc_per_node="$NPROC_PER_NODE" \
      "${REPO_ROOT}/src/experiments/run_integrated_end_to_end_8gpu.py" \
      --model "$MODEL" \
      --new-tokens "$NEW_TOKENS" \
      --warmup "$WARMUP" \
      --repeats "$REPEATS" \
      --num-bins "$NUM_BINS" \
      --output-dir "$out_dir"
  else
    out_dir="${REPO_ROOT}/${OUTPUT_BASE}/integrated_end_to_end"
    log_file="${LOG_DIR}/01_integrated_end_to_end.log"
    if stage_done "$out_dir"; then
      log "Experiment 1/3 already complete; skipping -> $out_dir"
      return 0
    fi
    mkdir -p "$out_dir"
    log "Experiment 1/3: integrated end-to-end benchmark"
    run_cmd "$log_file" "$PYTHON_BIN" -m src.experiments.run_integrated_end_to_end \
      --model "$MODEL" \
      --new-tokens "$NEW_TOKENS" \
      --warmup "$WARMUP" \
      --repeats "$REPEATS" \
      --num-bins "$NUM_BINS" \
      --output-dir "$out_dir"
  fi
  mark_done "$out_dir"
  log "Experiment 1/3 complete -> $out_dir"
}

# ------------------------------------------------------------------------------
# Experiment 2: heterogeneous real-workload entropy/chunk switching.
# ------------------------------------------------------------------------------
run_heterogeneous() {
  local out_dir log_file
  if [[ "$USE_8GPU" == "1" ]]; then
    out_dir="${REPO_ROOT}/${OUTPUT_BASE}/heterogeneous_corpus_8gpu"
    log_file="${LOG_DIR}/02_heterogeneous_corpus_8gpu.log"
    if stage_done "$out_dir"; then
      log "Experiment 2/3 already complete; skipping -> $out_dir"
      return 0
    fi
    mkdir -p "$out_dir"
    log "Experiment 2/3: heterogeneous corpus chunk switching (8GPU)"
    run_cmd "$log_file" torchrun --nproc_per_node="$NPROC_PER_NODE" \
      "${REPO_ROOT}/src/experiments/run_heterogeneous_corpus_8gpu.py" \
      --model "$MODEL" \
      --num-bins "$NUM_BINS" \
      --warmup "$HETERO_WARMUP" \
      --repeats "$HETERO_REPEATS" \
      --output-dir "$out_dir"
  else
    out_dir="${REPO_ROOT}/${OUTPUT_BASE}/heterogeneous_corpus"
    log_file="${LOG_DIR}/02_heterogeneous_corpus.log"
    if stage_done "$out_dir"; then
      log "Experiment 2/3 already complete; skipping -> $out_dir"
      return 0
    fi
    mkdir -p "$out_dir"
    log "Experiment 2/3: heterogeneous corpus chunk switching"
    run_cmd "$log_file" "$PYTHON_BIN" -m src.experiments.run_heterogeneous_corpus \
      --model "$MODEL" \
      --num-bins "$NUM_BINS" \
      --warmup "$HETERO_WARMUP" \
      --repeats "$HETERO_REPEATS" \
      --output-dir "$out_dir"
  fi
  mark_done "$out_dir"
  log "Experiment 2/3 complete -> $out_dir"
}

# ------------------------------------------------------------------------------
# Experiment 3: external baselines over LongBench subsets.
# ------------------------------------------------------------------------------
run_external_baselines() {
  local log_file="${LOG_DIR}/03_external_baselines.log"
  log "Experiment 3/3: external baseline harness"
  log "Baseline models: $BASELINE_MODELS"
  log "Baseline mode: $BASELINE_MODE"

  for subset in $TASKS; do
    local data_file="${REPO_ROOT}/${DATA_BASE}/${subset}/test.jsonl"
    local out_dir="${REPO_ROOT}/${OUTPUT_BASE}/external_baselines_${subset}"
    if [[ ! -f "$data_file" ]]; then
      log "Missing data file: $data_file"
      exit 2
    fi
    mkdir -p "$out_dir"
    if stage_done "$out_dir"; then
      log "External baselines subset=${subset} already complete; skipping -> $out_dir"
      continue
    fi
    log "External baselines subset=${subset}"
    run_cmd "$log_file" "$PYTHON_BIN" "${REPO_ROOT}/src/experiments/run_external_baselines.py" \
      --models $BASELINE_MODELS \
      --model-id "$MAMBA2_MODEL_ID" \
      --data-file "$data_file" \
      --max-prompts "$MAX_PROMPTS" \
      --mode "$BASELINE_MODE" \
      --device "$DEVICE" \
      --output-dir "$out_dir"
    mark_done "$out_dir"
    log "External baselines subset=${subset} complete -> $out_dir"
  done
  log "Experiment 3/3 complete"
}

run_flashattention3_matched() {
  local log_file="${LOG_DIR}/03b_flashattention3_matched.log"
  local out_dir="${REPO_ROOT}/${OUTPUT_BASE}/flashattention3_matched"
  local causal_args=()
  if stage_done "$out_dir"; then
    log "Experiment 3b already complete; skipping -> $out_dir"
    return 0
  fi
  if [[ "$FA3_CAUSAL" == "1" ]]; then
    causal_args+=(--causal)
  fi

  mkdir -p "$out_dir"
  log "Experiment 3b: FlashAttention-3 matched H800/H100 benchmark"
  run_cmd "$log_file" "$PYTHON_BIN" -m src.experiments.run_flashattention3_matched_benchmark \
    --seq-lens $FA3_SEQ_LENS \
    --batch-size 1 \
    --n-heads $FA3_N_HEADS \
    --head-dims $FA3_HEAD_DIMS \
    --dtype "$FA3_DTYPE" \
    --warmup "$FA3_WARMUP" \
    --repeats "$FA3_REPEATS" \
    --device "$DEVICE" \
    --output-dir "$out_dir" \
    "${causal_args[@]}"
  mark_done "$out_dir"
  log "Experiment 3b complete -> $out_dir"
}

main() {
  local start end elapsed
  start="$(date +%s)"

  log "============================================================"
  log "NeurIPS required experiment automation"
  log "Repo       : $REPO_ROOT"
  log "Output     : ${REPO_ROOT}/${OUTPUT_BASE}"
  log "Model      : $MODEL"
  log "Use 8GPU   : $USE_8GPU"
  log "Tasks      : $TASKS"
  log "Logs       : $LOG_DIR"
  log "============================================================"

  run_integrated
  run_heterogeneous
  run_external_baselines
  if [[ "$RUN_FA3_MATCHED" == "1" ]]; then
    run_flashattention3_matched
  fi

  end="$(date +%s)"
  elapsed="$((end - start))"
  log "============================================================"
  log "All three experiments completed in ${elapsed}s"
  log "Results: ${REPO_ROOT}/${OUTPUT_BASE}"
  log "Logs   : $LOG_DIR"
  log "============================================================"
}

main "$@"
