#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
OUTPUT_BASE="${OUTPUT_BASE:-src/outputs/h800_single_w1_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-mamba-370m}"
DISPATCH_MODULE="${DISPATCH_MODULE:-src.corey_selective_scan_dispatch}"
WARMUP="${WARMUP:-5}"
REPEATS="${REPEATS:-30}"
INTEGRATED_REPEATS="${INTEGRATED_REPEATS:-20}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/autodl-tmp/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/autodl-tmp/cache/triton}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

OUT_DIR="${REPO_ROOT}/${OUTPUT_BASE}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/main.log"
}

run_stage() {
  local name="$1"
  shift
  log "START ${name}: $*"
  set +e
  "$@" > "$LOG_DIR/${name}.log" 2>&1
  local status=$?
  set -e
  echo "$status" > "$OUT_DIR/${name}.exitcode"
  if [[ "$status" -eq 0 ]]; then
    log "DONE ${name}"
  else
    log "FAILED ${name} exit=${status}; continuing"
  fi
}

log "H800 single-card W1 experiment bundle"
log "repo=${REPO_ROOT}"
log "output=${OUT_DIR}"
log "python=${PYTHON_BIN}"
log "dispatch=${DISPATCH_MODULE}"
nvidia-smi | tee "$OUT_DIR/nvidia_smi_start.txt" >/dev/null

run_stage "00_dispatch_probe" \
  "$PYTHON_BIN" -m src.experiments.run_integrated_multiblock_dispatch \
    --dispatch-module "$DISPATCH_MODULE" \
    --output-dir "$OUT_DIR/dispatch_probe"

run_stage "01_integrated_dispatch_wrapper" \
  "$PYTHON_BIN" -m src.experiments.run_integrated_end_to_end \
    --model "$MODEL" \
    --new-tokens 32 \
    --warmup 2 \
    --repeats "$INTEGRATED_REPEATS" \
    --num-bins 256 \
    --selective-scan-dispatch-module "$DISPATCH_MODULE" \
    --output-dir "$OUT_DIR/integrated_dispatch_wrapper"

run_stage "02_w1_triton_triplet_fp16" \
  "$PYTHON_BIN" src/experiments/run_w1_triton_triplet.py \
    --batch-size 1 \
    --dim 1024 \
    --seq-len 4096 \
    --d-state 16 \
    --dtype float16 \
    --warmup-runs "$WARMUP" \
    --benchmark-repeats "$REPEATS" \
    --output-dir "$OUT_DIR/w1_triton_triplet_fp16"

run_stage "03_w1_chunk_sweep_fp16" \
  "$PYTHON_BIN" src/experiments/run_w1_chunk_sweep.py \
    --batch-size 1 \
    --dim 1024 \
    --seq-len 4096 \
    --d-state 16 \
    --dtype float16 \
    --warmup-runs "$WARMUP" \
    --benchmark-repeats "$REPEATS" \
    --sweep-chunks 32 64 128 256 512 \
    --output-dir "$OUT_DIR/w1_chunk_sweep_fp16"

run_stage "04_w1_perturbation_fp16" \
  "$PYTHON_BIN" src/experiments/run_w1_perturbation.py \
    --batch-size 1 \
    --dim 1024 \
    --seq-len 4096 \
    --d-state 16 \
    --dtype float16 \
    --warmup-runs "$WARMUP" \
    --benchmark-repeats "$REPEATS" \
    --output-dir "$OUT_DIR/w1_perturbation_fp16"

run_stage "05_active_hook_integration_n20" \
  "$PYTHON_BIN" -m src.experiments.run_active_hook_integration \
    --model "$MODEL" \
    --new-tokens 32 \
    --warmup 2 \
    --repeats "$INTEGRATED_REPEATS" \
    --num-bins 256 \
    --output-dir "$OUT_DIR/active_hook_integration_n20"

nvidia-smi | tee "$OUT_DIR/nvidia_smi_end.txt" >/dev/null || true
log "All queued H800 single-card stages finished. Check *.exitcode and logs/."
