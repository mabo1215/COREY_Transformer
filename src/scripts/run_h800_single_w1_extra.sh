#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-/root/miniconda3/bin/python}"
OUT="${OUTPUT_BASE:-src/outputs/h800_single_w1_extra_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${REPO_ROOT}/${OUT}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/autodl-tmp/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/autodl-tmp/cache/triton}"

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
  log "DONE ${name} exit=${status}"
}

log "H800 extra W1 kernel bundle"
log "output=${OUT_DIR}"
nvidia-smi > "$OUT_DIR/nvidia_smi_start.txt"

run_stage "01_triplet_bf16_4096" \
  "$PY" src/experiments/run_w1_triton_triplet.py \
    --batch-size 1 --dim 1024 --seq-len 4096 --d-state 16 \
    --dtype bfloat16 --warmup-runs 5 --benchmark-repeats 30 \
    --output-dir "$OUT_DIR/triplet_bf16_4096"

run_stage "02_sweep_bf16_4096" \
  "$PY" src/experiments/run_w1_chunk_sweep.py \
    --batch-size 1 --dim 1024 --seq-len 4096 --d-state 16 \
    --dtype bfloat16 --warmup-runs 5 --benchmark-repeats 30 \
    --sweep-chunks 32 64 128 256 512 \
    --output-dir "$OUT_DIR/sweep_bf16_4096"

run_stage "03_perturb_bf16_4096" \
  "$PY" src/experiments/run_w1_perturbation.py \
    --batch-size 1 --dim 1024 --seq-len 4096 --d-state 16 \
    --dtype bfloat16 --warmup-runs 5 --benchmark-repeats 30 \
    --output-dir "$OUT_DIR/perturb_bf16_4096"

run_stage "04_triplet_fp16_8192" \
  "$PY" src/experiments/run_w1_triton_triplet.py \
    --batch-size 1 --dim 1024 --seq-len 8192 --d-state 16 \
    --dtype float16 --warmup-runs 5 --benchmark-repeats 30 \
    --output-dir "$OUT_DIR/triplet_fp16_8192"

run_stage "05_sweep_fp16_8192" \
  "$PY" src/experiments/run_w1_chunk_sweep.py \
    --batch-size 1 --dim 1024 --seq-len 8192 --d-state 16 \
    --dtype float16 --warmup-runs 5 --benchmark-repeats 30 \
    --sweep-chunks 32 64 128 256 512 \
    --output-dir "$OUT_DIR/sweep_fp16_8192"

nvidia-smi > "$OUT_DIR/nvidia_smi_end.txt" || true
log "ALL_DONE"
