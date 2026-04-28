#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
OUTPUT_BASE="${OUTPUT_BASE:-src/outputs/rtx4090_closure_subset_$(date +%Y%m%d_%H%M%S)}"
MIN_GPUS="${MIN_GPUS:-4}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/autodl-tmp/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/autodl-tmp/cache/triton}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${REPO_ROOT}/${OUTPUT_BASE}/logs"
wait_log="${REPO_ROOT}/${OUTPUT_BASE}/logs/wait_for_${MIN_GPUS}gpu.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] waiting for ${MIN_GPUS} GPUs" | tee -a "$wait_log"
echo "repo=${REPO_ROOT}" | tee -a "$wait_log"
echo "output=${REPO_ROOT}/${OUTPUT_BASE}" | tee -a "$wait_log"

while true; do
  gpu_count=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] visible_gpus=${gpu_count}" | tee -a "$wait_log"
  if [[ "$gpu_count" -ge "$MIN_GPUS" ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting 4-GPU closure subset" | tee -a "$wait_log"
PYTHON_BIN="$PYTHON_BIN" \
OUTPUT_BASE="$OUTPUT_BASE" \
MAX_SAMPLES="$MAX_SAMPLES" \
  bash "${REPO_ROOT}/src/scripts/run_4x4090_closure_subset.sh" \
  2>&1 | tee -a "${REPO_ROOT}/${OUTPUT_BASE}/logs/main_4gpu_subset.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 4-GPU closure subset finished" | tee -a "$wait_log"
