#!/usr/bin/env bash
# ==============================================================================
# FlashAttention-3 matched benchmark launcher for H800/H100.
#
# Recommended machine: H800 80GB, CUDA toolkit 12.8, PyTorch CUDA build.
#
# First-time setup on a fresh node:
#   INSTALL_FA3=1 bash src/scripts/run_fa3_h800_matched_benchmark.sh
#
# Benchmark only:
#   bash src/scripts/run_fa3_h800_matched_benchmark.sh
#
# Useful overrides:
#   CUDA_VISIBLE_DEVICES=0 DTYPE=bf16 REPEATS=100 \
#     bash src/scripts/run_fa3_h800_matched_benchmark.sh
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/src/outputs/flashattention3_matched_h800}"
SEQ_LENS="${SEQ_LENS:-1024 2048 4096 8192}"
BATCH_SIZE="${BATCH_SIZE:-1}"
N_HEADS="${N_HEADS:-16}"
HEAD_DIMS="${HEAD_DIMS:-64}"
DTYPE="${DTYPE:-bf16}"
WARMUP="${WARMUP:-10}"
REPEATS="${REPEATS:-100}"
CAUSAL="${CAUSAL:-1}"
INSTALL_FA3="${INSTALL_FA3:-0}"
FA3_SRC_DIR="${FA3_SRC_DIR:-${HOME}/flash-attention}"
MAX_JOBS="${MAX_JOBS:-8}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

if [[ "$INSTALL_FA3" == "1" ]]; then
  log "Installing FlashAttention-3 from Dao-AILab/flash-attention hopper/"
  if [[ ! -d "$FA3_SRC_DIR/.git" ]]; then
    git clone https://github.com/Dao-AILab/flash-attention "$FA3_SRC_DIR"
  fi
  git -C "$FA3_SRC_DIR" pull --ff-only || true
  "$PYTHON_BIN" -m pip install -U pip packaging psutil ninja
  (
    cd "$FA3_SRC_DIR/hopper"
    MAX_JOBS="$MAX_JOBS" "$PYTHON_BIN" setup.py install
  )
fi

log "Environment probe"
"$PYTHON_BIN" - <<'PY'
import importlib
import torch

print("torch", torch.__version__, "torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
try:
    mod = importlib.import_module("flash_attn_interface")
    print("flash_attn_interface", getattr(mod, "__file__", "loaded"))
except Exception as exc:
    print("flash_attn_interface import failed:", repr(exc))
PY

mkdir -p "$OUTPUT_DIR"

extra_args=()
if [[ "$CAUSAL" == "1" ]]; then
  extra_args+=(--causal)
fi

log "Running FA3 matched benchmark -> $OUTPUT_DIR"
"$PYTHON_BIN" -m src.experiments.run_flashattention3_matched_benchmark \
  --seq-lens $SEQ_LENS \
  --batch-size "$BATCH_SIZE" \
  --n-heads $N_HEADS \
  --head-dims $HEAD_DIMS \
  --dtype "$DTYPE" \
  --warmup "$WARMUP" \
  --repeats "$REPEATS" \
  --output-dir "$OUTPUT_DIR" \
  "${extra_args[@]}"

log "Done. Summary: $OUTPUT_DIR/summary.json"
