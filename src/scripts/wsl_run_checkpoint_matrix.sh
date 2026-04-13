#!/usr/bin/env bash
set -euo pipefail

DEFAULT_REPO_ROOT="/mnt/c/source/Corey_Transformer"
if [[ ! -d "$DEFAULT_REPO_ROOT" && -d "/mnt/c/source/COREY_Transformer" ]]; then
  DEFAULT_REPO_ROOT="/mnt/c/source/COREY_Transformer"
fi

REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
TOOLS_DIR="${TOOLS_DIR:-${HOME}/.corey-wsl-tools}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-}"
ENV_NAME="${ENV_NAME:-}"
REPO_MICROMAMBA_BIN="$REPO_ROOT/.wsl-tools/bin/micromamba"
MICROMAMBA_BIN="$TOOLS_DIR/bin/micromamba"
MODES="${MODES:-longbench benchmark}"
MODELS="${MODELS:-mamba-370m mamba-1.4b mamba-2.8b}"
PRECISIONS="${PRECISIONS:-fp16}"
TASKS="${TASKS:-narrativeqa qasper multifieldqa_en gov_report}"
LM_DATASETS="${LM_DATASETS:-wikitext103 pg19}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/src/data/longbench_subset}"
DATASET_SOURCE="${DATASET_SOURCE:-hf}"
DATASET_NAME="${DATASET_NAME:-THUDM/LongBench}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
BENCHMARK_TASK="${BENCHMARK_TASK:-narrativeqa}"
BENCHMARK_MAX_SAMPLES="${BENCHMARK_MAX_SAMPLES:-1}"
BENCHMARK_MAX_NEW_TOKENS="${BENCHMARK_MAX_NEW_TOKENS:-32}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-5}"
LM_MAX_SAMPLES="${LM_MAX_SAMPLES:-20}"
PPL_MAX_SAMPLES="${PPL_MAX_SAMPLES:-20}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
OUTPUT_DIR="${OUTPUT_DIR:-src/outputs/checkpoint_matrix_wsl}"
SCHEDULER_POLICY="${SCHEDULER_POLICY:-corey}"
STATIC_TILE_SIZE="${STATIC_TILE_SIZE:-256}"
COLLECT_ENERGY="${COLLECT_ENERGY:-0}"
ENERGY_GPU_INDEX="${ENERGY_GPU_INDEX:-0}"
DISABLE_ENTROPY_HOOK="${DISABLE_ENTROPY_HOOK:-0}"
EVAL_PERPLEXITY="${EVAL_PERPLEXITY:-1}"
WINDOWS_USER="${WINDOWS_USER:-}"
if [[ -z "$WINDOWS_USER" ]]; then
  WINDOWS_USER="$USER"
fi
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

if [[ -z "$MAMBA_ROOT_PREFIX" ]]; then
  for candidate in "${HOME}/.adama-micromamba" "${HOME}/.corey-micromamba"; do
    if [[ -d "$candidate" ]]; then
      MAMBA_ROOT_PREFIX="$candidate"
      break
    fi
  done
fi
if [[ -z "$MAMBA_ROOT_PREFIX" ]]; then
  MAMBA_ROOT_PREFIX="${HOME}/.corey-micromamba"
fi

if [[ -z "$ENV_NAME" ]]; then
  if [[ -d "$MAMBA_ROOT_PREFIX/envs/adama-cuda128" ]]; then
    ENV_NAME="adama-cuda128"
  elif [[ -d "$MAMBA_ROOT_PREFIX/envs/corey-cuda128" ]]; then
    ENV_NAME="corey-cuda128"
  else
    ENV_NAME="corey-cuda128"
  fi
fi

if [[ -x "$REPO_MICROMAMBA_BIN" ]]; then
  MICROMAMBA_BIN="$REPO_MICROMAMBA_BIN"
fi

export MAMBA_ROOT_PREFIX
export HF_HOME
cd "$REPO_ROOT"

EXTRA_FLAGS=""
if [[ "$COLLECT_ENERGY" == "1" ]]; then
  EXTRA_FLAGS+=" --collect-energy --energy-gpu-index '$ENERGY_GPU_INDEX'"
fi
if [[ "$DISABLE_ENTROPY_HOOK" == "1" ]]; then
  EXTRA_FLAGS+=" --disable-entropy-hook"
fi
if [[ "$EVAL_PERPLEXITY" == "1" ]]; then
  EXTRA_FLAGS+=" --eval-perplexity"
fi

"$MICROMAMBA_BIN" run -n "$ENV_NAME" bash -lc "export PYTHONPATH='$REPO_ROOT'; export HF_HOME='$HF_HOME'; python -m src.experiments.run_checkpoint_matrix \
  --modes $MODES \
  --models $MODELS \
  --precisions $PRECISIONS \
  --tasks $TASKS \
  --lm-datasets $LM_DATASETS \
  --dataset-root '$DATASET_ROOT' \
  --dataset-source '$DATASET_SOURCE' \
  --dataset-name '$DATASET_NAME' \
  --max-samples '$MAX_SAMPLES' \
  --benchmark-task '$BENCHMARK_TASK' \
  --benchmark-max-samples '$BENCHMARK_MAX_SAMPLES' \
  --benchmark-max-new-tokens '$BENCHMARK_MAX_NEW_TOKENS' \
  --warmup-runs '$WARMUP_RUNS' \
  --benchmark-repeats '$BENCHMARK_REPEATS' \
  --ppl-max-samples '$PPL_MAX_SAMPLES' \
  --lm-max-samples '$LM_MAX_SAMPLES' \
  --max-length '$MAX_LENGTH' \
  --device '$DEVICE' \
  --dtype '$DTYPE' \
  --scheduler-policy '$SCHEDULER_POLICY' \
  --static-tile-size '$STATIC_TILE_SIZE' \
  --skip-existing \
  --output-dir '$OUTPUT_DIR' \
  $EXTRA_FLAGS"