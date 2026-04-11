#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/c/source/COREY_Transformer}"
TOOLS_DIR="${TOOLS_DIR:-${HOME}/.corey-wsl-tools}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/.corey-micromamba}"
ENV_NAME="${ENV_NAME:-corey-cuda128}"
REPO_MICROMAMBA_BIN="$REPO_ROOT/.wsl-tools/bin/micromamba"
MICROMAMBA_BIN="$TOOLS_DIR/bin/micromamba"
MODEL="${MODEL:-mamba-370m}"
TASK="${TASK:-narrativeqa}"
DATASET_ROOT="${DATASET_ROOT:-src/data/longbench_smoke}"
DATASET_SOURCE="${DATASET_SOURCE:-local}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-5}"
LM_DATASETS="${LM_DATASETS:-wikitext103}"
LM_MAX_SAMPLES="${LM_MAX_SAMPLES:-1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
PRECISION="${PRECISION:-fp16}"
OUTPUT_DIR="${OUTPUT_DIR:-src/outputs/official_hf_benchmark_wsl}"
WINDOWS_USER="${WINDOWS_USER:-$(cmd.exe /c \"echo %USERNAME%\" 2>/dev/null | tr -d '\r')}"
HF_HOME="${HF_HOME:-/mnt/c/Users/${WINDOWS_USER:-$USER}/.cache/huggingface}"

if [[ -x "$REPO_MICROMAMBA_BIN" ]]; then
  MICROMAMBA_BIN="$REPO_MICROMAMBA_BIN"
fi

export MAMBA_ROOT_PREFIX
export HF_HOME
cd "$REPO_ROOT"

"$MICROMAMBA_BIN" run -n "$ENV_NAME" bash -lc "export PYTHONPATH='$REPO_ROOT'; python -m src.experiments.run_official_mamba_benchmark \
  --model '$MODEL' \
  --dataset-root '$DATASET_ROOT' \
  --dataset-source '$DATASET_SOURCE' \
  --task '$TASK' \
  --max-samples '$MAX_SAMPLES' \
  --max-new-tokens '$MAX_NEW_TOKENS' \
  --warmup-runs '$WARMUP_RUNS' \
  --benchmark-repeats '$BENCHMARK_REPEATS' \
  --disable-entropy-hook \
  --lm-datasets $LM_DATASETS \
  --lm-max-samples '$LM_MAX_SAMPLES' \
  --max-length '$MAX_LENGTH' \
  --device '$DEVICE' \
  --dtype '$DTYPE' \
  --precision '$PRECISION' \
  --output-dir '$OUTPUT_DIR'"