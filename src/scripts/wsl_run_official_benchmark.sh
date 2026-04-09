#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/c/source/Corey_Transformer"
TOOLS_DIR="${HOME}/.corey-wsl-tools"
MAMBA_ROOT_PREFIX="${HOME}/.corey-micromamba"
ENV_NAME="corey-cuda128"
REPO_MICROMAMBA_BIN="$REPO_ROOT/.wsl-tools/bin/micromamba"
MICROMAMBA_BIN="$TOOLS_DIR/bin/micromamba"

if [[ -x "$REPO_MICROMAMBA_BIN" ]]; then
  MICROMAMBA_BIN="$REPO_MICROMAMBA_BIN"
fi

export MAMBA_ROOT_PREFIX
cd "$REPO_ROOT"

"$MICROMAMBA_BIN" run -n "$ENV_NAME" bash -lc "export PYTHONPATH='$REPO_ROOT'; python -m src.experiments.run_official_mamba_benchmark \
  --model mamba-370m \
  --dataset-root src/data/longbench_smoke \
  --dataset-source local \
  --task narrativeqa \
  --max-samples 1 \
  --max-new-tokens 32 \
  --warmup-runs 1 \
  --benchmark-repeats 5 \
  --disable-entropy-hook \
  --lm-datasets wikitext103 \
  --lm-max-samples 1 \
  --max-length 2048 \
  --device cuda \
  --dtype float16 \
  --precision fp16 \
  --output-dir src/outputs/official_hf_benchmark_wsl"