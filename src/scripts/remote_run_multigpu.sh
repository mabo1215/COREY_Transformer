#!/usr/bin/env bash
# remote_run_multigpu.sh  — executed directly on the remote server via SSH
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/home1/mabo1215/COREY_Transformer}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home1/mabo1215/.adama-micromamba}"
export ENV_NAME="${ENV_NAME:-quamba-py310}"
export MM="${MM:-/home1/mabo1215/.corey-wsl-tools/bin/micromamba}"
export GPU_IDS="${GPU_IDS:-2 3}"
export MODEL="${MODEL:-mamba-370m}"
export MAX_SAMPLES="${MAX_SAMPLES:-20}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"
export OUTPUT_BASE="${OUTPUT_BASE:-src/outputs/mgpu_longbench_remote}"

cd "$REMOTE_ROOT"
bash src/scripts/wsl_run_multigpu_longbench.sh
