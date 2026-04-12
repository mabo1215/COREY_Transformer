#!/usr/bin/env bash
# remote_run_multigpu.sh  — executed directly on the remote server via SSH
set -euo pipefail
cd /home1/mabo1215/COREY_Transformer
export MAMBA_ROOT_PREFIX=/home/mabo1215/.adama-micromamba
export GPU_IDS="2 3"
export MODEL=mamba-370m
export MAX_SAMPLES=20
export MAX_LENGTH=4096
export OUTPUT_BASE=src/outputs/mgpu_longbench_remote
bash src/scripts/wsl_run_multigpu_longbench.sh
