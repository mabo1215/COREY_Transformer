#!/usr/bin/env bash
set -euo pipefail
cd /root/Corey_Transformer
OUT=src/outputs/scheduler_ablation_h800_smoke_20260430
PY=/root/miniconda3/bin/python
export PYTHONPATH=/root/Corey_Transformer:/root/Corey_Transformer/Quamba/3rdparty/mamba:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0
export COREY_SELECTIVE_SCAN_BACKEND=auto
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
$PY -m src.experiments.run_scheduler_ablation_matrix \
  --model mamba-370m \
  --prompt-repeat 1 \
  --max-prompt-length 1024 \
  --new-tokens 2 \
  --warmup 0 \
  --repeats 1 \
  --rows static,no_entropy,random,sampled_hist \
  --sweep-chunks 128,256 \
  --selective-scan-dispatch-module src.corey_selective_scan_dispatch \
  --output-dir $OUT
