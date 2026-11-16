#!/usr/bin/env bash
set -euo pipefail
cd /root/Corey_Transformer
OUT=src/outputs/scheduler_ablation_h800_full_20260430
PY=/root/miniconda3/bin/python
export PYTHONPATH=/root/Corey_Transformer:/root/Corey_Transformer/Quamba/3rdparty/mamba:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0
export COREY_SELECTIVE_SCAN_BACKEND=auto
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
$PY -m src.experiments.run_scheduler_ablation_matrix \
  --model mamba-370m \
  --prompt-repeat 8 \
  --max-prompt-length 4096 \
  --new-tokens 32 \
  --warmup 3 \
  --repeats 50 \
  --sweep-chunks 128,256,512,1024,2048 \
  --rows static,no_entropy,random,hist,sampled_hist,cheap_proxy,variance_proxy,kurtosis_proxy,token_hist \
  --selective-scan-dispatch-module src.corey_selective_scan_dispatch \
  --output-dir $OUT
