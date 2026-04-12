#!/usr/bin/env bash
set -euo pipefail

cd /home1/mabo1215/COREY_Transformer
OUT=src/outputs/mgpu_longbench_remote_hf

rm -rf "${OUT}_shard_0" "${OUT}_shard_1" "${OUT}_merged"

CUDA_VISIBLE_DEVICES=2 /home1/mabo1215/.corey-wsl-tools/bin/micromamba run -r /home1/mabo1215/.adama-micromamba -n quamba-py310 \
  python -m src.experiments.run_longbench_inference \
  --model mamba-370m \
  --tasks narrativeqa qasper multifieldqa_en gov_report \
  --max-samples 10 \
  --sample-offset 0 \
  --dataset-source hf \
  --dataset-name THUDM/LongBench \
  --max-length 4096 \
  --device cuda \
  --dtype float16 \
  --batch-size 1 \
  --output-dir "${OUT}_shard_0" > "${OUT}_shard_0_run.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=3 /home1/mabo1215/.corey-wsl-tools/bin/micromamba run -r /home1/mabo1215/.adama-micromamba -n quamba-py310 \
  python -m src.experiments.run_longbench_inference \
  --model mamba-370m \
  --tasks narrativeqa qasper multifieldqa_en gov_report \
  --max-samples 10 \
  --sample-offset 10 \
  --dataset-source hf \
  --dataset-name THUDM/LongBench \
  --max-length 4096 \
  --device cuda \
  --dtype float16 \
  --batch-size 1 \
  --output-dir "${OUT}_shard_1" > "${OUT}_shard_1_run.log" 2>&1 &
PID1=$!

wait "$PID0"
wait "$PID1"

/home1/mabo1215/.corey-wsl-tools/bin/micromamba run -r /home1/mabo1215/.adama-micromamba -n quamba-py310 \
  python -m src.experiments.merge_sharded_results \
  --shard-dirs "${OUT}_shard_0" "${OUT}_shard_1" \
  --output-dir "${OUT}_merged" \
  --model mamba-370m \
  --precision fp16

echo "DONE"
