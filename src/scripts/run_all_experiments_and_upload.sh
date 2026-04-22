#!/bin/bash
# Usage: bash run_all_experiments_and_upload.sh
# 在 TPU VM 内部运行本脚本
set -e

MODEL="mamba-370m"
SEQ_LEN=4096
CHUNK_SIZE=512
REPEAT=30
OUTPUT_DIR=~/src/outputs/gcloud_tpu_all
GCS_BUCKET="corey-transformer-paper-results"
GCS_RESULTS_PREFIX="results/"

# 1. 运行所有实验
PJRT_DEVICE=TPU python3 ~/source/COREY_Transformer/src/experiments/run_corey_tpu_benchmark.py \
  --device tpu --model $MODEL --chunk-size $CHUNK_SIZE --seq-len $SEQ_LEN --repeat $REPEAT \
  --output-dir ~/source/COREY_Transformer/src/outputs/corey_tpu_benchmark

PJRT_DEVICE=TPU python3 ~/source/COREY_Transformer/src/experiments/run_integrated_end_to_end.py \
  --model $MODEL --output-dir ~/source/COREY_Transformer/src/outputs/integrated_end_to_end

PJRT_DEVICE=TPU python3 ~/source/COREY_Transformer/src/experiments/run_heterogeneous_corpus.py \
  --model $MODEL --output-dir ~/source/COREY_Transformer/src/outputs/heterogeneous_corpus

# 2. 合并所有结果到 OUTPUT_DIR
mkdir -p $OUTPUT_DIR
cp -r ~/source/COREY_Transformer/src/outputs/corey_tpu_benchmark $OUTPUT_DIR/
cp -r ~/source/COREY_Transformer/src/outputs/integrated_end_to_end $OUTPUT_DIR/
cp -r ~/source/COREY_Transformer/src/outputs/heterogeneous_corpus $OUTPUT_DIR/

# 3. 上传到 GCS
if [[ -n "$GCS_BUCKET" ]]; then
  echo "[INFO] Uploading results to GCS bucket: gs://$GCS_BUCKET/$GCS_RESULTS_PREFIX"
  gsutil -m cp -r $OUTPUT_DIR gs://$GCS_BUCKET/$GCS_RESULTS_PREFIX
  echo "[ALL DONE] Results uploaded to: gs://$GCS_BUCKET/$GCS_RESULTS_PREFIX"
else
  echo "[ALL DONE] Results saved to $OUTPUT_DIR"
fi
