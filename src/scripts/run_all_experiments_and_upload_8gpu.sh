#!/bin/bash
# Usage: bash run_all_experiments_and_upload_8gpu.sh
# Run this script on the 8x3090 CUDA server
set -e

MODEL="state-spaces/mamba2-2.7b"
SEQ_LEN=4096
CHUNK_SIZE=512
REPEAT=30
NUM_GPUS=1
OUTPUT_DIR=~/src/outputs/gpu1_all
SYNC_SOURCE_DIR=~/source/COREY_Transformer/src/outputs
SYNC_TARGET="gs://corey-transformer-paper-results/recdynamic/"
RESTORE_FROM_GCS="${RESTORE_FROM_GCS:-1}"
STAGE_DONE_MARKER=".stage_done"
ALLOW_LEGACY_NONEMPTY_AS_DONE="${ALLOW_LEGACY_NONEMPTY_AS_DONE:-0}"

restore_outputs_from_gcs() {
  if [[ "$RESTORE_FROM_GCS" == "1" ]]; then
    echo "[INFO] Restoring previous outputs from: $SYNC_TARGET"
    mkdir -p "$SYNC_SOURCE_DIR"
    gsutil -m rsync -r "$SYNC_TARGET" "$SYNC_SOURCE_DIR" || true
  else
    echo "[INFO] Skip restore from GCS (RESTORE_FROM_GCS=$RESTORE_FROM_GCS)"
  fi
}

is_stage_completed() {
  local stage_output_dir="$1"
  if [[ -f "$stage_output_dir/$STAGE_DONE_MARKER" ]]; then
    return 0
  fi
  if [[ "$ALLOW_LEGACY_NONEMPTY_AS_DONE" == "1" ]] && [[ -d "$stage_output_dir" ]] && [[ -n "$(ls -A "$stage_output_dir" 2>/dev/null)" ]]; then
    echo "[WARN] No $STAGE_DONE_MARKER found in $stage_output_dir, but directory is non-empty. Treating as completed because ALLOW_LEGACY_NONEMPTY_AS_DONE=1."
    return 0
  fi
  return 1
}

sync_stage_outputs() {
  echo "[INFO] Syncing stage outputs: $1"
  gsutil -m rsync -r "$SYNC_SOURCE_DIR" "$SYNC_TARGET"
}

# 1. Run all experiments
restore_outputs_from_gcs

if is_stage_completed ~/source/COREY_Transformer/src/outputs/corey_8gpu_benchmark; then
  echo "[SKIP] corey_8gpu_benchmark already exists, skipping run"
else
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    ~/source/COREY_Transformer/src/experiments/run_corey_8gpu_benchmark.py \
    --device cuda --model $MODEL --chunk-size $CHUNK_SIZE --seq-len $SEQ_LEN --repeat $REPEAT \
    --output-dir ~/source/COREY_Transformer/src/outputs/corey_8gpu_benchmark
  touch ~/source/COREY_Transformer/src/outputs/corey_8gpu_benchmark/$STAGE_DONE_MARKER
  sync_stage_outputs "corey_8gpu_benchmark"
fi

if is_stage_completed ~/source/COREY_Transformer/src/outputs/integrated_end_to_end_8gpu; then
  echo "[SKIP] integrated_end_to_end_8gpu already exists, skipping run"
else
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    ~/source/COREY_Transformer/src/experiments/run_integrated_end_to_end_8gpu.py \
    --model $MODEL --output-dir ~/source/COREY_Transformer/src/outputs/integrated_end_to_end_8gpu
  touch ~/source/COREY_Transformer/src/outputs/integrated_end_to_end_8gpu/$STAGE_DONE_MARKER
  sync_stage_outputs "integrated_end_to_end_8gpu"
fi

if is_stage_completed ~/source/COREY_Transformer/src/outputs/heterogeneous_corpus_8gpu; then
  echo "[SKIP] heterogeneous_corpus_8gpu already exists, skipping run"
else
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    ~/source/COREY_Transformer/src/experiments/run_heterogeneous_corpus_8gpu.py \
    --model $MODEL --output-dir ~/source/COREY_Transformer/src/outputs/heterogeneous_corpus_8gpu
  touch ~/source/COREY_Transformer/src/outputs/heterogeneous_corpus_8gpu/$STAGE_DONE_MARKER
  sync_stage_outputs "heterogeneous_corpus_8gpu"
fi


# 2. Merge all results into OUTPUT_DIR
mkdir -p $OUTPUT_DIR
cp -r ~/source/COREY_Transformer/src/outputs/corey_8gpu_benchmark $OUTPUT_DIR/
cp -r ~/source/COREY_Transformer/src/outputs/integrated_end_to_end_8gpu $OUTPUT_DIR/
cp -r ~/source/COREY_Transformer/src/outputs/heterogeneous_corpus_8gpu $OUTPUT_DIR/

# 3. Final sync to ensure the latest merged outputs are uploaded
sync_stage_outputs "final"
echo "[ALL DONE] Results synced to: $SYNC_TARGET"
