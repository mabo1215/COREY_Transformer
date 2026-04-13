#!/usr/bin/env bash
# Run hook microbenchmark on remote 4x RTX 3090 machine (10.147.20.176).
# Runs baseline (hook disabled) and hook-enabled on GPU 0, then collects
# results back to local outputs.
#
# Usage (from WSL or Git Bash):
#   bash src/scripts/remote_run_hook_micro.sh
set -euo pipefail

REMOTE="mabo1215@10.147.20.176"
REMOTE_ROOT="/home1/mabo1215/COREY_Transformer"
LOCAL_ROOT="${LOCAL_ROOT:-/mnt/c/source/COREY_Transformer}"
MM="/home1/mabo1215/.corey-wsl-tools/bin/micromamba"
ENV_NAME="quamba-py310"

MODELS="${MODELS:-mamba-370m mamba-1.4b}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
WARMUP_RUNS="${WARMUP_RUNS:-0}"
BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-1}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
TASK="${TASK:-narrativeqa}"

REMOTE_OUTPUT_BASE="$REMOTE_ROOT/src/outputs"

echo "[remote-hook-micro] syncing repo code to remote..."
# Sync src/ to remote (exclude outputs and data to save time)
rsync -avz --delete \
  --exclude='src/outputs/' \
  --exclude='src/data/' \
  --exclude='__pycache__/' \
  --exclude='.git/' \
  --exclude='Quamba/' \
  --exclude='paper/' \
  --exclude='*.pyc' \
  "$LOCAL_ROOT/src/algorithms/" "$REMOTE:$REMOTE_ROOT/src/algorithms/"
rsync -avz --delete \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  "$LOCAL_ROOT/src/experiments/" "$REMOTE:$REMOTE_ROOT/src/experiments/"

for model in $MODELS; do
  echo ""
  echo "========================================="
  echo "[remote-hook-micro] $model — baseline (hook disabled)"
  echo "========================================="
  BASELINE_DIR="$REMOTE_OUTPUT_BASE/remote_hook_micro_baseline"
  ssh -o BatchMode=yes "$REMOTE" "
    export MAMBA_ROOT_PREFIX=/home1/mabo1215/.adama-micromamba
    cd $REMOTE_ROOT
    $MM run -n $ENV_NAME python -m src.experiments.run_official_mamba_benchmark \
      --model $model \
      --dataset-source local \
      --dataset-root $REMOTE_ROOT/src/data/longbench_subset \
      --task $TASK \
      --max-samples $MAX_SAMPLES \
      --max-length $MAX_LENGTH \
      --max-new-tokens $MAX_NEW_TOKENS \
      --device $DEVICE \
      --dtype $DTYPE \
      --precision fp16 \
      --warmup-runs $WARMUP_RUNS \
      --benchmark-repeats $BENCHMARK_REPEATS \
      --disable-entropy-hook \
      --output-dir $BASELINE_DIR
  "

  echo ""
  echo "========================================="
  echo "[remote-hook-micro] $model — hook enabled (policy=corey)"
  echo "========================================="
  HOOK_DIR="$REMOTE_OUTPUT_BASE/remote_hook_micro_enabled"
  ssh -o BatchMode=yes "$REMOTE" "
    export MAMBA_ROOT_PREFIX=/home1/mabo1215/.adama-micromamba
    cd $REMOTE_ROOT
    $MM run -n $ENV_NAME python -m src.experiments.run_official_mamba_benchmark \
      --model $model \
      --dataset-source local \
      --dataset-root $REMOTE_ROOT/src/data/longbench_subset \
      --task $TASK \
      --max-samples $MAX_SAMPLES \
      --max-length $MAX_LENGTH \
      --max-new-tokens $MAX_NEW_TOKENS \
      --device $DEVICE \
      --dtype $DTYPE \
      --precision fp16 \
      --warmup-runs $WARMUP_RUNS \
      --benchmark-repeats $BENCHMARK_REPEATS \
      --scheduler-policy corey \
      --output-dir $HOOK_DIR
  "
done

echo ""
echo "[remote-hook-micro] collecting results..."
LOCAL_OUTPUT="$LOCAL_ROOT/src/outputs"
mkdir -p "$LOCAL_OUTPUT/remote_hook_micro_baseline" "$LOCAL_OUTPUT/remote_hook_micro_enabled"

for model in $MODELS; do
  rsync -avz "$REMOTE:$REMOTE_OUTPUT_BASE/remote_hook_micro_baseline/$model/" \
    "$LOCAL_OUTPUT/remote_hook_micro_baseline/$model/"
  rsync -avz "$REMOTE:$REMOTE_OUTPUT_BASE/remote_hook_micro_enabled/$model/" \
    "$LOCAL_OUTPUT/remote_hook_micro_enabled/$model/"
done

echo ""
echo "[remote-hook-micro] running analysis..."
cd "$LOCAL_ROOT"
python3 -m src.experiments.analyze_scheduler_hook_results \
  --baseline-root "$LOCAL_OUTPUT/remote_hook_micro_baseline" \
  --hook-root "$LOCAL_OUTPUT/remote_hook_micro_enabled" \
  --output-dir "$LOCAL_OUTPUT/remote_hook_micro_analysis"

echo ""
echo "[remote-hook-micro] done. Results in:"
echo "  $LOCAL_OUTPUT/remote_hook_micro_analysis/"
cat "$LOCAL_OUTPUT/remote_hook_micro_analysis/hook_overhead_summary.json"
