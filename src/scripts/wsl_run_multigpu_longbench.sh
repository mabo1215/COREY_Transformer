#!/usr/bin/env bash
# wsl_run_multigpu_longbench.sh
# -----------------------------------------------------------
# Run LongBench inference sharded across multiple GPUs on the
# remote ubuntu-4card server (4x RTX 3090).
#
# Strategy: data-parallel sharding — each GPU process handles
# an independent slice of the 20-sample set, all in parallel.
# Wall-clock time of the parallel run is measured and compared
# against the sequential single-GPU baseline.
#
# Configuration (override with env vars):
#   MAMBA_ROOT_PREFIX  default /home/mabo1215/.adama-micromamba
#   ENV_NAME           default adama-cuda128
#   MODEL              default mamba-370m
#   TASKS              default "narrativeqa qasper multifieldqa_en gov_report"
#   MAX_SAMPLES        default 20   (total, split evenly across GPUs)
#   MAX_LENGTH         default 4096
#   OUTPUT_BASE        default src/outputs/mgpu_longbench_remote
#   GPU_IDS            default "0 1 2 3"  (space-separated CUDA device indices)
# -----------------------------------------------------------
set -euo pipefail

MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mabo1215/.adama-micromamba}"
ENV_NAME="${ENV_NAME:-adama-cuda128}"
MODEL="${MODEL:-mamba-370m}"
TASKS="${TASKS:-narrativeqa qasper multifieldqa_en gov_report}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
OUTPUT_BASE="${OUTPUT_BASE:-src/outputs/mgpu_longbench_remote}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Resolve micromamba across local WSL and remote server layouts.
MM_CANDIDATES=(
    "${MM:-}"
    "${REPO_ROOT}/.wsl-tools/bin/micromamba"
    "/home1/mabo1215/.corey-wsl-tools/bin/micromamba"
    "/home/mabo1215/.corey-wsl-tools/bin/micromamba"
    "$(command -v micromamba || true)"
)
MM=""
for cand in "${MM_CANDIDATES[@]}"; do
    if [[ -n "$cand" && -x "$cand" ]]; then
        MM="$cand"
        break
    fi
done
if [[ -z "$MM" ]]; then
    echo "[mgpu] ERROR: micromamba not found. Set MM=/path/to/micromamba and retry."
    exit 127
fi

# Build GPU array
GPU_ARRAY=($GPU_IDS)
NUM_GPUS="${#GPU_ARRAY[@]}"

echo "[mgpu] Model        : $MODEL"
echo "[mgpu] Tasks        : $TASKS"
echo "[mgpu] Total samples: $MAX_SAMPLES  / $NUM_GPUS GPUs"
echo "[mgpu] Max length   : $MAX_LENGTH"
echo "[mgpu] Output base  : $OUTPUT_BASE"
echo "[mgpu] GPU IDs      : ${GPU_ARRAY[*]}"
echo "[mgpu] Env name     : $ENV_NAME"
echo "[mgpu] Micromamba   : $MM"
echo "[mgpu] HF endpoint  : $HF_ENDPOINT"

# Samples per shard (ceiling division)
SHARD_SIZE=$(( (MAX_SAMPLES + NUM_GPUS - 1) / NUM_GPUS ))

PIDS=()
WALL_START=$(date +%s%3N)   # milliseconds

for i in "${!GPU_ARRAY[@]}"; do
    GPU_ID="${GPU_ARRAY[$i]}"
    OFFSET=$(( i * SHARD_SIZE ))
    SHARD_SAMPLES="${SHARD_SIZE}"
    SHARD_DIR="${OUTPUT_BASE}_shard_${i}"

    if [[ "$OFFSET" -ge "$MAX_SAMPLES" ]]; then
        echo "[mgpu] GPU $GPU_ID: shard $i offset $OFFSET >= $MAX_SAMPLES, skipping"
        continue
    fi
    # Clamp last shard
    REMAINING=$(( MAX_SAMPLES - OFFSET ))
    if [[ "$REMAINING" -lt "$SHARD_SAMPLES" ]]; then
        SHARD_SAMPLES="$REMAINING"
    fi

    echo "[mgpu] GPU $GPU_ID: shard $i  offset=$OFFSET  samples=$SHARD_SAMPLES  → $SHARD_DIR"
    LOG_FILE="${SHARD_DIR}_run.log"
    mkdir -p "$(dirname "$LOG_FILE")"

    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX" \
    HF_ENDPOINT="$HF_ENDPOINT" \
    "$MM" run -n "$ENV_NAME" \
        python -m src.experiments.run_longbench_inference \
            --model "$MODEL" \
            --tasks $TASKS \
            --max-samples "$SHARD_SAMPLES" \
            --sample-offset "$OFFSET" \
            --dataset-root src/data/longbench_subset \
            --dataset-source local \
            --max-length "$MAX_LENGTH" \
            --device cuda \
            --dtype float16 \
            --batch-size 1 \
            --output-dir "$SHARD_DIR" \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    echo "[mgpu] shard $i PID=$!"
done

echo "[mgpu] Waiting for ${#PIDS[@]} shard processes ..."
FAILED=0
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    if wait "$PID"; then
        echo "[mgpu] shard $i (PID $PID) completed OK"
    else
        echo "[mgpu] shard $i (PID $PID) FAILED (exit $?)"
        LOG_FILE="${OUTPUT_BASE}_shard_${i}_run.log"
        if [[ -f "$LOG_FILE" ]]; then
            echo "[mgpu] Last 30 lines of $LOG_FILE"
            tail -n 30 "$LOG_FILE"
        fi
        FAILED=$(( FAILED + 1 ))
    fi
done

WALL_END=$(date +%s%3N)
WALL_MS=$(( WALL_END - WALL_START ))
echo "[mgpu] ── All shards done in ${WALL_MS} ms (wall-clock) ──"

if [[ "$FAILED" -gt 0 ]]; then
    echo "[mgpu] ERROR: $FAILED shard(s) failed — aborting merge."
    exit 1
fi

# Collect shard dirs that were actually created
SHARD_DIRS=()
for i in "${!GPU_ARRAY[@]}"; do
    SHARD_DIR="${OUTPUT_BASE}_shard_${i}"
    OFFSET=$(( i * SHARD_SIZE ))
    if [[ -d "$SHARD_DIR" && "$OFFSET" -lt "$MAX_SAMPLES" ]]; then
        SHARD_DIRS+=("$SHARD_DIR")
    fi
done

MERGED_DIR="${OUTPUT_BASE}_merged"
echo "[mgpu] Merging ${#SHARD_DIRS[@]} shards → $MERGED_DIR"
MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX" \
HF_ENDPOINT="$HF_ENDPOINT" \
"$MM" run -n "$ENV_NAME" \
    python -m src.experiments.merge_sharded_results \
        --shard-dirs "${SHARD_DIRS[@]}" \
        --output-dir "$MERGED_DIR" \
        --model "$MODEL" \
        --precision fp16

echo "[mgpu] ── Multi-GPU LongBench run complete ──"
echo "[mgpu] Wall-clock total: ${WALL_MS} ms"
echo "[mgpu] Merged results  : $MERGED_DIR"
echo ""

# Print quick score summary
SUMMARY_CSV="${MERGED_DIR}/${MODEL}/fp16/summary.csv"
if [[ -f "$SUMMARY_CSV" ]]; then
    echo "[mgpu] Merged summary (${SUMMARY_CSV}):"
    if command -v column >/dev/null 2>&1; then
        column -t -s',' "$SUMMARY_CSV" | head -20
    else
        cat "$SUMMARY_CSV"
    fi
fi
