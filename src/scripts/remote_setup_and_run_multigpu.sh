#!/usr/bin/env bash
# remote_setup_and_run_multigpu.sh
# Install minimal deps into quamba-py310 then run 2-GPU parallel LongBench
set -euo pipefail

export MAMBA_ROOT_PREFIX=/home1/mabo1215/.adama-micromamba
MM=/home1/mabo1215/.corey-wsl-tools/bin/micromamba
REPO=/home1/mabo1215/COREY_Transformer

echo "=== Installing packages into quamba-py310 ==="
$MM run -n quamba-py310 pip install --quiet --upgrade pip
$MM run -n quamba-py310 pip install --quiet \
    torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121

$MM run -n quamba-py310 pip install --quiet \
    transformers==4.41.2 \
    datasets==2.19.0 \
    sentencepiece==0.2.0 \
    numpy==1.26.4 \
    scipy==1.15.2 \
    scikit-learn==1.6.1 \
    zstandard==0.23.0 \
    huggingface_hub==0.27.0

echo "=== Smoke check ==="
$MM run -n quamba-py310 python -c 'import torch, transformers, datasets; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")'

echo "=== Launching 2-GPU parallel LongBench (GPU 2 & 3) ==="
cd "$REPO"

SHARD_SIZE=10
PIDS=()
WALL_START=$(date +%s%3N)

for i in 0 1; do
    GPU_ID=$(( i + 2 ))   # use GPU 2 and 3
    OFFSET=$(( i * SHARD_SIZE ))
    SHARD_DIR="src/outputs/mgpu_longbench_remote_shard_${i}"
    LOG="${SHARD_DIR}_run.log"
    mkdir -p "$(dirname "$LOG")"

    echo "[mgpu] GPU ${GPU_ID}: shard $i offset=${OFFSET} samples=10 → ${SHARD_DIR}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX} \
    $MM run -n quamba-py310 \
        python -m src.experiments.run_longbench_inference \
            --model mamba-370m \
            --tasks narrativeqa qasper multifieldqa_en gov_report \
            --max-samples 10 \
            --sample-offset "${OFFSET}" \
            --dataset-root src/data/longbench_subset \
            --dataset-source local \
            --max-length 4096 \
            --device cuda \
            --dtype float16 \
            --batch-size 1 \
            --output-dir "${SHARD_DIR}" \
        > "${LOG}" 2>&1 &
    PIDS+=($!)
    echo "  PID=$!"
done

echo "[mgpu] Waiting for ${#PIDS[@]} shards ..."
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[mgpu] shard $i OK"
    else
        echo "[mgpu] shard $i FAILED — see log"
        FAILED=$(( FAILED + 1 ))
        cat "src/outputs/mgpu_longbench_remote_shard_${i}_run.log" | tail -30
    fi
done

WALL_END=$(date +%s%3N)
WALL_MS=$(( WALL_END - WALL_START ))
echo "[mgpu] Wall-clock: ${WALL_MS} ms"

if [[ "$FAILED" -gt 0 ]]; then
    echo "[mgpu] $FAILED shard(s) failed."
    exit 1
fi

echo "[mgpu] Merging shards ..."
SHARD_DIRS=()
for i in 0 1; do
    SHARD_DIRS+=("src/outputs/mgpu_longbench_remote_shard_${i}")
done

MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX} \
$MM run -n quamba-py310 \
    python -m src.experiments.merge_sharded_results \
        --shard-dirs "${SHARD_DIRS[@]}" \
        --output-dir src/outputs/mgpu_longbench_remote_merged \
        --model mamba-370m \
        --precision fp16

echo "[mgpu] Done. Wall-clock: ${WALL_MS} ms"
SUMMARY=src/outputs/mgpu_longbench_remote_merged/mamba-370m/fp16/summary.csv
[[ -f "$SUMMARY" ]] && column -t -s',' "$SUMMARY"
