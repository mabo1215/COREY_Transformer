#!/usr/bin/env bash
# ==============================================================================
# run_all_experiments_4gpu.sh
# Run all four paper experiments concurrently on a 4x RTX 3090 server.
#
# Experiment assignment:
#   GPU 0  →  Exp 1: External baselines (RWKV / FA2 / Mamba2) on LongBench
#   GPU 1  →  Exp 2: Quamba INT4 quantized inference on LongBench
#   GPU 2  →  Exp 3: Policy COREY ablation sweep on Mamba2-2.7B
#   GPU 3  →  Exp 4: Fused kernel algorithm benchmark (CPU-bound, GPU for metadata)
#
# Usage:
#   bash src/scripts/run_all_experiments_4gpu.sh
#
# Override defaults via env vars, e.g.:
#   MODEL_ID=benchang1110/mamba2-2.7b-hf MAX_SAMPLES=20 \
#     bash src/scripts/run_all_experiments_4gpu.sh
#
# Remote execution:
#   ssh user@server "cd ~/COREY_Transformer && bash src/scripts/run_all_experiments_4gpu.sh"
# ==============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override with env vars)
# ---------------------------------------------------------------------------
MODEL_ID="${MODEL_ID:-benchang1110/mamba2-2.7b-hf}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
TASKS="${TASKS:-narrativeqa qasper gov_report multifieldqa_en}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
DEVICE_PREFIX="cuda"
OUTPUT_BASE="${OUTPUT_BASE:-src/outputs}"
DATA_BASE="${DATA_BASE:-src/data/longbench_subset}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-src}"

# Micromamba / conda env for the 4-card server (override if needed).
# Set USE_MICROMAMBA=0 to skip micromamba activation (use current env instead).
USE_MICROMAMBA="${USE_MICROMAMBA:-0}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mabo1215/.adama-micromamba}"
ENV_NAME="${ENV_NAME:-adama-cuda128}"
MM_CANDIDATES=(
    "${MM:-}"
    "/home1/mabo1215/.corey-wsl-tools/bin/micromamba"
    "/home/mabo1215/.corey-wsl-tools/bin/micromamba"
    "$(command -v micromamba 2>/dev/null || true)"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/${OUTPUT_BASE}/run_all_4gpu_logs"
mkdir -p "$LOG_DIR"

_python() {
    if [[ "$USE_MICROMAMBA" == "1" ]]; then
        local MM=""
        for cand in "${MM_CANDIDATES[@]}"; do
            [[ -n "$cand" && -x "$cand" ]] && { MM="$cand"; break; }
        done
        if [[ -z "$MM" ]]; then
            echo "[ERROR] micromamba not found; set MM=/path/to/micromamba" >&2; exit 127
        fi
        MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX" "$MM" run -n "$ENV_NAME" python "$@"
    else
        PYTHONPATH="${REPO_ROOT}/${PYTHONPATH_ROOT}" python "$@"
    fi
}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Experiment 1 — External baselines on LongBench (GPU 0)
# ---------------------------------------------------------------------------
run_exp1() {
    local LOG="${LOG_DIR}/exp1_external_baselines.log"
    log "EXP1 starting → $LOG"

    for subset in $TASKS; do
        CUDA_VISIBLE_DEVICES=0 _python \
            "${REPO_ROOT}/src/experiments/run_external_baselines.py" \
            --models rwkv flashattention mamba2 \
            --model-id "$MODEL_ID" \
            --data-file "${REPO_ROOT}/${DATA_BASE}/${subset}/test.jsonl" \
            --max-prompts "$MAX_SAMPLES" \
            --device cuda \
            --mode auto \
            --output-dir "${REPO_ROOT}/${OUTPUT_BASE}/external_baselines_${subset}" \
            >> "$LOG" 2>&1
        log "EXP1 subset=${subset} done"
    done
    log "EXP1 complete"
}

# ---------------------------------------------------------------------------
# Experiment 2 — Quamba INT4 quantized inference (GPU 1)
# ---------------------------------------------------------------------------
run_exp2() {
    local LOG="${LOG_DIR}/exp2_quamba_quant.log"
    log "EXP2 starting → $LOG"

    for subset in $TASKS; do
        CUDA_VISIBLE_DEVICES=1 _python \
            "${REPO_ROOT}/src/experiments/run_quamba_quant_benchmark.py" \
            --model-id "$MODEL_ID" \
            --quant-backend awq \
            --bits 4 \
            --group-size 128 \
            --data-file "${REPO_ROOT}/${DATA_BASE}/${subset}/test.jsonl" \
            --max-prompts "$MAX_SAMPLES" \
            --device cuda \
            --output-dir "${REPO_ROOT}/${OUTPUT_BASE}/quamba_quant_benchmark_${subset}" \
            >> "$LOG" 2>&1
        log "EXP2 subset=${subset} done"
    done
    log "EXP2 complete"
}

# ---------------------------------------------------------------------------
# Experiment 3 — Policy COREY ablation sweep (GPU 2)
# ---------------------------------------------------------------------------
run_exp3() {
    local LOG="${LOG_DIR}/exp3_policy_corey.log"
    log "EXP3 starting → $LOG"

    for policy in corey static; do
        for subset in $TASKS; do
            CUDA_VISIBLE_DEVICES=2 _python \
                "${REPO_ROOT}/src/experiments/run_policy_corey_ablation.py" \
                --model-id "$MODEL_ID" \
                --n "$MAX_SAMPLES" \
                --policy "$policy" \
                --data-file "${REPO_ROOT}/${DATA_BASE}/${subset}/test.jsonl" \
                --device cuda \
                --output-dir "${REPO_ROOT}/${OUTPUT_BASE}/policy_corey_ablation_${policy}_${subset}" \
                >> "$LOG" 2>&1
            log "EXP3 policy=${policy} subset=${subset} done"
        done
    done
    log "EXP3 complete"
}

# ---------------------------------------------------------------------------
# Experiment 4 — Fused kernel algorithm benchmark (CPU, runs on GPU 3 host)
# ---------------------------------------------------------------------------
run_exp4() {
    local LOG="${LOG_DIR}/exp4_fused_kernel.log"
    log "EXP4 starting → $LOG"

    CUDA_VISIBLE_DEVICES=3 _python \
        "${REPO_ROOT}/src/experiments/run_fused_kernel_benchmark.py" \
        --num-ops 32 \
        --device cuda \
        --repeats 1000 \
        --output-dir "${REPO_ROOT}/${OUTPUT_BASE}/fused_kernel_benchmark" \
        >> "$LOG" 2>&1

    log "EXP4 complete"
}

# ---------------------------------------------------------------------------
# Main — launch all four experiments in parallel
# ---------------------------------------------------------------------------
WALL_START=$(date +%s%3N)
log "============================================================"
log "Launching 4 experiments on 4x RTX 3090"
log "MODEL_ID        : $MODEL_ID"
log "MAX_SAMPLES     : $MAX_SAMPLES"
log "TASKS           : $TASKS"
log "OUTPUT_BASE     : $OUTPUT_BASE"
log "USE_MICROMAMBA  : $USE_MICROMAMBA"
log "============================================================"

run_exp1 &  PID1=$!
run_exp2 &  PID2=$!
run_exp3 &  PID3=$!
run_exp4 &  PID4=$!

log "PIDs: EXP1=$PID1  EXP2=$PID2  EXP3=$PID3  EXP4=$PID4"

FAILED=0
for i in 1 2 3 4; do
    VARNAME="PID${i}"
    PID="${!VARNAME}"
    if wait "$PID"; then
        log "EXP${i} (PID $PID) finished OK"
    else
        log "EXP${i} (PID $PID) FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

WALL_END=$(date +%s%3N)
WALL_MS=$((WALL_END - WALL_START))
log "============================================================"
log "All experiments done in ${WALL_MS} ms (wall-clock)"
log "Failed: $FAILED"
log "Logs: $LOG_DIR"
log "============================================================"

if [[ "$FAILED" -gt 0 ]]; then
    log "ERROR: $FAILED experiment(s) failed."
    exit 1
fi
log "SUCCESS: All 4 experiments completed."
