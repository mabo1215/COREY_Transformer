#!/usr/bin/env bash
# W1 Real-GPU Three-Policy Chunked Scan Benchmark (WSL2 runner)
#
# Runs run_w1_triton_triplet.py in the WSL2 CUDA environment to produce
# genuine GPU-timing differences between policy_off / policy_static / policy_corey.
#
# Unlike wsl_run_w1_triplet.sh (which routes through the full checkpoint matrix
# and produces identical Triton kernel paths for all three policies), this script
# calls the chunked-scan benchmark directly so that chunk_size actually changes
# the number of kernel invocations.
#
# Usage:
#   bash src/scripts/wsl_run_w1_triton_triplet.sh
#
# Environment variable overrides:
#   SEQ_LEN            (default: 4096)
#   DIM                (default: 1024)
#   D_STATE            (default: 16)
#   DTYPE              (default: float16)
#   WARMUP_RUNS        (default: 5)
#   BENCHMARK_REPEATS  (default: 30)
#   STATIC_CHUNK_SIZE  (default: 64)
#   COREY_MIN_CHUNK    (default: 32)
#   COREY_MAX_CHUNK    (default: 512)
#   OUTPUT_DIR         (default: src/outputs/w1_triton_triplet)
#   MICROMAMBA_ROOT    (auto-detected: $HOME/.corey-micromamba or $HOME/.adama-micromamba)
#   CONDA_ENV          (default: corey-cuda128, then adama-cuda128)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---------------------------------------------------------------------------
# Locate micromamba root
# ---------------------------------------------------------------------------
if [[ -n "${MICROMAMBA_ROOT:-}" ]]; then
    MAMBA_ROOT="$MICROMAMBA_ROOT"
elif [[ -d "$HOME/.corey-micromamba" ]]; then
    MAMBA_ROOT="$HOME/.corey-micromamba"
elif [[ -d "$HOME/.adama-micromamba" ]]; then
    MAMBA_ROOT="$HOME/.adama-micromamba"
elif [[ -d "$HOME/.local/share/mamba" ]]; then
    MAMBA_ROOT="$HOME/.local/share/mamba"
else
    echo "[W1] ERROR: micromamba root not found. Set MICROMAMBA_ROOT." >&2
    exit 1
fi

MICROMAMBA_BIN="$MAMBA_ROOT/bin/micromamba"
if [[ ! -x "$MICROMAMBA_BIN" ]]; then
    # Fallback: check .wsl-tools/bin
    ALT_BIN="$REPO_ROOT/.wsl-tools/bin/micromamba"
    if [[ -x "$ALT_BIN" ]]; then
        MICROMAMBA_BIN="$ALT_BIN"
    else
        echo "[W1] ERROR: micromamba not found at $MICROMAMBA_BIN" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Locate conda environment — must have mamba_ssm available
# ---------------------------------------------------------------------------
_find_env_with_mamba_ssm() {
    # Search across both micromamba roots for an env that has mamba_ssm
    local roots=( "$HOME/.adama-micromamba" "$HOME/.corey-micromamba" "$HOME/.local/share/mamba" )
    local candidates=( "adama-cuda128" "corey-cuda128" )
    for root in "${roots[@]}"; do
        [[ -d "$root/envs" ]] || continue
        for env in "${candidates[@]}"; do
            local py="$root/envs/$env/bin/python"
            [[ -x "$py" ]] || continue
            if "$py" -c "import mamba_ssm" 2>/dev/null; then
                echo "$root:$env"
                return 0
            fi
        done
    done
    return 1
}

if [[ -n "${CONDA_ENV:-}" ]]; then
    ENV_NAME="$CONDA_ENV"
    PYTHON="$MAMBA_ROOT/envs/$ENV_NAME/bin/python"
else
    # Auto-detect env that has mamba_ssm
    FOUND=$(_find_env_with_mamba_ssm 2>/dev/null || true)
    if [[ -z "$FOUND" ]]; then
        echo "[W1] ERROR: no env with mamba_ssm found. Install mamba_ssm in one of: adama-cuda128, corey-cuda128" >&2
        exit 1
    fi
    DETECTED_ROOT="${FOUND%%:*}"
    ENV_NAME="${FOUND##*:}"
    MAMBA_ROOT="$DETECTED_ROOT"
    MICROMAMBA_BIN="$MAMBA_ROOT/bin/micromamba"
    echo "[W1] Auto-detected env with mamba_ssm: $MAMBA_ROOT/envs/$ENV_NAME"
fi

PYTHON="$MAMBA_ROOT/envs/$ENV_NAME/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "[W1] ERROR: Python not found at $PYTHON" >&2
    exit 1
fi

echo "[W1] Using Python: $PYTHON"
echo "[W1] Repo root:    $REPO_ROOT"

# ---------------------------------------------------------------------------
# Parameters (env-var overrides)
# ---------------------------------------------------------------------------
SEQ_LEN="${SEQ_LEN:-4096}"
DIM="${DIM:-1024}"
D_STATE="${D_STATE:-16}"
DTYPE="${DTYPE:-float16}"
WARMUP_RUNS="${WARMUP_RUNS:-5}"
BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-30}"
STATIC_CHUNK_SIZE="${STATIC_CHUNK_SIZE:-64}"
COREY_MIN_CHUNK="${COREY_MIN_CHUNK:-32}"
COREY_MAX_CHUNK="${COREY_MAX_CHUNK:-512}"
OUTPUT_DIR="${OUTPUT_DIR:-src/outputs/w1_triton_triplet}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "[W1] Config: seq_len=$SEQ_LEN dim=$DIM d_state=$D_STATE dtype=$DTYPE"
echo "[W1] Chunks: static=$STATIC_CHUNK_SIZE corey=[$COREY_MIN_CHUNK,$COREY_MAX_CHUNK]"
echo "[W1] Repeats: warmup=$WARMUP_RUNS benchmark=$BENCHMARK_REPEATS"
echo "[W1] Output: $OUTPUT_DIR"

cd "$REPO_ROOT"

"$PYTHON" -m src.experiments.run_w1_triton_triplet \
    --seq-len "$SEQ_LEN" \
    --dim "$DIM" \
    --d-state "$D_STATE" \
    --dtype "$DTYPE" \
    --warmup-runs "$WARMUP_RUNS" \
    --benchmark-repeats "$BENCHMARK_REPEATS" \
    --static-chunk-size "$STATIC_CHUNK_SIZE" \
    --corey-min-chunk "$COREY_MIN_CHUNK" \
    --corey-max-chunk "$COREY_MAX_CHUNK" \
    --output-dir "$OUTPUT_DIR"

echo "[W1] Done. Results: $REPO_ROOT/$OUTPUT_DIR/summary.json"
