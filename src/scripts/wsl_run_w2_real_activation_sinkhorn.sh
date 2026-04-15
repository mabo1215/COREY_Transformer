#!/usr/bin/env bash
# W2 Real-Activation Sinkhorn Proxy Validation (WSL2 runner)
#
# Loads a real Mamba checkpoint, extracts in_proj activations via forward hooks,
# applies a simulated Hadamard rotation, and computes the Sinkhorn proxy residual
# validating Theorem 1 on real (not synthetic) data.
#
# Usage:
#   bash src/scripts/wsl_run_w2_real_activation_sinkhorn.sh
#
# Environment variable overrides:
#   MODEL              (default: mamba-370m)
#   LAYERS             (default: "0 1 2 3")
#   NUM_SAMPLES        (default: 20)
#   SINKHORN_BINS      (default: 64)
#   OUTPUT_DIR         (default: src/outputs/real_activation_sinkhorn)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Locate micromamba binary
if [[ -n "${MICROMAMBA_BIN:-}" ]]; then
    : # use as-is
elif [[ -x "$HOME/.corey-wsl-tools/bin/micromamba" ]]; then
    MICROMAMBA_BIN="$HOME/.corey-wsl-tools/bin/micromamba"
elif [[ -x "$HOME/.adama-wsl-tools/bin/micromamba" ]]; then
    MICROMAMBA_BIN="$HOME/.adama-wsl-tools/bin/micromamba"
elif command -v micromamba &>/dev/null; then
    MICROMAMBA_BIN="$(command -v micromamba)"
else
    echo "[W2] ERROR: micromamba not found." >&2; exit 1
fi

# Locate env root and env name
if [[ -d "$HOME/.corey-micromamba/envs/corey-cuda128" ]]; then
    MAMBA_ENVS="$HOME/.corey-micromamba/envs"
    ENV_NAME="corey-cuda128"
elif [[ -d "$HOME/.adama-micromamba/envs/adama-cuda128" ]]; then
    MAMBA_ENVS="$HOME/.adama-micromamba/envs"
    ENV_NAME="adama-cuda128"
else
    echo "[W2] ERROR: no suitable env (corey-cuda128 / adama-cuda128) found." >&2; exit 1
fi

PYTHON="$MAMBA_ENVS/$ENV_NAME/bin/python"
[[ -x "$PYTHON" ]] || { echo "[W2] ERROR: python not found at $PYTHON." >&2; exit 1; }

MODEL="${MODEL:-mamba-370m}"
LAYERS="${LAYERS:-0 1 2 3}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
SINKHORN_BINS="${SINKHORN_BINS:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-src/outputs/real_activation_sinkhorn}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "[W2] Model=$MODEL  Layers=[$LAYERS]  Samples=$NUM_SAMPLES"
echo "[W2] Output: $OUTPUT_DIR"

cd "$REPO_ROOT"

# Convert LAYERS string to array for argparse
read -r -a LAYERS_ARR <<< "$LAYERS"

"$PYTHON" -m src.experiments.run_real_activation_sinkhorn \
    --model "$MODEL" \
    --layers "${LAYERS_ARR[@]}" \
    --num-samples "$NUM_SAMPLES" \
    --sinkhorn-bins "$SINKHORN_BINS" \
    --output-dir "$OUTPUT_DIR"

echo "[W2] Done. Results: $REPO_ROOT/$OUTPUT_DIR/summary.json"
