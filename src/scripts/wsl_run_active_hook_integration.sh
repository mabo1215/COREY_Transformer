#!/usr/bin/env bash
set -euo pipefail

# Active-hook integration benchmark runner (adama-cuda128).
# Monkey-patches MambaMixer.cuda_kernels_forward so Shannon entropy and
# COREY chunk selection execute inline on every layer during
# model.generate(), then measures passive vs active end-to-end latency.

REPO_ROOT="${REPO_ROOT:-/mnt/c/source/COREY_Transformer}"
TOOLS_DIR="${TOOLS_DIR:-${HOME}/.adama-wsl-tools}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/.adama-micromamba}"
ENV_NAME="${ENV_NAME:-adama-cuda128}"
REPO_MICROMAMBA_BIN="$REPO_ROOT/.wsl-tools/bin/micromamba"
MICROMAMBA_BIN="$TOOLS_DIR/bin/micromamba"

MODEL="${MODEL:-mamba-370m}"
NEW_TOKENS="${NEW_TOKENS:-64}"
WARMUP="${WARMUP:-2}"
REPEATS="${REPEATS:-5}"
H_REF="${H_REF:-8.0}"
NUM_BINS="${NUM_BINS:-256}"

PROMPT="${PROMPT:-Selective state space models have emerged as a powerful alternative to Transformer architectures for long-sequence modeling. Unlike attention mechanisms that scale quadratically with sequence length, state space models maintain a fixed-size hidden state and process sequences with linear complexity. The Mamba architecture introduces input-dependent state transitions, allowing the model to selectively retain or discard information based on the current input. This selectivity mechanism enables efficient processing of long sequences while preserving relevant context. The entropy of the hidden activations reflects the information content and diversity of the internal representations, which the COREY scheduler uses to adaptively schedule chunk sizes for the selective scan kernel. By estimating Shannon entropy via a fixed-width histogram over the post-convolution hidden states, COREY maps high-entropy activations to larger chunk sizes and low-entropy activations to smaller chunks, reducing the number of kernel launches while maintaining numerical accuracy.}"

OUTPUT_DIR="${OUTPUT_DIR:-src/outputs/active_hook_integration}"

WINDOWS_USER="${WINDOWS_USER:-$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r')}"
HF_HOME="${HF_HOME:-/mnt/c/Users/${WINDOWS_USER:-$USER}/.cache/huggingface}"

if [[ -x "$REPO_MICROMAMBA_BIN" ]]; then
  MICROMAMBA_BIN="$REPO_MICROMAMBA_BIN"
fi
if [[ ! -x "$MICROMAMBA_BIN" ]]; then
  # Fall back to the adama toolchain layout if the repo-local binary is missing.
  MICROMAMBA_BIN="${HOME}/.adama-wsl-tools/bin/micromamba"
fi

export MAMBA_ROOT_PREFIX
export HF_HOME
cd "$REPO_ROOT"

echo "[active_hook_integ] Env        : $ENV_NAME"
echo "[active_hook_integ] Model      : $MODEL"
echo "[active_hook_integ] New tokens : $NEW_TOKENS  warmup=$WARMUP  repeats=$REPEATS"
echo "[active_hook_integ] H_ref      : $H_REF  num_bins=$NUM_BINS"
echo "[active_hook_integ] Output dir : $OUTPUT_DIR"
echo ""

"$MICROMAMBA_BIN" run -n "$ENV_NAME" bash -lc \
  "export PYTHONPATH='$REPO_ROOT'; \
   python -m src.experiments.run_active_hook_integration \
     --model '$MODEL' \
     --prompt '$PROMPT' \
     --new-tokens '$NEW_TOKENS' \
     --warmup '$WARMUP' \
     --repeats '$REPEATS' \
     --h-ref '$H_REF' \
     --num-bins '$NUM_BINS' \
     --output-dir '$OUTPUT_DIR'"

echo ""
echo "[active_hook_integ] Done. Results in $REPO_ROOT/$OUTPUT_DIR"
