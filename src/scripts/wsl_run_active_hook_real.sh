#!/usr/bin/env bash
set -euo pipefail

# Active-hook real-tensor benchmark runner.
# Captures SSM tensors from a real Mamba-370M forward pass and runs
# the three-policy (off / static-64 / corey) selective-scan benchmark.

REPO_ROOT="${REPO_ROOT:-/mnt/c/source/COREY_Transformer}"
TOOLS_DIR="${TOOLS_DIR:-${HOME}/.corey-wsl-tools}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/.corey-micromamba}"
ENV_NAME="${ENV_NAME:-corey-cuda128}"
REPO_MICROMAMBA_BIN="$REPO_ROOT/.wsl-tools/bin/micromamba"
MICROMAMBA_BIN="$TOOLS_DIR/bin/micromamba"

MODEL="${MODEL:-mamba-370m}"
LAYER_IDX="${LAYER_IDX:-0}"
PROMPT="${PROMPT:-Selective state space models have emerged as a powerful alternative to Transformer architectures for long-sequence modeling. Unlike attention mechanisms that scale quadratically with sequence length, state space models maintain a fixed-size hidden state and process sequences with linear complexity. The Mamba architecture introduces input-dependent state transitions, allowing the model to selectively retain or discard information based on the current input. This selectivity mechanism enables efficient processing of long sequences while preserving relevant context. The entropy of the hidden activations reflects the information content and diversity of the internal representations, which the COREY scheduler uses to adaptively schedule chunk sizes for the selective scan kernel. By estimating Shannon entropy via a fixed-width histogram over the post-convolution hidden states, COREY maps high-entropy activations to larger chunk sizes and low-entropy activations to smaller chunks, reducing the number of kernel launches while maintaining numerical accuracy. Experiments on Mamba-370M demonstrate that the entropy-guided scheduler achieves significant throughput improvements over the static baseline, with the speedup scaling with the entropy regime of the input sequence. The scheduler overhead is bounded analytically and confirmed empirically to be negligible relative to the scan kernel latency.}"
WARMUP_RUNS="${WARMUP_RUNS:-5}"
BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-30}"
OUTPUT_DIR="${OUTPUT_DIR:-src/outputs/active_hook_real_benchmark}"

WINDOWS_USER="${WINDOWS_USER:-$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r')}"
HF_HOME="${HF_HOME:-/mnt/c/Users/${WINDOWS_USER:-$USER}/.cache/huggingface}"

if [[ -x "$REPO_MICROMAMBA_BIN" ]]; then
  MICROMAMBA_BIN="$REPO_MICROMAMBA_BIN"
fi

export MAMBA_ROOT_PREFIX
export HF_HOME
cd "$REPO_ROOT"

echo "[active_hook_real] Environment : $ENV_NAME"
echo "[active_hook_real] Model       : $MODEL  layer $LAYER_IDX"
echo "[active_hook_real] Warmup/Reps : $WARMUP_RUNS / $BENCHMARK_REPEATS"
echo "[active_hook_real] Output dir  : $OUTPUT_DIR"
echo ""

"$MICROMAMBA_BIN" run -n "$ENV_NAME" bash -lc \
  "export PYTHONPATH='$REPO_ROOT'; \
   python -m src.experiments.run_active_hook_real_benchmark \
     --model '$MODEL' \
     --layer-idx '$LAYER_IDX' \
     --prompt '$PROMPT' \
     --warmup-runs '$WARMUP_RUNS' \
     --benchmark-repeats '$BENCHMARK_REPEATS' \
     --sweep-layers \
     --output-dir '$OUTPUT_DIR'"

echo ""
echo "[active_hook_real] Done. Results in $REPO_ROOT/$OUTPUT_DIR"
