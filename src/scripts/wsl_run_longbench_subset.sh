#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/c/source/COREY_Transformer}"
TOOLS_DIR="${TOOLS_DIR:-${HOME}/.corey-wsl-tools}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/.corey-micromamba}"
ENV_NAME="${ENV_NAME:-corey-cuda128}"
REPO_MICROMAMBA_BIN="$REPO_ROOT/.wsl-tools/bin/micromamba"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-$TOOLS_DIR/bin/micromamba}"
MODEL="${MODEL:-mamba-1.4b}"
PRECISION="${PRECISION:-fp16}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
TASKS="${TASKS:-narrativeqa qasper multifieldqa_en gov_report}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/src/data/longbench_subset}"
DATASET_SOURCE="${DATASET_SOURCE:-local}"
DATASET_NAME="${DATASET_NAME:-zai-org/LongBench}"
OUTPUT_DIR="${OUTPUT_DIR:-src/outputs/hf_mamba_longbench_subset_wsl_14b}"
WINDOWS_USER="${WINDOWS_USER:-$(cmd.exe /c \"echo %USERNAME%\" 2>/dev/null | tr -d '\r')}"
HF_HOME="${HF_HOME:-/mnt/c/Users/${WINDOWS_USER:-$USER}/.cache/huggingface}"
ZIP_URL="${LONG_BENCH_ZIP_URL:-https://hf-mirror.com/datasets/zai-org/LongBench/resolve/main/data.zip}"
ZIP_CACHE="${ZIP_CACHE:-$REPO_ROOT/src/data/.cache/longbench_data.zip}"

if [[ -x "$REPO_MICROMAMBA_BIN" ]]; then
  MICROMAMBA_BIN="$REPO_MICROMAMBA_BIN"
fi

mkdir -p "$DATASET_ROOT" "$(dirname "$ZIP_CACHE")"
export MAMBA_ROOT_PREFIX
export HF_HOME

needs_extract=0
for task in $TASKS; do
  if [[ ! -f "$DATASET_ROOT/$task/test.jsonl" ]]; then
    needs_extract=1
    break
  fi
done

if [[ "$needs_extract" -eq 1 ]]; then
  if [[ ! -f "$ZIP_CACHE" ]]; then
    curl -L "$ZIP_URL" -o "$ZIP_CACHE"
  fi
  "$MICROMAMBA_BIN" run -n "$ENV_NAME" python - "$ZIP_CACHE" "$DATASET_ROOT" $TASKS <<'PY'
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
dataset_root = Path(sys.argv[2])
tasks = sys.argv[3:]

with zipfile.ZipFile(zip_path) as archive:
    names = archive.namelist()
    for task in tasks:
        target_dir = dataset_root / task
        target_dir.mkdir(parents=True, exist_ok=True)
        candidates = [
            f"data/{task}.jsonl",
            f"LongBench/data/{task}.jsonl",
            f"{task}.jsonl",
        ]
        matched = next((name for name in candidates if name in names), None)
        if matched is None:
            raise FileNotFoundError(f"Unable to find {task}.jsonl inside {zip_path}")
        with archive.open(matched) as src, (target_dir / "test.jsonl").open("wb") as dst:
            dst.write(src.read())
PY
fi

cd "$REPO_ROOT"
"$MICROMAMBA_BIN" run -n "$ENV_NAME" bash -lc "export PYTHONPATH='$REPO_ROOT'; export HF_HOME='$HF_HOME'; python -m src.experiments.run_longbench_inference \
  --model '$MODEL' \
  --dataset-root '$DATASET_ROOT' \
  --dataset-source '$DATASET_SOURCE' \
  --dataset-name '$DATASET_NAME' \
  --tasks $TASKS \
  --max-samples '$MAX_SAMPLES' \
  --precision '$PRECISION' \
  --device '$DEVICE' \
  --dtype '$DTYPE' \
  --disable-entropy-hook \
  --eval-perplexity \
  --ppl-max-samples 1 \
  --lm-datasets wikitext103 \
  --lm-max-samples 1 \
  --max-length 8192 \
  --output-dir '$OUTPUT_DIR'"