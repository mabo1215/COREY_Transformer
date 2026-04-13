#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-state-spaces/mamba-370m-hf}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_HUB_VERSION="${HF_HUB_VERSION:-0.27.0}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN is not installed." >&2
  exit 1
fi

mkdir -p "$HF_HOME"
export MODEL_ID HF_HOME HF_HUB_VERSION HF_ENDPOINT FORCE_DOWNLOAD
export PATH="$HOME/.local/bin:$PATH"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  "$PYTHON_BIN" - <<'PY'
import importlib.util
import os
import subprocess
import sys

python_bin = sys.executable
module_name = "huggingface_hub"
required_version = os.environ.get("HF_HUB_VERSION", "0.27.0")

if importlib.util.find_spec(module_name) is None:
  print(f"INFO: installing {module_name}=={required_version}", file=sys.stderr)
  subprocess.check_call([
    python_bin,
    "-m",
    "pip",
    "install",
    f"{module_name}=={required_version}",
  ])
PY
fi

echo "INFO: downloading $MODEL_ID into HF_HOME=$HF_HOME via HF_ENDPOINT=$HF_ENDPOINT force=$FORCE_DOWNLOAD"
"$PYTHON_BIN" - <<'PY'
import os

from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
hf_home = os.environ["HF_HOME"]
force_download = os.environ.get("FORCE_DOWNLOAD", "0") == "1"

path = snapshot_download(
    repo_id=model_id,
    repo_type="model",
    resume_download=True,
    force_download=force_download,
    cache_dir=hf_home,
)

print(f"SNAPSHOT_PATH={path}")
PY

CACHE_DIR="$HF_HOME/hub/models--${MODEL_ID//\//--}"

echo "DONE"
du -sh "$CACHE_DIR"
echo "BLOBS: $(find "$CACHE_DIR/blobs" -type f | wc -l)"
echo "SNAPSHOT_FILES: $(find "$CACHE_DIR/snapshots" -type f | wc -l)"
if [ -f "$CACHE_DIR/refs/main" ]; then
  echo "REF_MAIN: $(cat "$CACHE_DIR/refs/main")"
else
  echo "REF_MAIN: missing"
fi