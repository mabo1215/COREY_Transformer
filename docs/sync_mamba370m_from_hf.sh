#!/usr/bin/env bash
set -euo pipefail

# Optional overrides:
#   MODEL_ID=state-spaces/mamba-370m-hf HF_HOME=~/.cache/huggingface ./sync_mamba370m_from_hf.sh
MODEL_ID="${MODEL_ID:-state-spaces/mamba-370m-hf}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is not installed." >&2
  exit 1
fi

mkdir -p "$HF_HOME"

python3 - <<'PY'
import os
import sys

try:
    from huggingface_hub import snapshot_download
except Exception as exc:
    print(f"ERROR: huggingface_hub is required: {exc}", file=sys.stderr)
    sys.exit(1)

model_id = os.environ.get("MODEL_ID", "state-spaces/mamba-370m-hf")

# Keep all files for full offline cache compatibility.
path = snapshot_download(
    repo_id=model_id,
    repo_type="model",
    resume_download=True,
)

print(f"SNAPSHOT_PATH={path}")
PY

CACHE_DIR="$HF_HOME/hub/models--${MODEL_ID//\//--}"

echo "DONE"
if [ -d "$CACHE_DIR" ]; then
  du -sh "$CACHE_DIR"
  if [ -d "$CACHE_DIR/blobs" ]; then
    echo "BLOBS: $(find "$CACHE_DIR/blobs" -type f | wc -l)"
  else
    echo "BLOBS: 0"
  fi
  if [ -d "$CACHE_DIR/snapshots" ]; then
    echo "SNAPSHOT_FILES: $(find "$CACHE_DIR/snapshots" -type f | wc -l)"
  else
    echo "SNAPSHOT_FILES: 0"
  fi
  if [ -f "$CACHE_DIR/refs/main" ]; then
    echo "REF_MAIN: $(cat "$CACHE_DIR/refs/main")"
  else
    echo "REF_MAIN: missing"
  fi
else
  echo "ERROR: cache directory not found: $CACHE_DIR" >&2
  exit 2
fi
