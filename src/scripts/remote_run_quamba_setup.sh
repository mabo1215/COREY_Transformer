#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOME="${2:-/home1/mabo1215}"
REMOTE_REPO_ROOT="${1:-$REMOTE_HOME/COREY_Transformer}"
LOG_FILE="${3:-$REMOTE_HOME/quamba_setup.log}"

export REPO_ROOT="$REMOTE_REPO_ROOT"
export QUAMBA_DIR="$REMOTE_REPO_ROOT/Quamba"
export HF_HOME="$REMOTE_HOME/.cache/huggingface"
export BUILD_THIRD_PARTY="${BUILD_THIRD_PARTY:-1}"
export BUILD_QUAMBA_PACKAGE="${BUILD_QUAMBA_PACKAGE:-1}"

set +e
bash -x "$REMOTE_REPO_ROOT/src/scripts/wsl_setup_quamba_env.sh" >"$LOG_FILE" 2>&1
status=$?
set -e

printf '[remote-runner] log=%s status=%s\n' "$LOG_FILE" "$status"
tail -n 120 "$LOG_FILE" || true
exit "$status"