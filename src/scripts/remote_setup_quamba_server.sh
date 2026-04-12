#!/usr/bin/env bash
set -euo pipefail

REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-$HOME/COREY_Transformer}"
FALLBACK_REPO_ROOT="/mnt/c/source/COREY_Transformer"
REPO_URL="${REPO_URL:-https://github.com/mabo1215/COREY_Transformer.git}"
QUAMBA_URL="${QUAMBA_URL:-https://github.com/enyac-group/Quamba.git}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

pick_repo_root() {
  if [[ -f "$REMOTE_REPO_ROOT/src/scripts/wsl_setup_quamba_env.sh" ]]; then
    printf '%s\n' "$REMOTE_REPO_ROOT"
    return
  fi

  if [[ -f "$FALLBACK_REPO_ROOT/src/scripts/wsl_setup_quamba_env.sh" ]]; then
    printf '%s\n' "$FALLBACK_REPO_ROOT"
    return
  fi

  mkdir -p "$(dirname "$REMOTE_REPO_ROOT")"
  if [[ ! -d "$REMOTE_REPO_ROOT/.git" ]]; then
    git clone --depth 1 "$REPO_URL" "$REMOTE_REPO_ROOT"
  else
    git -C "$REMOTE_REPO_ROOT" fetch --depth 1 origin main
    git -C "$REMOTE_REPO_ROOT" reset --hard origin/main
  fi
  printf '%s\n' "$REMOTE_REPO_ROOT"
}

REPO_ROOT="$(pick_repo_root)"
QUAMBA_DIR="$REPO_ROOT/Quamba"

if [[ ! -f "$REPO_ROOT/src/scripts/wsl_setup_quamba_env.sh" ]]; then
  printf '[error] setup script missing under %s\n' "$REPO_ROOT" >&2
  exit 1
fi

if [[ ! -f "$QUAMBA_DIR/setup.py" ]]; then
  rm -rf "$QUAMBA_DIR"
  git clone --depth 1 "$QUAMBA_URL" "$QUAMBA_DIR"
fi

export REPO_ROOT
export QUAMBA_DIR
export HF_HOME
export BUILD_THIRD_PARTY="${BUILD_THIRD_PARTY:-1}"
export BUILD_QUAMBA_PACKAGE="${BUILD_QUAMBA_PACKAGE:-1}"

cd "$REPO_ROOT"
bash src/scripts/wsl_setup_quamba_env.sh

if [[ -f "$REPO_ROOT/test_quamba_env.py" ]]; then
  MICROMAMBA_BIN="$REPO_ROOT/.wsl-tools/bin/micromamba"
  if [[ ! -x "$MICROMAMBA_BIN" ]]; then
    MICROMAMBA_BIN="$HOME/.corey-wsl-tools/bin/micromamba"
  fi
  MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.adama-micromamba}" \
    "$MICROMAMBA_BIN" run -n quamba-py310 python "$REPO_ROOT/test_quamba_env.py"
fi