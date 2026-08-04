#!/usr/bin/env bash
# Watch and pull H800 experiment outputs from WSL.
#
# Uses SSH public-key authentication via ssh-agent or ~/.ssh/config.
# Set COREY_H800_SSH_TARGET to a host alias or user@host before running.
#
# This script never deletes local backups.  It uses rsync without --delete so a
# destroyed remote instance cannot erase already-pulled results.
set -euo pipefail

SSH_TARGET="${COREY_H800_SSH_TARGET:-${SSH_TARGET:-}}"
SSH_PORT="${COREY_H800_SSH_PORT:-${SSH_PORT:-}}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/Corey_Transformer}"
LOCAL_BACKUP_ROOT="${LOCAL_BACKUP_ROOT:-/mnt/c/source/Corey_Transformer/src/outputs/h800_watch_backup}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
MAX_ITERATIONS="${MAX_ITERATIONS:-0}"
REMOTE_PATHS="${REMOTE_PATHS:-src/outputs fa3_h800_run.log}"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[watch-h800] missing command: $1" >&2
    echo "[watch-h800] install openssh-client and rsync in WSL" >&2
    exit 127
  fi
}

need_cmd rsync
need_cmd ssh

if [[ -z "$SSH_TARGET" ]]; then
  echo "[watch-h800] set COREY_H800_SSH_TARGET to an SSH host or ~/.ssh/config alias" >&2
  exit 2
fi

mkdir -p "$LOCAL_BACKUP_ROOT"
SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
)
if [[ -n "$SSH_PORT" ]]; then
  SSH_OPTS+=(-p "$SSH_PORT")
fi
SSH_COMMAND=(ssh "${SSH_OPTS[@]}")
printf -v RSYNC_SSH '%q ' "${SSH_COMMAND[@]}"

sync_once() {
  local changed=0
  local stamp
  stamp="$(date -Is)"
  echo "[$stamp] [watch-h800] syncing from ${SSH_TARGET}:${REMOTE_ROOT}"

  for rel in $REMOTE_PATHS; do
    local remote_path local_path
    if [[ "$rel" = /* ]]; then
      remote_path="$rel"
      local_path="${LOCAL_BACKUP_ROOT}${rel}"
    else
      remote_path="${REMOTE_ROOT}/${rel}"
      local_path="${LOCAL_BACKUP_ROOT}/${rel}"
    fi

    mkdir -p "$(dirname "$local_path")"
    if rsync -avz \
        -e "$RSYNC_SSH" \
        "${SSH_TARGET}:${remote_path}" "$local_path" ; then
      changed=1
    else
      echo "[watch-h800] warning: failed to sync $remote_path; keeping local backup" >&2
    fi
  done

  "${SSH_COMMAND[@]}" "$SSH_TARGET" \
    "cd '$REMOTE_ROOT' 2>/dev/null; date -Is; hostname; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true; find src/outputs -maxdepth 3 -type f 2>/dev/null | sort | tail -100" \
    > "${LOCAL_BACKUP_ROOT}/remote_status.txt" 2> "${LOCAL_BACKUP_ROOT}/remote_status.err" || true

  echo "[$(date -Is)] [watch-h800] sync complete"
  return 0
}

iteration=0
while true; do
  iteration=$((iteration + 1))
  sync_once
  if [[ "$MAX_ITERATIONS" -gt 0 && "$iteration" -ge "$MAX_ITERATIONS" ]]; then
    break
  fi
  sleep "$INTERVAL_SECONDS"
done
