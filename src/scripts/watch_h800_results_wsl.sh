#!/usr/bin/env bash
# Watch and pull H800 experiment outputs from WSL.
#
# Reads SSH command and password from /mnt/c/source/.env by default:
#   line N:   ssh -p <port> <user>@<host>
#   line N+1: password
#
# This script never deletes local backups.  It uses rsync without --delete so a
# destroyed remote instance cannot erase already-pulled results.
set -euo pipefail

ENV_PATH="${ENV_PATH:-/mnt/c/source/.env}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/Corey_Transformer}"
LOCAL_BACKUP_ROOT="${LOCAL_BACKUP_ROOT:-/mnt/c/source/Corey_Transformer/src/outputs/h800_watch_backup}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
MAX_ITERATIONS="${MAX_ITERATIONS:-0}"
REMOTE_PATHS="${REMOTE_PATHS:-src/outputs fa3_h800_run.log}"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[watch-h800] missing command: $1" >&2
    echo "[watch-h800] install in WSL: sudo apt-get update && sudo apt-get install -y sshpass rsync openssh-client" >&2
    exit 127
  fi
}

need_cmd sshpass
need_cmd rsync
need_cmd ssh

if [[ ! -f "$ENV_PATH" ]]; then
  echo "[watch-h800] env file not found: $ENV_PATH" >&2
  exit 2
fi

SSH_LINE="$(grep -m1 '^ssh ' "$ENV_PATH" || true)"
if [[ -z "$SSH_LINE" ]]; then
  echo "[watch-h800] no ssh line found in $ENV_PATH" >&2
  exit 2
fi

PASSWORD="$(awk -v ssh_line="$SSH_LINE" '
  $0 == ssh_line { getline; print; exit }
' "$ENV_PATH")"

PORT="$(printf '%s\n' "$SSH_LINE" | sed -nE 's/^ssh[[:space:]]+-p[[:space:]]+([0-9]+)[[:space:]]+.*$/\1/p')"
USER_HOST="$(printf '%s\n' "$SSH_LINE" | sed -nE 's/^ssh[[:space:]]+-p[[:space:]]+[0-9]+[[:space:]]+([^[:space:]]+).*$/\1/p')"

if [[ -z "$PORT" || -z "$USER_HOST" || -z "$PASSWORD" ]]; then
  echo "[watch-h800] failed to parse ssh credentials from $ENV_PATH" >&2
  exit 2
fi

mkdir -p "$LOCAL_BACKUP_ROOT"
SSH_OPTS="-p ${PORT} -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=3"

sync_once() {
  local changed=0
  local stamp
  stamp="$(date -Is)"
  echo "[$stamp] [watch-h800] syncing from ${USER_HOST}:${REMOTE_ROOT}"

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
    if sshpass -p "$PASSWORD" rsync -avz \
        -e "ssh ${SSH_OPTS}" \
        "${USER_HOST}:${remote_path}" "$local_path" ; then
      changed=1
    else
      echo "[watch-h800] warning: failed to sync $remote_path; keeping local backup" >&2
    fi
  done

  sshpass -p "$PASSWORD" ssh $SSH_OPTS "$USER_HOST" \
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
