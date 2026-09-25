#!/usr/bin/env bash
# Remote-side watcher for a GPU session (run it in its own screen on the host).
# Every INTERVAL seconds it checks the remote_run.sh markers of the given
# experiments. When one finishes or fails it updates that experiment's ledger
# file, commits the declared result files, and pushes to origin (falling back
# to a dedicated results branch if main rejects the push). When all
# experiments are done and --shutdown is set, it powers the host off.
#
#   bash src/revloop/remote_watch.sh [--shutdown] [--interval 600] <EXP_ID> [<EXP_ID> ...]
#
# Env: REVLOOP_HOST  registry name of this host (e.g. h800); remote.sh sets it.
#      PY            python interpreter (default: host 'python' field, else python3).
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT" || exit 1
HOST=${REVLOOP_HOST:-$(hostname)}
export REVLOOP_HOST=$HOST
BRANCH=${REVLOOP_BRANCH:-main}
MAX_MB=${REVLOOP_MAX_MB:-20}
INTERVAL=600
SHUTDOWN=0
IDS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --shutdown) SHUTDOWN=1 ;;
    --interval) INTERVAL=$2; shift ;;
    *) IDS+=("$1") ;;
  esac
  shift
done
[ ${#IDS[@]} -gt 0 ] || { echo "usage: remote_watch.sh [--shutdown] [--interval S] <EXP_ID> ..."; exit 2; }

PY=${PY:-python3}
command -v "$PY" >/dev/null 2>&1 || PY=python
state() { "$PY" src/revloop/revstate.py "$@"; }
host_field() { state host show "$HOST" --field "$1" 2>/dev/null; }
cfg_py=$(host_field python)
[ -n "$cfg_py" ] && [ -x "$cfg_py" ] && PY=$cfg_py

DIR=src/logs/revloop
mkdir -p "$DIR"
LOG=$DIR/watch.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

proxy=$(host_field git_proxy)
if [ -n "$proxy" ]; then eval "$proxy" >/dev/null 2>&1 || true; fi

push_results() {  # push_results <message>; returns 0 if main or a fallback branch accepted it
  local msg=$1 attempt branch
  git -c user.name="revloop-$HOST" -c user.email="revloop@$HOST" commit --quiet -m "$msg [revloop@$HOST]" \
    || { log "nothing to commit"; return 0; }
  for attempt in 1 2 3; do
    # -X theirs keeps this host's own result/state changes when replaying onto origin.
    if git -c user.name="revloop-$HOST" -c user.email="revloop@$HOST" pull --rebase --autostash -X theirs --quiet origin "$BRANCH" \
       && timeout 180 git push --quiet origin "HEAD:$BRANCH"; then
      log "pushed to origin/$BRANCH: $(git log -1 --format='%h %s')"
      return 0
    fi
    git rebase --abort >/dev/null 2>&1 || true
    sleep 20
  done
  branch="revloop-results-$HOST-$(date '+%Y%m%d-%H%M%S')"
  if timeout 180 git push --quiet origin "HEAD:refs/heads/$branch"; then
    log "WARN main rejected the push; results preserved on origin/$branch"
    return 0
  fi
  log "ERROR could not push results (main or fallback branch)"
  return 1
}

finalize() {  # finalize <id> <finished|failed>
  local id=$1 status=$2 note pattern
  # Start from the newest ledger so this update never replaces a fuller record.
  git pull --rebase --autostash --quiet origin "$BRANCH" || { git rebase --abort >/dev/null 2>&1; log "WARN pull before finalize failed"; }
  [ -f "docs/state/experiments/$id.json" ] || log "WARN no ledger file for $id on this host; writing a minimal record"
  if [ "$status" = finished ]; then
    note="remote: finished on $HOST (marker OK)"
  else
    note="remote: FAILED on $HOST exit $(cat "$DIR/$id.FAIL" 2>/dev/null); log tail: $(tail -n 3 "$DIR/$id.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
  fi
  state exp set "$id" --status "$status" --note "$note" >/dev/null
  git add -- "docs/state/experiments/$id.json"
  # Declared result files are force-added, so they sync even when they live under
  # an ignored directory (for example outputs/); oversized files are skipped.
  local f size
  while IFS= read -r f; do
    [ -f "$f" ] || continue
    size=$(wc -c < "$f")
    if [ "$size" -gt $((MAX_MB * 1024 * 1024)) ]; then
      log "WARN result file skipped ($((size / 1048576)) MB > ${MAX_MB} MB): $f"
      continue
    fi
    git add -f -- "$f" || log "WARN could not add $f"
  done < <(state results --exp "$id")
  log "$id -> $status"
  push_results "results: $id $status on $HOST"
}

log "watch started on $HOST: ${IDS[*]} (interval ${INTERVAL}s, shutdown=$SHUTDOWN)"
declare -A DONE=()
push_ok=1
while true; do
  for id in "${IDS[@]}"; do
    [ -n "${DONE[$id]:-}" ] && continue
    if [ -f "$DIR/$id.OK" ]; then
      finalize "$id" finished || push_ok=0
      DONE[$id]=1
    elif [ -f "$DIR/$id.FAIL" ]; then
      finalize "$id" failed || push_ok=0
      DONE[$id]=1
    fi
  done
  if [ ${#DONE[@]} -eq ${#IDS[@]} ]; then
    log "all watched experiments are done"
    break
  fi
  log "waiting: $(( ${#IDS[@]} - ${#DONE[@]} )) of ${#IDS[@]} still running; next check in ${INTERVAL}s"
  sleep "$INTERVAL"
done

if [ "$SHUTDOWN" = 1 ]; then
  persists=$(host_field disk_persists)
  if [ "$push_ok" != 1 ] && [ "$persists" != "True" ] && [ "$persists" != "true" ]; then
    log "NOT shutting down: results were not pushed and this host's disk may not persist"
    exit 1
  fi
  cmd=$(host_field shutdown_cmd)
  cmd=${cmd:-shutdown -h now}
  log "shutting down: $cmd"
  eval "$cmd"
fi
