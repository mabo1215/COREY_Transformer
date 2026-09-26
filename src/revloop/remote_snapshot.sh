#!/usr/bin/env bash
# Push growing partial result files of a running experiment at a fixed interval, so that
# a host that may disappear before the experiment ends (rental end, preemption) still
# leaves its completed rows in git. Stops when the experiment's OK/FAIL marker appears;
# the watcher then pushes the final files to main as usual.
#
# The snapshots are committed in a separate git worktree and force-pushed to the branch
# revloop-snapshot-<host>-<EXP_ID>. The main working tree is never touched: an earlier
# version pulled into it, git replaced the result file that the running experiment held
# open, and every later row went to the replaced (unlinked) file.
#
# Recover a snapshot:  git fetch origin revloop-snapshot-<host>-<EXP_ID> &&
#                      git show FETCH_HEAD:<file> > <file>
#
# Usage (on the GPU host, from the repo root, usually in its own screen):
#   bash src/revloop/remote_snapshot.sh <EXP_ID> <interval_s> <file> [file ...]
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
ID=${1:?experiment id}; INTERVAL=${2:?interval seconds}; shift 2
FILES=("$@")
HOST=${REVLOOP_HOST:-$(hostname)}
PY=${PY:-python3}
DIR=src/logs/revloop
LOG=$DIR/snapshot_$ID.log
MAX_MB=20
SNAP_BRANCH="revloop-snapshot-$HOST-$ID"
mkdir -p "$DIR"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
proxy=$("$PY" src/revloop/revstate.py host show "$HOST" --field git_proxy 2>/dev/null)
if [ -n "$proxy" ]; then eval "$proxy" >/dev/null 2>&1 || true; fi

WT=$(mktemp -d)/wt
git worktree add --quiet --detach "$WT" HEAD || { log "ERROR cannot create worktree"; exit 1; }
trap 'git worktree remove --force "$WT" >/dev/null 2>&1' EXIT

last=""
log "snapshot loop for $ID every ${INTERVAL}s to origin/$SNAP_BRANCH: ${FILES[*]}"
while true; do
  for m in OK FAIL; do
    if [ -f "$DIR/$ID.$m" ]; then log "marker $ID.$m present; stopping (watcher pushes the final files)"; exit 0; fi
  done
  sig=$(for f in "${FILES[@]}"; do [ -f "$f" ] && wc -c < "$f"; done | tr '\n' ' ')
  if [ -n "${sig// /}" ] && [ "$sig" != "$last" ]; then
    added=0
    for f in "${FILES[@]}"; do
      [ -f "$f" ] || continue
      [ "$(wc -c < "$f")" -gt $((MAX_MB * 1024 * 1024)) ] && { log "WARN skip oversized $f"; continue; }
      mkdir -p "$WT/$(dirname "$f")"
      cp -- "$f" "$WT/$f" && git -C "$WT" add -f -- "$f" && added=1
    done
    if [ "$added" = 1 ] && git -C "$WT" -c user.name="revloop-$HOST" -c user.email="revloop@$HOST" \
         commit --quiet -m "results: $ID partial snapshot on $HOST [revloop@$HOST]"; then
      if timeout 180 git -C "$WT" push --quiet --force origin "HEAD:refs/heads/$SNAP_BRANCH"; then
        log "pushed snapshot ($sig bytes) to origin/$SNAP_BRANCH: $(git -C "$WT" log -1 --format='%h')"
      else
        log "WARN snapshot push failed; will retry next interval"
      fi
    fi
    last=$sig
  fi
  sleep "$INTERVAL"
done
