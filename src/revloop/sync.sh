#!/usr/bin/env bash
# Git synchronisation for the revision loop. Brings this machine up to date
# with origin (main repository, paper submodule, ledger, pushed-back results)
# and publishes local progress so any other machine can resume.
# Works in Git Bash on Windows, in WSL, and on Linux.
#
#   bash src/revloop/sync.sh pull
#   bash src/revloop/sync.sh push "<message>" [pathspec ...]
#   bash src/revloop/sync.sh status
#
# Exit codes: 0 ok, 1 network/push failure, 2 conflict that needs a human or agent.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT" || exit 1
HOST=${REVLOOP_HOST:-$(hostname)}
BRANCH=${REVLOOP_BRANCH:-main}
PAPER=paper
MAX_MB=${REVLOOP_MAX_MB:-20}

pick_py() {
  local c
  for c in ${PY:-} python3 python; do
    # Skip the zero-byte Windows Store python3 stub and other broken shims.
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 8))' >/dev/null 2>&1; then
      echo "$c"; return 0
    fi
  done
  echo "sync.sh: no usable python found" >&2
  return 1
}
PY=$(pick_py) || exit 1

has_paper() { [ -e "$PAPER/.git" ]; }

unmerged_result_branches() {
  git branch -r --no-merged "origin/$BRANCH" 2>/dev/null | sed 's/^ *//' | grep -Ei 'result' || true
}

cmd_pull() {
  git fetch origin --prune --quiet || { echo "FETCH FAILED (network?)"; return 1; }
  if ! git pull --rebase --autostash --quiet origin "$BRANCH"; then
    echo "CONFLICT in main repository: resolve, 'git rebase --continue', then rerun"
    return 2
  fi
  if has_paper; then
    git -C "$PAPER" fetch origin --quiet || echo "WARN paper fetch failed"
    if [ "$(git -C "$PAPER" rev-parse --abbrev-ref HEAD)" != "$BRANCH" ]; then
      git -C "$PAPER" checkout --quiet "$BRANCH" || { echo "CONFLICT: cannot switch paper/ to $BRANCH"; return 2; }
    fi
    if ! git -C "$PAPER" pull --rebase --autostash --quiet origin "$BRANCH"; then
      echo "CONFLICT in paper submodule: resolve inside paper/, then rerun"
      return 2
    fi
  else
    git submodule update --init --quiet "$PAPER" || echo "WARN paper submodule not initialised"
  fi
  local b
  for b in $(unmerged_result_branches); do
    echo "UNMERGED_RESULT_BRANCH $b"
  done
  "$PY" src/revloop/revstate.py reconcile
  cmd_status
}

cmd_push() {
  local msg=${1:-}
  [ -n "$msg" ] || { echo "usage: sync.sh push \"<message>\" [pathspec ...]"; return 1; }
  shift
  local specs=("$@")
  [ ${#specs[@]} -gt 0 ] || specs=(docs src .claude/skills .gitignore .gitattributes)
  local tag="[revloop@$HOST]"

  # 1) paper submodule first, so the main repository can record its new pointer.
  if has_paper; then
    if [ -n "$(git -C "$PAPER" status --porcelain)" ]; then
      git -C "$PAPER" add -A
      git -C "$PAPER" commit --quiet -m "$msg $tag"
    fi
    if [ -n "$(git -C "$PAPER" log --oneline "origin/$BRANCH..HEAD" 2>/dev/null)" ]; then
      git -C "$PAPER" pull --rebase --quiet origin "$BRANCH" || { echo "CONFLICT in paper submodule"; return 2; }
      git -C "$PAPER" push --quiet origin "HEAD:$BRANCH" || { echo "PAPER PUSH FAILED"; return 1; }
      echo "paper pushed: $(git -C "$PAPER" log -1 --format='%h %s')"
    fi
  fi

  # 2) main repository.
  # A missing pathspec makes 'git add' reject the whole call, so pass only paths that exist.
  local existing=() spec
  for spec in "${specs[@]}"; do
    [ -e "$spec" ] && existing+=("$spec")
  done
  [ ${#existing[@]} -gt 0 ] && git add -A -- "${existing[@]}"
  has_paper && git add "$PAPER"
  # Result files declared in the ledger sync even under ignored directories
  # (e.g. outputs/); the size guard below still drops oversized ones.
  local f size
  while IFS= read -r f; do
    [ -f "$f" ] && git add -f -- "$f"
  done < <("$PY" src/revloop/revstate.py results --status finished,pulled,integrated 2>/dev/null)
  while IFS= read -r f; do
    [ -f "$f" ] || continue
    size=$(wc -c < "$f")
    if [ "$size" -gt $((MAX_MB * 1024 * 1024)) ]; then
      git reset --quiet -- "$f"
      echo "SKIPPED large file ($((size / 1048576)) MB > ${MAX_MB} MB): $f"
    fi
  done < <(git diff --cached --name-only --diff-filter=AM)
  if git diff --cached --quiet; then
    echo "nothing new to commit"
  else
    git commit --quiet -m "$msg $tag"
  fi
  local attempt
  for attempt in 1 2 3; do
    if ! git pull --rebase --autostash --quiet origin "$BRANCH"; then
      echo "CONFLICT while rebasing onto origin/$BRANCH: resolve, 'git rebase --continue', then rerun push"
      return 2
    fi
    if git push --quiet origin "HEAD:$BRANCH"; then
      echo "pushed: $(git log -1 --format='%h %s')"
      return 0
    fi
    sleep 3
  done
  echo "PUSH FAILED after 3 attempts"
  return 1
}

cmd_status() {
  local counts
  counts=$(git rev-list --left-right --count "origin/$BRANCH...HEAD" 2>/dev/null || echo "? ?")
  echo "repo: $(git log -1 --format='%h %s') | behind/ahead origin: ${counts/$'\t'/\/} | dirty files: $(git status --porcelain | wc -l | tr -d ' ')"
  if has_paper; then
    echo "paper: $(git -C "$PAPER" log -1 --format='%h %s') | dirty files: $(git -C "$PAPER" status --porcelain | wc -l | tr -d ' ')"
    local recorded
    recorded=$(git ls-tree HEAD "$PAPER" | awk '{print $3}')
    if [ -n "$recorded" ] && [ "$recorded" != "$(git -C "$PAPER" rev-parse HEAD)" ]; then
      echo "paper: checkout differs from the pointer recorded in the main repo (${recorded:0:7}); next push records it"
    fi
  fi
}

case "${1:-}" in
  pull)   shift; cmd_pull "$@" ;;
  push)   shift; cmd_push "$@" ;;
  status) shift; cmd_status "$@" ;;
  *) echo "usage: sync.sh pull | push \"<message>\" [pathspec ...] | status"; exit 1 ;;
esac
