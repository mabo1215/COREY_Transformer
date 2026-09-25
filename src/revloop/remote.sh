#!/usr/bin/env bash
# Local-side helper for the remote GPU hosts registered in docs/state/hosts.json.
# Repository rule: remote work is WSL-first, so on Windows call it via WSL:
#
#   wsl bash src/revloop/remote.sh ping     <host>
#   wsl bash src/revloop/remote.sh exec     <host> <local-script.sh> [args ...]
#   wsl bash src/revloop/remote.sh sync     <host>              # git pull on the host
#   wsl bash src/revloop/remote.sh launch   <host> <screen> <lane-script> [args ...]
#   wsl bash src/revloop/remote.sh watch    <host> [--shutdown] <EXP_ID> ...
#   wsl bash src/revloop/remote.sh patrol   <host>
#   wsl bash src/revloop/remote.sh fetch    <host> [EXP_ID ...]  # rsync results back
#   wsl bash src/revloop/remote.sh shutdown <host> --yes [--force]
#
# <host> is a registry name (h800, pro6000, ...). Paths of lane scripts are
# relative to the repository root on the remote host.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT" || exit 1

pick_py() {
  local c
  for c in ${PY:-} python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 8))' >/dev/null 2>&1; then
      echo "$c"; return 0
    fi
  done
  echo "remote.sh: no usable python found" >&2
  return 1
}
LPY=$(pick_py) || exit 1
state() { "$LPY" src/revloop/revstate.py "$@"; }

cmd=${1:-}; name=${2:-}
[ -n "$cmd" ] && [ -n "$name" ] || { sed -n '2,15p' "$0"; exit 2; }
shift 2

field() { state host show "$name" --field "$1" 2>/dev/null; }
state host show "$name" >/dev/null || exit 2
ALIAS=$(field ssh_alias)
RROOT=$(field root)
RPY=$(field python)
PROXY=$(field git_proxy)
[ -n "$ALIAS" ] && [ -n "$RROOT" ] || {
  echo "host '$name' needs ssh_alias and root in docs/state/hosts.json:"
  echo "  python src/revloop/revstate.py host set $name ssh_alias=<alias> root=<repo path> python=<python path>"
  exit 2
}
SSH_OPTS=(-o ConnectTimeout=20 -o ServerAliveInterval=30 -o BatchMode=yes)

# Environment prefix for every remote shell: repo root, host name, python, git proxy.
prefix() {
  printf 'cd %q || exit 1; export REVLOOP_HOST=%q; ' "$RROOT" "$name"
  [ -n "$RPY" ] && printf 'export PY=%q; ' "$RPY"
  [ -n "$PROXY" ] && printf '%s >/dev/null 2>&1 || true; ' "$PROXY"
}

run_script() {  # run_script <local-script> [args ...]
  local script=$1; shift
  [ -f "$script" ] || { echo "no such script: $script"; return 2; }
  # shellcheck disable=SC2029  # expanded locally on purpose
  ssh "${SSH_OPTS[@]}" "$ALIAS" "$(prefix) bash -s -- $(printf '%q ' "$@")" < "$script"
}

case "$cmd" in
  ping)
    ssh "${SSH_OPTS[@]}" "$ALIAS" "$(prefix) echo \"reachable: \$(hostname)\"; git log -1 --format='repo: %h %s'; nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'no GPU visible (no-GPU mode?)'; df -h . | tail -1"
    ;;
  exec)
    run_script "$@"
    ;;
  sync)
    ssh "${SSH_OPTS[@]}" "$ALIAS" "$(prefix) git pull --rebase --autostash origin main && git log -1 --format='remote repo now at %h %s'"
    ;;
  launch)
    screen_name=${1:?screen name}; lane=${2:?lane script}; shift 2
    ssh "${SSH_OPTS[@]}" "$ALIAS" "$(prefix) test -f $(printf '%q' "$lane") || { echo 'missing lane script on host (push + remote.sh sync first)'; exit 2; }; screen -dmS $(printf '%q' "$screen_name") bash -lc $(printf '%q' "export REVLOOP_HOST=$name; ${RPY:+export PY=$RPY; }bash $lane $*") && echo launched $screen_name && screen -ls | grep -F $(printf '%q' "$screen_name")"
    ;;
  watch)
    [ $# -gt 0 ] || { echo "watch needs experiment ids"; exit 2; }
    ssh "${SSH_OPTS[@]}" "$ALIAS" "$(prefix) screen -dmS revloop-watch bash -lc $(printf '%q' "export REVLOOP_HOST=$name; ${RPY:+export PY=$RPY; }bash src/revloop/remote_watch.sh $*") && echo 'watcher started (screen revloop-watch)'"
    ;;
  patrol)
    tmp=$(mktemp)
    {
      echo 'echo "== time: $(date "+%F %T %Z")"'
      echo 'echo "== screens"; screen -ls 2>/dev/null || true'
      echo 'echo "== gpu"; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "no GPU visible (no-GPU mode or powered off)"'
      echo 'echo "== disk"; df -h . | tail -1'
      echo 'echo "== watcher"; tail -n 3 src/logs/revloop/watch.log 2>/dev/null || echo "(no watcher log)"'
      state exp list --host "$name" --status planned,prepped,queued,running,finished,failed --format tsv |
      while IFS=$'\t' read -r id status screen_name logpath; do
        printf 'echo "== exp %s (ledger: %s, screen: %s)"\n' "$id" "$status" "$screen_name"
        printf 'for m in START OK FAIL; do [ -f src/logs/revloop/%s.$m ] && echo "marker $m $(cat src/logs/revloop/%s.$m)"; done\n' "$id" "$id"
        printf 'tail -n 15 %q 2>/dev/null || echo "(no log yet)"\n' "$logpath"
      done
    } > "$tmp"
    run_script "$tmp"
    rc=$?
    rm -f "$tmp"
    exit $rc
    ;;
  fetch)
    ids=("$@")
    if [ ${#ids[@]} -eq 0 ]; then
      mapfile -t ids < <(state exp list --host "$name" --status running,finished --format tsv | cut -f1)
    fi
    for id in "${ids[@]}"; do
      while IFS= read -r pattern; do
        [ -n "$pattern" ] || continue
        echo "fetch $id: $pattern"
        rsync -azR -e "ssh ${SSH_OPTS[*]}" "$ALIAS:$RROOT/./$pattern" ./ || echo "WARN rsync failed for $pattern"
      done < <(state exp show "$id" --field results)
    done
    state reconcile
    ;;
  shutdown)
    yes=0; force=0
    for a in "$@"; do
      case "$a" in --yes) yes=1 ;; --force) force=1 ;; esac
    done
    [ "$yes" = 1 ] || { echo "refusing: pass --yes once the user has authorised power-off"; exit 2; }
    active=$(state exp list --host "$name" --status queued,running,finished --format tsv | cut -f1 | tr '\n' ' ')
    if [ -n "${active// /}" ] && [ "$force" != 1 ]; then
      echo "refusing: experiments still queued/running or not pulled on $name: $active(use --force if results are safe)"
      exit 2
    fi
    sd=$(field shutdown_cmd); sd=${sd:-shutdown -h now}
    ssh "${SSH_OPTS[@]}" "$ALIAS" "$sd" ; echo "shutdown issued to $name ($sd)"
    state host set "$name" state=off >/dev/null
    ;;
  *)
    sed -n '2,15p' "$0"; exit 2 ;;
esac
