#!/usr/bin/env bash
# Run one ledger experiment on a remote host with a log and an OK/FAIL marker,
# so that remote_watch.sh and local patrols can tell when it has finished.
#
#   bash src/revloop/remote_run.sh <EXP_ID> -- <command> [args ...]
#
# Output : src/logs/revloop/<EXP_ID>.log
# Markers: src/logs/revloop/<EXP_ID>.START (UTC start time)
#          src/logs/revloop/<EXP_ID>.OK    (on success)
#          src/logs/revloop/<EXP_ID>.FAIL  (exit code, on failure)
# A lane script is simply a sequence of these calls; run each lane in its own screen.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT" || exit 1

id=${1:-}
[ -n "$id" ] || { echo "usage: remote_run.sh <EXP_ID> -- <command> [args ...]"; exit 2; }
shift
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "remote_run.sh: no command given for $id"; exit 2; }

dir=src/logs/revloop
mkdir -p "$dir"
rm -f "$dir/$id.OK" "$dir/$id.FAIL"
date -u +%Y-%m-%dT%H:%M:%SZ > "$dir/$id.START"
echo "[$(date '+%F %T')] START $id: $*" | tee -a "$dir/lanes.log"

"$@" > "$dir/$id.log" 2>&1
rc=$?

if [ $rc -eq 0 ]; then
  touch "$dir/$id.OK"
  echo "[$(date '+%F %T')] OK    $id" | tee -a "$dir/lanes.log"
else
  echo "$rc" > "$dir/$id.FAIL"
  echo "[$(date '+%F %T')] FAIL  $id (exit $rc, see $dir/$id.log)" | tee -a "$dir/lanes.log"
fi
exit $rc
