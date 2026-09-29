#!/usr/bin/env bash
# E11 starter: wait until the aria2 download of the E11 checkpoints has ended (ARIA_EXIT line in
# src/logs/tpds/e11_download.log), then run the E11 lane. If the download failed, write the E11 FAIL
# marker (with the reason in E11.log) so that a --shutdown watcher can still power the host off.
#
# Launch:  wsl bash src/revloop/remote.sh launch vgpu3090 tpds-r2-e11 src/experiments/remote/tpds_lane_r2_e11_after_download.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
DL_LOG=$LOG_DIR/e11_download.log
lanelog "E11 starter: waiting for the checkpoint download ($DL_LOG)"
waited=0
until grep -q "^ARIA_EXIT" "$DL_LOG" 2>/dev/null; do
  sleep 60; waited=$((waited + 60))
  if [ "$waited" -ge "${TPDS_WAIT_MAX_S:-14400}" ]; then
    echo "E11: checkpoint download did not finish within ${waited} s" >> "$MARK_DIR/E11.log"; : > "$MARK_DIR/E11.FAIL"; exit 1
  fi
done
rc=$(grep "^ARIA_EXIT" "$DL_LOG" | tail -1 | awk '{print $2}')
if [ "$rc" != 0 ]; then
  lanelog "E11 starter: download failed (aria2 exit $rc)"
  { echo "E11: checkpoint download failed (aria2 exit $rc)"; tail -20 "$DL_LOG"; } >> "$MARK_DIR/E11.log"
  : > "$MARK_DIR/E11.FAIL"; exit 1
fi
lanelog "E11 starter: download complete; launching the E11 lane"
exec bash "$(dirname "${BASH_SOURCE[0]}")/tpds_lane_r2_e11.sh"
