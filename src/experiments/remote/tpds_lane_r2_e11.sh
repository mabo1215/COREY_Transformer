#!/usr/bin/env bash
# TPDS cycle tpds-r2, E11 after the main lane: waits until E10 has ended (OK or FAIL), so that
# the trace capture and the chunk sweep of the larger checkpoints never overlap the timing of
# E8-E10, then runs the E11 step of tpds_lane_r2.sh.
#
# Launch:  wsl bash src/revloop/remote.sh launch vgpu3090 tpds-r2-e11 src/experiments/remote/tpds_lane_r2_e11.sh
# Markers: src/logs/revloop/E11.{START,OK,FAIL}
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
P=""; [ "${TPDS_SMOKE:-0}" = 1 ] && P=smoke-
lanelog "E11 lane: waiting for ${P}E10 to end"
waited=0
until [ -f "$MARK_DIR/${P}E10.OK" ] || [ -f "$MARK_DIR/${P}E10.FAIL" ]; do
  sleep 60; waited=$((waited + 60))
  if [ "$waited" -ge "${TPDS_WAIT_MAX_S:-43200}" ]; then lanelog "E11 lane: timeout waiting for E10"; exit 1; fi
done
TPDS_R2_ONLY=E11 bash "$(dirname "${BASH_SOURCE[0]}")/tpds_lane_r2.sh"
