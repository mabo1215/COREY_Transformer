#!/usr/bin/env bash
# TPDS cycle tpds-r3, review item M6b (counter-free substitute for E16b, whose Nsight Compute counters
# are denied in the AutoDL container): E16c times the placement control with a warm and a cold L2
# (e16c_l2_placement.py). Placements vs. same-buffer inputs, L in {8192, 2048}, all six chunks,
# act mode, three rounds. Waits for the GPU timing lock, so it can be launched while E16 runs and
# starts when E16 releases the GPU. A tiny smoke run precedes the full run. ~30-45 min.
#
# Launch:  wsl bash src/revloop/remote.sh launch pro6000 tpds-r3-e16c src/experiments/remote/tpds_lane_r3_e16c.sh
# Markers: src/logs/revloop/{E16c-smoke,E16c}.{START,OK,FAIL}
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
L=(); [ "${TPDS_NO_GPU_LOCK:-0}" = 1 ] || L=(flock "$LOCK")
check_space "$OUT_DIR" 2 || exit 1
lanelog "tpds-r3 E16c lane start on $TAG (waits for the GPU timing lock)"
$RUN E16c-smoke -- "${L[@]}" "$PY" -m src.experiments.tpds.e16c_l2_placement --lengths 512 --chunks 64,256 \
  --inputs gaussian,uniform,real --placements 4 --rounds 2 --warmup 2 --repeats 3 --exclusive-guard \
  --trace-dir "$TRACE_DIR" --out-dir src/experiments/results/tpds/smoke --tag "${TAG}_smoke" \
  || { lanelog "E16c smoke failed; full run skipped"; exit 1; }
$RUN E16c -- "${L[@]}" "$PY" -m src.experiments.tpds.e16c_l2_placement --exclusive-guard \
  --trace-dir "$TRACE_DIR" --out-dir "$OUT_DIR" --tag "$TAG"
lanelog "tpds-r3 E16c lane done"
