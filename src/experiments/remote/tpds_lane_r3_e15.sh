#!/usr/bin/env bash
# TPDS cycle tpds-r3, E15 (author 2026-10-01): both timing conventions on the RTX 3090.
#   "launch-inclusive": back-to-back calls timed with CUDA events (the protocol of E1-E11); for
#                       calls shorter than the host launch cost (~40-90 us) it includes that cost.
#   "device time":      the same with a GPU-side pre-spin in front of every timing block
#                       (TPDS_PRE_SPIN_MS / --pre-spin-ms), so only GPU execution is timed.
# E14 timed the 1.4B and 2.8B calls in device time. E15 adds, in this order:
#   1. the 1.4B and 2.8B calls again, launch-inclusive (comparable with the 370M results);
#   2. the 370M Mamba-1 calibration data of the cost model (E4 Mamba-1 launch sweep, E1 Mamba-1
#      medians) and the 2360 captured 370M calls, in device time.
# Starts when E14 has finished (marker E14.OK/FAIL), so it never overlaps E14's captures.
#
# Launch: screen -dmS tpds-r3-e15 bash -lc "export REVLOOP_HOST=vgpu3090; bash src/experiments/remote/tpds_lane_r3_e15.sh"
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
L=(); [ "${TPDS_NO_GPU_LOCK:-0}" = 1 ] || L=(flock "$LOCK")
P=""; [ "${TPDS_SMOKE:-0}" = 1 ] && P=smoke-
check_space "$OUT_DIR" 2 || exit 1
SPIN=${TPDS_E15_SPIN_MS:-20}
if [ "${TPDS_SMOKE:-0}" = 1 ]; then
  E2A="--batches 1 --warmup 2 --repeats 3 --max-rows 4"
  E4A="--parts m1 --lengths 128 --batches 1 --search-warmup 1 --search-repeats 2 --warmup 2 --repeats 3"
  E1A="--families mamba1 --lengths 512 --warmup 2 --repeats 3 --rounds 1 --inputs gaussian,real"
  BIG="${TPDS_E15_BIG:-}"
else
  E2A="--batches 1,4,16 --warmup 10 --repeats 50"
  E4A="--parts m1 --lengths 128,512,2048,8192 --batches 1,4,16 --search-warmup 3 --search-repeats 10 --warmup 10 --repeats 100"
  E1A="--families mamba1 --lengths 512,2048,8192 --batch 1 --warmup 10 --repeats 100 --rounds 3"
  BIG="1p4b 2p8b"
  wait_marker_any() {  # E14 may end OK or FAIL; either way it no longer uses the GPU
    local waited=0
    until [ -f "$MARK_DIR/E14.OK" ] || [ -f "$MARK_DIR/E14.FAIL" ]; do
      sleep 60; waited=$((waited + 60)); [ "$waited" -ge 43200 ] && return 1
    done
  }
  lanelog "E15: waiting for E14 to finish"
  wait_marker_any || { lanelog "E15: E14 did not finish within 12 h"; exit 1; }
fi
lanelog "tpds-r3 E15 start on $TAG (device-time spin ${SPIN} ms)"
# shellcheck disable=SC2086
$RUN ${P}E15 -- bash -c "set -e
  for tag in $BIG; do
    ${L[*]} $PY -m src.experiments.tpds.e2_oracle_sweep --model mamba1 $E2A --pre-spin-ms 0 \
        --trace-dir src/outputs/tpds_traces_\$tag --out-dir '$OUT_DIR' --tag ${TAG}_\${tag}_r3plain
  done
  export TPDS_PRE_SPIN_MS=$SPIN
  ${L[*]} $PY -m src.experiments.tpds.e4_autotune $E4A --trace-dir '$TRACE_DIR' --out-dir '$OUT_DIR' --tag ${TAG}_r3spin
  ${L[*]} $PY -m src.experiments.tpds.e1_value_invariance $E1A --trace-dir '$TRACE_DIR' --out-dir '$OUT_DIR' --tag ${TAG}_r3spin
  ${L[*]} $PY -m src.experiments.tpds.e2_oracle_sweep --model mamba1 $E2A --pre-spin-ms $SPIN \
      --trace-dir '$TRACE_DIR' --out-dir '$OUT_DIR' --tag ${TAG}_r3spin"
lanelog "tpds-r3 E15 done"
