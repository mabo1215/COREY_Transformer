#!/usr/bin/env bash
# TPDS cycle tpds-r3, E15b: the 2360 captured Mamba-2 370M calls in device time (GPU-side pre-spin),
# the counterpart of E15 for the SSD pipeline, whose launch-inclusive latency is floored by the host
# launches of its several kernels. Starts when E15 has finished.
#
# Launch: screen -dmS tpds-r3-e15b bash -lc "export REVLOOP_HOST=vgpu3090; bash src/experiments/remote/tpds_lane_r3_e15b.sh"
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
L=(); [ "${TPDS_NO_GPU_LOCK:-0}" = 1 ] || L=(flock "$LOCK")
P=""; [ "${TPDS_SMOKE:-0}" = 1 ] && P=smoke-
SPIN=${TPDS_E15_SPIN_MS:-20}
if [ "${TPDS_SMOKE:-0}" = 1 ]; then E2A="--batches 1 --warmup 2 --repeats 3 --max-rows 4"
else
  E2A="--batches 1,4,16 --warmup 10 --repeats 50"
  waited=0
  until [ -f "$MARK_DIR/E15.OK" ] || [ -f "$MARK_DIR/E15.FAIL" ]; do
    sleep 60; waited=$((waited + 60)); [ "$waited" -ge 43200 ] && { lanelog "E15b: E15 did not finish"; exit 1; }
  done
fi
lanelog "tpds-r3 E15b start on $TAG (Mamba-2 device time, spin ${SPIN} ms)"
# shellcheck disable=SC2086
$RUN ${P}E15b -- "${L[@]}" "$PY" -m src.experiments.tpds.e2_oracle_sweep --model mamba2 $E2A --pre-spin-ms "$SPIN" \
  --trace-dir "$TRACE_DIR" --out-dir "$OUT_DIR" --tag "${TAG}_r3spin"
lanelog "tpds-r3 E15b done"
