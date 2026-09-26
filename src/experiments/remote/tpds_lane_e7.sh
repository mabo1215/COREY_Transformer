#!/usr/bin/env bash
# TPDS E7 (GPU): Triton Mamba-1 scan vs the reference CUDA selective scan of mamba_ssm on the
# twelve E4 shape keys with captured activations (needs a mamba_ssm build with kernels for this
# GPU, e.g. the stock wheel on the RTX 3090; the patched build on the RTX PRO 6000 has none for
# sm_120). Every key waits for an exclusive GPU and is re-timed if another process appeared.
#
# Launch:  wsl bash src/revloop/remote.sh launch vgpu3090 tpds-e7 src/experiments/remote/tpds_lane_e7.sh
# Markers: src/logs/revloop/<prefix>E7.{START,OK,FAIL}
# Resumable: re-launching skips completed keys. TPDS_SMOKE=1 runs a tiny version.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
L=(); [ "${TPDS_NO_GPU_LOCK:-0}" = 1 ] || L=(flock "$LOCK")
P=$ID_PREFIX
check_space "$OUT_DIR" 1 || exit 1
if [ "${TPDS_SMOKE:-0}" = 1 ]; then
  E7_ARGS="--lengths 128,512 --batches 1 --warmup 2 --repeats 5"
else
  E7_ARGS="--lengths 128,512,2048,8192 --batches 1,4,16 --warmup 10 --repeats 50"
fi
lanelog "E7 start on $TAG"
# shellcheck disable=SC2086
$RUN ${P}E7 -- bash -c "
  $PY -c 'from mamba_ssm.ops.selective_scan_interface import selective_scan_fn' &&
  ${L[*]} $PY -m src.experiments.tpds.e7_reference_scan $E7_ARGS --exclusive-guard \
      --trace-dir '$TRACE_DIR' --out-dir '$OUT_DIR' --tag '$TAG' --e4-tag '$TAG'"
lanelog "E7 lane done"
