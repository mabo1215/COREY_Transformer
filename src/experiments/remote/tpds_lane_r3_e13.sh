#!/usr/bin/env bash
# TPDS cycle tpds-r3 E13 (review item M4c) on a data-centre GPU: wrapper around tpds_lane_r3.sh.
#   0. If the runtime-chunk reference scan (selective_scan_chunk_cuda) is not importable, build it
#      for the native architecture first (CPU-only compile, a few minutes; it cannot be built in
#      AutoDL no-GPU mode, whose 2 GB memory cap kills the compiler). E12 uses it when present;
#      a failed build only drops the reference rows of E12.
#   1. TPDS_R3_ONLY=E13 tpds_lane_r3.sh: kernel_stats, 370M Mamba-1 capture, E4/E1 calibration,
#      E2 over the 2360 captured calls, then E12. ~3 h, peak ~20 GB VRAM.
#   2. E13b (author 2026-10-01): the same calibration and evaluation data in device time, as E15 did
#      on the RTX 3090 (GPU-side pre-spin before every timing block, TPDS_E13_SPIN_MS, default 20):
#      E4 Mamba-1 launch sweep, E1 Mamba-1 medians and E2 over the 2360 calls, tag <host>_r3spin.
#      ~1.5-2 h. TPDS_E13_SKIP_SPIN=1 skips it.
#
# Launch:  wsl bash src/revloop/remote.sh launch h800 tpds-r3-e13 src/experiments/remote/tpds_lane_r3_e13.sh
# Env: TORCH_CUDA_ARCH_LIST (default: native compute capability), TPDS_E13_SKIP_REF=1.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/tpds_common.sh"
if [ "${TPDS_E13_SKIP_REF:-0}" != 1 ] && ! "$PY" -c "import torch, selective_scan_chunk_cuda" 2>/dev/null; then
  ARCH=${TORCH_CUDA_ARCH_LIST:-$("$PY" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())' 2>/dev/null)}
  lanelog "E13: building selective_scan_chunk_cuda for ${ARCH:-9.0}"
  PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST=${ARCH:-9.0} \
    MAX_JOBS=${MAX_JOBS:-8} bash "$HERE/tpds_r2_prep.sh" > "$LOG_DIR/e13_ref_build.log" 2>&1
  "$PY" -c "import torch, selective_scan_chunk_cuda" 2>/dev/null \
    && lanelog "E13: reference scan ready" || lanelog "E13: reference scan build failed (E12 runs without it)"
fi
TPDS_R3_ONLY=E13 bash "$HERE/tpds_lane_r3.sh"
[ "${TPDS_E13_SKIP_SPIN:-0}" = 1 ] && exit 0
L=(); [ "${TPDS_NO_GPU_LOCK:-0}" = 1 ] || L=(flock "$LOCK")
P=""; [ "${TPDS_SMOKE:-0}" = 1 ] && P=smoke-
SPIN=${TPDS_E13_SPIN_MS:-20}
if [ "${TPDS_SMOKE:-0}" = 1 ]; then
  E2A="--batches 1 --warmup 2 --repeats 3 --max-rows 4"
  E4A="--parts m1 --lengths 128 --batches 1 --search-warmup 1 --search-repeats 2 --warmup 2 --repeats 3"
  E1A="--families mamba1 --lengths 512 --warmup 2 --repeats 3 --rounds 1 --inputs gaussian,real"
else
  E2A="--batches 1,4,16 --warmup 10 --repeats 50"
  E4A="--parts m1 --lengths 128,512,2048,8192 --batches 1,4,16 --search-warmup 3 --search-repeats 10 --warmup 10 --repeats 100"
  E1A="--families mamba1 --lengths 512,2048,8192 --batch 1 --warmup 10 --repeats 100 --rounds 3"
fi
lanelog "tpds-r3 E13b (device time, spin ${SPIN} ms) start on $TAG"
# shellcheck disable=SC2086
$RUN ${P}E13b -- bash -c "set -e
  export TPDS_PRE_SPIN_MS=$SPIN
  ${L[*]} $PY -m src.experiments.tpds.e4_autotune $E4A --trace-dir '$TRACE_DIR' --out-dir '$OUT_DIR' --tag ${TAG}_r3spin
  ${L[*]} $PY -m src.experiments.tpds.e1_value_invariance $E1A --trace-dir '$TRACE_DIR' --out-dir '$OUT_DIR' --tag ${TAG}_r3spin
  ${L[*]} $PY -m src.experiments.tpds.e2_oracle_sweep --model mamba1 $E2A --pre-spin-ms $SPIN \
      --trace-dir '$TRACE_DIR' --out-dir '$OUT_DIR' --tag ${TAG}_r3spin"
lanelog "tpds-r3 E13b done"
