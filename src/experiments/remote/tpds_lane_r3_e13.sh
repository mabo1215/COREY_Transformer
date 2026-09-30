#!/usr/bin/env bash
# TPDS cycle tpds-r3 E13 (review item M4c) on a data-centre GPU: wrapper around tpds_lane_r3.sh.
#   0. If the runtime-chunk reference scan (selective_scan_chunk_cuda) is not importable, build it
#      for the native architecture first (CPU-only compile, a few minutes; it cannot be built in
#      AutoDL no-GPU mode, whose 2 GB memory cap kills the compiler). E12 uses it when present;
#      a failed build only drops the reference rows of E12.
#   1. TPDS_R3_ONLY=E13 tpds_lane_r3.sh: kernel_stats, 370M Mamba-1 capture, E4/E1 calibration,
#      E2 over the 2360 captured calls, then E12. ~3 h, peak ~20 GB VRAM.
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
export TPDS_R3_ONLY=E13
exec bash "$HERE/tpds_lane_r3.sh"
