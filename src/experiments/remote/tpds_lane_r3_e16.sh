#!/usr/bin/env bash
# TPDS cycle tpds-r3, review item M6b: placement control plus hardware counters on the RTX PRO 6000.
#   E16b  Nsight Compute counters of the E9 cells (designs A same buffers / B fresh buffers /
#         C placements), L in {8192, 2048}, batch 1, all six chunks, act and all modes, two rounds.
#         Natural clocks and a warm L2 (--clock-control none --cache-control none). ~30-60 min.
#         Runs first: it fails fast when the container has no counter permission (ERR_NVGPUCTRPERM).
#   E16   E9 placement timing on the RTX PRO 6000, where the 43% fixed-chunk effects were seen:
#         same designs, five rounds, exclusive-GPU guard, clock/power/throttle per cell; plus the
#         runtime-chunk reference CUDA scan (designs A and C) when selective_scan_chunk_cuda is
#         importable (tpds_r3_e16_prep.sh builds it for sm_120). ~1.5-2 h.
#
# Launch:  wsl bash src/revloop/remote.sh launch pro6000 tpds-r3-e16 src/experiments/remote/tpds_lane_r3_e16.sh
# Markers: src/logs/revloop/{E16b,E16}.{START,OK,FAIL}. TPDS_SMOKE=1 runs a tiny version (smoke- prefix).
# TPDS_E16_ONLY="E16" or "E16b" runs one step.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
L=(); [ "${TPDS_NO_GPU_LOCK:-0}" = 1 ] || L=(flock "$LOCK")
ONLY=${TPDS_E16_ONLY:-E16b E16}
P=""; [ "${TPDS_SMOKE:-0}" = 1 ] && P=smoke-
check_space "$OUT_DIR" 2 || exit 1
if [ "${TPDS_SMOKE:-0}" = 1 ]; then
  E16_ARGS="--lengths 512 --inputs gaussian,uniform,real --draws 1 --placements 4 --rounds 1 --warmup 2 --repeats 3 --cost-repeats 5"
  E16B_ARGS="--lengths 512 --chunks 64,256 --inputs gaussian,uniform --draws 1 --placements 2 --rounds 1 --warmup 1"
else
  E16_ARGS="--lengths 8192,2048 --rounds 5 --warmup 10 --repeats 100"
  E16B_ARGS=""
fi
REF_OK=0; "$PY" -c "import torch, selective_scan_chunk_cuda" 2>/dev/null && REF_OK=1
REF_FLAG=""; [ "$REF_OK" = 1 ] && REF_FLAG="--ref"
lanelog "tpds-r3 E16 lane start on $TAG (runtime-chunk module: $REF_OK; steps: $ONLY)"
want() { case " $ONLY " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

if want E16b; then
  # shellcheck disable=SC2086
  $RUN ${P}E16b -- "${L[@]}" "$PY" -m src.experiments.tpds.e16_counters $E16B_ARGS \
    --trace-dir "$TRACE_DIR" --out-dir "$OUT_DIR" --tag "$TAG" || lanelog "E16b failed (continuing with E16)"
fi

if want E16; then
  # shellcheck disable=SC2086
  $RUN ${P}E16 -- "${L[@]}" "$PY" -m src.experiments.tpds.e9_placement $E16_ARGS $REF_FLAG --exclusive-guard \
    --trace-dir "$TRACE_DIR" --out-dir "$OUT_DIR" --tag "${TAG}_r3"
fi
lanelog "tpds-r3 E16 lane done"
