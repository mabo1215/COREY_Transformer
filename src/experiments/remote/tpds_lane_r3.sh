#!/usr/bin/env bash
# TPDS cycle tpds-r3 GPU lane. Experiments are selected with TPDS_R3_ONLY (default "E12").
#   E12  chunk sweep at the tensor-parallel widths d/t of the 370M, 1.4B and 2.8B checkpoints
#        (synthetic inputs; Triton kernel, plus the runtime-chunk reference scan when the module
#        selective_scan_chunk_cuda is importable). Review item M1c. ~1 h on an RTX 3090.
#   E13  frozen cost model on a data-centre GPU (review item M4c): capture the 370M Mamba-1
#        traces, run the calibration data of the cost model (E4 Mamba-1 launch sweep and E1
#        Mamba-1 value-invariance medians) and the per-call sweep of the 2360 captured Mamba-1
#        calls (E2), then E12 on the same GPU. Compile facts of the native architecture are
#        extracted here as well (kernel_stats). ~3 h.
# All steps time kernels under the exclusive timing lock; E12 also uses the exclusive-GPU guard.
#
# Launch:  wsl bash src/revloop/remote.sh launch vgpu3090 tpds-r3 src/experiments/remote/tpds_lane_r3.sh
#          TPDS_R3_ONLY="E13" on the data-centre host.
# Markers: src/logs/revloop/{E12,E13-*}.{START,OK,FAIL}, prefixed with TPDS_R3_PREFIX when set.
# Resumable: every step skips completed rows. TPDS_SMOKE=1 runs a tiny version.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
L=(); [ "${TPDS_NO_GPU_LOCK:-0}" = 1 ] || L=(flock "$LOCK")
ONLY=${TPDS_R3_ONLY:-E12}
# Ledger IDs: E12 on the RTX 3090 (no prefix, unlike the replicated tpds-r1 runs of that host);
# on another host set TPDS_R3_PREFIX (e.g. "dc-") so that its E12 gets its own ID.
P=${TPDS_R3_PREFIX:-}
[ "${TPDS_SMOKE:-0}" = 1 ] && P=smoke-$P
check_space "$OUT_DIR" 2 || exit 1
if [ "${TPDS_SMOKE:-0}" = 1 ]; then
  E12_ARGS="--widths 1024,512 --lengths 128,512 --batches 1,4 --rounds 2 --warmup 2 --repeats 5"
  E13_E4="--parts m1 --lengths 128,512 --batches 1 --search-warmup 1 --search-repeats 2 --warmup 2 --repeats 3"
  E13_E1="--families mamba1 --lengths 512 --warmup 2 --repeats 3 --rounds 1 --inputs gaussian,sparse02,real"
  E13_E2="--model mamba1 --batches 1,4 --warmup 2 --repeats 3 --max-rows 12"
  E13_KS=""
else
  E12_ARGS="--rounds 3 --warmup 10 --repeats 50 --record-gpu-state"
  E13_E4="--parts m1 --lengths 128,512,2048,8192 --batches 1,4,16 --search-warmup 3 --search-repeats 10 --warmup 10 --repeats 100"
  E13_E1="--families mamba1 --lengths 512,2048,8192 --batch 1 --warmup 10 --repeats 100 --rounds 3"
  E13_E2="--model mamba1 --batches 1,4,16 --warmup 10 --repeats 50"
  E13_KS=""
fi
REF_OK=0; "$PY" -c "import torch, selective_scan_chunk_cuda" 2>/dev/null && REF_OK=1
REF_FLAG=""; [ "$REF_OK" = 1 ] && REF_FLAG="--ref"
lanelog "tpds-r3 lane start on $TAG (runtime-chunk module: $REF_OK; experiments: $ONLY)"
want() { case " $ONLY " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

if want E13; then
  # compile facts of the native architecture (no timing; fast)
  $RUN ${P}E13-kstats -- "$PY" -m src.experiments.tpds.kernel_stats $E13_KS --out-dir "$OUT_DIR" \
    || lanelog "kernel_stats failed (continuing; can be cross-compiled locally)"
  # 370M Mamba-1 traces (the captured calls of the paper)
  $RUN ${P}CAPTURE-mamba1 -- "$PY" -m src.experiments.tpds.capture_traces --model mamba1 $CAP_ARGS \
    --trace-dir "$TRACE_DIR" || { lanelog "capture failed"; exit 1; }
  # calibration data of the cost model: E4 Mamba-1 launch sweep, E1 Mamba-1 medians
  # shellcheck disable=SC2086
  $RUN ${P}E13-E4 -- "${L[@]}" "$PY" -m src.experiments.tpds.e4_autotune $E13_E4 \
    --trace-dir "$TRACE_DIR" --out-dir "$OUT_DIR" --tag "$TAG" || lanelog "E4 failed (continuing)"
  # shellcheck disable=SC2086
  $RUN ${P}E13-E1 -- "${L[@]}" "$PY" -m src.experiments.tpds.e1_value_invariance $E13_E1 \
    --trace-dir "$TRACE_DIR" --out-dir "$OUT_DIR" --tag "$TAG" || lanelog "E1 failed (continuing)"
  # evaluation data: every chunk on the 2360 captured Mamba-1 calls
  # shellcheck disable=SC2086
  $RUN ${P}E13-E2 -- "${L[@]}" "$PY" -m src.experiments.tpds.e2_oracle_sweep $E13_E2 \
    --trace-dir "$TRACE_DIR" --out-dir "$OUT_DIR" --tag "$TAG" || lanelog "E2 failed (continuing)"
  ONLY="$ONLY E12"
fi

if want E12; then
  # shellcheck disable=SC2086
  $RUN ${P}E12 -- bash -c "${L[*]} $PY -m src.experiments.tpds.e12_tp_widths $E12_ARGS $REF_FLAG --exclusive-guard \
      --out-dir '$OUT_DIR' --tag '$TAG'"
fi
lanelog "tpds-r3 lane done"
