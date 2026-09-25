#!/usr/bin/env bash
# TPDS lane B (GPU): trace capture for Mamba-1/Mamba-2 370M, E2 oracle sweep, then CPU analyses E3/E5
# and preview figures (written next to the results, never into paper/).
# Launch:   wsl bash src/revloop/remote.sh launch pro6000 tpds-b src/experiments/remote/tpds_lane_b.sh
# Markers:  src/logs/revloop/{CAPTURE-mamba1,CAPTURE-mamba2,E2,E3,E5}.{START,OK,FAIL}
# All steps are resumable. GPU steps share an exclusive flock with lane A.
# TPDS_SMOKE=1 runs a tiny version (see tpds_common.sh).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
L=(); [ "${TPDS_NO_GPU_LOCK:-0}" = 1 ] || L=(flock "$LOCK")
P=$ID_PREFIX
MAX_GB=${TPDS_TRACE_MAX_GB:-$MAX_GB_DEFAULT}
check_space "$TRACE_DIR" $((MAX_GB + 10)) || exit 1
check_space "$OUT_DIR" 2 || exit 1
lanelog "lane B start on $TAG (python $PY), traces in $TRACE_DIR (max ${MAX_GB} GB)"

for m in mamba1 mamba2; do
  # shellcheck disable=SC2086
  $RUN ${P}CAPTURE-$m -- "${L[@]}" "$PY" -m src.experiments.tpds.capture_traces --model "$m" $CAP_ARGS \
    --max-gb "$MAX_GB" --trace-dir "$TRACE_DIR" --log-name "capture_${m}_$TAG" || lanelog "CAPTURE-$m failed"
done

# One lock acquisition per model so lane A can interleave between the two sweeps.
$RUN ${P}E2 -- bash -c "
  ${L[*]} $PY -m src.experiments.tpds.e2_oracle_sweep --model mamba1 $E2_ARGS \
      --trace-dir '$TRACE_DIR' --out-dir '$OUT_DIR' --tag '$TAG' &&
  ${L[*]} $PY -m src.experiments.tpds.e2_oracle_sweep --model mamba2 $E2_ARGS \
      --trace-dir '$TRACE_DIR' --out-dir '$OUT_DIR' --tag '$TAG'"

$RUN ${P}E3 -- "$PY" -m src.experiments.tpds.e3_predictors --in-dir "$OUT_DIR" --hosts "$TAG"
wait_marker ${P}E4 || lanelog "E4 not finished; E5 uses whatever E4 rows exist"
$RUN ${P}E5 -- "$PY" -m src.experiments.tpds.e5_breakeven --in-dir "$OUT_DIR" --host "$TAG"
"$PY" -m src.experiments.tpds.make_figures --in-dir "$OUT_DIR" --host "$TAG" --out-dir "$OUT_DIR/figs_$TAG" \
  > "$LOG_DIR/figures_$TAG.log" 2>&1 || lanelog "figures failed (see $LOG_DIR/figures_$TAG.log)"
lanelog "lane B done"
