#!/usr/bin/env bash
# TPDS lane A (GPU): E1 value invariance + E4 autotuning / decision overhead.
# Launch:   wsl bash src/revloop/remote.sh launch pro6000 tpds-a src/experiments/remote/tpds_lane_a.sh
# Watch:    wsl bash src/revloop/remote.sh watch pro6000 E1 E2 E3 E4 E5
# Markers:  src/logs/revloop/{E1-synth,E4,E1}.{START,OK,FAIL}; python logs in src/logs/tpds/.
# All steps are resumable: re-launching the lane skips completed rows.
# GPU steps take an exclusive flock so they never time kernels concurrently with lane B.
# TPDS_SMOKE=1 runs a tiny version (see tpds_common.sh).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
L=(); [ "${TPDS_NO_GPU_LOCK:-0}" = 1 ] || L=(flock "$LOCK")
P=$ID_PREFIX
check_space "$OUT_DIR" 2 || exit 1
lanelog "lane A start on $TAG (python $PY)"

# E1, synthetic inputs first (does not need captured traces).
# shellcheck disable=SC2086
$RUN ${P}E1-synth -- "${L[@]}" "$PY" -m src.experiments.tpds.e1_value_invariance $E1_ARGS --skip-real \
  --trace-dir "$TRACE_DIR" --out-dir "$OUT_DIR" --tag "$TAG" || lanelog "E1-synth failed (continuing)"

# E4 uses real captured activations as inputs: wait for lane B's capture steps.
{ wait_marker ${P}CAPTURE-mamba1 && wait_marker ${P}CAPTURE-mamba2; } || lanelog "captures unavailable; E4 falls back to synthetic inputs"
# shellcheck disable=SC2086
$RUN ${P}E4 -- "${L[@]}" "$PY" -m src.experiments.tpds.e4_autotune $E4_ARGS \
  --trace-dir "$TRACE_DIR" --out-dir "$OUT_DIR" --tag "$TAG" || lanelog "E4 failed (continuing)"

# E1 again: resumes the same file and adds the real-activation rows.
# shellcheck disable=SC2086
$RUN ${P}E1 -- "${L[@]}" "$PY" -m src.experiments.tpds.e1_value_invariance $E1_ARGS \
  --trace-dir "$TRACE_DIR" --out-dir "$OUT_DIR" --tag "$TAG"
lanelog "lane A done"
