#!/usr/bin/env bash
# TPDS E1c (GPU): rerun of the E1b control (Mamba-1 and Mamba-2 scans at fixed shape, 10
# shuffled rounds, two extra random draws of every synthetic distribution, GPU state recorded
# around every timing) with an in-run exclusive-GPU guard. E1b started when the GPU was free, but
# other jobs claimed it 30 s later and ran for the whole measurement, so its timings are
# contaminated. Here every timed cell first waits until no other process holds the GPU, and a
# cell during which another process appeared is discarded and re-timed. Then the E1 analysis.
#
# Launch:  wsl bash src/revloop/remote.sh launch pro6000 tpds-e1c src/experiments/remote/tpds_lane_e1c.sh <guard-until-epoch>
# Markers: src/logs/revloop/<prefix>E1c.{START,OK,FAIL}
#
# <guard-until-epoch>: epoch seconds after which a paused run stops cleanly (exit 3) so that the
# completed rows can be pushed before the host goes away; default: no deadline. Resumable:
# re-launching skips completed rows. TPDS_SMOKE=1 runs a tiny version.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
L=(); [ "${TPDS_NO_GPU_LOCK:-0}" = 1 ] || L=(flock "$LOCK")
P=$ID_PREFIX
GUARD_UNTIL=${1:-}
E1C_TAG="${TAG}_e1c"
APPS_LOG="$LOG_DIR/e1c_gpu_apps_$TAG.log"
check_space "$OUT_DIR" 2 || exit 1

if [ "${TPDS_SMOKE:-0}" = 1 ]; then
  E1C_ARGS="--families mamba1 --lengths 512 --inputs gaussian,constant,sparse10 --modes act --chunks-mamba1 128,512 --warmup 2 --repeats 5 --rounds 2 --extra-seeds 1 --skip-real"
else
  E1C_ARGS="--families mamba1,mamba2 --lengths 512,2048,8192 --batch 1 --warmup 10 --repeats 100 --rounds 10 --extra-seeds 2"
fi
GUARD_ARGS="--exclusive-guard"
[ -n "$GUARD_UNTIL" ] && GUARD_ARGS="$GUARD_ARGS --guard-until $GUARD_UNTIL"
lanelog "E1c start on $TAG (guard until ${GUARD_UNTIL:-none})"

( while true; do
    apps=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null | tr '\n' ';')
    echo "[$(date '+%F %T')] ${apps:-none}" >> "$APPS_LOG"
    sleep 30
  done ) &
sampler=$!
trap 'kill $sampler 2>/dev/null' EXIT

# shellcheck disable=SC2086
$RUN ${P}E1c -- bash -c "
  ${L[*]} $PY -m src.experiments.tpds.e1_value_invariance $E1C_ARGS --record-gpu-state $GUARD_ARGS \
      --trace-dir '$TRACE_DIR' --out-dir '$OUT_DIR' --tag '$E1C_TAG' &&
  $PY -m src.experiments.tpds.e1_analysis --in-dir '$OUT_DIR' --tag '$E1C_TAG'"
lanelog "E1c lane done"
