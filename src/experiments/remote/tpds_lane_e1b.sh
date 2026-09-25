#!/usr/bin/env bash
# TPDS E1b (GPU): control run for E1 value invariance. Mamba-1 and Mamba-2 scans at fixed shape,
# 10 shuffled rounds, two extra random draws of every synthetic distribution (null spread between
# inputs of the same distribution), and GPU state (SM clock, power, temperature, throttle reasons)
# recorded around every timing. Then the E1 analysis on the new file.
#
# Launch:  wsl bash src/revloop/remote.sh launch pro6000 tpds-e1b src/experiments/remote/tpds_lane_e1b.sh
# Markers: src/logs/revloop/<prefix>E1b.{START,OK,FAIL}
#
# Kernel timing needs the GPU alone: the lane waits until no other process holds the GPU
# (TPDS_EXCLUSIVE_WAIT_MAX_S, default 12 h) and samples foreign GPU processes every 30 s during
# the run into src/logs/tpds/e1b_gpu_apps_<tag>.log, so contamination is visible afterwards.
# Resumable: re-launching skips completed rows. TPDS_SMOKE=1 runs a tiny version.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
L=(); [ "${TPDS_NO_GPU_LOCK:-0}" = 1 ] || L=(flock "$LOCK")
P=$ID_PREFIX
E1B_TAG="${TAG}_e1b"
APPS_LOG="$LOG_DIR/e1b_gpu_apps_$TAG.log"
check_space "$OUT_DIR" 2 || exit 1

foreign_gpu_pids() {  # PIDs of GPU compute processes that are not this lane's descendants
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' || true
}

if [ "${TPDS_SMOKE:-0}" = 1 ]; then
  E1B_ARGS="--families mamba1 --lengths 512 --inputs gaussian,constant,sparse10 --modes act --chunks-mamba1 128,512 --warmup 2 --repeats 5 --rounds 2 --extra-seeds 1 --skip-real"
else
  E1B_ARGS="--families mamba1,mamba2 --lengths 512,2048,8192 --batch 1 --warmup 10 --repeats 100 --rounds 10 --extra-seeds 2"
fi

waited=0; max=${TPDS_EXCLUSIVE_WAIT_MAX_S:-43200}
while [ "${TPDS_SMOKE:-0}" != 1 ] && [ -n "$(foreign_gpu_pids)" ]; do  # smoke runs check correctness only
  [ "$waited" -eq 0 ] && lanelog "E1b waiting for exclusive GPU (busy: $(foreign_gpu_pids | tr '\n' ' '))"
  if [ "$waited" -ge "$max" ]; then lanelog "E1b: GPU still busy after ${max}s; giving up"; exit 1; fi
  sleep 60; waited=$((waited + 60))
done
lanelog "E1b start on $TAG (GPU exclusive after ${waited}s wait)"

( while true; do
    apps=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null)
    echo "[$(date '+%F %T')] ${apps:-none}" >> "$APPS_LOG"
    sleep 30
  done ) &
sampler=$!
trap 'kill $sampler 2>/dev/null' EXIT

# shellcheck disable=SC2086
$RUN ${P}E1b -- bash -c "
  ${L[*]} $PY -m src.experiments.tpds.e1_value_invariance $E1B_ARGS --record-gpu-state \
      --trace-dir '$TRACE_DIR' --out-dir '$OUT_DIR' --tag '$E1B_TAG' &&
  $PY -m src.experiments.tpds.e1_analysis --in-dir '$OUT_DIR' --tag '$E1B_TAG'"
lanelog "E1b lane done"
