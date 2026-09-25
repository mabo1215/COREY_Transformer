#!/usr/bin/env bash
# Shared settings for the TPDS lane scripts (sourced, not executed).
#
# Env overrides:
#   PY               python interpreter (remote.sh exports the host's registry value)
#   TPDS_TRACE_DIR   captured-trace directory (default src/outputs/tpds_traces; git-ignored)
#   TPDS_TAG         result-file tag (default: $REVLOOP_HOST, else hostname)
#   TPDS_NO_GPU_LOCK set to 1 to let the lanes time kernels concurrently (NOT recommended:
#                    concurrent kernels on one GPU contaminate each other's latency)
#   TPDS_WAIT_MAX_S  maximum wait for a marker of the other lane (default 43200 = 12 h)

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT" || exit 1
# Host-local settings (untracked): e.g. PY=<venv python>, REVLOOP_HOST=<registry name>, HF_ENDPOINT.
# shellcheck disable=SC1091
[ -f "$ROOT/.tpds_env" ] && . "$ROOT/.tpds_env"
if [ -z "${PY:-}" ] && [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; fi
PY=${PY:-python3}
command -v "$PY" >/dev/null 2>&1 || PY=python
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
TAG=${TPDS_TAG:-${REVLOOP_HOST:-$(hostname)}}
export REVLOOP_HOST=${REVLOOP_HOST:-$TAG}
TRACE_DIR=${TPDS_TRACE_DIR:-src/outputs/tpds_traces}
OUT_DIR=src/experiments/results/tpds
LOG_DIR=src/logs/tpds
MARK_DIR=src/logs/revloop
LOCK=$LOG_DIR/gpu_timing.lock
RUN="bash src/revloop/remote_run.sh"
mkdir -p "$OUT_DIR" "$LOG_DIR" "$MARK_DIR"

lanelog() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/lanes.log"; }

# gpu <cmd...>: run a GPU step under an exclusive lock so timing never overlaps another lane.
gpu() {
  if [ "${TPDS_NO_GPU_LOCK:-0}" = 1 ]; then "$@"; else flock "$LOCK" "$@"; fi
}

# wait_marker <ID>: block until src/logs/revloop/<ID>.OK exists; fail on <ID>.FAIL or timeout.
wait_marker() {
  local id=$1 waited=0 max=${TPDS_WAIT_MAX_S:-43200}
  while [ ! -f "$MARK_DIR/$id.OK" ]; do
    if [ -f "$MARK_DIR/$id.FAIL" ]; then lanelog "dependency $id failed"; return 1; fi
    if [ "$waited" -ge "$max" ]; then lanelog "timeout waiting for $id"; return 1; fi
    sleep 60; waited=$((waited + 60))
  done
}

check_space() {  # check_space <path> <min_free_gb>
  local p=$1 need=$2 free_gb ifree
  mkdir -p "$p"
  free_gb=$(df -Pk "$p" | awk 'NR==2 {printf "%d", $4/1048576}')
  ifree=$(df -Pi "$p" | awk 'NR==2 {print $4}')
  lanelog "space at $p: ${free_gb} GB free, ${ifree} inodes free (need ${need} GB)"
  if [ "${free_gb:-0}" -lt "$need" ]; then lanelog "NOT ENOUGH DISK at $p"; return 1; fi
  case "$ifree" in ''|-) ;; *) if [ "$ifree" -lt 20000 ]; then lanelog "NOT ENOUGH INODES at $p"; return 1; fi ;; esac
}

# Experiment sizes. TPDS_SMOKE=1 selects a tiny configuration (local smoke test of the lanes;
# results go to the smoke directory and marker IDs get a "smoke-" prefix).
if [ "${TPDS_SMOKE:-0}" = 1 ]; then
  OUT_DIR=src/experiments/results/tpds/smoke
  TRACE_DIR=${TPDS_TRACE_DIR:-src/outputs/tpds_traces_smoke_lane}
  TAG=${TPDS_TAG:-smoke-lane}
  ID_PREFIX=smoke-
  E1_ARGS="--lengths 512 --warmup 2 --repeats 3 --rounds 1 --families mamba1,mamba2,gdn --inputs gaussian,sparse02,real"
  CAP_ARGS="--layers 0,24 --samples-per-regime 1 --longbench-per-task 1 --max-tokens 1024 --limit-prompts 12"
  E2_ARGS="--batches 1,4 --warmup 2 --repeats 3 --max-rows 12"
  E4_ARGS="--lengths 128,512 --batches 1 --search-warmup 1 --search-repeats 2 --warmup 2 --repeats 3 --overhead-repeats 10"
  MAX_GB_DEFAULT=2
else
  ID_PREFIX=
  E1_ARGS="--lengths 512,2048,8192 --batch 1 --warmup 10 --repeats 100 --rounds 3"
  CAP_ARGS="--layer-stride 6 --samples-per-regime 20 --longbench-per-task 8 --longbench-lengths 1024,2048,4096,8192 --max-tokens 8192"
  E2_ARGS="--batches 1,4,16 --warmup 10 --repeats 50"
  E4_ARGS="--lengths 128,512,2048,8192 --batches 1,4,16 --search-warmup 3 --search-repeats 10 --warmup 10 --repeats 100 --overhead-repeats 500"
  MAX_GB_DEFAULT=60
fi
mkdir -p "$OUT_DIR"
