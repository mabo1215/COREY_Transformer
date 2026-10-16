#!/usr/bin/env bash
# Cycle taco-r2 E35b on the 4c lab host: Nsight Compute counters for the mechanism of the chunk effect (review M5e).
# The driver restricts performance counters to administrators (RmProfilingAdminOnly=1), so this script is run by
# the account owner with sudo (the password is typed interactively, never stored):
#   ssh -t 4c sudo bash /home1/mabo1215/COREY_Transformer_taco/src/experiments/remote/taco_ncu_4c.sh
# It waits until physical GPU 0 or 1 has no compute process (timing lanes must not be disturbed), profiles the
# E35 cells on that GPU (one profiled call per cell, kernel replay, natural clocks, no cache control), parses the
# counters, and returns ownership of the new files to the repository owner. Needs the E35 breakdown file
# (part "breakdown", lane G1) for the per-cell kernel counts.
set -uo pipefail
R=/home1/mabo1215/COREY_Transformer_taco
OWNER=$(stat -c %U "$R")
cd "$R" || exit 1
# shellcheck disable=SC1091
. "$R/.tpds_env"
export PYTHONPATH=$R PYTHONUNBUFFERED=1
T=${TACO_TRACE_ROOT:-/data/mabo1215_taco/e30_inputs/autodl-fs/data/corey_r4_traces_vgpu3090}
[ -f src/experiments/results/tpds/e35_breakdown_4c_e35.json ] || { echo "run the E35 breakdown first (lane G1)"; exit 1; }
free_gpu() {
  local map uuids g
  map=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | tr -d ' ')
  uuids=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | tr -d ' ' | sort -u)
  for g in 0 1; do
    u=$(echo "$map" | awk -F, -v i="$g" '$1==i {print $2}')
    echo "$uuids" | grep -q "$u" || { echo "$g"; return 0; }
  done
  return 1
}
until G=$(free_gpu); do echo "[$(date '+%F %T')] GPUs 0 and 1 busy; waiting"; sleep 60; done
echo "[$(date '+%F %T')] profiling on physical GPU $G"
CUDA_VISIBLE_DEVICES=$G "$PY" -m src.experiments.tpds.e35_mechanism --part ncu --tag 4c_e35 --breakdown-tag 4c_e35 \
  --gdn-trace-dir "$T/tpds_traces_gdn_1p4b" --ncu /usr/local/cuda-12.4/bin/ncu
rc=$?
chown -R "$OWNER": src/experiments/results/tpds src/logs/tpds 2>/dev/null
echo "E35b exit code $rc"
exit $rc
