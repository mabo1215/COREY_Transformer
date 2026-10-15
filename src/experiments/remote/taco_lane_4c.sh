#!/usr/bin/env bash
# Cycle taco-r1 lane on the 4c host (4x RTX 3090, NVLink pairs 0-1 and 2-3; only two GPUs may be used).
#   E28 (review M7b). Two-rank tensor parallelism with NCCL collectives: the Triton Mamba-1 scan at the
#       t = 2 shard widths of the 370M, 1.4B and 2.8B checkpoints, every chunk, alone and with an
#       overlapping all-reduce of the layer output, three rounds, device time (e28_tp_collectives.py).
# The host is shared: the lane first waits until both GPUs of one NVLink pair (2-3 preferred, then 0-1)
# have no compute process, then runs a few-minute smoke test (E28-smoke), checks it, and only then starts
# the full run (E28) with the exclusive-GPU guard on the two physical GPUs.                ~1.5-2.5 GPU-h
#
# Launch:  bash src/revloop/remote.sh launch 4c taco-4c src/experiments/remote/taco_lane_4c.sh
# Markers: src/logs/revloop/{E28-smoke,E28}.{START,OK,FAIL}; the chosen pair in src/logs/tpds/taco_4c_pair.txt
# Env:     TACO_PAIRS (default "2,3 0,1"), TACO_WAIT_H (72), TACO_POLL_S (60), TACO_GUARD_H (72),
#          TACO_SKIP_WAIT=1 (no waiting), TACO_SMOKE_ONLY=1 (stop after the smoke test).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
PAIRS=${TACO_PAIRS:-"2,3 0,1"}
WAIT_S=$(( ${TACO_WAIT_H:-72} * 3600 )); POLL=${TACO_POLL_S:-60}
TR="$PY -m torch.distributed.run --nproc_per_node 2 --master_port ${TACO_PORT:-29531}"

busy_gpus() {   # indices of GPUs with a compute process
  local map uuids
  map=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | tr -d ' ')
  uuids=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | tr -d ' ' | sort -u)
  for u in $uuids; do echo "$map" | awk -F, -v u="$u" '$2==u {print $1}'; done
}
free_pair() {
  local busy p a b
  busy=" $(busy_gpus | tr '\n' ' ') "
  for p in $PAIRS; do
    a=${p%,*}; b=${p#*,}
    case "$busy" in *" $a "*|*" $b "*) ;; *) echo "$p"; return 0 ;; esac
  done
  return 1
}

lanelog "taco-r1 4c lane start (pairs: $PAIRS; waiting up to $(( WAIT_S / 3600 )) h for a free pair)"
waited=0; PAIR=""
until PAIR=$(free_pair); do
  [ "${TACO_SKIP_WAIT:-0}" = 1 ] && { PAIR=${PAIRS%% *}; lanelog "TACO_SKIP_WAIT=1: using $PAIR without waiting"; break; }
  if [ "$waited" -ge "$WAIT_S" ]; then lanelog "no free GPU pair after $(( waited / 3600 )) h; stopping"; exit 3; fi
  [ $(( waited % 3600 )) -eq 0 ] && lanelog "waiting for a free pair (busy GPUs: $(busy_gpus | tr '\n' ' '))"
  sleep "$POLL"; waited=$(( waited + POLL ))
done
echo "$PAIR" > "$LOG_DIR/taco_4c_pair.txt"
export CUDA_VISIBLE_DEVICES=$PAIR
lanelog "using GPUs $PAIR"

# smoke test: one width, one length, two chunks, both conditions, a few repeats (a few minutes)
SMOKE_DIR=src/experiments/results/tpds/smoke
mkdir -p "$SMOKE_DIR"
$RUN E28-smoke -- $TR -m src.experiments.tpds.e28_tp_collectives --models 370m --lengths 512,2048 --batches 1,4 \
    --chunks 64,256 --rounds 1 --warmup 2 --repeats 5 --out-dir "$SMOKE_DIR" --tag 4c_smoke_taco \
  || { lanelog "E28 smoke failed; not starting the full run"; exit 1; }
"$PY" - "$SMOKE_DIR/e28_tp_4c_smoke_taco.jsonl" <<'PYEOF' || { lanelog "E28 smoke check failed"; : > src/logs/revloop/E28-smoke.FAIL; exit 1; }
import json, math, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
t = [r for r in rows if r.get("kind") == "timing"]
assert len(t) == 2 * 2 * 2, f"expected 8 timing rows, got {len(t)}"
for r in t:
    for rk in ("0", "1"):
        for c, cell in r["ranks"][rk].items():
            assert math.isfinite(cell["mean_ms"]) and cell["mean_ms"] > 0, (r["key"], rk, c, cell)
for r in t:
    if r["condition"] == "overlap":
        a = next(x for x in t if x["condition"] == "alone" and x["L"] == r["L"] and x["batch"] == r["batch"])
        print(f"smoke L={r['L']} B={r['batch']}: overlap/alone at chunk 256, rank 0: "
              f"{r['ranks']['0']['256']['mean_ms'] / a['ranks']['0']['256']['mean_ms']:.3f}")
print("smoke ok:", len(t), "rows")
PYEOF
lanelog "E28 smoke passed"
[ "${TACO_SMOKE_ONLY:-0}" = 1 ] && { lanelog "TACO_SMOKE_ONLY=1: stopping after the smoke test"; exit 0; }

GUARD="--exclusive-guard --guard-until $(( $(date +%s) + ${TACO_GUARD_H:-72} * 3600 ))"
# shellcheck disable=SC2086
$RUN E28 -- $TR -m src.experiments.tpds.e28_tp_collectives --rounds 3 --warmup 10 --repeats 50 $GUARD \
    --out-dir "$OUT_DIR" --tag 4c_taco
lanelog "taco-r1 4c lane done"
