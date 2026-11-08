#!/usr/bin/env bash
# Cycle jpdc-r1 lane on the 4c lab host (stock RTX 3090 24 GB boards; physical GPUs 0 and 1 may be used).
#   E39 (review C3): whole prefill steps below 256 tokens (L in {16,32,64,128}, batch 1 and 4) captured as one
#       CUDA graph, with the protocol, models and two sessions of E32, so that the step-level study covers the
#       prompt lengths that public traces contain (e32_model_graph_prefill.py).
# Run:  CUDA_VISIBLE_DEVICES=0 bash src/experiments/remote/jpdc_lane_4c.sh
# Markers: src/logs/revloop/<ID>.{START,OK,FAIL}; results come back by rsync (remote.sh fetch 4c E39).
# Env: TACO_HF_HOME (/data/hf_home), TACO_HF_HUB_CACHE (/data/hf_cache), TACO_GUARD_H (12).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
H=4c
export HF_HOME=${TACO_HF_HOME:-/data/hf_home}
export HF_HUB_CACHE=${TACO_HF_HUB_CACHE:-/data/hf_cache}
GPU=${CUDA_VISIBLE_DEVICES:-0}
LOCK=$LOG_DIR/gpu_timing_g${GPU}.lock
SMOKE_DIR=$OUT_DIR/smoke
E32="$PY -m src.experiments.tpds.e32_model_graph_prefill"
M2_ID=${TACO_M2_ID:-AntonV/mamba2-370m-hf}
M27_ID=${TACO_M27_ID:-benchang1110/mamba2-2.7b-hf}
GDN_ID=${TACO_GDN_ID:-Idiap/gated-deltanet-swa-1.4B-30B}
GUARD="--exclusive-guard --guard-until $(( $(date +%s) + ${TACO_GUARD_H:-12} * 3600 )) --record-gpu-state"
SHAPES="--lengths 16,32,64,128 --batches 1,4"

e39() {  # e39 <session> <model> <model-id> <label>
  flock "$LOCK" $E32 --model "$2" --model-id "$3" --label "$4" $SHAPES $GUARD --tag ${H}_e39$1_$4
}

if [ "${1:-}" = "step" ]; then
  case "$2" in
    smoke)
      S="--lengths 16,128 --batches 1,4 --rounds 1 --warmup 2 --repeats 3 --eager-repeats 3 --out-dir $SMOKE_DIR"
      flock "$LOCK" $E32 --model mamba2 --model-id "$M2_ID" --label ssd370m $S --tag ${H}_e39smoke_ssd370m &&
      flock "$LOCK" $E32 --model gdn --model-id "$GDN_ID" --label gdn1p4b $S --tag ${H}_e39smoke_gdn1p4b ;;
    s1)
      e39 s1 mamba2 "$M2_ID" ssd370m && e39 s1 gdn "$GDN_ID" gdn1p4b && e39 s1 mamba2 "$M27_ID" ssd2p7b ;;
    s2)
      e39 s2 mamba2 "$M2_ID" ssd370m && e39 s2 gdn "$GDN_ID" gdn1p4b && e39 s2 mamba2 "$M27_ID" ssd2p7b ;;
    analyze)
      for lab in ssd370m gdn1p4b ssd2p7b; do
        $E32 --analyze --input-tags ${H}_e39s1_${lab},${H}_e39s2_${lab} --tag ${H}_e39s1_${lab} || exit 1
      done ;;
    *) echo "unknown step $2"; exit 2 ;;
  esac
  exit $?
fi

SELF="src/experiments/remote/jpdc_lane_4c.sh"
check_space "$OUT_DIR" 5 || exit 1
check_space "$HOME" 5 || exit 1   # Triton compile cache
lanelog "jpdc-r1 4c lane (CUDA_VISIBLE_DEVICES=$GPU)"
nvidia-smi -L || { lanelog "no GPU visible"; exit 1; }
$RUN E39-smoke -- bash "$SELF" step smoke || { lanelog "E39 smoke failed"; exit 1; }
$RUN E39s1 -- bash "$SELF" step s1
$RUN E39s2 -- bash "$SELF" step s2
$RUN E39 -- bash "$SELF" step analyze
