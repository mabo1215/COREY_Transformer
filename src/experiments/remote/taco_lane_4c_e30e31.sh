#!/usr/bin/env bash
# Cycle taco-r1 lanes on the 4c lab host (stock RTX 3090 24 GB boards, sm_86; two GPUs may be used).
#   E30 (review M6c) on physical GPU 0: SSD 370M, SSD 2.7B, Gated DeltaNet 1.4B autotuner state, pinned vs
#       re-tuned per shape (e30_autotune_state.py), on the stratified subset of the RTX 3090 sweep's calls.
#       Inputs: the subset's trace files and the on-disk Triton autotune results copied from the vGPU 3090
#       host (TACO_TRACE_ROOT; the autotune results are merged into this host's Triton cache beforehand).
#       The exclusive-GPU guard watches nvidia-smi GPU 0, so E30 must run on physical GPU 0.
#   E31 (review M8b) on physical GPU 1: output divergence of per-shape chunk switching (e31_divergence.py):
#       Mamba-2 370M and Mamba-1 370M concurrently (not a timing experiment), then Gated DeltaNet 1.4B.
# One lane per screen:
#   TACO_LANE=E30 CUDA_VISIBLE_DEVICES=0 bash src/experiments/remote/taco_lane_4c_e30e31.sh
#   TACO_LANE=E31 CUDA_VISIBLE_DEVICES=1 bash src/experiments/remote/taco_lane_4c_e30e31.sh
# Markers: src/logs/revloop/{E30-smoke,E30a,E30b,E30c,E31-smoke,E31a,E31b,E31c}.{START,OK,FAIL}
# Results come back by rsync:  bash src/revloop/remote.sh fetch 4c E30a E30b E30c E31a E31b E31c
# Env: TACO_TRACE_ROOT, TACO_HF_HOME (/data/hf_home), TACO_GUARD_H (24), TACO_SKIP_GDN31.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
SELF="src/experiments/remote/taco_lane_4c_e30e31.sh"
H=4c
T=${TACO_TRACE_ROOT:-/data/mabo1215_taco/e30_inputs/autodl-fs/data/corey_r4_traces_vgpu3090}
HFH=${TACO_HF_HOME:-/data/hf_home}
export HF_HOME=$HFH
E30="$PY -m src.experiments.tpds.e30_autotune_state"
E31="$PY -m src.experiments.tpds.e31_divergence --hf-home $HFH"
M2_ID=${TACO_M2_ID:-AntonV/mamba2-370m-hf}
GDN_ID=${TACO_GDN_ID:-Idiap/gated-deltanet-swa-1.4B-30B}
SMOKE_DIR=$OUT_DIR/smoke
LANE=${TACO_LANE:?set TACO_LANE=E30 or E31}
skip_ids() {  # mark dependent runs as failed so that a patrol closes them
  local why=$1; shift
  for id in "$@"; do echo 99 > "$MARK_DIR/$id.FAIL"; lanelog "skip $id: $why"; done
}
if [ "${1:-}" = "step" ]; then
  GUARD="--exclusive-guard --guard-until $(( $(date +%s) + ${TACO_GUARD_H:-24} * 3600 ))"
  S30="--n-calls 2 --max-rows 2 --rounds 1 --warmup 2 --repeats 5 --out-dir $SMOKE_DIR"
  case "$2" in
    e30_smoke)
      flock "$LOCK" $E30 --model mamba2 --trace-dir "$T" --label "SSD 370M" $S30 --tag ${H}_e30smoke_ssd370m &&
      flock "$LOCK" $E30 --model mamba2 --trace-dir "$T/tpds_traces_m2_2p7b" --label "SSD 2.7B" $S30 \
        --tag ${H}_e30smoke_ssd2p7b &&
      flock "$LOCK" $E30 --model gdn --trace-dir "$T/tpds_traces_gdn_1p4b" --label "GDN 1.4B" $S30 \
        --tag ${H}_e30smoke_gdn1p4b ;;
    e30a) flock "$LOCK" $E30 --model mamba2 --trace-dir "$T" --label "SSD 370M" $GUARD --tag ${H}_e30_ssd370m ;;
    e30b) flock "$LOCK" $E30 --model mamba2 --trace-dir "$T/tpds_traces_m2_2p7b" --label "SSD 2.7B" $GUARD \
            --tag ${H}_e30_ssd2p7b ;;
    e30c) flock "$LOCK" $E30 --model gdn --trace-dir "$T/tpds_traces_gdn_1p4b" --label "GDN 1.4B" $GUARD \
            --tag ${H}_e30_gdn1p4b ;;
    e31_smoke)
      $E31 --model mamba2 --model-id "$M2_ID" --smoke && $E31 --model mamba1 --smoke ;;
    e31a) for dt in float16 float32; do $E31 --model mamba2 --model-id "$M2_ID" --dtype $dt --tag ${H}_m2 || exit 1; done
          $E31 --analyze --tag ${H}_m2 ;;
    e31b) for dt in float16 float32; do $E31 --model mamba1 --dtype $dt --tag ${H}_m1 || exit 1; done
          $E31 --analyze --tag ${H}_m1 ;;
    e31c) $E31 --model gdn --model-id "$GDN_ID" --dtype bfloat16 --smoke &&
          $E31 --model gdn --model-id "$GDN_ID" --dtype bfloat16 --tag ${H}_gdn &&
          $E31 --analyze --tag ${H}_gdn ;;
    *) echo "unknown step $2"; exit 2 ;;
  esac
  exit $?
fi
check_space "$OUT_DIR" 5 || exit 1
check_space "$HOME" 5 || exit 1   # Triton compile cache
lanelog "taco-r1 4c lane $LANE on $TAG (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}; traces $T)"
nvidia-smi -L || { lanelog "no GPU visible"; exit 1; }
case "$LANE" in
  E30)
    if [ "${CUDA_VISIBLE_DEVICES:-}" != 0 ]; then lanelog "E30 must run on physical GPU 0 (the guard watches GPU 0)"; exit 2; fi
    if $RUN E30-smoke -- bash "$SELF" step e30_smoke; then
      $RUN E30a -- bash "$SELF" step e30a
      $RUN E30b -- bash "$SELF" step e30b
      $RUN E30c -- bash "$SELF" step e30c
    else
      skip_ids "E30 smoke failed" E30a E30b E30c
    fi ;;
  E31)
    if $RUN E31-smoke -- bash "$SELF" step e31_smoke; then
      $RUN E31a -- bash "$SELF" step e31a &
      $RUN E31b -- bash "$SELF" step e31b &
      wait
      if [ "${TACO_SKIP_GDN31:-0}" = 1 ]; then skip_ids "skipped by TACO_SKIP_GDN31" E31c
      else $RUN E31c -- bash "$SELF" step e31c; fi
    else
      skip_ids "E31 smoke failed" E31a E31b E31c
    fi ;;
  *) lanelog "unknown lane $LANE"; exit 2 ;;
esac
lanelog "taco-r1 4c lane $LANE done"
