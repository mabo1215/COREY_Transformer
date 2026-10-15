#!/usr/bin/env bash
# Cycle taco-r1 GPU lane on the vGPU 3090 (host tag vgpu3090; optional review items chosen by the author).
#   E30 (review M6c). Autotuner state of the SSD and Gated DeltaNet kernels: a stratified subset of the
#       captured calls (40 calls + the anchor call per model) timed with the inner configurations pinned
#       as in the sweeps and re-tuned per shape; CUDA graph, three rounds (e30_autotune_state.py).
#         E30a  SSD 370M      E30b  SSD 2.7B      E30c  Gated DeltaNet 1.4B             ~1.5-1.8 GPU-h
#   E31 (review M8b). Output divergence of per-shape chunk switching: 192 prompts, prefill at every chunk
#       against the library default, 32 greedy tokens, teacher-forced KL and first divergent token, float16
#       and float32 (e31_divergence.py). Not a timing experiment: the two models run concurrently.
#         E31a  Mamba-2 370M (SSD)    E31b  Mamba-1 370M (routed reference scan)      ~1.2 h wall
#         E31c  Gated DeltaNet 1.4B, bfloat16 (optional; skipped with TACO_SKIP_GDN31=1)   ~0.3 h
# Order: smoke tests first (E30-smoke, E31-smoke); a failed smoke marks the dependent runs as failed
# (exit 99) so the watcher closes them. E30 holds the timing lock and the exclusive-GPU guard; E31 runs
# after E30, so no other process shares the GPU while E30 times.
#
# Launch:  wsl bash src/revloop/remote.sh launch vgpu3090 taco-vgpu src/experiments/remote/taco_lane_vgpu3090.sh
# Markers: src/logs/revloop/{E30-smoke,E30a,E30b,E30c,E31-smoke,E31a,E31b,E31c}.{START,OK,FAIL}
# Env:     TACO_ONLY (default "E30 E31"), TACO_TRACE_ROOT, TACO_GUARD_H (24), TACO_SKIP_GDN31.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
SELF="src/experiments/remote/taco_lane_vgpu3090.sh"
T=${TACO_TRACE_ROOT:-/autodl-fs/data/corey_r4_traces_vgpu3090}
M2_370M=${TACO_M2_ID:-/autodl-fs/data/cache_pro6000/huggingface/hub/models--AntonV--mamba2-370m-hf/snapshots/c75f3beb380942ce1eb13352873548fbd282ffac}
GDN_ID=${TACO_GDN_ID:-/autodl-fs/data/corey_r4_models/gated-deltanet-swa-1.4B-30B}
E30="$PY -m src.experiments.tpds.e30_autotune_state"
E31="$PY -m src.experiments.tpds.e31_divergence"
SMOKE_DIR=$OUT_DIR/smoke
ONLY=${TACO_ONLY:-E30 E31}
want() { case " $ONLY " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
skip_ids() {  # mark dependent runs as failed so that the watcher closes them
  local why=$1; shift
  for id in "$@"; do echo 99 > "$MARK_DIR/$id.FAIL"; lanelog "skip $id: $why"; done
}

if [ "${1:-}" = "step" ]; then
  GUARD="--exclusive-guard --guard-until $(( $(date +%s) + ${TACO_GUARD_H:-24} * 3600 ))"
  S30="--n-calls 2 --max-rows 2 --rounds 1 --warmup 2 --repeats 5 --out-dir $SMOKE_DIR"
  case "$2" in
    e30_smoke)
      flock "$LOCK" $E30 --model mamba2 --trace-dir "$T" --label "SSD 370M" $S30 --tag vgpu3090_e30smoke_ssd370m &&
      flock "$LOCK" $E30 --model mamba2 --trace-dir "$T/tpds_traces_m2_2p7b" --label "SSD 2.7B" $S30 \
        --tag vgpu3090_e30smoke_ssd2p7b &&
      flock "$LOCK" $E30 --model gdn --trace-dir "$T/tpds_traces_gdn_1p4b" --label "GDN 1.4B" $S30 \
        --tag vgpu3090_e30smoke_gdn1p4b ;;
    e30a) flock "$LOCK" $E30 --model mamba2 --trace-dir "$T" --label "SSD 370M" $GUARD --tag vgpu3090_e30_ssd370m ;;
    e30b) flock "$LOCK" $E30 --model mamba2 --trace-dir "$T/tpds_traces_m2_2p7b" --label "SSD 2.7B" $GUARD \
            --tag vgpu3090_e30_ssd2p7b ;;
    e30c) flock "$LOCK" $E30 --model gdn --trace-dir "$T/tpds_traces_gdn_1p4b" --label "GDN 1.4B" $GUARD \
            --tag vgpu3090_e30_gdn1p4b ;;
    e31_smoke)
      $E31 --model mamba2 --model-id "$M2_370M" --smoke && $E31 --model mamba1 --smoke ;;
    e31a) for dt in float16 float32; do $E31 --model mamba2 --model-id "$M2_370M" --dtype $dt --tag vgpu3090_m2 || exit 1; done
          $E31 --analyze --tag vgpu3090_m2 ;;
    e31b) for dt in float16 float32; do $E31 --model mamba1 --dtype $dt --tag vgpu3090_m1 || exit 1; done
          $E31 --analyze --tag vgpu3090_m1 ;;
    e31c) $E31 --model gdn --model-id "$GDN_ID" --dtype bfloat16 --smoke &&
          $E31 --model gdn --model-id "$GDN_ID" --dtype bfloat16 --tag vgpu3090_gdn &&
          $E31 --analyze --tag vgpu3090_gdn ;;
    *) echo "unknown step $2"; exit 2 ;;
  esac
  exit $?
fi

check_space "$OUT_DIR" 5 || exit 1
check_space /root 5 || exit 1   # Triton compile cache
lanelog "taco-r1 vgpu3090 lane on $TAG (steps: $ONLY; traces $T)"
nvidia-smi -L || { lanelog "no GPU visible"; exit 1; }

if want E30; then
  if $RUN E30-smoke -- bash "$SELF" step e30_smoke; then
    $RUN E30a -- bash "$SELF" step e30a
    $RUN E30b -- bash "$SELF" step e30b
    $RUN E30c -- bash "$SELF" step e30c
  else
    skip_ids "E30 smoke failed" E30a E30b E30c
  fi
fi
if want E31; then
  if $RUN E31-smoke -- bash "$SELF" step e31_smoke; then
    $RUN E31a -- bash "$SELF" step e31a &
    $RUN E31b -- bash "$SELF" step e31b &
    wait
    if [ "${TACO_SKIP_GDN31:-0}" = 1 ]; then skip_ids "skipped by TACO_SKIP_GDN31" E31c
    else $RUN E31c -- bash "$SELF" step e31c; fi
  else
    skip_ids "E31 smoke failed" E31a E31b E31c
  fi
fi
lanelog "taco-r1 vgpu3090 lane done"
