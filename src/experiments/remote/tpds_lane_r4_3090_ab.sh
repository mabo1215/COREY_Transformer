#!/usr/bin/env bash
# TPDS cycle tpds-r4, optional packages A and B (author 2026-10-08), RTX 3090 (host tag vgpu3090).
# Starts after lane tpds_lane_r4_3090.sh (waits for its E21 marker); every timing step holds the GPU
# timing lock and the exclusive-GPU guard (the host is shared with other projects).
#   E24 (review M1b, package B). Chunk sweep of an independently written public kernel with a wide chunk
#       range: the TFLA mLSTM forward of mlstm_kernels (chunks 32..2048), three head layouts at the
#       370M/1.4B/2.8B value widths, L 512..8192, batch 1/4/16, three rounds, device time (tag
#       <host>_r4spin) and launch-inclusive (<host>_r4plain).                                ~1.5 GPU-h
#   E22 (review M3b, package A). CUDA-graph timing of every (call, chunk), >= 50 replays: the captured
#       370M Mamba-2 calls warm and L2-flushed, the 370M Mamba-1 calls warm, the Mamba-2 2.7B calls warm
#       (flushed for L >= 2048) and the Gated DeltaNet 1.4B calls warm and flushed. Captures the 370M
#       Mamba-2, 2.7B Mamba-2 and 1.4B Gated DeltaNet traces first (E17 settings).            ~3 GPU-h
#   E23 (review M9a, package A). Reference CUDA scan with 32 threads per block at every chunk next to the
#       stock pairing (module selective_scan_chunk_cuda rebuilt with fwd_with_chunk_threads), the 2360
#       captured 370M Mamba-1 calls, device time (tag <host>_r4t32).                       ~3 GPU-h
#
# Launch:  bash src/revloop/remote.sh launch vgpu3090 tpds-r4-ab src/experiments/remote/tpds_lane_r4_3090_ab.sh
#          (with TPDS_TRACE_DIR and TPDS_R4_MODEL_ROOT exported, see below)
# Markers: src/logs/revloop/{E24,CAPTURE-r4-m2,CAPTURE-r4-m2-2p7b,CAPTURE-r4-gdn,E22,E23}.{START,OK,FAIL}
# Env:     TPDS_R4AB_ONLY (default "E24 E22 E23"), TPDS_TRACE_DIR (370M traces; the 2.7B and GDN traces go
#          next to it), TPDS_R4_MODEL_ROOT (local copies of mamba2-2.7b-hf and gated-deltanet-swa-1.4B-30B),
#          TPDS_R4_HF_SHARED (an HF hub cache holding AntonV/mamba2-370m-hf), TPDS_R4AB_WAIT (1: wait for
#          the E21 marker), TPDS_R4_GUARD_H (24), TPDS_R4_SPIN_MS (20), TPDS_R4_E24_SUFFIX (appended to the
#          E24 tags, e.g. "b" for a rerun whose first files the watcher already committed).
# Resumable: every step skips completed rows. TPDS_SMOKE=1 runs a tiny version (smoke- markers).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
L=(); [ "${TPDS_NO_GPU_LOCK:-0}" = 1 ] || L=(flock "$LOCK")
P=""; [ "${TPDS_SMOKE:-0}" = 1 ] && P=smoke-
ONLY=${TPDS_R4AB_ONLY:-E24 E22 E23}
SPIN=${TPDS_R4_SPIN_MS:-20}
E24S=${TPDS_R4_E24_SUFFIX:-}
GUARD="--exclusive-guard --guard-until $(( $(date +%s) + ${TPDS_R4_GUARD_H:-24} * 3600 ))"
MODEL_ROOT=${TPDS_R4_MODEL_ROOT:-/root/autodl-tmp/hf_models}
TRACE_ROOT=$TRACE_DIR   # the 2.7B and GDN traces go next to the 370M ones (<root>/mamba1, <root>/mamba2)
if [ "${TPDS_SMOKE:-0}" = 1 ]; then
  TAG=${TPDS_TAG:-smoke-r4ab}
  GUARD=""   # a smoke run next to a real run must not wait for it (the real run's guard pauses instead)
  SPIN=${TPDS_R4_SPIN_MS:-5}
  E24A="--layouts tiny:2:64:128 --lengths 512,1024 --batches 1,2 --chunks 64,128,256 --rounds 1 --warmup 2 --repeats 3"
  E2A="--batches 1,4 --warmup 2 --repeats 3 --max-rows 4"
  M2_CAP="--layers 0,24 --samples-per-regime 1 --longbench-per-task 1 --max-tokens 1024 --limit-prompts 6"
  BIG_CAP="--samples-per-regime 1 --longbench-per-task 1 --max-tokens 1024 --limit-prompts 6"
  M2B_STRIDE=32; GDN_STRIDE=6; BIG_GB=2; COLD_MIN=512
  E23A="--batches 1 --warmup 2 --repeats 3 --max-rows 4"
  TRACE_ROOT=$TRACE_ROOT/smoke_r4ab; M2_TRACES=$TRACE_ROOT/m2_370m
else
  E24A="--rounds 3 --warmup 10 --repeats 50"
  E2A="--batches 1,4,16 --warmup 10 --repeats 50"
  M2_CAP=$CAP_ARGS
  BIG_CAP="--samples-per-regime 20 --longbench-per-task 8 --longbench-lengths 1024,2048,4096,8192 --max-tokens 8192"
  M2B_STRIDE=8; GDN_STRIDE=2; BIG_GB=30; COLD_MIN=2048
  E23A="--batches 1,4,16 --warmup 10 --repeats 50"
  M2_TRACES=$TRACE_DIR
fi
M2B_TRACES=$TRACE_ROOT/tpds_traces_m2_2p7b
GDN_TRACES=$TRACE_ROOT/tpds_traces_gdn_1p4b
check_space "$OUT_DIR" 2 || exit 1
want() { case " $ONLY " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
# a local checkpoint copy replaces the Hub id when it is complete
local_or_id() { local d="$MODEL_ROOT/${1##*/}"; if [ -f "$d/config.json" ] && ls "$d"/*.safetensors >/dev/null 2>&1
  then echo "$d"; else echo "$1"; fi; }
M2_SRC=AntonV/mamba2-370m-hf
M2_HUB=${TPDS_R4_HF_SHARED:-}/models--AntonV--mamba2-370m-hf
if [ -n "${TPDS_R4_HF_SHARED:-}" ] && [ -f "$M2_HUB/refs/main" ] \
   && [ -f "$M2_HUB/snapshots/$(cat "$M2_HUB/refs/main")/config.json" ]; then
  M2_SRC=$M2_HUB/snapshots/$(cat "$M2_HUB/refs/main")   # a snapshot folder in a shared hub cache
fi
M2B_SRC=$(local_or_id AntonV/mamba2-2.7b-hf); GDN_SRC=$(local_or_id Idiap/gated-deltanet-swa-1.4B-30B)
lanelog "tpds-r4 3090 A/B lane on $TAG (steps: $ONLY; traces $TRACE_ROOT; mamba2 $M2_SRC; 2.7b $M2B_SRC; gdn $GDN_SRC)"

if [ "${TPDS_R4AB_WAIT:-1}" = 1 ] && [ "${TPDS_SMOKE:-0}" != 1 ]; then
  wait_marker E21 || lanelog "E21 did not finish OK; continuing anyway (the timing lock serializes the GPU)"
fi

if want E24; then
  # shellcheck disable=SC2086
  $RUN ${P}E24 -- bash -c "rc=0
    ${L[*]} $PY -m src.experiments.tpds.e24_tfla_chunk $E24A --pre-spin-ms $SPIN $GUARD --out-dir '$OUT_DIR' \
        --tag ${TAG}_r4spin$E24S || rc=1
    ${L[*]} $PY -m src.experiments.tpds.e24_tfla_chunk $E24A --pre-spin-ms 0 $GUARD --out-dir '$OUT_DIR' \
        --tag ${TAG}_r4plain$E24S || rc=1
    $PY -m src.experiments.tpds.e24_tfla_chunk --analyze --out-dir '$OUT_DIR' --tag ${TAG}_r4spin$E24S || rc=1
    $PY -m src.experiments.tpds.e24_tfla_chunk --analyze --out-dir '$OUT_DIR' --tag ${TAG}_r4plain$E24S || rc=1
    exit \$rc"
fi

if want E22; then
  # traces: 370M Mamba-2 (E2 settings), Mamba-2 2.7B and Gated DeltaNet 1.4B (E17 settings)
  # shellcheck disable=SC2086
  "$PY" -m src.experiments.tpds.capture_traces --model mamba2 $M2_CAP --trace-dir "$M2_TRACES" --check-only \
      >> "$LOG_DIR/lanes.log" 2>&1 \
    || $RUN ${P}CAPTURE-r4-m2 -- "${GPU_LOCK[@]}" "$PY" -m src.experiments.tpds.capture_traces --model mamba2 \
         --model-id "$M2_SRC" $M2_CAP --trace-dir "$M2_TRACES" || lanelog "370M Mamba-2 capture failed"
  [ -f "$M2B_TRACES/mamba2/manifest.jsonl" ] \
    || $RUN ${P}CAPTURE-r4-m2-2p7b -- "${GPU_LOCK[@]}" "$PY" -m src.experiments.tpds.capture_traces --model mamba2 \
         --model-id "$M2B_SRC" --layer-stride $M2B_STRIDE $BIG_CAP --max-gb $BIG_GB --trace-dir "$M2B_TRACES" \
    || lanelog "Mamba-2 2.7B capture failed"
  [ -f "$GDN_TRACES/gdn/manifest.jsonl" ] \
    || $RUN ${P}CAPTURE-r4-gdn -- "${GPU_LOCK[@]}" "$PY" -m src.experiments.tpds.capture_gdn_traces \
         --model-id "$GDN_SRC" --layer-stride $GDN_STRIDE $BIG_CAP --max-gb $BIG_GB --trace-dir "$GDN_TRACES" \
    || lanelog "Gated DeltaNet 1.4B capture failed"
  # shellcheck disable=SC2086
  $RUN ${P}E22 -- bash -c "rc=0
    G='--cuda-graph $E2A $GUARD --out-dir $OUT_DIR'
    ${L[*]} $PY -m src.experiments.tpds.e2_oracle_sweep --model mamba2 \$G --trace-dir '$M2_TRACES' --tag ${TAG}_r4graph || rc=1
    ${L[*]} $PY -m src.experiments.tpds.e2_oracle_sweep --model mamba2 \$G --flush-l2 --trace-dir '$M2_TRACES' \
        --tag ${TAG}_r4graphcold || rc=1
    ${L[*]} $PY -m src.experiments.tpds.e2_oracle_sweep --model mamba1 \$G --trace-dir '$TRACE_DIR' --tag ${TAG}_r4graph || rc=1
    ${L[*]} $PY -m src.experiments.tpds.e2_oracle_sweep --model mamba2 \$G --trace-dir '$M2B_TRACES' \
        --tag ${TAG}_2p7b_r4graph || rc=1
    ${L[*]} $PY -m src.experiments.tpds.e2_oracle_sweep --model gdn \$G --trace-dir '$GDN_TRACES' \
        --tag ${TAG}_1p4b_r4graph || rc=1
    ${L[*]} $PY -m src.experiments.tpds.e2_oracle_sweep --model gdn \$G --flush-l2 --trace-dir '$GDN_TRACES' \
        --tag ${TAG}_1p4b_r4graphcold || rc=1
    ${L[*]} $PY -m src.experiments.tpds.e2_oracle_sweep --model mamba2 \$G --flush-l2 --min-length $COLD_MIN \
        --trace-dir '$M2B_TRACES' --tag ${TAG}_2p7b_r4graphcold || rc=1
    exit \$rc"
fi

if want E23; then
  # shellcheck disable=SC2086
  $RUN ${P}E23 -- bash -c "TPDS_PRE_SPIN_MS=$SPIN ${L[*]} $PY -m src.experiments.tpds.e8_refscan --thread-families 32 \
      $E23A $GUARD --trace-dir '$TRACE_DIR' --out-dir '$OUT_DIR' --tag ${TAG}_r4t32"
fi
lanelog "tpds-r4 3090 A/B lane done"
