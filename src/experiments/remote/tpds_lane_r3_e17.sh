#!/usr/bin/env bash
# TPDS cycle tpds-r3, review item M2b (E17): production kernels out of the launch floor, real Gated
# DeltaNet traces. Host: RTX PRO 6000 (shared with other projects: captures run alongside them, the
# kernel timings wait for an exclusive GPU).
#   E17a-cap  capture the Mamba-2 2.7B scan inputs (AntonV/mamba2-2.7b-hf, 8 of 64 layers, 192 prompts)
#   E17b-cap  capture the Gated DeltaNet kernel inputs of the 1.4B hybrid Idiap/gated-deltanet-swa-1.4B-30B
#             (6 of its 12 Gated DeltaNet layers, 192 prompts)
#   E17c      synthetic work-bound sweep: Mamba-2 SSD at the 370M/1.3B/2.7B widths and Gated DeltaNet
#             at five head layouts, L 512..8192, batch 1/4/16, every chunk, 3 rounds        ~30 min
#   E17a      per-call chunk sweep of the captured Mamba-2 2.7B calls, batch 1/4/16 (device time,
#             pre-spin 20 ms)                                                              ~1.5 h
#   E17b      per-call sweep of the captured Gated DeltaNet calls at chunks 16/32/64 (production
#             kernel path), batch 1/4/16                                                   ~30 min
# Checkpoints: tpds_r3_e17_prep.sh downloads them to $TPDS_E17_MODEL_ROOT (CPU; no-GPU mode is enough);
# without a local copy the Hub id is used (HF_ENDPOINT from .tpds_env).
#
# Launch:  wsl bash src/revloop/remote.sh launch pro6000 tpds-r3-e17 src/experiments/remote/tpds_lane_r3_e17.sh
# Markers: src/logs/revloop/{E17a-cap,E17b-cap,E17c,E17a,E17b}.{START,OK,FAIL}
# TPDS_E17_ONLY="E17c E17a" selects steps; TPDS_SMOKE=1 runs a tiny version with the 370M/360M checkpoints.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
L=(); [ "${TPDS_NO_GPU_LOCK:-0}" = 1 ] || L=(flock "$LOCK")
ONLY=${TPDS_E17_ONLY:-E17a-cap E17b-cap E17c E17a E17b}
P=""; [ "${TPDS_SMOKE:-0}" = 1 ] && P=smoke-
E17_ROOT=${TPDS_E17_MODEL_ROOT:-/root/autodl-tmp/hf_models}
TRACE_ROOT=${TPDS_E17_TRACE_ROOT:-src/outputs}
GUARD_UNTIL=$(( $(date +%s) + ${TPDS_E17_GUARD_H:-12} * 3600 ))
if [ "${TPDS_SMOKE:-0}" = 1 ]; then
  M2_ID=${TPDS_E17_M2_ID:-AntonV/mamba2-370m-hf}; M2_TAG=370m; M2_STRIDE=12
  GDN_ID=${TPDS_E17_GDN_ID:-puigde/gated-deltanet-360M-15B-slimpajama}; GDN_TAG=360m; GDN_STRIDE=7
  CAP="--samples-per-regime 1 --longbench-per-task 1 --max-tokens 1024 --limit-prompts 6"
  TPDS_E17_M2_MAX_GB=2; TPDS_E17_GDN_MAX_GB=2
  E2="--batches 1,4 --warmup 2 --repeats 3 --max-rows 8 --pre-spin-ms 5"
  E17C="--ssd-widths 2048 --gdn-shapes 4x256x256 --lengths 256,1024 --batches 1,4 --rounds 1 --warmup 2 --repeats 5"
  TRACE_ROOT=${TPDS_E17_TRACE_ROOT:-src/outputs/smoke_e17}
else
  M2_ID=${TPDS_E17_M2_ID:-AntonV/mamba2-2.7b-hf}; M2_TAG=2p7b; M2_STRIDE=8
  GDN_ID=${TPDS_E17_GDN_ID:-Idiap/gated-deltanet-swa-1.4B-30B}; GDN_TAG=1p4b; GDN_STRIDE=2
  CAP="--samples-per-regime 20 --longbench-per-task 8 --longbench-lengths 1024,2048,4096,8192 --max-tokens 8192"
  E2="--batches 1,4,16 --warmup 10 --repeats 50 --pre-spin-ms 20"
  E17C="--rounds 3 --warmup 10 --repeats 50 --record-gpu-state"
  # only the captures write traces (check_space creates the directory, so skip it when the traces are copied in)
  case " ${TPDS_E17_ONLY:-E17a-cap E17b-cap} " in *-cap*)
    check_space "$TRACE_ROOT" 60 || { lanelog "E17 needs ~60 GB for traces"; exit 1; } ;; esac
fi
check_space "$OUT_DIR" 2 || exit 1
# a local checkpoint copy replaces the Hub id when it is complete
local_or_id() { local d="$E17_ROOT/${1##*/}"; if [ -f "$d/config.json" ] && ! ls "$d"/*.aria2 >/dev/null 2>&1 \
  && ls "$d"/*.safetensors >/dev/null 2>&1; then echo "$d"; else echo "$1"; fi; }
M2_SRC=$(local_or_id "$M2_ID"); GDN_SRC=$(local_or_id "$GDN_ID")
M2_TRACES=$TRACE_ROOT/tpds_traces_m2_$M2_TAG; GDN_TRACES=$TRACE_ROOT/tpds_traces_gdn_$GDN_TAG
lanelog "tpds-r3 E17 lane start on $TAG (steps: $ONLY; mamba2 $M2_SRC; gdn $GDN_SRC)"
want() { case " $ONLY " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

if want E17a-cap; then
  # shellcheck disable=SC2086
  $RUN ${P}E17a-cap -- "$PY" -m src.experiments.tpds.capture_traces --model mamba2 --model-id "$M2_SRC" \
    --layer-stride $M2_STRIDE $CAP --max-gb ${TPDS_E17_M2_MAX_GB:-30} --trace-dir "$M2_TRACES" \
    || lanelog "E17a capture failed (E17a will be skipped)"
fi
if want E17b-cap; then
  # shellcheck disable=SC2086
  $RUN ${P}E17b-cap -- "$PY" -m src.experiments.tpds.capture_gdn_traces --model-id "$GDN_SRC" \
    --layer-stride $GDN_STRIDE $CAP --max-gb ${TPDS_E17_GDN_MAX_GB:-20} --trace-dir "$GDN_TRACES" \
    || lanelog "E17b capture failed (E17b will be skipped)"
fi
if want E17c; then
  # shellcheck disable=SC2086
  $RUN ${P}E17c -- "${L[@]}" "$PY" -m src.experiments.tpds.e17_workbound $E17C --exclusive-guard \
    --guard-until "$GUARD_UNTIL" --out-dir "$OUT_DIR" --tag "$TAG" || lanelog "E17c failed (continuing)"
fi
# Traces captured on another host and copied over (e.g. through /autodl-fs): with TPDS_E17_WAIT_TRACES=1
# the per-call sweeps start only once the copy has written $TRACE_ROOT/.complete (at most 6 h).
if [ "${TPDS_E17_WAIT_TRACES:-0}" = 1 ] && { want E17a || want E17b; }; then
  waited=0
  until [ -f "$TRACE_ROOT/.complete" ] || [ $waited -ge 21600 ]; do sleep 60; waited=$((waited + 60)); done
  [ -f "$TRACE_ROOT/.complete" ] && lanelog "traces complete at $TRACE_ROOT (waited ${waited}s)" \
    || lanelog "traces at $TRACE_ROOT still incomplete after 6 h"
fi
if want E17a; then
  if [ -f "$M2_TRACES/mamba2/manifest.jsonl" ]; then
    # shellcheck disable=SC2086
    $RUN ${P}E17a -- "${L[@]}" "$PY" -m src.experiments.tpds.e2_oracle_sweep --model mamba2 $E2 --exclusive-guard \
      --guard-until "$GUARD_UNTIL" --trace-dir "$M2_TRACES" --out-dir "$OUT_DIR" --tag "${TAG}_${M2_TAG}_r3" \
      || lanelog "E17a failed (continuing)"
  else lanelog "E17a skipped: no Mamba-2 traces at $M2_TRACES"; : > "$MARK_DIR/${P}E17a.FAIL"; fi
fi
if want E17b; then
  if [ -f "$GDN_TRACES/gdn/manifest.jsonl" ]; then
    # shellcheck disable=SC2086
    $RUN ${P}E17b -- "${L[@]}" "$PY" -m src.experiments.tpds.e2_oracle_sweep --model gdn $E2 --exclusive-guard \
      --guard-until "$GUARD_UNTIL" --trace-dir "$GDN_TRACES" --out-dir "$OUT_DIR" --tag "${TAG}_${GDN_TAG}_r3" \
      || lanelog "E17b failed"
  else lanelog "E17b skipped: no Gated DeltaNet traces at $GDN_TRACES"; : > "$MARK_DIR/${P}E17b.FAIL"; fi
fi
lanelog "tpds-r3 E17 lane done"
