#!/usr/bin/env bash
# TPDS cycle tpds-r4 GPU lane on the RTX 3090 (host tag vgpu3090): E19, then E21.
#   E19 (review M2b, required). The 2360 captured 370M Mamba-1 calls (E2, batch 1/4/16) in three
#       rounds, chunk order shuffled per call and per round, so the per-call oracle can be defined on
#       round medians, in both conventions: device time (GPU-side pre-spin; tag <host>_r4spin) and
#       launch-inclusive (tag <host>_r4plain). The device-time calibration of the cost model on this
#       GPU exists already (E15, tag <host>_r3spin).                                       ~1.4 GPU-h
#   E21 (review M10b). 64 calls of the serving key (length bucket 2048, batch 4, L mod 16 != 0), of
#       which the captured workload holds 11. The kernels are value-invariant, so the calls are
#       synthesized from the captured traces: each stacks four prompts of one cross-validation fold
#       whose captured length reaches a target length, cropped to it (a prefix of a causal scan input
#       is a valid scan input); the 64 targets are distinct unaligned lengths in 2049..4095, folds and
#       layers rotate. Three rounds in both conventions (tags <host>_r4spin_u2048b4,
#       <host>_r4plain_u2048b4); the rows have the E2 format, so they merge with the E19 tables.  ~5 min
# Before any timing the lane verifies the captured 370M Mamba-1 traces in $TRACE_DIR (no model load)
# and re-captures them when they are missing or incomplete (~22 GB, ~10 min; needs the
# state-spaces/mamba-370m-hf checkpoint in the HF cache or via HF_ENDPOINT).
#
# Launch:  wsl bash src/revloop/remote.sh launch vgpu3090 tpds-r4 src/experiments/remote/tpds_lane_r4_3090.sh
# Markers: src/logs/revloop/{CAPTURE-r4,E19,E21}.{START,OK,FAIL} (TPDS_SMOKE=1: smoke- prefix)
# Env:     TPDS_R4_ONLY (default "E19 E21"), TPDS_R4_SPIN_MS (20), TPDS_R4_ROUNDS (3),
#          TPDS_R4_E21_CALLS (64), TPDS_R4_GUARD (1: the E2 sweeps pause while other processes use the
#          GPU and re-time the calls they overlapped), TPDS_R4_GUARD_H (12).
# Every GPU step holds the exclusive timing lock. Resumable: every step skips completed rows and
# rounds; a failed step does not stop the others, its marker records it.
# TPDS_SMOKE=1 runs a tiny version on the smoke traces (outputs in the smoke directory).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
L=(); [ "${TPDS_NO_GPU_LOCK:-0}" = 1 ] || L=(flock "$LOCK")
P=""; [ "${TPDS_SMOKE:-0}" = 1 ] && P=smoke-
ONLY=${TPDS_R4_ONLY:-E19 E21}
ROUNDS=${TPDS_R4_ROUNDS:-3}
GUARD=""
if [ "${TPDS_R4_GUARD:-1}" = 1 ]; then
  GUARD="--exclusive-guard --guard-until $(( $(date +%s) + ${TPDS_R4_GUARD_H:-12} * 3600 ))"
fi
if [ "${TPDS_SMOKE:-0}" = 1 ]; then
  TAG=${TPDS_TAG:-smoke-r4vgpu3090}
  SPIN=${TPDS_R4_SPIN_MS:-5}
  E2A="--batches 1,4 --warmup 2 --repeats 3 --max-rows 4"
  # the smoke traces stop at 1024 tokens: synthesize the unaligned bucket-512 batch-4 key instead
  SYNTH="--synth-calls ${TPDS_R4_E21_CALLS:-6} --synth-bucket 512 --synth-batch 4 --synth-align unaligned"
  E21A="--warmup 2 --repeats 3"
  KEYTAG=u512b4
else
  SPIN=${TPDS_R4_SPIN_MS:-20}
  E2A="--batches 1,4,16 --warmup 10 --repeats 50"
  SYNTH="--synth-calls ${TPDS_R4_E21_CALLS:-64} --synth-bucket 2048 --synth-batch 4 --synth-align unaligned"
  E21A="--warmup 10 --repeats 50"
  KEYTAG=u2048b4
fi
check_space "$OUT_DIR" 2 || exit 1
want() { case " $ONLY " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
lanelog "tpds-r4 3090 lane start on $TAG (steps: $ONLY; pre-spin ${SPIN} ms; rounds $ROUNDS; guard ${GUARD:+on})"
ensure_traces "${P}CAPTURE-r4" mamba1 || { lanelog "370M Mamba-1 traces unavailable at $TRACE_DIR; stopping"; exit 1; }
COMMON="--trace-dir '$TRACE_DIR' --out-dir '$OUT_DIR'"

if want E19; then
  # shellcheck disable=SC2086
  $RUN ${P}E19 -- bash -c "rc=0
    ${L[*]} $PY -m src.experiments.tpds.e2_oracle_sweep --model mamba1 $E2A --rounds $ROUNDS --pre-spin-ms $SPIN \
        $GUARD $COMMON --tag ${TAG}_r4spin || rc=1
    ${L[*]} $PY -m src.experiments.tpds.e2_oracle_sweep --model mamba1 $E2A --rounds $ROUNDS --pre-spin-ms 0 \
        $GUARD $COMMON --tag ${TAG}_r4plain || rc=1
    exit \$rc"
fi

if want E21; then
  # shellcheck disable=SC2086
  $RUN ${P}E21 -- bash -c "rc=0
    ${L[*]} $PY -m src.experiments.tpds.e2_oracle_sweep --model mamba1 $SYNTH $E21A --rounds $ROUNDS \
        --pre-spin-ms $SPIN $GUARD $COMMON --tag ${TAG}_r4spin_$KEYTAG || rc=1
    ${L[*]} $PY -m src.experiments.tpds.e2_oracle_sweep --model mamba1 $SYNTH $E21A --rounds $ROUNDS \
        --pre-spin-ms 0 $GUARD $COMMON --tag ${TAG}_r4plain_$KEYTAG || rc=1
    exit \$rc"
fi
lanelog "tpds-r4 3090 lane done"
