#!/usr/bin/env bash
# TPDS cycle tpds-r4 GPU lane on the RTX PRO 6000 (host tag pro6000): E18, then E20.
#   E18 (review M2a + M2b, required). Device-time cost-model calibration of the 370M Mamba-1 scan,
#       with the E15 arguments and a GPU-side pre-spin in front of every timing block: the E4 Mamba-1
#       launch sweep plus re-measurement of the twelve tuning keys, and the E1 Mamba-1 medians
#       (tag <host>_r4spin). Then the 2360 captured 370M Mamba-1 calls (E2, batch 1/4/16) in three
#       rounds, chunk order shuffled per call and per round, so the per-call oracle can be defined on
#       round medians, in both conventions: device time (pre-spin; tag <host>_r4spin) and
#       launch-inclusive (tag <host>_r4plain).                                        ~1.2 GPU-h
#   E20 (review M3a). Cold L2: the L2 is overwritten before every timed repeat (as Triton's do_bench
#       does), device time (pre-spin): the E4 Mamba-1 launch sweep and the twelve tuning keys at the
#       six chunks, the E1 Mamba-1 medians (both for a cold refit of the cost model), and the captured
#       calls with L >= 2048 (544 of 2360) in three rounds (tag <host>_r4cold).             ~0.6 GPU-h
# Before any timing the lane verifies the captured 370M Mamba-1 traces in $TRACE_DIR (no model load)
# and re-captures them when they are missing or incomplete (~22 GB, a few minutes; needs the
# state-spaces/mamba-370m-hf checkpoint in the HF cache or via HF_ENDPOINT).
#
# Launch:  wsl bash src/revloop/remote.sh launch pro6000 tpds-r4 src/experiments/remote/tpds_lane_r4_pro6000.sh
# Markers: src/logs/revloop/{CAPTURE-r4,E18,E20}.{START,OK,FAIL} (TPDS_SMOKE=1: smoke- prefix)
# Env:     TPDS_R4_ONLY (default "E18 E20"), TPDS_R4_SPIN_MS (20), TPDS_R4_ROUNDS (3),
#          TPDS_R4_GUARD (1: every timing step (E4 keys, E1 cells, E2 calls) pauses while other processes
#          use the GPU and re-times the units they overlapped; 0 saves ~0.2 s of nvidia-smi queries per call), TPDS_R4_GUARD_H (12).
# Every GPU step holds the exclusive timing lock. Resumable: every step skips completed rows (the
# E2 sweeps also completed rounds); a failed step does not stop the others, its marker records it.
# TPDS_SMOKE=1 runs a tiny version on the smoke traces (outputs in the smoke directory).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
L=(); [ "${TPDS_NO_GPU_LOCK:-0}" = 1 ] || L=(flock "$LOCK")
P=""; [ "${TPDS_SMOKE:-0}" = 1 ] && P=smoke-
ONLY=${TPDS_R4_ONLY:-E18 E20}
ROUNDS=${TPDS_R4_ROUNDS:-3}
GUARD=""
if [ "${TPDS_R4_GUARD:-1}" = 1 ]; then
  GUARD="--exclusive-guard --guard-until $(( $(date +%s) + ${TPDS_R4_GUARD_H:-12} * 3600 ))"
fi
if [ "${TPDS_SMOKE:-0}" = 1 ]; then
  TAG=${TPDS_TAG:-smoke-r4pro6000}
  SPIN=${TPDS_R4_SPIN_MS:-5}
  E4A="--parts m1 --lengths 128 --batches 1 --search-warmup 1 --search-repeats 2 --warmup 2 --repeats 3"
  E1A="--families mamba1 --lengths 512 --warmup 2 --repeats 3 --rounds 1 --inputs gaussian,real"
  E2A="--batches 1,4 --warmup 2 --repeats 3 --max-rows 4"
  MINLEN=1024   # the smoke traces stop at 1024 tokens
else
  SPIN=${TPDS_R4_SPIN_MS:-20}
  E4A="--parts m1 --lengths 128,512,2048,8192 --batches 1,4,16 --search-warmup 3 --search-repeats 10 --warmup 10 --repeats 100"
  E1A="--families mamba1 --lengths 512,2048,8192 --batch 1 --warmup 10 --repeats 100 --rounds 3"
  E2A="--batches 1,4,16 --warmup 10 --repeats 50"
  MINLEN=2048
fi
check_space "$OUT_DIR" 2 || exit 1
want() { case " $ONLY " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
lanelog "tpds-r4 pro6000 lane start on $TAG (steps: $ONLY; pre-spin ${SPIN} ms; rounds $ROUNDS; guard ${GUARD:+on})"
ensure_traces "${P}CAPTURE-r4" mamba1 || { lanelog "370M Mamba-1 traces unavailable at $TRACE_DIR; stopping"; exit 1; }
COMMON="--trace-dir '$TRACE_DIR' --out-dir '$OUT_DIR'"

if want E18; then
  # shellcheck disable=SC2086
  $RUN ${P}E18 -- bash -c "rc=0
    TPDS_PRE_SPIN_MS=$SPIN ${L[*]} $PY -m src.experiments.tpds.e4_autotune $E4A $GUARD $COMMON --tag ${TAG}_r4spin || rc=1
    TPDS_PRE_SPIN_MS=$SPIN ${L[*]} $PY -m src.experiments.tpds.e1_value_invariance $E1A $GUARD $COMMON --tag ${TAG}_r4spin || rc=1
    ${L[*]} $PY -m src.experiments.tpds.e2_oracle_sweep --model mamba1 $E2A --rounds $ROUNDS --pre-spin-ms $SPIN \
        $GUARD $COMMON --tag ${TAG}_r4spin || rc=1
    ${L[*]} $PY -m src.experiments.tpds.e2_oracle_sweep --model mamba1 $E2A --rounds $ROUNDS --pre-spin-ms 0 \
        $GUARD $COMMON --tag ${TAG}_r4plain || rc=1
    exit \$rc"
fi

if want E20; then
  # shellcheck disable=SC2086
  $RUN ${P}E20 -- bash -c "rc=0
    TPDS_PRE_SPIN_MS=$SPIN ${L[*]} $PY -m src.experiments.tpds.e4_autotune $E4A --flush-l2 $GUARD $COMMON --tag ${TAG}_r4cold || rc=1
    TPDS_PRE_SPIN_MS=$SPIN ${L[*]} $PY -m src.experiments.tpds.e1_value_invariance $E1A --flush-l2 $GUARD $COMMON \
        --tag ${TAG}_r4cold || rc=1
    ${L[*]} $PY -m src.experiments.tpds.e2_oracle_sweep --model mamba1 $E2A --rounds $ROUNDS --flush-l2 \
        --min-length $MINLEN --pre-spin-ms $SPIN $GUARD $COMMON --tag ${TAG}_r4cold || rc=1
    exit \$rc"
fi
lanelog "tpds-r4 pro6000 lane done"
