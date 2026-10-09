#!/usr/bin/env bash
# TPDS cycle tpds-r4, independent re-check P3 (author 2026-10-10), RTX 3090 (host tag vgpu3090).
#   E25 CUDA-graph timing of E22 repeated in three rounds (chunk order shuffled per call and round,
#       round-median oracle, between-round variance in the tie set), warm L2, every (call, chunk)
#       captured into a CUDA graph per round, 10 warmup and 50 timed replays, batch 1, 4 and 16, on the
#       traces E22 used. One step per call set, in the order of the claims they carry:
#         E25a  370M Mamba-2 SSD calls        e2_mamba2_<tag>_r4graph3.{csv,json}, _rounds.jsonl
#         E25b  1.4B Gated DeltaNet calls     e2_gdn_<tag>_1p4b_r4graph3.*
#         E25c  2.7B Mamba-2 SSD calls        e2_mamba2_<tag>_2p7b_r4graph3.*
#         E25d  370M Mamba-1 Triton calls     e2_mamba1_<tag>_r4graph3.*
#       The L2-flushed E22 runs stay single-round (sensitivity only).                ~7 GPU-h in total
#
# Launch:  bash src/revloop/remote.sh launch vgpu3090 tpds-r4-graph3 src/experiments/remote/tpds_lane_r4_3090_graph3.sh
#          with TPDS_TRACE_DIR=/autodl-fs/data/corey_r4_traces_vgpu3090 exported (the E22 traces; the 2.7B and
#          GDN traces live next to the 370M ones)
# Markers: src/logs/revloop/{E25a,E25b,E25c,E25d}.{START,OK,FAIL}
# Env:     TPDS_R4G3_ONLY (default "E25a E25b E25c E25d"), TPDS_R4_GUARD_H (24), TPDS_R4G3_ROUNDS (3).
# Resumable: completed rounds of a call are reused on restart (``_rounds.jsonl``), finished calls skipped.
# TPDS_SMOKE=1 runs four calls per step into smoke tags without the exclusive-GPU guard.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
L=(); [ "${TPDS_NO_GPU_LOCK:-0}" = 1 ] || L=(flock "$LOCK")
P=""; [ "${TPDS_SMOKE:-0}" = 1 ] && P=smoke-
ONLY=${TPDS_R4G3_ONLY:-E25a E25b E25c E25d}
ROUNDS=${TPDS_R4G3_ROUNDS:-3}
GUARD="--exclusive-guard --guard-until $(( $(date +%s) + ${TPDS_R4_GUARD_H:-24} * 3600 ))"
SFX=r4graph3
if [ "${TPDS_SMOKE:-0}" = 1 ]; then
  GUARD=""   # a smoke run next to a real run must not wait for it
  E2A="--batches 1,16 --warmup 2 --repeats 3 --max-rows 4"
  SFX=smoke_r4graph3
else
  E2A="--batches 1,4,16 --warmup 10 --repeats 50"
fi
M2B_TRACES=$TRACE_DIR/tpds_traces_m2_2p7b
GDN_TRACES=$TRACE_DIR/tpds_traces_gdn_1p4b
check_space "$OUT_DIR" 2 || exit 1
want() { case " $ONLY " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
G="--cuda-graph --rounds $ROUNDS $E2A $GUARD --out-dir $OUT_DIR"
lanelog "tpds-r4 3090 graph lane on $TAG (steps: $ONLY; $ROUNDS rounds; traces $TRACE_DIR)"

# shellcheck disable=SC2086
want E25a && $RUN ${P}E25a -- "${L[@]}" "$PY" -m src.experiments.tpds.e2_oracle_sweep --model mamba2 $G \
    --trace-dir "$TRACE_DIR" --tag "${TAG}_$SFX"
# shellcheck disable=SC2086
want E25b && $RUN ${P}E25b -- "${L[@]}" "$PY" -m src.experiments.tpds.e2_oracle_sweep --model gdn $G \
    --trace-dir "$GDN_TRACES" --tag "${TAG}_1p4b_$SFX"
# shellcheck disable=SC2086
want E25c && $RUN ${P}E25c -- "${L[@]}" "$PY" -m src.experiments.tpds.e2_oracle_sweep --model mamba2 $G \
    --trace-dir "$M2B_TRACES" --tag "${TAG}_2p7b_$SFX"
# shellcheck disable=SC2086
want E25d && $RUN ${P}E25d -- "${L[@]}" "$PY" -m src.experiments.tpds.e2_oracle_sweep --model mamba1 $G \
    --trace-dir "$TRACE_DIR" --tag "${TAG}_$SFX"
lanelog "tpds-r4 3090 graph lane done"
