#!/usr/bin/env bash
# Cycle taco-r1 GPU lane on the RTX PRO 6000 (host tag pro6000).
#   E26 (review M6a). CUDA-graph timing of the captured calls on a second GPU, with the protocol of the
#       RTX 3090 runs (E25): three rounds, chunk order shuffled per call and round, round-median oracle,
#       warm L2, every (call, chunk) captured into a CUDA graph per round, 10 warmup and 50 timed
#       replays, batch 1, 4 and 16, on the same traces as the RTX 3090 (identical calls):
#         E26c  2.7B Mamba-2 SSD calls        e2_mamba2_<tag>_2p7b_taco_graph3.*
#         E26a  370M Mamba-2 SSD calls        e2_mamba2_<tag>_taco_graph3.*
#         E26b  1.4B Gated DeltaNet calls     e2_gdn_<tag>_1p4b_taco_graph3.*
#         E26d  370M Mamba-1 Triton calls     e2_mamba1_<tag>_taco_graph3.*
#   E27 (review M6b). Packed variable-length SSD and Gated DeltaNet calls sampled from serving traces
#       (packed_steps_taco.json), every chunk, fresh and carried state, three rounds, device time.
#                                                                                   ~6-8 GPU-h in total
#
# Launch:  wsl bash src/revloop/remote.sh launch pro6000 taco-r1 src/experiments/remote/taco_lane_pro6000.sh
# Markers: src/logs/revloop/{E26c,E26a,E26b,E27,E26d}.{START,OK,FAIL}
# Env:     TACO_ONLY (default "E26c E26a E26b E27 E26d"), TACO_TRACE_ROOT (the shared traces of the
#          RTX 3090 runs), TACO_GUARD_H (24), TACO_ROUNDS (3).
# Every GPU step holds the exclusive timing lock and the exclusive-GPU guard. Resumable: completed rounds
# and rows are reused on restart. TPDS_SMOKE=1 runs a tiny version without the guard.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
L=(); [ "${TPDS_NO_GPU_LOCK:-0}" = 1 ] || L=(flock "$LOCK")
P=""; [ "${TPDS_SMOKE:-0}" = 1 ] && P=smoke-
ONLY=${TACO_ONLY:-E26c E26a E26b E27 E26d}
ROUNDS=${TACO_ROUNDS:-3}
ROOT_TR=${TACO_TRACE_ROOT:-/autodl-fs/data/corey_r4_traces_vgpu3090}
M2B_TRACES=$ROOT_TR/tpds_traces_m2_2p7b
GDN_TRACES=$ROOT_TR/tpds_traces_gdn_1p4b
GUARD="--exclusive-guard --guard-until $(( $(date +%s) + ${TACO_GUARD_H:-24} * 3600 ))"
SFX=taco_graph3
E27A="--rounds $ROUNDS"
if [ "${TPDS_SMOKE:-0}" = 1 ]; then
  GUARD=""
  E2A="--batches 1,16 --warmup 2 --repeats 3 --max-rows 4"
  E27A="--rounds 1 --warmup 2 --repeats 3 --max-steps 3"
  SFX=smoke_taco_graph3
else
  E2A="--batches 1,4,16 --warmup 10 --repeats 50"
fi
check_space "$OUT_DIR" 2 || exit 1
want() { case " $ONLY " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
G="--cuda-graph --rounds $ROUNDS $E2A $GUARD --out-dir $OUT_DIR"
lanelog "taco-r1 pro6000 lane on $TAG (steps: $ONLY; $ROUNDS rounds; traces $ROOT_TR)"

# shellcheck disable=SC2086
want E26c && $RUN ${P}E26c -- "${L[@]}" "$PY" -m src.experiments.tpds.e2_oracle_sweep --model mamba2 $G \
    --trace-dir "$M2B_TRACES" --tag "${TAG}_2p7b_$SFX"
# shellcheck disable=SC2086
want E26a && $RUN ${P}E26a -- "${L[@]}" "$PY" -m src.experiments.tpds.e2_oracle_sweep --model mamba2 $G \
    --trace-dir "$ROOT_TR" --tag "${TAG}_$SFX"
# shellcheck disable=SC2086
want E26b && $RUN ${P}E26b -- "${L[@]}" "$PY" -m src.experiments.tpds.e2_oracle_sweep --model gdn $G \
    --trace-dir "$GDN_TRACES" --tag "${TAG}_1p4b_$SFX"
# shellcheck disable=SC2086
want E27 && $RUN ${P}E27 -- "${L[@]}" "$PY" -m src.experiments.tpds.e27_varlen $E27A $GUARD \
    --ssd370m-traces "$ROOT_TR" --ssd2p7b-traces "$M2B_TRACES" --gdn-traces "$GDN_TRACES" \
    --out-dir "$OUT_DIR" --tag "${TAG}_${P}taco"
# shellcheck disable=SC2086
want E26d && $RUN ${P}E26d -- "${L[@]}" "$PY" -m src.experiments.tpds.e2_oracle_sweep --model mamba1 $G \
    --trace-dir "$ROOT_TR" --tag "${TAG}_$SFX"
lanelog "taco-r1 pro6000 lane done"
