#!/usr/bin/env bash
# Cycle taco-r2 lanes on the 4c lab host (stock RTX 3090 24 GB boards, sm_86; physical GPUs 0 and 1 may be used).
#   E32 (review M3b, M7e): full-model prefill captured as one CUDA graph, default chunk vs the held-out shape
#       table vs a no-scan forward, SSD 370M, Gated DeltaNet 1.4B and SSD 2.7B (e32_model_graph_prefill.py).
#       Session 1 (E32a-c) and, after the other lane's work, session 2 (E32d-f) in a separate process.
#   E33 (review M5c, M7d): value invariance of the production SSD and Gated DeltaNet scans above the launch
#       floor in CUDA-graph replay, designs A (values only) and C (placement only), captured activations in the
#       same process (e33_value_invariance_graph.py); then aligned vs unaligned lengths (E33b).
#   E34 (review M2d): the Triton Mamba-1 scan's whole launch space (184 configurations) at the twelve tuning keys
#       under the full protocol, so that the chunk effect can be read at fixed launch configurations (e34_fixed_launch.py).
#   E35 (review M5e): per-kernel breakdown (time, grid, registers, estimated occupancy) of the Triton Mamba-1, SSD and
#       Gated DeltaNet scans versus chunk with the PyTorch profiler (e35_mechanism.py); the hardware-counter part
#       needs administrator rights and is run separately by the account owner (taco_ncu_4c.sh).
# One lane per GPU (each with its own timing lock, so the two lanes run concurrently):
#   TACO_LANE=G0 CUDA_VISIBLE_DEVICES=0 bash src/experiments/remote/taco_lane_4c_r2.sh
#   TACO_LANE=G1 CUDA_VISIBLE_DEVICES=1 bash src/experiments/remote/taco_lane_4c_r2.sh
#   TACO_LANE=G1b CUDA_VISIBLE_DEVICES=1 bash src/experiments/remote/taco_lane_4c_r2.sh   (after G1: E35)
#   G0b (after G0): E36 packed steps under CUDA graphs at padded capture sizes (review M3d);
#   G1c (after G1b): E37 graph-timed calibration keys of the work-term model (M4f), E33c the Triton scan's value
#       invariance in graph replay with captured Mamba-1 activations (M5c), E38 SSD bf16 vs fp16 (M7f),
#       E30s2 second E30 session (M5f).
# Update this file on the host by rename (scp to a temporary name, then mv), never in place: running lanes read it.
# Smoke tests only:  bash src/experiments/remote/taco_lane_4c_r2.sh step e32_smoke | e33_smoke
# Markers: src/logs/revloop/<ID>.{START,OK,FAIL}; results come back by rsync (remote.sh fetch 4c <IDs>).
# Env: TACO_TRACE_ROOT, TACO_HF_HOME (/data/hf_home), TACO_HF_HUB_CACHE (/data/hf_cache), TACO_GUARD_H (48),
#      TACO_S2_AFTER (marker the second E32 session waits for; default E33b).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
H=4c
T=${TACO_TRACE_ROOT:-/data/mabo1215_taco/e30_inputs/autodl-fs/data/corey_r4_traces_vgpu3090}
export HF_HOME=${TACO_HF_HOME:-/data/hf_home}
export HF_HUB_CACHE=${TACO_HF_HUB_CACHE:-/data/hf_cache}
GPU=${CUDA_VISIBLE_DEVICES:-0}
LOCK=$LOG_DIR/gpu_timing_g${GPU}.lock
SMOKE_DIR=$OUT_DIR/smoke
E32="$PY -m src.experiments.tpds.e32_model_graph_prefill"
E33="$PY -m src.experiments.tpds.e33_value_invariance_graph"
M2_ID=${TACO_M2_ID:-AntonV/mamba2-370m-hf}
M27_ID=${TACO_M27_ID:-benchang1110/mamba2-2.7b-hf}
GDN_ID=${TACO_GDN_ID:-Idiap/gated-deltanet-swa-1.4B-30B}
TRACES="ssd370m=$T,ssd2p7b=$T/tpds_traces_m2_2p7b,gdn1p4b=$T/tpds_traces_gdn_1p4b"
M1T=${TACO_M1_TRACES:-/data/mabo1215_taco/m1_traces}
GUARD="--exclusive-guard --guard-until $(( $(date +%s) + ${TACO_GUARD_H:-48} * 3600 )) --record-gpu-state"

e32() {  # e32 <session> <model> <model-id> <label> [extra args]
  local s=$1 m=$2 id=$3 lab=$4; shift 4
  flock "$LOCK" $E32 --model "$m" --model-id "$id" --label "$lab" $GUARD --tag ${H}_e32${s}_${lab} "$@"
}

if [ "${1:-}" = "step" ]; then
  case "$2" in
    e32_smoke)
      S="--lengths 512,3000 --batches 1 --rounds 1 --warmup 2 --repeats 3 --eager-repeats 3 --out-dir $SMOKE_DIR"
      flock "$LOCK" $E32 --model mamba2 --model-id "$M2_ID" --label ssd370m $S --tag ${H}_e32smoke_ssd370m &&
      flock "$LOCK" $E32 --model gdn --model-id "$GDN_ID" --label gdn1p4b $S --tag ${H}_e32smoke_gdn1p4b ;;
    e32_smoke27)
      S="--lengths 512,3000 --batches 1 --rounds 1 --warmup 2 --repeats 3 --eager-repeats 3 --out-dir $SMOKE_DIR"
      flock "$LOCK" $E32 --model mamba2 --model-id "$M27_ID" --label ssd2p7b $S --tag ${H}_e32smoke_ssd2p7b ;;
    e33_smoke)
      flock "$LOCK" $E33 --families ssd370m,gdn1p4b --shapes 2048x1 --trace-dirs "$TRACES" --draws 1 --placements 2 \
        --rounds 2 --warmup 2 --repeats 5 --out-dir "$SMOKE_DIR" --tag ${H}_e33smoke ;;
    e32a) e32 s1 mamba2 "$M2_ID" ssd370m ;;
    e32b) e32 s1 gdn "$GDN_ID" gdn1p4b ;;
    e32c) e32 s1 mamba2 "$M27_ID" ssd2p7b ;;
    e32d) e32 s2 mamba2 "$M2_ID" ssd370m ;;
    e32e) e32 s2 gdn "$GDN_ID" gdn1p4b ;;
    e32f) e32 s2 mamba2 "$M27_ID" ssd2p7b ;;
    e32_analyze)
      for lab in ssd370m gdn1p4b ssd2p7b; do
        $E32 --analyze --input-tags ${H}_e32s1_${lab},${H}_e32s2_${lab} --tag ${H}_e32s1_${lab} || exit 1
      done ;;
    e33a) flock "$LOCK" $E33 --trace-dirs "$TRACES" $GUARD --tag ${H}_e33 ;;
    e33b) flock "$LOCK" $E33 --trace-dirs "$TRACES" --shapes 2040x1,2047x1,2048x1,2049x1,2056x1 --designs A --draws 1 \
            $GUARD --tag ${H}_e33align ;;
    e34_smoke) flock "$LOCK" $PY -m src.experiments.tpds.e34_fixed_launch --lengths 512 --batches 1 --rounds 1 \
                 --warmup 2 --repeats 3 --out-dir "$SMOKE_DIR" --tag ${H}_e34smoke ;;
    e34) flock "$LOCK" $PY -m src.experiments.tpds.e34_fixed_launch $GUARD --tag ${H}_e34 ;;
    e35_smoke) flock "$LOCK" $PY -m src.experiments.tpds.e35_mechanism --part breakdown --families ssd,gdn \
                 --gdn-trace-dir "$T/tpds_traces_gdn_1p4b" --repeats 2 --out-dir "$SMOKE_DIR" --tag ${H}_e35smoke ;;
    e35) flock "$LOCK" $PY -m src.experiments.tpds.e35_mechanism --part breakdown \
           --gdn-trace-dir "$T/tpds_traces_gdn_1p4b" --tag ${H}_e35 ;;
    e36_smoke) flock "$LOCK" $PY -m src.experiments.tpds.e36_packed_graph --max-steps 3 --rounds 2 --repeats 5 \
                 --ssd370m-traces "$T" --ssd2p7b-traces "$T/tpds_traces_m2_2p7b" --gdn-traces "$T/tpds_traces_gdn_1p4b" \
                 --out-dir "$SMOKE_DIR" --tag ${H}_e36smoke ;;
    e36) flock "$LOCK" $PY -m src.experiments.tpds.e36_packed_graph --ssd370m-traces "$T" \
           --ssd2p7b-traces "$T/tpds_traces_m2_2p7b" --gdn-traces "$T/tpds_traces_gdn_1p4b" $GUARD --tag ${H}_e36 ;;
    e37) flock "$LOCK" $E33 --trace-dirs "$TRACES" --designs A --inputs 'gaussian#0' --rounds 3 \
           --shapes 128x1,512x1,2048x1,8192x1,128x4,512x4,2048x4,8192x4,128x16,512x16,2048x16,8192x16 \
           $GUARD --tag ${H}_e37calib ;;
    m1capture)  # a few Mamba-1 370M prefill calls on this host for the captured-activation input of E33c
      flock "$LOCK" $PY -m src.experiments.tpds.capture_traces --model mamba1 --layers 24 --samples-per-regime 1 \
        --longbench-per-task 1 --longbench-lengths 8192 --max-tokens 8192 --limit-prompts 9 --max-gb 5 \
        --trace-dir "$M1T" ;;
    e33c) flock "$LOCK" $E33 --families m1triton --shapes 2048x1,8192x1,2048x4 --trace-dirs "m1triton=$M1T" \
            $GUARD --tag ${H}_e33m1 ;;
    e38)  # M7f: SSD in bf16 and fp16 on the same shapes, Gaussian and captured inputs, same process order per dtype
      for dt in float16 bfloat16; do
        flock "$LOCK" $E33 --families ssd370m,ssd2p7b --shapes 512x1,2048x1,16384x1,4096x4 --designs A \
          --inputs 'gaussian#0,real' --ssd-dtype $dt --trace-dirs "$TRACES" $GUARD --tag ${H}_e38_$dt || exit 1
      done ;;
    e30s2)  # second session of E30 on this board: cross-session and cross-board reproducibility (review M5f)
      E30="$PY -m src.experiments.tpds.e30_autotune_state"
      G30="--exclusive-guard --guard-until $(( $(date +%s) + ${TACO_GUARD_H:-48} * 3600 ))"
      flock "$LOCK" $E30 --model mamba2 --trace-dir "$T" --label "SSD 370M" $G30 --tag ${H}_e30s2_ssd370m &&
      flock "$LOCK" $E30 --model mamba2 --trace-dir "$T/tpds_traces_m2_2p7b" --label "SSD 2.7B" $G30 --tag ${H}_e30s2_ssd2p7b &&
      flock "$LOCK" $E30 --model gdn --trace-dir "$T/tpds_traces_gdn_1p4b" --label "GDN 1.4B" $G30 --tag ${H}_e30s2_gdn1p4b ;;
    *) echo "unknown step $2"; exit 2 ;;
  esac
  exit $?
fi

SELF="src/experiments/remote/taco_lane_4c_r2.sh"
LANE=${TACO_LANE:?set TACO_LANE=G0 or G1}
check_space "$OUT_DIR" 5 || exit 1
check_space "$HOME" 5 || exit 1   # Triton compile cache
lanelog "taco-r2 4c lane $LANE (CUDA_VISIBLE_DEVICES=$GPU; traces $T)"
nvidia-smi -L || { lanelog "no GPU visible"; exit 1; }
case "$LANE" in
  G0)
    $RUN E32-smoke -- bash "$SELF" step e32_smoke || { lanelog "E32 smoke failed"; exit 1; }
    $RUN E32a -- bash "$SELF" step e32a
    $RUN E32b -- bash "$SELF" step e32b
    # the 2.7B checkpoint is downloaded separately; wait for it (at most TACO_WAIT_DL_H hours)
    waited=0
    until grep -q '^DONE' "$LOG_DIR/dl_mamba2_2p7b.log" 2>/dev/null; do
      [ "$waited" -ge $(( ${TACO_WAIT_DL_H:-6} * 3600 )) ] && { lanelog "2.7B checkpoint not downloaded"; break; }
      sleep 60; waited=$((waited + 60))
    done
    $RUN E32-smoke27 -- bash "$SELF" step e32_smoke27 && $RUN E32c -- bash "$SELF" step e32c
    # session 2: a separate process after the other lane's E33 work (a later time, a different GPU state)
    wait_marker "${TACO_S2_AFTER:-E33b}" || lanelog "continuing with session 2 without ${TACO_S2_AFTER:-E33b}"
    $RUN E32d -- bash "$SELF" step e32d
    $RUN E32e -- bash "$SELF" step e32e
    [ -f "$MARK_DIR/E32c.OK" ] && $RUN E32f -- bash "$SELF" step e32f
    bash "$SELF" step e32_analyze
    ;;
  G1)
    $RUN E33-smoke -- bash "$SELF" step e33_smoke || { lanelog "E33 smoke failed"; exit 1; }
    $RUN E33a -- bash "$SELF" step e33a
    $RUN E33b -- bash "$SELF" step e33b
    $RUN E34-smoke -- bash "$SELF" step e34_smoke && $RUN E34 -- bash "$SELF" step e34
    ;;
  G1b)  # follow-up work on GPU 1 (started separately while G1 runs; waits for E34)
    wait_marker E34 || lanelog "continuing without E34"
    $RUN E35-smoke -- bash "$SELF" step e35_smoke && $RUN E35 -- bash "$SELF" step e35
    ;;
  G0b)  # after the G0 lane (E32 sessions): packed steps under CUDA graphs (E36)
    until [ -f "$MARK_DIR/E32f.OK" ] || [ -f "$MARK_DIR/E32f.FAIL" ] || [ -f "$MARK_DIR/E32c.FAIL" ]; do sleep 120; done
    $RUN E36-smoke -- bash "$SELF" step e36_smoke && $RUN E36 -- bash "$SELF" step e36
    ;;
  G1c)  # after E35 on GPU 1: graph-timed calibration keys (E37), then the second E30 session (E30s2)
    wait_marker E35 || lanelog "continuing without E35"
    $RUN E37 -- bash "$SELF" step e37
    $RUN E33c-capture -- bash "$SELF" step m1capture && $RUN E33c -- bash "$SELF" step e33c
    $RUN E38 -- bash "$SELF" step e38
    $RUN E30s2 -- bash "$SELF" step e30s2
    ;;
  G1d)  # after G1c: resume E37 (resumable; chunks that do not fit in memory are skipped and recorded)
    wait_marker E30s2 || lanelog "continuing without E30s2"
    rm -f "$MARK_DIR/E37.FAIL"
    $RUN E37 -- bash "$SELF" step e37
    ;;
  *) echo "unknown lane $LANE"; exit 2 ;;
esac
lanelog "taco-r2 4c lane $LANE done"
