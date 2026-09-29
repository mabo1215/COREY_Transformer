#!/usr/bin/env bash
# TPDS cycle tpds-r2 GPU lane (RTX 3090): E8 -> E9 -> E10, one after the other. All three time
# kernels, so they run in one lane under the exclusive timing lock and with the exclusive-GPU
# guard (a job pauses while another process holds the GPU and re-times what it overlapped).
#   E8   reference CUDA scan at every runtime chunk on the 2360 captured Mamba-1 calls and the
#        twelve E4 keys (validates every chunk against the stock forward first)
#   E9   value vs placement control (designs A/B/C), with the reference scan when available
#   E10  single-session interleaved routed end-to-end rerun
#   E11  (optional, only when listed in TPDS_R2_ONLY) larger checkpoints: batch-1 traces of 8
#        layers of mamba-1.4b-hf and mamba-2.8b-hf, then the E2 chunk sweep of the Triton kernel
# Prerequisite: tpds_r2_prep.sh (P1 build of selective_scan_chunk_cuda, traces present).
#
# Launch:  wsl bash src/revloop/remote.sh launch vgpu3090 tpds-r2 src/experiments/remote/tpds_lane_r2.sh
# Markers: src/logs/revloop/{E8,E9,E10}.{START,OK,FAIL}. These are the tpds-r2 ledger IDs, so the
#          host's TPDS_ID_PREFIX (used for the replicated tpds-r1 experiments) is not applied.
# Resumable: re-launching skips completed rows/keys. TPDS_SMOKE=1 runs a tiny version.
# TPDS_R2_ONLY="E9 E10" restricts the lane to the listed experiments.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh"
L=(); [ "${TPDS_NO_GPU_LOCK:-0}" = 1 ] || L=(flock "$LOCK")
ONLY=${TPDS_R2_ONLY:-E8 E9 E10}
P=""; [ "${TPDS_SMOKE:-0}" = 1 ] && P=smoke-
check_space "$OUT_DIR" 2 || exit 1
if [ "${TPDS_SMOKE:-0}" = 1 ]; then
  E8_ARGS="--batches 1 --max-rows 6 --warmup 2 --repeats 5"
  E9_ARGS="--lengths 512 --rounds 1 --draws 1 --placements 2 --inputs gaussian,uniform --repeats 5 --cost-repeats 20"
  E10_ARGS="--rounds 1 --warmup-rounds 1 --lengths 256 --new-tokens 4"
else
  E8_ARGS="--batches 1,4,16 --e4-keys --warmup 10 --repeats 50"
  E9_ARGS="--lengths 2048,8192 --rounds 5 --draws 3 --placements 10 --repeats 100 --cost-repeats 500"
  E10_ARGS=""
fi
REF_OK=0; "$PY" -c "import torch, selective_scan_chunk_cuda" 2>/dev/null && REF_OK=1
lanelog "tpds-r2 lane start on $TAG (runtime-chunk module: $REF_OK; experiments: $ONLY)"

want() { case " $ONLY " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

if want E8; then
  if [ "$REF_OK" = 1 ]; then
    # shellcheck disable=SC2086
    $RUN ${P}E8 -- bash -c "${L[*]} $PY -m src.experiments.tpds.e8_refscan $E8_ARGS --exclusive-guard \
        --trace-dir '$TRACE_DIR' --out-dir '$OUT_DIR' --tag '$TAG'"
  else
    lanelog "E8 skipped: selective_scan_chunk_cuda not importable (run tpds_r2_prep.sh)"
    : > "$MARK_DIR/${P}E8.FAIL"
  fi
fi
if want E9; then
  REF_FLAG=""; [ "$REF_OK" = 1 ] && REF_FLAG="--ref"
  # shellcheck disable=SC2086
  $RUN ${P}E9 -- bash -c "${L[*]} $PY -m src.experiments.tpds.e9_placement $E9_ARGS $REF_FLAG --exclusive-guard \
      --trace-dir '$TRACE_DIR' --out-dir '$OUT_DIR' --tag '$TAG'"
fi
if want E10; then
  # shellcheck disable=SC2086
  $RUN ${P}E10 -- bash -c "${L[*]} $PY -m src.experiments.tpds.e10_routed_interleaved $E10_ARGS --exclusive-guard \
      --out-dir '$OUT_DIR' --tag '$TAG' && $PY -m src.experiments.tpds.e10_routed_interleaved --analyze \
      --out-dir '$OUT_DIR' --tag '$TAG'"
fi
if want E11; then
  # 8 captured layers per model: stride 6 of 48 (1.4B) and stride 8 of 64 (2.8B)
  # traces scale with the width: ~60 GB (1.4B, d_inner 4096) and ~80 GB (2.8B, 5120) vs 31 GB at 370M
  E11_CAP="--samples-per-regime 20 --longbench-per-task 8 --longbench-lengths 1024,2048,4096,8192 --max-tokens 8192 --max-gb 100"
  E11_E2="--batches 1 --warmup 10 --repeats 50"
  [ "${TPDS_SMOKE:-0}" = 1 ] && { E11_CAP="--samples-per-regime 1 --longbench-per-task 1 --max-tokens 1024 --limit-prompts 6"; E11_E2="--batches 1 --warmup 2 --repeats 3 --max-rows 6"; }
  check_space src/outputs 160 || { lanelog "E11 needs ~160 GB for traces"; : > "$MARK_DIR/${P}E11.FAIL"; }
  if [ ! -f "$MARK_DIR/${P}E11.FAIL" ]; then
    # shellcheck disable=SC2086
    $RUN ${P}E11 -- bash -c "set -e
      for spec in 1p4b:state-spaces/mamba-1.4b-hf:6 2p8b:state-spaces/mamba-2.8b-hf:8; do
        tag=\${spec%%:*}; rest=\${spec#*:}; mid=\${rest%:*}; stride=\${rest##*:}
        local_dir=\${TPDS_E11_MODEL_ROOT:-/root/autodl-tmp/hf_models}/\${mid##*/}
        [ -f \$local_dir/config.json ] && mid=\$local_dir   # pre-downloaded checkpoint (aria2), no network
        $PY -m src.experiments.tpds.capture_traces --model mamba1 --model-id \$mid --layer-stride \$stride $E11_CAP \
            --trace-dir src/outputs/tpds_traces_\$tag
        ${L[*]} $PY -m src.experiments.tpds.e2_oracle_sweep --model mamba1 $E11_E2 \
            --trace-dir src/outputs/tpds_traces_\$tag --out-dir '$OUT_DIR' --tag ${TAG}_\$tag
      done"
  fi
fi
lanelog "tpds-r2 lane done"
