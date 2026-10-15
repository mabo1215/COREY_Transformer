#!/usr/bin/env bash
# Cycle taco-r1 prep on the vGPU 3090 (CPU only; safe in no-GPU mode).
#   1. Disk (bytes and inodes) and the captured traces of the RTX 3090 runs on the shared file system.
#   2. E29 (review M2b): compile facts of the runtime-chunk reference CUDA scan that was timed, read from its
#      binary (cuobjdump) plus the dynamic shared memory of every instantiation (host-only nvcc program)
#      -> refscan_compile_facts_sm86.json.
#   3. E30 (review M6c) and E31 (review M8b): static checks, the stratified call plan (dry run) and the
#      checkpoint download for E31 (no GPU).
#
#   wsl bash src/revloop/remote.sh exec vgpu3090 src/experiments/remote/taco_vgpu_prep.sh
# Env: TACO_TRACE_ROOT (shared traces), TPDS_MAMBA_SRC (mamba_ssm source tree of the extension build),
#      TACO_PREP_ONLY (default "E29 E30 E31").
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]:-}")/tpds_common.sh" 2>/dev/null || source src/experiments/remote/tpds_common.sh
rc=0
ONLY=${TACO_PREP_ONLY:-E29 E30 E31}
want() { case " $ONLY " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
ROOT_TR=${TACO_TRACE_ROOT:-/autodl-fs/data/corey_r4_traces_vgpu3090}
lanelog "taco-r1 vGPU prep on $(hostname) with $PY (steps: $ONLY)"
check_space "$OUT_DIR" 5 || rc=1
nvidia-smi -L 2>/dev/null || echo "no GPU visible (expected in no-GPU mode)"

echo "== traces under $ROOT_TR"
for spec in "mamba2:$ROOT_TR" "mamba2:$ROOT_TR/tpds_traces_m2_2p7b" "gdn:$ROOT_TR/tpds_traces_gdn_1p4b"; do
  kind=${spec%%:*}; dir=${spec#*:}
  m="$dir/$kind/manifest.jsonl"
  if [ -f "$m" ]; then echo "$kind @ $dir: $(grep -c prompt_id "$m") manifest rows"; else echo "MISSING: $m"; rc=1; fi
done

if want E29; then
  echo "== E29: reference-scan compile facts (sm_86)"
  MAMBA_SRC=${TPDS_MAMBA_SRC:-/root/autodl-tmp/mamba_src}
  "$PY" -m src.experiments.tpds.refscan_facts --mamba-csrc "$MAMBA_SRC/csrc/selective_scan" --out-dir "$OUT_DIR" \
    || { echo "E29 facts FAILED"; rc=1; }
fi

if want E30; then
  echo "== E30: static checks and stratified plans"
  if [ -f src/experiments/tpds/e30_autotune_state.py ]; then
    "$PY" -m py_compile src/experiments/tpds/e30_autotune_state.py || rc=1
    "$PY" -m src.experiments.tpds.e30_autotune_state --model mamba2 --trace-dir "$ROOT_TR" --dry-run || rc=1
    "$PY" -m src.experiments.tpds.e30_autotune_state --model mamba2 --trace-dir "$ROOT_TR/tpds_traces_m2_2p7b" --dry-run || rc=1
    "$PY" -m src.experiments.tpds.e30_autotune_state --model gdn --trace-dir "$ROOT_TR/tpds_traces_gdn_1p4b" --dry-run || rc=1
  else
    echo "e30_autotune_state.py missing"; rc=1
  fi
fi

if want E31; then
  echo "== E31: static checks, checkpoint download and prompt plan"
  if [ -f src/experiments/tpds/e31_divergence.py ]; then
    "$PY" -m py_compile src/experiments/tpds/e31_divergence.py || rc=1
    for m in ${TACO_E31_MODELS:-mamba2 mamba1}; do
      "$PY" -m src.experiments.tpds.e31_divergence --model "$m" --download-only || rc=1
      "$PY" -m src.experiments.tpds.e31_divergence --model "$m" --dry-run || rc=1
    done
  else
    echo "e31_divergence.py missing"; rc=1
  fi
fi
lanelog "taco-r1 vGPU prep done (rc=$rc)"
exit $rc
