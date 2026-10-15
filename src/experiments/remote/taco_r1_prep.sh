#!/usr/bin/env bash
# Cycle taco-r1 prep on the RTX PRO 6000 (CPU only; safe in no-GPU mode).
#   1. Traces of the RTX 3090 graph runs on the shared file system (370M Mamba-1 and Mamba-2, 2.7B Mamba-2,
#      1.4B Gated DeltaNet): manifests and call counts.
#   2. Packed variable-length steps from the Azure 2023 serving traces (packed_steps.py -> packed_steps_taco.json).
#   3. Import and syntax checks of the new code (no GPU needed).
#   4. Review M2b: compile facts of the runtime-chunk reference CUDA scan for sm_86 (the RTX 3090 on which it
#      was timed): ptxas registers, spills and shared memory of every chunk and thread-count instantiation ->
#      refscan_compile_facts_sm86.json. Set TACO_REF_FACTS=0 to skip (about 10-20 min of nvcc).
#
#   wsl bash src/revloop/remote.sh exec pro6000 src/experiments/remote/taco_r1_prep.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]:-}")/tpds_common.sh" 2>/dev/null || source src/experiments/remote/tpds_common.sh
rc=0
lanelog "taco-r1 prep on $(hostname) with $PY"
check_space . 10 || rc=1
nvidia-smi -L 2>/dev/null || echo "no GPU visible (expected in no-GPU mode)"
ROOT_TR=${TACO_TRACE_ROOT:-/autodl-fs/data/corey_r4_traces_vgpu3090}

echo "== traces under $ROOT_TR"
for spec in "mamba1:$ROOT_TR" "mamba2:$ROOT_TR" "mamba2:$ROOT_TR/tpds_traces_m2_2p7b" "gdn:$ROOT_TR/tpds_traces_gdn_1p4b"; do
  kind=${spec%%:*}; dir=${spec#*:}
  m="$dir/$kind/manifest.jsonl"
  if [ -f "$m" ]; then
    echo "$kind @ $dir: $(grep -c prompt_id "$m") manifest rows, $(find "$dir/$kind" -name 'L*.pt' | wc -l) call files"
  else
    echo "MISSING: $m"; rc=1
  fi
done

echo "== packed steps (Azure 2023 traces)"
( [ -f /etc/network_turbo ] && source /etc/network_turbo >/dev/null 2>&1
  "$PY" -m src.experiments.tpds.packed_steps --out-dir "$OUT_DIR" --tag taco ) || { echo "packed_steps failed"; rc=1; }
ls -la "$OUT_DIR/packed_steps_taco.json" 2>/dev/null

echo "== static checks"
bash -n src/experiments/remote/taco_lane_pro6000.sh || rc=1
"$PY" -m py_compile src/experiments/tpds/e27_varlen.py src/experiments/tpds/packed_steps.py \
    src/experiments/tpds/kernels.py src/experiments/tpds/serving_key_churn.py || rc=1
"$PY" - <<'PYEOF' || rc=1
import inspect
from src.experiments.tpds import kernels as K
fn = K._load_ssd()
sig = inspect.signature(fn)
assert "seq_idx" in sig.parameters and "initial_states" in sig.parameters, sig
print("mamba_chunk_scan_combined:", sig)
g = K._load_gdn()
print("chunk_gated_delta_rule accepts cu_seqlens:", "cu_seqlens" in inspect.signature(g).parameters)
from src.experiments.tpds import e27_varlen  # noqa: F401
print("e27_varlen imports ok")
PYEOF

if [ "${TACO_REF_FACTS:-1}" = 1 ]; then
  echo "== M2b: reference-scan compile facts for sm_86"
  MAMBA_SRC=${TPDS_MAMBA_SRC:-/root/autodl-tmp/mamba_src}
  if [ ! -f "$MAMBA_SRC/csrc/selective_scan/selective_scan_fwd_kernel.cuh" ]; then
    REF=${TPDS_MAMBA_REF:-v2.2.2}
    ( [ -f /etc/network_turbo ] && source /etc/network_turbo >/dev/null 2>&1
      git clone -q --depth 1 --branch "$REF" https://github.com/state-spaces/mamba.git "$MAMBA_SRC" ) || echo "clone failed"
  fi
  BUILD=/root/autodl-tmp/taco_refscan_sm86
  rm -rf "$BUILD"; cp -r src/kernels/mamba1_runtime_chunk "$BUILD"
  ( cd "$BUILD" && MAMBA_CSRC="$MAMBA_SRC/csrc/selective_scan" TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=${MAX_JOBS:-8} \
      "$PY" setup.py build_ext --inplace > "$ROOT/$LOG_DIR/taco_refscan_sm86_build.log" 2>&1 ) \
    && echo "sm_86 build ok" || { echo "sm_86 build FAILED"; tail -30 "$LOG_DIR/taco_refscan_sm86_build.log"; rc=1; }
  "$PY" - "$LOG_DIR/taco_refscan_sm86_build.log" "$OUT_DIR/refscan_compile_facts_sm86.json" <<'PYEOF' || rc=1
import json, re, subprocess, sys
log, out = sys.argv[1], sys.argv[2]
facts, cur = [], None
for line in open(log, errors="replace"):
    m = re.search(r"Compiling entry function '([^']+)' for '(sm_\d+)'", line)
    if m:
        cur = {"mangled": m.group(1), "arch": m.group(2)}
        continue
    m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
    if m and cur is not None:
        cur.update(stack=int(m.group(1)), spill_stores=int(m.group(2)), spill_loads=int(m.group(3)))
        continue
    m = re.search(r"Used (\d+) registers.*?(?:(\d+) bytes smem)?", line)
    if m and cur is not None:
        cur["registers"] = int(m.group(1))
        s = re.search(r"(\d+) bytes smem", line)
        cur["smem"] = int(s.group(1)) if s else 0
        facts.append(cur)
        cur = None
names = subprocess.run(["c++filt"], input="\n".join(f["mangled"] for f in facts), capture_output=True,
                       text=True).stdout.splitlines() if facts else []
for f, n in zip(facts, names):
    f["name"] = n
    t = re.search(r"selective_scan_fwd_kernel<Selective_Scan_fwd_kernel_traits<(\d+), (\d+), (\d+)", n)
    if t:
        f.update(n_threads=int(t.group(1)), n_items=int(t.group(2)), n_rows=int(t.group(3)),
                 chunk=int(t.group(1)) * int(t.group(2)))
scan = [f for f in facts if "chunk" in f]
json.dump({"arch": "sm_86", "n_entries": len(facts), "scan_kernels": scan}, open(out, "w"), indent=1)
print(f"{len(facts)} entries, {len(scan)} scan instantiations ->", out)
for f in sorted(scan, key=lambda f: (f["n_threads"], f["chunk"]))[:40]:
    print(f"threads {f['n_threads']:4d} items {f['n_items']:3d} chunk {f['chunk']:5d}: {f['registers']} regs, "
          f"{f.get('spill_stores', 0)}/{f.get('spill_loads', 0)} B spill, {f.get('smem', 0)} B smem")
PYEOF
fi
lanelog "taco-r1 prep done (rc=$rc)"
exit $rc
