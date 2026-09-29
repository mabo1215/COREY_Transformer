#!/usr/bin/env bash
# TPDS cycle tpds-r2 prep (CPU only; safe in no-GPU mode, also fine with the GPU on).
#   1. P1: build the runtime-chunk reference selective scan (module selective_scan_chunk_cuda)
#      from a mamba_ssm source tree matching the installed wheel. nvcc compiles without a GPU.
#   2. Check that the captured traces, the checkpoint cache and the new scripts are in place.
# Idempotent: an existing build of the same source is kept (TPDS_R2_REBUILD=1 forces a rebuild).
#
#   wsl bash src/revloop/remote.sh exec vgpu3090 src/experiments/remote/tpds_r2_prep.sh
#
# Env: PY, TPDS_MAMBA_SRC (default /root/autodl-tmp/mamba_src), TPDS_MAMBA_REF (git tag of the
#      source; default: the installed mamba_ssm version prefixed with v), TORCH_CUDA_ARCH_LIST
#      (default: 8.6), TPDS_R2_REBUILD=1, TPDS_R2_E11=1 (also download the E11 checkpoints).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh" 2>/dev/null || source src/experiments/remote/tpds_common.sh
rc=0
lanelog "tpds-r2 prep on $(hostname) with $PY"
check_space . 10 || rc=1

echo "== toolchain"
"$PY" - <<'PYEOF' || rc=1
import torch, importlib.metadata as md
from torch.utils.cpp_extension import CUDA_HOME
print("torch", torch.__version__, "cuda", torch.version.cuda, "CUDA_HOME", CUDA_HOME)
print("mamba_ssm", md.version("mamba_ssm"))
try:
    import selective_scan_cuda
    print("stock selective_scan_cuda:", selective_scan_cuda.__file__, "has fwd_with_chunk_size:",
          hasattr(selective_scan_cuda, "fwd_with_chunk_size"))
except Exception as exc:  # noqa: BLE001
    print("stock selective_scan_cuda import failed:", exc)
PYEOF
NVCC=$("$PY" -c 'from torch.utils.cpp_extension import CUDA_HOME; print((CUDA_HOME or "") + "/bin/nvcc")')
if [ -x "$NVCC" ]; then "$NVCC" --version | tail -2; else echo "nvcc not found at $NVCC"; rc=1; fi

echo "== P1: runtime-chunk build"
MAMBA_SRC=${TPDS_MAMBA_SRC:-/root/autodl-tmp/mamba_src}
REF=${TPDS_MAMBA_REF:-v$("$PY" -c 'import importlib.metadata as m; print(m.version("mamba_ssm"))')}
if ! "$PY" -c "import selective_scan_chunk_cuda" 2>/dev/null || [ "${TPDS_R2_REBUILD:-0}" = 1 ]; then
  if [ ! -f "$MAMBA_SRC/csrc/selective_scan/selective_scan_fwd_kernel.cuh" ]; then
    ( [ -f /etc/network_turbo ] && source /etc/network_turbo
      git clone --depth 1 --branch "$REF" https://github.com/state-spaces/mamba.git "$MAMBA_SRC" ) || {
        lanelog "clone of mamba $REF failed"; rc=1; }
  fi
  if [ -f "$MAMBA_SRC/csrc/selective_scan/selective_scan_fwd_kernel.cuh" ]; then
    git -C "$MAMBA_SRC" log --oneline -1 2>/dev/null
    ( cd src/kernels/mamba1_runtime_chunk &&
      MAMBA_CSRC="$MAMBA_SRC/csrc/selective_scan" TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6} MAX_JOBS=${MAX_JOBS:-4} \
        "$PY" -m pip install --no-build-isolation --no-deps -v . > "$ROOT/$LOG_DIR/p1_build.log" 2>&1 ) \
      && lanelog "P1 build ok" || { lanelog "P1 build FAILED (see $LOG_DIR/p1_build.log)"; tail -40 "$LOG_DIR/p1_build.log"; rc=1; }
    grep -E "registers|spill" "$LOG_DIR/p1_build.log" | head -20
  fi
else
  echo "selective_scan_chunk_cuda already importable (TPDS_R2_REBUILD=1 to rebuild)"
fi
"$PY" -c "import selective_scan_chunk_cuda as m; print('selective_scan_chunk_cuda:', m.__file__, [n for n in dir(m) if not n.startswith('_')])" || rc=1

echo "== data and caches"
if [ -d "$TRACE_DIR" ]; then
  echo "traces: $(find "$TRACE_DIR" -type f | wc -l) files, $(du -sh "$TRACE_DIR" | cut -f1)"
else
  echo "traces missing at $TRACE_DIR (re-capture with capture_traces.py, ~20 min on the GPU)"; rc=1
fi
"$PY" - <<'PYEOF' || true
from huggingface_hub import scan_cache_dir
try:
    repos = sorted(r.repo_id for r in scan_cache_dir().repos)
    print("HF cache:", [r for r in repos if "mamba" in r.lower()])
except Exception as exc:  # noqa: BLE001
    print("HF cache scan failed:", exc)
PYEOF

if [ "${TPDS_R2_E11:-0}" = 1 ]; then
  echo "== E11 checkpoints (download only)"
  "$PY" - <<'PYEOF' || rc=1
from huggingface_hub import snapshot_download
for mid in ("state-spaces/mamba-1.4b-hf", "state-spaces/mamba-2.8b-hf"):
    print(mid, snapshot_download(mid, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]))
PYEOF
fi

echo "== static checks"
for f in src/experiments/tpds/e8_refscan.py src/experiments/tpds/e9_placement.py \
         src/experiments/tpds/e10_routed_interleaved.py src/experiments/tpds/kernel_stats.py; do
  "$PY" -m py_compile "$f" || { echo "py_compile failed: $f"; rc=1; }
done
bash -n src/experiments/remote/tpds_lane_r2.sh || rc=1
"$PY" -m src.experiments.tpds.e10_routed_interleaved --dry-run --tag "$TAG" 2>&1 | tail -5 || rc=1
nvidia-smi -L 2>/dev/null || echo "no GPU visible (no-GPU mode)"
lanelog "tpds-r2 prep done rc=$rc"
exit $rc
