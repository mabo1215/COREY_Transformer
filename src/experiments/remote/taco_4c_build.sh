#!/usr/bin/env bash
# 4c (stock RTX 3090, sm_86) environment build for E31 (review M8b): CPU only, no GPU needed.
#   1. causal-conv1d and mamba-ssm 2.3.2.post1 (the version of the vGPU 3090 runs) with their CUDA extensions,
#      compiled with CUDA 12.4 (major version 12 as the cu128 PyTorch build).
#   2. The runtime-chunk reference selective scan (module selective_scan_chunk_cuda) from the mamba-ssm
#      2.3.2.post1 sources (sdist), as on the vGPU 3090.
#   3. The checkpoints of E31 (Mamba-2 370M, Mamba-1 370M, Gated DeltaNet 1.4B) into HF_HOME.
#   4. Import and fast-path checks.
# Run on 4c in a screen session:  bash src/experiments/remote/taco_4c_build.sh  (log: src/logs/tpds/taco_4c_build.log)
set -uo pipefail
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$R" || exit 1
PY=${PY:-$R/.venv/bin/python}
export CUDA_HOME=${TACO_CUDA_HOME:-/usr/local/cuda-12.4}
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6}
export MAX_JOBS=${MAX_JOBS:-16}
export HF_HOME=${HF_HOME:-/data/hf_home}
export TMPDIR=${TACO_BUILD_TMP:-/tmp/taco_build_$USER}
mkdir -p "$TMPDIR" src/logs/tpds
log() { echo "[$(date '+%F %T')] $*"; }
rc=0
log "build on $(hostname) with $PY; nvcc: $(nvcc --version | tail -1)"
df -h "$R" /data /tmp | tail -3

log "== causal-conv1d"
"$PY" -c "import causal_conv1d" 2>/dev/null && log "causal_conv1d already importable" || {
  CAUSAL_CONV1D_FORCE_BUILD=TRUE "$PY" -m pip install --no-build-isolation --no-deps "causal-conv1d>=1.4,<1.6" || rc=1; }

log "== mamba-ssm 2.3.2.post1"
"$PY" -c "import selective_scan_cuda, importlib.metadata as m; assert m.version('mamba_ssm') == '2.3.2.post1'" 2>/dev/null \
  && log "mamba_ssm 2.3.2.post1 with CUDA extension already installed" || {
  MAMBA_FORCE_BUILD=TRUE "$PY" -m pip install --no-build-isolation --no-deps "mamba-ssm==2.3.2.post1" || rc=1; }

log "== runtime-chunk reference scan"
if ! "$PY" -c "import torch, selective_scan_chunk_cuda" 2>/dev/null; then
  # The sdist cannot be fetched with pip download (its metadata step needs torch); fetch the file itself, or
  # copy it into $TMPDIR/mamba_src beforehand from a machine with PyPI access.
  SRC=$TMPDIR/mamba_src
  mkdir -p "$SRC"
  [ -f "$SRC/mamba_ssm-2.3.2.post1.tar.gz" ] || curl -sL -o "$SRC/mamba_ssm-2.3.2.post1.tar.gz" \
    "https://files.pythonhosted.org/packages/af/48/aa7bec0efc7453fa0512276c0069c854bf684219050294266534bfef34bb/mamba_ssm-2.3.2.post1.tar.gz" || rc=1
  tar xzf "$SRC"/mamba_ssm-2.3.2.post1.tar.gz -C "$SRC" || rc=1
  CSRC=$(ls -d "$SRC"/mamba_ssm-2.3.2.post1/csrc/selective_scan)
  ( cd src/kernels/mamba1_runtime_chunk && MAMBA_CSRC="$CSRC" "$PY" -m pip install --no-build-isolation --no-deps -v . ) \
    > src/logs/tpds/taco_4c_refscan_build.log 2>&1 && log "runtime-chunk build ok" \
    || { log "runtime-chunk build FAILED (src/logs/tpds/taco_4c_refscan_build.log)"; tail -30 src/logs/tpds/taco_4c_refscan_build.log; rc=1; }
else
  log "selective_scan_chunk_cuda already importable"
fi

log "== checkpoints"
for m in mamba2 mamba1 gdn; do
  "$PY" -m src.experiments.tpds.e31_divergence --model $m --download-only --hf-home "$HF_HOME" || { log "download $m failed"; rc=1; }
done

log "== checks"
"$PY" - <<'PYEOF' || rc=1
import importlib.metadata as md
import torch, selective_scan_cuda, causal_conv1d, selective_scan_chunk_cuda  # noqa: F401
from transformers.models.mamba2 import modeling_mamba2 as m2
print("torch", torch.__version__, "mamba_ssm", md.version("mamba_ssm"), "causal_conv1d", md.version("causal_conv1d"))
print("selective_scan_chunk_cuda entry points:", [n for n in dir(selective_scan_chunk_cuda) if not n.startswith("_")])
print("transformers Mamba-2 fast path available:", getattr(m2, "is_fast_path_available", None))
PYEOF
log "build done (rc=$rc)"
exit $rc
