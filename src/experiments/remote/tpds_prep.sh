#!/usr/bin/env bash
# TPDS prep (CPU only; safe in no-GPU mode): install the Triton kernel libraries,
# download the two checkpoints, and run static checks. Idempotent.
#
#   wsl bash src/revloop/remote.sh exec pro6000 src/experiments/remote/tpds_prep.sh
#   (or on the host: bash src/experiments/remote/tpds_prep.sh)
#
# Env: PY (python), PIP_EXTRA (e.g. "--user" or "--break-system-packages"),
#      HF_ENDPOINT (optional Hugging Face mirror), TPDS_SKIP_INSTALL=1, TPDS_SKIP_DOWNLOAD=1.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh" 2>/dev/null || {
  # When run through remote.sh exec (script piped on stdin) BASH_SOURCE is not the file; use cwd.
  source src/experiments/remote/tpds_common.sh
}
rc=0
lanelog "tpds prep on $(hostname) with $PY"

echo "== disk"; df -h . "$TRACE_DIR" 2>/dev/null || df -h .; echo "== inodes"; df -i . 2>/dev/null | tail -1
check_space . 80 || { lanelog "WARNING: less than 80 GB free for traces + results"; rc=1; }

echo "== python/torch"
"$PY" - <<'PYEOF' || rc=1
import sys, torch, triton
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)
import typing_extensions as te
print("typing_extensions", getattr(te, "__version__", "?"))
PYEOF

if [ "${TPDS_SKIP_INSTALL:-0}" != 1 ]; then
  echo "== install (no dependency resolution, so the host's torch/triton are never replaced)"
  # shellcheck disable=SC2086
  "$PY" -m pip install ${PIP_EXTRA:-} --no-deps "flash-linear-attention==0.5.2" "fla-core==0.5.2" "einops>=0.8" || rc=1
  # torch._inductor (imported by fla) needs typing_extensions >= 4.12.
  # shellcheck disable=SC2086
  "$PY" -c "import torch._inductor.compile_fx" 2>/dev/null || "$PY" -m pip install ${PIP_EXTRA:-} -U "typing_extensions>=4.12" || rc=1
  # mamba-ssm: only the Triton SSD ops are used; the CUDA extensions are not built.
  # shellcheck disable=SC2086
  MAMBA_SKIP_CUDA_BUILD=TRUE CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE MAMBA_FORCE_BUILD=FALSE \
    "$PY" -m pip install ${PIP_EXTRA:-} --no-deps --no-build-isolation "mamba-ssm==2.3.2.post1" || rc=1
fi

echo "== import checks (CPU)"
"$PY" - <<'PYEOF' || rc=1
from src.experiments.tpds import kernels as K
print("mamba_ssm SSD (Triton) importable:", K.ssd_available())
print("fla chunk_gated_delta_rule importable:", K.gdn_available())
import sklearn, pandas, matplotlib
print("sklearn", sklearn.__version__, "pandas", pandas.__version__, "matplotlib", matplotlib.__version__)
PYEOF

if [ "${TPDS_SKIP_DOWNLOAD:-0}" != 1 ]; then
  echo "== checkpoints"
  "$PY" - <<'PYEOF' || rc=1
from huggingface_hub import snapshot_download
for rid in ("state-spaces/mamba-370m-hf", "AntonV/mamba2-370m-hf"):
    print(rid, "->", snapshot_download(rid))
PYEOF
fi

echo "== static checks"
for f in src/experiments/tpds/*.py; do "$PY" -m py_compile "$f" || { echo "py_compile FAILED: $f"; rc=1; }; done
for f in src/experiments/remote/tpds_*.sh src/revloop/remote_run.sh; do bash -n "$f" || { echo "bash -n FAILED: $f"; rc=1; }; done
"$PY" -c "from src.experiments.tpds.corpus import build_corpus; c=build_corpus(20, 8, (1024,2048,4096,8192)); print('corpus prompts:', len(c))" || rc=1

lanelog "tpds prep finished rc=$rc"
exit $rc
