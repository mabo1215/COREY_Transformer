#!/usr/bin/env bash
# TPDS cycle tpds-r3 E13 prep on a fresh data-centre host (CPU only; safe in no-GPU mode).
#   1. Build a venv at <repo>/.venv (on the data disk) with the software versions of the other
#      GPUs: torch 2.8.0 (cu128 wheel), Triton 3.4.0, transformers 5.6.2, numpy 2.3.2,
#      scikit-learn 1.8.0, fla 0.5.2 and mamba-ssm 2.3.2.post1 (Triton ops only, no CUDA build).
#   2. Write the untracked host-local settings file .tpds_env (PY, REVLOOP_HOST, HF_ENDPOINT).
#   3. Run the common prep (tpds_prep.sh: import checks, checkpoint download, static checks).
#   4. Build the runtime-chunk reference scan (module selective_scan_chunk_cuda) for the native
#      architecture with tpds_r2_prep.sh (nvcc compiles without a GPU); E12 uses it when present.
#      AutoDL no-GPU mode caps memory at 2 GB, which kills the compiler: the build then fails
#      here and is repeated by tpds_lane_r3_e13.sh at the start of the GPU session.
#   5. E13-specific static checks: bash -n of the r3 lane, the corpus of the capture step.
# Idempotent: an existing venv with the right torch is kept.
#
#   wsl bash src/revloop/remote.sh exec h800 src/experiments/remote/tpds_r3_e13_prep.sh
#
# Env: TPDS_BASE_PY (python used to create the venv; default /root/miniconda3/bin/python),
#      TPDS_HOST (registry name written to .tpds_env; default $REVLOOP_HOST),
#      TPDS_HF_ENDPOINT (Hugging Face mirror written to .tpds_env, e.g. https://hf-mirror.com),
#      TORCH_CUDA_ARCH_LIST (default 9.0), TPDS_SKIP_REF=1 (skip step 4).
set -uo pipefail
ROOT=$(pwd)
[ -f src/experiments/remote/tpds_common.sh ] || { echo "run from the repository root"; exit 2; }
rc=0
BASE_PY=${TPDS_BASE_PY:-/root/miniconda3/bin/python}
VENV=$ROOT/.venv
HOST=${TPDS_HOST:-${REVLOOP_HOST:-$(hostname)}}

echo "== venv ($VENV from $BASE_PY)"
[ -x "$VENV/bin/python" ] || "$BASE_PY" -m venv "$VENV" || { echo "venv creation failed"; exit 1; }
VPY=$VENV/bin/python
if ! "$VPY" -c 'import torch, sys; sys.exit(torch.__version__ != "2.8.0+cu128")' 2>/dev/null; then
  "$VPY" -m pip install -q -U pip wheel setuptools || rc=1
  "$VPY" -m pip install -q "torch==2.8.0" "triton==3.4.0" || rc=1
fi
"$VPY" -m pip install -q "transformers==5.6.2" "numpy==2.3.2" "scikit-learn==1.8.0" scipy pandas matplotlib \
  pynvml huggingface_hub "einops>=0.8" "typing_extensions>=4.12" ninja packaging || rc=1

echo "== .tpds_env"
{
  echo "PY=$VPY"
  echo "REVLOOP_HOST=$HOST"
  [ -n "${TPDS_HF_ENDPOINT:-}" ] && echo "export HF_ENDPOINT=${TPDS_HF_ENDPOINT}"
  echo "export HF_HOME=/root/autodl-tmp/hf_home"
} > .tpds_env
cat .tpds_env
export PY=$VPY
# shellcheck disable=SC1091
. ./.tpds_env

echo "== common prep"
bash src/experiments/remote/tpds_prep.sh || rc=1

if [ "${TPDS_SKIP_REF:-0}" != 1 ]; then
  echo "== runtime-chunk reference scan (native architecture)"
  export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0} MAX_JOBS=${MAX_JOBS:-16} \
    bash src/experiments/remote/tpds_r2_prep.sh || echo "(tpds_r2_prep rc!=0: traces are captured in the GPU session, so a missing-trace failure is expected)"
  "$VPY" -c "import torch, selective_scan_chunk_cuda" || { echo "selective_scan_chunk_cuda NOT importable"; rc=1; }
fi

echo "== E13 static checks"
bash -n src/experiments/remote/tpds_lane_r3.sh || rc=1
for f in src/experiments/tpds/*.py; do "$VPY" -m py_compile "$f" || { echo "py_compile FAILED: $f"; rc=1; }; done
"$VPY" -c "from src.experiments.tpds.corpus import build_corpus; c=build_corpus(20, 8, (1024,2048,4096,8192)); print('E13 capture corpus prompts:', len(c))" || rc=1
"$VPY" - <<'PYEOF' || rc=1
import importlib.metadata as md, sys, torch, triton
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)
for p in ("transformers", "numpy", "scipy", "scikit-learn", "pandas", "pynvml", "huggingface_hub",
          "mamba_ssm", "fla-core", "flash-linear-attention"):
    try:
        print(f"  {p} {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"  {p} MISSING")
print("cuda available (expected False in no-GPU mode):", torch.cuda.is_available())
from huggingface_hub import snapshot_download
print("mamba-370m-hf cached at", snapshot_download("state-spaces/mamba-370m-hf", local_files_only=True))
PYEOF
df -h /root/autodl-tmp | tail -1
nvidia-smi -L 2>/dev/null || echo "no GPU visible (no-GPU mode)"
echo "E13 prep rc=$rc"
exit $rc
