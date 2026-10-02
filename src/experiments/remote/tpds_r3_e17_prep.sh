#!/usr/bin/env bash
# tpds-r3 E17 (M2b) prep on the RTX PRO 6000: CPU only, safe in no-GPU mode and while other projects
# use the GPU.
#   1. Python environment: torch/triton, mamba_ssm Triton SSD ops, fla (Gated DeltaNet), transformers.
#   2. Disk: ~60 GB for traces plus the checkpoints.
#   3. Download the checkpoints to $TPDS_E17_MODEL_ROOT (huggingface_hub, HF_ENDPOINT from .tpds_env):
#      AntonV/mamba2-2.7b-hf and Idiap/gated-deltanet-swa-1.4B-30B (TPDS_E17_DOWNLOAD=0 skips).
#   4. Static checks: bash -n on the lane, py_compile, --help of the new scripts.
#
#   wsl bash src/revloop/remote.sh exec pro6000 src/experiments/remote/tpds_r3_e17_prep.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]:-}")/tpds_common.sh" 2>/dev/null || source src/experiments/remote/tpds_common.sh
rc=0
E17_ROOT=${TPDS_E17_MODEL_ROOT:-/root/autodl-tmp/hf_models}
lanelog "tpds-r3 E17 prep on $(hostname) with $PY"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "no GPU visible"
check_space "${TPDS_E17_TRACE_ROOT:-src/outputs}" 60 || rc=1
check_space "$E17_ROOT" 15 || rc=1

echo "== python environment"
"$PY" - <<'PYEOF' || rc=1
import importlib.metadata as md
import torch, triton
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)
for p in ("flash-linear-attention", "fla-core", "mamba_ssm", "transformers", "huggingface_hub", "scipy", "pandas"):
    try: print(p, md.version(p))
    except Exception: print(p, "MISSING")
from src.experiments.tpds import kernels as K
print("ssd ops importable:", K.ssd_available(), "| gdn ops importable:", K.gdn_available())
from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM  # noqa: F401
print("fla Gated DeltaNet model classes importable")
PYEOF

if [ "${TPDS_E17_DOWNLOAD:-1}" = 1 ]; then
  echo "== checkpoints -> $E17_ROOT"
  mkdir -p "$E17_ROOT"
  "$PY" - "$E17_ROOT" <<'PYEOF' || rc=1
import os, sys, time
from huggingface_hub import snapshot_download
root = sys.argv[1]
for mid, pats in (("AntonV/mamba2-2.7b-hf", ["*.json", "*.safetensors", "*.txt", "*.model"]),
                  ("Idiap/gated-deltanet-swa-1.4B-30B", ["*.json", "*.safetensors", "*.model", "*.txt"])):
    dest = os.path.join(root, mid.split("/")[-1])
    t0 = time.time()
    p = snapshot_download(mid, local_dir=dest, allow_patterns=pats, max_workers=8)
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs) / 2**30
    print(f"{mid}: {p} ({size:.1f} GB, {time.time() - t0:.0f} s)")
PYEOF
fi

echo "== static checks"
bash -n src/experiments/remote/tpds_lane_r3_e17.sh || rc=1
"$PY" -m py_compile src/experiments/tpds/capture_gdn_traces.py src/experiments/tpds/e17_workbound.py \
  src/experiments/tpds/e2_oracle_sweep.py src/experiments/tpds/kernels.py || rc=1
"$PY" -m src.experiments.tpds.e17_workbound --help >/dev/null || rc=1
"$PY" -m src.experiments.tpds.capture_gdn_traces --help >/dev/null || rc=1
"$PY" -m src.experiments.tpds.e2_oracle_sweep --help >/dev/null || rc=1
lanelog "tpds-r3 E17 prep done (rc=$rc)"
exit $rc
