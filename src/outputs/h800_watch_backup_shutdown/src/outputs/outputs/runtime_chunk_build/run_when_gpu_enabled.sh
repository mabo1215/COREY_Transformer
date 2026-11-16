#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=/root/Corey_Transformer
MAMBA_SRC="$REPO_ROOT/Quamba/3rdparty/mamba"
OUT="$REPO_ROOT/src/outputs/runtime_chunk_build_gpu"
mkdir -p "$OUT"
cd "$MAMBA_SRC"
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS="${MAX_JOBS:-2}"
export MAMBA_FORCE_BUILD=TRUE
export MAMBA_LOCAL_VERSION=corey_runtime_chunk_gpu
python -m pip install -e . --no-build-isolation --no-cache-dir 2>&1 | tee "$OUT/pip_install.log"
cd "$REPO_ROOT"
python - <<'PY' | tee "$OUT/symbol_check.txt"
import torch, selective_scan_cuda, mamba_ssm
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
print('mamba_ssm', getattr(mamba_ssm, '__version__', None), getattr(mamba_ssm, '__file__', None))
print('selective_scan_cuda', selective_scan_cuda.__file__)
for name in ['fwd','fwd_with_chunk_size','fwd_chunked','fwd_runtime_chunk','bwd']:
    print(name, hasattr(selective_scan_cuda, name))
PY
COREY_SELECTIVE_SCAN_BACKEND=runtime_cuda python -m src.experiments.run_integrated_multiblock_dispatch \
  --dispatch-module src.corey_selective_scan_dispatch \
  --output-dir "$OUT/multiblock_dispatch_probe" 2>&1 | tee "$OUT/multiblock_dispatch_probe.log"
COREY_SELECTIVE_SCAN_BACKEND=runtime_cuda python -m src.experiments.run_integrated_end_to_end \
  --model mamba-370m \
  --new-tokens 32 \
  --warmup 2 \
  --repeats 10 \
  --num-bins 256 \
  --selective-scan-dispatch-module src.corey_selective_scan_dispatch \
  --output-dir "$OUT/integrated_end_to_end" 2>&1 | tee "$OUT/integrated_end_to_end.log"
