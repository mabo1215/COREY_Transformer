"""Build the runtime-chunk Mamba-1 selective scan forward (module ``selective_scan_chunk_cuda``).

The kernel source is taken unchanged from a mamba_ssm source tree, given by MAMBA_CSRC (the
directory that contains selective_scan_fwd_kernel.cuh). Match its version to the installed
mamba_ssm wheel. Only the forward pass is built (three input types, real weights), which takes a
few minutes instead of the full mamba_ssm build.

    MAMBA_CSRC=/path/to/mamba/csrc/selective_scan TORCH_CUDA_ARCH_LIST=8.6 \
        pip install --no-build-isolation -v .
"""
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = Path(__file__).resolve().parent
MAMBA_CSRC = os.environ.get("MAMBA_CSRC")
if not MAMBA_CSRC or not (Path(MAMBA_CSRC) / "selective_scan_fwd_kernel.cuh").exists():
    raise SystemExit("set MAMBA_CSRC to the mamba_ssm csrc/selective_scan directory")

nvcc_flags = ["-O3", "-std=c++17", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
              "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
              "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
              "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math", "--ptxas-options=-v",
              "-lineinfo", "--threads", "4"]

setup(
    name="selective_scan_chunk_cuda",
    version="0.1.0",
    ext_modules=[CUDAExtension(
        name="selective_scan_chunk_cuda",
        sources=[str(HERE / "csrc" / f) for f in ("selective_scan_chunk.cpp", "selective_scan_fwd_chunk_fp16.cu",
                                                   "selective_scan_fwd_chunk_bf16.cu", "selective_scan_fwd_chunk_fp32.cu",
                                                   "selective_scan_fwd_chunk_threads_fp16.cu")],
        include_dirs=[str(Path(MAMBA_CSRC).resolve()), str(HERE / "csrc")],
        extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": nvcc_flags},
    )],
    cmdclass={"build_ext": BuildExtension},
)
