# Corey Transformer — GPU experiment environment
#
# Base image: NVIDIA CUDA 12.8 with cuDNN, Ubuntu 22.04
# Matches the corey-cuda128 WSL2 environment used for all src/ experiments.
#
# Build:
#   docker build -t corey-transformer .
#
# Run (requires NVIDIA Container Toolkit):
#   docker run --gpus all --rm -it \
#       -v $(pwd):/workspace \
#       -e HF_HOME=/workspace/.cache/huggingface \
#       corey-transformer bash

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# ── System packages ────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        wget \
        curl \
        build-essential \
        ninja-build \
        pkg-config \
        libssl-dev \
        ca-certificates \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 10 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 10 \
 && python -m pip install --upgrade pip setuptools wheel

# ── Build-time environment variables ──────────────────────────────────────────
# SM 8.0 = A100, 8.6 = RTX 3090/4090, 8.9 = RTX 4090 Ada, 9.0 = H100
# Extend TORCH_CUDA_ARCH_LIST as needed for your target GPU.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV CUDAARCHS="80;86;89;90"
ENV MAX_JOBS=4
ENV MAMBA_SRC_REF=v2.3.1

# ── PyTorch + CUDA 12.8 ────────────────────────────────────────────────────────
RUN pip install torch --index-url https://download.pytorch.org/whl/cu128

# ── Core Python dependencies ──────────────────────────────────────────────────
RUN pip install \
        numpy \
        triton \
        einops \
        transformers \
        datasets \
        sentencepiece \
        accelerate \
        psutil \
        huggingface_hub

# ── causal-conv1d (CUDA extension, compile from source) ───────────────────────
RUN pip install causal-conv1d

# ── mamba-ssm (CUDA extension, compile from source) ───────────────────────────
# Clones the exact tag used in the project, then installs with CUDA kernels.
RUN git clone --depth 1 --branch ${MAMBA_SRC_REF} \
        https://github.com/state-spaces/mamba.git /tmp/mamba_src \
 && pip install /tmp/mamba_src \
 && rm -rf /tmp/mamba_src

# ── Workspace ─────────────────────────────────────────────────────────────────
WORKDIR /workspace

# Copy only src/ so the layer is cacheable independently of paper/ and docs/
COPY src/ ./src/

# Install the src package in editable mode so `python -m src.experiments.*`
# resolves correctly without PYTHONPATH manipulation.
RUN pip install -e .

# ── Default command ────────────────────────────────────────────────────────────
CMD ["bash"]
