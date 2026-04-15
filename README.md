# Corey_Transformer

Entropy-Regularized Fusion Optimization for Efficient Mamba-based Transformer Inference

---

## Installation

### Requirements

- Linux or WSL2 (Ubuntu 22.04 recommended)
- NVIDIA GPU with CUDA 12.8 driver support
- Python 3.11
- NVIDIA Container Toolkit (for Docker path)

---

### Option A — Docker (recommended)

```bash
# Build the image (compiles mamba_ssm and causal_conv1d CUDA extensions)
docker build -t corey-transformer .

# Run with GPU access, mounting the repo into /workspace
docker run --gpus all --rm -it \
    -v $(pwd):/workspace \
    -e HF_HOME=/workspace/.cache/huggingface \
    corey-transformer bash
```

Inside the container, all `src/` experiments are ready to run:

```bash
# Example: W1 three-policy GPU benchmark
python -m src.experiments.run_w1_triton_triplet \
    --seq-len 4096 --dim 1024 --d-state 16 \
    --warmup-runs 5 --benchmark-repeats 30 \
    --output-dir src/outputs/w1_triton_triplet
```

---

### Option B — Local (micromamba / conda)

1. **Create and activate the environment**

   ```bash
   # Using micromamba (recommended)
   micromamba create -n corey-cuda128 -c nvidia -c conda-forge \
       python=3.11 pip setuptools wheel ninja cuda-nvcc=12.8

   micromamba activate corey-cuda128
   ```

2. **Install PyTorch with CUDA 12.8**

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```

3. **Install Python dependencies**

   ```bash
   pip install numpy triton einops transformers datasets \
               sentencepiece accelerate psutil huggingface_hub
   ```

4. **Build and install CUDA extensions**

   ```bash
   # causal-conv1d
   pip install causal-conv1d

   # mamba-ssm (compiled for your GPU architecture)
   export TORCH_CUDA_ARCH_LIST="8.6"   # adjust for your GPU (8.0=A100, 8.9=4090, 9.0=H100)
   export MAX_JOBS=4
   git clone --depth 1 --branch v2.3.1 https://github.com/state-spaces/mamba.git /tmp/mamba_src
   pip install /tmp/mamba_src
   ```

5. **Install this package in editable mode**

   ```bash
   # From the repository root
   pip install -e .
   ```

6. **Verify the installation**

   ```bash
   python -c "from src.algorithms import ExponentialMovingEntropy, select_fusion_groups; print('OK')"
   ```

---

### Running experiments

All experiment entry points live under `src/experiments/`. Run them as Python
modules from the repository root:

| Script | Purpose |
|--------|---------|
| `src/experiments/run_entropy_guided_experiments.py` | Full entropy-guided fusion scheduling sweep |
| `src/experiments/run_w1_triton_triplet.py` | W1 three-policy GPU benchmark (off / static / COREY) |
| `src/experiments/run_triton_selective_scan_benchmark.py` | Triton selective-scan micro-benchmark |
| `src/experiments/run_cuda_profile_three_policies.py` | CUDA kernel launch profile, three policies |
| `src/run_all.py` | Run the full entropy-guided experiment suite |

Example:

```bash
# Full experiment suite
python src/run_all.py

# W1 triplet benchmark with custom parameters
python -m src.experiments.run_w1_triton_triplet \
    --seq-len 4096 --dim 1024 --d-state 16 \
    --warmup-runs 5 --benchmark-repeats 30 \
    --output-dir src/outputs/w1_triton_triplet
```

Outputs (JSON summaries, CSV tables) are written to `src/outputs/`.