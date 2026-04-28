# Usage Guide — COREY Experiments

This document explains how to run the four experiment scripts,
how to reproduce them on a local GPU, and how to run all four
in parallel on the remote 4× RTX 3090 server.

---

## Experiment Overview

| # | Script | What it tests | Requires |
|---|--------|---------------|----------|
| 1 | `run_external_baselines.py` | RWKV / FlashAttention-2 / Mamba2 vs. Mamba-1.x on LongBench | 4× RTX 3090 (real); literature values (mock) |
| 2 | `run_quamba_quant_benchmark.py` | Mamba2-2.7B INT4 (Quamba) vs. FP16 on LongBench | 4× RTX 3090 + Quamba CUDA extensions |
| 3 | `run_policy_corey_ablation.py` | policy\_corey vs. policy\_static on Mamba2-2.7B LongBench | 4× RTX 3090 (preferred); local GPU (slow) |
| 4 | `run_fused_kernel_benchmark.py` | Entropy-guided fusion algorithm efficiency sweep | Any machine (CPU-level timing) |

All scripts accept `--data-file` pointing to a LongBench JSONL subset under
`src/data/longbench_subset/`.  Run with `PYTHONPATH=src` from the repo root.

---

## Quick Start — Local Single GPU

```bash
# Activate your environment (adjust env name as needed)
conda activate corey-env
# or: source venv/bin/activate

cd /path/to/COREY_Transformer
export PYTHONPATH=src

# Experiment 4 (runs fully on any machine):
python src/experiments/run_fused_kernel_benchmark.py \
  --num-ops 16 --repeats 1000

# Experiments 1–3 (mock mode, no model loading required):
python src/experiments/run_external_baselines.py \
  --models rwkv flashattention mamba2 \
  --data-file src/data/longbench_subset/narrativeqa/test.jsonl \
  --max-prompts 20 --mode mock

# Experiment 3 (real, single GPU, slow without mamba_ssm extensions):
python src/experiments/run_policy_corey_ablation.py \
  --model-id benchang1110/mamba2-2.7b-hf \
  --policy corey \
  --data-file src/data/longbench_subset/narrativeqa/test.jsonl \
  --device cuda
```

---

## Remote 4× RTX 3090 — Concurrent Execution

The 4-card server (`ubuntu-4card`, 4× RTX 3090 24 GiB, CUDA 12.1) is the
recommended target for all GPU-intensive experiments.  Each experiment is
pinned to one GPU via `CUDA_VISIBLE_DEVICES`; all four run in parallel.

---

## Remote H800 -- nohup launch, status, and WSL result watcher

For the H800 rental node, keep final outputs on the system disk so they are
included when saving the machine image, but keep large caches on the data disk:

```bash
cd /root/Corey_Transformer
export PATH=/root/miniconda3/bin:$PATH

mkdir -p /root/autodl-tmp/cache/huggingface
mkdir -p /root/autodl-tmp/cache/torch
mkdir -p /root/autodl-tmp/cache/triton
mkdir -p src/outputs/neurips_required/logs

export HF_HOME=/root/autodl-tmp/cache/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface
export TORCH_HOME=/root/autodl-tmp/cache/torch
export TRITON_CACHE_DIR=/root/autodl-tmp/cache/triton
export OUTPUT_BASE=src/outputs/neurips_required
```

### H800 background run with nohup

```bash
cd /root/Corey_Transformer
nohup bash -lc '
  export PATH=/root/miniconda3/bin:$PATH
  export HF_HOME=/root/autodl-tmp/cache/huggingface
  export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface
  export TORCH_HOME=/root/autodl-tmp/cache/torch
  export TRITON_CACHE_DIR=/root/autodl-tmp/cache/triton
  export OUTPUT_BASE=src/outputs/neurips_required
  RUN_FA3_MATCHED=1 BASELINE_MODE=auto PYTHON_BIN=/root/miniconda3/bin/python \
    bash src/scripts/run_neurips_required_experiments.sh
' > src/outputs/neurips_required/logs/nohup_main.log 2>&1 &

echo $! > src/outputs/neurips_required/nohup_main.pid
```

### H800 status checks

If the PID file does not exist, or `ps` prints nothing, the background job is
not running. Check the last log lines and completed stages:

```bash
cd /root/Corey_Transformer
tail -n 120 src/outputs/neurips_required/logs/nohup_main.log
find src/outputs/neurips_required -name .stage_done -print
```

General status checks:

```bash
cd /root/Corey_Transformer
ps -ef | grep run_neurips_required | grep -v grep
ps -ef | grep python | grep -v grep
watch -n 5 nvidia-smi
find src/outputs/neurips_required -name .stage_done -print
find src/outputs/neurips_required -maxdepth 3 -type f | sort | tail -50
cat src/outputs/neurips_required/flashattention3_matched/summary.json
```

If the run stopped, rerun the same `nohup` command. Completed stages are
skipped by default because each successful stage writes `.stage_done`; use
`FORCE_RERUN=1` only when intentionally recomputing.

Expected H800 wall time after FA3 has already been built:

```text
FA3 matched benchmark:              < 1 minute
Integrated end-to-end benchmark:     5-30 minutes
Heterogeneous corpus benchmark:      20-90 minutes
External baselines, auto mode:       minutes if mock/fallback; 2-6 hours if Mamba2 real runs

Typical total:                       2-4 hours
Conservative total:                  4-8 hours
```

### Local WSL watcher, every 10 minutes

```bash
sudo apt-get update
sudo apt-get install -y sshpass rsync openssh-client

cd /mnt/c/source/Corey_Transformer
bash src/scripts/watch_h800_results_wsl.sh
```

One-shot sync:

```bash
cd /mnt/c/source/Corey_Transformer
MAX_ITERATIONS=1 bash src/scripts/watch_h800_results_wsl.sh
```

The watcher syncs `src/outputs` and `fa3_h800_run.log` by default. To pull
additional remote paths, override `REMOTE_PATHS`:

```bash
REMOTE_PATHS='src/outputs fa3_h800_run.log /root/autodl-tmp/outputs' \
bash src/scripts/watch_h800_results_wsl.sh
```

### Prerequisites on the server

```bash
# 1. SSH to server
ssh mabo1215@<server-ip>

# 2. Sync latest code (from local machine)
rsync -avz --exclude '.git' \
  /mnt/c/source/COREY_Transformer/ \
  mabo1215@<server-ip>:~/COREY_Transformer/

# 3. Verify GPU availability
nvidia-smi   # should list 4× RTX 3090

# 4. Set up micromamba env (if not already present)
micromamba create -n adama-cuda128 python=3.11 -c conda-forge -y
micromamba run -n adama-cuda128 pip install \
  torch==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121
micromamba run -n adama-cuda128 pip install transformers datasets \
  mamba-ssm causal-conv1d

# 5. (Optional) Install Quamba for INT4 experiment:
# Follow https://github.com/state-spaces/mamba for mamba_ssm build.
# Then: git clone https://github.com/enyac-group/Quamba && cd Quamba
#       pip install -e .  (requires sm_89, CUDA 12.8)
```

### One-command parallel launch

```bash
# On the server (or trigger via SSH from local):
cd ~/COREY_Transformer

# Default: all 4 experiments, 20 samples/task, all 4 LongBench subsets
nohup bash src/scripts/run_all_experiments_4gpu.sh \
  > src/outputs/run_all_4gpu_logs/main.log 2>&1 &

# Monitor live:
tail -f src/outputs/run_all_4gpu_logs/main.log
tail -f src/outputs/run_all_4gpu_logs/exp1_external_baselines.log
tail -f src/outputs/run_all_4gpu_logs/exp3_policy_corey.log
```

### Override defaults via env vars

```bash
MODEL_ID=benchang1110/mamba2-2.7b-hf \
MAX_SAMPLES=20 \
USE_MICROMAMBA=1 \
ENV_NAME=adama-cuda128 \
  bash src/scripts/run_all_experiments_4gpu.sh
```

### GPU assignment (summary)

| GPU | Experiment | Est. wall-time (20 samples, 4 tasks) |
|-----|------------|--------------------------------------|
| 0   | External baselines (mock + Mamba2 real) | ~2 h |
| 1   | Quamba INT4 (falls back to FP16 if Quamba unavailable) | ~2 h |
| 2   | Policy COREY ablation (corey + static, Mamba2-2.7B) | ~4 h |
| 3   | Fused kernel algorithm sweep (CPU-bound) | ~1 min |

Total wall-clock (parallel): approximately **4 hours** for 20 samples per task.

---

## Running Individual Experiments Manually

### Experiment 1 — External Baselines

```bash
# Mock mode (no model loading, uses literature reference values):
PYTHONPATH=src python src/experiments/run_external_baselines.py \
  --models rwkv flashattention mamba2 \
  --data-file src/data/longbench_subset/narrativeqa/test.jsonl \
  --max-prompts 20 --mode mock \
  --output-dir src/outputs/external_baselines_narrativeqa

# Real mode (on GPU server with real models installed):
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python src/experiments/run_external_baselines.py \
  --models mamba2 \
  --model-id benchang1110/mamba2-2.7b-hf \
  --data-file src/data/longbench_subset/narrativeqa/test.jsonl \
  --max-prompts 20 --mode real --device cuda

# Background + log:
nohup bash -c '
  for subset in narrativeqa qasper gov_report multifieldqa_en; do
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python \
      src/experiments/run_external_baselines.py \
      --models rwkv flashattention mamba2 \
      --data-file src/data/longbench_subset/${subset}/test.jsonl \
      --max-prompts 20 --mode auto \
      --output-dir src/outputs/external_baselines_${subset}
  done
' > exp1.log 2>&1 &
tail -f exp1.log
```

### Experiment 2 — Quamba INT4 Quantization

```bash
# Single task (falls back to FP16 if Quamba not installed):
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
python src/experiments/run_quamba_quant_benchmark.py \
  --model-id benchang1110/mamba2-2.7b-hf \
  --quant-backend awq --bits 4 --group-size 128 \
  --data-file src/data/longbench_subset/narrativeqa/test.jsonl \
  --max-prompts 20 --device cuda

# All four tasks in background:
nohup bash -c '
  for subset in narrativeqa qasper gov_report multifieldqa_en; do
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python \
      src/experiments/run_quamba_quant_benchmark.py \
      --model-id benchang1110/mamba2-2.7b-hf \
      --quant-backend awq --bits 4 --group-size 128 \
      --data-file src/data/longbench_subset/${subset}/test.jsonl \
      --max-prompts 20 --device cuda \
      --output-dir src/outputs/quamba_${subset}
  done
' > exp2.log 2>&1 &
tail -f exp2.log
```

**Note on Quamba:** The INT4 quantization path requires:
- `quamba==2.0.0a1` compiled from source against CUDA 12.8 on sm_89 hardware.
- `mamba_ssm==2.2.2`, `causal_conv1d==1.6.1`, `fast_hadamard_transform==1.0.4.post1`.
- Without these, the script falls back to FP16 inference and documents the limitation.
- AutoAWQ 0.2.9 and auto-gptq 0.7.1 do **not** support Mamba checkpoints.

### Experiment 3 — Policy COREY Ablation

```bash
# Single policy, single task:
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src \
python src/experiments/run_policy_corey_ablation.py \
  --model-id benchang1110/mamba2-2.7b-hf \
  --policy corey \
  --data-file src/data/longbench_subset/narrativeqa/test.jsonl \
  --device cuda

# Full ablation (both policies, all four tasks):
nohup bash -c '
  for policy in corey static; do
    for subset in narrativeqa qasper gov_report multifieldqa_en; do
      CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python \
        src/experiments/run_policy_corey_ablation.py \
        --model-id benchang1110/mamba2-2.7b-hf \
        --n 20 --policy ${policy} \
        --data-file src/data/longbench_subset/${subset}/test.jsonl \
        --device cuda \
        --output-dir src/outputs/policy_corey_ablation_${policy}_${subset}
    done
  done
' > exp3.log 2>&1 &
tail -f exp3.log
```

**Note on fast path:** Without `mamba_ssm` CUDA extensions, the HF naive path
is used.  On a single RTX 3070 this yields ~34 s/sample for Mamba2-2.7B.
On the 4-card RTX 3090 server with `mamba_ssm` installed, expected throughput
is ~15 tokens/s (~2.1 s/sample for 32 generated tokens).

### Experiment 4 — Fused Kernel Algorithm Benchmark

```bash
# Local (no GPU required for the algorithm benchmark itself):
PYTHONPATH=src python src/experiments/run_fused_kernel_benchmark.py \
  --num-ops 32 --repeats 1000

# With sweep over chain lengths and entropy regimes (default):
PYTHONPATH=src python src/experiments/run_fused_kernel_benchmark.py \
  --num-ops 64 --repeats 1000 --device cuda

# Background + monitor:
nohup bash -c '
  PYTHONPATH=src python src/experiments/run_fused_kernel_benchmark.py \
    --num-ops 64 --repeats 2000
' > exp4.log 2>&1 &
tail -f exp4.log
```

---

## Output Locations

| Experiment | Output directory |
|------------|-----------------|
| 1 (external baselines) | `src/outputs/external_baselines_<task>/` |
| 2 (Quamba) | `src/outputs/quamba_quant_benchmark_<task>/` |
| 3 (policy COREY) | `src/outputs/policy_corey_ablation_<policy>_<task>/` |
| 4 (fused kernel) | `src/outputs/fused_kernel_benchmark/` |

Per-run parallel logs: `src/outputs/run_all_4gpu_logs/`

---

## LongBench Subset Data

Local JSONL files are under `src/data/longbench_subset/`:

| Task | File | Samples |
|------|------|---------|
| NarrativeQA | `narrativeqa/test.jsonl` | 200 |
| Qasper | `qasper/test.jsonl` | 200 |
| GovReport | `gov_report/test.jsonl` | 200 |
| MultifieldQA-EN | `multifieldqa_en/test.jsonl` | 150 |

Use `--max-prompts 20` for quick runs (matches the paper's 20-sample protocol).

---

## Example Prompts for Smoke Testing

The following representative prompts match each LongBench task style:

### NarrativeQA
```
Summarize the following story in one paragraph: [story text]
```

### Qasper
```
Answer the following question based on the passage:
Passage: [passage text]
Question: What was the main finding of the study?
```

### MultifieldQA-EN
```
Extract the main entities and their relationships from: [text]
```

### GovReport
```
Summarize the following government report in two sentences: [report text]
```

Use `--prompt "Your test prompt here"` to override data-file loading for smoke tests.
