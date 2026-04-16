# Quamba WSL2 Local Environment - Next Phase Experiments

**Status:** ✅ **PRODUCTION READY**
- Environment: quamba-py310 (Python 3.10, CUDA 12.1, all third-party libs)
- Validation: ✓ mamba-370m inference + LongBench narrativeqa evaluation complete
- Last run: 2/2 samples evaluated, metrics computed, latency ~27s/sample

---

## Quick Reference Commands

### 1. Fast Environment Verification (< 1min)
```bash
# Confirms CUDA visibility + core stack (torch, mamba_ssm, transformers)
wsl bash -lc 'export MAMBA_ROOT_PREFIX=/home/mabo1215/.adama-micromamba && \
  /mnt/c/source/COREY_Transformer/.wsl-tools/bin/micromamba run -n quamba-py310 \
  python /mnt/c/source/COREY_Transformer/test_quamba_env.py'
```

### 2. LongBench Evaluation (Parameterized)
```bash
# Template: replace TASK, NUM_SAMPLES, OUTPUT_NAME as needed
wsl bash -lc 'cd /mnt/c/source/COREY_Transformer && \
  export MAMBA_ROOT_PREFIX=/home/mabo1215/.adama-micromamba && \
  ./.wsl-tools/bin/micromamba run -n quamba-py310 \
  python -m src.experiments.run_longbench_inference \
    --model mamba-370m \
    --tasks TASK \
    --max-samples NUM_SAMPLES \
    --max-length 4096 \
    --device cuda \
    --dtype float16 \
    --batch-size 1 \
    --output-dir src/outputs/quamba_longbench_OUTPUT_NAME'

# Example: narrativeqa with 10 samples
wsl bash -lc 'cd /mnt/c/source/COREY_Transformer && \
  export MAMBA_ROOT_PREFIX=/home/mabo1215/.adama-micromamba && \
  ./.wsl-tools/bin/micromamba run -n quamba-py310 \
  python -m src.experiments.run_longbench_inference \
    --model mamba-370m \
    --tasks narrativeqa \
    --max-samples 10 \
    --max-length 4096 \
    --device cuda \
    --dtype float16 \
    --batch-size 1 \
    --output-dir src/outputs/quamba_longbench_nqa10'
```

### 3. Language Model Perplexity (WikiText-103, PG19)
```bash
# Not yet pre-configured; requires adding --lm-datasets flag
# Placeholder for future implementation
```

---

## Recommended Next Steps (Choose One)

### Option A: Local Comprehensive Evaluation (2-4 hours)
**Goal:** Establish comprehensive mamba-370m baseline on multiple LongBench tasks + perplexity

1. **Multi-task LongBench** (narrativeqa, qasper, multifieldqa_en, gov_report)
   ```bash
   # Run each task with larger sample count (e.g., --max-samples 50)
   ```
2. **WikiText-103 Perplexity** (if --lm-datasets flag available)
3. **Output:** Detailed metrics CSV for comparison with future quantization runs

**Pros:** 
- All results local; no network latency or coordination issues
- GPU (RTX 3070) sufficient for 370m model
- Can quickly iterate and debug

**Cons:**
- Single-GPU bottleneck (370m fits; 1.4b may need optimization)
- Results not yet useful for paper (small scale)

---

### Option B: Remote Server Synchronization + Distributed Runs (4-8 hours)
**Goal:** Sync repo to GPU server, set up quamba-py310 there, run larger experiments in parallel

**Prerequisites:**
1. Complete SSH key auth to 10.147.20.176:
   ```bash
   # MANUAL STEP (user must run once and input server password):
   ssh-copy-id -i ~/.ssh/id_ed25519.pub mabo1215@10.147.20.176
   ```

2. After SSH works:
   ```bash
   # Check remote GPU availability
   ssh mabo1215@10.147.20.176 'nvidia-smi'
   
   # Sync local repo to remote
   rsync -av /mnt/c/source/COREY_Transformer mabo1215@10.147.20.176:~/
   
   # Run remote setup (same as local)
   ssh mabo1215@10.147.20.176 'bash ~/COREY_Transformer/src/scripts/wsl_setup_quamba_env.sh'
   ```

3. Launch distributed experiments:
   - Local: Run mamba-370m on LongBench (this machine)
   - Remote: Run mamba-1.4b or mamba-2.8b on LongBench (peer GPU)
   - Both in parallel → 2x throughput

**Pros:**
- Access to larger Mamba models (1.4b, 2.8b)
- Parallel evaluation reduces total time by ~50%
- Preparation for actual paper-scale experiments

**Cons:**
- Requires SSH authorization (manual step by user)
- Network dependency (rsync, ssh tunneling)
- More complex debugging if issues arise

---

### Option C: Local Quantization Trial (1-3 hours)
**Goal:** Test AutoAWQ or GPTQ quantization pathway with mamba-370m as proof-of-concept

**Available in quamba-py310:**
- `--quant-backend awq` or `--quant-backend gptq` flags in run_longbench_inference.py
- Megatron-LM already installed (may need adaptation to Mamba architecture)

**Estimated Workflow:**
1. Run AWQ quantization on mamba-370m (4-bit, group_size=128)
2. Benchmark W4A8 vs FP16 latency + accuracy on narrativeqa (5 samples)
3. Collect metrics → show quantization effectiveness

**Pros:**
- Clear demo of quantization capability
- Directly supports paper narrative ("we also tested quantization")
- Fast turnaround (single GPU sufficient)

**Cons:**
- Requires debugging AutoAWQ integration with Mamba (may fail)
- Only proof-of-concept scale (5 samples)

---

## Current Environment Details

### Installed Runtime Stack
| Package | Version | Status |
|---------|---------|--------|
| Python | 3.10.16 (micromamba) | ✅ Active |
| PyTorch | 2.4.0+cu121 | ✅ CUDA visible |
| transformers | 4.41.2 | ✅ Loaded |
| datasets | 2.19.0 | ✅ LongBench working |
| mamba_ssm | 2.2.2 | ✅ Inference functional |
| fast-hadamard-transform | 1.0.4.post1 | ✅ Compiled |
| lm-evaluation-harness | 0.4.2 | ✅ Metrics computed |
| Megatron-LM | 0.10.0 | ✅ Available (unused) |

### GPU Details
- Device: NVIDIA GeForce RTX 3070
- Memory: 8 GB GDDR6
- Compute Capability: 8.6
- Torch confirmed access: ✅ torch.cuda.is_available() = True

### Data Locations
- Model cache: `~/.cache/huggingface/hub/` (shared)
- Experiment outputs: `src/outputs/quamba_longbench_*` (tracked)
- LocalBench data: `data/longbench_subset/` + `data/longbench_smoke/` (versioned)

---

## Execution Timeline Estimates

| Experiment | Duration | Resource |
|-----------|----------|----------|
| Env verify | 1 min | CPU (startup only) |
| Smoke test (2 samples) | 1 min | GPU only during inference |
| 50-sample task | 25 min | GPU (700+ token sequences) |
| 4-task LongBench | ~2 hours | GPU (sequential, 50 samples each) |
| WikiText-103 (baseline PPL) | 30 min | GPU (varies by sequence length) |
| Quantization trial (AWQ) | 45 min | GPU + CPU (calibration, then inference) |
| **Full local suite** | **~4 hours** | GPU (can overlap with remote runs) |

---

## Decision Matrix

| Goal | Local | Remote | Quantization |
|------|-------|--------|--------------|
| **Validate baseline functionality** | ✅ Done | ⏳ Pending SSH | N/A |
| **Establish reference metrics** | 🟨 In-progress | ✅ Better (1.4b, 2.8b) | 🟥 Not needed |
| **Test quantization pathway** | ⏳ Can do | ⏳ After sync | 🟥 Primary focus |
| **Prepare paper results** | 🟥 Too small | ✅ Needed | ✅ If shows benefit |
| **Time-to-first-result** | 🟩 Fast (1-2h) | 🟨 Medium (4h) | 🟨 Medium (1.5h) |
| **Complexity** | 🟩 Low | 🟨 Medium | 🟨 Medium |

---

## User Decision Required

**Please choose ONE of the following options and let me know:**

1. **"Continue local: run full multi-task LongBench + perplexity baseline"**
   - I will execute Option A
   - Estimated time: 3-4 hours
   - Output: Comprehensive mamba-370m metrics CSV

2. **"Set up remote: complete SSH auth, sync repo, run distributed experiments"**
   - I will guide you through SSH (manual step required), then execute remote setup
   - Estimated time: 2 hours setup + 4 hours experiments
   - Output: Parallel 370m (local) + 1.4b/2.8b (remote) metrics

3. **"Test quantization: AWQ trial on mamba-370m"**
   - I will run quantization pipeline and benchmark W4A8 vs FP16
   - Estimated time: 1-2 hours
   - Output: Quantization effectiveness analysis

4. **"Do all three sequentially"**
   - Starting with Option A, then B, then C
   - Estimated time: 10-12 hours total (can be split across sessions)

**Which would you prefer? (Or provide custom next steps)**
