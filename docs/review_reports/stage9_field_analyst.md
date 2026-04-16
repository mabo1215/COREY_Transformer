# Stage 9 — Field Analysis and Reviewer Configuration Card

**Date:** 2026-04-16  
**Stage:** Pipeline Stage 9 (Independent Full Review)  
**Target Venue:** NeurIPS 2026  
**Reviewer Framework:** 5-Reviewer Panel (EIC + R1 + R2 + R3 + DA)

---

## Paper Summary

**Title:** COREY: A Prototype Study of Entropy-Guided Kernel-Level Scheduling and Hadamard Reparameterization for Selective State Space Models

**Research Problem:** Improving inference efficiency of Mamba-family Selective State Space Models (SSMs) by reducing memory-bandwidth overhead through better kernel scheduling and activation smoothing.

**Core Contribution:**
1. An entropy-driven chunk-size selection policy for the Triton `selective_scan_fn` kernel — COREY selects a larger chunk (e.g., 256) from a runtime entropy observation, reducing kernel invocations and improving HBM coalescing.
2. A Hadamard reparameterization layer that absorbs forward/inverse transforms into projection blocks to smooth activation outliers.

**Two-Tier Evidence Structure:**
- Tier 1: Python cost-model prototype (illustrative diagnostics only, NOT GPU measurements)
- Tier 2: Real-GPU chunked selective-scan benchmark (genuine kernel-level timing)

**Key Result:** COREY achieves 3.24× latency speedup vs. Static-64 on RTX 3070 (Triton kernel, seq_len=4096). An oracle-tuned Static-256 baseline achieves identical latency, confirming the speedup comes from chunk-size selection, not the entropy mechanism per se.

---

## Field Classification

- **Primary domain:** ML Systems / Efficient Inference
- **Secondary domain:** Selective State Space Models, Quantization-Oriented Architectures
- **Research paradigm:** Empirical systems study + mathematical formalization
- **Methodology type:** Quantitative (hardware benchmarks + theoretical analysis)
- **Target journal tier:** NeurIPS 2026 (Tier 1 ML conference, systems track)
- **Paper maturity:** Prototype / Proof-of-Concept, explicitly scoped as such

---

## Reviewer Configuration

| Reviewer | Identity | Focus |
|----------|----------|-------|
| **EIC** | NeurIPS 2026 Area Chair, specializing in efficient ML inference and systems papers | Journal fit, contribution scope, submission readiness |
| **R1** | GPU Systems Researcher — Triton/CUDA expert, operator fusion, memory-bandwidth optimization | Methodological rigor, kernel-level evidence quality, cost-model validity |
| **R2** | SSM/Mamba Architecture Expert — familiar with mamba-ssm codebase, SSM inference optimization, Mamba-2 architecture | Domain contribution, related work coverage, architectural analysis |
| **R3** | Applied Quantization Researcher — AWQ, GPTQ, Quamba, deployment considerations | Hadamard/quantization contribution, practical utility, deployment gap |
| **DA** | Devil's Advocate — challenges core claims about entropy guidance value, novelty, and causal attribution | Fundamental validity of entropy-guidance claim, circular evidence, practical value |

---

## Stage 9 Protocol Notes

- **READ-ONLY constraint**: This review process does NOT modify any files in `paper/`. All output is reports and revision guidance only.
- **Venue compliance check**: Conducted before scoring (Phase 1).
- **Independence**: Each reviewer has reviewed the paper independently without cross-referencing other reviewers.
