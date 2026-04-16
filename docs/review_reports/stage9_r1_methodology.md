# Stage 9 — R1 Methodology Review Report

**Reviewer Role:** Peer Reviewer 1 — GPU Systems / Kernel Methodology  
**Paper:** COREY — Entropy-Guided Kernel-Level Scheduling and Hadamard Reparameterization for SSMs  
**Review Date:** 2026-04-16  
**Expertise:** Triton/CUDA kernel design, operator fusion, memory-bandwidth analysis, GPU profiling

---

## Scores

| Dimension | Score | Notes |
|-----------|-------|-------|
| Novelty | 62 | Entropy-driven chunk selection is a non-trivial idea; implementation is narrow |
| Methodological Rigor | 67 | Real kernel timing is solid; cost model correctly labeled as illustrative |
| Evidence Sufficiency | 65 | Core GPU claim well-supported; entropy variance evidence weak |
| Argument Coherence | 68 | Clear logical flow, good scope demarcation |
| Writing Quality | 70 | Technically precise |

**Weighted Total: 66.3/100**

---

## Detailed Assessment

### Methodology Strengths

1. **The chunked selective-scan benchmark is well-designed.** Table 2 (tab:w1_chunked_scan) uses the real `mamba_ssm.ops.selective_scan_interface.selective_scan_fn` Triton kernel. The comparison across three policies (No-Fusion, Static-64, COREY) with n=30 repeats and 5 warmup iterations follows good benchmarking practice. Cross-validation on RTX 3070 (WSL2/CUDA 12.8) and RTX 3090 (native Linux/CUDA 12.1) strengthens reproducibility.

2. **The Static-256 oracle is correctly included.** Including Static-256 (analytically equivalent to COREY for the tested configuration) makes the result interpretable: the speedup comes from chunk size, not from entropy computation per se. This is honest and scientifically correct.

3. **Entropy overhead negligible (Table 3 in appendix).** The hook microbenchmark correctly shows that entropy computation adds no measurable latency. The negative deltas are attributed to noise, which is appropriate for a feasibility check.

4. **Cost model correctly labeled.** All Tier-1 tables are properly labeled as "illustrative cost-model behavior" in the appendix. The near-zero variance is explained as a deterministic simulation artifact.

### Methodology Weaknesses

**[R1-M1] Critical: Entropy variance not characterized for real inference workloads (MAJOR).**  
The core systems claim is that entropy-guided chunk selection enables DYNAMIC adaptation to varying activation statistics. However, Table 1 reports only one entropy observation: H=4.18 nats for a single NarrativeQA 512-token prompt on Mamba-370M, yielding tile recommendation 288. This is insufficient to establish that:
- Entropy varies meaningfully across different prompts within a real workload
- Entropy varies across different sequence lengths during inference
- Entropy varies across different model sizes or domains

If real-world entropy stays consistently in the "high" regime (as the single observation suggests), then a fixed large static chunk (e.g., 512) would always be optimal, eliminating the need for runtime entropy measurement. The perturbation study (Appendix, tab:perturbation) uses synthetic distributions rather than real-inference traces, so it does not address this concern. **This is the primary gap in the systems argument.**

**[R1-M2] COREY selects chunk=256 while oracle is chunk=512 — the selection criterion is suboptimal for uniform-random activations (MAJOR).**  
The appendix chunk-size sweep (tab:chunk_sweep) shows latency decreases monotonically: chunk=512 is 53.4% faster than COREY's chunk=256 on uniformly random activations. The paper explains that "conservative selection is intentional, as very large chunks can accumulate larger numerical error over the recurrent state when activations are non-uniform." However:
- The benchmark uses uniformly random activations, for which this concern does not apply.
- No experiment demonstrates when chunk=512 actually causes numerical error in practice.
- If the paper advocates entropy-guided selection, it should show a case where entropy correctly predicts that chunk=256 is safer than chunk=512 because of numerical concerns.
Without this evidence, the conservative selection appears to be a hardcoded policy decision rather than an entropy-guided one.

**[R1-M3] The "fusion" language in the title and algorithm overstates the implementation (MINOR).**  
Algorithm 1 is titled "Entropy-Guided Boundary Selection" and describes grouping operators into fusion groups. Section 3.1 discusses fusion depth and Theorem 2 bounds fusion depth. However, the Tier-2 implementation is specifically chunk-size selection for the selective scan operator — not general operator fusion. The paper correctly clarifies this in several places (following C7 revisions), but the overall framing still carries "kernel-level scheduling" and "fusion" terminology that exceeds the current implementation scope.

**[R1-M4] Real-GPU DRAM measurement absent (MINOR).**  
The paper reports "estimated HBM bytes" as a parsed tensor-volume proxy, not hardware counters. For a NeurIPS systems paper claiming to reduce DRAM traffic, an actual DRAM measurement (e.g., via Nsight `dram__bytes_read` counters) for the Tier-2 result would substantially strengthen the memory-efficiency claim. The appendix notes this is blocked by driver limitations, which is an honest disclosure.

---

## Questions for Authors

1. Can the authors provide an entropy distribution histogram across a representative set of real-world prompts (e.g., NarrativeQA, QaspeR, a code completion corpus) for Mamba-370M and Mamba-1.4B? If this distribution shows high variance, it supports the dynamic selection argument. If it shows consistently high entropy, it suggests static tuning is sufficient.

2. Is there a specific activation condition (e.g., sparse attention patterns, long zero-padded sequences) where large chunk sizes cause measurable perplexity degradation? This would provide the evidence needed to motivate conservative selection.

3. Appendix Algorithm 3 (Triton Fused SSM Kernel) is labeled as a prospective design target. What is the estimated engineering effort to implement this, and is there a path to implementation before the camera-ready deadline?

---

## Summary

The kernel benchmarking methodology is solid and the result is reproducible. The primary remaining gap is the lack of evidence that entropy actually varies in practice in a way that makes runtime measurement valuable over static configuration. This gap should be addressable with a relatively small experiment: logging entropy across a few hundred real prompts and reporting the distribution.
