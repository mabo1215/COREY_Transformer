# Stage 9 — R3 Perspective Review Report

**Reviewer Role:** Peer Reviewer 3 — Applied Quantization / Deployment Researcher  
**Paper:** COREY — Entropy-Guided Kernel-Level Scheduling and Hadamard Reparameterization for SSMs  
**Review Date:** 2026-04-16  
**Expertise:** Post-training quantization (AWQ, GPTQ), hardware-efficient inference, deployment optimization

---

## Scores

| Dimension | Score | Notes |
|-----------|-------|-------|
| Novelty | 58 | Entropy for scheduling is fresh; Hadamard smoothing builds on Quamba/SmoothQuant |
| Methodology | 62 | GPU experiments credible; quantization component absent |
| Evidence Sufficiency | 60 | Scheduling story supported; quantization story hypothetical |
| Clarity | 68 | Good separation of tiers; some sections remain dense |
| Significance/Impact | 56 | Potential value for deployment, but gap to usable system is wide |

**Weighted Total: 60.8/100**

---

## Detailed Assessment

### Perspective Strengths

1. **The scheduling + quantization co-design perspective is a worthwhile research direction.** The observation that activation outlier suppression (via Hadamard) and fusion scheduling (via entropy) can be addressed jointly is conceptually valuable. If validated end-to-end, this could inform how future SSM inference runtimes are designed.

2. **Quamba installation is verified.** The paper reports `import quamba` passes with the Quamba 2.0.0a1 build on sm_89/CUDA 12.8. This establishes a technical path to quantization experiments in the future.

3. **Honest about quantization access constraints.** The note that "full W8A8/W4A8 perplexity evaluation remains future work, pending access to sm_89 hardware" (Section 8, Limitations item 8) is a clear and honest disclosure.

4. **The Hadamard-quantization connection is theoretically sound.** Theorem 3's $\ell_\infty$ bound ($\|z\|_\infty \leq \|x\|_1/\sqrt{d}$) is correct and directly relevant to clipping-limited low-bit quantization. The caveat for $D = \Theta(d)$ outliers is now in the main text (following M4 revisions).

### Perspective Weaknesses

**[R3-P1] The quantization contribution is entirely theoretical — no measured quality results (MAJOR).**  
Section 3.2 (Fused Hadamard Layer), Section 4.3 (Theorem 3: Stability Under Quantization), and the "Empirical implication" discussion in Section 4.3 all connect Hadamard reparameterization to improved low-bit inference. However:
- No perplexity measurement under any quantization scheme is reported anywhere in the paper.
- The "Empirical implication" paragraph for Theorem 3 explicitly says that the analogous effect on real checkpoints "remains to be confirmed with full checkpoint quantization experiments."
- The Appendix's "bit-width sensitivity" ablation (tab:ablation_precision) is labeled as a "Diagnostic Proxy" from the illustrative cost model — not actual quantization results.

For a deployment-focused reader, the Hadamard contribution amounts to: "we absorb Hadamard rotation into projections and prove it should help quantization; we haven't measured it." This is a significant gap if the paper aims to position COREY as relevant to the deployment community.

**[R3-P2] The energy efficiency benefit is speculative (MINOR).**  
Section 8 (Broader Impact) states: "lower memory traffic per token could reduce the energy footprint of long-context SSM serving — though this remains provisional until hardware power measurements are collected." While honest, this means the energy efficiency benefit is entirely unverified. For a practical deployment paper, even a rough GPU power measurement (e.g., using `nvidia-smi dmon`) during the Tier-2 benchmark would provide a data point.

**[R3-P3] The "forward-modifying" gap limits deployment story (MAJOR).**  
The Hadamard layer is described as a fused operator that preserves functional equivalence (Section 3.2, $y = W' \hat{x}$). However, the Tier-2 evaluation hook is passive — it observes activations but does not apply the Hadamard transformation in the forward pass. This means:
- The actual forward computation is NEVER modified by COREY in any reported experiment.
- The "fused Hadamard layer" exists as a design description, not an implementation.
- Checkpoint perplexity with and without Hadamard reparameterization cannot be compared.

From a deployment perspective, the most compelling experiment would be: run Mamba-370M with fused Hadamard absorbed into `in_proj`, measure perplexity and latency versus baseline. This remains unexecuted.

**[R3-P4] Chunk-size selection granularity is unclear for non-uniform sequences (MINOR).**  
The COREY chunk-size formula uses the entropy of the ENTIRE input tensor as a single scalar. For long sequences with non-uniform statistical properties (e.g., long documents where early tokens are narrative and later tokens are data-heavy), a single global entropy estimate may be a poor proxy for the optimal chunk size at each position. The paper does not discuss whether per-chunk or per-layer entropy estimates would be beneficial.

---

## Questions for Authors

1. Is the sm_89 constraint for Quamba a fundamental limitation, or is there a CPU/FP32 quantization evaluation path that could demonstrate perplexity preservation at lower precision without requiring Ampere Ada hardware?

2. What is the overhead of absorbing Hadamard into `in_proj` at the model loading stage (a one-time weight transformation)? If it's low, why not implement the full forward-modifying variant for a small-scale proof of concept?

3. For real-world inference with variable-length inputs, does the entropy estimate stabilize within the first few tokens (EMA decay λ=0.85), or does it remain noisy? An entropy trajectory plot across a few hundred generation steps would help characterize the signal's practical stability.

---

## Summary

COREY's scheduling contribution is practical and well-supported. The quantization contribution (Hadamard reparameterization) is theoretically sound but experimentally absent. For deployment researchers, the most valuable next step is a small-scale forward-modifying Hadamard experiment that connects the theoretical framework to an actual inference quality measurement.
