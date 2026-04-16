# Stage 9 — R2 Domain Expert Review Report

**Reviewer Role:** Peer Reviewer 2 — SSM/Mamba Domain Expert  
**Paper:** COREY — Entropy-Guided Kernel-Level Scheduling and Hadamard Reparameterization for SSMs  
**Review Date:** 2026-04-16  
**Expertise:** State Space Models, Mamba architecture, SSM inference optimization, sequence modeling efficiency

---

## Scores

| Dimension | Score | Notes |
|-----------|-------|-------|
| Novelty | 62 | New application of entropy to chunk scheduling; incremental relative to existing SSM optimization work |
| Literature Integration | 65 | Key SSM papers cited; Mamba-2 and RWKV-6 noted as future work rather than baselines |
| Evidence Sufficiency | 63 | Mamba-370M and 1.4B covered; 2.8B results are partial |
| Argument Coherence | 68 | Contribution story is now clearer after prior revisions |
| Significance | 58 | Important for SSM deployment; limited to one scheduling decision |

**Weighted (custom domain weights): 63.2/100**

---

## Detailed Assessment

### Domain Strengths

1. **Correct treatment of selective scan as the bottleneck.** The paper correctly identifies that selective state updates in Mamba-family models are decomposed into fragmented kernels with repeated intermediate tensor materialization. This is an accurate characterization of the current `mamba_ssm` implementation and the source of the memory-bandwidth bottleneck.

2. **Good coverage of related SSM work.** Mamba, Mamba-2 (structured state space duality), Quamba, and MambaQuant are cited. The paper correctly positions COREY relative to quantization-oriented SSM work (Quamba, SmoothQuant, AWQ).

3. **Real checkpoint validation on multiple scales.** Mamba-370M and Mamba-1.4B are both evaluated. The Mamba-2.8B results (partial, with PPL anomaly handled honestly) extend the scale coverage.

4. **Negative finding on Theorem 1 is a genuine scientific contribution.** The empirical falsification of the doubly-stochastic mixing condition on real Mamba `in_proj` outputs (160/160 pairs showing entropy decrease) is an informative finding about the distributional regime of real SSM activations. This is correctly reported and documented.

### Domain Weaknesses

**[R2-D1] Absence of Mamba-2 comparison leaves a major gap for domain readers (MAJOR).**  
Mamba-2 (Gu & Dao, 2024) introduces a fundamentally different scan algorithm (state space duality, SSD) with substantially different hardware characteristics. The paper studies Mamba-1.x exclusively. For domain readers, the natural question is: does entropy-guided chunk selection work for Mamba-2's SSD scan? If Mamba-2 eliminates the fragmentation problem through its scan formulation, COREY's scheduling approach may be irrelevant to the future of the SSM ecosystem. The paper notes this as future work in Limitations, but the paper's contribution claims implicitly cover "SSMs" broadly. Scoping the title/abstract to "Mamba-1.x" or "first-generation SSMs" would be more accurate.

**[R2-D2] The claim that entropy of INPUT activations motivates aggressive fusion is theoretically unresolved (MAJOR).**  
The paper uses input activation entropy (H_input) as the scheduling signal. The theoretical motivation (Theorems 1–3) establishes:
- Theorem 1: Hadamard rotation increases histogram entropy (in synthetic heavy-tailed regime; falsified on real checkpoints)
- Theorem 2: Fusion depth is bounded by memory constraints
- Theorem 3: Hadamard reduces peak coordinate magnitude

However, the claim that HIGH INPUT ENTROPY → SAFE TO USE LARGE CHUNKS is based on a different informal reasoning: "activation mass is more evenly distributed and resource use is more predictable" (Section 3.3). This reasoning is not formalized or experimentally validated on real checkpoints. After Theorem 1 is falsified, the theoretical bridge between the scheduling signal (H_input) and the safety of large chunks on real data is informal reasoning only. The paper acknowledges this implicitly but does not clearly state it.

**[R2-D3] LongBench scores are too limited to characterize model quality (MINOR).**  
The paper reports LongBench results: 4 tasks × 20 samples. For Mamba-370M, scores are reported as "token-F1" and "ROUGE-L" but the discussion notes that "MF-EN exact-match is zero across all models due to strict exact-match evaluation on 32-token generation." This suggests harness configuration issues rather than genuine quality measurement. While the paper correctly labels these as harness verification rather than quality claims, it would be cleaner to either fix the evaluation setup or remove these scores from the main text entirely.

**[R2-D4] No comparison with non-Mamba SSMs (MINOR).**  
The paper claims to address "Selective State Space Models" broadly, but all experiments are on the `state-spaces/mamba-*` checkpoints. RWKV-6 and similar architectures are mentioned only as future work. For a paper titled about "Selective State Space Models" (plural, generic), scoping to Mamba-1.x more explicitly would be appropriate.

---

## Domain-Specific Questions

1. Does the structured state space duality (SSD) formulation in Mamba-2 change the memory access pattern in a way that would make COREY's chunk-size selection approach fundamentally different? Is there a simple experiment on Mamba-2 that could clarify the scope?

2. The Hadamard layer is described as absorbing into `in_proj`. In practice, `in_proj` in Mamba-1 is a large linear projection. How does fusing Hadamard into this projection affect the overall computation graph, and does it interact with the existing fast path (SSM fast scan) in the `mamba_ssm` library?

3. The Quamba installation is verified and `import quamba` passes. Even a qualitative test (e.g., does Quamba-quantized Mamba-370M produce reasonable outputs?) would be valuable to confirm that the quantization path is technically accessible for future work.

---

## Summary

For domain readers, the key gaps are: (1) the scope is implicitly broad (SSMs) but explicitly narrow (Mamba-1.x); (2) Mamba-2 is the obvious next step that the paper cannot currently address; (3) the theoretical motivation for input-entropy-guided scheduling on real checkpoints remains informal after Theorem 1 is falsified. These are not blocking concerns for a prototype paper, but they limit how broadly the community will find the contribution applicable.
