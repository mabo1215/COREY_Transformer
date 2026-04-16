# Stage 9 — EIC Review Report

**Reviewer Role:** Editor-in-Chief (NeurIPS 2026 Area Chair)  
**Paper:** COREY — Entropy-Guided Kernel-Level Scheduling and Hadamard Reparameterization for SSMs  
**Review Date:** 2026-04-16  
**Mode:** Full Review (Phase 1)

---

## Phase 0: Venue Compliance Check

| Compliance Dimension | Status | Notes |
|---------------------|--------|-------|
| LaTeX Template | ✅ PASS | `neurips_2026.sty` used in submission mode |
| Anonymous Authors | ✅ PASS | Author block replaced with "Anonymous Authors/Institution" |
| Page Limit | ⚠️ MARGINAL | Main body appears close to 9-page limit; appendix is lengthy. Needs careful final check before submission. |
| Double-Blind | ✅ PASS | No author-identifying information in text; anonymous.4open.science URL used |
| NeurIPS Checklist | ❌ MISSING | No NeurIPS 2026 checklist section visible in main.tex. This is **MUST FIX** for submission. |
| Reproducibility Statement | ✅ PASS | Code URL provided; experimental setup detailed |
| Broader Impact | ✅ PASS | Section 8 present |

**MUST FIX (Venue Compliance):** Add the NeurIPS 2026 mandatory checklist section.

---

## Scores

| Dimension | Score | Weight | Notes |
|-----------|-------|--------|-------|
| Novelty | 60 | 20% | Entropy-based chunk selection is a new lens, but limited to a specific scheduling decision |
| Methodology | 65 | 25% | Honest, two-tier design; real GPU evidence exists but is narrow |
| Experiments | 62 | 25% | Genuine speedup but equivalent to static oracle; limited diversity |
| Clarity | 72 | 15% | Well-written, clear scope of claims |
| Significance | 58 | 15% | Important problem, but prototype-level impact |

**Weighted Total: 63.25/100** → Borderline Major Revision

---

## Strengths

1. **Exemplary scientific honesty.** The explicit "Scope of claims" and "Not claimed" sections (Section 1, bullet 4) are rare and commendable. The authors acknowledge that their central theorem is falsified on real checkpoints (Theorem 1, Remark 1) and that the 3.24× speedup matches a static oracle (Table 2, Static-256 footnote). This level of transparency significantly strengthens the paper's credibility.

2. **Genuine two-tier evidence structure.** Unlike many prototype papers, COREY provides actual GPU kernel timing (Table 2) on real Triton kernels, cross-validated on RTX 3070 and RTX 3090. This is legitimate systems evidence, not simulation.

3. **Mathematical formalization of an engineering heuristic.** The connection between Hadamard-induced entropy increase, outlier suppression, and fusion safety (Theorems 1–3) provides a formal backbone for what could otherwise be ad hoc engineering.

4. **Perturbation study demonstrates adaptive behavior.** Appendix Table (tab:perturbation) shows that COREY adapts chunk size across 5 different activation distributions, yielding 2.64–4.34× speedup for typical high-entropy inputs and conservative selection for pathological sparse inputs.

---

## Weaknesses

### Major Issues

**[EIC-M1] NeurIPS 2026 checklist is absent (MUST FIX — venue compliance).**  
The NeurIPS checklist has been mandatory since 2023. The submitted version does not include this section. This is a hard requirement and will prevent acceptance regardless of scientific quality.

**[EIC-M2] The practical value of entropy guidance over static tuning is not quantified.**  
The paper's strongest result (Table 2) shows COREY equals Static-256 in latency. The paper argues the value is "automatic selection without manual oracle tuning." However, the empirical case for this argument rests on a single entropy observation (H=4.18 nats for one config/prompt in Table 1). No experiment demonstrates:
- That entropy varies meaningfully across real-world workloads (different models, prompts, sequence lengths), AND
- That a fixed static chunk size would be suboptimal across this variation.
Without this evidence, the practical value claim is asserted rather than demonstrated. A reader may reasonably ask: "If optimal chunk size is consistently high (e.g., always in {256, 512}), why not just use a fixed large chunk?"

**[EIC-M3] The Hadamard reparameterization layer is theoretically well-motivated but experimentally absent.**  
Theorems 1–3 together argue that Hadamard rotation increases entropy (under synthetic heavy-tailed regime), enabling deeper fusion, and reduces peak coordinate magnitude, improving quantization stability. However:
- On real checkpoints, Theorem 1 is falsified (entropy DECREASES in 160/160 pairs).
- No experiment measures the effect of actually applying the Hadamard layer to forward computation.
- Quantization results (W8A8/W4A8 perplexity) remain future work.
This means the Hadamard layer contributes theorems but zero measured results. For a NeurIPS systems paper, this is a significant gap.

### Minor Issues

**[EIC-m1] The two entropy signals are underspecified in the paper narrative.**  
COREY uses the entropy of INPUT activations as the chunk-size signal (Tables 1, 2). The Hadamard layer is motivated by the entropy GAIN after rotation (Theorem 1). These are different signals. Theorem 1's falsification affects the motivation for the Hadamard layer but NOT the chunk-size scheduling. The paper acknowledges this (Remark 1, Section 4.1) but the narrative flow from motivation to implementation would benefit from a clearer explicit separation.

**[EIC-m2] Section structure reflects revision history rather than optimal narrative flow.**  
The paper has undergone many revision cycles, and some structural decisions reflect incremental patching rather than a clean top-down design. For example, the "signal-to-decision-to-latency" bridge paragraph in Section 6.1 appears to respond to a prior reviewer request but is awkwardly placed before the main result table.

---

## Questions for Authors

1. Is there empirical evidence that activation entropy varies enough across real inference scenarios (different models, prompts, domains, sequence lengths) to justify per-prompt adaptive chunk sizing? If entropy is consistently high (or consistently low) across typical workloads, what is the practical benefit of runtime measurement over a one-time profiling pass?

2. Can the authors provide even a preliminary result on the forward-modifying Hadamard variant? Even a small perplexity experiment on Mamba-370M would dramatically strengthen the Hadamard contribution.

---

## EIC Recommendation

**Major Revision (Score: 63/100)**

The paper has a genuine contribution (runtime entropy-guided chunk-size selection that automates what static tuning achieves manually) and commendable scientific honesty. The immediate barrier to NeurIPS acceptance is the missing NeurIPS checklist (venue compliance MUST FIX) and the need to better quantify the practical advantage of entropy guidance over a well-tuned static baseline. The Hadamard contribution needs experimental grounding.
