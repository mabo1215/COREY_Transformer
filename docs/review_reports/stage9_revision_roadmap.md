# Stage 9 — Revision Roadmap

**Stage:** Pipeline Stage 9 — Independent 5-Reviewer Panel  
**Editorial Decision:** Major Revision  
**Score:** 63/100 (Borderline Major Revision)  
**Date:** 2026-04-16  
**Reviewers:** EIC, R1 (Methodology), R2 (Domain), R3 (Perspective), DA (Devil's Advocate)

---

## Editorial Decision Letter Summary

Dear Authors,

Thank you for submitting COREY to NeurIPS 2026. This is a fresh independent review conducted after the prior review cycle (C1–C8, M1–M9) has been addressed. The reviewers acknowledge substantial improvements: the paper now has genuine two-tier evidence, honest disclosure of limitations, a correctly included Static-256 oracle baseline, and proper labeling of Tier-1 cost-model results as illustrative diagnostics.

**Editorial Decision: Major Revision**

The paper cannot be accepted in its current form for the following reasons:

1. **Missing NeurIPS 2026 checklist** — this is a mandatory venue requirement.
2. **The practical value of entropy guidance is not demonstrated empirically** — the core "automatic vs. manual tuning" argument lacks the evidence that entropy varies enough in real workloads to justify runtime measurement.
3. **The Hadamard reparameterization contribution has no experimental grounding** — theoretically well-motivated but zero measured quality results.
4. **The entropy-guidance theoretical motivation is internally inconsistent** after Theorem 1's falsification, and needs cleaner separation between the two entropy signals.

The paper has a genuine contribution — runtime entropy-guided chunk-size selection that automates an optimization otherwise requiring per-model profiling. Strengthening the evidence that this automation is needed (entropy varies across real workloads) and providing at least a preliminary Hadamard quality experiment would bring the paper to acceptance territory.

Sincerely,  
NeurIPS 2026 Program Committee

---

## Consensus Summary

**CONSENSUS-3** on the following core findings:
- The paper is scientifically honest and the GPU benchmarking methodology is sound (R1, R2, EIC)
- The primary remaining gap is the missing evidence that entropy varies in real workloads (R1, DA, EIC)
- The Hadamard contribution needs experimental grounding (EIC, R3, DA)
- The NeurIPS checklist is absent and must be added (EIC — venue requirement)
- The theoretical narrative needs cleaner separation of the two entropy signals (R2, DA)

**DA-CRITICAL findings:**
- DA-C1: Practical value of entropy guidance is undemonstrated (CRITICAL)
- DA-C2: Theoretical motivation is internally inconsistent (CRITICAL)

**Per DA-CRITICAL findings, Accept is not possible.**

---

## Revision Roadmap (Prioritized)

| ID | Priority | Description | Reviewer | Type |
|----|----------|-------------|----------|------|
| REV-001 | must_fix | Add NeurIPS 2026 mandatory checklist | EIC | Venue Compliance |
| REV-002 | must_fix | Characterize entropy variance across real workloads (histogram of H across ≥50 prompts for Mamba-370M) | R1, DA, EIC | Experiment |
| REV-003 | must_fix | Clarify the two entropy signals (H_input for scheduling vs. ΔH from Hadamard) as separate contributions with separate justifications | R2, DA | Writing/Theory |
| REV-004 | should_fix | Add at least a preliminary forward-modifying Hadamard experiment (e.g., absorb H into in_proj offline, measure perplexity vs. baseline on Mamba-370M) | EIC, R3 | Experiment |
| REV-005 | should_fix | Justify the conservative chunk selection (chunk=256 vs. oracle chunk=512): show a case where chunk=512 causes numerical error, or revise the selection formula | R1, DA | Experiment/Writing |
| REV-006 | should_fix | Replace Table 1 (single-row entropy observation) with an entropy distribution plot or table showing variance across ≥50 real-inference prompts | R1, DA, EIC | Experiment/Writing |
| REV-007 | should_fix | Scope the title/abstract/introduction to Mamba-1.x rather than "SSMs" generically, since Mamba-2's SSD algorithm has different hardware characteristics | R2 | Writing |
| REV-008 | should_fix | Add a discussion of the "profile-once-and-fix" alternative (one-time chunk-size calibration) and explain why runtime entropy guidance is preferable | DA | Writing |
| REV-009 | consider | Report a power/energy measurement during the Tier-2 chunked scan benchmark (e.g., via nvidia-smi dmon) to ground the energy efficiency claim in data | R3 | Experiment |
| REV-010 | consider | Clarify the relationship between per-step entropy estimate stability (EMA decay) and the practical latency of reaching a stable recommendation | R3 | Writing |

---

## Verification Criteria

| ID | How to verify fix is adequate |
|----|------------------------------|
| REV-001 | NeurIPS 2026 checklist section appears in compiled main.pdf |
| REV-002 | A figure or table with entropy distribution statistics (mean, std, percentiles) for ≥50 real prompts from ≥2 tasks; should show whether H varies below the current chunk-size threshold |
| REV-003 | A clear paragraph or figure that names "Signal A: H_input for chunk scheduling" and "Signal B: ΔH from Hadamard for fusion depth" as separate; fixes the "entropy → Hadamard → scheduling" conflation |
| REV-004 | A perplexity measurement (even WikiText-103 20-sample) with H absorbed into in_proj vs. unmodified checkpoint |
| REV-005 | Either: (a) an experiment showing chunk=512 degrades perplexity for some activation regime, or (b) revised formula with entropy threshold for selecting chunk=512 |
| REV-006 | New table/figure in Section 6.1 or Appendix showing entropy distribution; caption should state the % of prompts that fall in each chunk recommendation bucket |
| REV-007 | Title, abstract, and introduction use "Mamba" or "Mamba-1.x" rather than generic "SSMs"; Limitations note that Mamba-2 is out of scope |
| REV-008 | A paragraph in Limitations or Discussion comparing runtime entropy guidance to one-time profiling calibration |
| REV-009 | Power measurement in Table 2 or Appendix with explicit caveat about measurement precision |
| REV-010 | A note on EMA convergence time in Appendix |
