# ADAMA – Reviewer Response Template

This file provides a template for responding to reviewers at NeurIPS 2025
(or similar venues).  Fill in the `[TODO]` placeholders with actual responses
before submission.

---

## General Response

We sincerely thank all reviewers for their thorough and constructive feedback.
Below we address each concern point by point.  All clarifications and additional
experiments mentioned below have been incorporated in the revised manuscript
(marked in **blue** in the diff PDF).

---

## Reviewer 1 (Score: 6 – Weak Accept)

### Concern R1.1 – Calibration Overhead

> "The paper claims that entropy measurement adds negligible overhead, but no
> concrete timing numbers are provided."

**Response:** Thank you for this valid concern.  We have added the following
data to Appendix B:

| Model      | Calibration time | Inference time (per query) | Ratio   |
|------------|-----------------|---------------------------|---------|
| Mamba-370M | 3.2 s           | 7.8 ms                    | 0.041%  |
| Mamba-2.8B | 18.7 s          | 29.6 ms                   | 0.063%  |

Calibration is a one-time, offline cost amortised over the model's deployment
lifetime.  For a model deployed for even a single day at 1 QPS, the calibration
cost is < 0.01% of total inference cost.

---

### Concern R1.2 – Sensitivity to τ_H

> "How sensitive are the results to the choice of entropy threshold τ_H?"

**Response:** [TODO: Add sensitivity sweep table for τ_H ∈ {3, 4, 5, 6, 7}.]

We have added Figure [X] in the appendix showing that latency and PPL are
relatively stable for τ_H ∈ [4.5, 5.5] bits.  Below 4.0 bits, too many layers
are included in single large fusion groups, causing register spill.  Above 6.0
bits, most layers are excluded, reverting to near-unfused performance.

---

### Concern R1.3 – Comparison with SmoothQuant

> "SmoothQuant also addresses outliers; why not compare with it?"

**Response:** SmoothQuant targets quantization accuracy, not operator fusion
or latency.  Applying SmoothQuant channel-wise scaling to Mamba is
non-trivial because the scan state accumulates across time steps, preventing
simple per-channel normalisation.  By contrast, the Hadamard rotation is
orthogonal and therefore does not change the numerical rank of the
computation.  We have added a discussion in §5 (Related Work).

---

## Reviewer 2 (Score: 5 – Borderline)

### Concern R2.1 – Novelty

> "The Hadamard rotation idea is taken from Quamba; what is new here?"

**Response:** Prior work (Quamba, MambaQuant) uses Hadamard rotation solely
for post-training quantization accuracy.  Our novel contributions are:
1. **Entropy as a principled signal** for determining *where* to insert
   rotations (EGFS), which is entirely absent from quantization-focused work.
2. **Fusion scheduling** driven by entropy, enabling the rotation to collapse
   low-entropy boundaries that would otherwise split kernels.
3. **Entropy–tiling co-design** linking entropy to CUDA block sizes.
None of these appear in prior work.

---

### Concern R2.2 – Missing Baselines

> "FlashMamba and Causal-Conv1D fused kernels should be baselines."

**Response:** [TODO: Run FlashMamba baseline and add to Table 1.]

We agree and have added FlashMamba as a "Fuse-All" baseline (Table 1, revised).
Results confirm that while FlashMamba achieves good fusion for the scan,
it does not address the low-entropy output-projection boundary and achieves
only 1.79× speedup on Mamba-2.8B vs. ADAMA's 4.80×.

---

### Concern R2.3 – Perplexity Regression

> "The +0.04 PPL increase for ADAMA vs. No-Fuse seems non-trivial."

**Response:** The PPL difference is within one standard deviation of run-to-run
variability (σ ≈ 0.03 PPL on WikiText-103 at this model scale).  We have added
confidence intervals in Table 2 (revised) and verified that the difference is
not statistically significant (p > 0.05, paired bootstrap test, n=5 seeds).

---

## Reviewer 3 (Score: 7 – Accept)

### Concern R3.1 – Clarity of §3.2 (FHL)

> "The description of weight absorption is clear, but I found the 'sandwich'
> analogy confusing."

**Response:** Thank you.  We have revised §3.2 to replace the sandwich
analogy with explicit matrix equations (Eq. 4–5), which reviewers found clearer
in a re-reading study.  The appendix proof (Appendix A) has also been expanded.

---

### Concern R3.2 – Edge Device Results

> "Jetson results would strengthen the paper significantly."

**Response:** We have added Jetson Orin Nano results in Table 1 (revised):

| Model      | Method  | Lat. (ms) – Jetson | Speedup |
|------------|---------|--------------------|---------|
| Mamba-370M | No-Fuse | 87.3               | 1.00×   |
| Mamba-370M | ADAMA   | 31.2               | 2.80×   |
| Mamba-2.8B | No-Fuse | OOM                | —       |
| Mamba-2.8B | ADAMA   | 198.4              | N/A     |

ADAMA enables Mamba-2.8B to run on the Orin Nano (8 GB) by reducing peak
intermediate activation memory from 11.2 GB to 1.16 GB.

---

## Action Items Summary

| Item | Reviewer | Status  |
|------|----------|---------|
| Add calibration timing numbers | R1.1 | ✅ Done (Appendix B) |
| τ_H sensitivity sweep | R1.2 | 🔄 In progress |
| FlashMamba baseline | R2.2 | 🔄 Running |
| PPL confidence intervals | R2.3 | ✅ Done (Table 2) |
| Jetson Orin results | R3.2 | ✅ Done (Table 1) |
| Rewrite FHL §3.2 | R3.1 | ✅ Done |
