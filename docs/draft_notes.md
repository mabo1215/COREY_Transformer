# ADAMA – Draft Revision Notes

This file tracks modification comments on successive drafts of the paper
**"ADAMA: Entropy-Guided Operator Fusion with Hadamard Transform for Efficient
State Space Model Acceleration"**.

---

## Draft v0.1 → v0.2  (Internal Review)

### Abstract
- [ ] Shorten to ≤ 200 words for NeurIPS submission portal.
- [x] Add explicit speedup number ($4.8\times$) in the last sentence.
- [ ] Replace "SSM" with the spelled-out form "State Space Model (SSM)" on
      first use.

### Introduction
- [x] Add citation for Mamba-2 in the paragraph on Fuse-All strategies.
- [ ] Strengthen the motivation paragraph: explain *why* entropy is a better
      signal than layer type or static heuristics.
- [ ] Move Table 1 (latency) reference forward into the intro paragraph to
      hook readers earlier.

### Methodology – EGFS (§3.1)
- [x] Add Algorithm 1 (EGFS pseudocode).
- [ ] Clarify that $\tau_H = 5.0$ bits is a tunable hyperparameter and provide
      sensitivity analysis in the appendix.
- [ ] Add a worked example showing how the fusion groups change as $\tau_H$
      varies from 3.0 to 7.0 bits.

### Methodology – FHL (§3.2)
- [x] Add proof of weight absorption to Appendix A.
- [ ] Clarify that the FHL is compatible with BF16 and FP8 weight formats.
- [ ] Note that odd-dimension models require padding to the next power of 2
      before applying the WHT.

### Experiments (§4)
- [ ] Add Jetson Orin Nano latency numbers to Table 1.
- [x] Add perplexity table (Table 2).
- [ ] Add a memory-footprint column to Table 1.
- [ ] Include throughput numbers (TPS) alongside latency.

### Related Work (§5)
- [ ] Add discussion of Jamba (hybrid SSM-Attention) to the SSM paragraph.
- [ ] Mention FineGrained-Quant work in the quantization paragraph.

### Conclusion
- [x] Add hardware design implications paragraph.
- [ ] Expand future-work bullet on online entropy estimation.

---

## Known Issues / TODO

1. **Figure 1 (overview)**: The current placeholder PDF needs to be replaced
   with a properly rendered TikZ or Inkscape figure before camera-ready.
2. **Figure 2 (entropy gain)**: Synthetic data used; replace with actual
   calibration measurements from the Mamba-2.8B checkpoint.
3. **Table 1**: Latency numbers are from a preliminary run on a single seed;
   average over 5 seeds before final submission.
4. **Appendix B (entropy estimation)**: Expand with a comparison of histogram
   vs. kernel density estimation vs. quantile-based entropy.

---

## Style Checklist (NeurIPS 2025 Requirements)

- [ ] Page limit: 9 pages + unlimited references.
- [ ] Font size: 10pt body, as required by the NeurIPS style file.
- [ ] Author anonymisation: all identifying information removed from PDF
      metadata (`pdfauthor` = empty).
- [ ] ArXiv pre-print: do not upload before review period ends.
- [ ] Code availability: add footnote with anonymised GitHub link.
