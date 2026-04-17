# EIC Review (Stage 9 Reset, 2026-04-18)

Overall recommendation: Borderline Reject

NeurIPS-style overall score:
- Overall: 3 / 6 (Borderline Reject)
- Confidence: 4 / 5

Weighted assessment:
- Novelty: 54 / 100
- Methodology: 49 / 100
- Experiments: 43 / 100
- Clarity: 66 / 100
- Significance: 47 / 100
- Editorial overall: 52 / 100

Summary:
The paper presents an interesting runtime signal for chunk-size selection in Mamba-1.x kernels and is refreshingly honest about its current prototype status. However, the current submission still falls short of NeurIPS 2026 main-track expectations on three fronts: venue compliance, closed-loop validation, and significance relative to simpler alternatives.

Major findings:
1. The current submission is not format-compliant. In the rebuilt `paper/main.pdf`, the main text continues through page 11 and references begin on page 12, exceeding the NeurIPS 2026 limit of 9 content pages.
2. The most important systems claim remains unclosed at checkpoint level. Section 6.3 measures the inline overhead of entropy computation and chunk selection, but the selected chunk is still not passed into the real checkpoint scan path. The only direct speedup evidence is the synthetic-input W1 kernel benchmark in Section 6.2.
3. The paper's own evidence weakens the case for runtime adaptivity on the measured workload. The 80 real prompts all map to the same coarse chunk bucket, while Static-512 remains materially faster than COREY in the W1 benchmark.
4. The manuscript spends substantial page budget on a Hadamard/quantization extension that is explicitly not validated in the real-checkpoint path and whose main theoretical assumption is empirically violated on real activations.
5. Evaluation breadth is still below the bar for a systems paper at this venue.

Strengths:
- Clear scoping and explicit caveats.
- Real kernel measurements on two GPU environments.
- Thoughtful appendix and reproducibility disclosures.

Editorial verdict:
- I would encourage revision, but not acceptance in the present form.
- The paper needs a sharper concept-and-feasibility framing, a compliant page budget, and stronger evidence that runtime entropy guidance delivers value beyond one-time profiling on real checkpoints.
