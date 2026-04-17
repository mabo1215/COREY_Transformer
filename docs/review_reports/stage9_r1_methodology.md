# R1 Methodology Review (2026-04-18)

Recommendation: Weak Reject

Scores:
- Novelty: 56 / 100
- Methodology: 45 / 100
- Experiments: 44 / 100
- Clarity: 64 / 100
- Significance: 46 / 100
- Overall: 51 / 100
- Confidence: 4 / 5

Main strengths:
- The manuscript carefully distinguishes Signal A (runtime input entropy) from Signal B (Hadamard entropy gain).
- The authors openly acknowledge when a theoretical condition fails on real checkpoints.
- The resource-aware formulation in Sections 3-4 is cleaner than in many prototype systems papers.

Main concerns:
1. Claim-to-evidence mismatch persists. The validated real-checkpoint result is an inline monitoring-and-decision path with 8.3% overhead, not a deployed scheduler that changes scan execution and demonstrates end-to-end benefit.
2. Theorem 1 is no longer central to the validated contribution. The paper itself reports that the doubly-stochastic mixing condition is falsified on real checkpoint activations, so the main theorem explains only the synthetic calibration regime. That makes the theorem more of a side analysis than a pillar of the paper's empirical story.
3. The current evaluation knowingly uses a miscalibrated default. Section 6.2 recommends `H_ref = log K`, while the main experiments retain `H_ref = 8.0`; Appendix A.16-A.17 then shows that this choice systematically prevents the method from reaching the oracle chunk in the measured regime.
4. Statistical quality remains uneven. Some kernel results use `n=30`, the active integration uses `n=5`, and the passive hook feasibility table still uses `n=1`; several estimates are explicitly extrapolated rather than measured.

What would change my recommendation:
- Either integrate the selected chunk into the real checkpoint scan path, or reduce the paper to a narrower feasibility note that no longer implies a systems speedup contribution.
- Remove or compress the non-essential Hadamard material from the main body.
- Re-evaluate the runtime policy with the principled calibration and matched static baselines on the same real-checkpoint path.
