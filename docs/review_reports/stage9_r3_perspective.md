# R3 Perspective Review (Deployment / Practical Value, 2026-04-18)

Recommendation: Weak Reject

Scores:
- Novelty: 55 / 100
- Methodology: 54 / 100
- Experiments: 47 / 100
- Clarity: 69 / 100
- Significance: 51 / 100
- Overall: 55 / 100
- Confidence: 3 / 5

What I like:
- The paper is honest about being a prototype.
- The broader-impact and limitations sections are much better than average for an efficiency paper.
- The code/data availability story appears reasonably mature for a revision-stage manuscript.

What limits practical impact:
1. The measured real workload appears homogeneous enough that runtime selection currently behaves like a one-time auto-calibration mechanism, not a truly adaptive serving policy.
2. The practical trade-off is not favorable yet: the active inline scheduler costs 8.3% overhead, while the deployment benefit is still inferred from a separate kernel benchmark rather than measured on the real checkpoint serving path.
3. The strongest operational baseline is not runtime entropy guidance but one-time profiling. The paper recognizes this, but the current draft still frames COREY as if the runtime decision is the main practical win.
4. The submission is too dense for its page budget. Important caveats are pushed into appendix cross-references, which makes the main story harder to evaluate quickly.

Practical recommendation:
- Reframe the paper as a concept-and-feasibility submission centered on "runtime entropy as a control signal" rather than "a validated deployment scheduler."
- Show one real workload mixture where runtime switching is actually necessary, or concede that the current benefit is mostly automated static tuning.
