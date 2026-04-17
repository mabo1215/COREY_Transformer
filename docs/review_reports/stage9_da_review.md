# Devil's Advocate Review (2026-04-18)

Recommendation: Reject

Overall score:
- Overall: 44 / 100
- Confidence: 4 / 5

Adversarial reading:
1. The paper is over budget for NeurIPS. That alone makes the current form unready for review.
2. The central end-to-end story is not closed. The runtime policy chooses chunks, but the chosen chunk is not actually used in the real checkpoint inference path that generates the paper's practical claims.
3. The strongest empirical message seems to be that a static oracle beats COREY by 35% on the showcased benchmark, while all measured real prompts land in the same bucket anyway. This makes the adaptive runtime argument look weak.
4. A large fraction of the main body is spent on Hadamard theory that is explicitly not validated on the real system and is empirically contradicted on real checkpoints.
5. The paper repeatedly asks the reader to accept future work as part of the present contribution: more heterogeneous workloads, stronger baselines, end-to-end fused integration, and real low-bit evaluation.

Why I am not convinced:
- If the contribution is "runtime entropy is a nice signal," then the paper should be much shorter and cleaner.
- If the contribution is "COREY improves real checkpoint serving," then the critical experiment is still missing.

Required bar for a stronger recommendation:
- Compress to a compliant main body.
- Remove non-essential side stories.
- Demonstrate benefit against matched static baselines on the same real checkpoint path where the scheduler is executed.
