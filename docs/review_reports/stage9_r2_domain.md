# R2 Domain Review (Mamba / SSM Systems, 2026-04-18)

Recommendation: Borderline Reject

Scores:
- Novelty: 50 / 100
- Methodology: 53 / 100
- Experiments: 40 / 100
- Clarity: 63 / 100
- Significance: 44 / 100
- Overall: 50 / 100
- Confidence: 4 / 5

Strengths:
- The paper now positions itself narrowly around Mamba-1.x rather than over-claiming across the full SSM family.
- The kernel-level W1 benchmark is concrete and useful.
- The manuscript cites relevant SSM quantization and systems work.

Weaknesses:
1. The real comparison set is too weak for the claimed problem. The paper still lacks matched checkpoint-path comparisons against Static-256 and Static-512 on the same real workload where COREY is supposed to help.
2. The external baseline story is underdeveloped. Pythia-410M is presented as an architectural sanity check rather than a fair baseline, and no strong Transformer-side memory-management or FlashAttention-style reference system is actually compared on the target workload.
3. The workload coverage is too narrow to support the adaptivity story. Four LongBench tasks with 20 samples each, plus 80 prompts all mapping to bucket 256, suggest that the current paper mostly demonstrates stable calibration inside one regime.
4. The appendix still exposes incompleteness: the Mamba-2.8B `policy_static` row is pending, and one passive-hook perplexity entry is suppressed because it is unreliable.

Bottom line:
- I can believe that entropy is a useful scheduler signal for a Mamba kernel.
- I do not yet see evidence that this signal delivers a strong publishable advantage over simple per-model static profiling or over stronger systems baselines.
