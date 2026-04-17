# Stage 9 Revision Roadmap (2026-04-18)

Decision baseline:
- Current recommendation: Borderline Reject / Major Revision
- Target for Stage 10: convert the paper into a venue-compliant Concept & Feasibility submission with one closed-loop real-checkpoint story

Priority 0: Venue compliance
1. Cut the main body from 11 content pages to 9.
   - Move non-essential Hadamard/quantization detail and secondary diagnostics out of the main body.
   - Keep only the minimum theory needed to support the validated scheduler story.

Priority 1: Close the main empirical loop
2. Either pass the entropy-selected chunk into the real checkpoint scan path and report the resulting end-to-end latency, or explicitly narrow the paper so it no longer implies that this closed-loop gain has already been demonstrated.
3. Run the same real-checkpoint path with matched static baselines (`static-64`, `static-256`, and `static-512` if feasible) so the operational value of runtime entropy can be judged directly.

Priority 2: Show value beyond one-time profiling
4. Re-run the runtime policy with the principled calibration `H_ref = log K`, because the current appendix already shows that the default `H_ref = 8.0` is systematically conservative for `K = 256`.
5. Add at least one heterogeneous real workload mixture where different prompts genuinely map to different chunk buckets, or else rewrite the claims so COREY is presented as an automatic static tuner within one workload regime.

Priority 3: Strengthen positioning and evidence hygiene
6. Tighten the contribution type to "Concept & Feasibility" in the main narrative and submission metadata.
7. Strengthen or narrow the baseline story.
   - Preferred: add one stronger matched system baseline.
   - Minimal fallback: explicitly state that Pythia-410M is only an architectural sanity check and that the current contribution is intrafamily Mamba scheduling.
8. Make all appendix-facing artifacts self-consistent.
   - The separate appendix build should not contain unresolved references or an empty bibliography.
   - Replace estimated quantities with measured ones where they are central; otherwise label them even more prominently as estimates.

Expected Stage 10 output:
- A 9-page-compliant main body
- A single clearly validated runtime-scheduling claim
- A cleaner separation between validated scheduler evidence and prospective low-bit/Hadamard analysis

