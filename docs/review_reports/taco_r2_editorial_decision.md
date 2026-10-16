# Editorial decision - independent TACO assessment (round taco-r2), 2026-10-16

Inputs: five independent reviewers who read only the current manuscript (main.pdf, appendix.pdf, sources) and
the result files (`taco_r2_eic.md`, `taco_r2_r1_methodology.md`, `taco_r2_r2_domain.md`,
`taco_r2_r3_perspective.md`, `taco_r2_devils_advocate.md`), and a re-check of the taco-r1 review items
(`taco_r2_recheck.md`). Venue: ACM TACO, regular paper, at most 20 pages excluding references, double-anonymous.

## Decision: Major Revision

| Reviewer | Novelty | Method. | Experiments | Clarity | Significance | Recommendation |
|---|---|---|---|---|---|---|
| EIC | 45 | 74 | 68 | 38 | 45 | Major Revision |
| R1 methodology | 55 | 60 | 66 | 48 | 52 | Major Revision |
| R2 domain | 35 | 65 | 60 | 40 | 40 | Major Revision |
| R3 practitioner | 52 | 72 | 63 | 38 | 45 | Major Revision |
| Devil's advocate | 30 | 60 | 55 | 35 | 28 | Reject (resubmit refocused) |
| **Mean** | **43.4** | **66.2** | **62.4** | **39.8** | **42.0** | 4 x Major, 1 x Reject |

Against the taco-r1 panel (means 42.4 / 60.4 / 61.2 / 40.0 / 40.4): methodology improved by about 6 points,
the other dimensions are flat. No dimension regressed by more than 0.2 points.

Re-check of taco-r1: 47 of 66 items addressed, 11 partial, 8 declined with scoped claims, 0 not addressed;
acceptance criteria 5 of 8 met, 3 partial (A1 artifact anonymity, A5 latency-weighted a-priori headline, A8
supplement length and notation). Stage 11 gate: page limit, PDF and source anonymity, references, citations
pass; the artifact archive fails anonymity.

## Why not Accept or Minor Revision

1. **Significance and novelty (all five).** For data-oblivious kernels a shape-only optimum is expected; the
   lemma is elementary; the refuted design is one unreviewed preprint. The positive production-kernel result
   is small (1.0 to 5.9% of scan time in isolated CUDA-graph replay) and its recoverable part has wide
   intervals (GDN on the RTX 3090: 0.45% [-0.90, 1.78]).
2. **The steep chunk effect belongs to the in-house Triton kernel (all five).** It is 1.23x the reference
   scan, its launch table couples the chunk with grid size and warps and was chosen on another GPU, and its
   "default 256" is borrowed. A free length rule (clip(nextpow2(L), 32, 1024)) already reaches 5.9 / 7.8 /
   10.7% per call. With channel block and warps fixed, the optimum barely moves on two GPUs (R2, short
   timings; to be confirmed under the full protocol).
3. **No engine-level or data-centre evidence for the device-bound claims (EIC, R2, R3, DA).** Per-shape graph
   replays are not how engines run prefill; packed steps, padded capture sizes and the scan's share of the
   forward pass are not measured; SSD and GDN under graphs are timed on two workstation-class GPUs only.
4. **The cost model does not pay on the paper's own numbers (all five).** Calibration 130 to 843 s against 4 to
   7 s to profile every key; a-priori regret depends on which of six selection protocols is reported (0.38 to
   6.2%; the reported one is the most pessimistic, and its latency-weighted loss on the RTX PRO 6000 equals
   the per-batch static chunk); simple rankers and MLPs tie or win on some GPU; transfer to another kernel is
   untested.
5. **Clarity (all five; mean 39.8).** Five or more timing conventions, several definitions of "default" and
   "best static", number-dense sentences, two figures in 20 pages, about 50 pointers into a 31-page supplement.
6. **Measurement reliability (R1, DA, EIC).** A launch-inclusive session that did not reproduce, unguarded data
   sets, a possibly virtualized and power-capped RTX 3090 whose round-to-round noise exceeds the shard effect,
   value invariance of SSD/GDN untested above their floor, H800 input effects (28 of 36 configurations, up to
   48%) not in the main text.

## Consensus strengths (keep)

Recurrence-preserving timing with output checks; held-out prompt-cluster evaluation with bootstrap, TOST and
permutation tests; the placement-versus-values and L2 interventions; the compiler-specialization bit in the
key; candid reporting of model failures; the autotuner-state and output-divergence controls; every one of
about 70 recomputed numbers matched the result files (two definitional differences: 93.5% vs 94% tied cells,
0.40 vs 0.38% between two tables).

## What would move the paper to Minor Revision

Text and offline (no GPU):
- Reframe around the robust findings: shape key with compiler specialization (stride/alignment signature),
  host-bound versus device-bound regimes, placement and L2 artifacts in tuning, eager-fitted models failing
  under graphs, and profiling-versus-modelling economics; condense the refutation of data-dependent dispatch
  to one subsection; present the in-house kernel openly as a stress case.
- Fair baselines everywhere: the free length rule, per-batch static chunk, length-bucket table; one
  definition of "default" per kernel; recoverable gains (default minus held-out table) with cluster-bootstrap
  intervals; noise floors next to every regret.
- One pre-specified a-priori protocol for the cost model, all six reported with latency-weighted regret;
  chunk effects at fixed launch configuration from the 184-configuration sweep; workload reweighting to the
  Azure/BurstGPT length and batch distributions.
- Recipe corrections: batch invariance (the key includes batch and packed length), 32-bit indexing failures,
  padding to a multiple of 16, synthetic inputs only above the floor; one convention per table; a
  claims-to-evidence table and a mechanism figure; supplement towards 20 to 25 pages.
- Artifact anonymity (scrub result-file metadata and directory names; never upload `paper/` as is: it holds
  author-named profile photos and prior-venue backups); confidential disclosure of the preprint to the editor.

New measurement (the decisive lever for significance):
- A shape-keyed chunk table inside a serving engine (vLLM or SGLang) with CUDA graphs, one Mamba-2 hybrid and
  one Gated DeltaNet hybrid, reporting time to first token, throughput and switching overhead, ideally on a
  data-centre GPU; or, at least, a full-model prefill captured as one CUDA graph, default versus table, in two
  sessions.
- Smaller: SSD/GDN value invariance above the floor in device time or graphs; a second RTX PRO 6000 session;
  hardware counters (occupancy, stalls, spills) for a few keys where the profiler is permitted.

## Assessment of the outcome

As submitted today the paper would most likely receive Major Revision at TACO, with a real risk of Reject
from a reviewer who weighs significance as the devil's advocate does. The text and offline items can be done
without GPU time and address clarity, fairness of baselines and most methodological points; they are unlikely
on their own to lift significance above the bar. Engine-level evidence that the device-bound chunk effect
matters end to end (or a clear demonstration that it does not, reframed as the finding) is what separates
Major Revision from Minor Revision.
