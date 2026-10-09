# tpds-r4 independent re-check after the precheck fixes and the compsoc template (2026-10-10)

Fresh agent without context: no access to the ledger, progress notes, design notes, the precheck
report or git history. Paper as built 2026-10-10 08:01 (paper commit `e6e23e4`, compsoc 10pt,
14 pp main, 25 pp supplement). Line numbers: M = main.tex, A = appendix.tex.

**Recommendation: Minor Revision.** Every one of the more than 30 traced headline numbers matches
the result files. The template is correct: `[10pt,journal,compsoc]`, Palatino 9.5 pt, abstract and
keywords in `\IEEEtitleabstractindextext`. The abstract has 246 words; there are no undefined
references and no visible overfull boxes; the PDF metadata is clean.

## Review items not fully addressed

| Item | Missing |
|---|---|
| M1a/A1 | "1.23x the reference scan" absent from abstract and Contribution 1; abstract omits that the TFLA default loses only 1.4% |
| M2a/A2 | abstract's 2.7-4.5% is launch-inclusive single-block and unlabeled |
| M3b | graph replays one round; Mamba-2 rows of Tables 4/5 not labeled eager-only or paired with graph results |
| M4a/M8c/A5 | abstract lacks "one GPU, synthetic inputs" for shard widths, retuning cost 0.03-0.05 s per key, reference-scan width insensitivity |
| M5c | Conclusion gives no p95; criterion selectivity (P1) |
| M8a | abstract does not name the comparators (host sync 20 us, device reduction 7.3-42 us); the device-time bucket headroom on H800 and the large checkpoints (17-20 us) exceeds the 7.3 us device reduction at L=2048 |
| M12a | M:986-1012 and M:1234-1248 hold about 22 and 21 numbers |
| N1.5 | no sentence reconciling 23.4% (chunk 256, per-call) with 10.6% (chunk 512, latency-weighted) |
| N3 | visible gap above "8 Discussion" on p. 12 |
| N4 | Table S1 has no rows for the CUDA-graph, TFLA, fixed-thread and thin-key runs or the PRO 6000 device-time calibration |
| N5 | README says E22 used three rounds; data and A:2152 say one |
| V1 | capsule link (intentional, author decision 2026-10-08) |
| V4 | Idiap identifier (authors not affiliated, recorded 2026-10-07; the reviewer could not see this) |
| A3 | untabulated: "up to 65%" (H800 per shard), "14 to 49%", bucket headroom 0.8-3.8%, device-time batch-1 26%/28% |

## Number mismatches

- Cross-GPU parameter transfer: the recipe (M:1490) says 2.5-5.4%; the supplement and the file
  (`cost_model_baselines_r4_summary.json` cross_gpu_baselines) give 0.87-5.44%.
- RTX 3090 input effects: A:843-844 says "no input effect exceeds 4.3%", but the file has a 19.0%
  non-significant effect; 4.3% is the largest significant one.

## Inconsistencies

1. Parameter transfer 2.5-5.4% (M:1490) vs 0.87-5.44% (A:1270-1272).
2. RTX 3090 input effects (A:843-844 vs M:1379 vs data).
3. M:987-989 says the form selects best "at the H800 shard widths", but M:1151-1152 and
   A:1012-1014 say the d=2048 table does better there (2.2%).
4. The abstract (M:52-55) and M:1301-1302 call the SSD 0.24%, GDN 154/155 and 0.09 ms figures
   "eager", but they come from device-time timings above the floor.
5. "Equivalent within 0.5 points" (M:166) vs two "inc." verdicts (A:538-539); "no information
   beyond its permutation baseline" (M:913-915) vs 0.012 bits at p=0.001 (A:528).
6. M:1305 attributes 0.48 ms to "the routed study", but it is the kernel-level estimate x48; the
   routed paired measurement is +0.64+-1.16 ms.
7. E22 rounds: the README says three, the paper and data say one.
8. H800 reference-scan shard data exist (e12_analysis_h800.json) but are not reported.

## New problems

- **P1 (major).** The headline 2.7-4.5% (p95 14-26%) is the most favourable clean-selection
  criterion. Leave-one-shape-out gives 7.25/62.5 on the H800, and the two-other-GPUs criterion with
  the three-parameter candidate gives 4.48/50.1 on the PRO 6000. Table S22 omits that row. Report the
  range across criteria, or pre-declare one criterion.
- **P2 (major, scope).** The cost model is supported only on the two Mamba-1 kernels: TFLA 7.5% vs
  a default of 1.4%; reference scan at 32 threads 49.5%; work-term model under graphs 12-22% vs
  defaults 16-22%. The abstract omits the TFLA default and the 49.5%.
- **P3 (moderate).** The CUDA-graph claim rests on one GPU and one round:
  - relative regret is dominated by calls under 100 us;
  - in absolute terms it is 0.10 / 0.57 / 1.42 ms per forward pass;
  - a different single chunk recovers most of it (SSD 370M best static 4.7% vs default 18.2%);
  - "serving engines replay captured CUDA graphs" (A:2148) has no citation.
- **P4 (minor).** "0.05 to 0.37 us over an exact-shape table" holds at tuned shapes only; over all
  calls it is 14.6 us (RTX 3090 device time).
- **P5 (minor).** H800 reference-scan shards: unsharded table 6.6% pooled, up to 15.3% per shard.
  This weakens "width-insensitive (2.2%)".
- **P6 (minor).** In the TFLA comparison the batch table is in-sample while the model is
  leave-one-shape-out.
- **P7 (minor).** "14 to 49%" are batch-1 single-block launch-inclusive values; this is not stated.
- **P8 (minor).** "MLP on shape and chunk" (M:994-996): the basic learner includes compile facts
  (A:1260-1263).
- **P9 (minor).** GDN 154/155 covers 9% of the 1770 calls (those above the device-time floor).
- **P10 (minor).** "Exact-shape table" is held out in Table 5 and in-sample "all profiled" in
  Table 6.

Camera-ready note: biographies will push the paper to 15 pages.
