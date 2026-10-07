# Revision Roadmap -- cycle tpds-r4 (opened 2026-10-07)

Manuscript: "Shape Sets the Chunk: A Compile-Informed Cost Model and a No-Gain Bound for Data-Dependent Dispatch of Chunked Linear-Recurrence Kernels" -- IEEE TPDS regular paper, double-anonymous, 14 formatted pages.
Decision: **Major Revision** (unanimous; panel mean 64.7/100). Sources: `docs/review_reports/tpds_r4_editorial_decision.md` (decision, arbitration, acceptance criteria A1-A10) and `docs/revision_suggestions.tex` (items M1-M12, N1-N8, V1-V6). Item IDs below are those of the review file.

Effort: (a) text only; (b) offline analysis of existing result files; (c) new GPU measurement. Host: CPU-only for (a)/(b); single GPU (named where the reviewers specified it); 2 GPUs for the tensor-parallel run. GPU-hours are the reviewers' estimates.

## 1. Prioritized action table

| # | ID | Action (short) | Effort | GPU-h | Host | Depends on | Serves |
|---:|---|---|---|---:|---|---|---|
| 1 | V1 | Anonymized availability link; verify logged-out; DOI at camera-ready | a | 0 | CPU | -- | A10 (blocks re-review) |
| 2 | V2 | Confirm nothing public carries "COREY", else generic name in review version | a | 0 | CPU | -- | A10 |
| 3 | V3 | Remove or re-target the AAAI reproducibility checklist; fix the "no" on statistical tests | a | 0 | CPU | -- | A10 |
| 4 | M1a | Rescope title/abstract/Contribution 1/Sec. VII opening/Sec. X to the study's Triton kernel; SSD/GDN "defaults suffice" as a finding; reword the "runtimes tune offline" premise | a | 0 | CPU | -- | A1 |
| 5 | M2a | Device-time re-timing of 2360 captured calls + calibration sweep on the RTX PRO 6000; refit; PRO 6000 rows in Table S-conventions | c | 0.5 | PRO 6000 | -- | A2 (required) |
| 6 | M2b | Three shuffled rounds per captured call, oracle on round medians, 370M on PRO 6000 and RTX 3090; re-derive Tables II-III and headroom | c | 0.6 | PRO 6000 + RTX 3090 | run with M2a on PRO 6000 | A2 (required) |
| 7 | M8a | Replace superseded headroom rows; held-out per-shape maximum column; provenance of 4.7 us; convention and decision cost named in abstract | b | 0 | CPU | M2b | A3 |
| 8 | M8b | Definitions paragraph for the three regret estimators; reconcile 23.4% vs 10.6%; latency-weighted/absolute form where it feeds Eqs. (7)-(8) | a+b | 0 | CPU | -- | A3 |
| 9 | M8c | Abstract/contributions carry mean and p95, three GPUs, both kernels, both policies, device-time result | a | 0 | CPU | M2a, M8a | A3 |
| 10 | M2c | Tie shares under 2%/5% equivalence margins; between-round CI if overlap kept; fix "conservative" | b | 0 | CPU | M2b | A2 |
| 11 | M4a | Separate cross-device (14-49%) from unsharded-to-sharded (63%) loss; TP result as "no retuning at 0.7-15%" with retuning cost; simulated trace study stated; reframe or drop "mixed GPU fleets"/"LLM serving" unless M4b/M4c | a | 0 | CPU | author decision D-2 | A5 |
| 12 | M5c | Lead abstract/conclusion with form-blind numbers (H800, width transfer, shards); label 0.9/3.2 as form-selected; mean and p95 | a | 0 | CPU | M5a | A4 |
| 13 | M5a | Clean form selection (calibration + leave-one-key-out, or freeze on PRO 6000 and test on RTX 3090; select on one GPU, test on the other two) | b | 0 | CPU | existing timings; redo after M2a if device-time fit changes | A4 |
| 14 | M5b | Baseline table: full model, 3-param restriction, nearest key, log-linear interpolation, length tree, GBT + MLP on calibration timings, TFLA expression; three GPUs, shards, cross-GPU | b | 0 | CPU | M5a | A4 |
| 15 | M5d | Three instantiations as one framework table; term selection as a procedure | a | 0 | CPU | M5a | A4 |
| 16 | M7c | Replace "cannot help" by the quantitative bound at zero decision cost; information result leads in Sec. IV and abstract | a | 0 | CPU | -- | A7 |
| 17 | M7a | Introduction: external proponent of value-dependent dispatch, or state none; bound as closing a design question | a | 0 | CPU | -- | A7 |
| 18 | M7b | Sec. III to half a column; Eqs. (1)-(2), variants, Fig. 4 saturation material to supplement | a | 0 | CPU | -- | A7, V5 |
| 19 | M7d | Bootstrap CIs on Tables III-IV; exact-key stratified permutation test; conditional MI given exact key; GBT/depth-6 predictors; headroom leads Sec. VII-C | b | 0 | CPU | M2b (optional refresh) | A7 |
| 20 | M3c | Launch-bound default of the work-term model to 256; re-report Mamba-2 regret; restate the eager boundary for the 320 us floor | b | 0 | CPU | -- | A6 |
| 21 | M3d | State the Triton do_bench L2-flush discrepancy (VI-D, VII-F); eager-mode scope of Mamba-2 claims if M3b not run | a | 0 | CPU | author decision D-3 | A6 |
| 22 | M6a | Deployment-units table: ms per request per kernel/length/batch; share of prefill and total GPU time per trace and mode; SSD/GDN gain ~0 | b | 0 | CPU | M2a (device time preferred) | A8 |
| 23 | M6b | Model-vs-profiling break-even per key and share of trace calls below it; startup cost of profiling the bucket grid vs 8-14 min calibration; address the headroom-vs-model-regret tension | b | 0 | CPU | -- | A8 |
| 24 | M6c | Deployment recipe box; top-2 hybrid as the recommendation; niche stated | a | 0 | CPU | M6b | A8 |
| 25 | M6d | p95/p99 regret of top-2/top-3 on three GPUs and shard widths | b | 0 | CPU | M2b | A8 |
| 26 | M11a | Add and position against the 13+ references (compile-informed models, cross-device, shape-generic, learned, Triton docs, chunkwise-form origin, online autotuning, Mooncake, Jamba, Qwen3-Next); fix the Ansor sentence | a | 0 | CPU | -- | A9 |
| 27 | M11b | Scope the Sec. V novelty sentence; compare top-k profiling with Triton perf_model/top_k; one sentence on what Eq. (5) adds | a | 0 | CPU | M11a | A9 |
| 28 | M1c | Chunk-taxonomy table; rewrite Sec. I para. 1 | a | 0 | CPU | -- | A1 |
| 29 | M1d | GDN chunk 64 = library default and design limit; not tested beyond | a | 0 | CPU | -- | A1 |
| 30 | M9b | Delete "at least as chunk-sensitive as ours"; stock-dispatch statement (VII-E, Contribution 1, IX) | a | 0 | CPU | required unless M9a | A1 |
| 31 | M10c | Padded budget-step numbers (42/93%) to supplement; production key (packed length, sequences, alignment); distinct prompts per batched call stated | a | 0 | CPU | -- | A5, A3 |
| 32 | N1 | Reconcile the nine number pairs; add missing supplement entries | a+b | 0 | CPU | M8a | A10 |
| 33 | N2-N4 | Bibliography duplicate/uncited/booktitle; eqref, underfull boxes, duplicate labels, sec:res_datacentre; protocol summary table, seeds, mu formula, 48 GB board, Kendall | a | 0 | CPU | -- | A10 |
| 34 | N5-N8 | README mapping and stale statements; Table V histogram rows; subnormal/specialization/autotune/mamba_ssm statements; "data-centre GPUs", keyword, prompt reuse | a | 0 | CPU | -- | A10 |
| 35 | M12a-c | Compress VII-D/VII-F/VII-G; secondary numbers to tables; regret-vs-budget figure; Table VI H800 rows; 14-page check | a | 0 | CPU | all text items | A10, V5 |
| 36 | V4-V6 | Idiap identifier check; page limit; pdfinfo metadata check | a | 0 | CPU | M12 | A10 |
| 37 | M3a | Cold-L2 re-timing of 12 keys x 6 chunks (+ captured calls L >= 2048) on the RTX PRO 6000; best chunk, regrets, spread, headroom warm vs cold | c | 1-2 | PRO 6000 | author decision D-3; run after M2a | A6 |
| 38 | M10b | Capture >= 50 calls of the unaligned bucket-2048 batch-4 key on the RTX 3090; re-run replay; per-row count of thin keys | c | 0.2 | RTX 3090 | -- | A5 |
| 39 | M3b | CUDA-graph capture of SSD (370M, 2.7B), GDN and Mamba-1 at 12 keys / captured calls; warm and flushed; re-derive Mamba-2 rows and 1.4B/2.7B headroom | c | 3 | RTX 3090 | author decision D-3 | A6 |
| 40 | M9a | Fixed-thread reference-scan instantiation; re-sweep 2360 calls; spread, oracle, model regret next to stock variant in Table IV | c | 1-2 | RTX 3090 | author decision D-4 | A1 |
| 41 | M10a | Varlen entry points on ~200 packed steps at T = 512/2048/8192, every chunk, three rounds; packed-length key; proxy error | c | 3 | single GPU | author decision D-5 | A5 |
| 42 | M7e | Zero-sync / lagged-readback variant in the routed study; critical-path cost at three lengths, batch 1 and 16; fix the learned-table control | c | <1 | RTX 3090 | author decision D-6 | A7 |
| 43 | M10d | Batched routed configuration (batch 8 x 2048), 30 rounds; optionally under CUDA-graph capture (+1 h) | c | 1-2 (+1) | RTX 3090 | author decision D-6 | A5 |
| 44 | M1b | One public chunked kernel with a wide chunk range (TFLA mLSTM / FLA) at 370M-2.8B layouts, L 512-8192, batch 1/4/16, three rounds, model fit | c | 4-6 | RTX 3090 | author decision D-1 | A1 (keeps general title) |
| 45 | M4d | Width sweep at unsharded widths 8192/16384 and t = 2/4/8 shards | c | 2 | single GPU | author decision D-2 | A5 |
| 46 | M4b | One measured 2-way TP prefill run (1.4B/2.8B), three lengths, two batches, 30 rounds; table vs model vs per-shard oracle; all-reduce included | c | 4-8 | 2 GPUs (2x RTX 3090 or 2x H800) | author decision D-2 | A5 |
| 47 | M4c | vLLM replay of one hour of the Azure trace, three dispatch policies, TTFT/TPOT/throughput at three loads; once at TP = 2 (subsumes M4b; also serves A8) | c | 8-10 (+2 engineer-days) | single GPU + 2 GPUs | author decision D-2 (alternative to M4b) | A5, A8 |
| 48 | N6(c) | Triton-fused histogram for Table V bandwidth-floor row | c | 0.5 | single GPU | optional | A10 |
| 49 | M11c | Mooncake trace replay through the existing simulator | b | 0 | CPU | optional | A9 |
| 50 | M12d | Main-text paragraph on the placement finding | a | 0 | CPU | optional, page budget | A10 |

GPU totals: required (rows 5-6) about 1.1 GPU-h. All optional (c) rows with M4b: about 20-31 GPU-h; with M4c instead of M4b: about 24-33 GPU-h. All (c) rows including the required ones: about 21-32 GPU-h (25-34 with M4c).

## 2. Suggested execution order

**Phase 0 -- venue compliance (same day, CPU).** V1, V2, V3. Nothing is sent for re-review before V1 is verified in a logged-out browser.

**Phase 1 -- text that does not wait for data (CPU).** M1a, M1c, M1d, M9b, M7a, M7b, M7c, M4a (text branch; revisit once D-2 is decided), M3d, M8b (definitions), M10c, M11a, M11b, N2-N4, N5-N8. Keep a running page count (V5): M7b and the VII-D/VII-F/VII-G compression of M12a are the only sources of space.

**Phase 2 -- offline analysis on existing result files (CPU, parallelizable).** M5a -> M5b -> M5d -> M5c; M3c; M6b -> M6c; M6a (can start on launch-inclusive data, refreshed after M2a); M7d; N1 (items not depending on M8a); M11c if selected.

**Phase 3 -- GPU work, grouped by host.**
- *RTX PRO 6000, one session (required first):* M2a and M2b (PRO 6000 part) in one run (device time, three shuffled rounds); then M3a if funded (the L2-overwrite intervention already exists). About 1-3 GPU-h.
- *RTX 3090, one session:* M2b (RTX 3090 part, 0.3 h) and M10b (0.2 h) first because M8a, M2c, M6d and the replay depend on them; then, in the order the author funds them, M3b (3 h), M9a (1-2 h), M10a (3 h), M7e (<1 h), M10d (1-2 h), M1b (4-6 h), M4d (2 h), N6(c) (0.5 h). Prepare each script and smoke-test it locally while the previous run is on the GPU.
- *Two GPUs (2x RTX 3090 or 2x H800), one session:* M4b (4-8 h) or M4c (8-10 h plus engineering); M4c should be prepared in full (engine patch, policies, trace loader) before the 2-GPU window opens.

**Phase 4 -- backfill and integration (CPU).** M8a, M2c, M6d, M5a/M5b refresh on device-time data, N1 remaining items, M8c, M5c, then M12a-c (figure, tables, page budget), then V4-V6 and a full rebuild; final consistency pass of abstract <-> contributions <-> Sec. X <-> tables.

**Phase 5 -- response letter.** One entry per item M1a-M12d, N1-N8, V1-V6 with the location of the change and, for every (c) item not funded, the (a)/(b) fallback applied and the reason.

## 3. Decisions the author must make

- **D-1 (scope, A1).** Fund M1b (4-6 GPU-h, one public kernel with a wide chunk range) to try to keep the general title and claim, or rescope the title, abstract, contributions and conclusion to register-tiled Triton scans and present "defaults suffice on SSD/GDN; the stock rule is within 6.6% on the reference scan" as the thesis (R2 calls the latter "still publishable but must be stated"). Rescoping M1a is required in both branches.
- **D-2 (parallel dimension, A5).** Choose one: (i) M4b, one measured 2-way TP run (4-8 GPU-h on 2 GPUs; the EIC's preferred option); (ii) M4c, the vLLM trace replay plus TP = 2 (8-10 GPU-h plus about two engineer-days; R3's option, which also answers A8 end to end); (iii) no multi-GPU run, and the abstract/keywords reframed to single-device shapes produced by parallel deployments with "mixed GPU fleets" and "LLM serving" dropped. Optionally add M4d (2 GPU-h) in any branch. M4a's wording changes are required in all branches.
- **D-3 (timing regime, A6).** Fund M3a (cold-L2, 1-2 GPU-h on the PRO 6000) and/or M3b (CUDA-graph capture, 3 GPU-h on the RTX 3090), or state the warm-L2 and eager-mode scope at claim level (M3d). M3c (launch-bound default) is done regardless.
- **D-4 (reference scan, A1).** Fund M9a (fixed-thread instantiation, 1-2 GPU-h) or delete the "at least as chunk-sensitive" claim (M9b).
- **D-5 (production call, A5).** Fund M10a (varlen entry points, 3 GPU-h) or move the padded budget-step numbers to the supplement and state the proxy's limitation (M10c).
- **D-6 (optional routed-study extensions).** M7e (zero-sync variant, <1 GPU-h), M10d (batched routed configuration, 1-2 GPU-h, +1 h under CUDA graph), N6(c) (fused histogram, 0.5 GPU-h). None is required for acceptance; M7e is the cheapest way to retire the cost-side objection entirely.
- **D-7 (hosts).** Which GPUs are available in this cycle: the PRO 6000 session (required for M2a) is the critical path; the RTX 3090 session can run in parallel; the 2-GPU window is needed only if D-2 is (i) or (ii).
- **D-8 (anonymity, V1/V2).** Whether the Code Ocean capsule can be switched to a private peer-review link or must be withdrawn from the review package, and whether anything public carries the name COREY.

## 4. Acceptance criteria served (from the decision letter)

A1 scope attributed and production kernels stated; A2 PRO 6000 numbers in device time with a multi-round oracle; A3 every abstract number tabulated and estimators defined; A4 clean form selection and full baseline table; A5 parallel claims match the measurement; A6 timing-regime caveats measured or stated and the launch-bound default fixed; A7 quantitative bound replaces "cannot help", information result leads, entropy test strengthened; A8 deployment units, break-even, recipe, tail regret; A9 positioning; A10 venue MUST FIX, 14 pages, mismatches reconciled.
