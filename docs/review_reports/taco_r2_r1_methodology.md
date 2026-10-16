# TACO Review -- Reviewer 1 (Methodology and Statistics)

**Manuscript:** "Shape Sets the Chunk: Selecting and Keying the Chunk of Chunked Linear-Recurrence GPU Kernels" (new regular paper, double-anonymous)
**Material reviewed:** main.pdf / main.tex (25 pp. incl. references), Supplementary Material appendix.pdf / appendix.tex, the released result files and analysis code.
**Reviewer focus:** timing protocol, oracle and tie definitions, regret summaries, cross-validation and held-out protocols, bootstrap and equivalence inference, information-theoretic tests, cost-model fitting and a-priori term selection, and whether the claims hold at the stated convention and uncertainty.

---

## 1. Summary

The paper asks how a runtime should choose the chunk size of chunked linear-recurrence GPU kernels: a custom Triton Mamba-1 selective scan, the reference Mamba-1 CUDA scan rebuilt with a runtime chunk, Mamba-2 SSD and FLA's Gated DeltaNet. Every measurement varies the chunk inside a single recurrence-preserving launch. The study covers RTX PRO 6000, RTX 3090 and (Mamba-1 only) H800 GPUs, about 2360 captured prefill calls per checkpoint, and several timing conventions: launch-inclusive, device time behind a GPU-side spin, and CUDA-graph replay. It makes three claims:

1. **What the key must contain.** The optimal chunk is a function of the shape key: dimensions, compiler-specialization bits and device. Activation statistics, including the entropy used by a prior scheduler, add nothing of practical value. An elementary lemma bounds any data-dependent policy's gain by the held-out residual of a shape-keyed table, which is below one host synchronization at every tuned shape of the steep-curve scans.
2. **Where the chunk matters.** In host-bound execution the library defaults are adequate. In device-bound execution (CUDA graphs) the defaults leave 1.0 to 5.9% of scan time.
3. **When a model pays.** A compile-informed cost model calibrated once per GPU selects the Triton scan's chunk with 3.0 to 6.2% mean regret when its terms are chosen a priori. Profiling every enumerable key is still far cheaper.

The empirical work is extensive and unusually transparent. Every number I recomputed from the released files matches the manuscript (Section 7). The core negative result (entropy carries no practically useful information beyond the exact shape) is well supported. The weaknesses are mainly these:

- Several positive headline claims are stated against a noisy per-call oracle, without intervals, and without end-to-end validation in the regime they are about.
- The a-priori accuracy of the cost model is one of six selection protocols the authors computed.
- Some data sets come from a measurement platform whose round-to-round or session-to-session variation is as large as the effects being claimed.

---

## 2. Scores

| Criterion | Score (0-100) |
|---|---|
| Novelty | 55 |
| Methodology | 60 |
| Experiments | 66 |
| Clarity | 48 |
| Significance | 52 |

**Confidence:** 4 / 5. I read both PDFs' sources in full, re-ran parts of the analysis on the released files, and ran new analyses with the authors' own helpers. I did not re-time anything.

---

## 3. Recommendation: **Major Revision**

**Why not Reject.** The main negative result is robust:

- It replicates across three GPUs, two timing conventions, four kernels and larger checkpoints.
- It is evaluated on held-out prompt folds.
- It holds under a battery of tests: paired cluster-bootstrap differences, an exact-key permutation test, conditional mutual information, and deeper learners.
- I could reproduce every recomputed table cell exactly.

The deployment recipe also follows from evidence the paper contains. This is a sound measurement study with a useful negative result for TACO's readership.

**Why not Minor Revision.** Three of the paper's positive quantitative claims cannot be accepted as stated without new analysis, and partly new measurement:

- (a) "The defaults leave 1.0 to 5.9% of scan time that a shape-keyed table recovers" (abstract, contributions, conclusion). It is measured against a noisy per-call oracle on isolated single-kernel graph replays. It has no interval and no end-to-end graph-captured validation. For one of the six kernel/GPU cells, the amount a realizable table actually recovers is not distinguishable from zero (Major 1).
- (b) "3.0 to 6.2% mean regret with terms chosen a priori." This rests on one of six selection protocols in the released analysis, which give 0.4 to 6.2% depending on the protocol. On the only form-blind GPU, the full model is not distinguishable from a three-parameter restriction (Major 3).
- (c) Several data sets come from a platform whose cross-round or cross-session variation is as large as the effects being reported. Examples are the RTX 3090 tensor-parallel sweep and the launch-inclusive PRO 6000 sessions (Major 2). Part of the value-invariance foundation is also measured in a launch-dominated regime (Major 5).

These issues require restated claims, sensitivity analyses and two or three targeted GPU runs (hours, not weeks). That is the scope of a major revision, not a rejection.

---

## 4. Strengths

- **S1. Recurrence-preserving timing.** The paper correctly identifies and avoids the split-call chunk-timing pitfall (Section 5). Every chunk is validated against an fp32 reference on a length that is not a multiple of any chunk (Table S4).
- **S2. Multiple timing conventions, defined explicitly.** Launch-inclusive, device time and CUDA-graph replay are defined in a conventions box. A three-criterion tie table (Table S3) and a per-experiment protocol table (Table S1) make the measurement conditions auditable.
- **S3. The held-out design is sound.**
  - Folds are assigned per prompt.
  - Batched calls never straddle folds.
  - Exact-shape tables are chosen on other folds.
  - Covered and fallback calls are reported separately.
  - Intervals use a prompt-cluster bootstrap over connected prompt clusters.
- **S4. The headroom argument is well posed.** The lemma is elementary but turns an open-ended question ("can any statistic help?") into a measurable upper bound. The authors also try to de-bias it: split-half, null-with-noise, and cross-round estimates (Table S37).
- **S5. Placement controls.** Designs A/B/C plus a warm vs flushed L2 test show that the large "input effects" are buffer-placement effects. This is a valuable methodological contribution in its own right (Table S10, Section S1.3).
- **S6. Transparency and reproducibility.** Result files carry seeds, versions and command lines, and the analysis code reproduces the paper's numbers (Section 7). The paper also reports many of its own weaknesses openly:
  - the cost model's fit is a local optimum (50 random starts find a better one);
  - the launch-inclusive sessions are not reproducible;
  - the oracle includes timing noise.

---

## 5. Major issues

### Major 1. The device-bound headline compares against a noisy per-call oracle, has no intervals, and is not validated end to end; for one cell the recoverable gain is not significant

**Location:** Abstract ("the defaults leave 1.0 to 5.9% of scan time on the table"); Contributions bullet 2; Section 6.5 and Table 7; Fig. 2; Conclusion ("the production defaults leave 1.0 to 5.9% of scan time that a shape-keyed table recovers").

**Problem.**

- The 1.0 to 5.9% figure is the latency-weighted (LW) excess of the default over the *per-call oracle*. That oracle is the minimum over chunks of round medians, which is biased low by noise.
- What a deployable table can recover is the default's LW loss minus the held-out table's LW loss, and that difference has no interval anywhere in the paper.
- Every graph measurement captures one kernel per graph and replays it 50 times back to back on resident inputs. This leaves out the real producer/consumer context of a graph-captured forward pass: L2 state left by the preceding kernels, interleaved kernels, and the inter-kernel launch gaps that remain inside a graph.
- The only end-to-end study (Section 6.6) is eager, on the reference CUDA scan, on one RTX 3090. That is a host-bound setting, where chunk effects are hidden at 256 and 976 tokens.

**Recomputation.** I used the authors' `round_median_view`, a held-out exact-shape table with bucket fallback as in `cv_exact_headroom`, and a 2000-replicate bootstrap over the 42 connected prompt clusters. The recoverable LW share (default minus held-out table) is:

| Graph replay (warm L2, 3 rounds) | Default LW (paper) | Held-out table LW | Recoverable LW [95% CI] |
|---|---|---|---|
| SSD 370M, RTX 3090 | 2.80% (2.8) | 0.01% | 2.79% [1.51, 5.30] |
| SSD 370M, PRO 6000 | 1.75% (1.7) | 0.03% | 1.72% [0.89, 3.57] |
| SSD 2.7B, RTX 3090 | 2.20% (2.2) | 0.03% | 2.17% [1.15, 4.61] |
| SSD 2.7B, PRO 6000 | 5.85% (5.9) | 0.46% | 5.39% [2.77, 7.63] |
| GDN 1.4B, RTX 3090 | 1.02% (1.0) | 0.57% | **0.45% [-0.90, 1.78]** |
| GDN 1.4B, PRO 6000 | 3.19% (3.2) | 0.25% | 2.94% [1.88, 4.93] |

The ms-per-forward-pass values of the held-out table (0.00, 0.00, 0.02, 0.10, 0.06, 0.01) match Table 7's "Tree / table" columns.

So:

- The lower end of the abstract's range (GDN on the RTX 3090) is not a gain that a shape-keyed table delivers: the recoverable share is 0.45% and its interval includes zero.
- The other intervals are wide (a factor of about 2 to 3), because only 42 clusters carry the inference.
- The statement that a table "recovers to within 0.7%" refers to per-call means, not to scan time.

**Fix.**

- (i) Restate the headline as recoverable scan time (default minus held-out table), with cluster-bootstrap intervals, in Table 7, the abstract and the conclusion.
- (ii) Run at least one end-to-end validation in the regime the claim is about. Capture a full-model prefill forward as a CUDA graph at fixed shapes (Mamba-2 370M and 2.7B, GDN 1.4B), once with the default chunk and once with the held-out table. Measure the prefill latency on both GPUs, in at least two sessions.
- (iii) Report per-forward-pass and end-to-end percentages next to "% of scan time".

**Effort:** offline analysis (i, iii); new GPU measurement (ii).

---

### Major 2. Measurement reliability: single sessions, unreproducible launch-inclusive sessions, and an RTX 3090 tensor-parallel sweep dominated by cross-round variation

**Location:** Section 5.3 and the conventions box; Section 6.5 ("Tensor-parallel widths"); Section 6.8; Tables S1, S2, S24, S28, S37; Limitations.

**Problems.**

- **Launch-inclusive sessions do not reproduce.** A second PRO 6000 session moved the shape tree and the cost model from 0.7 / 0.9% to 4.7 / 4.9%, and the median oracle latency from 60 to 98 µs (Section S8). Host CPU, frequency governor, host load and board power limit were not recorded (Section S1). Yet several conclusions rest on single-session launch-inclusive data:
  - "library defaults are adequate when kernel launches dominate";
  - the Mamba-2 rows of Table 4;
  - the reference-scan Table 6;
  - the H800 launch-inclusive results;
  - the value-invariance table on the PRO 6000 (Major 5).
- **The RTX 3090 tensor-parallel sweep is noise-dominated.** In the authors' stored analysis (`taco_report_numbers.json`, M7a), a table retuned at the *exact shard shape* and scored on another round loses 14.87% on the Triton kernel. Paired against it, the cost model is *better* by -8.35 points [-12.38, -4.59]. On the reference scan it is better by -4.35 [-7.06, -1.86]. A model cannot beat an exact-shape table except through scoring noise. Cross-round variation on this platform therefore exceeds the chunk effects that the sweep claims to measure, including "a table tuned at the unsharded width loses 36.8%". The text notes this in passing ("whose rounds differ by more than the chunk effect there"), but Section 6.5, Contribution 2, Table S24 and the shard columns of Table S19 still report the RTX 3090 numbers at face value. On the H800 the retuned table loses 0.75%, so only the H800 supports the shard-width claim.
- **The RTX 3090 platform itself.** It is a non-stock 48 GB board in a cloud container, where virtualization or time-slicing "cannot be excluded". The software power cap was active in 44 to 92% of readings in several sweeps. All of the following are RTX 3090 only:
  - the routed end-to-end study;
  - the reference-kernel sweep;
  - the larger checkpoints;
  - the production kernels at larger widths;
  - the TFLA kernel.
- **Headline device-time results are single-session.** The PRO 6000 device-time captured-call sweep and the PRO 6000 graph replays each come from one session. Cross-session stability in device time is shown only on the RTX 3090 (single-block r3spin vs three-round r4spin). There it holds, which is encouraging, but it is not shown for the GPU that carries the headline numbers.
- **Unguarded data and mixed protocols.** Many data sets were timed without the exclusive-GPU guard (Table S1), including the value-invariance table, the H800 captured calls and the cost-model calibration on two GPUs. The selector table (Table 5) mixes three-round medians (PRO 6000, RTX 3090) with single blocks (H800). Its RTX 3090 column also mixes two sessions: most rows are on the three-round r4spin data, while the a-priori row (6.23%) is on the single-block r3spin data.

**Fix.**

- (i) For every data set, report a noise floor next to the regrets: the cross-round or cross-session regret of the per-shape best chunk (the authors already compute this for the TP sweep and in Table S37). Interpret effects only above it.
- (ii) Repeat the PRO 6000 device-time captured-call sweep and the PRO 6000 graph replays in a second, independent session (ideally a different instance) and report the session-to-session change of each headline number.
- (iii) Demote the RTX 3090 TP numbers to the supplement as uninformative, and rest the shard-width claim on the H800, or re-run the 3090 sweep with more rounds and the exclusive-GPU guard.
- (iv) Evaluate every row of Table 5 on the same data set per GPU.
- (v) Record host CPU model, governor and load in future runs.

**Effort:** text and offline (i, iii, iv); new GPU measurement (ii; about one day of GPU time).

---

### Major 3. Cost-model accuracy "with terms chosen a priori" depends on which of six selection protocols is reported; the form-blind evidence is one GPU and does not separate the model from simpler forms

**Location:** Abstract ("3.0 to 6.2% mean regret when its terms are chosen without the target GPU's calls"); Section 4 ("Model form and data", "What transfers"); Section 6.4; Table 5; Tables S14 to S19; Conclusion.

**Problems.**

- **Forking paths.** The released analysis (`taco_model_baselines.json`, block M4c) evaluates six a-priori protocols: two candidate sets (ablation12, ablation12+work_only) times three criteria (LOGO on the other GPUs' captured calls, LOGO on calibration LOSO, target-calibration LOSO). The paper reports only one (ablation12+work_only, LOGO captured calls). Device-time results of all six:

  | Target | Primary (reported) | All six protocols |
  |---|---|---|
  | PRO 6000 | 4.16% (work_only) | 0.38 to 4.16% (2.24, 0.38, 0.56, 4.16, 0.38, 0.56) |
  | RTX 3090 | 6.23% (drop_shuffle) | 0.88 to 6.23% (6.23, 2.49, 1.74, 6.23, 2.49, 0.88) |
  | H800 | 3.01% (add_l2_miss / add_cross_warp) | 2.14 to 3.01% |

  The *target-calibration LOSO* criterion selects the form by cross-validation on the target GPU's own calibration timings, without its calls. That is the procedure the "What transfers" paragraph recommends ("calibrate at about twelve shape keys, and choose the terms without the target calls"), and it yields 0.56 / 0.88 / 2.14%. The reported protocol is the most pessimistic one on the PRO 6000, so this is not cherry-picking in the authors' favour. It does mean, however, that "the method's accuracy" is not a well-defined quantity in the paper. The tie rule ("fewest parameters within 0.1 points") alone moves the PRO 6000 between 0.38, 0.56 and 4.16%.
- **Form-blind evidence is n = 1.** On the H800, the frozen eight-parameter model (3.01%) is not distinguishable from the three-parameter restriction: paired -0.53 points [-1.35, +0.60], Table S15. It is beaten by the structural MLP (0.57%; 1.1% over five seeds). Across the three GPUs, the full form is significantly better than the three-parameter restriction only on the PRO 6000 (-3.80 [-4.78, -2.89]). That is one of the two GPUs on whose calls the terms were selected. On the RTX 3090 it is significantly worse (+2.66 [+2.12, +3.17]). The specific terms of Eq. (3) therefore have no out-of-sample support so far.
- **Fit fragility.**
  - The 12-key bootstrap gives PRO 6000 [0.38, 2.26] with the point estimate at the lower edge of its own interval, so the stored fit is a favourable draw.
  - The RTX 3090 fit is a local optimum: 50 random starts give a 16% lower objective and 1.7% regret.
  - An ordinary least-squares fit gives 2.1 / 1.8 / 3.0%.
  - 2208 of the 2298 calibration timings are brief autotuner-style timings (3 warmups, 10 repeats) taken in key order without randomization, so drift is confounded with key. On two GPUs they were also taken without the guard.
  - With only 12 clusters, a percentile bootstrap undercovers. The leave-one-key-out results (0.8, 7.4 and 3.7%) are the more honest uncertainty statement.
- **Only per-call regret is reported for the a-priori forms.** Deployment-relevant latency-weighted regret is missing. For the frozen form it is much larger on the RTX 3090 (9.7% LW vs 3.5% per call).

**Fix.**

- (i) Pre-specify and justify one selection protocol, and report all six in a supplementary table, with latency-weighted regret and cluster intervals.
- (ii) Replace "3.0 to 6.2%" in the abstract and conclusion by a range that reflects protocol sensitivity, or by the pre-specified protocol with an interval.
- (iii) State plainly that, on the only form-blind GPU, the compile-informed form is not better than a three-parameter restriction.
- (iv) Ideally, freeze all selectors and protocols and evaluate on a fourth, previously unseen GPU (for example an A100, L40S or H100 SXM). This is the only way to substantiate "a new GPU needs its compile facts but hardly its own calibration".

**Effort:** offline (i to iii); optional new GPU measurement (iv).

---

### Major 4. Where the headroom exceeds one synchronization, the information-side exclusion is under-powered, and the equivalence margin cannot fail in the direction that matters

**Location:** Section 6.3; Section 6.7 ("On Mamba-2 at the 370M width ... five RTX 3090 shapes exceed one synchronization; there the exclusion rests on the information side"); Tables 5, S6 and S37.

**Problems.**

- **The test relied on is powerless where it matters.** The information tests (exact-key permutation, conditional mutual information (CMI), paired tree differences) predict the per-call oracle *label*. Where the paper relies on them (Mamba-2 370M, RTX 3090, launch-inclusive), the label is tie- and noise-dominated:
  - H(oracle | exact key) = 1.97 bits, against a maximum of log2(5) = 2.32 bits;
  - median within-block CV is about 4 to 5% near the floor;
  - on average 3.2 chunks are within 2% of the oracle.
  
  A CMI excess of -0.0003 bits (p = 0.50) on such a label says almost nothing. Any real value effect would be swamped by arbitrary tie-breaking.
- **Exploratory check.** For the seven largest-headroom covered shapes of that data set, I tested a continuous outcome: the within-shape log-latency ratio between the table's chunk and the runner-up chunk, against entropy. No association was significant (Spearman rho -0.06 to 0.31, all p >= 0.13; n = 24 to 224 calls per shape). This agrees with the authors, but with 24 to 32 calls per shape at the two batch-16 shapes it can exclude only large effects. The largest headroom (43 µs at L = 8192, batch 16) is 0.4% of a 10.7 ms call, the same size as the 0.9% within-block CV.
- **Inert equivalence margin.** The 0.5-point margin (Table S6) is larger than the shape tree's entire residual regret on the primary data sets (0.38 to 0.46%). Adding entropy cannot lower the regret below zero, so an "equivalent at 0.5 points" verdict cannot fail in the beneficial direction on those data sets.
- **Multiplicity.** P-values are reported across about 19 data sets, 3 conditions and 3 statistics, with Holm correction only within a data set.
- **Fixed predictions in the bootstrap.** The cluster bootstrap resamples fixed held-out predictions without refitting, so the intervals ignore training variability. With 42 clusters, percentile intervals are also somewhat narrow.

**Fix.**

- (i) Add a continuous-outcome test: per-chunk log latency regressed on entropy within the exact key, with layer and prompt-cluster random effects, or the top-2 latency ratio as above. Report effect sizes in µs per call with intervals, and a minimum detectable effect.
- (ii) Re-time the five Mamba-2 RTX 3090 shapes whose headroom exceeds one synchronization in graph mode, with at least 3 rounds, and report whether the headroom survives (the authors' own analysis suggests it is block noise).
- (iii) Express equivalence margins in absolute terms tied to the decision cost (for example half a synchronization, about 10 µs per call) or relative to the residual regret.
- (iv) Use refit-in-bootstrap or repeated grouped cross-validation for the paired differences.
- (v) Control multiplicity across data sets, or label the tests as descriptive.

**Effort:** offline (i, iii to v); small new GPU measurement (ii).

---

### Major 5. The value-invariance foundation on the PRO 6000 is launch-inclusive, unguarded and partly launch-dominated; the real-activation input was timed in a separate process

**Location:** Section 6.1, Table 3; Table S1 (value invariance: launch-inclusive on PRO 6000, no guard, no state logging); Table S5 (TOST).

**Problems.**

- **Launch-dominated at L = 512.** Table 3 (414 cells) is launch-inclusive on the PRO 6000. At L = 512 the selected chunk's latency has a median of 62 µs (my recomputation from `e1_pro6000.jsonl`), inside the 56 to 90 µs host launch cost given in the conventions box. The L = 512 rows therefore largely time the launch path, and "invariance" there is partly the invariance of the launch.
- **The real input is confounded with process.** According to the meta records in `e1_pro6000.jsonl`, the first run used `--skip-real`. The captured-activation input, the most relevant of the eight, was then timed in a second process 36 minutes later, and the additive input-plus-round model treats it like the others. Section 6.1 shows that placement moves latency by up to 43.6%, so comparing the real input across processes, and therefore allocations, is exactly the confound the paper identifies.
- **TOST power is limited.** With three rounds as blocks, only 4 of 18 configurations pass at a 2% margin on the PRO 6000.
- **Production kernels above their floor are untested.** Value invariance of SSD and GDN above their launch floor is not measured (Limitations), yet the abstract states the shape-key conclusion for "dense chunked kernels" in general. The captured-call predictor tests in graph mode partially cover this, but only on oracle labels (Major 4).

**Fix.** Re-run the value-invariance sweep in device time or graph replay, with:

- all eight inputs, including the captured activations, interleaved in one process;
- fixed placement (design A) plus the multi-allocation design C;
- at least 5 rounds;
- for Mamba-1 at L in {2048, 8192} and for SSD and GDN at shapes above their floor.

Report TOST bounds per configuration. This is a few GPU-hours with existing code.

**Effort:** new GPU measurement (small).

---

### Major 6. Headline means are taken over a constructed workload, not a deployed one

**Location:** Section 5.2 ("per-call means are therefore not workload-weighted"); Tables 4, 5, 7 and 8; Abstract.

**Problem.**

- Prompts are truncated in rotation to 64 to 1024 tokens or kept whole.
- Batches of 4 and 16 reuse prompts. A batch-16 call holds 2 to 10 distinct prompts, and 177 of the 488 batch-4 calls contain fewer than 4 distinct prompts.
- The batch mix is fixed at 1536 / 488 / 336.

Per-call means and latency-weighted shares are averages over this artificial mixture. Every headline percentage (for example chunk 256 losing 49% per call and 34% of scan time, or the 1.0 to 5.9% under graphs) changes with the mixture. The serving replay addresses the mixture only through a step model (2.3 times faster than measured prefill), nearest-key mapping, and costing packed calls as batch-1 Triton calls.

**Fix.** Post-stratify the captured-call results to trace-derived (length bucket, batch) distributions. The key distributions of the Azure and BurstGPT replays are already computed. Report the headline per-call and latency-weighted numbers under those weights, with cluster intervals, next to the current unweighted ones.

**Effort:** offline.

---

### Major 7. "Chunk" effects are confounded with launch configuration, grid size and thread count, and the decomposition is undefined

**Location:** Section 4 (default launch table), Section 6.4 ("29 points due to the chunk and 3 to the launch configuration"), Table 1, Fig. S1, Tables S7 and S13.

**Problem.**

- **Triton scan.** In the default launch table, chunks 32/64/128 map to (D_b, n_w) = (8, 2), 256/512 to (4, 2) and 1024 to (4, 4). At width 2048 and batch 1, the grid therefore doubles from 256 to 512 programs between chunk 128 and chunk 256, which changes the number of waves on 188 or 82 SMs. Part of the "chunk effect" (for example the 2.5 to 8.5 times spreads, and chunk 256 vs 512 at batch 1) is a grid-size and occupancy effect of the paired launch entry.
- **Reference scan.** The thread count varies with the chunk. The authors address this with the 32-thread family.
- **The "29 + 3 points" decomposition is not defined** (sequential? Shapley?), and its order dependence is unknown.

**Fix.**

- Define the decomposition.
- Use the 184-configuration sweep at the 12 keys, which already exists, to report the chunk effect at fixed (D_b, n_w, stages): the spread over chunks within each fixed configuration, and the best-chunk-at-fixed-configuration regret.
- State in Table 1 and in Section 6.2 that the chunk results are results for launch-table entries.

**Effort:** offline.

---

## 6. Minor issues

1. **Inconsistent values for the same quantity.**
   - The PRO 6000 device-time cost model is 0.40 / 3.8 (Table 5, Tables S30 and S31) but 0.38 in Tables S14, S15 and S16, and 0.38 / 3.6 in Table S19. Both exist in the files: `e3b_information_mamba1_pro6000_r4spin_rmed.json` has a recomputed 0.405 and a stored 0.383.
   - The RTX 3090 cost model is 3.52 vs 3.55, from two sessions.
   
   Harmonize, or state which evaluation each table uses.
2. **Reference-scan stock dispatch.** The 6.6% regret compares a *separately timed stock binary* against an oracle from the runtime-chunk build. The runtime-chunk build at the stock chunk gives 5.35%. Re-timings of the same chunk in the two binaries differ by a factor of 0.86 to 1.13 (10th to 90th percentile, launch-inclusive; 2360 calls). The difference between 6.6 and 5.35 is block noise, and so is part of "a shape tree halves that". Use the same binary, and repeated rounds, for this comparison.
3. **Seed choice in Table 5.** The MLP rows report seed 0 (0.37% on the PRO 6000) where the five-seed mean is 1.59%. Report the seed mean and range in the table, not a single favourable seed.
4. **Upper bounds for "at every tuned shape".** "Below one host synchronization at every tuned shape" should be supported by per-shape upper confidence bounds, not point estimates. On the reference scan the held-out per-shape maximum is 14.5 µs against 21 µs. Also state that the isolated 20 µs synchronization is a *lower* bound on the critical-path cost in device-bound pipelines, where a synchronization drains the queue. This makes the comparison conservative, which helps the authors.
5. **Arbitrary floor definitions.** The launch floors ("twice the median batch-1 latency at L <= 256") and the 100 µs convention threshold drive the "above the floor" counts (for example 375 of 397, and 17% of calls). Add a sensitivity analysis (factor 1.5 / 2 / 3).
6. **Per-call oracle bias.** Regrets against the per-call oracle are biased upward by the minimum-of-noisy-means effect. Report the noise floor (cross-round regret of the round-median oracle itself) in each table caption, so readers know the resolvable level. In device time and graph mode the held-out tables reach 0.06 to 0.5%, which bounds this bias; say so.
7. **Isolation of the device-time and graph conventions.** Both time identical kernels back to back on resident inputs, without the producer's write or interleaved kernels. The cold-L2 control exists only for Mamba-1 on the PRO 6000 and, under graphs, one flushed round on the RTX 3090. State this limitation in the conventions box.
8. **Routed study.** Decode is about 39 ms per token (HF eager; 1.24 s for 32 tokens), which dilutes "% of total time". The statistic implementations are acknowledged as not bandwidth-optimal, and the guarded and learned-table variants pay for a histogram they do not use. Report prefill-only figures as primary, and give a fused or device-side cost estimate for each statistic.
9. **Serving replay.** It is a simulation: a step model, nearest-key mapping, and P_k and h_k extrapolated beyond L = 8192 or batch 16. Label its shares as model outputs everywhere they appear in the main text (Section 6.7 "Deployment units"), with the load sensitivity ranges.
10. **Tie criterion.** The 2% tie criterion is below launch-inclusive noise (CV 4 to 5%), so "share within 2%" understates ties there. Table S3 helps; reference it wherever launch-inclusive shares appear.
11. **Different boards.** The autotuner-state experiment (Table S34) and the divergence experiment (Table S36) ran on a different, stock 24 GB RTX 3090 instance. State that these are cross-instance checks.
12. **Abstract conventions.** The abstract mixes conventions: per-call means for the cost model, latency-weighted shares for the defaults. Name the convention for each number.
13. **Cluster structure.** Report the number and size distribution of the 42 prompt clusters, and use BCa or studentized intervals given the small number of clusters.
14. **Clarity.** The paper is extremely dense: nearly every sentence carries two to four numbers with different conventions. Moving the secondary conventions to the supplement and keeping one primary number (with an interval) per claim in the main text would greatly improve readability.
15. **Label provenance.** Table 5's "Best chunk per batch, held out" row is computed in sample in the stored analysis (`best_static_per_batch_in_sample`). I verified that the held-out values are identical (20.0 / 17.4 / 24.7), so this only concerns the label's provenance.

---

## 7. Recomputation table

All recomputations ran on CPU from the released files, using the authors' helpers (`analysis.round_median_view`, `analysis.cv_exact_headroom`, `e3_predictors.evaluate`) where applicable. "Stored" means the value was read from the authors' analysis JSON rather than re-estimated.

| # | Quantity (location) | Paper | Recomputed | Source file(s) | Status |
|---|---|---|---|---|---|
| 1 | Slowest / fastest chunk per call, Mamba-1, PRO 6000 device time, median (p95) (Section 6.2) | 4.8x (15.2x) | 4.77x (15.22x) | e2_mamba1_pro6000_r4spin.csv (round medians) | Match |
| 2 | Chunks within 2% of the oracle per call, Mamba-1 / Mamba-2 (Section 6.2) | 1.1 / 3.2 | 1.11 / 3.24 | same; e2_mamba2_pro6000.csv | Match |
| 3 | Table 4, Mamba-1, B = 1/4/16: oracle != static; best static; regret mean/p95; LW chunk 512; rule mean/p95 | 85/64/64%; 256/64/64; 18.8/50.5, 20.5/75.2, 24.6/93.3; 8.8/18.3/17.6; 18.8/50.5, 88.8/313.6, 126.3/386.1 | identical to one decimal | e2_mamba1_pro6000_r4spin.csv | Match |
| 4 | Table 4, Mamba-2, B = 1/4/16; spread median (p95) | 70/73/57%; 64/256/256; 1.5/5.9, 1.0/2.8, 2.0/6.8; rule 1.8/6.3, 1.1/2.9, 2.3/11.4; 1.04x (2.7x) | identical; 1.038x (2.71x) | e2_mamba2_pro6000.csv | Match |
| 5 | Chunk 256 per call / scan time; best chunk per batch per call / scan time (Section 6 intro) | 49% / 34%; 20% / 27% | 48.6% / 33.8%; 20.0% / 27.0% | e2_mamba1_pro6000_r4spin.csv | Match |
| 6 | Table 5, profiled selectors, all three GPUs (default, length only, entropy only, moments, shape, shape + entropy; mean / p95) | e.g. PRO 6000 48.6/309, 5.7/28.5, 45.1/247, 33.4/181, 0.46/3.7, 0.46/3.7; RTX 3090 shape 0.38/1.9, +entropy 0.35/2.4; H800 shape 1.63/12.5 | identical (grouped 5-fold CV re-run) | e2_mamba1_{pro6000_r4spin, vgpu3090_r4spin, h800_r3spin}.csv | Match |
| 7 | Exact-shape table held out, mean regret; covered headroom, PRO 6000 device time | 0.14 / 0.35 / 0.20%; 0.14 µs | 0.14 / 0.35 / 0.20%; 0.14 µs (29 covered shapes, 2156 calls) | same | Match |
| 8 | Table 3 (value invariance), all 9 rows: C*, spread, tied, max regret; cell count; tied share | e.g. Mamba-1 L512: 256, 2.50x, 92%, 1.36%; 414 cells, 94% | identical per row; 414 cells, 93.5% | e1_pro6000.jsonl | Match (rounding) |
| 9 | Table 7 (CUDA graphs), 8 rows: floor, share above, default mean/p95, LW, ms/forward pass (oracle) | e.g. SSD 370M 3090: 45 µs, 64%, 19.1/72.5, 2.8, 0.56 (20.1); Triton 3090 5.32 (29.6) | identical; Triton 3090 5.31 (29.6) | e2_*_r4graph3.csv, e2_*_taco_graph3.csv | Match (rounding) |
| 10 | Abstract: defaults leave "1.0 to 5.9% of scan time" | 1.0 to 5.9% | 1.02 to 5.85% | as row 9 | Match |
| 11 | Table 6, reference scan: spread median (p95); best single chunk; stock dispatch | 5.4x (12.9x); 30.2; 6.6 | 5.42x (12.9x); 30.20; 6.58 (on 2360 calls; the file also holds 12 calibration rows) | e8_refscan_vgpu3090.csv | Match |
| 12 | Table 8, Triton 370M: default and per-batch static, ms per forward pass (PRO / 3090 / H800) | 4.42 / 6.40 / 4.42; 2.03 / 0.79 / 1.76 | 4.42 / 6.40 / 4.42; 2.03 / 0.79 / 1.76 | e2_mamba1_* device-time tables | Match |
| 13 | CMI of entropy given exact key; log L given batch (PRO 6000 device time) | 0.0003 bits, p = 0.40; 1.79 bits | 0.0003 bits, p = 0.402; 1.789 bits | e3b_information_mamba1_pro6000_r4spin_rmed.json (stored) | Match |
| 14 | Key-churn break-even medians (one synchronization) | 1900 / 2660 calls (40 / 55 requests) | 1901 / 2662 calls (39.6 / 55.5) | e5b_pro6000.json, e5b_vgpu3090.json (stored) | Match |
| 15 | A-priori form regret (device time); cost-model 12-key bootstrap interval | 4.2 / 6.2 / 3.0%; [0.4, 2.3], [1.7, 6.1] | 4.16 / 6.23 / 3.01%; [0.38, 2.26], [1.74, 6.09] | taco_model_baselines.json (stored) | Match (but see Major 3: one of six protocols) |
| 16 | Cost model, PRO 6000 device time, mean regret, across tables | 0.40 (Table 5) vs 0.38 (Tables S14, S15, S19) | 0.405 (recomputed) and 0.383 (stored) both present | e3b_..._r4spin_rmed.json | Inconsistent across tables (Minor 1) |

**New analyses by the reviewer (not in the manuscript):**

- **A. Recoverable latency-weighted share under CUDA graphs** (default minus held-out exact table, 42-cluster bootstrap). See the table in Major 1. GDN 1.4B on the RTX 3090 gives 0.45% [-0.90, 1.78], not significant.
- **B. Sensitivity of the a-priori form to the selection protocol** (six protocols in the stored M4c block): PRO 6000 0.38 to 4.16%, RTX 3090 0.88 to 6.23%, H800 2.14 to 3.01% (Major 3).
- **C. RTX 3090 tensor-parallel sweep.** The cost model minus a table retuned at the exact shard shape and scored cross-round is -8.35 points [-12.38, -4.59] (Triton) and -4.35 [-7.06, -1.86] (reference): the sweep is noise-dominated (Major 2). On the H800 the same comparison gives +2.16 [1.16, 3.22], as expected.
- **D. Continuous-outcome entropy test** on the largest-headroom Mamba-2 RTX 3090 launch-inclusive shapes: Spearman rho -0.06 to 0.31, all p >= 0.13. This supports the authors' conclusion but with low power (Major 4).
- **E. Same-chunk re-timing across the two reference-scan binaries:** ratio 10th to 90th percentile 0.86 to 1.13. The stock-dispatch regret from the runtime-chunk build at the stock chunk is 5.35%, not 6.6% (Minor 2).

**Bottom line of the recomputation.** The released results reproduce the manuscript's numbers exactly, apart from rounding and one cross-table inconsistency. The concerns in this report are about what the numbers mean, how they are summarized, and how uncertain they are, not about errors in the data.
