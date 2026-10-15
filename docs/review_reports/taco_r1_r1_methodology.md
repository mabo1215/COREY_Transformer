# ACM TACO Submission: Review by Reviewer 1 (Methodology)

**Paper:** "Shape Sets the Chunk: A Compile-Informed Cost Model and a No-Gain Bound for Data-Dependent Dispatch of Chunked Linear-Recurrence Kernels" (main article, 23 pp. with references, plus a 48-page Supplementary Material, reviewed together)

**Reviewer identity:** GPU performance measurement methodology: benchmarking rigor, statistics of timing experiments, and fitting and validating performance models (cross-validation, overfitting, model selection).

**Location conventions:** `main.tex Lxxx` and `appendix.tex Lxxx` are source line numbers. Table, figure, section and equation numbers follow the compiled main PDF:

- Tables: Tab. 1 taxonomy, Tab. 2 framework, Tab. 3 value invariance, Tab. 4 oracle/static regret, Tab. 5 predictors, Tab. 6 selector baselines, Tab. 7 reference scan / larger checkpoints, Tab. 8 deployment units.
- Sections: Sec. 4 no-gain bound, Sec. 5 cost model, Sec. 6 methodology, Sec. 7.x results.
- Supplementary items are cited by their LaTeX label (e.g. `S-tab:supp_conventions`).

---

## Summary (5 lines)

1. The paper asks whether chunk size in chunked linear-recurrence GPU kernels should be chosen from the shape alone or also from the activation values. It times every chunk inside one recurrence-preserving launch on three GPUs.
2. It reports three results: (i) a value-invariance study with placement and L2 controls; (ii) a per-call oracle study showing that activation statistics add nothing beyond the exact shape; (iii) an elementary no-gain bound (Prop. 1) that turns "can data-dependent dispatch pay?" into measurable per-shape headroom.
3. The second contribution is an 8-parameter compile-informed latency model for an in-house Triton Mamba-1 scan. It is calibrated once per GPU and selects chunks with 0.4–3.6% mean regret (device time) or 0.9–3.5% (launch-inclusive), but it fails on every external kernel tried.
4. The measurement work is unusually careful (rounds, guard, placement and L2 interventions, CUDA-graph replays, cluster bootstrap, TOST, permutation tests), and I spot-checked 11 groups of headline numbers against the result files: all match.
5. The main methodological problems are: (a) convention switching, with launch-inclusive headline numbers that do not reproduce across sessions; (b) model-form selection on the test calls and unequal treatment of baselines; (c) noisy calibration data and opaque fitting; (d) per-call relative regret driving the CUDA-graph headlines; (e) overstated scope for "without timing target shapes" and for Prop. 1's empirical instantiation; (f) single-kernel validation of the cost model.

## Recommendation and confidence

- **Recommendation: Major Revision.**
- **Confidence: 4/5.** I checked the source, both PDFs, the result files and parts of the analysis code. I could not re-run the model fits, because SciPy is not installed in the provided environment.

**Statistical reporting completeness (per the reviewer standard):**

- Overall: Needs Improvement (about 65/100).
- The information/equivalence tests are close to Adequate: cluster bootstrap, Holm correction, permutation nulls and TOST are all present.
- The cost-model and selector comparisons are Inadequate: no CIs, no paired tests, model selection on the test data, single-seed learned baselines.

## Scores (0–100)

| Criterion | Score | Comment |
|---|---|---|
| Novelty | 52 | The bound is elementary (the authors say so). Compile-fact latency models and input-aware selection are established. The value lies in carefully closing a design question. |
| Methodology | 58 | Excellent controls in places. Undermined by convention switching, test-set form selection, noisy calibration data and missing uncertainty on the model results. |
| Experiments | 68 | Very broad: 3 GPUs, 4 kernel families, captured calls, TP widths, CUDA graphs, routed end-to-end runs, serving replay. Thin where it matters for the model contribution (one in-house kernel, one form-blind GPU, 57 distinct test shapes). |
| Clarity | 42 | The paper is dense with numbers. The abstract mixes conventions and metrics, and key qualifications are only in the 48-page supplement. |
| Significance | 50 | A useful negative result for practitioners. The cost model's practical value over simple tables or learned selectors is not established. |

## Strengths

- **S1. Recurrence-preserving timing.** The paper identifies and avoids the "split-call chunk timing" pitfall (Sec. 6.1, main.tex L570–579), and checks every kernel/chunk against an fp32 reference (`S-tab:supp_validation`).
- **S2. Value vs placement is disentangled.** Three buffer designs separate values from placement (`S-tab:supp_placement`, `S-tab:supp_placement_pro6000`). A warm/cold L2 intervention follows (`S-tab:supp_l2`). The 43% "input effects" are correctly attributed to buffer placement rather than values, and the authors say openly that the attribution rests on an intervention, not on hardware counters.
- **S3. Sound held-out design for the information question.** Folds are grouped by prompt, and batched calls never straddle folds (appendix.tex L403–408). Intervals use a prompt-cluster bootstrap with connected clusters. There is an exact-key permutation test with Holm correction, a permutation-corrected CMI, and an equivalence margin (`S-tab:supp_information`).
- **S4. The oracle's noise bias is acknowledged and partly quantified.** The paper uses held-out, split-half, null-simulation and cross-round estimates (`S-sec:supp_headroom_noise`), and checks round-median against single-block oracles (`S-tab:supp_conventions`).
- **S5. Negative results are reported.** Failures appear in the abstract: TFLA kernel 7.5%, fixed-thread reference scan 49.5%, production kernels under graphs 12–22%. Unstable form selection is shown in `S-tab:supp_clean_selection`. The non-reproducing launch-inclusive session is shown in `S-tab:supp_conventions`.

---

## Major concerns

### M1. The timing convention changes from claim to claim, and the launch-inclusive headline numbers do not reproduce across sessions

**Location**
- Abstract: L50–59. Device time is used for "19 to 28%" and "up to 25%"; launch-inclusive is used for "1.3 to 7.3%".
- Sec. 6 timing protocol: L654–666, and its last sentence, "tables are first-session launch-inclusive single blocks unless stated".
- Tab. 6 (`tab:baselines`, L943–976): launch-inclusive, one session.
- Sec. 7.8: L1354–1357, "hold for one session".
- `S-tab:supp_conventions` (appendix.tex L2078–2117) and appendix.tex L2223–2232: second PRO 6000 session.

**Problem**
- The same 2360 calls, re-timed launch-inclusive on the same GPU in a second session, give very different results:
  - shape-tree regret 4.7% instead of 0.7%;
  - cost-model regret 4.9% instead of 0.9%;
  - held-out headroom 3.9 µs instead of 0.45 µs;
  - median oracle latency 98 µs instead of 60 µs.
- On the RTX 3090, single launch-inclusive blocks are contaminated by the first block after loading the inputs: shape tree 10.7% using the first block vs 1.65% using the second or third block (appendix.tex L2205–2208). This suggests that 10 warm-ups are not enough after loading new inputs.
- The launch-inclusive convention measures how fast Python issues the kernel and records two CUDA events per call (floor 56–90 µs, L655–656), plus host CPU contention in a shared cloud container. Host CPU load is not logged except in one run.
- Calling these numbers "the cost in eager execution" (L657) is an oversimplification. Inside a real forward pass, whether launch latency is exposed depends on queue depth, not on the isolated call. The routed study (Sec. 7.6) shows prefill is host-bound and chunk-insensitive at 256 and 976 tokens.
- Each claim uses the convention where its effect is largest:
  - the motivating static-chunk loss uses device time (18.8% at batch 1, against 1.7% launch-inclusive);
  - the main baseline table, where the cost model "wins", uses launch-inclusive;
  - in device time the cost model is beaten by an MLP on two of three GPUs (0.35 vs 0.38 on the PRO 6000; 0.47 vs 3.55 on the RTX 3090) and by the three-parameter restriction on the RTX 3090 (0.88 vs 3.55).

**Required action**
1. Declare one primary convention a priori, preferably device time over three or more shuffled rounds, or CUDA-graph replay. State every headline number in it: abstract, contributions, Tabs. 4–8, Fig. 3 and the deployment recipe.
2. Move launch-inclusive results to the supplement, with both sessions side by side and the between-session difference as an explicit uncertainty.
3. Support any "eager execution" claim with in-model eager timings, as in the routed study, rather than isolated launch-inclusive microbenchmarks.
4. Log host CPU model, governor and load for every launch-inclusive run.

**Effort**
- (b) For Mamba-1, device-time data already exist for all three GPUs (`S-tab:supp_baselines`, device-time columns).
- (c) About 3 GPU-h: CUDA-graph replays of the 2360 Mamba-1 calls on the PRO 6000 and the H800 (about 1.5 GPU-h each, by analogy with the about 1.2–1.4 GPU-h three-round sweeps), so that the "graph ≈ device time" equivalence is not established on the RTX 3090 alone.

### M2. The model form was selected on the test calls, and the baselines did not get the same advantage

**Location**
- Sec. 5 "Model form and data": L492–515.
- Tab. 2 regrets 0.9/3.2 (L531).
- Tab. 6 (L963–973), with bold "wins".
- Abstract L57–59 and Contributions L152–167.
- `S-tab:supp_model_ablation`, `S-tab:supp_clean_selection` (appendix.tex L1283–1327), `S-tab:supp_baselines`.

**Problem**
- The terms of Eq. (3) were chosen by an ablation scored on the captured calls of the PRO 6000 and RTX 3090. Only the H800 is form-blind, so generalization of the form rests on one GPU.
- Re-selecting the terms without the target GPU is unstable:
  - leave-one-calibration-shape-out picks "−τe" on the H800, giving 7.25% mean and 62.5% 95th-percentile regret;
  - the held-out-error criterion picks "−W1" on the PRO 6000, giving 4.82% / 27.9%.
- The learned and simple baselines (gradient boosting, MLP, three-parameter, interpolation) never had test-set access to their structure. The bold entries in Tab. 6 for the PRO 6000 therefore compare a form tuned on that test set with baselines that were not.
- No paired uncertainty is reported for any selector difference: 0.88 vs 1.26 (PRO 6000), 3.54 vs 4.13 (H800), 3.81 vs 5.06 (reference scan). With 42 prompt clusters and 57 distinct shapes, several of these differences are probably not resolved.
- The learned baselines appear to be single-seed. Their results swing widely (MLP-basic PRO 6000: 9.63% launch-inclusive vs 0.35% device time), which suggests high variance.
- The 8-parameter form is not the minimal one. On the PRO 6000, dropping τe or τp gives identical regret (0.8775%; `cost_model_baselines_r4_summary.json`, `exact_tie_alternatives`). The paper's own recipe step (4), "preferring fewer parameters on ties" (L549–550), would select a 7-parameter form.

**Required action**
1. Make the clean, nested protocol the primary result in the main text: terms chosen by leave-one-shape-out on calibration data, or leave-one-GPU-out, then evaluated on the held-out GPU. Replace the "frozen form" column in Tab. 6 accordingly.
2. Report paired cluster-bootstrap CIs for model-minus-baseline differences on every GPU, in both conventions.
3. Run the learned baselines over at least 5 seeds, with a hyperparameter budget comparable to the 12-form search given to the model.
4. Apply the stated tie rule consistently.
5. Pre-register the frozen form now and evaluate it on a fourth GPU that the authors have not yet measured (e.g. A100, L40S or H100).

**Effort**
- (b) for actions 1–4, CPU only.
- (c) about 3–4 GPU-h for a fourth GPU: compile facts, 12-key calibration of about 15 min, a 2360-call three-round device-time sweep of about 1.3 GPU-h, and TP widths of about 1 GPU-h.

### M3. "Selects the chunk without timing target shapes" is inaccurate, and the test set has few distinct shapes

**Location**
- Abstract L56–57, "without timing target shapes".
- Contributions L153–154.
- Sec. 7.4 L925–941.
- Captions of Tab. 5 ("time no captured call of the target") and Tab. 6 ("none on a captured call").
- Workload construction L618–627: truncation to 64/128/256/512/1024 tokens in rotation.

**My check (`e2_mamba1_*.csv`)**
- The 2360 captured Mamba-1 calls contain only 57 distinct (L, B) pairs.
- 773 calls (32.8%) have exactly the (L, B) of a calibration key: L ∈ {128, 512, 2048, 8192}, B ∈ {1, 4, 16}, at the same width.
- 79.5% of calls have power-of-two lengths, because of the truncation scheme.
- So the calibration data time one third of the test calls' shapes, with different values.
- Regret stratified by overlap: the leakage does not drive the result, but the claim is literally false for a third of the calls, and "2360 calls" overstates the effective sample.

| GPU | On calibration shapes | Off calibration shapes |
|---|---|---|
| PRO 6000 | 0.76% | 0.94% |
| RTX 3090 | 2.90% | 3.42% |
| H800 | 3.55% | 3.54% |

- The unit of generalization for a shape-to-chunk model is the shape, and about 45 untimed shapes is a small test set. Per-call means weight shapes by call count; for example, L=1024 at batch 1 has 232 calls.

**Required action**
1. Restate the claim, e.g. "no captured call is timed; 33% of calls share a calibration shape".
2. Report regret on off-calibration shapes and per-shape (unweighted) metrics with shape-level CIs.
3. Add a held-out shape set with non-power-of-two lengths, batch sizes outside {1, 4, 16} (e.g. 2, 3, 8, 12, 24, 32) and lengths that are not multiples of 16. Synthetic inputs are acceptable, given the value-invariance results.

**Effort**
- (b) for the stratification.
- (c) about 1 GPU-h per GPU for a new synthetic shape sweep (3 rounds, 6 chunks, about 100 shapes).

### M4. The calibration data are noisy and the model fitting is opaque

**Location**
- Sec. 5: L476–480, "fitted once per GPU by robust least squares on log latency".
- Sec. 7.4: L925–930 (2298 calibration timings), and the auto-tuning baseline paragraph, L1015–1026.
- appendix.tex L1166–1207 (fit and validation), L558–572 (search protocol, 3 warmup / 10 repeats), L60–94 (`S-tab:supp_protocol`: calibration unguarded, keys in fixed order).
- `S-tab:supp_model_ablation`, `S-tab:supp_autotune_3090`.

**Problem: noisy calibration timings**
- 2208 of the 2298 calibration timings are auto-tuner search timings: 3 warm-up runs, 10 repeats, no guard, keys in fixed order.
- My check (`e4_*.jsonl`) compared each default-launch static chunk's search timing with its full-protocol re-measurement (10/100) for the same configuration and key:

| GPU | 90th-pct \|Δ\| | Max \|Δ\| |
|---|---|---|
| RTX 3090 | 36.7% | 226% |
| PRO 6000 | 4.8% | 61% |
| H800 | 2.6% | 11% |

- On the RTX 3090 the discrepancies concentrate at L=128, where compilation happens. Example at batch 1, chunk 32: 0.304 ms in the search vs 0.093 ms re-measured. At L=128, batch 4 the error goes the other way (0.095 vs 0.159 ms), which suggests the host launch floor shifted between phases.
- Consistent with this, the RTX 3090 "tuned" kernel is slower than the best static chunk at several keys (`S-tab:supp_autotune_3090`):
  - Mamba-1, L=512, B=1: 0.114 vs 0.096 ms;
  - SSD, L=128, B=16: 0.810 vs 0.770 ms;
  - SSD, L=2048, B=4: 0.854 vs 0.811 ms.
- This also weakens the auto-tuning baseline ("within 2.8% of the tuned kernel", L1021–1023).

**Problem: fitting procedure**
- The ablation shows R² increasing when a term is removed. On the PRO 6000: full model 0.951, without τe 0.956, without W1 0.959.
- For nested least-squares fits at the global optimum this cannot happen. Either the reported R² is not the minimized objective (soft-ℓ1 loss with scale 0.1), or the optimizer stops at different local optima. The PRO 6000 fit reports 18 function evaluations from three fixed restarts (`cost_model_pro6000.json`, `fit`).
- No parameter uncertainty is given, yet the parameters are interpreted physically: "0.44 and 0.76 µs of exposed latency", "saturation at four to seven warps" (appendix.tex L1474–1477).

**Problem: the implementation differs from the printed equation**
- The code bounds the prediction from below by bytes divided by datasheet bandwidth: `max(T, t_mem)`. Eq. (3) does not show this.
- I checked: the floor never binds on the captured calls of any GPU. It is harmless there, but Eq. (3) must describe the implemented model.

**Required action**
1. Refit using only full-protocol timings, or weight by inverse within-block variance, and exclude compile-contaminated first-key timings. Report how regret and parameters change.
2. Report an ordinary least-squares fit alongside the robust one.
3. Use at least 50 random multi-starts and report the spread of objective and selection regret.
4. Bootstrap over the 12 calibration keys to give parameter CIs and a CI for selection regret.
5. Report R² for the objective that is actually minimized.
6. Disclose the bandwidth floor or remove it.

**Effort**
- (b) for the refits.
- (c) about 0.5 GPU-h per GPU if the 184-configuration search must be re-timed at 10/100.

### M5. The CUDA-graph headline and the "eager suffices, graphs do not" contrast come from the regret metric

**Location**
- Abstract L53–54: "the SSD and Gated DeltaNet defaults lose 16 to 22%".
- Contributions L145; Results intro L703–708; Sec. 7.5 graph paragraph L1160–1166.
- Recipe preamble L1381–1383; Conclusion L1444–1447: "a shape-keyed table recovers the 16 to 22% their defaults lose".

**My check (`e22_graph_analysis_vgpu3090_rounds3.json`)**

| Kernel under graphs | Per-call mean regret | Latency-weighted share | ms per forward pass |
|---|---|---|---|
| SSD 370M | 19.1% | 2.7% | 0.56 |
| SSD 2.7B | 22.3% | 2.2% | 1.44 |
| GDN 1.4B | 16.2% | 1.0% | 0.10 |

- The per-call mean is driven by sub-floor calls with about 28 µs oracle latency, where the default loses 35–42%. Above the floor, the GDN default loses 2.0% and SSD 370M loses 10.4%.
- For comparison, the SSD default's latency-weighted share in eager execution is 2.0%, close to the 2.7% under graphs.
- So the qualitative contrast in the abstract and conclusion rests on how the mean of ratios weights microsecond-scale calls. The absolute gain (0.56 ms per forward pass for SSD 370M, out of about 20 ms of SSD scan time) is real but modest.
- All graph data come from a single GPU (RTX 3090).

**Required action**
1. Lead in the abstract, contributions and conclusion with the latency-weighted share, ms per forward pass and share of end-to-end model time. Keep the per-call mean as a secondary metric.
2. Do the same for every "defaults lose X%" statement, including the Mamba-1 "19 to 28%".
3. Replicate the graph timings on a second GPU.

**Effort**
- (a) and (b).
- (c) about 2–3 GPU-h; this overlaps with M1.

### M6. What the value-invariance and placement experiments establish is overstated in places

**Location**
- Sec. 7.1 (L710–776), Tab. 3, Prop. 1(b).
- appendix.tex L229–235 (ANOVA), L318–356 (TOST), L358–392 and L888–987 (placement and L2).
- Sec. 7.8 / `S-sec:supp_h800`: H800 without placement control.

**Problems**
1. **Mamba-2 and GDN short-length tests are uninformative.** For SSD and GDN at L ≤ 2048 at batch 1, the launch-inclusive latency is the host launch floor (about 0.44–0.52 ms on the PRO 6000; `S-tab:supp_latency`). The value-invariance and TOST tests on these cells measure host overhead and say little about value dependence of the kernels. Yet they enter the "94% of 414 cells" headline.
2. **Placement is confounded with input and interacts with the chunk.** Input and buffer placement are confounded in the main run (design B) and in every captured-call sweep, where each call has its own buffers. Placement interacts with the chunk: under warm L2, 29–41% at chunks 32–256 vs 2.6–8.2% at 512/1024 (`S-tab:supp_l2`). Per-call oracles at L2-sized working sets are therefore partly placement-determined.
3. **The H800 has no placement control.** Its input effects reach 48%, yet its captured calls and TP sweep feed headline cost-model and TP results.
4. **Number of inputs and F-test df.** The paper says "eight inputs" and "F test (7 and 14 degrees of freedom)" (L714, appendix.tex L233–234). The GDN analysis uses 7 inputs: `e1_analysis_pro6000.json` has `n_inputs`=7 and 21 cells per (L, mode). Its df are therefore (6, 12).
5. **The TOST bound is stated against the wrong reference.** It bounds each input's deviation from the pooled mean, so the difference between two inputs can be up to twice the reported bound (median 3.6% becomes about 7%).
6. **No power analysis.** With three rounds, TOST has low power, and no minimum detectable effect is reported.
7. **ANOVA assumptions unchecked.** Homoscedasticity and normality of round means are not checked for the 84 F-tests.

**Required action**
1. Repeat the fixed-shape value-invariance test for SSD and GDN under CUDA-graph replay or device time.
2. Make the fixed-placement design (A) the primary design, or randomize placement per round.
3. Run the placement and warm/cold-L2 controls on the H800, or qualify H800-dependent claims.
4. Correct the inputs/df statements.
5. Report pairwise equivalence bounds and an MDE or sensitivity analysis.

**Effort**
- (c) about 1.5–2 GPU-h: graph-mode invariance about 0.5 GPU-h; placement-randomized rerun about 1 GPU-h; H800 controls about 0.5 GPU-h.
- (a) for the corrections.

### M7. Prop. 1 is correct, but its empirical instantiation is presented more broadly than the evidence allows

**Location**
- Sec. 4: L362–446. Abstract L63–68; Contribution 3, L168–178; Sec. 7.7, L1210–1240.
- `S-tab:supp_headroom_noise` (appendix.tex L2388–2435), `S-tab:supp_static_bucket`, `S-tab:supp_information`.

**Correctness**
- Part (a) follows in one line from T(s, π(s, φ(Z)); Z) ≥ min_C T(s, C; Z), and the proof is correct.
- Part (b) is true by construction given its assumptions. Its "proof" (L416–418) lists instruction and address streams and buffer placement, but omits other state the latency depends on: L2 residency left by preceding kernels, and DVFS reacting to power. The text acknowledges the power issue.
- Neither part is "tested" by the experiments. What is tested is whether the assumptions hold and how large the estimated h(s) is.

**Problems with the instantiation**
1. **Against the coarse table, the bound does not exclude anything.** Against the length-bucket × batch table, the measured device-time headroom is 0.8–3.8% (2.2–20 µs; I verified this in `bucket_headroom.json`). On the H800 and the larger checkpoints, about 20 µs is roughly one host synchronization and above the cost of one device-side reduction kernel (7.3 µs at L=2048). Entropy also carries 0.03–0.09 bits about the oracle given the length bucket (`S-tab:supp_information`, "Bucket" column). Against the coarse table, therefore, the bound does not rule out a device-side policy. The exclusion holds only against the exact-shape table, and the abstract should say so plainly.
2. **The H800 host synchronization was never measured.** It is nominally 20 µs (`S-tab:supp_headroom_noise` caption; "--" in `S-tab:replication`), yet H800 comparisons with "one synchronization" are made.
3. **"At every tuned shape" is fragile.** It is a maximum over about 29 shapes of point estimates, some from fewer than 10 calls, with no CIs. Untuned shapes (204 of 2360 calls; per-shape headroom up to 784 µs) are excluded by definition.
4. **The CMI result cannot be interpreted as reported.** "Entropy carries 0.0003 bits given the exact key" (L906–907) needs the ceiling H(oracle | key). Report the normalized conditional mutual information.
5. **The degenerate interval is not equivalence evidence.** The paired difference "0.00 [0.00, 0.00]" in Tab. 5 means the depth-limited tree never split on entropy. The exact-key permutation test and the boosted learners are the real evidence; the table should say which test carries the inference.

**Required action**
1. Present Prop. 1(b) as a modeling assumption to be checked, not a prediction "tested" by the experiments.
2. Measure the H800 synchronization.
3. Give per-shape headroom with cross-round CIs, and report the share of calls and shapes that are untuned.
4. Report normalized CMI.
5. Rephrase the bucket-table result as "not excluded by the bound".

**Effort**
- (a) and (b).
- (c) under 0.1 GPU-h for the H800 synchronization.

### M8. The cost model is validated on a single in-house kernel

**Location**
- Contribution 2 (L152–167); Sec. 5; Tab. 2; Secs. 7.4–7.5 (L1107–1111, L1166, L1168–1176).
- Kernels paragraph L583–591, where the Triton scan is 1.23× the reference latency (geometric mean), up to 1.54×.
- `S-tab:supp_model_ablation`.

**Problem**
- All positive cost-model evidence comes from a Triton scan written for this study, which is slower than the production CUDA scan.
- Its chunk dependence is dominated by register spilling. Dropping the spill terms lowers R² from 0.951 to 0.788 and raises regret from 0.88% to 4.80%.
- The form fails on every external kernel:
  - TFLA mLSTM: 7.5%, while the library default loses only 1.4%;
  - the reference CUDA scan at fixed thread count: 49.5%;
  - the production SSD/GDN kernels under graphs: 12–22%.
- The contribution is therefore a model of one suboptimal kernel's spill and occupancy behaviour. For a TACO cost-model contribution, external validity is needed. Alternatively, the contribution should be reframed explicitly as a case study, and its prominence in the title and abstract reduced.

**Required action**
- Apply the full pipeline (compile facts, term derivation, calibration, held-out chunk selection) to at least one independently written kernel with a tunable tile and nontrivial register pressure (e.g. an FLA chunked GLA or recurrent kernel). Freeze the terms before seeing its test calls.
- Or reframe the contribution as described above.

**Effort**
- (c) about 4–8 GPU-h, plus engineering.

### M9. Isolation and environment reporting are incomplete

**Location**
- Sec. 6 timing protocol: L666–672.
- `S-tab:supp_protocol` (appendix.tex L60–94) and `S-tab:supp_platforms` (L103–135).
- appendix.tex L52–58: non-stock 48 GB RTX 3090 board.

**Problem**
- Clocks run at the default boost policy, and clock, power and throttle state are logged only in the value-invariance control runs.
- Several data sets were collected without the exclusive-GPU guard: the first value-invariance run, the calibration sweeps, and the first-session captured-call sweeps. These are the data behind Tabs. 3, 6 and 7.
- The guard can only see processes visible to `nvidia-smi` inside the container. It cannot rule out other tenants or GPU time-slicing, and host CPU contention, which affects launch-inclusive timings, is not monitored.
- The supplement mentions earlier batch-1 sweeps of the larger checkpoints that were "timed under host load", with a 129% coefficient of variation. This shows interference occurred in this environment.
- The RTX 3090 is a modified 48 GB board, and its memory clocks and timings are not reported.

**Required action**
1. Report SM and memory clock telemetry, including throttle-reason counts, for every sweep that feeds a table.
2. State how whole-device exclusivity was ensured: passthrough, MIG/vGPU off, no MPS.
3. Report host CPU model, governor and load.
4. Re-time the unguarded data sets, or show agreement with guarded re-timings.
5. Report the memory clock of the modified board.

**Effort**
- (a) and (b) where telemetry exists.
- (c) about 2 GPU-h to re-time the unguarded sets.

### M10. Cost-model, transfer, TP and deployment results lack uncertainty, and pooled means hide heterogeneity

**Location**
- Tab. 6; Tab. 7 (cost model 3.7 / 0.8); TP paragraph (L1124–1141); Tab. 8.
- `S-tab:supp_baselines`, `S-tab:supp_topk`, `S-tab:supp_tp`.

**Problem**
- None of these regrets has a CI.
- The calls are clustered three ways: 57 shapes, 42 prompt clusters, and 8 layers per prompt.
- Pooled means hide large heterogeneity. Example: for the 1.4B transfer, the pooled 3.7% contains a batch-4 mean of 13.9% (`cost_model_transfer_e2_mamba1_vgpu3090_1p4b_r3.json`).
- "Better than" statements have no test behind them. Example: the model at 0.8% vs the shape tree at 2.9% on 2.8B, L1117–1118.
- The cross-validated shape tree is a capacity-limited learner (depth ≤ 4, at least 5 calls per leaf; the exact-shape floor is 0.05%). Beating it says more about the tree than about the model.
- For TP, the per-round argmin agrees across rounds for only 52–70% of shapes (appendix.tex L1772–1776). The per-shape oracle is therefore noisy, and regrets against it need CIs.

**Required action**
1. Give cluster-bootstrap CIs (by shape and by prompt cluster) for every regret in Tabs. 4–8 and the TP results, and paired CIs for comparisons.
2. Add per-batch breakdowns to the main tables.
3. Use the held-out exact-shape table, not the depth-4 tree, as the learned reference.

**Effort**
- (b).

### M11. Reproducibility statement and artifacts

**Location**
- None of the following appears in main.tex or appendix.tex: a code or data availability statement, an artifact appendix, a listing or pseudo-code of the Triton Mamba-1 kernel, or the patches to the locally built mamba_ssm 2.2.2. The result metadata show a locally patched build tagged `2.2.2+corey.runtime.chunk.gpu`.
- The runtime-chunk reference-scan build is described only functionally (L1086–1092).

**Problem**
- The central kernel cannot be inspected, and the measurements depend on specific Triton, PyTorch and driver versions.
- TACO supports ACM artifact badges. A paper whose main claims are timing measurements should provide enough for an artifact evaluation.

**Required action**
1. Add a data and artifact availability statement.
2. Give kernel pseudo-code (tile loop, scan, state carry, launch table) and document the library patches.
3. Release, at least anonymously for review, the timing harness, raw per-repeat timings and analysis scripts.

**Effort**
- (a).

---

## Minor concerns

1. **Abstract (L47–69).** Nearly every clause carries a number, and conventions and metrics switch within sentences. Rewrite it around two or three effect sizes in one primary convention and metric.
2. **Routed study (L1189–1191).** "Slower ... by 24 to 71% in prefill" excludes the full histogram, which costs 56–193% (`S-tab:supp_routed_full`). Say so in the main text.
3. **Routed reference choice.** Routed differences are taken against "the fastest static chunk of each length", chosen post hoc on the same 30 rounds, which carries a winner's-curse bias. Use a pre-specified reference, such as stock dispatch or chunk 512, or the round-min analysis the authors already compute.
4. **Tie criterion (L678–683).** Overlap of two 95% CIs corresponds roughly to a test at α ≈ 0.005, and for three-round means t₂ = 4.30 makes it extremely lenient. Prefer a CI on the difference, or a practical equivalence margin. The 2% and 5% shares in `S-tab:supp_ties` are better primary tie measures.
5. **CIs of the mean (L643–644).** CIs from 50 back-to-back repeats assume i.i.d. samples, while GPU timings are autocorrelated and skewed. Report medians with block-bootstrap CIs, or between-round variability.
6. **Bootstrap 95th percentiles.** Percentile-bootstrap intervals of the 95th percentile with 42 clusters are unreliable; e.g. "93.3 [29.6, 93.5]" in `S-tab:supp_tables_ci`. Flag them or drop them.
7. **Mamba-2 device-time exclusion (L910–914; appendix.tex L2070–2074).** This data set was excluded after the fact ("timings we otherwise do not use"), and it is the one where entropy significantly lowers regret: −0.47 [−0.92, −0.07]. State the exclusion criterion and when it was set. Better, run the information tests of `S-tab:supp_information` on the clean CUDA-graph data of SSD 370M, SSD 2.7B and GDN.
8. **TFLA baseline (Tab. 6; appendix.tex L1354–1368).** TFLA's runtime expression instantiated for a scan with d_hv = 1 is a category mismatch. Present it as a sanity check rather than a competing baseline.
9. **TP framing (Abstract L60–61; Sec. 7.5).** "The unsharded table loses up to 65%" is a straw man, as the paper itself notes (L1132–1134). Lead with the comparison against retuning at the shard width.
10. **Output agreement (L1091–1092).** "Every chunk reproduces the stock output": the maximum absolute output difference is 0.0156 (`e8_analysis_vgpu3090.json`). State the tolerance.
11. **Tab. 4.** The "Best static" column is chosen in-sample per batch; label it so. The rule's 95th percentile of 313.6% is reported while its mean is 88.8%; consider reporting the median as well.
12. **Key churn (L1295–1301; appendix.tex L1630–1645).** P_k is scaled from wall-clock times using summed chunk latency, i.e. it is modelled, not measured. Give a sensitivity range.
13. **Serving replay (L1282–1284).** The replay's step model is 2.3× faster than measured eager prefill. Quantify the direction and size of the resulting bias in the reported shares.
14. **Mean of ratios.** The per-call relative regret is a mean of ratios and is sensitive to near-floor calls. Report the latency-weighted share next to it in every main table, not only in the supplement.
15. **Device-time spin.** Specify whether the warm-ups fall inside the 20 ms spin window, and for multi-kernel pipelines show that the spin covers enqueueing of the whole block. The paper says it does not for SSD; a per-pipeline check is needed.
16. **Supplement source header.** appendix.tex L1–10 still refers to an "IEEE TPDS submission" and an IEEEtran wrapper. Check that the compiled supplement uses the ACM format and that no previous-venue traces remain.
17. **Page limit.** The main PDF's conclusion ends on page 20 and the AI-use disclosure is on page 21. Confirm the 20-page limit (excluding references) is met.

---

## Number spot-check results

All values were recomputed or read from `src/experiments/results/tpds/` with Python run in isolated mode (`-I`).

| # | Paper location and value | File / key | File value | Verdict |
|---|---|---|---|---|
| 1 | Tab. 3: Mamba-1 L=8192: C* 512, spread 8.52×, tied 100%, max regret 0.57%. Overall 94% tied, 12/84 significant | `e1_analysis_pro6000.json` `decision[mamba1,8192,act]`; `overall` | 512, 8.52, 1.00, 0.567; tie-aware 0.944; 12 | Match |
| 2 | Tab. 4: Mamba-1 B=1 best static 256, 18.8/50.5; B=16: 64, 24.6/93.3; rule p95 386.1 | `r4_rounds_analysis.json` `tags.pro6000_r4spin.comparison.table_II_single_block_to_round_median` (round median) | 18.81/50.50; 24.60/93.31; 386.09 | Match |
| 3 | Tab. 5 (Mamba-1): shape 0.46/3.7 (93%); best single chunk 48.6/308.6 (20%); exact-shape held-out 0.14/0.0 (98%); cost model 0.40/3.8 (92%) | same file, `metrics`, round median | 0.465/3.75/0.934; 48.59/308.6/0.202; 0.138/0.0/0.977; 0.405/3.76/0.924 | Match |
| 4 | Tab. 6: cost model 0.88/4.0, 3.25/15.4, 3.54/14.1; three-parameter 4.48/50.1, 2.49/13.1, 4.13/17.4; GB structural 1.26/6.6; TFLA 12.70/88.1 | `cost_model_baselines_r4_summary.json` `captured_calls.{pro6000,vgpu3090,h800}` | identical | Match |
| 5 | Sec. 7.4 text, device time: MLP best on PRO 6000 / RTX 3090, loses 8.0% on H800 | same, `*_r4spin` / `*_r3spin` | MLP-basic 0.35 / 0.47 / 7.99; model 0.38 / 3.55 / 3.01 | Match. Note: model beaten in the primary convention |
| 6 | Tab. 7: reference spread 5.4× (12.9×); single chunk 30.2 (17%); stock 6.6 (72%); shape 3.0; entropy rule 185.5 | `e8_analysis_vgpu3090.json` | 5.43 (12.94); 30.20 (0.169); 6.58 (0.721); 3.00; 185.5 | Match |
| 7 | Tab. 7: Triton 1.4B/2.8B cost model 3.7/0.8; table tuned on 370M 5.3/6.8; floor 0.07/0.05 | `cost_model_transfer_e2_mamba1_vgpu3090_{1p4b,2p8b}_r3.json` | 3.65/0.79; 5.31/6.81; 0.071/0.050 | Match. Batch-4 regret for 1.4B is 13.9%, hidden by the pooled value |
| 8 | TP: cost model 6.5% (3090), 2.9% (H800); unsharded 36.8/22.1; d=2048 table 16.5/2.2; reference unsharded 2.2/6.6 | `e12_analysis_{vgpu3090,h800}.json` `families.*.pooled_shards` | 6.52/2.91; 36.82/22.10; 16.46/2.20; 2.22/6.65 | Match |
| 9 | Sec. 7.7 / abstract: bucket-table headroom 0.8–3.8% (2.2–20 µs), device time | `bucket_headroom.json` | 0.785%/2.15 µs (PRO 6000) … 3.78%/19.76 µs (H800); 2.8B 20.4 µs | Match |
| 10 | Sec. 7.5 / abstract: graph defaults lose 19, 22, 16% | `e22_graph_analysis_vgpu3090_rounds3.json` `tables.*_graph.policies.default.all` | 19.09 / 22.31 / 16.17. Latency-weighted: 2.73 / 2.15 / 1.01 | Match. Metric concern (M5) |
| 11 | Sec. 5 / abstract: clean selection 1.3–7.3% mean, 95th percentile up to 63% | `cost_model_baselines_r4_summary.json` `clean_selection_*` | 1.34 … 7.25; p95 62.5 | Match |
| 12 | Sec. 7.6: 24–71% prefill, 0.8–5% total | `e10_routed_vgpu3090.json` `paired_vs_best_static` | prefill 24.3–71.1; total 0.76–5.22 (full histogram excluded: 56–193% / 3.5–16%) | Match. Qualification missing (minor 2) |
| 13 | `S-tab:supp_deployment_buckets` row L=16, B=1, PRO 6000: 0.04 / 0.06 / 0.06 ms | `deployment_units.json` | 0.041 / 0.064 / 0.065 | Match |

**Discrepancies and undisclosed details found**

- "Eight inputs; F with (7, 14) df" (L714; appendix.tex L233–234). The GDN analysis has 7 inputs, so df are (6, 12).
- Eq. (3) vs implementation: the code takes `max(T, bytes/bandwidth)` with datasheet bandwidth. The floor never binds on the captured calls, but it is not described.
- "Without timing target shapes": 32.8% of captured calls share a calibration (L, B, width), and the test set has only 57 distinct shapes (M3).
- Calibration data quality: search timings differ from full-protocol re-measurement by up to 226% on the RTX 3090 (M4).
- The 8-parameter form ties exactly with two 7-parameter forms on the PRO 6000, contrary to the stated "fewer parameters on ties" rule (M2).

---

## What would be needed for Accept

1. **One primary convention** (device time over ≥3 rounds, or CUDA-graph replay) for all headline numbers, with launch-inclusive moved to the supplement and its session-to-session variability shown (M1). Headline effects stated as latency-weighted share and ms per forward pass (M5).
2. **A nested, clean evaluation of the cost model.** Terms selected without the target GPU's calls; paired cluster-bootstrap CIs against every baseline; multi-seed learned baselines; one additional unseen GPU with the form frozen before measurement (M2, M10).
3. **A corrected and qualified claim about target shapes**, with off-calibration-shape and per-shape results plus a new held-out shape set (M3).
4. **A refit on full-protocol calibration data**, with multi-start, parameter CIs and the bandwidth floor disclosed (M4).
5. **Value-invariance for SSD and GDN out of the launch floor** (graph or device time), a fixed-placement primary design, and either H800 placement controls or qualified H800 claims (M6).
6. **Prop. 1's empirical scope stated precisely.** Exclusion only against the exact-shape table; measured H800 synchronization; per-shape headroom CIs; normalized CMI (M7).
7. **External validity for the cost model** on at least one independently written kernel, or an explicit reframing as a single-kernel case study (M8).
8. **Environment and isolation reporting**, and a data/code availability statement with kernel pseudo-code and library patches (M9, M11).

Estimated new GPU time for the full set: about 15–25 GPU-h. Most other items are offline re-analyses of existing result files.
