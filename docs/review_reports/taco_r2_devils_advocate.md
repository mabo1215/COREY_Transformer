# TACO Review: Devil's Advocate Report

**Manuscript:** "Shape Sets the Chunk: Selecting and Keying the Chunk of Chunked Linear-Recurrence GPU Kernels" (new regular paper, double-anonymous, acmsmall)
**Material reviewed:** main.pdf / main.tex (20 pages plus references), Supplementary Material appendix.pdf / appendix.tex, ref.bib, the released result files and analysis code.
**Role:** Devil's Advocate. I set out to find the strongest reasons to reject while keeping every claim tied to the manuscript or to a recomputation from the authors' own result files. I did not consult any earlier review.

---

## 1. Summary

The paper studies how a runtime should choose the chunk (time tile or intra-chunk block) of chunked linear-recurrence kernels: a Triton Mamba-1 selective scan written for this study, the reference Mamba-1 CUDA scan rebuilt with a runtime chunk, Mamba-2 SSD and FLA Gated DeltaNet. It makes three claims. (1) The optimal chunk is a function of the shape key; activation statistics add no practically relevant information. An elementary "headroom" inequality (Lemma 3.1) bounds what any data-dependent policy can gain. (2) Library defaults are adequate in host-bound execution. Under CUDA-graph replay they lose 1.0 to 5.9% of scan time, which a shape-keyed table recovers. (3) A compile-informed cost model of the in-house Triton scan selects chunks with 3.0 to 6.2% regret when its terms are chosen a priori. Profiling every enumerable key is still far cheaper than calibrating the model. The paper closes with a deployment "recipe".

The measurement work is careful and extensive. Every chunk is timed inside a recurrence-preserving launch, rounds are shuffled, placement and L2 controls are run, intervals use a prompt-cluster bootstrap, and limitations are stated openly. Most numbers I spot-checked reproduce from the result files.

**The case for rejection** rests on six points:

1. **Little is new and the conclusions are expected.** The kernels are data-oblivious by construction, so a shape-only optimum is the null expectation. The paper says so itself in Section 3. The only foil is one unreviewed preprint.
2. **The large chunk effects come from the authors' own kernel.** That Triton kernel is 23% slower than the reference scan, uses a launch table tuned on a different GPU (an RTX 3070), and spills registers at the chunks that win on long calls. Every independent kernel the authors measured (reference CUDA scan, TFLA, SSD, Gated DeltaNet) shows that the default or a stock length rule is within a few percent of the best chunk.
3. **The production-kernel benefit is small and not validated in a real system.** Under graph replay of isolated calls it is 1 to 6% of scan time. There is no engine-level or end-to-end measurement.
4. **The one serving-like experiment contradicts the recipe.** On packed variable-length prefill steps, the shape-keyed table does not recover the default's loss.
5. **The cost model loses on fair terms.** Once its terms are chosen a priori, the compile-informed model is no better than its own three-parameter restriction or a generic MLP. Its accuracy depends heavily on which selection protocol the analyst picks.
6. **Several summaries are selective.** Examples: the H800 value-invariance results, the device-side reduction cost on the primary GPU, and medians reported where worst cases matter.

---

## 2. Scores

| Dimension | Score (0-100) | Confidence (1-5) | One-line rationale |
|---|---|---|---|
| Novelty | 30 | 4 | Shape-keyed tile selection is standard practice. Value invariance follows from data-oblivious kernels. The model is a narrow case study whose own economics favour profiling. |
| Methodology | 60 | 4 | Rigorous timing protocol and statistics. Weakened by the in-house kernel, an unrepresentative call mix, protocol-dependent model accuracy, and noisy or virtualized platforms. |
| Experiments | 55 | 4 | Very extensive, but production-kernel evidence rests on two non-datacenter GPUs. There is no engine or end-to-end validation, and the packed-call result contradicts the recipe. |
| Clarity | 35 | 4 | Extremely dense: five or more timing conventions, several definitions of "best static", and an abstract built from qualifiers. The main narrative depends on about 40 supplementary tables. |
| Significance | 28 | 4 | The actionable outcome is "keep the defaults, or key a table on shape; don't build a model". For production kernels the gain is 1 to 6% of scan time in isolated replay. |

**Overall confidence:** 4/5. I recomputed several key quantities from the released files; see Section 6.

---

## 3. Recommendation: **Reject**, with resubmission of a refocused paper possible

**Where the decision boundary sits.** A Major Revision would be appropriate if the defects were in presentation or in missing analyses that the existing data could supply. Two defects are not of that kind (C1, C2).

- Once the in-house kernel is set aside, the empirical content says that production defaults or stock length rules are within a few percent of the best chunk. The device-bound exceptions are 1 to 6% of scan time in isolated graph replay and have not been shown to matter end to end.
- No rewrite turns that into a significant TACO contribution. A new, substantially different study is needed. It would have to show, for production kernels in a real serving engine on a datacenter GPU, a chunk effect large enough to matter, together with evidence that a shape-keyed cache captures it in packed variable-length prefill.

If the authors produced that evidence and demoted the in-house kernel and its cost model to a supporting case study, the paper could cross into Major Revision territory. The 10 MAJOR findings below would each also need addressing, but none of them alone would justify rejection.

**Strengths, for fairness:**
- The recurrence-preserving timing methodology is correct, and the critique of split-call chunk timing is useful.
- Placement and L2 controls separate buffer placement from value effects.
- Paired, prompt-cluster-bootstrapped information tests (exact-key permutation, conditional mutual information) are unusually thorough. For the Mamba-1 kernels the information conclusion is convincing: paired intervals are [0.00, 0.00].
- Limitations are disclosed candidly, and the artifact appears complete.

---

## 4. Findings

Severity labels: **CRITICAL** blocks acceptance; **MAJOR** must be fixed for any acceptance; **MINOR** should be fixed. Effort classes: *text*, *offline analysis* (existing data), *new GPU measurement*.

### C1. CRITICAL: The contribution is confirmatory and of low significance

- **Location:** Abstract; Section 1 (Contributions); Section 2 (Positioning); Section 3 (paragraph after Lemma 3.1); Section 7 (Conclusion).

- **Evidence that the value-invariance result is expected:**
  - The paper establishes it a priori. The three kernels "contain no data-dependent branches, early exits or sparse paths". "If, in addition, the input buffers sat at fixed addresses and every instruction took a time independent of its operand values, h(s) would be zero" (Section 3).
  - Related work concedes the same: "All of them key the result on shape and device, not on data values" (Section 2).
  - TFLA (Beck et al. 2025) already derives the chunk from shape and hardware.

- **Evidence that the foil is weak:**
  - The only proposal refuted is a single 2026 arXiv preprint (ref.bib: primaryClass cs.CV) proposing an entropy-to-chunk map.
  - Its refutation is listed as a contribution ("we test and refute that design").
  - Refuting an unreviewed, unadopted design is not by itself a research contribution at TACO's level.

- **Evidence that "where the chunk matters" is also expected:**
  - Launch overhead hides kernel-time differences of short calls, and CUDA graphs expose them. This is the standard motivation for CUDA graphs.

- **Evidence that the production-kernel benefit is small:**
  - Under isolated CUDA-graph replay, the latency-weighted loss of the defaults is 1.0 to 5.9% of scan time (Table 7). I verified this in `taco_e26_analysis_pro6000.json`: 1.75, 5.85, 3.19 (PRO 6000) and 2.80, 2.20, 1.02 (RTX 3090).
  - In absolute terms that is 0.10 to 1.44 ms per forward pass, of unknown share of end-to-end prefill.

- **Evidence on the model and the lemma:**
  - The cost model's own economics end in "not a substitute for profiling an enumerable key set" (Section 6.7).
  - Lemma 3.1 is a one-line inequality (`T(π) ≥ min_C T`).

- **Why it matters:** TACO expects either a new technique or new, non-obvious knowledge that changes practice. The take-away is "key a tuning cache on shape and device, as every autotuner does; defaults are mostly fine; a model is not worth it". The value of that conclusion is mostly confirmatory.

- **What would resolve it:** One of two routes.
  - (a) Show, for production kernels inside a real serving engine on a datacenter GPU, a chunk effect that matters end to end, and show that the proposed key captures it.
  - (b) Recast the work as a short negative-result or characterization paper, with claims scaled accordingly.

- **Effort:** new GPU measurement plus text.

### C2. CRITICAL: The headline chunk sensitivity is largely an artifact of the in-house Triton kernel

- **Location:** Abstract ("For a register-bound Triton scan ..."); Section 1; Section 4; Section 5.1; Section 6 opening paragraph; Tables 2, 4, 5, 6 (Triton columns), 8; Supplement S1.2, Fig. S1, Table S9, Table S13.

- **Evidence that the kernel is in-house and slower than the reference:**
  - The kernel was "written for this study" (Section 5.1).
  - It takes 0.81 to 1.54 times the latency of the reference CUDA scan, geometric mean 1.23 (Table S9; I recomputed 1.229 from the twelve ratios). This comparison exists only on the RTX 3090.

- **Evidence that its launch configuration is poorly tuned:**
  - "The default launch table was chosen once by a short offline search on a fourth GPU (an RTX 3070) at d = 2048 ... and is used unchanged on every GPU" (S1.2).
  - The chunk is coupled to the channel block and the warp count (Fig. S1: 32, 64 and 128 map to (8, 2); 256 and 512 to (4, 2); 1024 to (4, 4)). A "chunk" change is therefore partly a launch-configuration change.
  - The best table entry is 7 to 18% behind the fully tuned kernel on the RTX 3090, the H800 and in device time, and up to 53% behind at a single key (Section 6.4).

- **Evidence that the steepness comes from register pressure:**
  - The winning long-call chunks sit at the register cap and spill. Chunks 512 and 1024 use 255 registers with 12 and 64 B of spill stores on sm_120 (Table S13).
  - The spill terms are the most important part of the model (S18: removing them raises regret to 4.8% and 12.7%).

- **Evidence that independent kernels show no comparable effect:**
  - Reference CUDA scan: the stock length rule is within 3.1% in device time (Table S21) and 6.6% launch-inclusive (Table 6).
  - TFLA mLSTM: the default loses 1.4% (Table S27).
  - Mamba-2 SSD and Gated DeltaNet host-bound: at most 2.5% (Table S32/S33).
  - Above the device-time floor, SSD chunk 256 is the oracle for 375 of 397 calls (loss at most 0.24%), and Gated DeltaNet chunk 64 for 154 of 155 calls (Section 6.5).

- **Evidence on disclosure:**
  - The abstract never says that the "register-bound Triton scan" is the authors' own kernel and slower than the reference.
  - The introduction's contribution bullets refer to "the Triton scan" without defining it.
  - Most quantitative tables and economics are about this kernel: Tables 2, 4, 5 and 8, the tensor-parallel widths, the break-even analysis, key churn and the serving replay.

- **Why it matters:** The steep chunk curve motivates the cost model, the economics and much of the recipe. If it reflects an under-engineered kernel (register spills, foreign launch table), then those parts do not tell a practitioner using production kernels anything.

- **What would resolve it:**
  - State the provenance and the 1.23× gap in the abstract and introduction.
  - Re-measure chunk sensitivity after per-GPU tuning of the launch table, or with the fully autotuned configuration per chunk.
  - Add at least one independently written, production Triton scan with a tunable chunk.

- **Effort:** text plus new GPU measurement.

### C3. MAJOR: Strawman baselines inflate the headline chunk effect

- **Location:** Section 6 opening ("chunk 256, the Mamba-2 default, serves as its untuned reference and loses 49% per call ... (34% of the scan time)"); Table 5 rows "Default chunk 256"; Table 8 "Default" column for Triton rows ("4.4 to 25 ms per forward pass"); Supplement S3.

- **Evidence that the reference default is borrowed:** The Triton Mamba-1 kernel "has no library default". The reference used is borrowed from a different kernel family (Mamba-2 SSD).

- **Evidence from my recomputation** (`e2_mamba1_*` per-call tables; round medians where available):
  - The obvious zero-cost alternative mirrors the stock CUDA dispatch: `chunk = clip(nextpow2(L), 32, 1024)`.
  - Per-call mean regret of that rule: 5.86% (PRO 6000), 7.76% (RTX 3090), 10.66% (H800), in device time.
  - Latency-weighted regret: 5.96%, 16.31%, 13.94%.
  - Compare the "default 256" (48.6/96.4/50.9% per call) and the a priori cost model (4.16/6.23/3.01%).
  - So the 130 to 322 s calibration (warm compile cache) buys only 1.7, 1.5 and 7.7 points of per-call regret over a free rule.
  - The best static chunk by latency on the PRO 6000 (512) loses 15.5% of scan time, not 34%.
  - The paper's own calibration-only "length tree" (16.6/20.4/38.4% in Table S19) is far worse than this trivial rule. That suggests the calibration-key baselines are weak.

- **Why it matters:** The motivating magnitudes (49% per call; 4.4 to 25 ms per forward pass) compare against an arbitrary default. Against a sensible untuned rule the effect shrinks by an order of magnitude.

- **What would resolve it:** Add a stock-style length rule as the untuned baseline in Tables 4, 5 and 8 and in the Section 6 opening, and restate the headline numbers against it.

- **Effort:** offline analysis plus text.

### C4. MAJOR: The cost model's value does not survive a priori term selection and calibration cost

- **Location:** Abstract; Section 4; Section 6.4; Table 2; Table 5; Section 6.7 ("Calibration versus profiling"); Supplement S3, Tables S14 to S19.

- **Evidence: the model is worst on average under fair selection.**
  - With a priori terms the model scores 4.16/6.23/3.01% (Table 5): mean over the three GPUs 4.47, worst 6.23.
  - Its own three-parameter restriction scores 4.20/0.85/3.53 (mean 2.86, worst 4.20).
  - The MLP averaged over five seeds scores 1.59/1.31/7.13 (mean 3.34). The structural MLP scores 3.75/4.90/1.11 (mean 3.25).
  - So with fairly chosen terms, the compile-informed eight-parameter model has the worst GPU-mean regret of these calibration-only competitors.
  - The text's "within 3.5% on every GPU" and "best on the RTX PRO 6000" use the frozen form whose terms were selected on the PRO 6000 and RTX 3090 captured calls. That comparison is biased in the model's favour, because the competitors' hyperparameters never saw captured calls.

- **Evidence: accuracy depends on the analyst's choices.** In `taco_model_baselines.json` (M4c), the "a priori" result depends on the candidate set and the criterion:
  - PRO 6000: 0.38% to 4.16%.
  - RTX 3090: 0.88% to 6.23%.
  - H800: 2.14% to 3.01%.
  - The 50-start optimizer reaches 1.7% on the RTX 3090 instead of 3.55%; the reported fit is a local optimum (Section 6.4).
  - An ordinary least-squares fit gives 2.1/1.8/3.0% (S3).
  - The bootstrap intervals are 0.4 to 2.3, 1.7 to 6.1 and 2.4 to 3.0%.
  - "The method's accuracy" is therefore not well defined.

- **Evidence: the H800 "nine-parameter variant" is not independent.** The selected form (`add_l2_miss`) reproduces the frozen form's regret exactly (3.0077%, same p95 and tie share). The added term is inert, so it gives no independent a priori evidence.

- **Evidence: per-batch regrets are hidden.** On the RTX 3090 the model loses 15.9% at batch 16, against 1.5% at batches 1 and 4. On the H800 it loses 6.3% at batch 4 (`cost_model_*_r3spin.json`, by_batch). Batched calls are where the device-bound regime matters.

- **Evidence: the economics favour profiling.**
  - Calibration takes 130/322/202 s warm (493/843/664 s cold).
  - Profiling all 57 exact keys takes 3.9/7.4/6.1 s.
  - The model pays only beyond about 1600 to 2500 keys; I verified 1622 to 2475 in the result file.
  - Request-level and fixed-slice serving produce 15 to 23 bucket keys (S13).

- **Evidence: the production-pipeline models fail where the chunk matters.** The work-term model calibrated eagerly loses 9 to 22% under graphs, and a refit repairs it only partially (Table 7). On the PRO 6000 the refit loses 20% of scan time on SSD 370M, against 1.7% for the default (S9).

- **Further limits:** Only one GPU is form-blind, and "whether its procedure transfers is untested" (Section 7).

- **Why it matters:** Contribution 3 is presented as the paper's methodological novelty. On fair comparison it is dominated by simpler models and by profiling.

- **What would resolve it:**
  - Report all calibration-only selectors under the same a priori protocol in the main table.
  - Pre-specify the selection protocol.
  - Report per-batch and shape-weighted regrets.
  - Either identify where the full model is needed or demote it.

- **Effort:** offline analysis plus text; new GPU measurement for any additional form-blind kernel or GPU.

### C5. MAJOR: The packed variable-length experiment contradicts the recipe, and its 370M data are noise-dominated

- **Location:** Section 6.5 ("Packed variable-length calls"); Recipe box ("Key: length (packed length of a variable-length call) ..."); Section 1 Contribution 2 ("a shape-keyed table recovers what the defaults lose"); Supplement S10, Table S35.

- **Evidence: tables do not recover the loss** (`e27_varlen_pro6000_taco.json`):
  - SSD 370M, fresh state: default 9.75% per step (7.46% latency-weighted, LW). The held-out packed-token-bucket table is worse, at 14.14% (10.23% LW). Adding the number of sequences to the key gives 13.46%.
  - SSD 370M, carried state: 10.68% (8.11% LW) falls only to 9.21% (6.77% LW).
  - SSD 2.7B: 9.04% falls to 2.77% (fresh) and 7.53% to 4.48% (carried).
  - The 370M packed default loss of 7.5 to 8.1% LW exceeds the abstract's "1.0 to 5.9%". It is not mentioned in the abstract or the conclusion.

- **Evidence: the 370M measurement is noise-dominated (my recomputation from the raw rounds):**
  - The per-round argmin is identical in all three rounds for only 34% of SSD 370M step-variants (70% for 2.7B, 83% for Gated DeltaNet).
  - The median round-to-round deviation is 1.4%, with a 90th percentile of 6.3%.
  - These steps were timed in device time with the same 20 ms pre-spin. Supplement S1.1 and S8 state that this spin does not cover the enqueueing of multi-kernel SSD pipelines near the floor ("we do not use them").

- **Why it matters:** Packed chunked-prefill steps are the production-like regime. There are two possible readings, and both hurt the paper:
  - If the measured default loss is real, the proposed key is incomplete for packed calls, and the recipe fails where it is most needed.
  - If the loss is noise, the reported loss (7.5 to 10.7% per step) is overstated, and the protocol used is one the paper itself rejects elsewhere.

- **What would resolve it:**
  - Re-time the packed steps under CUDA-graph replay with more rounds, and report a noise floor (split-half or a null as in S12).
  - Investigate richer keys, such as per-sequence length structure.
  - Revise the recipe and Contribution 2 accordingly.

- **Effort:** new GPU measurement plus text.

### C6. MAJOR: No engine-level or end-to-end evidence for the device-bound regime in which the chunk is claimed to matter

- **Location:** Section 1 ("production engines run prefill both host-bound and as captured CUDA graphs"); Sections 6.5 to 6.7; Tables 7 and 8; Supplement S9, S13, S15, Table S39.

- **Evidence: the graph replay is an isolated-kernel proxy.**
  - All device-bound production-kernel evidence comes from capturing each (call, chunk) pair into its own CUDA graph.
  - "ms per forward pass" is mean per-call regret multiplied by the number of layers (Table 8 caption), not a measured forward pass.
  - The share of end-to-end prefill time is never reported for SSD or Gated DeltaNet.

- **Evidence: the vLLM premise is doubtful.** The vLLM CUDA-graph design notes (cited by the paper as `vLLM Team 2025`) state:
  - Under the default `FULL_AND_PIECEWISE` mode, prefill and mixed batches are not captured as full graphs.
  - In piecewise mode, "attention or other CUDA Graphs-incompatible operations stay eager".
  - Mamba attention backends list full-graph support only for uniform single-token decode.
  - So the isolated graph replay may not represent how these kernels run during prefill in a production engine.

- **Evidence: the routed end-to-end study cannot answer the question.**
  - It is eager only, on one RTX 3090, with the reference CUDA scan and the 370M model.
  - At 256 and 976 tokens "the prefill takes about 46 ms whatever the chunk" (S15).
  - Its "learned table" variant "compute[s] the sampled histogram first and pay[s] its full cost" (S15). A zero-overhead shape-keyed policy is therefore never measured end to end.

- **Evidence: the serving replay is simulated.** It uses a step model "2.3 times faster than the measured eager prefill" and costs only the in-house Triton kernel.

- **Why it matters:** The practical claim "the chunk matters where execution is device-bound" is never shown to translate into measurable time-to-first-token or throughput gains.

- **What would resolve it:** Integrate a shape-keyed chunk table into vLLM or SGLang for a Mamba-2 and a Gated DeltaNet hybrid model on an H100, H200 or B200. Measure prefill latency and throughput with and without the table under the engine's real execution mode, and report the scan's share of prefill.

- **Effort:** new GPU measurement.

### C7. MAJOR: Value invariance is not established for production kernels and is summarized selectively

- **Location:** Abstract; Section 6.1; Section 6.8; Recipe ("synthetic inputs suffice, as latency is value-invariant"); Limitations; Supplement S1.3, S2, Tables S5, S10, S11.

- **Evidence: production kernels are not tested above their launch floor.**
  - Limitations: "value invariance of SSD and Gated DeltaNet above their launch floor are not measured".
  - The sweep in Table 3 is at batch 1 and launch-inclusive. There, SSD is launch-bound and flat (chunk spread 1.01 to 1.03× at L ≤ 2048), so invariance is trivially satisfied.
  - The Gated DeltaNet synthetic sweep calls a different kernel entry than the captured calls, and the authors note the two are "not directly comparable" (S6).
  - The recipe's "synthetic inputs suffice" is therefore unvalidated for the kernels and regime where tuning is recommended.

- **Evidence: the H800 result is omitted from the main text** (`e1_analysis_h800.json`, `e1_analysis_h800_r3spin.json`):
  - 28 of 36 fixed-chunk configurations show a significant input effect launch-inclusive, and 36 of 36 in device time.
  - Input ranges reach 47.6% (launch-inclusive) and 39.9% (device time) at L = 2048.
  - At L = 2048 the per-input best chunk flips between 256 and 1024. Only 54 to 71% of cells are tied, and the regret reaches 5.7%.
  - The main text reports only the PRO 6000's "12 of 84". It says only that the H800 "has no placement or L2 control".
  - Section 6.8 discusses only the RTX 3090 ("at most 4.3%").
  - The placement-not-values explanation is therefore untested on the only Hopper GPU.

- **Evidence: medians are reported where worst cases matter.**
  - Main text: the equivalence bound at the selected chunk has "a median of 3.6% (RTX PRO 6000) and 3.2% (RTX 3090)".
  - Table S5: at the 2% margin used for ties everywhere else, only 4 of 18 selected-chunk configurations pass on each GPU.
  - The maximum bounds are 14.8% (Mamba-1) and 34.0% (SSD) on the PRO 6000.

- **Evidence: the placement control lacks power at L = 2048.** At L = 2048 on the PRO 6000, values-only ranges reach 22.8% but are non-significant because of noise (`e9_placement_pro6000_r3.json`). The headline comparison of 0.1% (values) against 21% (placement) is made at L = 8192 only.

- **Why it matters:** "Activation statistics add no practically relevant information" is stated for chunked linear-recurrence kernels in general. The controlled evidence covers the in-house kernel and the reference scan on two GPUs.

- **What would resolve it:**
  - Run a placement and L2 control on the H800.
  - Run a value-invariance sweep of SSD and Gated DeltaNet above the floor, or under graph replay, comparing synthetic and captured inputs through the same entry point.
  - Report worst-case equivalence bounds in the main text.

- **Effort:** new GPU measurement plus text.

### C8. MAJOR: The headroom (Lemma 3.1) argument is narrower than stated, and its cost side is parameterized selectively

- **Location:** Abstract ("below the cost of one host–device synchronization at every tuned shape"); Section 3; Section 6.7 ("against the exact-shape table, g > o is out of reach at tuned shapes"); Supplement S12, S13, Tables S37, S38.

- **Evidence: the bound holds only for a key the paper finds impractical.**
  - The "below one synchronization" statement holds for an exact-shape table on 29 covered shapes.
  - The recommended bucket key, even with the specialization bit, leaves 26 µs on one RTX 3090 key. Bucket 2048, batch 4, unaligned leaves 30 to 32 µs (S12, S13), above one synchronization (21.4 µs).
  - Under real traffic the exact-length key places 51.5 to 83.8% of calls on keys below the break-even (Table S38).

- **Evidence: the device-side reduction cost is quoted only for the RTX 3090.**
  - The main text quotes 7.3 µs (L = 2048) and 42 µs (L = 8192) for the RTX 3090.
  - The same measurement on the primary GPU, the RTX PRO 6000, gives a marginal cost of +2.7 µs and −3.7 µs, i.e. zero within noise (`e9_placement_pro6000_r3.json`, `decision_cost`).
  - On that GPU, "g > o is out of reach" cannot be claimed for device-side policies.
  - The exclusion then rests entirely on the information tests, which cover six statistics and depth-limited learners. The lemma's "every statistic at once" guarantee does not carry the conclusion.

- **Why it matters:** The abstract presents the headroom result as a general cost-side exclusion. It holds only for host-side decisions against an impractical key.

- **What would resolve it:** Report the PRO 6000 reduction cost. Restate the abstract and Section 6.7 as an information result with a host-side cost corollary.

- **Effort:** text.

### C9. MAJOR: The evaluation workload is unrepresentative, and per-call weighting hides where models fail

- **Location:** Section 5.2; Sections 6.2 to 6.4; Table 5; Supplement S1.2.

- **Evidence from my recomputation** (`e2_mamba1_pro6000_r4spin.csv`):
  - The 2360 captured calls span only 57 distinct (length, batch) shapes: 34, 13 and 10 distinct lengths at batch 1, 4 and 16.
  - 79.5% of calls have power-of-two lengths, an artifact of truncation in rotation. All of these are multiples of 16 and hit the aligned compiler variant. In real traffic only about 1 in 16 lengths would.
  - 773 calls (33%) lie exactly on the twelve calibration keys.
  - Twelve shapes hold 65% of the calls; 160 calls share one length (25).
  - The cost model's regret differs by alignment: unaligned 1.01/2.25/4.24% against aligned 0.22/3.89/2.69% (PRO 6000 / RTX 3090 / H800). The ranking therefore depends on this mix.
  - Batch-1 calls are 65% of the data. On the PRO 6000 the median oracle latency is 60 µs, and 20% of calls take under 20 µs in device time.
  - Batched calls reuse prompts: 177 of 488 batch-4 calls contain fewer than 4 distinct prompts.
  - Latency-weighted shares (for example the 1.0 to 5.9% claim) depend on the authors' truncation mix.

- **Why it matters:** Model accuracy, table accuracy and latency-weighted shares are all evaluated on a designed grid, not a serving distribution.

- **What would resolve it:**
  - Evaluate on lengths and batch compositions sampled from the public traces the paper already replays, with natural unaligned lengths and workload weighting.
  - Report per-batch and shape-weighted results in the main tables.

- **Effort:** offline analysis (reweighting) plus new GPU measurement (natural-length calls).

### C10. MAJOR: The measurement platform limits validity and reproducibility

- **Location:** Section 5.2; Section 6.5; Section 6.8; Limitations; Supplement S1.1, Tables S1, S2, S8 text, S5 text.

- **Evidence about the RTX 3090 platform:**
  - It is a non-stock 48 GB board in a cloud container where "virtualization or time-slicing below the container cannot be excluded".
  - The software power cap was active in 38 to 92% of readings in several sweeps (S1.1).
  - In the shard sweep the per-round argmin agrees across rounds for only 52% of shapes. Its "rounds differ by more than the chunk effect", and a retuned table loses 14.9% (Section 6.5).

- **Evidence that key experiments ran only on that GPU:** the reference-scan sweep, the routed study, the larger checkpoints, the production kernels at larger widths, TFLA and the autotuner-state test.

- **Evidence on host-side reproducibility:**
  - The host CPU, frequency governor and host load were not recorded (S1.1), although the host-bound regime is a property of the host.
  - Launch-inclusive results did not reproduce across sessions on the PRO 6000: the shape tree went from 0.7% to 4.7% and the model from 0.9% to 4.9%, and the median oracle latency from 60 to 98 µs (S8).

- **Further gaps:**
  - Several data sets were timed without the exclusive-GPU guard (Table S1).
  - No datacenter GPU was used for SSD or Gated DeltaNet; the H800 runs only the in-house kernel.
  - Two different `mamba_ssm` versions were used across GPUs.

- **Why it matters:** The production-kernel conclusions rest on consumer or workstation GPUs with poorly controlled timing. The host-bound conclusions cannot be reproduced without the host description.

- **What would resolve it:** Repeat the production-kernel graph-replay and packed experiments on datacenter GPUs (H100, H200 or B200), with the guard, clock and host state recorded, and report the host CPU for all launch-inclusive data.

- **Effort:** new GPU measurement plus text.

### C11. MAJOR: The scope tunes the chunk alone, although the full launch configuration matters as much

- **Location:** Section 6.4 ("Auto-tuning baseline"); Section 5.3 conventions box (cold L2); Recipe; Limitations ("no model replaces a search of the full launch space"); Table S17.

- **Evidence:**
  - The best entry of the fixed launch table is 7 to 18% behind full launch-space tuning on the RTX 3090, the H800 and in device time, and up to 53% behind at a single key.
  - The best full configuration differs between warm and cold L2 at 7 of 12 keys.
  - The cost model used as a full autotuner loses 8.9 to 23% on average (53 to 95% maximum) when one key is left out (S17).
  - Any production tuning cache (for example Triton's autotuner, which the paper cites) tunes the full configuration per key. The recipe itself says to time the full configuration.

- **Why it matters:** A cache that tunes the chunk alone is a strict subset of standard practice. The chunk-only regrets and models answer a question a practitioner would not ask in isolation.

- **What would resolve it:** Use the existing 184-configuration sweeps at the twelve keys to report chunk selection jointly with full configuration tuning. Alternatively, justify the chunk-only scope (for example, library APIs that expose only the chunk) and restrict the claims.

- **Effort:** offline analysis plus text.

### C12. MAJOR: Clarity and density

- **Location:** whole paper; especially the abstract, Section 6 opening, Tables 2 and 7, Section 6.7.

- **Evidence:**
  - The abstract (about 270 words) is a chain of qualified numbers. For example: "a data-dependent policy can gain at most the residual regret of a shape-keyed table, which on the scans with a steep chunk curve is below the cost of one host–device synchronization at every tuned shape".
  - The paper uses at least five timing conventions (launch-inclusive, device time, graph replay, warm or cold L2, single block or round medians) and several sessions.
  - "Best static" has four definitions: per-call held-out, by total latency, in-sample, and per batch.
  - Table 7's "Default" column means the library default for SSD and Gated DeltaNet but the best single chunk for Triton.
  - Coined terms such as "form-blind", "covered" and "tuned shapes" go undefined.
  - The main narrative depends on roughly 40 supplementary tables.

- **Why it matters:** A TACO reader cannot easily extract what to do, or how large the effects are, for the kernels they actually use.

- **What would resolve it:** Rewrite around one primary convention and one baseline definition. Move secondary conventions to the supplement, and put one summary table per question in the main text.

- **Effort:** text.

### C13. MINOR: The same quantity appears with different values across tables

- **Location:** Table 5, Section 6.4, Tables S14, S15, S19, S30, S31.

- **Evidence:**
  - PRO 6000 device-time cost-model regret is 0.40% in Table 5, Section 6.4 and Tables S30/S31 (`tables_iv_v_ci.json`: 0.4047), but 0.38% in Tables S14, S15 and S19 (`taco_model_baselines.json`: 0.3827). Its 95th percentile is 3.8 against 3.6.
  - MLP: 0.37 against 0.35.
  - Top-2 and top-3: 0.23/0.04 in Table 5 against 0.21/0.02 in Table S15.
  - RTX 3090 top-2 and top-3: 0.72/0.55 against 0.67/0.51.

- **Why it matters:** Two analysis pipelines produce different headline values for the same selector and session.

- **What would resolve it:** Reconcile them, or state which session or oracle each table uses.

- **Effort:** offline analysis plus text.

### C14. MINOR: Inaccurate qualifiers in the abstract

- **Location:** Abstract.

- **Evidence:**
  - "0.4 to 3.5% when its terms are selected on the same GPUs' calls" includes the H800 (3.0%), whose calls did not select the terms.
  - "As under CUDA graphs on two workstation GPUs" includes a GeForce RTX 3090, a consumer board.

- **What would resolve it:** Correct both qualifiers.

- **Effort:** text.

### C15. MINOR: The "shape" feature set includes the layer index

- **Location:** Section 6.3 ("shape (length, batch, width, layer)"); code `e3_predictors.py` (`SHAPE = ["L","batch","width","layer"]`).

- **Evidence:** The layer index is not part of the shape key defined in Section 3. I re-ran the trees without it and the regrets are identical (0.465 PRO device time, 0.383 RTX 3090 device time, 1.626 H800, 0.732 PRO launch-inclusive). The concern is therefore only descriptive, but the feature set should match the definition.

- **What would resolve it:** Describe the feature set accurately, or drop the layer index.

- **Effort:** text.

### C16. MINOR: Lemma 3.1 overstates the theoretical content

- **Location:** Section 3.

- **Evidence:** The "lemma" is the observation that no chunk is faster than the minimum over chunks.

- **What would resolve it:** Present it as a definition or observation, not a lemma with proof.

- **Effort:** text.

### C17. MINOR: Related work misses the closest production practice

- **Location:** Section 2.

- **Evidence:** No citation or discussion of:
  - PyTorch 2 / TorchInductor max-autotune with dynamic-shape bucketing.
  - vLLM's per-device tuned configuration files for its fused-MoE Triton kernels, keyed on token-count buckets. This is essentially the paper's recipe, in production.
  - Recent autotuners such as BaCO, GPTune and ytopt.

- **Why it matters:** These practices sharpen the novelty concern in C1.

- **What would resolve it:** Discuss them and position the recipe against them.

- **Effort:** text.

### C18. MINOR: Output divergence is under-weighted as a deployment cost

- **Location:** Section 7 ("Switching chunks is not output-neutral"); Supplement S11, Table S36.

- **Evidence:** In half precision, another chunk changes the greedy output of 2.6 to 9.4% of prompts, and a per-length table changes up to 4.7%. This is a first-order cost of any per-shape chunk switching, for reproducibility and batch-invariant serving. It appears only in the discussion.

- **What would resolve it:** Move it into the recipe and the abstract-level trade-off.

- **Effort:** text.

### C19. MINOR: Mixed baseline definitions within tables

- **Location:** Tables 7 and 8; Section 6 opening.

- **Evidence:**
  - Table 7's "Default" column mixes library defaults with the Triton kernel's best single chunk.
  - Table 8's "Best static" is chosen by total latency per batch, while Table 5's is held out by per-call regret.
  - The Section 6 opening mixes both ("20% (27%)").

- **What would resolve it:** Use one definition throughout and label any exceptions.

- **Effort:** text.

### C20. MINOR: The end-to-end overhead of the refuted design is measured through a deliberately unoptimized implementation

- **Location:** Section 6.6; Table S39; Supplement S15.

- **Evidence:** The full histogram is "not bandwidth-optimal", and two variants pay the sampled histogram even when they do not use it. The 19 to 191% prefill penalties therefore say more about this implementation than about data-dependent dispatch.

- **What would resolve it:** Present them as an upper bound for one implementation. The information side of the argument (C8) is the real evidence.

- **Effort:** text.

---

## 5. Answers to the specific attack questions

- **Is the contribution significant or obvious?** Mostly obvious; see C1. For data-oblivious kernels, "shape sets the chunk" is the null hypothesis, and every autotuner already keys on shape and device.
- **Is the headline effect an artifact of the in-house kernel?** Largely yes; see C2 and C3. The steep curve (8.5× spread, 49% default loss, 4.4 to 25 ms per forward pass) belongs to a kernel that is 1.23× slower than the reference, uses a launch table tuned on another GPU, and spills registers.
- **Does production-kernel evidence support the claims?** Only partly; see C5 to C7.
  - Host-bound: defaults are fine.
  - Device-bound: 1 to 6% of scan time in isolated graph replay.
  - No engine or end-to-end validation, and the packed-call result contradicts the recipe.
- **Internal contradictions?**
  - The packed-call result against Contribution 2 and the recipe (C5).
  - "g > o out of reach" against the PRO 6000's zero-cost reduction (C8).
  - Main-text value-invariance summaries omit the H800 (C7).
  - Numerical mismatches across tables (C13) and inaccurate abstract qualifiers (C14).
- **Cherry-picked conventions or sessions?**
  - L = 8192 is the only length used for the placement headline (C7).
  - Median, not worst-case, equivalence bounds (C7).
  - RTX 3090 values only for the device-side decision cost (C8).
  - The frozen, in-sample-selected cost-model form is compared against a priori baselines (C4).
  - The "default 256" baseline for a kernel without a default (C3).
  - To their credit, the authors state headline numbers in device time and explain why.
- **Do the statistics support "no practically relevant information"?**
  - Yes for the Mamba-1 kernels: paired intervals [0.00, 0.00], and exact-key splits that are statistically detectable but practically nil. For example, on the 1.4B device-time data a variance split is Holm-significant (p = 0.018) but changes regret by only 0.004 points.
  - Not established for SSD and Gated DeltaNet above the floor. Two of three device-time SSD data sets are "inconclusive" at the authors' own 0.5-point margin, and the H800 shows large, unexplained input effects (C7).
- **Is the cost model's value real?** No, once calibration cost and a priori term selection are included; see C3 and C4.
  - It beats a free length rule by only 1.5 to 1.7 points on two of three GPUs.
  - It is beaten on average by its own three-parameter restriction.
  - It costs 130 to 843 s per GPU, against seconds to profile every key.
- **Overclaims?**
  - Abstract-level generality of value invariance (C7).
  - The cost-side headroom exclusion (C8).
  - "A shape-keyed table recovers what the defaults lose" (C5).
  - The non-disclosure of the kernel's provenance in the abstract (C2).

---

## 6. Verification log (recomputed from the released files; no GPU used)

| Check | Source | Result |
|---|---|---|
| Triton/reference geometric mean | Table S9 ratios | 1.229 (paper: 1.23) ✓ |
| CUDA-graph default latency-weighted (LW) shares | `taco_e26_analysis_pro6000.json` | 1.75/5.85/3.19 (PRO), 2.80/2.20/1.02 (3090), range 1.0 to 5.9 ✓ |
| Cost model, device time | `taco_model_baselines.json` | 0.383/3.554/3.008 ✓ (Table 5 PRO shows 0.40; see C13) |
| A priori selections and protocol spread | same, M4c | PRO 0.38 to 4.16, 3090 0.88 to 6.23, H800 2.14 to 3.01; H800 `add_l2_miss` equals frozen form |
| Calibration payback in keys | same, M5b | 1622 to 2475 (paper: "about 1600 to 2500") ✓ |
| Cost model per batch | `cost_model_*_r3spin.json` | RTX 3090 batch 16: 15.9%; H800 batch 4: 6.3% |
| Cost model aligned / unaligned | same | 0.22/1.01 (PRO), 3.89/2.25 (3090), 2.69/4.24 (H800) |
| Length rule `clip(nextpow2(L),32,1024)` | `e2_mamba1_*` per-call tables | per call 5.86/7.76/10.66%; LW 5.96/16.31/13.94% (device time) |
| Shape tree with and without the layer index | rerun of `e3_predictors.evaluate` | identical (0.465/0.383/1.626/0.732) |
| Captured-call shape structure | `e2_mamba1_pro6000_r4spin.csv` | 57 shapes; 79.5% power-of-two lengths; 773 calls on calibration keys |
| H800 value invariance | `e1_analysis_h800*.json` | 28/36 (launch-inclusive), 36/36 (device time) significant; best chunk flips at L = 2048 |
| PRO 6000 device-side reduction cost | `e9_placement_pro6000_r3.json` | +2.7 µs (L = 2048), −3.7 µs (L = 8192); RTX 3090: 7.3/42.0 µs ✓ |
| Packed calls | `e27_varlen_pro6000_taco.json` + raw rounds | SSD 370M table worse than default; per-round argmin stable for 34% of steps |

*End of report.*
