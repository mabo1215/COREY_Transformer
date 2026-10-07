# Devil's Advocate Review — TPDS Round 4

**Manuscript:** "Shape Sets the Chunk: A Compile-Informed Cost Model and a No-Gain Bound for Data-Dependent Dispatch of Chunked Linear-Recurrence Kernels" (main body 14 pp. + Supplementary Material 16 pp.)
**Venue:** IEEE Transactions on Parallel and Distributed Systems, regular paper, double-anonymous
**Reviewer role:** Devil's Advocate (adversarial; independent of the other four panel members)
**Sources read:** `paper/main.tex`, `paper/appendix.tex`, `paper/ref.bib` only. No prior review, progress file, or source code was consulted.

Section, table, equation and figure numbers below follow the main paper (Sec. 1 Introduction ... Sec. 10 Conclusion; Tables 1–7; Eqs. 1–7; Prop. 1) and the supplement ("S-" prefix, by label where the number is not fixed in the source).

---

## 1. Scores (0–100)

| Dimension | Weight | Score | Two-line justification |
|---|---|---|---|
| Novelty | 20 | **55** | The no-gain bound is, in the authors' own words, "elementary" (Sec. 4): part (a) is a one-line consequence of the definition of a minimum and part (b) restates that dense kernels are data-oblivious. The cost model is a sensible but incremental instance of the Hong–Kim/Volkov tradition with compiler-reported inputs; the central negative result refutes the authors' own earlier scheduler (COREY) rather than a position anyone in the literature defends. |
| Methodology | 25 | **62** | The timing discipline, placement controls, TOST tests and the recurrence-preserving pitfall section are careful and candid. But the cost-model form was selected on the test calls (Sec. 5, "not for the choice of terms"), the reference-scan chunk sweep is confounded with thread count (Sec. 6.5), the headline chunk-sensitivity numbers come from a warm-L2 microbenchmark the paper itself shows produces up to 43% placement artefacts (Tables S-placement_pro6000, S-l2), and the decisive information-side test uses depth-3 trees. |
| Experiments | 25 | **66** | Three GPUs, four kernels, 2360 captured calls per checkpoint, replication, tensor-parallel widths, larger widths and five serving traces: the volume is well above the TPDS norm. The gap is in baselines, not in volume: no TFLA instantiation, no interpolating or learned model fitted on the calibration data alone, no fused or lagged statistic, and a serving analysis that is a simulation. |
| Clarity | 15 | **65** | The English is precise and the argument structure (bound → model → results) is clear. The reader must, however, track three GPUs, two timing conventions, four kernels and two key definitions through an abstract and results section that are walls of percentages; several headline sentences are only true for one GPU or one kernel, and the reader has to find that out in the supplement. |
| Significance | 15 | **48** | For the kernels people actually run, the paper's own data says the default chunk is within 0.24% (Mamba-2 SSD), 0.05% (Gated DeltaNet) or 6.6% (reference Mamba-1 stock dispatch) of the per-call oracle. The cost model and the bound therefore change nothing for a serving system today; the actionable residue is measurement guidance and a cold-key default for one register-tiled Triton scan. |
| **Weighted overall** | | **60** | 0.20×55 + 0.25×62 + 0.25×66 + 0.15×65 + 0.15×48 = 59.95 → Major Revision band (50–64). |

---

## 2. Recommendation: **Major Revision**

Decisive reasons, in order of weight:

1. **Scope–evidence mismatch (Challenge 1, CRITICAL).** The title, abstract, keywords ("LLM serving") and conclusion speak of chunked linear-recurrence kernels in general, but the only kernel on which the optimal chunk moves with the shape by enough to need a table, a model or a bound is the Triton scan the authors wrote for this study (and, confounded with thread count, the rebuilt reference scan). The paper concedes this in one sentence of Sec. 9 and nowhere else. As submitted, a TPDS reader is told to deploy shape-keyed tables plus a cost model for kernels on which the paper's own measurements show a fixed default chunk is already within 0.25%.
2. **The cost model's advantage over trivial selectors is one-GPU-strong and the model form was chosen on the test calls (Challenges 4–5).** On the RTX 3090 a three-parameter restriction with a far worse fit selects better than the eight-parameter model; on the H800 a length-only tree and the nearest calibration key tie it. The closest prior model (TFLA) and the obvious interpolation baselines are not compared.
3. **"Cannot help" is overstated (Challenge 3).** The cost-side exclusion is established only for a blocking host synchronization. The paper's own strongest result is quantitative — any data-dependent policy can gain at most 0.9% over a bucket table and 0.2% over an exact-shape table, before paying any decision cost — and that is the sentence that should appear in the abstract and conclusion.

None of these requires new science; most are text or offline analysis. One is strengthened substantially by a modest new measurement (an independently written public kernel on which the optimum moves). With those changes, the paper is a careful, honest, negative-result systems paper with one useful cold-key model; without them, its headline claims outrun its evidence.

---

## 3. Strongest counter-argument (what a hostile expert would write)

The paper does three things well: it measures carefully, it documents its own limitations with unusual candor, and it correctly identifies a measurement pitfall (split-call timing) that would mislead anyone tuning these kernels. That said, here is the case against it.

The paper discovers that a dense kernel's latency does not depend on the values it processes, states this as a proposition whose proof is one line, and then spends fourteen pages verifying it on a scheduler that the authors built and that no one else uses. The bound's numerical "verdict" is a comparison of a sub-microsecond headroom against a 20 µs blocking synchronization that a competent implementation would never pay; the paper admits that fused, overlapped and device-side conditional mechanisms cost less and does not measure any of them. The information-side defense is the real result, and it is small: 0.9% over a bucket table. Meanwhile the positive contribution — a cost model — is demonstrated chiefly on a Triton kernel written for this study, whose 8.5× chunk sensitivity arises from register spills at large tiles, i.e., from how this particular kernel was written; on the Mamba-2 SSD and Gated DeltaNet kernels that people run, the library default chunk is within 0.25% of the oracle and nothing in the paper is needed; on the reference Mamba-1 CUDA scan, the stock length dispatch is within 6.6% and the cost model (3.8%) is *worse* than a shape table (3.0%). The model's advantage over a nearest-key lookup or a length-only tree holds clearly on one of three GPUs — the one whose captured calls were used to select the model's terms. The serving analysis is a simulation whose per-key costs are mapped to the nearest of 33 measured keys. The "so what" for a serving-system engineer is: keep the default chunk; if you run a Mamba-1 Triton scan under tensor parallelism and cannot profile six chunks for 0.05 s per key, calibrate this model instead. That is a workshop-sized conclusion carried by a journal-sized measurement campaign.

---

## 4. Challenges

Effort classes: (a) text only; (b) offline analysis of existing results; (c) new GPU measurement with estimated GPU-hours.

### C1 — CRITICAL — The chunk effect the machinery addresses exists only on the authors' own kernel

**Claim attacked.** Title; Abstract ("Chunked linear-recurrence kernels such as the Mamba selective scan expose a chunk size that trades on-chip reuse against occupancy ... Runtimes tune it offline per shape, which scales poorly when continuous batching, chunked prefill, model parallelism and mixed GPU fleets multiply the shapes"); Conclusion, Sec. 10 ("Chunk selection should be keyed on the complete shape and device, with such a model for the shapes a tuning cache has not yet seen").

**Counter-argument.** The paper's own data give a more parsimonious story for three of the four kernels: *use the default chunk*.
- Mamba-2 SSD at 370M: chunk spread 1.01–1.25× (Table 1); best single chunk within 2.0% (Table 2). At 2.7B above the launch floor: "chunk 256 is ... the per-call oracle of 375 of the 397 captured calls above the floor, where it loses at most 0.24% at any batch size" (Sec. 6.5). Chunk 256 is the library default (Sec. S-breakeven).
- Gated DeltaNet: chunk 64 "is the fastest at 73 of 75 synthetic shapes and the oracle of 154 of the 155 captured calls above the floor"; static regret 0.00/0.00/0.05% (Table S-largewidth_calls).
- Reference Mamba-1 CUDA scan: "the stock length-based dispatch, itself a shape rule, comes within 6.6% of the per-call oracle" (Sec. 6.5); under tensor-parallel sharding "the single-GPU table transfers (2.2% pooled; cost model 4.0%)".
- Only the Triton scan "written for this study" (Sec. 5.2) shows the 8.5× spread, the 13–24% static regret at batch 4/16 (Table 2) and the 63% sharding loss (Table S-tp). Its sensitivity is traceable to its own design: 255 registers plus spills at C ≥ 512 (Table S-model_compile), which the reference scan avoids with a fixed 168-register count.

So for every kernel in production use, neither a shape table nor a cost model nor a no-gain bound changes the chunk a system would pick. The conclusion's recommendation is unsupported for those kernels and the abstract's motivation ("scales poorly") describes a problem the paper shows does not arise there.

**What the paper already answers.** Sec. 9 concedes: "Only the Mamba-1 scan has an optimal chunk that moves with the shape by enough to matter at the sizes we study ... The per-call, predictor and cost-model results therefore rest mainly on Mamba-1, measured on a Triton kernel written for this study." This is honest but confined to Limitations; the title, abstract, contributions and conclusion do not reflect it.

**What would settle it.** (a) Rescope title/abstract/conclusion to register-tiled Triton scans and present "defaults suffice on SSD/GDN" as a positive finding rather than a limitation. (c) The convincing fix: show one independently written, publicly used chunked kernel on which the optimum moves with shape or width — e.g., the FLA library's Mamba-1/GLA chunk kernels or its SSD implementation, swept over chunk at the 12 tuning keys and the shard widths. Estimated 4–6 GPU-hours on the RTX 3090.

**Status.** Unresolved in the submitted text. Resolvable; primarily (a), decisively (c).

### C2 — MAJOR — The reference-scan chunk sweep measures thread count, not chunk

**Claim attacked.** Sec. 6.5: "The reference scan is at least as chunk-sensitive as ours (Table 4): the slowest chunk of a call is 5.4× the fastest at the median"; Contribution 1: the cost model "extends to the reference CUDA scan".

**Counter-argument.** The same paragraph states: "the thread count changes with the chunk, from 32 to 128, and the sweep measures the chunk as the stock dispatch realizes it rather than the chunk alone." A 4× change in threads per block over the same per-channel work can by itself account for a large part of a 5.4× spread. The claim "at least as chunk-sensitive" is therefore not established; what is established is that the stock *dispatch configuration* is sensitive. This matters because the reference scan is the only Mamba-1 kernel in production use, and it is the paper's only evidence that the Triton kernel's sensitivity is "a property of chunked scans" rather than "of that kernel" (the question Sec. 6.5 itself poses).

**What the paper already answers.** The confound is stated in Sec. 6.5 and Sec. 9, and the cost-model instance for the reference scan includes n_t explicitly (Sec. S-cost_model). The headline sentence is nevertheless retained.

**What would settle it.** (c) Re-sweep the runtime-chunk reference scan with the thread count held fixed (one sweep at 32 threads for all chunks, one at 128), report spread and oracle. Estimated 2 GPU-hours on the RTX 3090. Or (a) delete "at least as chunk-sensitive as ours" and state the confound in the abstract-level claim.

### C3 — MAJOR — The no-gain bound is definitional, and its stated verdict depends on a blocking synchronization that cheaper mechanisms avoid

**Claim attacked.** Abstract: "Data-dependent dispatch ... cannot help"; Prop. 1 and its proof ("The bound is elementary"); Sec. 6.8: "any host-side decision at least one synchronization (20 µs) ... So g > o is out of reach."

**Counter-argument.** Part (a) says that no policy can beat the per-call minimum by more than the distance of the table from that minimum, which is the definition of a minimum. Its entire content is in the two measured numbers, and the cost number is chosen as a *blocking* device-to-host readback. Four mechanisms the paper names or implies pay essentially nothing on the critical path and are not measured:
1. *Lagged asynchronous readback*: copy the layer-l statistic to pinned memory with an asynchronous copy, poll the event without blocking, and use it to pick the chunk of layer l+1 (or of the next chunked-prefill slice of the same request). No synchronization, no extra kernel on the critical path.
2. *Fused statistic* in the epilogue of the causal-convolution kernel (Sec. 4: "could cost less, and we do not measure such implementations").
3. *Device-side conditional graph nodes* (Sec. 4, same sentence).
4. *Reuse of a prefill statistic across chunked-prefill slices or decode* (not discussed).
For all four, o → 0 and part (a) is silent; the device-side reduction cost quoted (7.3 µs at L = 2048) is measured "at batch 1 only, where it costs more than the pooled headroom, so for them the comparison is indicative rather than per shape" (Sec. 4). The categorical "cannot help" is therefore not what the cost side establishes.

**What the paper already answers.** The information side, and it is the stronger argument: the held-out headroom over a bucket table is 2.5 µs (0.9%) and over an exact-shape table 0.45/2.2/2.2 µs per call, at most 4.4/6.7/14.5 µs at any tuned shape (Sec. 6.8); split-half estimates are 0.13–2.1 µs (Table S-headroom_noise); shape + entropy equals shape on every kernel and GPU (Tables 3, 4, 7); conditional mutual information 0.046 bits (Sec. 6.3). Sec. 4 also says explicitly that for zero-cost mechanisms "the argument rests on information rather than cost." This answers the core of the attack. What remains is the overstatement: the defensible sentence is "any data-dependent policy, whatever its decision mechanism, can gain at most 0.9% over a bucket table and 0.2% over an exact-shape table on these kernels", not "cannot help".

**What would settle it.** (a) Replace "cannot help" in abstract and conclusion with the quantitative bound and state that it holds for o = 0. (c) Optional and cheap: a lagged-readback variant in the routed study (decide layer l+1 from layer l) to retire the cost-side argument altogether; ~1 GPU-hour on the RTX 3090.

### C4 — MAJOR — The cost-model form was chosen on the test calls, and where the test is clean the model ties trivial selectors

**Claim attacked.** Contribution 1: "Without timing any of the target shapes, it selects the chunk of 2360 captured prefill calls with 0.9% mean regret on an RTX PRO 6000 and 3.2% on an RTX 3090"; Sec. 6.4: "The model adds information beyond its calibration data".

**Counter-argument.** Sec. 5 states: "The terms of Eq. (3) were chosen with an ablation that scored candidate terms on the captured calls of the RTX PRO 6000 and the RTX 3090 ... The captured calls are therefore a clean test set for the fitted parameters ... but not for the choice of terms." The headline 0.9% is on the GPU on which the form was selected. Where the comparison is cleaner, the eight-term model has no clear advantage:
- RTX 3090: "A three-parameter restriction ... selects chunks with ... 2.5% on the RTX 3090, better than the full model there, although it fits the RTX 3090 calibration data far worse (R² 0.64 against 0.97)" (Sec. 6.4). A worse-fitting simpler model selecting better is the signature of terms that are not validated out of sample.
- H800 (frozen form, Sec. S-h800): model 3.5% vs "a tree on the length alone reaches 4.0% and the launch-and-work-only restriction 4.1%"; at shard widths "a table tuned at the calibration width (2.2%) and the nearest calibration key (3.7%) do as well as the model" (2.9%).
- Device time on the RTX 3090 (the convention closest to a CUDA-graph deployment): cost model 3.6% vs shape tree 0.4% (Table S-conventions).
The paper's gloss — "a good fit of the calibration data does not by itself make a good selector" — concedes the point without drawing the conclusion: the five parameters beyond the launch-and-work restriction are justified on one GPU.

**What the paper already answers.** The H800 is a genuine frozen-form test (3.5%); leave-one-key-out regret is 1.0%/2.0%; cross-GPU parameter transfer gives 2.5–5.4%; the ablation (Table S-model_ablation) shows each removed term hurts on at least one GPU. These support "the model is not grossly overfit"; they do not support "the full form is needed".

**What would settle it.** (b) For all three GPUs, one table with: full model, 3-parameter restriction, nearest calibration key, log-linear interpolation of the 12 calibration keys in (log L, log B), and a length-only tree — all fitted on calibration data only and scored on the captured calls. (b) Freeze the form on the PRO 6000 alone and re-report the RTX 3090 as a clean test. No GPU time.

### C5 — MAJOR — The closest competing model (TFLA) and the natural learned baseline are not compared

**Claim attacked.** Sec. 2: "Tiled Flash Linear Attention (TFLA) is the closest analysis to ours. It models the runtime of chunkwise linear-RNN kernels from the architectural dimensions and the arithmetic intensity of the accelerator and derives the chunk size that minimizes it"; Sec. 2: learned cost models such as Ansor's "are trained on such timings per operator and device; the analytical model of Section 5 needs no timing at a new shape".

**Counter-argument.** TFLA's model is, by the paper's description, a function of shape and hardware with no fitted timings; instantiated for this scan it is a direct competitor to Eq. (3) and would show whether compile facts add anything over architectural constants. Equally, a gradient-boosted regressor on (shape, launch configuration, compile facts) fitted on the same 2298 calibration timings is the "learned" alternative the related-work paragraph alludes to; it needs no timing at a new shape either. The only interpolator the model is compared with is the weakest one (nearest key, 5.2%/4.5%/14.4%). "The model's value lies in interpolating between calibrated shapes" (Sec. 6.4) is thus shown against a straw interpolator. A TPDS reviewer will cite TFLA and ask for the number.

**What the paper already answers.** Nothing numerical for TFLA; the nearest-key row only.

**What would settle it.** (b) Instantiate TFLA's runtime expression for the Mamba-1 scan at the captured shapes; fit a GBT on the calibration set; score both on the captured calls alongside Eq. (3). Offline.

### C6 — MAJOR — The abstract carries the most favorable number of each pair

**Claim attacked.** Abstract and Contributions.

**Counter-argument.** Each of the following is a choice between two numbers the paper reports, and each time the abstract takes the favorable one:
1. "0.9% ... 3.2%" are means; the 95th percentiles are 4.0% and 15.4%, and on the RTX 3090 only 69% of chosen chunks are tied with the oracle (Table S-model_params). The H800's 3.5% appears only in the conclusion.
2. "a table tuned for one GPU loses up to 63%" is the t = 8 shard of the 1.4B model on the authors' Triton kernel (Table S-tp). On the reference scan the same table loses 2.2% pooled and at most 11.2%; at width 1024 the cost model itself loses 15.2% against the table's 17.3%. The phrase "tuned for one GPU" also invites a cross-GPU reading, whereas the cross-GPU loss is 48.8% (Sec. 6.9).
3. "about 2% of calls fall on keys rare enough for a histogram policy to beat tuning" is for the every-8th-element histogram (o = 0.19–0.72 ms). For a policy that pays one synchronization, the body reports that keys including the batch of a budget step "leave up to 42 and 93%" of calls below break-even (Sec. 6.8). The abstract's next sentence ("the cost model covers keys not yet tuned") only makes sense in the light of the 93%, which the abstract omits.
4. Launch-inclusive timing is used for every headline. In device time on the RTX 3090 — the convention the paper says "approximates the cost inside a CUDA graph" — the cost model is 3.6% against 0.4% for the shape tree (Table S-conventions), i.e., nine times the table's regret; the abstract reports neither.
5. "Tied" shares (90%, 94%) are computed from overlapping 95% confidence intervals (Sec. 5.4), a criterion that is conservative for declaring a *difference* and therefore lenient for declaring a tie; the paper calls it "conservative" in the opposite sense.
6. The reference-scan comparison in the abstract ("extends to the reference CUDA scan", Contribution 1) omits that there the cost model (3.8%) is worse than the shape tree (3.0%) and only 2.8 points better than the stock dispatch (6.6%).

**What the paper already answers.** Every number is present in the body or supplement; nothing is hidden. The objection is to selection at the abstract level.

**What would settle it.** (a) Abstract and contributions carry mean and p95, all three GPUs, both kernels for the sharding claim, both policies for the trace claim, and the device-time result. No GPU time.

### C7 — MAJOR — The headline chunk-sensitivity and headroom numbers come from a warm-L2 microbenchmark that the paper shows produces up to 43% placement artefacts

**Claim attacked.** Sec. 6.1: "The chunk changes latency by up to 8.5× on Mamba-1"; Sec. 6.2: "the ratio between the slowest and the fastest chunk of a call has a median of 2.6× (95th percentile 8.1×)"; the per-call oracle and the headroom of Sec. 6.8.

**Counter-argument.** Sec. 5.4: "Repeats run back to back without flushing the L2 cache." Sec. S-invariance_details: on the RTX PRO 6000 at L = 8192 "identical values at different addresses differ by a median of 21% and up to 43.6%"; "With a warm L2 the placements differ by 29 to 41% at chunks 32 to 256 ... with a cold L2 the same placements differ by 0.5 to 2.3%" (Table S-l2); and "In serving, where other kernels run between two scans of a layer, the warm-L2 advantage of a favorable placement is not guaranteed either." The per-call oracle, the chunk spread at L = 8192 and the exact-shape headroom are therefore properties of a back-to-back, warm-L2 microbenchmark in which each buffer's placement effect is "fully reproducible" (same section); the paper does not report the chunk curve, oracle or headroom under a cold L2, which is the serving-relevant condition. A reader cannot tell whether the 8.5× is 6× or 10× cold, nor whether the headroom at L = 8192 is unchanged.

**What the paper already answers.** At the selected chunks (512, 1024) the warm-L2 placement range is 8.2% and 2.6%; the input-agnostic chunk stays tied with each input's best in 90–100% of cells under every control; the value-vs-placement attribution is itself a strength of the paper. The decision is probably robust; the magnitudes are not shown to be.

**What would settle it.** (c) Re-time the 12 tuning keys × 6 chunks on the RTX PRO 6000 with the L2 overwritten between repeats (the intervention already implemented for Table S-l2) and report spread, oracle and exact-shape headroom under both conditions. Estimated 2 GPU-hours.

### C8 — MAJOR — The information-side "no gain from entropy" test has too little capacity to find a gain

**Claim attacked.** Sec. 6.3: "The decisive comparison is between the shape and the shape plus activation statistics: on Mamba-1 ... adding entropy or all activation moments to the shape leaves the regret at 0.7%."

**Counter-argument.** Sec. S-kernel_experiments: "Every predictor except the entropy rule is a classification tree of depth at most 3 with at least 5 calls per leaf." Eight leaves are consumed by splits on length and batch (the shape tree already needs them to reach 0.7%); an entropy split can only enter by displacing a shape split. A null result from "shape + entropy" is therefore partly guaranteed by model capacity, and a reviewer can reasonably say the test could not have found a 0.3-point gain even if one existed.

**What the paper already answers.** Adequately, by a different route: the exact-shape held-out headroom (0.45 and 2.2 µs per call; 0.13 and 1.8 µs split-half) bounds what *any* per-call feature can add over the exact shape, independent of tree capacity, and the conditional MI excess (0.046 bits vs 1.23 bits for log length) points the same way. The tree row is redundant evidence presented as the "decisive comparison".

**What would settle it.** (b) Re-run the predictor study with a gradient-boosted model or depth-6 trees, or lead Sec. 6.3 with the headroom and present the tree as corroboration. Offline.

### C9 — MAJOR — The cost model does not pass the paper's own break-even test against profiling

**Claim attacked.** Sec. 7: "Calibrating it once per device type and kernel and using it as the default for keys that a cache has not yet tuned combines the two"; Abstract: "the cost model covers keys not yet tuned."

**Counter-argument.** Sec. 6.8 gives the model's regret on the captured keys as "7 and 15 µs per call on average but above one synchronization on 5 and 10 of the 33 keys", and Sec. S-breakeven gives profiling six chunks at 0.03–0.05 s per key. Applying Eq. (6) to the model itself: a profiled entry pays back after P_k / regret ≈ 0.03–0.05 s / (7–15 µs) ≈ 2,000–7,000 calls, i.e., within minutes on keys that "receive hundreds to thousands of calls per hour each" (Sec. 6.8). The model's window of usefulness is the first minutes after a key appears, or the padded budget-step key explosion that the paper itself says to fix with a coarser key. Two further internal tensions: (i) 7–15 µs per call is 3–30× the headroom (0.45–2.2 µs) used to dismiss data-dependent dispatch, so the paper rejects one mechanism for a sub-sync loss and recommends another with a super-sync loss; (ii) on the kernels where the model is actually needed (C1), profiling six chunks takes "1.2 s" for twelve keys (Sec. 6.6), which is cheaper than the model's calibration (8–14 minutes, Sec. 6.4) for any deployment with fewer than a few hundred keys.

**What the paper already answers.** Profiling the model's top-2 chunks is proposed (0.5%/2.3%), which is a hybrid, and Sec. 6.8 states the per-key regret honestly. The break-even of the model against a table is not computed.

**What would settle it.** (b) Compute N* for the model vs. a six-chunk profile per key, and report the share of trace calls below it exactly as done for the online policies in Table S-serving. Offline.

### C10 — MINOR — The routed study headlines a histogram the paper itself declares irrelevant

**Claim attacked.** Table 5 and Sec. 6.7: "Full histogram +192.7% ... decision cost 188.8 ms".

**Counter-argument.** Sec. 6.7: "our histogram is not bandwidth-optimal, and the conclusions of Section 6.8 therefore rest on the host synchronization ... not on these histogram costs"; Sec. S-cost: the two passes read "about 1.5 MB ... a few microseconds, about two orders of magnitude below the measured 1.10 ms per call." A main-text table whose entries are ~100× an implementation floor inflates the negative result for a skimming reader and invites the reply "you benchmarked a bad histogram". The useful content of the study is the chunk-effect confirmation (chunk 128 +16.8%, predicted 17.4 ms) and the sync-only cost.

**What would settle it.** (a) Move the statistic rows to the supplement or add a bandwidth-floor row to Table 5; (c) optionally a Triton-fused histogram, ~0.5 GPU-hour.

### C11 — MINOR — Over-generalisation from single-GPU, eager-mode prefill of ≤2.8B models to "LLM serving" and tensor parallelism

**Claim attacked.** Keywords ("LLM serving"); Abstract ("transfers across GPUs and to tensor-parallel shard widths"; "In replayed public LLM serving traces").

**Counter-argument.** Sec. 9: "We do not run multi-GPU jobs"; "the tensor-parallel widths are timed with synthetic inputs on one GPU"; "The serving analysis replays request lengths through a simulated scheduler rather than running a serving system, costs packed calls as batch-1 scans, and maps shapes beyond the measured lengths and batches to the nearest measured key." In the one end-to-end study, "At 256 and 976 tokens the prefill takes about 46 ms whatever the chunk ... and no static chunk is resolved from the best one" (Sec. 6.7), so the end-to-end evidence that the chunk matters is one length (4096) on one 370M model on one GPU in eager mode. The "2360 captured calls" at batch 4 and 16 are built by reusing prompts ("a batch-16 call contains 2 to 10 distinct prompts", Sec. S-kernel_experiments), which the main text does not say. Decode is excluded by construction.

**What the paper already answers.** All of the above is disclosed in Sec. 9 and the supplement; the TP shapes are a legitimate shape test given value invariance.

**What would settle it.** (a) Narrow the keywords and the abstract to "single-GPU prefill shapes produced by parallel deployments" and state the simulated nature of the trace study in the abstract; (c) one CUDA-graph end-to-end run at three lengths, ~1 GPU-hour on the RTX 3090.

---

## 5. Ignored alternative explanations

1. **Kernel design, not "chunked scans", produces the chunk sensitivity.** The Triton kernel holds a [D_b, C] tile in registers and spills at C ≥ 512 (Table S-model_compile); the reference scan uses a fixed register count and is insensitive once thread count is held fixed (unknown — see C2). The simplest explanation of "only Mamba-1 has a moving optimum" is "only the register-tiled kernel has one".
2. **The entropy rule is constant by construction.** Eq. (2) saturates at r ≈ 0.688 and real activations give r = 0.29–0.66, so the rule returns 256 for every call (Fig. 4, Sec. S-estimator). Its failure is a property of the mapping, not evidence about information; the paper's entropy-only tree is the fair test and should carry the argument alone.
3. **Mamba-2's flat curve is partly eager-mode launch overhead.** "Without the Python-level launch cost, its latency floor at batch 1 falls from about 740 to about 320 µs" (Sec. S-conventions). Under CUDA graphs the SSD chunk leaves the floor at smaller BL; the larger-width data show the optimum still does not move, so the conclusion survives, but the stated mechanism ("bound by the fixed cost of its several kernel launches") is a description of the harness.

## 6. Unexamined premise

The paper treats "the chunk" as one knob with one meaning across four kernels: a register time tile (Triton), a sequential loop length coupled to thread count (reference CUDA), a matmul block (SSD), a Householder block bounded at 64 (GDN). The cross-kernel sentence "Chunked linear-recurrence kernels execute the same instruction and memory stream for every input ... so their optimal chunk is a function of shape and hardware" (Sec. 10) is true for all four and informative for one. The paper also assumes mean kernel latency is the objective; in serving, the SSD chunk also fixes the size of the chunk-state tensor (which exceeds 32-bit indexing at chunk 32, Sec. S-largewidth) and hence memory headroom and batch capacity — a shape-keyed consideration the bound does not cover.

## 7. Observations (not defects)

- The placement finding — identical values at different addresses differing by up to 43.6%, vanishing with a cold L2 — is the most surprising empirical result in the manuscript and is buried in the supplement. It deserves a main-text paragraph of its own; it is also directly relevant to anyone auto-tuning kernels with back-to-back repeats.
- The split-call pitfall (Sec. 5.1) is a genuinely useful warning and is the part of the paper most likely to be cited.
- The paper's candor in Sec. 9 is well above the norm. Most of the challenges above are already conceded somewhere in the text; the revision needed is to let those concessions reach the title, abstract and conclusion.

---

## 8. Steel-man: the paper's strongest defensible claim

For a dense, data-oblivious chunked scan, the per-call latency gain available to any policy that reads the activations — whatever its decision mechanism and even at zero decision cost — is bounded by the oracle headroom over an exact-shape table, and on three GPUs, two Mamba-1 kernels and the Mamba-2 and Gated DeltaNet pipelines that headroom at tuned shapes is 0.03–2.2 µs per call (at most 0.2% of a call, and within the error of re-timing one kernel), so chunk selection for such kernels is a shape-and-device problem and any activation statistic can only proxy the shape, which is exact and free. For one register-tiled Triton selective scan whose chunk curve spans up to 8.5× and whose optimum moves with length, batch, width and GPU, an eight-parameter model that reads register, spill and tile-layout facts from the compiled kernel selects the chunk at shapes it has never timed within 0.9–3.5% of the per-call oracle on three GPUs, beats a nearest-key lookup on all three, transfers to unseen checkpoint widths and to tensor-parallel shards where a width-tuned table loses tens of percent, and costs one 8–14-minute calibration per GPU instead of an entry per shape. The measurement discipline that underpins both results — chunk changed inside one recurrence-preserving launch, tuning keys that include compiler specializations, placement and cold-L2 controls, and both eager and graph-like timing conventions — is correct, reusable, and more careful than most kernel-tuning papers at this venue.
