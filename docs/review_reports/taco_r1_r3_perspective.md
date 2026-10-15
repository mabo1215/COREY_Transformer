# TACO Review, Round 1: Reviewer 3 (Perspective: Production Inference Systems)

**Manuscript:** "Shape Sets the Chunk: A Compile-Informed Cost Model and a No-Gain Bound for Data-Dependent Dispatch of Chunked Linear-Recurrence Kernels" (main article, 23 pp. including references, plus a 48-page Supplementary Material, reviewed together as one paper)

**Reviewer identity.** I work on production LLM inference: serving engines (vLLM, SGLang, TensorRT-LLM), attention and recurrence kernels (FlashInfer, FLA, mamba_ssm), CUDA-graph capture, continuous batching, chunked prefill and tensor parallelism, and I run kernel tuning caches in deployed fleets. I read TACO as a practitioner who wants results I can act on. I am not an SSM modelling specialist. Where I comment on modelling conventions or statistics, I am speaking from the deployment side.

---

## Summary (5 lines)

1. The paper asks whether the chunk size of chunked linear-recurrence kernels (Mamba-1 scan, Mamba-2 SSD, Gated DeltaNet) should be chosen from the activations, from the shape, or not at all. It answers with an elementary no-gain bound, value-invariance tests, held-out predictors and a routed end-to-end study.
2. Practical findings: the production kernels' defaults are adequate in "eager" timing. Under single-GPU CUDA-graph replay of the scan alone, the SSD and Gated DeltaNet defaults lose 16 to 22% per call. Only the authors' own Triton Mamba-1 scan, which is 1.23x slower than the reference kernel, is strongly chunk-sensitive.
3. For that kernel, an eight-parameter compile-informed cost model predicts the chunk without timing the target shapes (1.3 to 7.3% mean regret). It does not transfer to TFLA, to the reference scan at fixed threads, or to the production kernels under graphs.
4. The kernel-level measurement work is careful: shuffled rounds, confidence intervals, placement and L2 controls, and three GPUs. The deployment-facing claims (eager vs. graph, serving replay, tensor-parallel deployment, the recipe) are not demonstrated in any real serving engine, packed varlen call or multi-GPU job.
5. As it stands, an engineer working on an inference runtime would take away "keep the defaults in eager mode, use a shape-keyed table when device-bound, never dispatch on data". That conclusion is plausible but under-evidenced exactly where it matters, and it is buried under a very dense presentation.

## Recommendation and confidence

- **Recommendation:** Major Revision
- **Confidence:** 4 / 5. I am confident about the systems and deployment aspects and less certain about TACO-specific modelling-theory conventions.

## Scores (0-100)

| Criterion | Score | One-line rationale |
|---|---|---|
| Novelty | 45 | The bound is elementary (the authors say so). The cost model is narrow and works on one non-production kernel. The graph-mode default regret is the most novel practical finding but is under-developed. |
| Methodology | 62 | Kernel timing methodology is unusually careful. However, the framing of "eager vs. graph" by isolated per-call timing is not how execution regimes behave in engines, and the replay simulator is under-documented. |
| Experiments | 55 | Broad at kernel level. Missing: a real serving engine, packed or varlen calls, a stock data-center GPU for the graph-mode claims, any multi-GPU run, and model sizes beyond 2.8B. |
| Clarity | 38 | Extremely number-dense. Six timing conventions are interleaved, terms are used in the abstract before they are defined, and headline results live only in the supplement. |
| Significance | 42 | For practitioners the actionable content is real but small, and the most actionable part (graph and device-bound regimes for production kernels) is the least evidenced. |

---

## Strengths (brief)

1. **Measurement hygiene.** The paper does several things that kernel papers usually skip: it identifies the split-call pitfall (Sec. 6.1), times in three shuffled rounds, separates value effects from buffer-placement and L2 effects (Supp. Tables S8, S17, S18, S38), and states which numbers do not reproduce across sessions (Sec. 7.8; Supp. Sec. S9).
2. **Honest negative results.** The cost model's failures (TFLA 7.5% vs. default 1.4%; the reference scan at 32 threads, 49.5%; production kernels under graphs, 12 to 22%) are reported in the abstract and conclusion. This is rare and valuable.
3. **Deployment units and break-even.** Table 8 and Eqs. (4) and (5) take the right step from per-call relative regret to ms per forward pass and calls per key, which is how practitioners decide whether something is worth doing.
4. **A useful corollary, under-sold.** Value invariance (Sec. 7.1) implies that a tuning cache can be filled at engine warm-up with synthetic inputs. Practitioners would benefit from this statement directly.

---

## Major concerns

### P1. "Eager vs. CUDA graph" is the wrong axis. The relevant axis is host-bound vs. device-bound execution, and isolated back-to-back timing measures neither eager nor graph cost inside a model.

- **Where:**
  - Abstract, main.tex lines 49-56 ("In eager execution the defaults suffice ... Replayed as CUDA graphs ... lose 16 to 22%").
  - Contribution 1, lines 141-151.
  - Sec. 6.4, lines 654-660 ("these *launch-inclusive* latencies ... are the cost in eager execution"; "device time ... approximates the cost inside a CUDA graph").
  - Sec. 7 opening, lines 697-708.
  - Deployment-recipe preface, lines 1380-1383.
  - Limitations, lines 1414-1419.
  - Conclusion, lines 1444-1447.
  - Supp. Sec. S9 and Table S39.
- **Why it matters:**
  - In an engine forward pass, kernel launches are asynchronous. Whether a scan's chunk matters depends on whether the host runs ahead of the device for that layer. This is set by model size, tokens per step and per-layer host overhead, not by graph capture as such.
  - The paper's own routed study shows exactly this (Sec. 7.6; Supp. Table S45). At 256 and 976 tokens the 370M prefill takes about 46 ms "whatever the chunk, because the host-side dispatch of the 48 layers hides the differences". At 4096 tokens the device-time effect appears in full (+16.8 ms).
  - A 46 ms prefill for 256 tokens of a 370M model is a framework-eager stack, roughly an order of magnitude slower than a production engine. In production engines with low host overhead, and for 2 to 8B+ hybrid models, prefill layers are typically device-bound even without full graphs.
  - Conversely, as far as I know, current piecewise-graph modes in vLLM-style engines keep the SSM or linear-attention mixer outside the captured pieces for mixed prefill batches. Prefill SSD and Gated DeltaNet kernels therefore usually run eagerly, but after long GEMMs, with the host ahead.
  - Hence the device-time and graph-mode finding ("defaults lose 16 to 22%") may well apply to production eager serving of realistic models. The eager finding ("defaults suffice") may be an artifact of Python launch overhead in isolated repeats; note the about 0.46 to 0.79 ms SSD "launch floor" in Supp. Tables S11 and S26.
  - As written, the headline guidance can mislead an engine developer in either direction.
- **Required action:**
  1. Reframe the dichotomy as host-bound vs. device-bound and give an operational criterion, for example per-layer host enqueue time vs. device time, measured or estimated per (model, tokens per step).
  2. Measure the chunk effect of Mamba-2 SSD and Gated DeltaNet inside a real engine forward pass:
     - vLLM or SGLang, in eager, piecewise-graph and (for decode-only or short steps) full-graph modes;
     - at least one Mamba-2-style hybrid (2 to 8B) and one Gated-DeltaNet-style hybrid;
     - chunk in {64, 128, 256} for SSD and {16, 32, 64} for GDN, vs. a shape-keyed table;
     - report TTFT and prefill throughput at token budgets of 2K to 8K.
  3. If an engine integration is impossible, at minimum capture the whole Mamba-2 2.7B forward pass (not the single scan) as one CUDA graph per prompt length and chunk, and also run it eagerly with a low-overhead launcher. Report the end-to-end difference.
- **Effort:** (a) reframing text. (c) Engine measurement: about 15 to 25 GPU-hours on one H100- or H800-class GPU plus engineering. The whole-model-graph fallback: about 4 GPU-hours.

### P2. The graph-mode result for production kernels is a headline claim but rests on one consumer GPU, per-call relative regret, a supplement table, and an untraceable ms figure.

- **Where:**
  - Abstract, lines 53-54.
  - Contribution 1, line 145.
  - Sec. 7.5, "Replayed as CUDA graphs", lines 1160-1166.
  - Deployment units, lines 1277-1279 ("SSD default costs 0.56 (370M) and 1.44 ms (2.7B) per forward pass"): I could not find these numbers in Table 8 or in any supplementary table.
  - Limitations, lines 1417-1419.
  - Supp. Table S39.
- **Why it matters:**
  - This is the one result that tells an engine developer to change something in production kernels. It is measured only on a non-stock 48 GB RTX 3090 board (line 606; Supp. Sec. S1.1).
  - Mean per-call relative regret over captured calls weights the many short calls equally. That is not what TTFT or throughput weights.
  - The 16 to 22% figure has no denominator in deployment terms. 1.44 ms per forward pass may be 1% or 10% of a 2.7B prefill.
- **Required action:**
  1. Repeat the Supp. Table S39 replays (SSD 370M and 2.7B, GDN 1.4B) on the H800 and on the RTX PRO 6000. The captured inputs already exist.
  2. Report latency-weighted regret and ms per forward pass next to the per-call mean, together with the forward-pass time it is a share of.
  3. Add a main-body table for the graph-mode results, since they appear in the abstract and conclusion.
  4. Add the supporting table for the 0.56 and 1.44 ms figures.
- **Effort:** (b) for the reweighting and the table. (c) About 4 to 8 GPU-hours for two more GPUs (6490 captured calls x 5 or 3 chunks x 3 rounds).

### P3. Batching and call shapes do not match what continuous-batching engines issue. The recipe's key for packed calls is unvalidated.

- **Where:**
  - Sec. 6.3, lines 623-627 (batches of 4 and 16 stack equal-length prompts cropped to the shortest; prompts reused, so a batch-16 call holds 2 to 10 distinct prompts).
  - Lines 636-639 (packed variable-length calls "were not timed").
  - Recipe key, line 1388.
  - Supp. Sec. S12 ("A packed call is costed as a batch-1 scan of its packed length, since we did not measure a variable-length kernel").
- **Why it matters:**
  - Engines with continuous batching and chunked prefill issue packed varlen prefill calls (batch dimension 1 with sequence offsets or `seq_idx`), mixed with decode, and with nonzero initial states for continued chunked-prefill slices and prefix-cache hits.
  - The padded equal-length batch-4 and batch-16 regime that drives the largest Mamba-1 regrets is close to irrelevant for these engines. For SSD, chunks that straddle sequence boundaries change the work, so packed length alone may not be a sufficient key.
  - The recipe tells practitioners to key on "packed length ... batch" without any measurement behind it.
- **Required action:**
  1. Time the varlen entry points of SSD and of FLA's Gated DeltaNet (sequence offsets, with and without initial states) on packed steps sampled from the replay, for example 200 steps per trace mode and token budget.
  2. Report whether the optimum depends only on packed length and alignment, or also on the number of sequences and the length mix.
  3. Revise the recipe's key accordingly and replace "costed as batch-1" in the replay with measured varlen costs.
- **Effort:** (c) About 3 to 6 GPU-hours; (a) recipe text.

### P4. The cost model's practical niche is not demonstrated. It works on a kernel no one should deploy, and its economics are inverted relative to warm-up profiling.

- **Where:**
  - Contribution 2, lines 152-167.
  - Sec. 5.
  - Sec. 7.4, lines 925-1001.
  - Recipe, lines 1396-1409.
  - Supp. Table S13: Triton vs. reference ratio 0.81 to 1.54, geometric mean 1.23.
  - Supp. Sec. S5, "Per-key tuning costs": profiling all 33 keys costs 2.5 and 5.0 s.
  - Supp. Table S41: 11 to 132 bucket keys per trace and mode.
  - Sec. 7.4, line 930 and recipe line 1397: calibration takes 8 to 14 min per GPU.
- **Why it matters:**
  - Even at its oracle chunk, the Triton Mamba-1 scan is slower than the reference CUDA scan at 9 of 12 keys. A practitioner would deploy the reference scan, whose stock rule is within 6.6% launch-inclusive and 3.1% in device time (Supp. Table S27), and whose end-to-end penalty is +0.7% at 4096 tokens (Supp. Table S45).
  - On every production kernel and on TFLA, the model either fails or loses to the default.
  - Economics: the paper's own numbers show that a complete bucket-key table for any replayed trace can be profiled at engine warm-up in seconds (plus a one-time compile, which persistent compile caches amortize). That is cheaper than the 8 to 14 min per-GPU calibration.
  - Common practice is a persisted per-device table shipped with the engine, in the same style as per-device-name tuned config files for MoE GEMMs. The recipe's "When the model is worth it" (keys that cannot be enumerated: new shard widths, chunked-prefill steps, untimed GPUs) lists exactly the cases that practitioners do enumerate at configuration or warm-up time.
- **Required action:**
  1. Quantify, per replayed trace and mode under the recommended key, (i) the warm-up cost of profiling every key, including compile time with and without a warm compile cache, and (ii) the cold-start regret and TTFT impact, against the model's calibration cost and residual regret.
  2. Either (a) demonstrate the model on a kernel that is competitive in deployment (for example a production Triton SSD or varlen kernel, or a Mamba-1 Triton kernel that beats the reference), or (b) reframe the cost model as a methodology case study and remove it from the title and abstract headline.
- **Effort:** (a) and (b) for the economics and reframing. (c) About 5 GPU-hours if a new kernel is calibrated.

### P5. Parallel-deployment claims go beyond the single-GPU, synthetic-input evidence.

- **Where:**
  - Abstract, lines 59-61 ("At tensor-parallel shard widths ... it needs no retuning (2.9 to 6.5%), where the unsharded table loses up to 65%").
  - Contribution 2, lines 159-162.
  - Sec. 1, lines 115-116.
  - Sec. 5, lines 560-565 ("sequence parallelism at length L/p ... which the model covers").
  - Sec. 7.5, lines 1124-1141.
  - Discussion, lines 1373-1377.
  - Conclusion, lines 1448-1452.
  - Supp. Table S31.
- **Why it matters:**
  1. **The baseline is a straw man.** The unsharded-width table is not something anyone deploys. The tensor-parallel degree is fixed at configuration time, as the authors state themselves (line 1132), and retuning costs 0.03 to 0.05 s per key. The honest comparison is the model against a per-shard table (0% by construction), plus the cost of building that table.
  2. **The model is not uniformly better.** On the H800 the calibration-width table (2.2%) beats the model (2.9%). On the RTX 3090 the model loses up to 15.2% per shard (Supp. Table S31, 370M t=2 and 1.4B t=4).
  3. **Co-running communication is ignored.** Under tensor parallelism the scan shares the device with collectives. Communication kernels occupy SMs, and compute-communication overlap is common. This changes the n_SM, wave and occupancy terms that Eq. (3) hard-codes. None of this is exercised on one GPU.
  4. **Sequence parallelism is not measured.** Sequence or context parallelism for recurrences needs the initial-state input path and a state exchange, which are different kernel variants. "L/p" shapes are just shorter calls that are already in the captured set, so the "covers" claim is unsupported.
  5. **Ranks can disagree.** If ranks profile independently on noisy timings, they can choose different chunks, producing per-rank latency skew and stragglers at the next collective. The recipe is silent on rank consensus.
- **Required action:**
  - (a) Rewrite the abstract, contributions, Discussion and Conclusion to say "per-device shard shapes timed on one GPU with synthetic inputs". Compare against the per-shard table and its cost. Drop sequence parallelism from the claims, or measure the initial-state variant. Add rank consensus to the recipe.
  - (c) A 2- to 4-GPU tensor-parallel run (for example 2x H800, or 4x RTX 3090 under a serving engine with TP=2 and TP=4) measuring per-rank scan latency with collectives active: about 4 to 8 wall-clock hours on 2 to 4 GPUs (8 to 32 GPU-hours).

### P6. The deployment recipe is incomplete in ways that matter for a runtime.

- **Where:** Discussion, recipe box, lines 1379-1409.
- **What is missing:**
  1. **Where and when to profile.** "On first sight of a key, profile the best two" (line 1398) would add 30 to 50 ms to a live request's TTFT if done on the critical path. It cannot happen inside a graph replay at all. Profile at warm-up with synthetic inputs, which the value-invariance result justifies and the paper should say so explicitly. For graph-captured sizes, choose the chunk per captured size at capture time. Persist the table.
  2. **Cache invalidation and key completeness.** Add the compiler or Triton version, kernel-library version, driver, GPU SKU (PCIe vs. SXM, power cap, MIG slice, all of which change n_SM or clocks), and whether initial states or intermediate states are passed or returned.
  3. **Tensor-parallel rank consensus** (see P5).
  4. **Engine constraints that pin the chunk.** In Mamba-2 the chunk is a model-configuration value. Engines precompute per-step chunk metadata from it, and, as far as I am aware, SSM-state prefix caching in some engines checkpoints states at chunk-aligned block boundaries. Per-shape chunk switching must stay compatible with these mechanisms.
  5. **Numerical batch invariance.** If the chunk depends on packed length or batch, a request's floating-point result depends on which other requests share its step. Several engines now offer batch-invariant or deterministic modes because RL rollouts and reproducibility require it. The paper's evidence on this is thin: outputs agree to 1e-3 (line 591) and greedy tokens match on 3 prompts x 32 tokens (Sec. 7.6), which is too small a test.
- **Required action:**
  - (a) Rewrite the recipe as a runtime procedure covering items 1 to 5, and state explicitly when not to switch chunks (for example in batch-invariant mode).
  - (c) Measure output divergence of per-shape chunk switching vs. a fixed chunk on 200 prompts x 256 generated tokens for SSD and GDN: first divergent token and logit KL. About 1 to 2 GPU-hours.

### P7. The serving-trace replay is under-documented and reports inconsistent denominators.

- **Where:**
  - Sec. 7.7, "Deployment units", lines 1280-1284 ("2.1 to 2.6% of prefill on the RTX PRO 6000 and 5.5 to 10% on the RTX 3090 ... over all 88 primary settings ... the replay's step model is 2.3 times faster than the measured eager prefill").
  - "Serving traces", lines 1314-1334.
  - Supp. Sec. S12 and Table S41.
- **Why it matters:**
  - The "step model" (how non-scan layers and decode steps are costed), the "88 primary settings" and the share-of-prefill metric are not defined anywhere in the main paper or the supplement. Line 1283 is the only mention. A practitioner cannot reproduce or judge the replay.
  - The main text reports a share of prefill (2.1 to 10%). Supp. Table S41 reports a share of total GPU time including decode (0.31 to 2.30% for the default). These are different quantities presented as the same finding.
  - Up to 76.5% of the calls in a setting are mapped to measured keys with fewer than 30 captures ("Thin"). The replay uses the 370M model (48 calls per step) and other models' tokenizers.
- **Required action:**
  1. Fully specify the step model, the decode model and the settings in the supplement.
  2. Report TTFT p50 and p99, prefill throughput and GPU-time share with one consistent denominator, and give the decode share explicitly, so readers see that the chunk affects only prefill.
  3. Replay with the 2.7B and hybrid shapes as well as the 370M ones.
  4. Validate the simulator once against a real engine serving a trace slice (for example 10 minutes of the Azure code trace).
- **Effort:** (b) for documentation and denominators. (c) About 4 to 8 GPU-hours for validation and larger shapes.

### P8. Readability: the paper is very hard to use for a reader outside the SSM niche.

- **Where:**
  - Abstract, lines 47-68: about 35 numbers. Terms such as "launch floor", "device time", "launch-inclusive", "three criteria", "tuned shapes" and "at fixed threads" are used before definition, and "(one GPU)" is ambiguous.
  - Contribution bullets, lines 141-187: 10 to 15 numbers each, with nested parentheticals.
  - Sec. 7 opening, lines 697-708: repeats the abstract's numbers.
  - "Mamba-1 denotes [the Triton scan] below unless stated otherwise" (lines 702-703), while the reference Mamba-1 scan appears in the same tables.
  - Timing conventions vary across the tables: Table 4 uses device time for Mamba-1 and launch-inclusive for Mamba-2; Table 5 the same; Table 6 launch-inclusive single session; Table 7 mixed; Table 8 mixed per row. The same selector therefore appears as 0.4% (Table 5) and 0.88% (Table 6) on the same GPU.
  - The graph-mode headline has no main-body table.
- **Why it matters:** TACO readers include runtime and compiler engineers who will not reconstruct six timing conventions to extract the advice.
- **Required action:**
  1. An abstract with at most about 6 numbers and no undefined terms.
  2. A main-body "practitioner summary" table: kernel x regime (host-bound eager, device-bound eager, graph) x call-size class, with recommended policy, expected ms per forward pass, and share of TTFT.
  3. One "regime map" figure: default regret vs. tokens per call, per kernel, eager and graph.
  4. One primary timing convention for all main tables, with the others in the supplement.
  5. Distinct names for the two Mamba-1 kernels (for example "T-scan" and "R-scan").
  6. Move the six-convention discussion into one boxed definition.
- **Effort:** (a) and (b).

### P9. Space allocation: most of the body refutes a policy no runtime uses, while the deployment-relevant evidence sits in the supplement.

- **Where:**
  - Line 121-122 ("no published runtime or paper proposes value-dependent dispatch for dense chunked kernels").
  - Secs. 3, 4, 7.1-7.3, 7.6 and 7.7 (bound, entropy scheduler, value invariance, predictors, routed study, break-even, key churn): roughly 40% of the body.
- **Why it matters:**
  - Practitioners already exclude host-synchronizing, data-dependent dispatch in async-scheduled or graph-captured engines. In such engines one host synchronization per layer costs far more than the isolated 20 µs (line 1218), because it collapses the CPU-GPU overlap.
  - The no-gain result has two practical corollaries: warm-up tuning with synthetic inputs is sufficient, and device-side conditional dispatch is bounded by the information side. These fit in about a page.
- **Required action:** Condense Secs. 4 and 7.1-7.3 and 7.6 (keeping the bound and one decisive table), move the information-theoretic detail to the supplement, and use the freed space for the evidence requested in P1 to P3 and P7.
- **Effort:** (a).

---

## Minor concerns

1. **Lines 1189-1191 (Sec. 7.6) misstate the routed study.** The text says "Every variant that computes an activation statistic is slower ... by 24 to 71% in prefill and 0.8 to 5% in total time". Supp. Table S45 shows the full-histogram variant at +56 / +125 / +193% prefill and +16.2% total at 4096 tokens. The supplement restricts the 24 to 71% range to the sampled, per-token and moment variants. Fix the main text.
2. **Sec. 7.6 omits the end-to-end bottom line.** Even on the most chunk-sensitive kernel at 4096 tokens, static chunk 128 costs only +3.1 ± 2.4% of total time with 32 decode tokens (Supp. Table S45), and less with realistic output lengths. State this in the main text.
3. **Lines 1422-1423 (Limitations).** "The reference CUDA scan was timed at one fixed thread count (32)" reads as if all reference timings used 32 threads. Say "besides the stock thread pairing, only one fixed thread count (32) was tested".
4. **Abstract, line 52.** The reference scan's stock rule is "within 6.6%" launch-inclusive but 3.1% in device time (Supp. Table S27). State the convention.
5. **Table 8 (lines 1242-1270).** Add the oracle forward-pass time as a denominator column (Supp. Table S28 has it for the Triton rows), add graph-mode rows, and explain "pooled".
6. **Fig. 2 (lines 781-789).**
   - The SSD panel shows a flat about 0.48 ms eager floor, which is Python launch overhead and tells a practitioner nothing about the kernel. Add device-time or graph curves.
   - The panel titles sit on the x tick labels of the panel above.
   - Batch 1 only; add one batched or packed panel.
7. **Line 596-597.** "The model ... puts the RTX 3090 optimum near 85" for Gated DeltaNet is beyond the library cap of 64. Either test chunk 128 by lifting the cap (shared-memory limits permitting, especially on Hopper), or drop the speculation.
8. **Lines 605-607.** The modified 48 GB RTX 3090 board should be characterised (memory clock and bandwidth, power limit). Practitioners will question its representativeness, which is another reason to move graph-mode headline claims to a stock data-center GPU (P2).
9. **Lines 666-668.** Clocks were not locked. Data-center fleets often run power-capped. Please report the power limit and, ideally, check whether the optimum shifts under a lower cap (one GPU-hour).
10. **Line 1218.** The 20 µs "one host synchronization" cost is an isolated lower bound. In engines with asynchronous scheduling, a per-layer synchronization also exposes all subsequent launch latency. Note that the real cost is larger, which strengthens your conclusion.
11. **Recipe line 1396.** "About 2300 calibration points" are mostly the 184-configuration launch-space sweep. For chunk-only selection, show regret as a function of calibration size: the supplement has 18 batch-1 medians giving 4.6 / 5.1%. Practitioners want the minimum calibration.
12. **Sec. 7.4, lines 1024-1026, and Supp. Sec. S5.** Compilation dominates the tuning cost (363 of 493 s). With a persistent compile cache it is paid once per (architecture, compiler version). Redo the break-even analysis with and without a warm compile cache.
13. **Ties by CI overlap.** The supplement shows this criterion overstates ties by up to 56 points on noisy tables (Supp. Table S3). Use "within 2% of the oracle" in the main tables, as practitioners would.
14. **Lines 1355-1357.** Launch-inclusive values hold "for one session" only. Since the abstract mixes launch-inclusive (1.3 to 7.3%) and device-time numbers, state all headline numbers in one reproducible convention.
15. **Production layouts.** The Gated DeltaNet captured calls come from one research checkpoint (8 heads of 256). Production-style layouts (for example 16 qk / 32 v heads of 128) are covered only synthetically. Add a table that maps the per-device shapes of current production hybrids (at their typical TP degree) onto the measured width and layout ranges, so readers can see coverage. This is (b) only.
16. **Related work is missing deployed tuning-cache practice:**
    - SGLang and TensorRT-LLM;
    - TensorRT's timing cache and optimization profiles;
    - cuBLASLt's heuristic-then-benchmark algorithm selection, which is exactly the "model top-k, then profile" hybrid of Sec. 7.4;
    - PyTorch Inductor max-autotune caching;
    - per-device tuned kernel config files in serving engines;
    - SSM-state prefix caching in hybrid-model serving.
17. **Double-anonymous hygiene.** If LaTeX sources are uploaded:
    - appendix.tex lines 1-10 read "Supplementary Material for the IEEE TPDS submission ... loads IEEEtran";
    - figure files are named `tpds_*.pdf`;
    - the main.tex header comments refer to earlier formats.
    Clean these up. Also verify the page limit: the body (with the AI-use disclosure) runs onto page 21 before the references.
18. **Line 54, "(one GPU)".** This means one GPU model (RTX 3090), not one device. Say so.
19. **Sec. 6.3.** Prompts are truncated "in rotation" to 64 to 1024 tokens to cover lengths, so the length distribution of the captured calls is synthetic. That is fine for kernel coverage, but say explicitly that the per-call means are not workload-weighted.
20. **Abstract and Sec. 7.7.** State once, plainly, that decode does not use the chunked kernel, and give the decode share of GPU time in the replay, so readers can bound the end-to-end relevance.

---

## Assumption audit (perspective-specific, brief)

- **Explicit assumptions.**
  - The kernels are data-oblivious (Prop. 1b). This holds for the studied kernels, and the paper tests rather than assumes it. Good.
  - Device time approximates graph cost. Mostly validated (Supp. Table S39, Mamba-1).
- **Implicit assumptions.**
  1. Isolated per-call timing represents the eager cost in a model. It does not when the host runs ahead (P1).
  2. Equal-length padded batches represent serving calls. They do not for packed continuous batching (P3).
  3. A single-GPU shard shape represents a tensor-parallel deployment. It ignores co-running collectives and rank consensus (P5).
  4. Changing the chunk per shape is free of side effects. It interacts with batch invariance, chunk metadata and state caching (P6).
  5. Keys must be tuned online because they cannot be enumerated. Bucketed keys are enumerable in every replayed setting (P4).
- **Paradigmatic assumption.** Per-call relative regret is the figure of merit. Deployments optimise TTFT, throughput and GPU-time share, which weight long calls and include non-scan work. The paper has the tools (Table 8, Supp. Table S28) but leads with the per-call metric.

## Cross-disciplinary connections and reading recommendations

- **Frigo and Johnson, "The Design and Implementation of FFTW3", Proc. IEEE 93(2), 2005.**
  - FFTW_ESTIMATE (cost model) vs. FFTW_MEASURE (profiling), plus persisted "wisdom", is exactly the paper's model-vs-profile trade-off with a shape-keyed persisted cache.
  - FFTW's long experience with when ESTIMATE suffices would sharpen the "when the model is worth it" discussion.
- **Vuduc, Demmel and Yelick, "OSKI: A library of automatically tuned sparse matrix kernels", J. Phys.: Conf. Ser. 16, 2005.** Install-time machine characterisation plus a runtime model. This is the closest precedent for "calibrate once per GPU, predict at runtime", and it also treats when runtime tuning amortises.
- **Ioannidis, Ng, Shim and Sellis, "Parametric Query Optimization", VLDB Journal 6(2), 1997; Dutt and Haritsa, "Plan Bouquets", SIGMOD 2014.** Plan caching keyed on parameter regions, with the cost-model vs. execution-feedback tension. This literature formalises key granularity and churn (Eq. (5)) and offers robustness criteria beyond mean regret.
- **He and Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", Connectionism blog, Sep. 2025.** Shape-dependent reduction or decomposition choices break batch invariance. This is directly relevant to shape-keyed chunk switching (P6).
- **Whaley and Dongarra, "Automatically Tuned Linear Algebra Software", SC 1998.** Install-time empirical tuning per platform is the baseline the cost model must beat economically (P4).

## Questions for the authors

1. In a production engine forward pass of a 2 to 8B hybrid at a 2K to 8K token budget, is the SSD or GDN layer host-bound or device-bound? Which of your two conclusions ("defaults suffice" or "defaults lose 16 to 22%") then applies?
2. For packed varlen SSD calls, does the optimal chunk depend on the number of sequences and their lengths, or only on packed length and alignment?
3. Under the recommended bucket key, how long does it take to profile every key of each replayed trace at engine warm-up, with and without a warm compile cache? In which realistic scenario is the 8 to 14 min calibration cheaper?
4. Is shape-keyed chunk switching compatible with SSM-state prefix caching and with batch-invariant inference modes? If not, what should a runtime do?
5. Under tensor parallelism with overlapped collectives, does the per-rank optimum (and the model's ranking) survive reduced SM availability?

---

## What would be needed for Accept

1. **Real-engine evidence (P1, P2).** Chunk-effect measurements of the production kernels (SSD, Gated DeltaNet) in a real serving engine, or at least in whole-model CUDA-graph and low-overhead eager forward passes, on a stock data-center GPU (H100 or H800 class). Report TTFT, throughput and GPU-time share with consistent denominators. Reframe the conclusions around host-bound vs. device-bound execution.
2. **Packed calls (P3).** Timing of packed varlen calls, with and without initial states, drawn from the replay, plus a corrected recipe key.
3. **Honest positioning of the cost model (P4).** Either demonstrate value on a competitive kernel and against warm-up profiling economics, or demote it from the headline.
4. **Rewritten parallel-deployment claims (P5).** State them as per-device shard shapes timed on one GPU, compare against the per-shard table, and either drop sequence parallelism or measure it. A small multi-GPU tensor-parallel run would substantially strengthen the claim.
5. **A runtime-grade recipe (P6).** Warm-up tuning with synthetic inputs, cache invalidation keys, rank consensus, compatibility with chunk-pinning engine features, and a batch-invariance caveat backed by a divergence measurement.
6. **A documented, validated serving replay (P7).**
7. **A substantially more readable paper (P8, P9).** A short abstract, a practitioner summary table and regime-map figure, one primary timing convention, a condensed data-dependent-dispatch part, and the minor fixes above, in particular minor items 1 and 3.

**Estimated new GPU time for items 1 to 6:** roughly 40 to 80 GPU-hours on H100- or H800-class hardware plus a short multi-GPU allocation. Most of the remaining work is analysis of existing data and rewriting.
