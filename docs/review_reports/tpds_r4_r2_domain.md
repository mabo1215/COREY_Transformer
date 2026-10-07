# Domain Review Report (Reviewer 2) -- IEEE TPDS regular paper

**Manuscript:** "Shape Sets the Chunk: A Compile-Informed Cost Model and a No-Gain Bound for Data-Dependent Dispatch of Chunked Linear-Recurrence Kernels" (main body `paper/main.tex`, 14 pp.; Supplementary Material `paper/appendix.tex`, 16 pp.; bibliography `paper/ref.bib`, 64 entries).

**Reviewer identity:** researcher working on GPU kernels for state space models and linear attention (Mamba selective scan, Mamba-2 SSD, Gated DeltaNet / flash-linear-attention, chunked recurrences) and on auto-tuning and input-aware kernel selection (Triton autotuner, TVM/Ansor, Nitro, roofline and analytical GPU cost models).

**Independence:** this review was written from the manuscript sources and the reviewer rule files only. No revision history, progress notes, earlier reports or source code were consulted. Cited related work was verified by web lookup where marked "verified"; two items could not be verified in this session and are marked as such.

**Confidence:** 4/5 (high on kernel structure, autotuning and measurement practice; medium on the serving-trace replay, which is outside my core area).

---

## 1. Scores (0-100)

| Dimension | Weight | Score | Justification |
|---|---|---|---|
| Novelty | 20 | **58** | The no-gain bound is, by the authors' own admission, elementary, and the "model-then-profile-top-k" workflow and compile-fact-based pruning already exist in the Triton autotuner (`prune_configs_by={'perf_model','top_k','early_config_prune'}`). The fitted per-kernel model and the key-churn framing are useful but incremental against analytical GPU models that already consume compiled code (PPoPP 2010/2012) and shape-generic cost models for dynamic shapes (DietCode, MLSys 2022), none of which is cited. |
| Methodology | 25 | **63** | The recurrence-preserving measurement discipline, the placement/L2 control, the TOST tests and the headroom-vs-noise analysis are exemplary. But the central protocol (warm L2, repeats back to back) is the opposite of what the Triton autotuner does (L2 flush per repeat), the paper's own control shows 29-43% placement effects under that protocol at the lengths that matter, and the Mamba-2 "launch floor" is an eager-mode host artefact that the paper could not measure in device time. |
| Experiments | 25 | **66** | Three GPUs, 2360 captured calls per checkpoint, 370M-2.8B widths, TP shard widths, serving-trace replay: broad. But the headline numbers (8.5x spread, 0.9% regret) come from a kernel written for this study that is 23% slower (geomean) than the reference scan, the two production kernels are nearly chunk-insensitive in the tested ranges, the reference-scan sweep is confounded with thread count, and there is no learned-cost-model baseline on the same calibration data. |
| Clarity | 15 | **70** | Dense but precise prose; honest limitations. The word "chunk" is used for three mechanically different parameters (sequential time tile of one program; block size of a chunkwise-parallel SSD/delta-rule form) without a taxonomy, and the abstract's "a table tuned for one GPU loses up to 63%" conflates cross-device transfer with unsharded-to-sharded transfer. |
| Significance | 15 | **62** | A well-documented negative result on activation-driven dispatch and a practical cold-start recipe for tuning caches are useful to TPDS readers running hybrid/linear-attention models in fleets. Impact is capped because the strongest effect is specific to the authors' kernel and the TP argument is a strawman (TP degree is fixed at deployment). |
| **Weighted overall** | | **63.7 -> 64** | (0.20x58 + 0.25x63 + 0.25x66 + 0.15x70 + 0.15x62) |

---

## 2. Recommendation: **Major Revision**

Decisive reasons:

1. The generalisation in the title, abstract and contribution list ("chunked linear-recurrence kernels expose a chunk size that ...", "selects the chunk ... with 0.9% mean regret") is carried almost entirely by a Triton kernel the authors wrote, whose chunk dependence arises from register spills of its own tiling (255 registers and spills at C >= 512, Supp. Table S-compile) and which is slower than the reference CUDA scan. The two production kernels (Mamba-2 SSD, Gated DeltaNet) show <= 2.4% static regret or a fixed optimum at the library's largest chunk. The paper needs at least one production kernel with a genuine, wide chunk parameter (TFLA's mLSTM kernels are the obvious candidate and are already cited as the closest related work) before the general claim can stand.
2. The timing protocol under which every per-call oracle, every regret and the cost-model fit were obtained (warm L2, back-to-back repeats) is one the paper itself shows to be placement-sensitive by up to 43% at L = 8192, and it is not the protocol of the Triton autotuner the paper says it emulates (which flushes L2 before every repeat). The Mamba-2 "launch floor" could not be measured in device time at all; CUDA-graph replay, the serving-relevant protocol, is the right fix and is cheap.
3. Positioning against prior art is incomplete in exactly the areas the paper claims novelty: compile-informed analytical GPU models (Baghsorkhi et al., Sim et al.), shape-generic cost models for dynamic shapes (DietCode, Nimble), transfer across devices (TLP, Transfer-Tuning), and Triton's own perf-model pruning. The related-work rewrite is text-only, but the novelty claims must be re-scoped accordingly.

All three are fixable with bounded effort (my estimates below total roughly 10-14 GPU-hours on one RTX 3090 / RTX PRO 6000 plus offline analysis), and the measurement discipline already in the paper gives me confidence the authors can do it well.

---

## 3. Findings

Effort classes: (a) text only; (b) offline analysis of existing results; (c) new GPU measurement, with estimated GPU-hours.

### F1 [CRITICAL] The headline chunk effect is a property of a bespoke kernel, not of "chunked linear-recurrence kernels"

**Evidence.** Sec. 6.2 "Kernels": "(i) Mamba-1 selective scan: a Triton kernel written for this study in which the chunk is the time tile ... this kernel at its best chunk takes 0.81 to 1.54 times the latency of the reference scan ... (geometric mean 1.23)". Table 1: chunk spread 8.52x (Mamba-1, L=8192) versus 1.25x (Mamba-2 SSD) and 2.00x (Gated DeltaNet). Supp. Table S-compile (default launch table): 255 registers with spill stores at C = 512 and 1024 on both architectures, 64-202 registers below. Sec. 9 Limitations: "Only the Mamba-1 scan has an optimal chunk that moves with the shape by enough to matter ... The per-call, predictor and cost-model results therefore rest mainly on Mamba-1". Abstract and Contribution 1 do not say this.

**Why it matters.** In the authors' kernel each program walks all L/C tiles sequentially and holds a [D_b, C] tile in registers; the 8.5x spread is the signature of per-tile overhead at small C and spills at large C, i.e., of this tiling choice. A kernel that stages the tile through shared memory, or the reference scan at fixed thread count, need not show it. The abstract's "0.9% mean regret", "8.5x" and "Shape sets the chunk" are therefore statements about one kernel of the authors' own design that is 23% slower than the production kernel for the same model. Readers from the FLA/SSD community will read "chunk" as the chunkwise-parallel block size and will expect the claim to hold there, where the paper's own data show almost nothing to select.

**Action.** (a) State in the abstract, the contribution list and the first paragraph of Sec. 7 that the large chunk effect and the 0.9/3.2/3.5% regrets are for the authors' Triton scan, and give the reference-scan ratio (1.23x) there, not only in Sec. 6.2. (c) Add one production kernel with a genuine chunk parameter and a wide supported range: the TFLA mLSTM kernels (Beck et al., NeurIPS 2025, already cited; open source, chunk sizes well beyond 64 by design) at the 370M-2.8B head layouts, L in {512 ... 8192}, batch {1,4,16}, three rounds, plus a work-term or Eq. (5)-style fit. Estimated 4-6 GPU-hours on the RTX 3090 including compilation. If the chunk optimum moves with shape there, the general claim is supported; if it does not, the paper's thesis becomes "shape-keyed tuning is enough and often unnecessary", which is still publishable but must be stated.

### F2 [MAJOR] "Chunk" means three different things and the paper's single definition fits none of them exactly

**Evidence.** Sec. 1, para. 1: "Inside a chunk the recurrence is evaluated in a parallel form; across chunks a state is carried forward. The chunk size trades on-chip reuse against register and shared-memory pressure and against the length of the sequential inter-chunk pass". Sec. 5: in the authors' Mamba-1 kernel "each program walks ceil(L/C) time tiles of C steps" (no parallelism across chunks; the "inter-chunk pass" is the program's own loop). Sec. 2 and Supp. Sec. S-cost-model "Multi-kernel pipelines": for Mamba-2 SSD and Gated DeltaNet the chunks are processed in parallel across the grid, the state passing is a separate sequential kernel, and the intra-chunk work grows with C (W = BL(a + bC) + B ceil(L/C) g), i.e., a FLOP trade-off (quadratic-in-C block-diagonal part against the number of state-passing steps), not a register trade-off.

**Why it matters.** The three parameters have different cost structures, different natural ranges (16-64 for delta-rule kernels because of the C x C triangular solve in the WY path; 64-256 for SSD; 32-1024 for the authors' time tile) and different reasons to depend on shape. The paper's conclusions are per kernel and correct, but the framing invites the reader to transfer the Mamba-1 result to the other two.

**Action.** (a) Add a short taxonomy table in Sec. 2 or 6.2 with columns: kernel; what C controls; parallel across chunks (yes/no); where the sequential pass runs (same program / separate kernel); intra-chunk cost scaling in C; dominant trade-off (registers-spills / FLOPs-vs-state-passing / launch count). Re-phrase Sec. 1 para. 1 so that it does not claim one mechanism for all three.

### F3 [MAJOR] Warm-L2 protocol: not the Triton autotuner's protocol, and shown by the authors to move latency by up to 43% at the lengths where the chunk matters

**Evidence.** Sec. 6.4: "Repeats run back to back without flushing the L2 cache, so the latencies are those of a warm cache." Sec. 7.6: tuning is done "in the style of the Triton autotuner". Supp. Table S-l2 (RTX PRO 6000, L = 8192, 20 placements of one Gaussian draw): warm-L2 placement range 40.9 / 37.1 / 29.3 / 32.6 / 8.2 / 2.6% at chunks 32 ... 1024, cold-L2 range 0.5-2.3%. Sec. 7.1: "on the RTX PRO 6000 at L=8192 they vanish when the L2 is overwritten before every repeat". Supp. Sec. S-invariance-details: "In serving, where other kernels run between two scans of a layer, the warm-L2 advantage of a favorable placement is not guaranteed either."

**Why it matters.** `triton.testing.do_bench`, which the Triton autotuner uses to time each config, flushes L2 with a 256 MB buffer before every repeat (verified against the Triton documentation). An autotuner "in the style of Triton" would therefore rank chunks under a cold L2. The paper's per-call oracles, Table 1-4 regrets, the 8.5x spread and the Eq. (5) fit were all obtained warm. At L = 8192 the working set equals the 128 MB L2 of the PRO 6000, and the paper shows that under warm L2 the ranking of chunks 32-256 is placement-determined (Spearman 0.96 across rounds) while chunk 512 moves by 8.2%. Whether the oracle chunk and the model's regret survive a cold-L2 protocol is not reported, and in prefill the scan does not run back to back on the same buffers.

**Action.** (c) Re-time the captured Mamba-1 calls with L >= 2048 (and the 12 tuning keys) on the RTX PRO 6000 with an L2 flush before every repeat (the `do_bench` convention) and report, for that subset: per-call oracle agreement with the warm protocol, the Table 3 regrets of the shape tree and of the cost model (refitted on cold calibration data), and the chunk spread. About 1-2 GPU-hours (the flush dominates). (a) In Sec. 6.4 and 7.6 state explicitly that the Triton autotuner flushes L2 and that the paper's protocol does not, and say which one a serving runtime should use for its tuning cache.

### F4 [MAJOR] The Mamba-2 SSD "launch floor" is an eager-mode host artefact that the paper could not measure in device time; CUDA-graph replay is the correct and cheap protocol

**Evidence.** Sec. 7.1: "at batch 1 its multi-kernel pipeline is bound by the fixed cost of its several kernel launches, so its latency is nearly flat in the chunk and the length up to L=2048"; Supp. Table S-latency: SSD latency 0.476-0.493 ms for every chunk at L = 512 and 2048, batch 1. Supp. Sec. S-conventions: "For the Mamba-2 SSD pipeline, whose Python-level call enqueues several kernels, the spin does not cover the enqueueing of a full block of repeats, and the device-time timings mix both regimes (median within-block variation 85%); we do not use them ... its latency floor at batch 1 falls from about 740 to about 320 us". Supp. Table S-largewidth-calls: of the 2360 captured Mamba-2 2.7B calls only 397 are "above floor" and used.

**Why it matters.** A 0.46-0.74 ms floor for a pipeline of about six Triton kernels is the Python/Triton launch path, which serving runtimes remove by capturing CUDA graphs. All Mamba-2 rows of Tables 1-3 and the 2.7B headroom therefore describe eager execution, and the paper's Mamba-2 conclusions ("nothing to predict", "flat") are conclusions about host overhead at small token counts. The paper's spin trick fails precisely for multi-kernel pipelines; `torch.cuda.CUDAGraph` capture per (shape, chunk) and replay gives clean device time with no spin and is what the paper's own "device time ... approximates the cost inside a CUDA graph" (Sec. 6.4) describes.

**Action.** (c) Capture the SSD pipeline (and the Gated DeltaNet pipeline) in a CUDA graph per (shape, chunk), replay >= 50 times after warm-up, and re-derive the Mamba-2 rows of Tables 1 and 2, the Mamba-2 column of Table 3, and the 2.7B / 1.4B headroom of Supp. Table S-largewidth-calls on all 2360 / 1770 calls rather than the 17% / 9% above the eager floor. About 3 GPU-hours on the RTX 3090 (graph capture is milliseconds per cell). Report both conventions, as Supp. Table S-conventions already does for Mamba-1.

### F5 [MAJOR] The reference CUDA scan sweep confounds chunk with thread count and load width; the confound is removable

**Evidence.** Sec. 7.5: "by dispatching each chunk in {128,...,2048} to the thread configuration that the stock code uses for a sequence of that length, so the thread count changes with the chunk, from 32 to 128, and the sweep measures the chunk as the stock dispatch realizes it rather than the chunk alone ... The reference scan is at least as chunk-sensitive as ours". Supp. Sec. S-cost-model "Reference CUDA scan": "n_t is the stock thread count of each chunk (32 for C <= 512, 64 for 1024 and 128 for 2048)". Sec. 9 repeats the caveat.

**Why it matters.** The stock kernel template is `<kNThreads, kNItems>` with chunk = kNThreads x kNItems; the five stock instantiations change the number of threads per block by 4x (and kNItems = 4 uses 8-byte instead of 16-byte vector loads). A 5.4x median spread between a 32-thread and a 128-thread block is dominated by intra-block parallelism, not by the chunk. Calling this "at least as chunk-sensitive as ours" misleads, and it weakens the one production-kernel data point that supports F1.

**Action.** (c) Instantiate the template at a fixed thread count (kNThreads = 32, kNItems in {4, 8, 16, 32, 64} gives chunks 128-2048; or kNThreads = 128 with kNItems in {1 ... 16}), re-sweep the 2360 captured calls, and report the spread, the per-call oracle and the Eq. (5)-instance regret for the fixed-thread variant next to the stock-dispatch variant in Table 4. About 1-2 GPU-hours including the five compilations. (a) Until then, replace "at least as chunk-sensitive as ours" by a statement about the stock dispatch.

### F6 [MAJOR] Novelty of the compile-informed cost model is overstated relative to prior art that is not cited

**Evidence.** Sec. 2 "Kernel auto-tuning": "Search-based tuners time every candidate at every new shape key, and learned cost models such as Ansor's are trained on such timings per operator and device; the analytical model of Section 5 needs no timing at a new shape once its parameters are fitted". Sec. 2 "Performance models and measurement" cites only Hong & Kim 2009, Volkov 2016 and the Roofline paper and says "Our cost model is of this kind but reads its occupancy and spill inputs from the compiled kernel". Sec. 7.4: "Profiling only the two chunks that the model ranks best at a shape lowers the regret to 0.5% and 2.3%".

**Why it matters.** (i) Triton's autotuner has, for years, accepted `prune_configs_by={'perf_model': ..., 'top_k': ..., 'early_config_prune': ...}` (verified against the Triton documentation): an analytical performance model ranks the configs and only the top-k are benchmarked, and the shipped matmul `early_config_prune` reads register and shared-memory usage of the compiled kernel to discard configs that would spill. The "model, then profile top-2/3" workflow of Sec. 7.4 and the "compile-informed" ingredient are therefore both present in the tool the paper benchmarks against, and Sec. 7.6 should say so. (ii) Analytical GPU models that take the compiled instruction stream, occupancy and register usage as inputs predate this paper by 15 years (Baghsorkhi et al., PPoPP 2010; Sim et al., GPUPerf, PPoPP 2012; Zhang & Owens, HPCA 2011; Konstantinidis & Cotronis, JPDC 2017). (iii) Shape-generic cost models that pick a micro-kernel for shapes never tuned are the subject of DietCode (MLSys 2022) and of Nimble (MLSys 2021); this is the paper's "keys not yet tuned" use case. (iv) Cross-device transfer of tuned configurations and of learned cost models exists (TLP / MTL-TLP, ASPLOS 2023; Transfer-Tuning, PACT 2022).

**Action.** (a) Rewrite the two Sec. 2 paragraphs with the references in Sec. 4 of this report; in Sec. 5 scope the novelty to "a fitted, eight-parameter latency model for a chunked scan in which the chunk dependence enters only through assembler-reported registers and spills and the Triton tile layout, calibrated once per GPU"; in Sec. 7.4 and 7.6 compare the top-k profiling against Triton's `perf_model`/`top_k` mechanism and state what the paper adds (a model that works at new shape keys without timing them, and the transfer evidence of Sec. 7.5 and 7.9).

### F7 [MAJOR] No learned-cost-model baseline on the same calibration data

**Evidence.** Sec. 7.4: "The model adds information beyond its calibration data: a table that serves every call with the best chunk of the nearest of the twelve calibration keys ... reaches 5.2% and 4.5%"; "A three-parameter restriction ... selects chunks with 4.5% ... but with 2.5% on the RTX 3090, better than the full model there". Supp. Sec. S-h800: "On this GPU its margin over simple selectors is small: a tree on the length alone reaches 4.0% and the launch-and-work-only restriction 4.1%" against 3.5% for the full model.

**Why it matters.** The paper argues for an analytical form against search and learned models (Sec. 2), but the standard control in the autotuning literature, a regressor trained on exactly the same 2298 calibration timings with the same inputs (L, B, d, C, D_b, n_w, registers, spills, e, e_D, l_C, waves, resident warps), is missing. Without it the reader cannot tell whether the eight-parameter physics or merely the compile facts carry the result; the paper's own sentence "The compile facts carry most of the information" (Sec. 7.4) suggests the latter. On the H800 the margin over a length-only tree is 0.5 points.

**Action.** (b) Train, per GPU, a gradient-boosted regressor and a small MLP on log latency of the 2298 calibration timings with the features above (Ansor/TLP style), evaluate chunk-selection regret on the 2360 captured calls, on the held-out twelfth key, on the TP shard widths and across GPUs (features from the other GPU's compile facts), and add the rows to Table 3, Table 4 and Supp. Table S-tp. No GPU time needed. Report where the analytical form wins (extrapolation to new widths / devices is where I would expect it to) and where it does not.

### F8 [MAJOR] The tensor-parallel "loses up to 63%" claim compares against a strawman and is misworded in the abstract

**Evidence.** Abstract: "transfers across GPUs and to tensor-parallel shard widths, where a table tuned for one GPU loses up to 63%". Contribution 1: same wording. Supp. Table S-tp, column "Unsharded": 62.7% for the 1.4B model at t = 8 (width 4096 -> 512), i.e., a table tuned at the unsharded width applied to the shard width; the cross-GPU transfer loss (Sec. 7.9) is 14.1-48.8%. Sec. 8: "one tuned for a single GPU loses up to 63% after tensor-parallel sharding".

**Why it matters.** "Tuned for one GPU" reads as device transfer; the 63% is width transfer. More importantly, the TP degree is fixed when a deployment is configured, so every runtime tunes at the shard width once (seconds, per Sec. 7.6: profiling six chunks takes 1.2 s for twelve keys). Nobody would apply an unsharded table to shards. The realistic comparison is "cost model with no tuning (6.5% pooled regret, 95th percentile 30%)" versus "a few seconds of profiling at the shard width (about 0%)", which favours tuning. The cost model's genuine value for TP is cold start and heterogeneous fleets, which the paper should say.

**Action.** (a) Reword the abstract, Contribution 1 and Sec. 8: separate the cross-device loss (14-49%) from the unsharded-to-sharded loss (up to 63%), and present the TP result as "no retuning needed after sharding, at 0.7-15% regret", with the cost of retuning stated next to it.

### F9 [MINOR] Gated DeltaNet's "largest chunk is optimal" is a statement inside the library's design envelope

**Evidence.** Sec. 6.2: "(iii) Gated DeltaNet: the chunkwise gated delta-rule kernel of the FLA library, whose chunk size can be set to 16, 32 or 64". Sec. 7.4: the work-term model "places the optimum at 61 and 85, near or above the largest supported chunk (64)". Sec. 9: "Gated DeltaNet is measured at chunks up to 64, the limit of its library".

**Why it matters.** To my knowledge the public `chunk_gated_delta_rule` API of FLA fixes the chunk at 64 (the paper must have called the internal functions), and the WY-representation / triangular-solve path is written for chunks up to 64 because its intra-chunk cost is quadratic in C and 64 is the tensor-core sweet spot for head dimension 128-256. "The largest supported chunk is best" is thus the library authors' design decision being confirmed, not evidence that shape has nothing to say; the fitted optimum of 85 on the RTX 3090 says a chunk of 128 might win. (I could not verify the exact assertion in FLA 0.5.2 in this session; the authors should state the constraint from the version they used.)

**Action.** (a) Say that 64 is the library default and the design limit, and that the paper does not test beyond it. (c, optional, folds into F1) TFLA's kernels exist precisely to lift this limit; measuring them covers this point.

### F10 [MINOR] Origin of the chunkwise-parallel form is under-cited

**Evidence.** Sec. 2: "Gated linear attention introduced hardware-efficient chunkwise training for gated linear recurrences".

**Why it matters.** The chunked form (exact quadratic attention within a chunk, linear recurrence across chunks) was introduced by Hua et al. (ICML 2022, FLASH) and used for retention by Sun et al. (RetNet, 2023); GLA contributed the data-dependent gates and the secondary-level chunking that make it hardware-efficient with gates. As written, a reader may attribute the chunkwise form itself to GLA.

**Action.** (a) Add the two citations and reword to "the chunkwise parallel form of linear attention [Hua 2022; Sun 2023], made hardware-efficient for gated recurrences by GLA".

### F11 [MINOR] Key-churn analysis is not connected to the online-autotuning and serving literature it belongs to

**Evidence.** Sec. 7.8 "Key churn" and "Serving traces"; Sec. 2 "Serving and parallelism" cites Orca, vLLM, Sarathi-Serve, FlashInfer, Marconi, Splitwise and BurstGPT only.

**Why it matters.** Eq. (6)-(7) are the classic offline-versus-online autotuning amortisation (Active Harmony; OpenTuner), and the "coarser key" remedy the paper proposes is what vLLM/SGLang already do when they bucket batch sizes for CUDA-graph capture; both deserve a sentence. The Mooncake trace (FAST 2025, best paper) comes from the Kimi service, the one production deployment of a linear-attention model the paper cites (Kimi Linear), and would be the most relevant public trace for this study.

**Action.** (a) Add the citations and one sentence each; (b, optional) replay the Mooncake trace through the existing simulator (the released data cover "every trace", so the code exists).

### F12 [MINOR] Precision and consistency items

1. Sec. 7.7 text: "prefill with chunk 128 is 16.8 +/- 1.4 ms slower"; Table 5: "+16.8 +/- 1.5" in percent. Unit and interval disagree; both cannot be right unless the prefill is exactly 100 ms. (a)
2. Sec. 4: "The second assumption of (b) can fail through subnormal arithmetic". NVIDIA FP32/FP16 FMA pipelines handle denormals at full rate; this is not a GPU-relevant failure mode for these kernels. Replace by the value-dependent switching activity / power management already cited (Gregersen et al.) and by tensor-core data-dependent power, which the paper's "constant input is fastest" observation (Supp. Sec. S-invariance-details) actually reflects. (a)
3. Sec. 5 "Compiler specialization": correct and valuable. Add that Triton also specializes on arguments equal to 1 and on 16-byte pointer alignment, both of which also belong in the key. (a)
4. Mamba-2's own `@triton.autotune` decorators key the inner block sizes on chunk size, head dimension and state size, not on L or batch; stating this explains why the SSD inner tiles do not adapt to L and strengthens the "shape-keyed" narrative. (a; authors should confirm against the mamba_ssm versions they used.)
5. Supp. Table S-platforms lists mamba_ssm 2.2.2 (local build) on the PRO 6000 and 2.3.2.post1 on the RTX 3090 / H800; Sec. 7.9 says "same software versions (except mamba_ssm)". State whether the SSD kernel source differs between the two (the chunk_scan kernels changed between those releases, to my knowledge) because the Mamba-2 cross-GPU comparison in Table 6 otherwise mixes a kernel change with a device change. (a)
6. Sec. 3.1, Eq. (2) and Supp. Sec. S-estimator: the "decision saturation" analysis shows the entropy rule maps every real input to one chunk by construction. This is a flaw of the rule's normalisation, not evidence about activation statistics in general; the paper's stronger evidence is Table 3 (shape + entropy = shape). Consider shortening the COREY material in Sec. 3.1 and Fig. 4 in favour of F1-F5. (a)

---

## 4. Missing or misrepresented references

Verified = existence, authors and venue confirmed by web lookup in this session. Unverified = well-known to me but not looked up here.

1. **Bojian Zheng, Ziheng Jiang, Cody Hao Yu, Haichen Shen, Joshua Fromm, Yizhi Liu, Yida Wang, Luis Ceze, Tianqi Chen, Gennady Pekhimenko. "DietCode: Automatic Optimization for Dynamic Tensor Programs." MLSys 2022.** -- verified. Shape-generic search space and cost model that selects a micro-kernel for dynamic shapes without tuning each shape; the direct precedent for "the cost model covers keys not yet tuned" (F6).
2. **Haichen Shen, Jared Roesch, Zhi Chen, Wei Chen, Yong Wu, Mu Li, Vin Sharma, Zachary Tatlock, Yida Wang. "Nimble: Efficiently Compiling Dynamic Neural Networks for Model Inference." MLSys 2021.** -- verified. Establishes the dynamic-shape dispatch problem that the paper's key-churn analysis re-derives for one kernel (F6, F11).
3. **Sara S. Baghsorkhi, Matthieu Delahaye, Sanjay J. Patel, William D. Gropp, Wen-mei W. Hwu. "An Adaptive Performance Modeling Tool for GPU Architectures." PPoPP 2010.** -- verified. Analytical GPU model built from the compiled kernel's work-flow graph, intended "to provide performance information to an auto-tuning compiler and assist it in narrowing down the search": the lineage of Sec. 5 (F6).
4. **Jaewoong Sim, Aniruddha Dasgupta, Hyesoon Kim, Richard Vuduc. "A Performance Analysis Framework for Identifying Potential Benefits in GPGPU Applications." PPoPP 2012.** -- verified. Analytical model instantiated from static and dynamic profiles of the compiled code, with latency-hiding and issue terms comparable to Eq. (5) (F6).
5. **Yao Zhang, John D. Owens. "A Quantitative Performance Analysis Model for GPU Architectures." HPCA 2011.** -- unverified in this session. Microbenchmark-calibrated throughput model; relevant to the "eight parameters calibrated once per GPU" design (F6).
6. **Elias Konstantinidis, Yiannis Cotronis. "A quantitative roofline model for GPU kernel performance estimation using micro-benchmarks and hardware metric profiling." Journal of Parallel and Distributed Computing 107:37-56, 2017.** -- verified. Calibrated roofline-type model with reported absolute error (about 28%) that gives the paper a yardstick for its own 15-24% median error (F6, F7).
7. **Yi Zhai, Yu Zhang, Shuo Liu, Xiaomeng Chu, Jie Peng, Jianmin Ji, Yanyong Zhang. "TLP: A Deep Learning-based Cost Model for Tensor Program Tuning." ASPLOS 2023.** -- verified. Learned cost model with multi-task transfer to a new device using 7% of target-hardware data; the natural learned baseline and transfer comparison for Sec. 7.4 and 7.9 (F6, F7).
8. **Perry Gibson, Jose Cano. "Transfer-Tuning: Reusing Auto-Schedules for Efficient Tensor Program Code Generation." PACT 2022.** -- verified. Reuse of tuned schedules across programs/devices; relevant to the "table tuned on 370M" and cross-GPU rows of Tables 4 and 6 (F6).
9. **Weizhe Hua, Zihang Dai, Hanxiao Liu, Quoc V. Le. "Transformer Quality in Linear Time." ICML 2022.** -- verified. Origin of the mixed chunk (quadratic within chunk, linear across chunks) form that the paper attributes implicitly to GLA (F10).
10. **Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, Furu Wei. "Retentive Network: A Successor to Transformer for Large Language Models." arXiv 2307.08621, 2023.** -- unverified in this session. Chunkwise recurrent form for retention; same role as item 9 (F10).
11. **Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu. "Mooncake: Trading More Storage for Less Computation -- A KVCache-centric Architecture for Serving LLM Chatbot." USENIX FAST 2025 (best paper).** -- verified. Public request trace from the Kimi service, the production deployment of the Kimi Linear model the paper cites; the most relevant trace for a linear-attention serving replay (F11).
12. **Triton documentation, `triton.autotune` (`prune_configs_by`: `perf_model`, `top_k`, `early_config_prune`) and `triton.testing.do_bench` (L2 flush with a 256 MB buffer before each repeat).** -- verified (software documentation, cite as such). Prior art for "model, then profile top-k" and for compile-fact pruning; and the evidence that an autotuner "in the style of Triton" flushes L2, which the paper's protocol does not (F3, F6).
13. **Jason Ansel, Shoaib Kamil, Kalyan Veeramachaneni, Jonathan Ragan-Kelley, Jeffrey Bosboom, Una-May O'Reilly, Saman Amarasinghe. "OpenTuner: An Extensible Framework for Program Autotuning." PACT 2014.** -- unverified in this session. Offline/online autotuning amortisation that Eq. (6)-(7) re-express (F11).

Misrepresentations to correct (all text-only):

- Sec. 7.6 "in the style of the Triton autotuner": the Triton autotuner flushes L2 before every timed repeat; the paper's protocol does not (F3).
- Sec. 7.5 "The reference scan is at least as chunk-sensitive as ours": the measured sensitivity is to the stock thread configuration (F5).
- Abstract / Contribution 1 "a table tuned for one GPU loses up to 63%": the 63% is unsharded-to-sharded width transfer on one GPU; the cross-GPU loss is 14-49% (F8).
- Sec. 2 "Gated linear attention introduced hardware-efficient chunkwise training": the chunkwise form predates GLA (F10).
- Sec. 2 "learned cost models such as Ansor's are trained on such timings per operator and device": transfer-learned and shape-generic cost models exist (items 1, 7, 8).

Venue checks of existing entries (spot-checked): `beck2025tfla` NeurIPS 2025 -- verified; `guo2026loglinear` ICLR 2026 -- verified; `mamba2023` COLM 2024, `yang2025gateddeltanet` ICLR 2025, `pan2025marconi` MLSys 2025, `ye2025flashinfer` MLSys 2025, `wang2025burstgpt` KDD 2025, `gregersen2024inputdependent` SC-W 2024, `stojkovic2025dynamollm` HPCA 2025 -- consistent with my knowledge, not re-verified. `dao2024flashlinearattention` duplicates `gu2024mamba2` (same title, authors, venue) and should be removed; `vaswani2017attention`, `merity2017pointer`, `biderman2023pythia`, `marshall2011inequalities`, `mambaforgekernel2024` appear not to be cited in the main text and should be pruned or used.

---

## 5. Strengths

1. **Measurement discipline that the field lacks.** The split-call pitfall (Sec. 6.1) is real and common; recurrence-preserving sweeps with an fp32 reference check for every kernel and chunk (Supp. Table S-validation) set the right standard.
2. **The compiler-specialization observation** (Sec. 5: a length not divisible by 16 compiles to a different kernel with scalar loads and a different optimal chunk; Supp. Sec. S-headroom-noise shows the headroom it leaves in an incomplete key) is correct, practically important for tuning caches, and rarely stated.
3. **Honest treatment of residual value effects**: the two-way analysis, TOST equivalence bounds, redraw control, placement designs A/B/C and the warm/cold-L2 intervention (Sec. 7.1 and Supp. Sec. S-invariance-details) are a model of how to argue data-obliviousness empirically rather than assume it.
4. **Headroom-versus-noise analysis** (Supp. Sec. S-headroom-noise: held-out, split-half and null constructions) correctly treats the oracle as a minimum over noisy means; most papers in this area do not.
5. **Breadth**: three GPU generations, two timing conventions, 370M-2.8B widths, TP shard widths and five public serving traces, with the legacy measurements clearly quarantined (Supp. Sec. S-legacy).
6. **Clear negative result on the authors' own scheduler (COREY)**, reported without hedging; the routed study (Sec. 7.7) with paired, interleaved rounds and output-token checks is well designed.
7. **Code and data are released** (Code Ocean capsule), and the AI-use disclosure is explicit.
