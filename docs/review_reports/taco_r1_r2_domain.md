# ACM TACO, Round 1: Reviewer 2 (Domain) Report

**Manuscript:** "Shape Sets the Chunk: A Compile-Informed Cost Model and a No-Gain Bound for Data-Dependent Dispatch of Chunked Linear-Recurrence Kernels" (main article, 23 pp. incl. references, plus 48 pp. Supplementary Material, reviewed as one paper; kernel and cost-model code inspected for technical claims)

**Reviewer identity:** I work on GPU code generation and auto-tuning (Triton/CUDA kernel tuning, TVM/Ansor-style learned cost models, Kernel Tuner), analytical GPU performance models (Hong & Kim, GPUMech, roofline, occupancy models), and linear-recurrence/SSM kernels (Mamba selective scan, Mamba-2 SSD, FLA/Gated DeltaNet, TFLA).

---

## 1. Summary (5 lines)

1. The paper asks whether the chunk (time tile) of chunked linear-recurrence kernels should be chosen per call from activation values, from the shape, or fixed. It measures this on a Triton Mamba-1 scan written for the study, the reference CUDA Mamba-1 scan, Mamba-2 SSD, FLA Gated DeltaNet and TFLA's mLSTM kernel, on an RTX PRO 6000, a 48 GB RTX 3090 and an H800 PCIe.
2. Finding: activation statistics add nothing to the shape. The paper calls this an elementary "no-gain bound": data-dependent dispatch gains at most the headroom of a shape-keyed table, which it measures as small.
3. Finding: in eager execution the library defaults are adequate for the production kernels. Under CUDA graphs (RTX 3090 only) the SSD and Gated DeltaNet defaults lose 16–22% per call.
4. Proposal: an 8-parameter latency model per GPU that uses compile facts (registers, spills, Triton tile layout) to pick the chunk of the study's own Triton kernel without timing the target shapes. Its regret is 1.3–7.3% when the terms are selected without the target GPU, and it also covers tensor-parallel shard widths.
5. The paper is unusually honest about where the model fails (TFLA, the reference scan at fixed threads, production kernels under graphs). Measured against TACO's bar, though, the main technical artefact is specific to one non-production kernel. Its "compile facts" are partly mis-specified, the strongest auto-tuning and model baselines are missing, and the production-relevant regime (graphs, packed variable-length calls, data-centre GPUs) is thinly covered.

## 2. Recommendation and confidence

- **Recommendation:** Major Revision (borderline Reject-and-Resubmit). The measurements are careful and the negative results are useful. In its current form, though, the paper is a measurement study wrapped around a kernel-specific fitted model, and its novelty claims over the auto-tuning and analytical-model literature are not established.
- **Confidence:** 4/5. I am very familiar with Triton/CUDA tuning, performance models and SSM kernels. I checked the kernel code and the cost-model code against the text. I did not re-run any experiment.

## 3. Scores (0–100)

| Criterion | Score | One-line reason |
|---|---|---|
| Novelty | 35 | The no-gain bound is elementary (the paper says so). The entropy policy has prior art (a public preprint) or is a straw man. The cost model is a fitted occupancy/issue model of a well-known family, differentiated from tritonBLAS, Roller, Pruner, ArchTune-style ptxas pruning and Habitat only by its inputs. |
| Methodology | 55 | Recurrence-preserving timing, multiple rounds, CIs, placement and L2 controls are commendable. The "chunk" is confounded with channel block and warps, the spill input is static code size rather than traffic, the internal autotuner state of production kernels is uncontrolled, and the default timing convention is dominated by Python launch overhead. |
| Experiments | 55 | Very extensive, but on the wrong axes for TACO. Only one self-written kernel exercises the model. Graph timing of production kernels exists on one consumer GPU only. Packed variable-length calls are absent, and there are no non-recurrent tiled kernels. |
| Clarity | 40 | The abstract and contribution list are dense strings of numbers in mixed timing conventions. The text also switches between per-call, latency-weighted and absolute regret, and between launch-inclusive and device time. Hard for a TACO reader to extract what to reuse. |
| Significance | 35 | The headline chunk sensitivity belongs to a kernel that is 1.23x slower (geometric mean) than the reference it replaces. On the production kernels the eager defaults suffice. A practitioner's take-away (key on shape plus specialization bits; profile 6 chunks in 0.03–0.05 s per key) does not need the model. |

## 4. Strengths

- **S1. Measurement discipline.** All chunk changes are made inside one recurrence-preserving launch, with output checks. Calls are timed in three shuffled rounds with CIs. The placement intervention, warm versus cold L2, CUDA-graph replays and exclusive-GPU guard are rare in kernel papers and should be kept.
- **S2. Honest negative results.** The paper reports where its own model fails: TFLA 7.5% against a 1.4% default, the reference scan at 32 threads 49.5%, and the production pipelines under graphs.
- **S3. A concrete, reusable systems observation.** Triton's divisibility-by-16 and alignment specializations change the compiled kernel (vector width, registers, spills) and therefore the optimal tile. A tuning cache key that drops these bits leaves headroom that looks like a data effect.
- **S4. The CUDA-graph result for production kernels.** Defaults chosen for long calls lose 16–22% per call once the launch floor disappears, and a shape-keyed table recovers this. This is the practically most relevant finding and deserves to be the centre of the paper.

---

## 5. Major concerns

### D1. The cost model, and most of the chunk sensitivity, rest on one study-specific, non-production kernel; generality is not demonstrated

- **Location:** Abstract; Contributions bullet 2; Sec. 5 (cost model); Sec. 7 intro paragraph ("The chunk effects that could motivate dispatch are large only on the Triton Mamba-1 scan of this study"); Table 2 (framework); Sec. 7.6 ("A public kernel"); Sec. 10 (Limitations).
- **Why it matters for TACO:**
  - Every large chunk effect belongs to the authors' own Triton Mamba-1 scan: the 8.5x spread, the 19–28% static regret, the 65% at shard widths, and the 0.9/3.2/3.5% model regrets. That kernel is 0.81–1.54x the reference CUDA scan (geometric mean 1.23x, RTX 3090 only).
  - Its chunk sensitivity comes from design choices of that kernel. The whole [D_b, C] tile is register-resident; there is a non-unrolled per-state loop that reloads B_n, C_n and A_n and does masked reductions over the state tile every tile; and the batch-1 grid has only 256–512 programs.
  - On every production kernel the defaults suffice in eager execution. The compile-informed form fails on the one independent kernel with a wide chunk range (TFLA: 7.5% vs. a 1.4% default) and on the reference scan at fixed threads (49.5%).
  - Critically, the TFLA, SSD and Gated DeltaNet tests did **not** use compile facts. Table 2 lists "none" for SSD and Gated DeltaNet, and the TFLA "five-term model" has no register or spill terms, although all three are Triton kernels whose compile facts are as easy to read as those of the authors' kernel.
  - So the paper's distinguishing ingredient (compile facts) has been tested on exactly one kernel, written by the authors. A TACO reader cannot tell whether the method transfers to tile-size selection in attention, GEMM or FLA kernels, which is the question this venue cares about.
- **Required actions:**
  1. Apply the full compile-informed recipe (Sec. 5, steps 1–5, *with* registers, spills, shared memory and layout read per configuration) to at least two independent Triton kernels whose tile choice matters. Natural candidates:
     - the SSD inner tiles. Their autotune key omits L and batch (the paper notes this in Sec. 2), so whether the optimal inner tile moves with L and B is exactly the paper's question for a production kernel;
     - FLA Gated DeltaNet's BT/BK/BV tiles;
     - TFLA's Triton mLSTM kernels with compile facts.
  2. Apply it to one non-recurrent tiled kernel, such as Triton FlashAttention (BLOCK_M/BLOCK_N, num_stages) or a Triton GEMM. Compare against the kernel's own autotuner and against tritonBLAS-style analytical selection (see Missing related work).
  3. If transfer cannot be shown, re-position the paper as a measurement study of chunk dispatch. Remove "Compile-Informed Cost Model" from the title and present the model as a case study.
  4. Motivate the Triton Mamba-1 kernel: is any production or portable stack (e.g. ROCm, torch.compile) running a Triton Mamba-1 selective scan whose design resembles yours? If so, evaluate that kernel too.
- **Effort:** (c), about 10–15 GPU-hours on two GPUs (one data-centre GPU such as H100/H800 plus one consumer part). Compilation is CPU-only.

### D2. What "the chunk" means in the Triton kernel is misstated: chunk, channel block and warps are changed together

- **Location:**
  - Sec. 6.2: "so the chunk changes only the tiling";
  - Table 1, row "Mamba-1, Triton": "registers grow with C";
  - Sec. 7.1 and Fig. 2: chunk spread up to 8.5x;
  - Supplementary Table "Compile facts of the default launch table";
  - Sec. 7.4 ("Auto-tuning baseline").
- **Why it matters:**
  - The per-call sweeps vary the chunk "at the default launch configuration". That table couples C with the channel block D_b and the warp count n_w: C <= 128 uses D_b = 8 and 2 warps; C = 256 and 512 use D_b = 4 and 2 warps; C = 1024 uses D_b = 4 and 4 warps.
  - Going from C = 128 to C = 256 therefore **doubles the number of programs** (P = ceil(d/D_b)·B). Elements per thread (16) and registers (202 on sm_120) stay identical, contradicting "registers grow with C".
  - At batch 1 and d = 2048 on a 188-SM GPU this is 1.4 versus 2.7 programs per SM, a large change in latency hiding for a latency-bound kernel.
  - The reported "chunk effect" therefore mixes tiling with grid parallelism and warps. This is the same confound the paper (correctly) criticizes in the reference CUDA scan, whose stock dispatch pairs chunk with thread count (Sec. 7.5, "Fixed thread count").
  - The provenance of the default table is not stated in the paper. A comment in the released code says it came from a short search on an RTX 3070, a GPU used in the paper only for overhead measurements.
- **Required actions:**
  1. State the coupling and the provenance in the main text, and correct Sec. 6.2 and Table 1.
  2. From the existing 184-configuration sweep at the 12 keys, report the chunk effect at **fixed** (D_b, n_w, num_stages), and decompose the 8.5x spread and the 19–28% static regret into a tiling part and a grid/warp part.
  3. Re-state the dispatch knob honestly as a launch tuple (C, D_b, n_w), or select D_b and n_w independently.
- **Effort:** (a) + (b), offline re-analysis of existing sweep data. Optionally (c), under 1 GPU-hour, to fill gaps in the sweep at the captured-call shapes.

### D3. The "compile facts" are partly mis-specified or unused, and the published model differs from the implementation

- **Location:** Sec. 5 and Eq. (3) (definitions of ω and μ); Supplementary "Compile facts" and "Reference CUDA scan"; Table 5 caption; Table 2 (SSD and Gated DeltaNet: "none").
- **Why it matters:** The paper's novelty claim is that the chunk dependence "enters only through the assembler-reported registers and spills and the Triton tile layout". The following points undermine that claim as written:
  1. **ω, the spill input, is static code size, not traffic.** The text calls ω "the spill traffic per thread in words". The implementation uses the ptxas "bytes spill stores/loads" counts (or Triton's equivalent). These are the static sizes of spill instructions in the binary, not dynamic traffic. The model then multiplies ω by N per tile, implicitly assuming every spill instruction sits in the per-state inner loop. Likewise, the L1 footprint behind μ should be the local-memory stack frame, not the spill-store byte count.
  2. **Occupancy ignores shared memory.** k is computed from registers, warp and block limits only. Shared memory is not used, although the cross-warp associative scan (C = 512 and 1024 span 2–4 warps along the scan axis) and Triton layout conversions use it, and the recipe (Sec. 5, step 1) lists shared memory as an input. The L1 capacity also ignores the shared-memory carveout.
  3. **Eq. (3) does not match the code.** The implemented predictor returns max(T, bytes/bandwidth), using vendor DRAM bandwidth, plus fixed-at-zero optional terms. The bandwidth floor appears nowhere in the paper.
  4. **The reference CUDA scan does not use the compile facts that are available.** Table 5's caption says "a prebuilt kernel exposes no compile facts". Yet the authors built the kernel themselves (chunk as a runtime argument), and the Supplement says the fixed count of 168 registers is "the count that the assembler reports for most instantiations". Per-instantiation registers, spills and shared memory are available from `ptxas -v` at build time or `cuobjdump -res-usage` on the shared object.
     - The 49.5% failure at a fixed 32 threads is explained by "its fixed register count does not describe the long per-thread tiles". That is, the failure stems from **not** using the compile facts, and is not a limit of the method.
  5. **SSD and Gated DeltaNet are Triton kernels** whose per-chunk compile facts could be read the same way. Instead they get only a work-term model with no compile facts.
- **Required actions:**
  1. Correct the definitions of ω and μ, or measure dynamic spill traffic (Nsight Compute local-memory load/store sectors) at the calibration keys and show the static proxy is adequate.
  2. Include shared memory in occupancy, or justify omitting it per configuration.
  3. Make Eq. (3) match the implementation, including the bandwidth floor and the fixed-zero terms.
  4. Read per-instantiation compile facts for the reference scan, refit both the stock-pairing and the 32-thread families, and report whether the 49.5% failure disappears.
  5. Read compile facts for the SSD and FLA kernels when testing transfer (see D1).
- **Effort:** (b) for items 1 (proxy check), 3 and 4, because compile facts are CPU-only and the fits reuse existing timings. (c) about 1 GPU-hour for the Nsight Compute spill-traffic check.

### D4. The comparison baselines are not the strongest or most relevant ones, and the calibration budget is not compared at equal cost

- **Location:** Table 5 and Sec. 7.4; Supplementary "Baselines", "TFLA's runtime expression" and the TFLA kernel section; Sec. 8 ("Where a new shape cannot be tuned … TFLA's analytical runtime … selects worse").
- **Why it matters:**
  1. **The TFLA comparison is a straw man in one direction and missing in the other.**
     - TFLA's runtime expression models chunkwise-parallel kernels with an intra-chunk matrix product whose FLOPs grow with C². The study's Mamba-1 kernel has no such product; it is an associative scan with linear per-tile work. Fitting TFLA's expression to it and concluding "the timings reject its quadratic arithmetic term" is unsurprising. Citing it in the Discussion as evidence that the paper's model "is a cheaper default" than TFLA's is not a fair comparison.
     - Conversely, TFLA's own expression was never evaluated on TFLA's own kernel, the one place where it applies (the Supplement's TFLA section fits a different five-term model).
  2. **No Triton-native model-plus-top-k baseline.** Sec. 7.4 notes that Triton's autotuner can already rank configurations with `prune_configs_by={'perf_model', 'top_k'}`. The "cost model, top-2" row is that mechanism with the authors' model. What is missing is the same mechanism with a **generic** off-the-shelf model, such as an occupancy-plus-roofline model with no kernel-specific terms, or a zero-parameter compile-facts rule (e.g. the largest chunk without spills that maximizes resident warps or minimizes waves). Without these, the value of the eight fitted parameters and the hand-selected terms is not isolated. Note that the Supplement's three-parameter restriction already beats the full model on the RTX 3090.
  3. **No search-based tuner at equal profiling budget.**
     - Kernel Tuner with Bayesian optimization (Willemsen et al. 2021) and KTT's dynamic (online) tuning (Petrovič et al. 2020) are the standard tools, and should be compared with top-2 at a budget of 2–3 profiles per new key.
     - Draft-then-verify tuners (Pruner, ASPLOS 2025) are the published form of "rank with a cheap model, verify the top candidates".
  4. **The calibration cost dwarfs the cost of the alternative.**
     - The model needs 2298 timings per GPU (8–14 minutes, including the full 184-configuration sweep at 12 keys).
     - By the paper's own numbers (Supplementary "Per-key tuning costs"), profiling all six chunks at **all 33 keys of the captured workload costs 2.5 s and 5.0 s**, and retuning a shard width costs 0.03–0.05 s per key.
     - The model's practical niche (keys that cannot be enumerated or timed before serving) is therefore narrow, and the paper must show the regret-versus-calibration-budget curve, e.g. calibration on only the 72 static timings or on k keys.
  5. **Learned and analytical cross-device predictors are cited but not positioned or compared.** For transfer to an untimed GPU (H800 with another GPU's parameters: 0.9–5.4%), Habitat, TLP, CDMPP and NeuSight are the relevant comparisons. At minimum, the paper must argue quantitatively why they do not apply.
- **Required actions:**
  1. Remove or re-frame the TFLA-on-Mamba-1 comparison, and add TFLA's expression fitted on TFLA's kernel.
  2. Add a zero-parameter compile-facts rule, and a generic occupancy/roofline `perf_model` with top-k.
  3. Add a Bayesian-optimization or online tuner at an equal profile budget per key.
  4. Plot regret versus calibration timings (learning curve), and compare total cost (calibration plus per-key profiling) against "profile every key on first sight" for the captured workload and the serving traces.
- **Effort:** (b) for items 1 (first half), 2 and 4, since all timings exist. (c) about 2–4 GPU-hours for TFLA-on-TFLA re-timing (if needed) and the BO/online-tuner runs.

### D5. The production-relevant regime is under-covered: eager-mode Python launch overhead dominates many conclusions; graphs, packed variable-length calls and data-centre GPUs are missing for the production kernels

- **Location:**
  - Sec. 6.4 (timing protocol; launch-inclusive tables "unless stated otherwise");
  - Abstract and Contributions bullet 1 ("In eager execution the defaults suffice");
  - Sec. 7.6 (production kernels; CUDA graphs on the RTX 3090 only);
  - Sec. 6.3 (packed variable-length prefill "not timed");
  - Supplementary "CUDA-Graph Timing" and "Timing Conventions";
  - Table 7.
- **Why it matters:**
  1. **The host launch cost decides the eager-mode verdicts.**
     - The launch-inclusive floor is 56–90 µs for the Mamba-1 Triton call and about 740 µs for the 5-kernel SSD pipeline at batch 1. That is a property of the Python harness and Triton launcher, not of the kernels.
     - 46–64% of the 370M Mamba-1 calls have an optimal latency below 100 µs. Consequently, for example, "one chunk loses only 1.7% at batch 1" launch-inclusive but 19% in device time.
     - "Defaults suffice in eager execution" for SSD and Gated DeltaNet is largely the statement that a large host overhead masks kernel time.
     - Production serving engines replay CUDA graphs (the paper cites vLLM's), and recent work fuses the five SSD kernels into one Triton kernel, removing the floor entirely. The PyTorch "Accelerating Mamba2 with Kernel Fusion" blog (2026) reports that the optimal SSD chunk for Mamba-2 2.7B moves from 256 (unfused) to 128 (fused). This directly shows that the paper's SSD conclusions are a property of the unfused pipeline and its launch regime.
  2. **Graph and production-kernel timings exist only on the RTX 3090**, a modified 48 GB consumer board. The H800 covers only the Mamba-1 kernels.
  3. **Device time is unreliable for the multi-kernel pipelines.** The median within-block CV is 60–85% (Supplementary Table "CUDA-graph replays …"), yet the SSD 2.7B and Gated DeltaNet "above the floor" claims of Sec. 7.6 rest partly on these timings.
  4. **Packed variable-length prefill is not timed.** Production prefill for SSD and Gated DeltaNet runs packed variable-length calls (cu_seqlens), not equal-length padded batches cropped to the shortest member. Moreover, serving engines constrain the chunk:
     - vLLM's Mamba-2 path builds chunk-aligned variable-length metadata ("pseudo-chunks"), and the mamba state can be retrieved every chunk_size tokens for prefix caching;
     - so per-call chunk dispatch interacts with prefix-cache alignment and state checkpointing. The deployment recipe (Sec. 8) does not mention this.
  5. **Headline regrets overweight short calls.** Per-call relative regret weights every call equally, so a 25-token call counts as much as an 8192-token one. Table 7 gives absolute milliseconds but never the share of the forward pass.
- **Required actions:**
  1. Make device time or CUDA-graph replay the primary convention throughout, and keep launch-inclusive as a secondary column.
  2. Replicate the graph study (SSD 370M/2.7B, Gated DeltaNet 1.4B, Mamba-1) on the H800 or an H100.
  3. Time packed variable-length calls of the production kernels for a few serving-trace batches.
  4. Discuss chunk-alignment constraints in serving engines in the recipe.
  5. Report latency-weighted shares and percent of forward/prefill time next to every per-call headline.
  6. If the fused SSD kernel is available, repeat the SSD chunk sweep with it.
- **Effort:** (a) for items 1, 4 and 5 (re-presentation). (c) about 6–10 GPU-hours on one data-centre GPU for items 2, 3 and 6.

### D6. Uncontrolled internal autotuner state in the production kernels confounds their chunk sweeps

- **Location:** Sec. 2 ("the autotuning decorators of the SSD pipeline key its inner tiles on the chunk, head and state sizes, not on the length or the batch"); Supplementary "Auto-tuning" ("its own internal block-size autotuning is part of the measured tuning time"); Sec. 7.6 and the graph results.
- **Why it matters:**
  - The SSD kernels, and FLA's kernels, carry `@triton.autotune` decorators whose keys exclude L and B.
  - Within a process, the inner-tile configuration used at chunk C is therefore the one tuned at the **first shape seen with that chunk**. It differs between sessions, rounds and shuffled call orders, and is generally not optimal for later shapes.
  - Under graphs the defaults lose 16–22%, mostly on short calls. Part of this may be inner tiles tuned for a different (L, B), not the chunk itself.
  - The paper does not say how the autotune cache was seeded, reset or pinned across rounds and sessions.
- **Required actions:**
  1. Document the autotune-state policy.
  2. Re-time a stratified subset (e.g. 200 calls) with the inner configurations pinned and, separately, re-tuned per shape. Report how much of the graph-mode regret is attributable to the chunk versus the inner tiles.
  3. This doubles as a natural transfer case for D1.
- **Effort:** (c) about 2–3 GPU-hours.

### D7. Novelty and positioning: prior art for the entropy policy, an elementary bound promoted to the title, and the cost model not differentiated from the closest work

- **Location:** Sec. 1 ("to our knowledge, no published runtime or paper proposes value-dependent dispatch for dense chunked kernels"); title; Sec. 4 (Proposition 1, "The bound is elementary"); Sec. 5 ("its novelty is narrow"); Sec. 6.1 (split-call pitfall) and Contributions bullet 5.
- **Why it matters:**
  1. **There is public prior art for the entropy policy.** arXiv:2604.10597 (April 2026, v3 May 2026), "COREY: Entropy-Guided Runtime Chunk Scheduling for Selective Scan Kernels", proposes a runtime scheduler that maps fixed-bin activation entropy to the chunk of selective-scan kernels. That is precisely the policy studied and refuted here. The "no published paper" sentence is therefore incorrect.
     - If this is the authors' own prior work, TACO's double-anonymous policy still requires citing it in the third person and stating what the present paper changes or retracts. For example, the preprint's reported latency gains versus this paper's finding that entropy adds nothing.
     - If it is not the authors' work, it is the direct prior art that the paper refutes, and should be engaged explicitly.
     - Either way, without it the entropy-driven scheduler reads as a straw man constructed to be refuted.
  2. **Proposition 1 is elementary.** Part (a) says a selection is never faster than the per-input minimum. Part (b) is the standard data-obliviousness argument. Neither belongs in the title of a TACO article; it is a lemma that organizes the measurements.
  3. **The cost model is not differentiated from the closest work.** It belongs to the occupancy/issue/latency-hiding family of models, so its novelty must be argued against:
     - tritonBLAS: analytical Triton GEMM parameter selection without autotuning, over 95% of autotuned performance;
     - ptxas-based pre-compilation pruning of Triton configurations;
     - Roller's micro-performance model;
     - Pruner's draft-then-verify search;
     - CRAT, Kayiran et al., and related work on register allocation versus thread-level parallelism (the core trade-off the chunk controls here);
     - sample-free dynamic-shape compilers (Vortex, MikPoly, DietCode);
     - the classic model-driven versus empirical-optimization comparison (Yotov et al. PLDI 2003).
     Currently only the analytical GPU models of 2009–2017, Habitat and TenSet/TLP are discussed.
  4. **The "measurement pitfall" (split-call chunk timing) is not substantiated as a contribution.** It is presented as a contribution without a single citation showing that published work committed it. Either document instances or move it into the methodology section as a protocol note.
- **Required actions:**
  1. Cite and position arXiv:2604.10597, and correct the novelty sentence.
  2. Demote the bound from the title to a lemma.
  3. Add a positioning paragraph or table against the works listed in Section 7 of this report, stating for each what it needs (timings, traces, training data), what it predicts (latency, configuration rank), and whether it handles untimed shapes or GPUs.
  4. Substantiate or demote the pitfall contribution.
- **Effort:** (a).

### D8. Kernel-quality artefacts are treated as shape effects, and the register/occupancy trade-off is not explored with the standard knobs

- **Location:** Sec. 5 ("Compiler specialization and parallel shapes": unaligned lengths "about twice as slow at batch 1"); Sec. 7.4 (search space: chunk, channel tile, warps, stages); Supplementary "Compile facts" (255-register cap and spills at C >= 512).
- **Why it matters:**
  1. **The 2x slowdown for lengths that are not multiples of 16 is a fixable kernel deficiency.** It comes from the row stride (= L) and the mask bound not being divisible by 16, which forces scalar loads. Allocating the producer's output with a row stride padded to a multiple of 16, or peeling the tail tile, would restore vector loads. Treating this as a shape effect that a cache must key on, and that the model must "capture", inflates the problem.
  2. **The register-versus-occupancy trade-off has a direct knob that is not explored.** Triton exposes `maxnreg` (`CUDAOptions`, the PTX `.maxnreg` directive), and CUDA exposes `-maxrregcount`. The paper instead varies the chunk to move along this trade-off, and the search space omits the register cap. If capping registers at a large chunk recovers occupancy, much of the chunk sensitivity disappears.
  3. **The batch-1 grid is small.** With only 256–512 programs, the kernel is latency-bound by construction. A kernel parallelized along L (inter-chunk state passing, as in SSD) would behave differently.
  - A TACO reader will ask whether the 19–28% static regret survives a better-engineered kernel.
- **Required actions:**
  1. Add `maxnreg` to the launch search space at the 12 keys.
  2. Test stride padding on the unaligned calls.
  3. Report the static-chunk regret and the model's advantage for the improved kernel.
- **Effort:** (c) about 2–4 GPU-hours per GPU (two GPUs suffice).

### D9. The tensor-parallel "no retuning" claim relies on a straw-man baseline

- **Location:** Abstract ("At tensor-parallel shard widths … needs no retuning (2.9 to 6.5%), where the unsharded table loses up to 65%"); Sec. 7.5 "Tensor-parallel widths"; Supplementary Table "Mean per-call relative regret at tensor-parallel widths".
- **Why it matters:**
  - The tensor-parallel degree t is fixed when a deployment is configured, so nobody would serve shards with a table tuned at the unsharded width. The paper itself notes that retuning at the shard width costs 0.03–0.05 s per key.
  - On the H800, a table tuned at the calibration width beats the model (2.2% vs. 2.9%).
  - On the RTX 3090 the per-shard model regret reaches 15.2% (t = 2 for 370M, and t = 4 for 1.4B, both at width 1024).
  - All shard timings use synthetic inputs on a single GPU.
- **Required action:** Compare the model against "profile at deployment time" in total cost (calibration plus profiling), drop the unsharded-table comparison from the abstract, and report per-shard worst cases next to the pooled means.
- **Effort:** (a), using existing data.

---

## 6. Minor concerns

1. **Abstract and contribution list.** Over 25 numbers in mixed conventions (launch-inclusive and device time; per-call, latency-weighted and absolute). Rewrite around 3–4 numbers in one convention, and state explicitly that the model applies to one kernel.
2. **Triton specialization.** Vectorization of the masked loads depends on the divisibility of both the row stride (L) and the mask bound (L), not only on the stride (Sec. 5). Also show concretely how a real cache loses the bit: Triton's autotuner caches on the values of `key` arguments, so with bucketed keys (custom key functions, Inductor size hints) the divisibility bit is dropped. One sentence plus a citation to the Triton documentation would make the recommendation actionable.
3. **The state loop and the latency term.** The kernel's state loop (`for n in range(N)`) is not a `tl.static_range`. State whether it is unrolled after compilation, because the interpretation of the latency term A = N·τ_m (serialized per-state loads) depends on it.
4. **Hardware realism.** The RTX 3090 is a non-stock 48 GB board, and the released code labels the host "vgpu3090". Please confirm the GPU was not virtualized or time-sliced (vGPU scheduling would invalidate microsecond timings), and justify a modified consumer board for the production-kernel graph study.
5. **Reference scan coverage.** It was built and compared only on the RTX 3090 (Sec. 6.2: "for which a build … is available"). An sm_90 build for the H800 is straightforward, and the 1.23x claim should cover at least two GPUs.
6. **Work-term model for SSD and Gated DeltaNet.**
   - W = BL(a + bC) + B⌈L/C⌉g has no padding term; for L < C the intra-chunk work scales with ⌈L/C⌉·C, not L. This likely explains why it keeps chunk 64 for Gated DeltaNet under graphs (12.2% even after refitting) while short calls prefer 16/32.
   - Add the padding term, and test the trivial rule C = min(C_default, next_pow2(L)) on the existing graph timings.
7. **The abstract misattributes the graph-mode model failure.** "It does not transfer … 12 to 22% on the production kernels under graphs" is mostly a calibration-regime mismatch: with the graph-measured floor, SSD is at 4.4% and 1.2%, and refitted on the replays at 1.3% and 0.8% (Supplementary "CUDA-Graph Timing"). Say so in the main text and the abstract.
8. **Limitations wording.** "The reference CUDA scan was timed at one fixed thread count (32)" is ambiguous, since the main results use the stock pairing (32–128 threads). Rephrase.
9. **Histogram cost.** The 4.43 ms per-call histogram cost (Sec. 7.8) comes from an implementation the paper itself calls not bandwidth-optimal. Since the argument rests on the information side, drop it from headline claims, or implement a fused statistic in the convolution epilogue with a device-side decision (CUDA-graph conditional nodes) to bound the cost side credibly.
10. **Unlocked clocks.** Clocks are unlocked and sessions differ (a second PRO 6000 session gives larger launch-inclusive regrets). The fitted τ parameters are clock-dependent; report the clock distribution at calibration versus evaluation.
11. **Form selection.** The cost-model form was chosen on the test calls of two of the three GPUs (Sec. 5, "Model form and data"). Use only the form-blind numbers (H800 frozen form; leave-calibration-shape-out selection) in the abstract and conclusion. The current "1.3 to 7.3%" range is the right one, but 0.9% and 3.2% still appear in Table 2 and Sec. 7 without a qualifier at first use.
12. **Taxonomy (Table 1).** Add a column for whether the chunk is a constexpr (one compiled variant per chunk) or a runtime argument. This determines compilation cost, cache size and graph-capture behaviour, all relevant to deployment.
13. **Related-work hygiene.**
    - Cite FlashAttention-2/3 alongside FlashAttention-1 when discussing tile choice.
    - Move the vLLM CUDA-graph citation into the main text.
    - Cite Lightning Attention-2, the first tiled intra-/inter-block linear attention.
    - Cite TileLang as a modern tile-level DSL whose autotuner faces the same question.
14. **Source hygiene.** The Supplement's LaTeX header still says "Supplementary Material for the IEEE TPDS submission" and refers to an IEEEtran wrapper, and figure files carry a "tpds_" prefix. This is invisible in the PDF but visible in any source upload; clean it before submitting sources.
15. **Recipe (Sec. 8).**
    - For Triton, say which compiled-kernel fields are read (registers, spills, shared memory), and that Triton's spill count, like ptxas's, is static.
    - "Time about 2300 calibration points" should come with the learning curve requested in D4.

## 7. Missing related work (each verified online during this review)

**Prior art for the studied policy**

- B. Ma, J. Wu, W. Yan. "COREY: Entropy-Guided Runtime Chunk Scheduling for Selective Scan Kernels." arXiv:2604.10597, Apr. 2026 (v3 May 2026). It proposes the entropy-to-chunk runtime scheduler that the manuscript studies; it must be cited and positioned (see D7).

**Analytical or model-driven parameter selection without (or with less) autotuning**

- R. Swann, M. Osama, X. Guo, B. Nelson, L. Zhang, A. Brown, Y. Ong, A. Yazdani, S. Siddens, G. Dasika, A. Underwood. "tritonBLAS: Triton-based Analytical Approach for GEMM Kernel Parameter Selection." arXiv:2512.04226, Dec. 2025. The closest concurrent work: an analytical Triton parameter selector reaching over 95% of autotuned performance with zero tuning time.
- H. Zhu, R. Wu, Y. Diao, S. Ke, H. Li, C. Zhang, J. Xue, L. Ma, Y. Xia, et al. "ROLLER: Fast and Efficient Tensor Compilation for Deep Learning." USENIX OSDI 2022. A micro-performance model that replaces on-device search.
- K. Yotov, X. Li, G. Ren, M. Cibulskis, G. DeJong, M. Garzarán, D. Padua, K. Pingali, P. Stodghill, P. Wu. "A Comparison of Empirical and Model-Driven Optimization." PLDI 2003, pp. 63–76, doi:10.1145/781131.781140. The classic framing of the paper's model-versus-profiling question.
- L. Qiao, J. Shi, X. Hao, X. Fang, S. Zhang, M. Zhao, Z. Zhu, J. Chen, H. An, X. Tang, B. Li, H. Yuan, X. Wang. "Pruner: A Draft-then-Verify Exploration Mechanism to Accelerate Tensor Program Tuning." ASPLOS 2025, doi:10.1145/3676641.3716269 (arXiv:2402.02361). The published form of "rank with a cheap model, verify the top-k".

**Dynamic shapes and sample-free tuning**

- Y. Zhou, H. Zhu, Q. Qiu, W. Cui, Z. Liu, C. Guo, S. Feng, J. Meng, H. Lan, J. Leng, W. Zhu, M. Deng. "Vortex: Efficient Sample-Free Dynamic Tensor Program Optimization via Hardware-aware Strategy Space Hierarchization." arXiv:2409.01075, 2024.
- F. Yu, G. Li, J. Zhao, H. Cui, X. Feng, J. Xue. "Optimizing Dynamic-Shape Neural Networks on Accelerators via On-the-Fly Micro-Kernel Polymerization" (MikPoly). ASPLOS 2024.

**Cross-device performance prediction (relevant to the untimed-GPU claims)**

- S. Lee, A. Phanishayee, D. Mahajan. "Forecasting GPU Performance for Deep Learning Training and Inference" (NeuSight). ASPLOS 2025 (arXiv:2407.13853). Tile-granularity prediction on unseen GPUs.
- H. Hu, J. Su, J. Zhao, Y. Peng, Y. Zhu, H. Lin, C. Wu. "CDMPP: A Device-Model Agnostic Framework for Latency Prediction of Tensor Programs." EuroSys 2024 (arXiv:2311.09690).
- B. H. Ahn, S. Kinzer, H. Esmaeilzadeh. "Glimpse: Mathematical Embedding of Hardware Specification for Neural Compilation." DAC 2022.

**Runtime kernel selection, tuning caches and online tuning**

- R. Swann, M. Osama, K. Sangaiah, J. Mahmud. "Seer: Predictive Runtime Kernel Selection for Irregular Problems." CGO 2024 (arXiv:2403.17017). It weighs cheap versus expensive input features for runtime selection, directly relevant to the decision-cost argument.
- S. Heldens, B. van Werkhoven. "Kernel Launcher: C++ Library for Optimal-Performance Portable CUDA Applications." iWAPT 2023 at IPDPS (arXiv:2303.12374). It captures kernel launches and serves tuned configurations by problem size at runtime, i.e. a tuning cache.
- F. Petrovič, D. Střelák, J. Hozzová, J. Oľha, R. Trembecký, S. Benkner, J. Filipovič. "A Benchmark Set of Highly-efficient CUDA and OpenCL Kernels and its Dynamic Autotuning with Kernel Tuning Toolkit." Future Generation Computer Systems 108 (2020) 161–177. Online/dynamic autotuning as the alternative to offline caches and to the paper's online-policy break-even.

**Auto-tuning search methods**

- F.-J. Willemsen, R. van Nieuwpoort, B. van Werkhoven. "Bayesian Optimization for Auto-Tuning GPU Kernels." PMBS@SC 2021, doi:10.1109/PMBS54543.2021.00017.
- A. Rasch, R. Schulze, M. Steuwer, S. Gorlatch. "Efficient Auto-Tuning of Parallel Programs with Interdependent Tuning Parameters via Auto-Tuning Framework (ATF)." ACM TACO 18(1), 2021, doi:10.1145/3427093. Interdependent tile parameters, the coupled (C, D_b, n_w) space of D2; also a TACO precedent.

**Register / thread-level-parallelism trade-off (the mechanism the Triton chunk controls)**

- X. Xie, Y. Liang, X. Li, Y. Wu, G. Sun, T. Wang, D. Fan. "Enabling Coordinated Register Allocation and Thread-level Parallelism Optimization for GPUs" (CRAT). MICRO 2015, doi:10.1145/2830772.2830813.
- O. Kayiran, A. Jog, M. T. Kandemir, C. R. Das. "Neither More Nor Less: Optimizing Thread-level Parallelism for GPGPUs." PACT 2013, pp. 157–166.

**Analytical GPU models newer than those cited**

- J. Lee et al. "GCoM: A Detailed GPU Core Model for Accurate Analytical Modeling of Modern GPUs." ISCA 2022, doi:10.1145/3470496.3527384.
- Y. Arafa, A.-H. Badawy, A. ElWazir, A. Barai, A. Eker, G. Chennupati, N. Santhi, S. Eidenbenz. "Hybrid, Scalable, Trace-Driven Performance Modeling of GPGPUs" (PPT-GPU). SC 2021.

**Wave quantization (the paper's "waves" term)**

- M. Osama, D. Merrill, C. Cecka, M. Garland, J. D. Owens. "Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU." PPoPP 2023 (arXiv:2301.03598). Wave-quantization effects of tile choice, and a single-tile-size alternative to kernel-selection heuristics.

**Linear-attention and SSM kernels relevant to chunk choice**

- Z. Qin, W. Sun, D. Li, X. Shen, W. Sun, Y. Zhong. "Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models." arXiv:2401.04658, 2024.
- PyTorch Blog. "Accelerating Mamba2 with Kernel Fusion." Feb. 2026 (updated May 2026), https://pytorch.org/blog/accelerating-mamba2-with-kernel-fusion/. A fused single-launch SSD Triton kernel, 1.50–2.51x on A100/H100. It reports that the optimal chunk for Mamba-2 2.7B moves from 256 (unfused) to 128 (fused), and notes that serving code may assume a fixed chunk. This directly bears on D5.
- L. Wang et al. "TileLang: A Composable Tiled Programming Model for AI Systems." arXiv:2504.17577, 2025.

**Tooling and serving facts referenced in this report**

- Triton `autotune` documentation (`key`, `prune_configs_by` with `perf_model` and `top_k`), https://triton-lang.org/main/python-api/generated/triton.autotune.html.
- Triton NVIDIA backend `CUDAOptions.maxnreg` (maps to the PTX `.maxnreg` directive), triton-lang/triton, `third_party/nvidia/backend/compiler.py`.
- vLLM Mamba-2 attention backend and metadata documentation (chunk-aligned variable-length metadata and prefix-caching alignment), https://docs.vllm.ai/en/latest/api/vllm/v1/attention/backends/mamba2_attn/.

## 8. Questions for the authors

1. How was the default launch table chosen, and on which GPU? What is the chunk effect at fixed (D_b, n_w)?
2. Are the ptxas or Triton spill counts used for ω static byte counts? Have you checked them against Nsight Compute local-memory traffic?
3. How was the internal `@triton.autotune` state of the SSD and FLA kernels controlled across rounds, sessions and shuffled orders?
4. Why does Table 5 say the reference scan "exposes no compile facts" when the Supplement uses its assembler-reported register count?
5. What fraction of a 2.8B forward or prefill pass on an H100/H800 under CUDA graphs is the chunk effect, for the production kernels and for your Triton kernel?
6. Does the Triton Mamba-1 chunk sensitivity survive `maxnreg` capping and stride padding?
7. What is the relationship between this submission and arXiv:2604.10597?

## 9. What would be needed for Accept

1. **Generality (D1):** the full compile-informed recipe, with compile facts, applied to at least two independent Triton kernels (e.g. SSD inner tiles, FLA Gated DeltaNet tiles, TFLA) and one non-recurrent tiled kernel (attention or GEMM), each compared with its own autotuner. Alternatively, an honest re-scoping of the paper as a measurement study, with the title changed accordingly.
2. **Technical correctness (D2, D3):** the (C, D_b, n_w) coupling disclosed and disentangled. ω, μ and occupancy corrected or validated. Eq. (3) aligned with the implementation. Real compile facts used for the reference scan and the production kernels.
3. **Baselines (D4):** a zero-parameter compile-facts rule, a generic `perf_model` with top-k, a Bayesian-optimization or online tuner at equal budget, TFLA's model on TFLA's kernel, and a calibration-budget learning curve with an equal-cost comparison against "profile every key".
4. **Production regime (D5, D6):** device time or graphs as the primary convention, graph replication of the production kernels on a data-centre GPU, packed variable-length calls, autotune state controlled, latency-weighted and percent-of-forward-pass reporting, and serving-engine chunk-alignment constraints discussed.
5. **Positioning (D7):** prior art for the entropy policy cited and positioned, the bound demoted to a lemma, and explicit differentiation from tritonBLAS, Roller, Pruner, NeuSight/CDMPP, CRAT and Yotov et al.
6. **Kernel quality (D8):** `maxnreg` and stride padding tested, showing the chunk sensitivity is not an artefact of an under-engineered kernel.

**Estimated additional GPU time for the requested experiments:** about 25–35 GPU-hours. Of these, roughly half are on one data-centre GPU (H100/H800): graphs, packed variable-length calls, transfer kernels. The rest are split across one consumer GPU and the H800 (maxnreg, padding, autotune-state and BO runs). Items D2, D3 (mostly), D4 (mostly), D7 and D9 need no new GPU time.
