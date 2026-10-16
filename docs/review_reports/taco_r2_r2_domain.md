# Review Report: Reviewer 2 (domain expert)

**Submission:** "Shape Sets the Chunk: Selecting and Keying the Chunk of Chunked Linear-Recurrence GPU Kernels" (ACM TACO, new regular paper, double-anonymous)
**Reviewer expertise:** GPU kernels for state-space models and linear attention (Mamba-1 selective scan, Mamba-2 SSD, FLA/Gated DeltaNet, TFLA), Triton and CUDA, autotuning, analytical and learned GPU performance models.
**Material examined:** main paper (PDF and source, 20 body pages plus references, so within the page limit), Supplementary Material (31 pages), bibliography, the kernel and cost-model code in the artifact (Triton Mamba-1 kernel, runtime-chunk build of the reference CUDA scan, cost model, timing and graph-capture harness), and the launch-space sweep result files, which I re-analysed offline (see M1).

---

## 1. Summary

The paper asks how a runtime should choose the chunk size of chunked linear-recurrence kernels. It studies a Triton Mamba-1 selective scan that the authors wrote for the study, a runtime-chunk rebuild of the reference CUDA Mamba-1 scan, the Mamba-2 SSD Triton pipeline (mamba_ssm), FLA's chunked Gated DeltaNet kernel and, briefly, TFLA's mLSTM kernel. The experiments run on an RTX PRO 6000 (Blackwell), a modified 48 GB RTX 3090 and, for the Triton Mamba-1 scan only, an H800 PCIe. The inputs are prefill calls captured from 370M to 2.8B research checkpoints. There are four main claims:

1. **Values do not matter.** The optimal chunk is a function of the shape key: dimensions, Triton specialization bits and device. Activation statistics add no useful information. An elementary lemma bounds any data-dependent policy by the residual regret (headroom) of a shape-keyed table, and that headroom is below one host synchronization.
2. **The execution regime decides.** Library defaults are adequate host-bound. Device-bound (CUDA-graph replay), the optimum of SSD and Gated DeltaNet moves with the shape and the defaults lose 1.0 to 5.9% of scan time.
3. **A cost model is a case study.** An 8-parameter model of the Triton scan uses compiler resource reports (registers, spills, Triton tile layout), is calibrated once per GPU, and selects chunks with 3.0 to 6.2% regret when its terms are chosen a priori. Profiling enumerable keys is cheaper than calibrating the model.
4. **A practical recipe** covers keying and filling chunk tuning caches.

The measurement work is careful and unusually self-critical: recurrence-preserving timing, shuffled rounds, held-out evaluation, a placement/L2 intervention, an autotuner-state control, and explicit limitations. As a domain reviewer, however, I find three problems that limit what the paper establishes:

- **The large chunk effects come from a weak kernel.** The steep curve, the 49% default regret, the TP-width sensitivity and the whole cost model belong to the authors' own Triton kernel. That kernel is under-optimized (1.23x the reference scan, geometric mean), and its default launch table couples the chunk to the channel block and warp count. That table was tuned on a fourth GPU (an RTX 3070).
- **Production relevance is argued, not measured.** The paper uses kernel-level proxies (graph replay of single calls, a synthetic serving replay) rather than a serving engine.
- **The cost-model contribution is thin.** The model is narrow, its closest prior art is missing, and by the authors' own economics it is dominated by profiling.

---

## 2. Scores

| Dimension | Score (0-100) | Comment |
|---|---|---|
| Novelty | 35 | One knob of one kernel family. The "values do not matter" result is expected for data-oblivious kernels and largely implied by TFLA's shape-and-hardware analysis. The cost model is a per-kernel occupancy/issue model in the Hong-Kim/Volkov tradition, and the closest pruning-by-static-analysis prior art is not cited. |
| Methodology | 65 | Timing protocol, held-out evaluation, placement and L2 controls are strong. Weaknesses: the chunk effect is confounded with the launch table (M1); there are no hardware counters (M2); the cost model has a physically questionable latency term (M3); the device-time spin is too short for the multi-kernel pipelines. |
| Experiments | 60 | Extensive and well documented. Production-kernel evidence rests on a non-stock RTX 3090 and an RTX PRO 6000. There is no serving-engine evaluation, no datacenter-GPU graph measurement of SSD/GDN, and no TP shards for the production kernels (M4, M5). |
| Clarity | 40 | Extremely dense: numbers in nearly every clause, five timing conventions, and tables that mix conventions. "Mamba-1" means the authors' Triton kernel rather than the production CUDA kernel. The main text depends on more than 35 supplementary tables (M7). |
| Significance | 40 | The actionable result is "profile per shape at warm-up in the regime you serve". This is already standard practice (Triton autotune keys, cuDNN benchmark mode, TensorRT timing caches, per-device MoE configs in vLLM/SGLang). Measured production gains are 1-6% of scan time, an unknown and likely small share of end-to-end time. |

**Reviewer confidence:** 4 / 5.

---

## 3. Recommendation: **Major Revision**

**Why not Reject.**
- The measurement methodology is sound and transparent.
- The claims are stated honestly within their scope.
- The artifact is extensive.
- Several findings are useful to practitioners: placement and L2 residency can masquerade as chunk or data effects; the Triton specialization bit must be in the key; device-bound execution exposes chunk effects that host-bound timing hides; and the autotuner-state control.
- The problems below can be addressed by reframing plus a bounded set of measurements on existing infrastructure.

**Why not Minor Revision.**
- **Main significance claims are not demonstrated in production.** "Where the chunk matters" (production kernels under device-bound execution) and "when a model pays" are not shown in a production setting.
- **The headline effect sizes are confounded.** The steep chunk curve, the 49% regret and the shard-width sensitivity are partly a property of the authors' launch table and kernel design rather than of chunked scans (M1).
- **The cost model needs stronger positioning or demotion** against prior art it does not cite (M3).

**Path forward.** If a revision (a) shows the effect for production kernels inside a serving engine on a datacenter GPU, (b) separates the chunk from the launch configuration, and (c) re-positions the model as a pruning heuristic against the relevant prior art, I would expect to support acceptance. If the engine-level effect turns out to be at most about 1% end to end, the paper would read better as a shorter characterization note than as a TACO regular paper.

---

## 4. Strengths

- **S1. Recurrence-preserving methodology.** Every chunk is timed inside one launch and verified against a reference scan. The runtime-chunk rebuild of the reference CUDA scan reuses the stock kernel body and is validated against the stock binary.
- **S2. Value versus placement.** The intervention separating buffer placement from values (Supplementary Table S11, warm/cold L2 section) is a useful benchmarking lesson. Placement changes latency by a median 21% (up to 44%) at an L2-sized working set on the RTX PRO 6000, while values change it by 0.1%. Autotuner developers should know this.
- **S3. Statistical care.** Grouped cross-validation by prompt, prompt-cluster bootstrap intervals, split-half and null analyses of oracle headroom, and TOST equivalence tests. The oracle-minimum bias is treated explicitly.
- **S4. Specialization bit.** The divisibility-by-16 finding (one bit changes the compiled kernel, its register count and its optimal chunk, and a length-bucket key without it leaves 22-90 us of headroom) is a concrete, generalizable pitfall for Triton tuning caches.
- **S5. Autotuner-state control.** The pinned versus re-tuned control for the production kernels (Supplementary S9.1) addresses an obvious confounder of chunk sweeps on autotuned Triton pipelines.
- **S6. Honest economics.** Calibration versus profiling (Section 6.6) leads the authors to conclude against their own model as a substitute for profiling.
- **S7. Artifacts and protocol.** A complete artifact, an explicit protocol table (Supplementary Table S1), and a limitations section that is specific rather than generic.

---

## 5. Major issues

### M1. The large chunk effects belong to an under-optimized kernel whose launch table confounds chunk and parallelism

**Location:** Section 4 (launch table, Eq. (3)); Section 5.1(i); Section 6 lead paragraph; Section 6.2; Section 6.4 "Auto-tuning baseline"; Section 6.5 "Tensor-parallel widths"; Table 8; Supplementary Figure S1 (pseudo-code and default launch table) and Table S9.

**Why it matters.** The paper itself says the chunk effects "large enough to motivate tuning" belong to the Triton Mamba-1 scan. Several properties of that kernel inflate or shape its chunk sensitivity in ways that do not transfer to production scans.

1. **Launch-table coupling.** The default table maps C to (D_b, n_w): 32/64/128 -> (8,2), 256/512 -> (4,2), 1024 -> (4,4). Moving from chunk 128 to 256 therefore halves D_b and doubles the grid. At d = 2048 and batch 1 this is 256 versus 512 programs of 64 threads on 82-188 SMs. A "chunk" sweep along the table changes the number of programs and resident warps as much as the tile length.
   - The decomposition in Section 6.4 ("29 points due to the chunk and 3 to the launch configuration") compares the best table entry with the fully tuned kernel. It does not hold (D_b, n_w) fixed while varying C, so it does not separate the two.
   - **My offline check of the artifact's launch-space sweep** (the short device-time search timings of the 184 configurations at the twelve keys) suggests the separation matters:
     - **RTX 3090, fixed (D_b, n_w) = (4, 2):** chunk 128 is fastest at 11 of 12 keys (64 at the twelfth), and a single chunk loses about 0% across the keys. Along the default table the optimum ranges over {32, 64, 128, 512} and the best single chunk loses 17%.
     - **H800, fixed (8, 2):** chunk 128 is fastest at 11 of 12 keys (1.2% single-chunk regret), versus 21% along the default table.
     - **RTX PRO 6000, fixed (4, 2):** the optimum still moves (128/64 at L = 128, 512 from L = 512), so the confound is GPU dependent.

     These are noisy 10-repeat timings and need confirming under the full protocol, but they indicate that much of the shape dependence of the "chunk" on two of three GPUs is the table's coupling of the chunk to the grid.
   - **The table is mistuned off the GPU it was chosen on.** The fully searched configuration beats the best default-table entry by 24-27% at several keys (for example RTX 3090, L = 8192, B = 1: 0.78 versus 1.07 ms; H800, L = 2048, B = 1: 0.17 versus 0.23 ms). This is consistent with the paper's own "7 to 18% behind" (Section 6.4). It means the "chunk regret" is measured conditional on a launch table tuned on an RTX 3070.
2. **Grid under-fill at batch 1.** The kernel parallelizes only over ceil(d/D_b) x B programs; there is no split over the state index or the sequence. At batch 1, which covers 65% of the captured calls, it runs 2-6 warps per SM. This is far below the register limit, so at batch 1 the kernel is grid-bound and latency-bound, not "register-bound" as Section 4 and the abstract say. The chunk then mainly trades the length of each program's serial tile chain. The same mechanism explains the shard-width sensitivity (Section 6.5): n_p scales with d/n_tp. The reference CUDA scan instead launches one block per (batch, channel) and is much less width-sensitive (Supplementary Table S25).
3. **Avoidable per-tile overheads.** The carried state is extracted by a masked full-tile reduction, `h_last = sum(where(t == C-1, hs, 0))`, and h[:, n] by a masked reduction over the state axis, once per state index per tile. That adds log2(l_C) shuffle levels per state. The cost model then devotes a term (tau_s) to these shuffles, so the model partly fits the kernel's own inefficiencies.
4. **Unaligned lengths.** These run about 2x slower at batch 1 (Section 4). A production kernel would run a vectorized main loop with a masked tail, or pad the input.
5. **Arbitrary default.** "Chunk 256, the Mamba-2 default" is used as the untuned reference for a Mamba-1 kernel that has no default. The 49% / 34% losses quoted in the Section 6 lead paragraph and Table 8 (4.4-25 ms per forward pass) compare against an arbitrary choice.

**Concrete fixes.**
- (a) **Fixed-configuration analysis.** Report chunk sensitivity, the optimum's movement and selector regrets at fixed (D_b, n_w, stages). This is available offline from the existing sweep; re-time the relevant slice under the full protocol.
- (b) **Retune the launch table per GPU** before studying the chunk, or define the knob as the full launch configuration.
- (c) **Reposition the kernel.** Either bring it to parity with the reference CUDA scan (for example a slice-based carry instead of the full-tile reduction, more programs per channel block at small B·d, an aligned main loop plus a masked tail), or present it explicitly as a stress case and move its numbers out of the abstract and conclusion.
- (d) **Use a real default.** Use the reference scan's stock length rule as the Mamba-1 default, not SSD's 256.

**Effort:** offline analysis, plus new GPU measurement for (a)'s re-timing and for (c).

### M2. The chunk mechanism is asserted rather than measured; no hardware-counter evidence

**Location:** Abstract ("register-bound Triton scan"); Section 4; Table 1; Section 6.1 "Placement, not values"; Section 6.5 "Production kernels at larger widths"; Section 7; Supplementary Sections S3, S6 and S9.

**Why it matters.** TACO readers expect the paper to explain why the chunk curve has its shape on each kernel. Specific gaps:

- **Which resource binds is not shown.** The paper never shows, per regime, whether the binding resource is grid parallelism, register-limited occupancy, local-memory (spill) traffic, issue slots, or the launch floor.
- **The placement explanation lacks counters.** The authors acknowledge it "rests on an intervention, not on hardware counters".
- **The production pipelines are not decomposed.** For SSD and Gated DeltaNet, the multi-kernel pipelines are summarized by a launch floor and work terms. There is no per-kernel breakdown versus chunk (chunk_cumsum, chunk_state, state_passing, bmm_chunk, chunk_scan; for FLA: cumsum, kkt, solve_tril, recompute_w_u, fwd_h, fwd_o).
- **The cause of short-call losses is not shown.** The main loss mechanism for short calls under graphs is plausibly empty or masked programs when C > L, and the serialization of the state-passing kernel, but this is not demonstrated. The work-term model "has no padding term for calls shorter than the chunk", which is exactly where the defaults lose up to 63% (Figure 2).

**Concrete fixes.**
- **Counters for the Triton scan.** Collect Nsight Compute metrics for the twelve tuning keys at each chunk on at least one GPU: achieved occupancy, warps per scheduler, issue-slot utilization, top stall reasons, local-memory load/store sectors, and L2 hit rate. If cloud containers block counters (ERR_NVGPUCTRPERM), a local GPU of the same generation suffices.
- **Counters for placement.** Measure the L2 hit rate for two placements of the same input to confirm the placement explanation.
- **Per-kernel pipeline breakdown.** Report a per-kernel time breakdown of SSD and GDN versus chunk under graph replay at short and long calls.
- **Padding term.** Add a min(L, C) padding term to the work-term model.

**Effort:** new GPU measurement, plus text.

### M3. The cost model: limited novelty, missing closest prior art, a questionable latency term, and a contribution that the paper's own economics negates

**Location:** Section 2 ("Performance models"); Section 4 (Eq. (3), "Its novelty is narrow"); Section 6.4; Section 6.6 "Calibration versus profiling"; Table 2; Table 5; Supplementary S3.

**Why it matters.**

- **(a) Positioning.** The procedure is: compile every candidate without running it, read the compiler's resource report, predict with an analytical occupancy/issue model, then profile the top-k. This is essentially the program-optimization-space pruning of Ryoo et al. (CGO 2008), which pruned GPU configurations using static efficiency and utilization metrics derived from compiled code and then measured the Pareto subset. It is also close to:
  - the static-analysis autotuning of Lim, Norris and Malony (ICPP 2017);
  - ML-predicted autotuning that profiles only the best-predicted configurations (Bergstra et al. 2012; Falch and Elster 2017);
  - Triton's own perf_model/top_k pruning, which the paper does cite.

  None of the first three is cited. The paper's own "work-term model" for SSD and GDN, BL(a + bC) + B·ceil(L/C)·g plus a launch floor, is essentially TFLA's runtime form with a floor. This should be stated rather than presenting TFLA only as a failing baseline on a scan kernel it was not designed for (see m9).
- **(b) Physical consistency.** In Eq. (3) the latency part enters as A·min(1, W1/W) per tile per wave. When W > W1, the predicted wave time per tile drops below A. A is described as the per-tile latency of the N_s state iterations, whose loop is not unrolled, so it lies on each program's serial chain. A wave cannot finish before its programs' dependency chains, so the term lets the model predict wave makespans below the critical path.
  - The standard formulations take a max of latency-bound and throughput-bound time (Hong and Kim's MWP/CWP; Volkov's latency-bound versus throughput-bound analysis).
  - The fitted W1 is 6.6-10.5 warps, and W exceeds it in batched calls with 4-warp configurations, so the issue can affect batch-4/16 selections.
  - Please justify the form (is A meant to be overlappable, non-dependent latency only?) or refit a max-form and report whether the selected chunks change.
- **(c) Evidence of value.**
  - The a-priori accuracy (3.0-6.2%) is matched or beaten on some GPU by an MLP, a three-parameter restriction or gradient boosting (Table 5).
  - On the RTX 3090 the per-batch static chunk beats the model per forward pass (Table 8).
  - The learning curve is flat or non-monotone on two of three GPUs (Supplementary Table S16).
  - Full calibration costs 130-843 s against 4-7 s to profile all 57 enumerable keys (Section 6.6).
  - With three chunks profiled, a three-parameter ranker matches it.

  So the model's demonstrated contribution is a pruning heuristic for keys that cannot be enumerated. The paper offers no evidence that such keys exist in practice: the serving replays show 11-23 bucket keys per trace, and the authors recommend enumerating.
- **(d) Calibration data.** 2208 of the 2298 calibration timings are short-search timings of non-default launch configurations, while selection uses only the six default-table configurations. The 72 static re-measurements alone already reach 0.56/1.8/6.1%. It is unclear why the model must be calibrated on the full launch space.
- **(e) Transfer.** "Whether [the procedure] transfers is untested." For a TACO contribution on performance modelling, at least one transfer test is needed: another Triton kernel whose register and spill footprint varies with the chunk, for example FLA's chunk_fwd_o or the SSD chunk_scan kernel with the inner tile fixed.

**Concrete fixes.**
- Cite and compare against the static-pruning and predictive-autotuning lines above.
- Reframe the model explicitly as a top-k pruner and compare it against Triton's perf_model/top_k mechanism with an off-the-shelf occupancy-only ranker.
- Fix or justify the latency term.
- Report calibration on the six default-table configurations only.
- Either add one transfer test or move the model to a shorter subsection and demote it from the contributions list.

**Effort:** text plus offline analysis (re-fits); the transfer test needs a new GPU measurement.

### M4. Relevance to production serving is not established

**Location:** Abstract; Section 1 (second question); Section 5.2; Section 6.5 ("CUDA-graph replay", "Packed variable-length calls"); Section 6.6 ("Deployment units", "Serving traces"); recipe box; Supplementary S9, S10 and S14.

**Why it matters.** The paper's second contribution ("where the chunk matters") is framed around production engines, but every production-facing number comes from proxies.

1. **The host-bound regime is an artifact of the Python wrapper.** For SSD it is measured through the mamba_ssm Python wrapper, with a batch-1 floor of about 0.46 ms (RTX PRO 6000) and 0.74-0.8 ms (RTX 3090). That floor is a property of the wrapper, not of the kernels: graph replay brings it to 33-57 us. "Library defaults are adequate when kernel launches dominate" is therefore true only for a host overhead that serving engines work hard to avoid.
2. **The device-bound short-call regime may not exist inside engines.** It is where SSD and GDN defaults lose 10-24% per call (Table 7, Figure 2), and it is measured by graph-replaying single calls in isolation. In vLLM V1 and SGLang, prefill and mixed batches of hybrid models typically run the Mamba/GDN mixer outside captured graphs: it is a splitting op of the piecewise graphs, and full graphs are used mainly for uniform decode batches. Short prefill chunks are also usually packed with decode tokens. Whether short, device-bound, chunk-sensitive mixer calls occur at all in these engines, and how often, is the crux of the claim and is not measured.
3. **Switching the chunk per call has engine costs.** It interacts with engine metadata: chunk-aligned varlen metadata, the alignment of prefix-cache or state-checkpoint blocks to the chunk size, and graph capture per chunk. The recipe mentions this but its cost is not measured.
4. **The serving replay is synthetic.** It costs packed calls as batch-1 calls of the authors' Triton Mamba-1 scan, which supports neither varlen nor the production Mamba-1 path (vLLM/SGLang use the CUDA selective scan with varlen support). Its step model is 2.3x faster than measured eager prefill. The "share of prefill" numbers are therefore illustrative upper bounds and should not carry claims.
5. **TP is measured only for the Mamba-1 kernels**, in isolation. The production kernels are not measured at the shard layouts that matter: SSD with n_heads/n_tp and ngroups > 1, as in Nemotron-H or Falcon-H1; GDN at Qwen3-Next-like head counts divided by n_tp.
6. **No end-to-end share.** Only percentages of scan time and ms per forward pass are reported, so the effect on TTFT or throughput is unknown. The one end-to-end experiment (routed study: HF Transformers eager, reference scan, batch 1, RTX 3090) shows the stock dispatch within 0.7% of the best static chunk in prefill at 4096 tokens.

**Concrete fix.** Integrate a shape-keyed chunk table into one serving engine (vLLM or SGLang). Use at least one Mamba-2 hybrid (for example Nemotron-H 8B, Falcon-H1, or a Granite-4.0-H model) and one GDN hybrid (a small Qwen3-Next-style or the paper's GDN 1.4B checkpoint if the engine supports it). Then:
- run the engine's default CUDA-graph configuration with a realistic chunked-prefill token budget on public traces;
- report TTFT and throughput with the default versus the tuned chunk, the share of mixer calls that are device-bound and below 4096 tokens, and the overhead of per-call chunk switching (metadata rebuild, graph count);
- include one TP configuration (TP = 2 suffices).

If an engine integration is out of reach, at minimum time whole-layer (projection, conv, mixer, gating) device-bound calls rather than isolated kernels, and report the end-to-end share.

**Effort:** new GPU measurement.

### M5. Platform validity of the production-kernel evidence

**Location:** Section 5.2; Section 6.5; Supplementary Tables S1 and S2 and the "Power cap" paragraph.

**Why it matters.**
- **Where the evidence comes from.** All CUDA-graph, larger-width, reference-scan, routed, TFLA and output-divergence results for the production kernels come from the RTX 3090 and the RTX PRO 6000; packed calls come from the RTX PRO 6000 only.
- **The RTX 3090 is not a reliable measurement platform.** It is a non-stock 48 GB board in a cloud container where sharing below the container "cannot be excluded". The software power cap was active in 38-92% of readings for several sweeps, clocks are unlocked, and the board power limit was not recorded.
- **No datacenter-GPU measurement of the production kernels.** The only datacenter GPU (H800) covers the authors' Triton kernel only, without graphs, placement or L2 controls. Production prefill for hybrid models runs on Hopper/Blackwell datacenter parts, where the Triton SSD and FLA kernels may select different inner configurations (and, in recent library versions, different code paths).

**Concrete fix.** Replicate, on a stock datacenter GPU (H100/H800/H200 or B200; an A100 or L40S would also help):
- Table 7 (graph replay of SSD 370M/2.7B and GDN 1.4B);
- the packed-call study (Supplementary S10);
- the autotuner-state control.

Record board power limits and lock application clocks where the platform permits; otherwise report clock distributions per cell as already done.

**Effort:** new GPU measurement.

### M6. The emphasis on refuting data-dependent dispatch crowds out the kernel content

**Location:** Section 1 (first question and first contribution); Section 3 (Lemma 3.1); Sections 6.1, 6.3 and 6.6 (break-even, key churn); Supplementary S12-S15.

**Why it matters.**
- **The result is expected.** For kernels that contain "no data-dependent branches, early exits or sparse paths" (Section 3), value invariance of the optimal configuration is the a-priori expectation of a TACO reader. TFLA already derives the optimal chunk from shape and hardware.
- **The control is weak.** It is a single preprint's scheduler [Ma et al. 2026] whose mapping saturates and selects chunk 256 for essentially every call (Supplementary Figure S6). Refuting it does not require a lemma, a mutual-information analysis, a routed study and a key-churn model.
- **The space is needed elsewhere.** These parts occupy roughly a third of the main text, space that M1, M2 and M4 need.
- **Section 5's pitfall paragraph** ("Timing ... by cutting the sequence into pieces ... is not a chunked scan") addresses an error the kernel community does not make. It reads as a correction of the cited preprint.

**Concrete fix.**
- Condense the data-dependence material into one subsection. Keep Lemma 3.1 as a remark, keep the placement/L2 finding (S2), and keep one predictor table. Move the routed study, break-even and key churn to the supplement.
- If the cited preprint is the authors' own, the relationship should be disclosed to the editor per ACM policy (outside the anonymized text).

**Effort:** text.

### M7. Clarity and presentation

**Location:** throughout; especially the Abstract, the Section 6 lead paragraph, Sections 6.4 and 6.6, and Tables 2, 4, 5 and 7.

**Why it matters.** The paper is hard to read even for a specialist:

- **Abstract.** One sentence carries the lemma, the headroom, the synchronization cost and the restriction to "scans with a steep chunk curve".
- **Timing conventions.** Device time, launch-inclusive, graph replay, warm/cold L2, single block versus round medians, and first versus second session: these alternate within paragraphs and within tables. Table 4 mixes device-time Mamba-1 rows with launch-inclusive Mamba-2 rows. Section 6.6 quotes launch-inclusive and device-time headroom in one sentence.
- **Naming.** "Mamba-1" denotes the authors' Triton kernel "unless stated otherwise", while the production Mamba-1 kernel is the CUDA scan. Readers will misattribute results.
- **Figures.** There are only two figures. The kernel-mechanism story (M1/M2) needs at least one figure decomposing latency versus chunk into the model's terms, or into profiler metrics, per regime.
- **Supplement dependence.** Many main-text claims are only verifiable in more than 35 supplementary tables.

**Concrete fix.**
- Rewrite the abstract to three or four plain claims with one number each.
- Fix one primary convention per table and state it in the column header.
- Rename the kernels ("Triton scan (ours)", "CUDA scan (mamba_ssm)").
- Add a claims-to-evidence table.
- Replace a dense paragraph with a mechanism figure.
- Move secondary numbers to the supplement.

**Effort:** text.

### M8. "What the key must contain" is derived from one kernel and stated too narrowly for production layouts

**Location:** Section 3 (shape-key definition); Section 4 "Compiler specialization and parallel shapes"; recipe box; Table 1; Supplementary S12 "Key completeness".

**Why it matters.**
- **What Triton specializes on.** It specializes on the divisibility (and, depending on version, unit value) of integer arguments, including strides, and on pointer alignment. It does not specialize on "the length".
  - The paper's kernel sees stride = L because the captured inputs are made contiguous ([B, d, L]).
  - In production layouts the relevant stride is B·L, or the packed token count for varlen. Mamba's x comes out of xz.chunk(2, dim=1) of a [B, 2d, L] projection, with a pointer offset that may break 16-byte alignment.
  - So the recipe's "length divisible by 16" is not the right bit in general.
- **Which specialization flags are active.** Please verify which attributes Triton 3.4 actually applies to this kernel, since the default set of integer specializations has changed across Triton 3.x releases.
- **The production kernels are untested.** Divisibility effects are shown only for the authors' Triton scan. The SSD and FLA Triton kernels were not checked for alignment-dependent compiled variants and optima.
- **The CUDA scan has a chunk-coupled specialization.** It instantiates kIsEvenLen = (L mod (kNThreads·kNItems) == 0). In the runtime-chunk sweep this bit therefore changes with the chunk (for C > L, or L not divisible by C). It is omitted from Table 1 and from the interpretation of the 5.4x spread.

**Concrete fix.**
- Define the key as the kernel's actual specialization signature (Triton exposes it through its compilation cache key and specialization attributes) rather than "L mod 16".
- Report the aligned/unaligned effect for the SSD and GDN kernels on captured or packed calls.
- Discuss kIsEvenLen in Table 1 and in Section 6.5.

**Effort:** offline analysis plus a small new GPU measurement.

---

## 6. Minor issues

- **m1. Reference scan dispatch** (Section 6.5): "selects its chunk ... from the length at compile time". The stock scan selects among compile-time template instantiations at run time (host-side dispatch on seqlen). Table 1's caption is closer to correct; align the wording.
- **m2. Abstract wording.** "as under CUDA graphs on two workstation GPUs": the RTX 3090 is a consumer GeForce card. Say "on an RTX 3090 and an RTX PRO 6000".
- **m3. Device-time spin.** The 20 ms GPU-side spin does not cover the enqueueing of 50 repeats of the multi-kernel pipelines (about 23-40 ms at their eager floor), so their device-time data are discarded (Supplementary S8). A longer spin, or one sized to the measured enqueue time, would have recovered a clean device-time convention for SSD and GDN. Please fix it or state why it was not done.
- **m4. Spread statistics.** Chunk-spread statistics ("8.5x", "15.2x at p95") are dominated by heavily spilling configurations (for example D_b = 16 with 2 warps). The regret of the default or best static choice is the meaningful quantity; demote the spreads.
- **m5. Spill proxy.** omega is the static ptxas spill byte count used as per-tile traffic inside the state loop. Please validate it against measured local-memory traffic for at least the 255-register configurations (C = 512, 1024).
- **m6. Occupancy bookkeeping.** m = min(r, ceil(n_p/n_SM)) assumes an even distribution across SMs. The last partial wave and its residency are not modelled, which matters at batch 1, where w = 1 and n_p/n_SM is 1.4-2.7.
- **m7. GDN chunk limit.** Gated DeltaNet is measured only at C ≤ 64 because FLA's solve_tril path caps it. Say explicitly that the "fixed optimum at 64" may be a library ceiling: the work-term model places the optimum at 61-85. The paper partially says this; make it prominent in Section 6.5 and the conclusion.
- **m8. Packed-call design.** The carried state is applied to the first sequence for SSD but to every sequence for GDN (Supplementary S10). Use the same design, or justify the difference.
- **m9. TFLA baseline.** TFLA's runtime expression is applied to the scan kernel (Table 5), a kernel class it does not model; the outcome ("rejects the quadratic term") is foregone. Move it to the supplement and instead compare TFLA's expression with the work-term model on SSD and GDN, where it applies.
- **m10. Cost-model tuning loop.** The leave-one-GPU-out term selection (Section 6.4) uses only two "other" GPUs per held-out GPU, with a 0.1-point parsimony rule. Report the candidate forms and their scores per fold so the "a priori" accuracy can be judged.
- **m11. Output divergence.** "Switching chunks is not output-neutral" (Section 7) is important for engines with batch-invariant or deterministic modes. Give the main-text number together with its run-to-run baseline: the default run twice was bitwise identical, which is good.
- **m12. Table 2.** The asterisk and parentheses convention ("0.4*/3.5*/3.0 (4.2/6.2/3.0)") is hard to parse. Split it into two labelled columns.
- **m13. Figure 2.** The host-bound (RTX 3090) series is visible only in the Triton panel. If it lies at zero elsewhere, say so in the caption.
- **m14. Workload weighting.** Per-call means over truncated prompts are not workload-weighted (acknowledged). Report latency-weighted regret wherever per-call regret appears in the abstract or conclusion.
- **m15. Thread pairing.** Section 6.5's stock-pairing analysis shows that a chunk at fixed 32 threads behaves very differently (34x spread). It would help to state for the reference kernel which part of the "chunk effect" is thread count, items per thread and kIsEvenLen.
- **m16. Recipe evidence.** "Persist the table and invalidate it on a change of GPU model, power limit, driver, compiler or kernel library" is reasonable but only the GPU-model part is evaluated. Mark the others as untested.
- **m17. Term "compile facts".** It is idiosyncratic. "Compiler resource report" or "static resource usage" is the established terminology.

---

## 7. Missing references

Each entry lists what should be cited and why it matters.

**Static/predictive pruning of GPU autotuning (closest prior art for the cost model and top-k hybrid):**
- **Ryoo et al. (CGO 2008).** Prunes GPU configuration spaces using static metrics derived from compiled code, then measures the Pareto set. This is the closest precedent for "compile, read resource report, rank, profile top-k". Shane Ryoo, Christopher I. Rodrigues, Sam S. Stone, Sara S. Baghsorkhi, Sain-Zee Ueng, John A. Stratton, and Wen-mei W. Hwu. 2008. Program Optimization Space Pruning for a Multithreaded GPU. In *Proc. 6th Annual IEEE/ACM International Symposium on Code Generation and Optimization (CGO)*. 195-204. https://doi.org/10.1145/1356058.1356084
- **Lim, Norris and Malony (ICPP 2017).** Robert V. Lim, Boyana Norris, and Allen D. Malony. 2017. Autotuning GPU Kernels via Static and Predictive Analysis. In *Proc. 46th International Conference on Parallel Processing (ICPP)*. https://doi.org/10.1109/ICPP.2017.61
- **Bergstra, Pinto and Cox (InPar 2012).** James Bergstra, Nicolas Pinto, and David Cox. 2012. Machine Learning for Predictive Auto-Tuning with Boosted Regression Trees. In *Innovative Parallel Computing (InPar)*.
- **Falch and Elster (CCPE 2017).** Predicts kernel time from configuration and profiles only the best-predicted candidates. Thomas L. Falch and Anne C. Elster. 2017. Machine Learning-Based Auto-Tuning for Enhanced Performance Portability of OpenCL Applications. *Concurrency and Computation: Practice and Experience* 29, 8 (2017), e4029.
- **Balaprakash et al. (Proc. IEEE 2018).** Survey that should anchor the autotuning positioning. Prasanna Balaprakash, Jack Dongarra, Todd Gamblin, Mary Hall, Jeffrey K. Hollingsworth, Boyana Norris, and Richard Vuduc. 2018. Autotuning in High-Performance Computing Applications. *Proc. IEEE* 106, 11 (2018), 2068-2083.

**Occupancy and latency modelling (for M3(b)):**
- **Volkov (GTC 2010).** The classic statement of the register/occupancy versus ILP trade-off this kernel exhibits. Vasily Volkov. 2010. Better Performance at Lower Occupancy. GPU Technology Conference (GTC).
- **Hong and Kim (ISCA 2010).** Companion to the cited 2009 model. Sunpyo Hong and Hyesoon Kim. 2010. An Integrated GPU Power and Performance Model. In *Proc. 37th Annual International Symposium on Computer Architecture (ISCA)*.
- **Kaufman et al. (MLSys 2021).** Learned per-kernel performance model used to rank configurations in a production compiler. Samuel J. Kaufman, Phitchaya Mangpo Phothilimthana, Yanqi Zhou, Charith Mendis, Sudip Roy, Amit Sabne, and Mike Burrows. 2021. A Learned Performance Model for Tensor Processing Units. In *Proc. Machine Learning and Systems (MLSys)*.

**Shape-keyed tuning and specialization in production ML stacks (the recipe's closest practice):**
- **PyTorch 2 / TorchInductor (ASPLOS 2024).** Dynamic-shape handling specializes on 0/1 and divisibility, autotunes Triton kernels per shape, and uses CUDA graphs per shape. Jason Ansel et al. 2024. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation. In *Proc. 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), Volume 2*.
- **FlashDecoding++ (MLSys 2024).** Offline shape-keyed (token-count) selection among GEMM implementations for LLM inference, a direct precedent for shape-keyed kernel-variant tables in serving. Ke Hong, Guohao Dai, Jiaming Xu, Qiuli Mao, Xiuhong Li, Jun Liu, Kangdi Chen, Yuhan Dong, and Yu Wang. 2024. FlashDecoding++: Faster Large Language Model Inference on GPUs. In *Proc. Machine Learning and Systems (MLSys)*.
- **MIOpen (arXiv 2019).** Find-db/perf-db, a persistent tuning cache keyed on problem descriptors. Jehandad Khan et al. 2019. MIOpen: An Open Source Library For Deep Learning Primitives. arXiv:1910.00078.
- **Software artifacts to mention:**
  - cuDNN's benchmark mode (as exposed by `torch.backends.cudnn.benchmark`, a per-shape algorithm cache);
  - TensorRT's persistent timing cache;
  - the per-device, per-shape tuned Triton configuration files for fused MoE kernels in vLLM and SGLang.

  Together these show the practice the recipe recommends already exists, and the paper should position against it.

**Serving engines:**
- **SGLang (NeurIPS 2024).** Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. 2024. SGLang: Efficient Execution of Structured Language Model Programs. In *Advances in Neural Information Processing Systems (NeurIPS)*.

**Parallel scans for linear recurrences (background for the kernel design discussion in M1):**
- **Blelloch (1990).** Guy E. Blelloch. 1990. Prefix Sums and Their Applications. Technical Report CMU-CS-90-190, Carnegie Mellon University.
- **Martin and Cundy (ICLR 2018).** Eric Martin and Chris Cundy. 2018. Parallelizing Linear Recurrent Neural Nets Over Sequence Length. In *International Conference on Learning Representations (ICLR)*.
- **Merrill and Garland (2016).** Chunked scan with carried inter-tile state is a decoupled-look-back pattern. Duane Merrill and Michael Garland. 2016. Single-pass Parallel Prefix Scan with Decoupled Look-back. NVIDIA Technical Report NVR-2016-002.
- **Smith, Warrington and Linderman (ICLR 2023).** Jimmy T. H. Smith, Andrew Warrington, and Scott W. Linderman. 2023. Simplified State Space Layers for Sequence Modeling. In *International Conference on Learning Representations (ICLR)*.

**Measurement variability (supports the per-device tuning and power-cap discussion):**
- **Sinha et al. (SC 2022).** Prasoon Sinha, Akhil Guliani, Rutwik Jain, Brandon Tran, Matthew D. Sinclair, and Shivaram Venkataraman. 2022. Not All GPUs Are Created Equal: Characterizing Variability in Large-Scale, Accelerator-Rich Systems. In *Proc. International Conference for High Performance Computing, Networking, Storage and Analysis (SC)*.

**Mischaracterization to correct.** TFLA (Beck et al. 2025) is described as "the closest analysis to ours", yet it is used only as a failing baseline on the scan kernel. Its runtime form is essentially the paper's own work-term model for the chunkwise-parallel kernels (see M3(a), m9). This should be stated.

---

## 8. Questions to the authors

1. With (D_b, n_w, stages) fixed, how much of the Triton scan's chunk regret and of its optimum's movement with L, B and d/n_tp remains on each GPU (M1)?
2. In vLLM or SGLang with default CUDA-graph settings, what fraction of Mamba-2/GDN mixer calls during chunked prefill are device-bound and below 4096 tokens? What end-to-end TTFT or throughput change does a shape-keyed chunk table produce (M4)?
3. Is the term A·min(1, W1/W) intended to model overlappable latency only? How do the selections change with a max(latency, throughput) formulation (M3(b))?
4. Which Triton specialization attributes are active for your kernel in Triton 3.4? Do the SSD and FLA kernels show alignment-dependent optima on packed calls (M8)?
5. What is the relationship between this submission and [Ma et al. 2026]? (This may be answered to the editor if anonymity requires.)
