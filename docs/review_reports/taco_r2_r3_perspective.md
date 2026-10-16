# Review Report: Reviewer 3 (Practitioner Perspective)

**Manuscript:** "Shape Sets the Chunk: Selecting and Keying the Chunk of Chunked Linear-Recurrence GPU Kernels"
**Venue:** ACM Transactions on Architecture and Code Optimization (TACO), new regular paper (double-anonymous, acmsmall)
**Material reviewed:** main article (20 pages plus references), Supplementary Material (31 pages, Tables S1 to S39, Figs. S1 to S6), and spot checks of the released result files
**Reviewer profile:** engineer who builds LLM inference and serving systems and kernel libraries, and would act on the paper's advice

---

## 1. Summary

The paper asks how a runtime should pick the chunk size of chunked linear-recurrence GPU kernels: the Mamba-1 selective scan (an in-house Triton kernel plus the reference CUDA scan rebuilt with a runtime chunk), Mamba-2 SSD and FLA's Gated DeltaNet. It studies three options: choosing from the call's shape, choosing from the activation values, or predicting the chunk with a model of the compiled kernel. Every measurement changes the chunk inside one launch that keeps the recurrence intact. The calls are captured from 370M to 2.8B checkpoints and timed on an RTX PRO 6000 and an RTX 3090, with Mamba-1 also on an H800. The main findings are:

1. The optimal chunk is a function of the shape key, provided the key includes the Triton specialization bits. Activation statistics add nothing measurable. An elementary lemma bounds what any data-dependent policy can gain by the residual regret of a shape-keyed table, and that residual is below one host synchronization.
2. For the production kernels, library defaults are adequate when execution is host-bound. Under CUDA-graph replay the defaults lose 10 to 24% per call but only 1.0 to 5.9% of scan time (0.10 to 1.44 ms per forward pass). The loss is concentrated in short calls.
3. A cost model built from compile facts (registers, spills, tile layout) of the in-house Triton scan selects the chunk without timing the target calls, with 3.0 to 6.2% regret when its terms are chosen a priori. Profiling every enumerable key is still far cheaper than calibrating the model.
4. The paper closes with a recipe for keying and filling chunk tuning caches, plus economics for key churn and calibration versus profiling, using replays of public serving traces.

The measurement methodology is careful. The negative result about data-dependent dispatch is convincing, and the recipe is close to something I would adopt. However, I could not find the effect size in the units that would change an engineering decision: share of prefill or TTFT in a realistic engine, on datacenter GPUs. The CUDA-graph regime is measured in a way that does not match how engines capture prefill. The recipe has correctness gaps around batch invariance and kernel indexing limits. The cost model's practical niche is not shown. The prose is too dense for a TACO reader to extract the advice without the 31-page supplement.

---

## 2. Scores

| Criterion | Score (0 to 100) |
|---|---|
| Novelty | 52 |
| Methodology | 72 |
| Experiments | 63 |
| Clarity | 38 |
| Significance | 45 |

**Confidence:** 4 / 5. I work on serving runtimes and kernel tuning and have read the full supplement. I spot-checked several main-text numbers against the result files: Table 7's ms per forward pass for SSD 370M, SSD 2.7B and GDN 1.4B; Table 8's Triton defaults; and the routed-study prefill and decode times. All of them reproduced.

---

## 3. Recommendation: **Major Revision**

**Why not Reject.** The core measurements look sound. The timing keeps the recurrence intact, inputs are held out, prompts are cross-validated in groups, buffer placement and L2 state are controlled, and tie sets and bootstrap intervals are reported. The paper also flags the unusual weaknesses of its platforms. The negative result, the warning about specialization bits, the separation of placement effects from value effects, and the warm-up recipe are useful to kernel-library and serving engineers, and no comparable characterization exists for this kernel class.

**Why not Minor Revision.** The paper's positive practical claim is that production defaults leave 1.0 to 5.9% of scan time under CUDA graphs. That claim currently rests on three things:

- isolated per-shape graph replays that engines do not perform;
- two workstation GPUs, one of which may be virtualized;
- a unit (percent of scan time, averaged over a call mix that is not weighted by any workload) that does not say what an engine gains.

The recipe also gives advice that is wrong for batch-invariant serving, and it omits a crash hazard the authors measured themselves.

Fixing these issues needs restructuring and offline re-analysis of data the authors already hold. It also needs two targeted GPU measurements:

- graph replay of the production kernels in the form engines actually capture, on a datacenter GPU;
- the share of the forward pass at a few representative shapes.

That is more than a minor revision, but the path is clear and bounded.

---

## 4. Strengths

1. **Recurrence-preserving timing (Section 5).** The point that timing "at chunk C" by splitting the sequence measures launch count rather than tiling is correct. It is a useful warning for anyone benchmarking these kernels. Checking every chunk against a reference scan is good practice.
2. **Placement versus values (Section 6.1, S1.3, Table S10).** Buffer placement moved latency by up to 43.6% when the working set approached the 128 MB L2, while values moved it by 0.1%. This is the most transferable finding for autotuner designers, because a cache that times a key once on one allocation records placement as a chunk effect. The intervention (fixed versus re-placed buffers, warm versus flushed L2) is well designed.
3. **Specialization bits belong in the key (Sections 3, 4, S12).** Triton's divisibility-by-16 and alignment specializations change the compiled kernel and its optimum. This applies to every Triton tuning cache, not only to scans.
4. **Honest reporting.** Results are given per call and latency-weighted, in µs and ms. Results that do not transfer are reported too: the work-term model under graphs, the RTX 3090 shard-width retune, and the second launch-inclusive session. The limitations section is specific.
5. **Practical side studies.** The supplement contains several experiments that serving engineers rarely see:
   - packed variable-length steps with carried states (S10);
   - the autotuner-state control for the production kernels (S9.1);
   - output divergence from chunk switching (S11);
   - key churn on real traces (S13).
6. **Clear negative result for data-dependent dispatch (Sections 6.3, 6.6).** The routed end-to-end study shows that the cost of computing the statistic, not the chosen chunk, dominates.
7. **Artifact.** The code, result files and table-generating scripts are provided. The few numbers I recomputed from the result files matched the paper.

---

## 5. Major Issues

### M1. The deployed effect size is never stated in the unit an engineer acts on, and what is stated suggests a small effect

**Location.** Abstract; Section 6 opening; Section 6.5 (Table 7, Fig. 2); Section 6.6; Section 6.7 "Deployment units" and "Serving traces" (Table 8); Supplementary S13, S15 and Table S39.

**Why it matters.** A serving engineer decides from "what fraction of prefill (TTFT) or throughput do I recover, on my engine and GPU". The paper reports percent of scan time, ms of scan time per forward pass, and per-call relative regret. All of these are averaged over a call mix that is deliberately not weighted by any workload: prompts are truncated in rotation (Section 5.2). Four specific problems:

- **Graph-mode SSD and GDN results.** For SSD 2.7B on the RTX 3090, the default costs 1.44 ms of 65.4 ms scan time per forward pass (Table 7). I recomputed this from the graph-replay result files. In those files, 99% of the default's absolute regret (92% for SSD 370M, 100% for GDN 1.4B) comes from calls under 4096 tokens. The non-scan work (in/out projections, convolution, norms, attention layers of the hybrid) is never reported.
  - My back-of-envelope estimate counts projection GEMMs only, at 50 to 70 TFLOPS on the RTX 3090, for the mean call of about 5.2k tokens. It puts the default's cost at roughly 0.3% (2.7B) to 0.7% (370M) of an average forward pass. This is an estimate, not a measurement, and the authors should replace it with a measurement.
- **Routed end-to-end study (Section 6.6).** It uses an eager Transformers testbed. In the result files, prefill takes about 47 ms at 256 tokens whatever the chunk, and 32 greedy decode steps take about 1.26 s, roughly 40 ms per token for a 370M model.
  - The comparison the text leads with, "chunk 128 adds 16.8 ms", is a deliberately bad static chunk against the best one.
  - The comparison that matters is stock dispatch against the best chunk. In Table S39 it is +0.7 ± 1.1% in prefill at 4096 tokens, so there is nothing measurable to gain end to end at batch 1 for the deployed Mamba-1 kernel.
- **Serving replay (Section 6.7).** The "2.1 to 2.6% / 5.5 to 10% of prefill" figures come from three choices: the in-house Triton scan, an arbitrary default of 256, and a synthetic step model (2 ms + 10 µs per token).
  - The paper calls them upper bounds because the step model is 2.3 times faster than eager prefill. Against an optimized engine, whose prefill may be at least as fast as the step model, the scan's share is larger, not smaller. The direction of the bound therefore depends on the baseline.
  - The replay is not run for the production kernels at all.
- **Abstract framing.** "Leave 1.0 to 5.9% of scan time on the table" will be read as a deployment gain. The paper should say what that is per forward pass and per request.

**Concrete fix.**

1. Add a main-text "deployment effect" table for the kernels people actually run: the reference Mamba-1 CUDA scan, SSD 370M and 2.7B, and GDN 1.4B. Per regime, give the oracle scan time, the default and table regret in ms, and the measured share of a full forward pass (or of a graph-captured full model) at three or four representative tokens-per-call points.
2. Reweight the existing captured-call timings with the Azure 2023/2024 and BurstGPT length distributions already used in S13. The per-call and latency-weighted numbers then become workload-weighted.
3. Re-run the serving replay with the SSD and GDN graph-replay and packed-step timings in place of the Triton scan.
4. In Section 6.6, lead with stock-versus-tuned and state that the testbed is eager Transformers.

**Effort.** Items 2 to 4: offline analysis. Item 1: new GPU measurement (modest: a few shapes per model on the existing GPUs).

### M2. The device-bound regime, where the only positive production-kernel result lives, does not match how engines run prefill

**Location.** Section 1 ("production engines run prefill both host-bound and as captured CUDA graphs", no citation in the main text); Section 5.3 Conventions; Section 6.5 (Table 7, Fig. 2, packed calls); Supplementary S9 and S10; Recipe box.

**Why it matters.**

- **One graph per exact call.** In S9 each (call, chunk) pair is captured into its own graph at its exact shape, with equal-length batched sequences. Engines that graph-capture prefill do it differently. They capture a fixed set of padded token counts and replay them with packed variable-length metadata (sequence boundaries, carried-state flags) supplied at run time. The chunk is frozen into the captured graph.
  - Under that scheme the only available key is the capture size. The keys the recipe recommends (length bucket, batch, carried state) cannot be used, and the recoverable gain may differ substantially from Table 7.
- **Whether prefill scans are captured at all.** To my knowledge, in at least one widely used engine (vLLM V1) the Mamba and linear-attention mixer ops are split out of piecewise CUDA graphs and run eagerly, and full graphs are limited to uniform decode batches. If so, prefill chunked scans in that engine never run under CUDA graphs. The authors should check this against the engine versions and modes they cite and state precisely which engines and modes put prefill scans in device-bound execution, and at which token counts. Graph capture, C++ launch paths, or host run-ahead under async scheduling are the candidate mechanisms.
- **The host-bound regime is an artifact of the measurement setup.** It is defined by Python-level launch overhead: 56 to 90 µs per Triton call, and an SSD launch floor of about 0.5 ms (RTX PRO 6000) and about 1.5 ms (RTX 3090, per the result files) for the multi-kernel pipeline. C++ runtimes and async schedulers largely hide this. "Library defaults are adequate in host-bound execution" is therefore a statement about eager Python benchmarking more than about a deployment regime, yet it is the first practical claim in the abstract.

**Concrete fix.**

1. Cite and describe, per engine and mode, which prefill scan calls run device-bound. Qualify or drop the general claim.
2. Re-measure the graph regime as engines would run it:
   - packed variable-length prefill steps captured at padded capture sizes (for example 16 to 512 tokens, and up to the largest prefill budget an engine would capture), replayed with varying composition;
   - report the regret of a table keyed on capture size alone.
3. Rename "host-bound" to something like "eager, launch-bound (Python launch path)" and report the harness's per-call launch overhead next to it.

**Effort.** Item 1: text. Item 2: new GPU measurement (one session per GPU; the packed-step generator of S10 and the graph harness of S9 already exist). Item 3: text.

### M3. Platform coverage for the production-kernel claims is thin and partly unreliable

**Location.** Abstract ("under CUDA graphs on two workstation GPUs"); Sections 5.2, 6.5, 6.8, 8; Supplementary S1.1, Table S1, Table S2, S5.

**Why it matters.**

- **Which GPUs carry which results.** The SSD and GDN device-bound results, the routed study, the reference-scan sweep, the larger checkpoints, the tensor-parallel widths and the TFLA sweep rest on two GPUs:
  - an RTX 3090 that is a non-stock 48 GB board on a cloud instance where "virtualization or time-slicing below the container cannot be excluded" (the result files label this host `vgpu3090`);
  - an RTX PRO 6000, a workstation Blackwell (sm_120).

  The only datacenter GPU, the H800 PCIe, covers the in-house Triton scan only, with no graphs and no placement or L2 controls.
- **Measurement conditions.** Clocks are not locked and the power limit is not recorded. The software power cap is active in up to 92% of RTX 3090 samples in some sweeps (S1.1). On the RTX 3090, "rounds differ by more than the chunk effect": a table retuned at shard width loses 14.9% when scored on another round (Section 6.5).
- **Transfer to datacenter GPUs.** Serving fleets run H100/H200/B200 (or MI300) parts. Triton code generation on sm_120 differs from sm_90 and sm_100 (different tensor-core instruction paths), so the matrix-product-heavy SSD and GDN pipelines may have different short-call optima there.

**Concrete fix.**

1. Repeat Table 7 (graph replay of SSD 370M and 2.7B and GDN 1.4B) and the packed-step experiment (Table S35) on a datacenter GPU. The authors already have access to an H800.
2. State explicitly whether the RTX 3090 is a virtual GPU. If it is, say in the main text which conclusions rest on it alone, and corroborate the key ones on the stock 24 GB RTX 3090 already used for S9.1 and S11.
3. Report the cross-session stability of a tuned table: tune in session A, serve in session B, on each GPU. A cache filled at warm-up is always served in a later "session".

**Effort.** Items 1 and 3: new GPU measurement. Item 2: text.

### M4. The recipe is actionable but has correctness gaps and unhedged items

**Location.** Recipe box (p. 19); Section 7; Supplementary S1.3, S3/Fig. S3, S6, S7, S11/Table S36.

**Why it matters and fix.** Engineers will copy this box. Each item should be correct and point to its evidence.

1. **Batch invariance is mishandled.** The box says "a batch-invariant or deterministic mode should keep one chunk per key". That is not sufficient.
   - The key contains the batch size and the packed length. The same request therefore gets a different chunk, and a different summation order, depending on what it is batched with. Table S36 shows that this flips 2 to 9% of greedy outputs in half precision.
   - Batch-invariant serving must either fix the chunk or key it only on per-request quantities that batching cannot change.
   - Independently tuned replicas with noisy timings will also give replica-dependent outputs. The table should be frozen, versioned and shared across replicas.
2. **Crash hazard missing from "Failure modes".** S6 and S7 report that small chunks overflow the 32-bit indexing of the chunk-state tensor and fault with an illegal memory access: SSD chunk 32 at width 5120, batch 16, L > 6528; TFLA chunks 32 and 64 at the largest shapes. An illegal address poisons the CUDA context, so a warm-up profiler that tries every candidate chunk can take a server down.
   - Add a rule for the candidate set: exclude chunks whose state tensor would exceed 2^31 elements at the largest batch and length the key admits. Note that this tensor grows as L/C.
3. **"Synthetic inputs suffice, as latency is value-invariant" needs a qualifier.** S1.3 shows constant and very sparse inputs are systematically faster, by up to 6.7% on the Triton scan and 3.4% on the reference scan. Warm-up code commonly uses zeros or uninitialized tensors.
   - Specify dense random inputs of realistic scale.
   - Specify several allocations or a flushed L2, given the placement effect of up to 43.6%.
4. **Alignment deserves its own action.** The unaligned specialization is about twice as slow at batch 1 (Section 4). Fig. S3 shows L = 2382 far slower than L = 2048. For this kernel, padding lengths or strides to a multiple of 16 is worth more than any chunk choice, yet the recipe only says to key on the bit.
   - Recommend padding or aligned allocation where the engine allows it.
   - Check offline whether the production Triton kernels show the same sensitivity. The captured calls include unaligned lengths.
5. **Unsupported items should be labelled as precautions.** Three recipe items have no supporting measurement in the paper:
   - invalidating on a power-limit or driver change;
   - broadcasting the chunk across TP ranks;
   - consistency with prefix-cache checkpoints.

   For the prefix-cache item, be concrete: some engines tie the state-checkpoint granularity of hybrid prefix caching, and the per-step chunk metadata, to the model's configured chunk size. A per-shape chunk may then be unsupported without engine changes.
6. **"Seconds per GPU" holds only for eager timing.** Under graphs, tuning means capturing one graph per candidate chunk per capture size. Report the capture time and graph-pool memory this costs, or recommend tuning in device time before capture. Note that S8 shows device-time timings of the multi-kernel pipelines are unreliable near the launch floor, which is exactly where the graph-mode gains are.

**Effort.** Items 1 to 3 and 5: text. Item 4: text plus offline analysis. Item 6: small new GPU measurement or text.

### M5. The cost model's practical role is not established, and its strongest use case is buried

**Location.** Abstract; Section 4; Section 6.4; Section 6.7 "Calibration versus profiling"; Conclusion; Supplementary S2, S3, Table S18.

**Why it matters.** The paper's own numbers leave the model little room:

- Calibration costs 130 to 322 s warm (493 to 843 s cold). Profiling all six chunks of all 57 exact keys costs 3.9 to 7.4 s warm.
- Compilation, not timing, dominates both (363 of 493 s). The model needs the compile facts of every candidate configuration, so it saves only the timing: about 0.03 to 0.05 s per key.
- Pruning to the model's top two or three candidates saves only a few tens of milliseconds per key.
- A length-only tree is within one to three points of the model on several GPUs.
- The model form was selected on the test calls of two of the three GPUs. A priori accuracy is 3.0 to 6.2%.

"Keys that cannot be enumerated" is never made concrete. With the bucketed keys the paper itself recommends, the key set is always enumerable at warm-up.

The one scenario where a model cannot be replaced by profiling is producing a table for a GPU that cannot be timed, for example ahead-of-time builds for a heterogeneous fleet. S2 supports this case: RTX PRO 6000 parameters combined with H800 compile facts select with 3.8% regret (3.3% in device time). It appears only in the supplement.

**Concrete fix.**

1. Reposition the model around cross-device and ahead-of-time table generation. Move the S2 transfer results, for all source and target pairs, into Section 6.4.
2. Either give a concrete case where keys cannot be enumerated, or shorten Sections 4 and 6.4 and the abstract accordingly.
3. Drop the Gated DeltaNet "instance" from Table 2. Five parameters fitted to nine timings, with no regret reported, is not a model result.

**Effort.** Text plus offline analysis.

### M6. The large effects come from an in-house kernel that is slower than the deployed one, measured against an arbitrary "default"

**Location.** Section 5.1; Section 6 opening; Tables 4, 5, 8; Abstract; Supplementary Tables S9, S13.

**Why it matters.**

- **The Triton scan is slower than the reference.** At its best chunk it runs at 1.23 times the reference CUDA scan's latency (geometric mean; up to 1.54 times) and is slower at 10 of the 12 keys (Table S9).
- **Its launch configuration is not tuned for these GPUs.** The launch table was picked on an RTX 3070. It is 7 to 18% behind the fully tuned kernel on the other GPUs and in device time. Several configurations reach 255 registers with spills (Table S13). The steep chunk curve is therefore partly the curve of a kernel that is not fully tuned.
- **The default is borrowed.** Its "default" chunk of 256 comes from Mamba-2.
- **Consequence for the headline numbers.** "4.4 to 25 ms per forward pass" (Table 8) and "49% per call" do not tell an engineer what they would gain on kernels they run. The deployed Mamba-1 CUDA scan loses 6.6% (launch-inclusive) or 3.1% (device time) per call with its stock rule, and nothing measurable end to end at batch 1 (Table S39).
- **"Default" is used inconsistently.** It means chunk 256 for the Triton scan in Table 8, the "best single chunk of each regime" for the Triton scan in Table 7 and Fig. 2, and the stock dispatch for the reference scan.

**Concrete fix.**

1. Lead with the production kernels.
2. Label the Triton-scan results as a "steep-curve case study" in every caption, and rename the kernel instead of redefining "Mamba-1" (Section 6 opening).
3. Use one definition of "default" throughout.
4. Using the 184-configuration sweep the authors already have, show whether the chunk curve stays steep when every chunk gets its best launch configuration per GPU.

**Effort.** Text plus offline analysis (the data exists).

### M7. Readability and self-containedness fall short of what a TACO reader needs

**Location.** Throughout; especially the Abstract, the Section 6 opening, Sections 6.4 and 6.7, Tables 5 and 7, and the Conventions box.

**Why it matters.**

- **Number density.** Most sentences carry three to six numbers with nested parenthetical qualifiers, spread across about six measurement contexts: device time, launch-inclusive, graph replay, warm or cold L2, one block or three rounds, and "single-block session" or "second session". The Section 6 opening and the "Calibration versus profiling" paragraph each take several readings.
- **Dependence on the supplement.** The main text refers to the supplement 47 times, and the evidence behind the recipe lives only there: Tables S33, S35 and S36, Fig. S3, and the S2 transfer result.
- **Few figures.** The article has two figures in 20 pages.
- **Table typesetting.** Tables 5 and 7 are set in \scriptsize with 2 to 2.2 pt column separation.

**Concrete fix.**

1. Restructure Section 6 around the three questions of Section 1. For each, give one answer paragraph and one figure, and move the secondary statistics to the supplement: conditional mutual information, permutation tests, depth-6 trees, bootstrap intervals of the fit.
2. Replace the Conventions box with a small table that maps each measurement convention to the deployment regime it approximates and the tables that use it.
3. Bring into the main text Table S33 (merged with Table 7), Fig. S3, the S2 cross-GPU transfer result and the S11 divergence result.
4. To make room, compress Section 2 (dense citation lists), the information-theoretic tests in Section 6.3, and Section 6.7's key-churn derivation. Its practical conclusion fits in a sentence: profile at warm-up, at 30 to 50 ms per key.
5. Limit the abstract to one number per claim.

**Effort.** Text.

---

## 6. Minor Issues

### Notation and terminology

1. Section 4, "Compiler specialization and parallel shapes": "Tensor parallelism over *t* devices runs the scan at width d/n_tp". The symbol t is never defined; use n_tp throughout.
2. Table S14 lists "t_0 (µs)" where Eq. (3) uses τ_0.
3. Redefining "Mamba-1" as the in-house Triton kernel "below unless stated otherwise" (Section 6 opening) will mislead readers, who associate Mamba-1 with the reference CUDA kernel. Use a distinct name.
4. Many technical terms are defined across Sections 3 and 5 and the Conventions box: oracle headroom, headroom, residual regret, floor, covered, tie, and LW share. Add a short glossary to the notation table, or keep fewer terms.
5. "ms per forward pass" is an average over a call mix that is not workload-weighted. Define it precisely in Section 5.3, or switch to ms per prefill at stated (L, B) points.

### Figures

6. **Fig. 1.**
   - The caption does not state the timing convention. Table S1 suggests launch-inclusive, which explains the 0.5 ms SSD and GDN floors and the 0.06 ms Mamba-1 floor at L = 512. This matters because the paper's primary convention is device time.
   - The panel titles ("Mamba-2 SSD", "Gated DeltaNet") sit on the tick labels of the panel above.
   - Three stacked narrow panels leave half the page width empty. Use a 1 x 3 layout with a shared legend.
7. **Fig. 2.**
   - The fonts are too small for print, and the x-axis has only three ticks.
   - In the Triton panel, "default" is a different chunk for each curve.
   - A host-bound RTX PRO 6000 series is missing.
   - Bins are not annotated with call counts.
   - Add the latency-weighted series, which is the one that matters for throughput.
8. **Fig. S5.** The rotated labels "1/8 histogram" and "full histogram" overlap the plotted lines.

### Tables

9. **Table 5.**
   - The MLP rows report seed 0 while the text reports five-seed means. Report mean ± s.d. in the table.
   - The † row on the RTX 3090 comes from a different session than the rest of the column. Footnote this more prominently or use the same session.
   - The first row merges "default chunk 256" and "entropy rule" because they coincide. Split it into two rows.
10. **Table 7.** The parenthetical "(20.1)" in the ms column (oracle scan time) is easy to miss; give it its own column. Move the two work-term-model columns to the supplement.
11. **Table 8.**
    - The rows mix conventions (device time, launch-inclusive, above-floor only), so they cannot be compared row to row.
    - Add the oracle scan time per forward pass. The identical 4.42 ms for the RTX PRO 6000 and the H800 is correct per the result files: the oracle scan times are 13.1 and 24.2 ms. Readers will still suspect a copy error unless the oracle column is shown.
12. **Table 4.** The headers "Oracle ≠ static" and "Rule regret" are cryptic; write "calls whose oracle differs from the best static chunk" and "entropy-rule regret".
13. **Table 2.** The "Regret" cell "0.4*/3.5*/3.0 (4.2/6.2/3.0)" is hard to parse. Split it into "terms selected on these calls" and "a priori" columns.

### Specific claims and wording

14. Abstract: "three GPU generations" applies to Mamba-1 only; the production-kernel results cover two workstation GPUs.
15. Section 6.6: "chunk 128 adds 16.8 ± 1.4 ms", while Table S39 gives +16.8 ± 1.5 in percent. Both are right only because the baseline is about 100 ms. Write "16.8 ms (16.7%)" and keep the units consistent.
16. Section 6.6 and S15: "3.1% of end-to-end time" depends on an eager decode loop of about 40 ms per token. State this, or drop end-to-end percentages from this testbed.
17. Section 7 cites "Table S33" and "Section S11" without "Supplementary", inconsistent with the rest of the article.
18. Section 1, end of the third question: "... the trade-off between model-driven and empirical optimization [Yotov et al. 2003]?" The appended noun phrase makes the question ungrammatical. Rephrase.
19. Data types: only fp16 (Mamba-1, SSD) and bf16 (GDN) were measured, yet the key includes the data type. Either show one bf16 versus fp16 comparison for SSD, which ships in bf16, or note that the dtype dependence is untested.
20. Section 6.5, "Packed variable-length calls": these are the closest thing to how hybrid-model engines run prefill, yet they get one paragraph and device-time timing only. Promote them (see M2).
21. Recipe, "Which keys": the claim "profile every chunk of each (seconds per GPU)" should state that it excludes compilation, which costs 27 to 40 s for the captured workload.
22. Section 6.4: "on the RTX 3090 one of 50 random starts reaches a 16% lower objective and 1.7% regret, so the fit we report there is a local optimum". Fixing the three-restart protocol before scoring is commendable. For practitioners, though, say what protocol you would recommend.
23. Speculative-decoding verification and chunked-prefill slices produce very short chunked-kernel calls, the regime where Fig. 2 shows the largest default regret. A sentence in the Discussion would tie the result to a practical use.
24. Line-level typography: the Conventions and Recipe boxes are set in \footnotesize inside \fbox with little padding. Consider a lightly shaded box with normal-size text, since these are the parts practitioners will reuse.
