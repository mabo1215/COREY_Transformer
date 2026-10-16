# ACM TACO: Editor-in-Chief Assessment

**Manuscript:** "Shape Sets the Chunk: Selecting and Keying the Chunk of Chunked Linear-Recurrence GPU Kernels" (double-anonymous, acmsmall; main text 20 pages + references; Supplementary Material 31 pages)
**Type:** New regular paper (declared "New Paper, Not an Extension of a Conference Paper")
**Basis of this assessment:** main.pdf/main.tex, appendix.pdf/appendix.tex, ref.bib, and the result files in `src/experiments/results/tpds/`. I recomputed 20 numbers independently from the raw per-call timing files where I could (Section 7).

---

## 1. Summary

The paper asks how a runtime should pick the chunk size of chunked linear-recurrence GPU kernels: the Mamba-1 selective scan (a Triton kernel the authors wrote, plus the reference CUDA scan rebuilt so the chunk is a runtime argument), Mamba-2 SSD and FLA's Gated DeltaNet. It also asks whether the chunk can be predicted rather than profiled. Every chunk is timed inside one launch that carries the recurrence across chunks. The measurements cover prefill calls captured from 370M–2.8B checkpoints on an RTX PRO 6000 (Blackwell) and an RTX 3090 (Ampere), and, for Mamba-1 only, an H800 PCIe. The paper reports four findings:
- **The shape decides the chunk.** The optimal chunk depends on the shape key (dimensions, compiler specialization bits, device). Activation statistics add nothing: adding entropy changes a shape tree's regret by at most 0.03 points. An elementary lemma bounds what any data-dependent policy can gain by the oracle headroom of a shape-keyed table, and that headroom is below one host synchronization. Apparent "data effects" come from buffer placement and a warm L2.
- **Library defaults are fine when host-bound.** Under CUDA graphs (device-bound) the optimum moves with the shape, and the defaults lose 1.0–5.9% of scan time.
- **A compile-informed cost model works but rarely pays.** For the authors' register-bound Triton scan, a model of compiler resource facts selects the chunk with 3.0–6.2% regret when its terms are chosen a priori. Profiling every enumerable key, however, costs seconds, while calibrating the model costs minutes.
- **The study ends with a recipe** for keying and filling chunk tuning caches, plus an economic analysis based on replayed serving traces.

## 2. Scores

| Dimension | Score (0–100) |
|---|---|
| Novelty | 45 |
| Methodology | 74 |
| Experiments | 68 |
| Clarity | 38 |
| Significance | 45 |

**Confidence:** 4 / 5

## 3. Recommendation: **Major Revision**

**Why not Reject, and why not Minor Revision.**
- **Within scope and rigorous.** The submission is squarely in TACO's scope: autotuning, tuning caches and compile-informed performance modelling of GPU kernels. It is also unusually rigorous. It uses recurrence-preserving timing, shuffled rounds, held-out prompt clusters, cluster bootstrap, TOST and permutation tests, placement and L2 interventions, three timing regimes, and an autotuner-state control.
- **Honest negative results, verified numbers.** It reports its negative results honestly, and the numbers I checked are traceable to the released result files.
- **The decision boundary.** TACO accepts characterization papers that give practitioners insight they did not already have. Rigor is not the question; significance and clarity are.

**What keeps it from Minor Revision:**
1. **The practical case is unproven where these kernels run.** On production kernels the measured effect is 1–6% of scan time (0.1–1.4 ms per forward pass), and only under CUDA graphs. It is shown only on two workstation-class GPUs (one a non-stock board that may be virtualized), with no datacenter GPU and no engine-level measurement.
2. **The large headline effects come from a custom kernel and a borrowed default.** They belong to the authors' own Triton scan, which is 1.23× slower than the reference, scored against a default borrowed from another kernel.
3. **The core question is close to settled, and the cost model undercuts itself.** Whether data values matter for a data-oblivious kernel's tile is close to settled by first principles, and the refuted design is a single non-peer-reviewed preprint. The cost model's value is negated by the paper's own profiling economics.
4. **The manuscript is very hard to read.**

None of these is fatal, and all can be addressed in one revision cycle. If they are not, the paper would not clear TACO's significance bar.

## 4. Strengths

1. **Scope fit.** Chunk/tile selection, shape-keyed tuning caches, compile-fact-driven performance models and the choice between profiling and modelling are core TACO topics (cf. Kernel Launcher, Triton autotune, ATF, TVM cost models).
2. **Measurement methodology is a contribution in itself:**
   - **Recurrence-preserving timing.** The split-call pitfall (Section 5, main.tex:523–532) is a real trap for anyone benchmarking these kernels.
   - **Placement and L2 artifacts.** The placement-versus-values intervention (main.tex:731–744; Supplementary Table S10) shows that a cache timing one allocation with a warm L2 can mistake a 21% (up to 44%) placement effect for a data effect. Autotuner designers need to know this.
   - **The execution regime decides whether the chunk matters.** Host-bound (launch-inclusive), device-bound (device time) and CUDA-graph timings are kept separate, and the paper shows the regime determines whether the chunk matters at all (Fig. 2, Table 7).
3. **Statistical care.** Held-out evaluation by prompt fold, 42-cluster bootstrap intervals, paired differences, equivalence tests, permutation tests of conditional information, noise-null and split-half headroom estimates (Supplementary S12), and three-round re-timing.
4. **Honest reporting.** The authors separate in-sample term selection from a priori selection (3.0–6.2% versus 0.4–3.5%). They report that the RTX 3090 fit is a local optimum, that the eager-calibrated work-term model fails under CUDA graphs, and that profiling beats calibration for enumerable keys.
5. **Breadth.** Three kernel families plus the reference CUDA scan and an independently written public kernel (TFLA mLSTM). The study also covers larger checkpoints (1.4B, 2.7B/2.8B), tensor-parallel shard widths, packed variable-length steps, output divergence of chunk switching, and the production kernels' autotuner state.
6. **Actionable recipe** (main.tex:1320–1344): key on compiler specialization bits (one divisibility bit changes the kernel and its optimal chunk); time in the regime and L2 state that will be served; invalidate on driver or compiler change; broadcast the chunk under TP; keep one chunk per key for deterministic modes.
7. **Reproducibility.** Every number I spot-checked is in the result files; 18 of 20 checks match exactly, and 2 depend on definitional scope (Section 7).

## 5. Major issues

### M1. Practical significance on production kernels is modest and is not shown where these kernels are deployed
- **Location:** Abstract (main.tex:59–62); Section 6 intro (main.tex:661–664); Section 6.5, Table 7 and Fig. 2 (main.tex:1096–1150); Table 8 (main.tex:1242–1269); Limitations (main.tex:1425–1429); Conclusion (main.tex:1447–1449).
- **Why it matters:**
  - **Effect size.** For the kernels people actually deploy (Mamba-2 SSD, Gated DeltaNet, the reference Mamba-1 CUDA scan), defaults suffice host-bound. Device-bound, they lose 1.0–5.9% of *scan* time, i.e. 0.10–1.44 ms per forward pass (Table 7). The paper never says what share of end-to-end prefill (TTFT) the scan occupies in these models, so a reader cannot translate "5.9% of scan time" into a serving benefit.
  - **Hardware.** The device-bound evidence for the production kernels comes only from an RTX 3090 and an RTX PRO 6000. The RTX 3090 is a non-stock 48 GB board in a cloud container where virtualization or time-slicing "cannot be excluded" (main.tex:560–561). The datacenter GPU (H800) covers only Mamba-1.
  - **No engine measurement.** No measurement is made inside a serving engine with CUDA graphs. The "serving" analysis is a simulated replay (see M7).
- **Concrete fix:**
  - (a) Repeat the CUDA-graph replay of SSD 370M/2.7B and Gated DeltaNet 1.4B on a datacenter GPU (H800/H100/B200).
  - (b) Add one end-to-end measurement in a real engine with CUDA graphs enabled (e.g., vLLM or SGLang serving a Mamba-2 or Gated DeltaNet hybrid). Report TTFT or prefill throughput with the default chunk versus a shape-keyed table.
  - (c) Report, per checkpoint, the scan's share of the forward pass (this can be derived offline from the routed study), so the scan-time percentages map to end-to-end impact.
- **Effort:** new GPU measurement (c: offline analysis).

### M2. The large headline effects come from the authors' own, slower kernel scored against a borrowed default
- **Location:** main.tex:536–545 (Triton kernel at 0.81–1.54× the reference, geometric mean 1.23×); main.tex:653–660 (chunk 256, "the Mamba-2 default, serves as its untuned reference and loses 49% per call ... 34% of the scan time"); Table 4; first rows of Table 5; Table 8 "Default" column (4.4–25 ms per pass); Supplementary S1.2 (default launch table chosen on a fourth GPU, an RTX 3070).
- **Why it matters:**
  - **Inflated headline numbers.** The 8.5× chunk spread, the 49% default regret and the 4.4–25 ms-per-pass numbers measure a kernel that has no library default. It is scored against another kernel's default, and its launch table was tuned on an unrelated GPU.
  - **Realistic baselines are much smaller.** A per-batch chunk loses 20%, and a length-bucket table, built in seconds, leaves 2–20 μs per call (Supplementary Table S23). The abstract and Section 6 intro lead with the inflated numbers.
  - **Production kernels show small effects.** On the production kernels the effects are an order of magnitude smaller.
  - **Steep curve may be an artifact.** A reader may suspect that the steep chunk curve reflects the kernel being under-optimized: it spills at chunks 512/1024 (Supplementary Table S13) and is 1.45× slower than the reference at L=8192, batch 1.
- **Concrete fix:**
  - (a) Present the realistic untuned baselines (best chunk per batch; length-bucket table) as the reference points in the abstract, Section 6 and Table 8, and keep "chunk 256" as a secondary row.
  - (b) State up front that the paper's tuning gains for production kernels are those of Table 7.
  - (c) Either bring the Triton kernel to parity with the reference scan and show that the chunk sensitivity persists, or explicitly reposition it as a deliberately steep-curve testbed for the cost-model case study.
  - (d) In Table 5, merge the identical "Default chunk 256, entropy rule" and "Best single chunk, held out" rows (they coincide on two of three GPUs) or explain why they coincide.
- **Effort:** text only for (a), (b), (d); new GPU measurement for (c).

### M3. Novelty and framing: the central question is close to settled, and the target is a single preprint
- **Location:** Introduction (main.tex:112–139); Contributions (main.tex:141–162); Section 2 "Input-aware variant selection" and "Positioning" (main.tex:264–280, 322–327); Section 3, Lemma 3.1 (main.tex:364–404).
- **Why it matters:**
  - **The answer is expected.** For dense kernels with no data-dependent control flow and index-only addressing (which the authors themselves state, main.tex:386–391), the expected answer to "does the input distribution change the optimal tile?" is "no". Shape-keyed caches (Triton autotune, Kernel Launcher, FFTW wisdom) already embody it.
  - **The lemma is trivial, and the target is narrow.** The lemma is acknowledged as elementary. The design being "refuted" is one non-peer-reviewed arXiv preprint. For TACO readers this reads as a straw man.
  - **Relation to TFLA is thin.** TFLA [Beck et al. 2025] already derives the optimal chunk from shape and hardware. The relation is covered in two sentences (main.tex:221–231).
  - **The valuable insights are buried.** They lie elsewhere: the placement/L2 artifact; the host- versus device-bound regime map; the compiler-specialization bit in the key; eager-calibrated models failing under graphs; and the economics of profiling versus modelling.
- **Concrete fix:**
  - Reframe the paper as a characterization of chunk selection for chunked linear-recurrence kernels and of the measurement pitfalls that mislead tuning caches. Make those four or five insights the contributions.
  - Demote the entropy scheduler to the control it actually is (one paragraph plus the routed study).
  - Add a short "what was not known before" paragraph that positions each finding against TFLA, the Triton autotuner, Kernel Launcher and input-aware autotuning (Nitro, Ding et al., Seer).
  - Consider turning Lemma 3.1 into a remark.
- **Effort:** text only.

### M4. The cost-model contribution is undercut by the paper's own economics, and it is evaluated on one kernel only
- **Location:** Section 4 (main.tex:406–518); Table 2; Section 6.4 (main.tex:907–960); Table 5 "Calibration only" block; "Calibration versus profiling" (main.tex:1346–1356); Table 8 (RTX 3090 row); Supplementary Tables S14–S18.
- **Why it matters:**
  - **Weak a priori accuracy.** Chosen a priori, the model reaches 3.0–6.2% mean regret (95th percentile up to 50%).
  - **Calibration costs more than the profiling it would replace.** Calibration costs 493–843 s per GPU, against 3.9–7.4 s to profile every chunk of all 57 exact keys (main.tex:1347–1350). Calibration pays only beyond about 1600–2500 keys, and in the serving replays profiling every bucket key is cheaper in 99–100% of settings.
  - **Simpler alternatives match or beat it.** A three-parameter ranker plus three profiled chunks matches it. On the RTX 3090 the per-batch static chunk beats it in ms per forward pass (0.79 versus 2.90 ms, Table 8).
  - **Instability.** As a full launch-space auto-tuner it is unreliable (leave-one-key-out 8.9–23% mean, up to 95%; Supplementary Table S17). Its learning curve is flat or non-monotone on two of three GPUs (Table S16). The reported RTX 3090 fit is a local optimum (main.tex:932–935).
  - **Generality untested.** The form encodes this one kernel's recurrence and spilling, and "whether it transfers is untested" (main.tex:501–506). The one transfer attempt, a five-term fit on TFLA, selects with 7.5% regret, worse than the default (main.tex:1171–1173).
  - **Net effect.** As written, the third contribution ("When a model pays") is that it essentially does not pay. That is a legitimate result, but it currently occupies about a quarter of the paper.
- **Concrete fix:** either
  - (a) demonstrate the *procedure* (compile without running, read the resource report, calibrate at about 12 keys, choose terms a priori) on a second, independent kernel with a non-trivial chunk curve, with terms fixed before seeing its calls; or
  - (b) shrink Section 4 and the cost-model parts of Section 6.4 to a compact case study, move the ablation and baseline detail to the supplement, and use the space for M1 and M5.

  In both cases, state the honest conclusion as the contribution: profiling dominates for enumerable keys, and the model is a cold-start default or pruning aid. Quantify how often non-enumerable keys actually arise; Supplementary Table S38 suggests at most about 2% of calls under bucketed keys.
- **Effort:** (a) new GPU measurement; (b) text only.

### M5. Clarity and presentation are below TACO's standard
- **Location:**
  - Abstract (main.tex:48–69).
  - Section 6 intro (main.tex:647–664): one paragraph with a seven-clause sentence (653–660) that also defines what "Mamba-1" means for the rest of the paper.
  - Section 6.3 (876–902) and Section 6.4 (907–960): about 40 numbers in one paragraph.
  - Section 6.5 (1031–1175) and Section 6.7 (1199–1372).
  - Tables 5, 7, 8.
  - Discussion (1397–1412: two short paragraphs); Limitations (1416–1440: a single run-on paragraph).
- **Why it matters:**
  - **Prose is overloaded.** It routinely packs 5–10 numbers into a sentence.
  - **Too many parallel conventions.** Readers must track device time, launch-inclusive and graph replay; warm and cold L2; per-call, latency-weighted and absolute regret; in-sample, held-out, covered and split-half estimates; session 1 and session 2; single-block and three-round timings. There are also about 60 cross-references into a 31-page supplement with more than 40 tables.
  - **Tables mix conventions.** Table 5 packs three metrics into each cell across 17 rows and 3 GPUs, mixing timing sessions (RTX 3090 a priori row from another session) and seeds. Table 8 mixes conventions row by row (device time, launch-inclusive, above-floor only), so its rows are not comparable.
  - **Key figures are in the supplement.** These include the cost-model fit (Fig. S3) and the profiling-budget curve (Fig. S4).
  - **Discussion is thin.** It does not discuss implications for compilers or autotuners (e.g., integration into Triton's autotuner, `do_bench_cudagraph`, vLLM graph capture), generality, or what to expect on datacenter GPUs.
  - **Consequence.** Reviewers and readers will struggle to extract the message.
- **Concrete fix:**
  - Restructure Section 6 around four or five findings, each with one table or figure and a one-sentence takeaway.
  - Fix one primary convention per table and footnote exceptions.
  - Split Table 5 into "profiled selectors (information test)" and "calibration-only selectors".
  - Move Fig. S3 or S4 into the main text.
  - Cut the abstract to about 180 words with three findings.
  - Itemize the Limitations and expand the Discussion to about one page.
  - Reduce numbers in prose to those that carry the argument.
- **Effort:** text only.

### M6. Measurement environment, and consistency across sessions and software builds
- **Location:** Section 5.2–5.3 (main.tex:557–606); Conventions box (608–642); Section 6.8 (main.tex:1392–1395); Supplementary S1.1 and Tables S1–S2; "Repeated rounds" (Supplementary S8).
- **Why it matters:**
  - **Uncontrolled hardware.** All GPUs are cloud containers with unlocked clocks. The RTX 3090 is a non-stock board with possible sharing below the container. The power cap is active in up to 92% of RTX 3090 readings.
  - **Unguarded data sets.** Several data sets were timed without the exclusive-GPU guard: the value-invariance table (Table 3), the launch-inclusive and H800 captured calls, and most calibration data.
  - **Session non-reproducibility.** Launch-inclusive timings did *not* reproduce between sessions on the same RTX PRO 6000: median oracle latency 98 versus 60 μs, and the shape tree's regret rose from 0.7 to 4.7%.
  - **Inconsistent builds and protocols.** mamba_ssm differs between GPUs: the PRO 6000 uses a "local build" whose result-file metadata carries a non-release version suffix. The H800 uses one block per chunk.
  - **Possible bias from the spin.** The device-time convention places a 20 ms GPU spin before each block. This changes the clock and power state relative to real execution with idle gaps.
  - **Why this matters for the argument.** The decision-level conclusions (which chunk wins) look robust. However, the lemma-based exclusion argument rests on *absolute* microsecond headroom compared with a 20 μs synchronization, and those values are exactly what such effects perturb.
- **Concrete fix:**
  - Re-time at least the device-time captured calls and the CUDA-graph replays on a bare-metal or stock GPU with exclusive access and, if possible, locked clocks.
  - Document precisely what the local mamba_ssm build changes, and confirm the SSD kernels are unmodified.
  - Ensure that every number in the main tables comes from guarded runs.
  - Report session-to-session variation of the headroom estimates.
- **Effort:** new GPU measurement (partly text).

### M7. Workload representativeness; the serving and economic conclusions rest on modelling, not measurement
- **Location:** Section 5.2 (main.tex:570–591); "Key churn" (1291–1318); "Serving traces" (1358–1372); "Deployment units" (1284–1289); Supplementary S13 and Table S38.
- **Why it matters:**
  - **Unweighted workload.** Prompts are truncated in rotation to cover all lengths, so every per-call mean is unweighted by any real workload (acknowledged at main.tex:574–575). Batched calls are crops of 2–10 distinct prompts reused across batches.
  - **Synthetic replay model.** The serving replay uses a synthetic step model (2 ms + 10 μs per token) that is 2.3× faster than measured prefill.
  - **Borrowed costs.** The replay maps every call shape to the nearest measured key of the *Triton Mamba-1* kernel, including packed calls costed as batch-1 Triton calls, and scales P_k from wall clocks.
  - **Consequence.** The break-even counts (1900/2660 calls) and "share of prefill" numbers are therefore model outputs. They inherit the inflated Triton effects of M2.
- **Concrete fix:**
  - Reweight the captured-call regrets by a real length and batch distribution (e.g., the Azure 2023/2024 or BurstGPT length histograms) and report the weighted figures beside the unweighted ones (offline).
  - Cost the production kernels' packed steps with their own measured timings (170 steps already exist on the PRO 6000).
  - Label clearly in the text which conclusions are measured and which are modelled.
- **Effort:** offline analysis (optionally new GPU measurement).

### M8. Editorial compliance: anonymity of the artifact (must be fixed before reviewer distribution)
- **Location:** Artifact statement (main.tex:1455–1461); metadata of the result files that back the tables.
- **Why it matters:**
  - **Identifying strings in the artifact.** The result files contain absolute host paths, hostnames, a project directory name, and a package version suffix. Together these name the system of the cited preprint [Ma et al. 2026] and a previous venue.
  - **Breach of double-anonymous review.** If the "anonymized archive" is built from these files, double-anonymous review is breached.
  - **Disclosure.** If the cited preprint is the authors' own work, the current third-person citation is acceptable, but the relationship should be disclosed confidentially to the editor. The cover letter should confirm that the "new paper" declaration accounts for it.
- **Concrete fix:** Scrub paths, hostnames, project names and version tags from all artifact files. Confirm the relationship to the cited preprint in the confidential comments to the editor.
- **Effort:** text only / artifact cleanup.

## 6. Minor issues

1. **main.tex:714.** "In 94% of the 414 cells the input-agnostic chunk is tied." Under the 2% tie criterion the Conventions box makes default, the pooled share is 93.5% (taco_within2_ties.json); 94.4% is the CI-overlap value. State the criterion or report 93%.
2. **main.tex:814–815.** "within 0.14% of the per-call oracle ... (0.04% in sample; 0.14 μs per call)" mixes populations.
   - The 0.14% is over all 2360 calls and depends on the fallback rule for the 204 uncovered calls. My plain length-bucket fallback gives 0.22%.
   - The 0.14 μs is over covered calls only; over all calls it is 0.37 μs.
   - Specify the population and the fallback rule.
3. **Table 5 versus Supplementary Table S14.** Two discrepancies:
   - Cost model on the PRO 6000: 0.40% versus 0.38%. Recomputed: 0.405% with the oracle on round medians, 0.383% on round means.
   - RTX 3090: 3.52% versus 3.55% (two sessions), and three-parameter ranker 0.85% versus 0.88%.

   Harmonize or footnote.
4. **main.tex:516–517.** "Tensor parallelism over $t$ devices runs the scan at width $d/n_{\mathrm{tp}}$" uses inconsistent symbols.
5. **Eq. (3).** Waves $w$, resident warps $W$ and the thresholds $W_0, W_1$ are easily confused. Rename the waves (e.g., $n_{\text{wave}}$).
6. **Table 2.** The "Calib." column needs a unit ("timings per GPU"). The "Regret" cell mixes conventions and an asterisk convention; simplify.
7. **Fig. 1 caption versus Table 3.** Fig. 1 says chunk 512 is fastest at batch 1, but Table 3 has C* = 256 at L = 512. Say "tied" or qualify by length.
8. **main.tex:1186–1187.** "3.1% of its end-to-end time" includes 32 decode steps. For a prefill-focused paper, report the prefill/TTFT effect (16.7%) as the primary number.
9. **main.tex:1211–1216.** "The oracle headroom bounds g" is computed there against a single static chunk, whereas Lemma 3.1's h(s) is relative to the shape-keyed policy. Separate the two baselines explicitly.
10. **Supplement-only citations.** main.tex:1409 and 1412 cite "Table S..." and "Section S..." without the "Supplementary" prefix used elsewhere. More generally, the main text should be readable without the supplement; at present Fig. S2, S3 and S4 carry arguments the text relies on.
11. **Abstract "three GPU generations".** Only Mamba-1 was measured on Hopper; qualify this. Also state in the abstract that Gated DeltaNet is limited by the library to chunks ≤ 64 (a three-point design space).
12. **Floor definition.** The launch floor ("twice the median oracle latency of batch-1 calls with L ≤ 256") is ad hoc. Give a sensitivity check (e.g., 1.5× and 3×) for the shares above the floor in Table 7.
13. **Spin bias.** Discuss whether the 20 ms GPU-side spin in the device-time convention biases clocks or power relative to graph replay. A short comparison of device time against graph replay on the same calls (both exist for Triton Mamba-1) would settle it.
14. **Reference-scan session mismatch.** The reference-scan results come from two sessions (launch-inclusive stock dispatch 6.6% in Table 6; device-time stock pairing 3.1% in Supplementary Table S21). Say which number supports the claim "a shape tree halves that".
15. **Table 7.** The "Work-term model: Eager / Refit" columns use a different metric (per-call regret) from the neighbouring columns; separate them visually.
16. **Lemma 3.1.** Since the result is admittedly elementary, a "Proposition" or "Remark" would better match its weight.
17. **Bibliography.** The cited preprint is filed under cs.CV, which is odd for a GPU-kernel scheduler; check the entry. Several 2026 arXiv-only references (Mamba-3, log-linear memory, Astra) should be updated to their archival versions if they exist.

## 7. Number spot-check

Recomputed means I re-derived the value from the raw per-call files (`e2_*.csv`, using `lat_c*_med` or `lat_c*_mean` as stated in the captions) with my own scripts. Analysis file means I read the value from the authors' stored analysis.

| # | Claim | Location | File / key | Match |
|---|---|---|---|---|
| 1 | Mamba-1, batch 1: best static 256, 18.8 / 50.5%, oracle ≠ static 85% | Table 4, main.tex:785 | `e2_mamba1_pro6000_r4spin.csv` (round medians), recomputed | yes |
| 2 | Entropy rule regret 88.8 / 313.6 (B=4) and 126.3 / 386.1 (B=16); rule selects 256 for every call | Table 4, main.tex:786–787, 804–806 | same CSV, `entropy_rule_chunk`, recomputed | yes |
| 3 | Latency-weighted loss of chunk 512: 8.8 / 18.3 / 17.6% | Table 4 | same CSV, recomputed | yes |
| 4 | Mamba-2 rows: 64: 1.5/5.9; 256: 1.0/2.8; 256: 2.0/6.8; rule 1.8/6.3, 1.1/2.9, 2.3/11.4 | Table 4, main.tex:789–791 | `e2_mamba2_pro6000.csv` (block means), recomputed | yes |
| 5 | Chunk 256 loses 49% per call (34% of scan time); spread median 4.8× (p95 15.2×); Mamba-2 1.04× (2.7×) | main.tex:656–657, 761–763 | both CSVs above, recomputed (48.6%, 33.8%, 4.77×, 15.2×; 1.04×, 2.71×) | yes |
| 6 | Table 3: Mamba-1 spreads 2.50/5.52/8.52×, max regret 1.36/0.00/0.57%; Mamba-2 3.50/5.79/0.82%; tied 92/71/88/88% | Table 3, main.tex:695–705 | `e1_analysis_pro6000.json` → `decision`; `taco_within2_ties.json` | yes |
| 7 | "94% of the 414 cells tied" | main.tex:714 | `taco_within2_ties.json` pooled 0.935 (2% rule); `e1_analysis_pro6000.json` → `overall.tie_aware_agreement` 0.944 (CI rule) | partial (criterion) |
| 8 | 12 of 84 significant input effects, up to 43% | main.tex:722–724 | `e1_analysis_pro6000.json` → `overall`, `effects` (12; 43.0%) | yes |
| 9 | TOST bound at the selected chunk: median 3.6% (PRO 6000), 3.2% (RTX 3090) | main.tex:725–727 | `e1_analysis_{pro6000,vgpu3090}.json` → `equivalence.summary.all.at_best_chunk` (3.58, 3.16) | yes |
| 10 | Exact-shape table: 0.04% in sample, 0.14 μs per call, 0.14% held out | main.tex:814–815, Table 5 | recomputed from the r4spin CSV: in-sample 0.042%, covered calls 0.14 μs; all-call 0.138% only with the authors' fallback (`taco_report_numbers.json`); plain bucket fallback gives 0.22% | partial (scope) |
| 11 | Cost model, PRO 6000 device time: 0.40 / 3.8 / 92% | Table 5, main.tex:859 | `cost_model_pro6000_r4spin.json` → `per_call_choice`, scored on CSV round medians: 0.405 / 3.76 / 92% (round means 0.383) | yes |
| 12 | H800 default chunk 256: 50.9 / 259 / 37% | Table 5, main.tex:845 | `e2_mamba1_h800_r3spin.csv`, recomputed | yes |
| 13 | CUDA-graph default regret, LW share, ms per pass (oracle ms), floor and share above floor, all 8 rows (e.g. SSD 2.7B PRO 19.2/69.1, 5.9%, 1.29 (22.1) ms, 34 μs, 64%) | Table 7, main.tex:1114–1121 | `e2_{mamba2,gdn,mamba1}_*graph3.csv`, recomputed | yes |
| 14 | Tuning 12 keys 493 s, 363 s compilation; profiling six chunks 1.2 s | main.tex:979–980 | `e4_pro6000.json` → `summary.mamba1` (492.7 s; Σ first call 362.9 s; 1.17 s) | yes |
| 15 | Routed: chunk 128 adds 16.8 ± 1.4 ms to the 100.4 ms prefill (3.1% total); proxies +19–66%; full histogram +49–191% | main.tex:1185–1191 | `e10_routed_vgpu3090.json` → `lengths` | yes |
| 16 | TP pooled: unsharded table 36.8/22.1%; calibration-width table 16.5/2.2%; model 6.5/2.9%; reference 2.2/6.6%, model 1.8% | main.tex:1068–1077 | `e12_analysis_{vgpu3090,h800}.json` → `families.*.pooled_shards` | yes |
| 17 | Key churn: online beats profiling below 1900 / 2660 calls; model break-even 4564 / 3688 calls | main.tex:1303–1313 | `serving_key_churn.json` → `hosts.*.e5b_summary_n_star`; `deployment_units.json` → `m6b_captured_keys` | yes |
| 18 | Packed steps, SSD 2.7B fresh: default 9.0% / LW 3.6%; bucket table 2.8% | main.tex:1157–1159 | `e27_varlen_pro6000_taco.json` → `families.ssd2p7b.fresh` | yes |
| 19 | Chunk switching changes the greedy output of 2.6–9.4% of prompts; per-length table up to 4.7% | main.tex:1410–1412 | `e31_divergence_4c_{m1,m2,gdn}.json` | yes |
| 20 | Table 8, Triton PRO 6000: 4.42 / 2.03 / 0.30 / 0.02 ms per pass | Table 8, main.tex:1257 | `taco_report_numbers.json` → `M4a_selector_tables.mamba1.pro6000_device` (`deployment_units.json` holds the launch-inclusive variant, 4.30/0.34/0.03) | yes |

**Verdict on number integrity:** 18 of 20 checks match exactly. The two partial matches (items 7 and 10) are definitional: which tie criterion is used, and which call population with which fallback rule. They are not data errors, and both appear in Minor issues 1–2. No fabricated or untraceable number was found in the sample.
