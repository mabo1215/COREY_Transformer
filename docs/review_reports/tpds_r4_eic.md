# EIC Review Report — TPDS round 4 (independent panel)

**Manuscript:** "Shape Sets the Chunk: A Compile-Informed Cost Model and a No-Gain Bound for Data-Dependent Dispatch of Chunked Linear-Recurrence Kernels"
**Venue:** IEEE Transactions on Parallel and Distributed Systems, regular paper, double-anonymous
**Material reviewed:** `paper/main.tex` (main body, 14 formatted pages, IEEEtran journal class), `paper/appendix.tex` (Supplementary Material, 16 pages), `paper/ReproducibilityChecklist.tex`, `paper/ref.bib`, build logs `paper/build/main.log` and `paper/build/appendix_only.log`. The body and the supplement were reviewed jointly as one paper. No PDF renderer was available; page facts come from the build logs and `.aux` files.
**Reviewer identity:** Editor-in-Chief persona, parallel and distributed systems journal; bird's-eye assessment of fit, originality, significance, positioning and submission readiness. Deep methodology checking is left to Reviewer 1.
**Confidence:** 4/5 (GPU performance modeling and LLM serving are within my area; I did not re-derive the cost model or re-run any measurement).

---

## 1. Scores (0–100)

| Dimension | Weight | Score | Justification |
|---|---|---|---|
| Novelty | 20 | **66** | The compile-informed cost model (register/spill/thread-layout inputs read from the compiled kernel, eight per-GPU parameters, cross-GPU and cross-width transfer) is a real but incremental advance over TFLA-style and occupancy models. The no-gain bound is, by the authors' own words (Sec. IV, "The bound is elementary"), a one-line inequality, and the negative result refutes a dispatch policy (COREY) that the authors built themselves. |
| Methodology | 25 | **74** | Measurement discipline is above the TPDS norm: recurrence-preserving launches, CIs, BH-corrected F tests, TOST equivalence, placement and L2 controls, prompt-grouped folds, two timing conventions, three GPUs. Gaps an editor must weigh: the model form was selected on the captured calls that then serve as the test set (Sec. V, "Model form and data"); Mamba-2 evidence is eager-mode only because its device-time timings were discarded (Supp. S7); clocks unlocked on cloud instances. |
| Experiments | 25 | **72** | Breadth is strong: three kernel families, 2360 captured calls per checkpoint at every chunk, 370M–2.8B checkpoints, tensor-parallel widths, five public serving traces, replication on RTX 3090 and H800. But the dramatic numbers (8.5x spread, 63% loss after sharding, 48.8% cross-GPU loss) all come from the study's own Triton kernel; on the production CUDA scan, SSD and Gated DeltaNet kernels the paper's own data show little to gain, and no multi-GPU job is run. |
| Clarity | 15 | **66** | Terminology (regret, headroom, tied, shape key) is defined once and used consistently, and the Limitations section is exemplary. The results prose is, however, extremely number-dense (Sec. VII-D alone packs more than twenty-five figures into three paragraphs), Sec. III spends a full column and two equations on a policy the paper then shows cannot work, and the reader must hold three different cost-model instantiations in mind. |
| Significance | 15 | **58** | For TPDS readers the transferable value is the measurement guidance and a cost model that avoids per-(key, device) tuning. Its practical payoff is undercut by the paper's own end-to-end study (Table V: static 512, static 2048 and the stock length dispatch within 0.9% of each other at 4096 tokens; "about 46 ms whatever the chunk" at 256 and 976 tokens) and by the finding that the production kernels have a single good chunk. The distributed dimension is argued through shapes, not measured. |
| **Weighted overall** | | **68.3** | 0.20·66 + 0.25·74 + 0.25·72 + 0.15·66 + 0.15·58. |

---

## 2. Recommendation: **Major Revision**

The paper is not at the bar for acceptance at TPDS in its current form, but the path to acceptance is bounded (mostly text and offline analysis; at most about 10 GPU-hours of new measurement). The decisive reasons:

1. **Journal fit rests on an unmeasured parallel/distributed dimension.** Tensor parallelism enters only as a per-device width timed on one GPU with synthetic inputs, sequence parallelism is not measured at all, and serving is a simulated scheduler replaying trace lengths (Sec. IX: "We do not run multi-GPU jobs"). A TPDS regular paper that advertises transfer "to tensor-parallel shard widths" and "mixed GPU fleets" in its abstract should contain at least one measured parallel deployment, or must reframe those claims as single-device consequences of parallel deployments.
2. **The headline numbers are attributed to the wrong kernel.** The 63% and 48.8% losses and the 8.5x chunk spread belong to a Triton kernel written for this study that is on average 1.23x slower than the reference CUDA scan (Sec. VI-B). The abstract and contribution list do not say so; the reference scan's single-GPU table "transfers (2.2% pooled)" and the two production kernels have one optimal chunk. As written, a reader of the abstract will over-estimate the practical stakes.
3. **The negative result targets a self-built policy.** No external proposal of activation-driven chunk or tile dispatch for dense kernels is cited; COREY is "the instance we measure" (Sec. II). Roughly five of nine results subsections refute it. The editorial question "who holds the hypothesis being refuted?" must be answered in the Introduction, and the space devoted to COREY must be reduced in favor of the positive contribution.
4. **A double-anonymity defect must be fixed before any re-review** (Code Ocean capsule link; see Venue compliance).

The measurement rigor, replication across three GPUs and the candid Limitations section are already at the journal's standard; the decision reflects fit, framing and positioning rather than validity. If items 1–4 and the MAJOR findings below are addressed, I would expect the paper to be acceptable after one further round.

---

## 3. Findings

Effort classes: (a) text only; (b) offline analysis of existing results; (c) new GPU measurement with estimated GPU-hour cost.

**F1. CRITICAL (conditional) — The availability link can de-anonymize the authors.**
Evidence: Section "Code and Data Availability": "The code and data for this article are available at \url{https://codeocean.com/capsule/3457679/tree}". A numeric `codeocean.com/capsule/<id>/tree` URL is the public published-capsule form, which displays capsule authors and affiliations.
Action (a): For the review version, replace the URL with an anonymized review link (Code Ocean private peer-review link, an anonymized repository mirror, or "an anonymized artifact is provided to the editor") and state that the DOI will be inserted at camera-ready. If the capsule is already published with names, it must be withdrawn from the review package. Verify before resubmission by opening the link in a logged-out browser.

**F2. MAJOR — Headline chunk-sensitivity numbers are not attributed to the study's own kernel.**
Evidence: Abstract: "transfers across GPUs and to tensor-parallel shard widths, where a table tuned for one GPU loses up to 63%"; Contributions: the same 63%; Sec. VI-B: the Triton kernel "takes 0.81 to 1.54 times the latency of the reference scan ... (geometric mean 1.23)"; Sec. VII-E "Tensor-parallel widths": "On the reference CUDA scan the optimal chunk hardly depends on the width, and the single-GPU table transfers (2.2% pooled; cost model 4.0%)"; Sec. IX: "Only the Mamba-1 scan has an optimal chunk that moves with the shape by enough to matter".
Action (a): In the abstract and in the first contribution bullet, name the kernel to which 63%, 48.8% and 8.5x apply ("the Triton Mamba-1 scan of this study") and place the reference-scan (2.2% pooled), Mamba-2 SSD (at most 0.24% above the launch floor) and Gated DeltaNet (fixed optimum) figures next to them. Add one sentence to the Introduction stating that the production kernels studied are far less chunk-sensitive.

**F3. MAJOR — The parallel/distributed dimension is argued, not measured.**
Evidence: Sec. V "Parallel deployments": "Model parallelism changes the shape key of every scan call and nothing else"; Sec. VII-E: shard widths "timed ... in three rounds on the RTX 3090, with synthetic inputs"; Sec. VII-H "Serving traces": "through a simulated prefill scheduler"; Sec. IX: "We do not run multi-GPU jobs ... the state exchange of sequence parallelism is not measured".
Action, preferred (c): run one end-to-end confirmation on a 2-GPU node (2x RTX 3090 or 2x H800): the 1.4B or 2.8B Mamba-1 checkpoint under 2-way tensor parallelism (4-way if a node is available), prefill at three lengths (512, 2048, 8192) and two batch sizes, 30 interleaved rounds, comparing per-shard chunk from (i) the table tuned at the unsharded width, (ii) the cost model calibrated at 370M width, (iii) the per-shard oracle. Report prefill ms and the paired difference. Estimated cost: 4–8 GPU-hours including setup. Alternative (a): retitle the "Parallel deployments" paragraph and the abstract claim to "the single-device shapes that tensor parallelism and chunked prefill produce", and remove "mixed GPU fleets" from the abstract unless F3(c) is done.

**F4. MAJOR — The negative result refutes a policy class with no external proponent.**
Evidence: Sec. I: "and for these kernels we built one, COREY, which estimates the entropy of the kernel input and maps it to a chunk"; Sec. II "Input-aware variant selection": "In all of these the features describe the structure of the work (sizes, sparsity, routing). Our bound asks what remains when that structure is fully described by the shape key, and COREY is the instance we measure." Sec. III, Eqs. (1)–(2), Fig. 1 and Secs. VII-A to VII-C, VII-G are devoted to it.
Action (a): In the Introduction, state explicitly whether any published runtime or paper proposes value-dependent dispatch for dense chunked kernels; if none, say so and present the bound as closing a design question that practitioners raise (the paper's Sec. VIII (ii) already lists the fused-statistic and device-side-launch variants that would keep the question open). Reduce Sec. III to at most half a column: keep the three policy classes and Fig. 1, move Eqs. (1)–(2) and the variant list to the supplement (Sec. S10 already defines them). Use the freed space for F2, F5 and F7.

**F5. MAJOR — The practical gain for a deployer is never stated in deployment units.**
Evidence: Table V and Sec. VII-G: "Static 512 +0.0 ... Static 2048 +0.3 ... Stock length dispatch +0.7" prefill at 4096 tokens; "chunks 1024 and 2048 and the stock dispatch are within 0.9%"; "At 256 and 976 tokens the prefill takes about 46 ms whatever the chunk"; Sec. VII-E: "the stock length-based dispatch, itself a shape rule, comes within 6.6% of the per-call oracle"; Sec. VII-E "Production kernels at larger widths": chunk 256 "loses at most 0.24% at any batch size"; Gated DeltaNet chunk 64 "the oracle of 154 of the 155 captured calls".
Action (b): Add a short paragraph (or a four-row table) to Sec. VIII or VII-H that converts kernel regret into end-to-end prefill time per request for each kernel at the measured lengths and batch sizes: (cost-model chunk minus stock/default chunk) x number of layers, from the existing captured-call timings, with the routed-study calibration (48 x kernel difference predicted 17.4 ms against 16.8 ms measured) as the sanity check. State plainly that for Mamba-2 SSD and Gated DeltaNet at the studied widths the gain is approximately zero and that the cost model's value is for cold keys and new devices.

**F6. MAJOR — Three cost-model instantiations are presented as one model, and the term selection used the test calls.**
Evidence: Sec. V "Model form and data": "The terms of Eq. (5) were chosen with an ablation that scored candidate terms on the captured calls of the RTX PRO 6000 and the RTX 3090 ... we then froze the form. The captured calls are therefore a clean test set for the fitted parameters, which never see them, but not for the choice of terms"; Supp. S3 "Reference CUDA scan": "It has no spill terms; instead, per chunk, A = tau_f + N tau_m and I = N(... tau_x [n_w>1] log2(n_w+1))" with eight different parameters; Supp. S3 "Multi-kernel pipelines": "T = (F^p + W^p)^{1/p}" with five parameters; Contributions: "extends to the reference CUDA scan".
Action (a)+(b): (a) Present the three instantiations as one framework in a compact table in Sec. V (kernel; compile facts read; terms; number of parameters; calibration set size and time; regret), and write the term-selection step as a procedure a reader can apply to a new kernel. (b) Using existing timings only, repeat the term-selection ablation on one GPU (e.g., RTX 3090) and evaluate the selected form on the other (RTX PRO 6000, then H800) to show that the chosen form is not GPU-specific; report the resulting regret next to the 0.9%/3.2%/3.5% figures.

**F7. MAJOR — Positioning against performance predictors that use compiled-code or cross-device features is missing.**
Evidence: Sec. II "Performance models and measurement" cites only Hong and Kim (2009), Volkov (2016) and the Roofline model; Sec. V: "it takes the register allocation, spills and thread layout from the compiler rather than from the source, since that is where the chunk dependence of a compiled Triton kernel arises"; Contributions: "transfers between GPUs".
Action (a): Add a paragraph that positions the model against (i) cross-device DNN performance predictors such as Habitat (Yu et al., USENIX ATC 2021), which also transfer timings between GPU generations from hardware constants, (ii) mechanistic GPU models that read occupancy and issue behavior from compiled code (e.g., GPUMech, Huang et al., MICRO 2014), and (iii) learned tensor-program cost models beyond Ansor (e.g., TenSet, TLP). State in one sentence what Eq. (5) adds (spill traffic and distributed tile layout as first-class inputs; eight parameters calibrated in 8–14 minutes).

**F8. MINOR — Mamba-2 conclusions are eager-mode conclusions.**
Evidence: Sec. VII-A: "at batch 1 its multi-kernel pipeline is bound by the fixed cost of its several kernel launches"; Supp. S7: "For the Mamba-2 SSD pipeline ... the device-time timings mix both regimes (median within-block variation 85%); we do not use them, and the Mamba-2 results of the main paper are launch-inclusive"; "Without the Python-level launch cost, its latency floor at batch 1 falls from about 740 to about 320 us"; Supp. S3: the pipeline "stays launch-bound up to about 17,000 (RTX PRO 6000) and 9,500 (RTX 3090) tokens per call".
Action (c), optional but recommended: capture the SSD forward of the twelve tuning keys at the 370M and 2.7B widths in a CUDA graph and re-time the per-chunk curve (about 1–2 GPU-hours on the RTX 3090); report whether chunk 256 and the flat regime persist. Otherwise (a): add to Sec. IX that all Mamba-2 chunk conclusions hold for eager execution and that graph-captured serving may expose a different curve below the current launch floor. Reviewer 1 should weigh this.

**F9. MINOR — Number density impairs readability for a journal audience.**
Evidence: Sec. VII-D, third paragraph, reports in prose the Kendall rank correlation (0.81 and 0.61 against 0.79 and 0.72), top-2 and top-3 profiling (0.5%, 2.3%, 0.36%, 2.1%, floors 0.36% and 1.6%), leave-one-key-out (1.0% and 2.0%), and device-dependence of selected chunks, in one paragraph; Sec. VII-H "Serving traces" similarly stacks eight percentages in two sentences.
Action (a): Move secondary numbers (rank correlation, top-k profiling, leave-one-key-out, calibration-on-batch-1) into Table III or a new compact table, and keep one number per claim in the prose. Supplementary Table S-model-params already holds most of them.

**F10. MINOR — The body is at the 14-page limit with zero slack.**
Evidence: `build/main.log`: "Output written on ... main.pdf (14 pages"; `build/main.aux`: Sec. X Conclusion begins on page 13, references occupy the remainder of pages 13–14.
Action (a): Any addition from F2–F7 will exceed the TPDS limit and incur mandatory overlength charges. Compensate by compressing Sec. III (F4) and Sec. VII-G to roughly half a column each (keep Table V; the full routed table is already Supplementary Table S-routed-full), and by moving the "Auto-tuning baseline" (Sec. VII-F, nine lines of numbers) into Sec. VII-D.

**F11. MINOR — The reproducibility checklist in the package is an AAAI artifact and contradicts the paper.**
Evidence: `paper/ReproducibilityChecklist.tex`: "\usepackage[submission]{aaai2027}"; item 4.11 "The significance of any improvement or decrease in performance is judged using appropriate statistical tests" answered "no", while Sec. VI-D and Supp. S1.3 use BH-corrected F tests, TOST and paired within-round confidence intervals; items on novel datasets answered "partial" although the captured-call data set is released through the availability statement.
Action (a): TPDS has no reproducibility checklist; either remove the file from the submission package or, if the authors wish to attach it as supplementary information, re-format it without the AAAI class and correct 4.11 to "yes" and the dataset items to match the availability statement.

**F12. MINOR — Bibliography and typesetting hygiene.**
Evidence: `ref.bib` contains `dao2024flashlinearattention`, a duplicate of `gu2024mamba2` (same title, authors, year); six entries are uncited (`vaswani2017attention`, `merity2017pointer`, `biderman2023pythia`, `mambaforgekernel2024`, `marshall2011inequalities`, the duplicate); `booktitle` forms mix "Proceedings of the ..." and "Proc. ... Int. Conf."; Table III cites "Eq.~\ref{eq:chunk_rule}" (prints "Eq. 2") while every other place uses "\eqref" (prints "Eq. (2)"); `build/main.log` reports an underfull \hbox (badness 10000) in the proof of Proposition 1 (main.tex lines 388–393) and one underfull \vbox; the proposition carries two labels (`thm:nogain`, `prop:invariance`); `\label{sec:res_datacentre}` is placed mid-paragraph inside Sec. VII-I so that the supplement's "Section VII-I" reference for the H800 points to a paragraph, not a section.
Action (a): Remove the duplicate and uncited entries, normalize the booktitle abbreviation style to the IEEE form, use \eqref in Table III, fix the underfull box (the proof's displayed inequality should be set on its own line), and give the H800 replication its own paragraph heading or drop the extra label.

---

## 4. Venue compliance (IEEE TPDS, regular paper, double-anonymous)

| Item | Status | Evidence / note |
|---|---|---|
| Template / class | Compliant | `\documentclass[journal]{IEEEtran}` (main.tex line 5); supplement wrapper uses the same class (build.bat). |
| Page limit (14 formatted pages incl. references) | Compliant, zero slack | main.log: 14 pages. References on pp. 13–14. Any addition triggers overlength charges (F10). |
| Supplementary material | Compliant | Separate 16-page PDF (appendix_only.log), anonymized title block "Anonymous Author(s)", own IEEEtran bibliography, cross-referenced with "S" prefixes. |
| Abstract length / keywords | Compliant | Abstract 246 words (under 250); `\IEEEkeywords` present. |
| Author block / affiliations | Compliant | `\author{Anonymous Author(s)}` with a `\thanks` note stating names are withheld; `\markboth` uses "Anonymous". |
| Acknowledgments / funding | Compliant (keep as is) | The Acknowledgment section contains only the IEEE-required AI-use disclosure (Grammarly; Claude for code debugging and script writing); no names, institutions or grants. |
| Code/data availability statement | Present, but **MUST FIX** | Public Code Ocean capsule URL (`codeocean.com/capsule/3457679/tree`) is a de-anonymization vector (F1). Replace with an anonymized review link; restore DOI at camera-ready. |
| Self-identifying wording | **MUST FIX (verify)** | Sec. I: "for these kernels we built one, COREY". If an artifact, preprint or talk named COREY exists publicly under the authors' names, the name plus first-person attribution identifies them. Either confirm nothing public carries the name, or refer to the policy generically ("an entropy-driven scheduler") in the review version. No "our prior work"/"we previously showed" phrasing was found elsewhere. |
| Third-party identifiers | Check | Supp. S6 names the Hugging Face model `Idiap/gated-deltanet-swa-1.4B-30B`. Acceptable as a third-party workload; confirm the authors are not affiliated with that organization, otherwise cite it via a reference rather than inline. Legacy platform rows (RTX 3070 on WSL2) are not identifying. |
| PDF metadata | Compliant | `\pdfinfoomitdate=1` and `\pdftrailerid{}` in both main.tex and the supplement wrapper; hyperref is not loaded, so no `pdfauthor`. Re-check `/Author` and `/Title` with `pdfinfo` on the final PDFs before upload (no renderer was available here). |
| Reference style | Compliant | `\bibliographystyle{IEEEtran}` with the `cite` package; numeric citations. Hygiene items in F12 (duplicate entry, uncited entries, mixed "Proc." forms). Five arXiv-only references (Megatron-LM, LASP-2, Nemotron-H, MiniMax-01, Kimi Linear) are acceptable with arXiv identifiers. |
| Reproducibility checklist | **MUST FIX** | `ReproducibilityChecklist.tex` is an AAAI-2027 checklist (`\usepackage[submission]{aaai2027}`), not a TPDS artifact, is not `\input` into the paper, and answers "no" to statistical testing in contradiction with the paper (F11). Remove from the package or re-target and correct. |
| Figures | Compliant | All eight referenced PDFs exist under `paper/figs/` (five in the body, three in the supplement); build has 0 overfull boxes and no undefined references. |
| Build warnings | Minor | One underfull \hbox in the proof of Proposition 1 and one underfull \vbox (main.log); five "T1/ptm/m/scit undefined" font warnings in the supplement (cosmetic, small caps italic substituted). |

---

## 5. Strengths

1. **Measurement rigor well above the TPDS norm.** Chunk effects are timed inside one recurrence-preserving launch with output checks (Sec. VI-A, Supp. Table S-validation), with BH-corrected F tests, TOST equivalence, placement and warm/cold-L2 controls (Supp. S1.3), prompt-grouped cross-validation, two timing conventions, and replication on three GPU generations (Table VI).
2. **A cost model that needs no timing at the target shape.** Eq. (5) reaches 0.9%, 3.2% and 3.5% mean regret on three GPUs from 8–14 minutes of calibration per GPU, interpolates between calibrated shapes better than a nearest-key table (5.2% and 4.5%), and transfers its parameters across GPUs (2.5%, 5.1%, 3.8%) and to unseen widths (0.8% on the 2.8B checkpoint).
3. **Honest negative result with a clean decision rule.** The headroom-versus-decision-cost framing (Prop. 1, Eq. (7)) turns "should dispatch read the data?" into two measurable quantities, and the paper measures both, including the finding that two-percent re-timing noise alone explains most of the residual headroom (Supp. S8).
4. **Serving-trace replay ties kernel results to deployment.** The key-churn analysis over Azure and BurstGPT traces (Sec. VII-H, Supp. S9) is the kind of evidence TPDS readers expect and is rarely provided in kernel-tuning papers.
5. **Exemplary Limitations section.** Sec. IX states every scope restriction a reviewer would otherwise have to discover (single-kernel sensitivity, eager-mode Mamba-2, no multi-GPU runs, simulated scheduler, non-bandwidth-optimal histogram).

---

## Recommendation to peer reviewers

- Reviewer 1 (methodology): weigh F6 (term selection on the captured calls) and F8 (eager-mode Mamba-2); check whether the exact-shape "floor" and "held-out headroom" definitions in Table IV and Supp. S8 are used consistently across the three timing data sets.
- Reviewer 2 (domain): verify the related-work gap in F7 and whether any published runtime actually dispatches dense kernels on activation statistics (F4).
- Reviewer 3 (perspective): judge whether a TPDS reader gains enough from the single-GPU evidence (F3, F5) and whether the title should lead with the measured result rather than the "no-gain bound".
