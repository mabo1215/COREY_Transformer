# EIC Review Report: ACM TACO, Round 1

> Note added 2026-10-16: supplement numbers in this report (Table S..., Fig. S..., Sec. S...) refer to the reviewed 48-page supplement. The supplement has since been renumbered; see `taco_r1_supplement_numbering.md` for the current numbers. The report text itself is unchanged.

**Submission:** "Shape Sets the Chunk: A Compile-Informed Cost Model and a No-Gain Bound for Data-Dependent Dispatch of Chunked Linear-Recurrence Kernels" (main article, 23 pp. including references, plus a 48-page Supplementary Material)

**Reviewer role:** Editor-in-Chief / handling editor, ACM Transactions on Architecture and Code Optimization. I reviewed independently and saw no other reports.

**Materials examined:** `paper/main.tex` and `main.pdf`, `paper/appendix.tex` and `appendix.pdf` (reviewed as one paper), `paper/ref.bib`, and `paper/figs/`. I also spot-checked the selector and headroom result files that back Table 6 and the bucket-headroom numbers. Line numbers refer to `main.tex` unless they are prefixed with "Supp." (`appendix.tex`).

---

## 1. Summary of the paper

1. The paper studies chunk-size selection for chunked linear-recurrence GPU kernels: a Triton Mamba-1 scan written by the authors, the reference Mamba-1 CUDA scan, Mamba-2 SSD and FLA's Gated DeltaNet. It uses captured prefill calls on RTX PRO 6000, RTX 3090 (a modified 48 GB board) and H800 GPUs.
2. It reports that library defaults are adequate in eager execution, but under CUDA-graph replay the SSD and Gated DeltaNet defaults lose 16 to 22% (RTX 3090 only). On the authors' Triton scan, a single chunk loses 19 to 28%.
3. It proposes an 8-parameter cost model per GPU. Its inputs are compiler-reported registers, spills and Triton tile layout. It selects the chunk of the authors' Triton scan with a mean regret of 0.9 to 3.5%, or 1.3 to 7.3% when the terms are selected without the target GPU. The model does not transfer to TFLA, to the reference scan at fixed threads, or to production kernels under CUDA graphs.
4. It states an elementary "no-gain bound": a policy that reads activation values can gain at most the oracle headroom of a shape-keyed table. Measurements show that headroom to be tiny, and an entropy-driven scheduler is slower end to end.
5. It adds deployment analyses: break-even of profiling against online policies, key churn on replayed serving traces, tensor-parallel shard widths (synthetic inputs, one GPU), and a deployment recipe.

---

## 2. Recommendation

**Major Revision.** Confidence: **4/5.**

The measurement work is careful, extensive and unusually honest about its limits. As submitted, however, the paper is not yet a TACO paper:

- **Focus.** The contribution TACO readers would value is a compile-informed performance model and an auto-tuning cost/benefit analysis for a code-variant parameter. Most of the narrative and the title instead go to refuting a data-dependent policy that, by the authors' own account, nobody has proposed.
- **Generality.** The cost model is demonstrated on one kernel, the authors' own.
- **Readability.** The text is very hard to read: there are six timing conventions, and the abstract is a dense list of numbers.
- **Self-containment.** The main article depends heavily on a 48-page supplement.

There is also a small page-limit violation that must be fixed before the paper goes to referees. With refocusing, broader validation of the model and a rewrite for this audience, I can see a publishable TACO article. Without these, I would expect referees to return Reject.

---

## 3. Scores

| Criterion (weight) | Score (0-100) | Justification |
|---|---|---|
| **Novelty (20%)** | **50** | The no-gain bound is, in the authors' words, "elementary" (l. 421). Part (a) is min <= choice. Part (b) assumes data-obliviousness, fixed placement and operand-independent instruction timing, which is close to assuming the conclusion. The data-dependent policy it rules out has no proponents (l. 121-122). The cost model is a narrow, fitted variant of known analytical GPU models (Hong-Kim, Volkov, GPUMech; l. 481-490). What is new is feeding it Triton compile facts (registers, spills, tile layout) to rank one tile parameter. That is useful but incremental. |
| **Methodology (25%)** | **65** | **Strengths:** recurrence-preserving timing (Sec. 6.1), shuffled rounds, placement and L2 interventions, TOST equivalence, prompt-cluster bootstrap and a protocol table. **Weaknesses:** the model form was chosen on the test calls of 2 of 3 GPUs (l. 493-509), so only one GPU (H800) is form-blind. Headline launch-inclusive numbers "hold for one session" (l. 1355-1357); in a second session the cost model's regret rose from 0.88% to 4.9% (Supp. l. 1025-1029). Device-time timing is invalid for multi-kernel pipelines below the launch floor (CV 64-85%), so 83-91% of the SSD 2.7B and GDN calls are discarded. Clocks were unlocked on rented instances. The primary captured-call data were collected without the exclusive-GPU guard (Supp. Table S1). |
| **Experiments (25%)** | **68** | Coverage is broad: 3 GPUs, 4 kernel families, 3 model scales, TP widths, CUDA graphs, TFLA and a serving replay. It is uneven where it matters: CUDA-graph results exist on one consumer GPU only; the H800 covers Mamba-1 only; the compile-informed model is never applied to an independent Triton kernel; and top-k results are reported only for the proposed ranker (see E4). There is no stock-board replication of the RTX 3090 results, no multi-GPU run, and the serving study is a simulation. |
| **Clarity (15%)** | **45** | The abstract (l. 47-69) packs about 30 numbers and five timing qualifiers into 248 words. The contributions run to 465 words (l. 141-187). Six timing conventions are mixed across the main tables. There is heavy jargon ("launch floor", "headroom", "tied", "form-blind", "three criteria") and notation collides. The main text refers to 38 distinct supplementary items 46 times. The framing (prefill, checkpoints, LongBench, "requests of 48 layers") reads as written for an ML-systems venue. |
| **Significance (15%)** | **50** | There are practical takeaways: graph-mode defaults lose 16-22%; tuning caches should be keyed on compiler specializations; a model is a reasonable cold-start default. But the large chunk sensitivity belongs to the authors' own kernel, which is 1.23x slower than the reference scan (l. 587-591). Profiling all six chunks costs 0.03-0.05 s per key and repays the model's excess within a median of about 95 requests (l. 1303-1312). The authors themselves concede that "enumerable keys are cheaper to profile directly" (l. 1409). |
| **Weighted total** | **57.5 / 100** | 0.20(50) + 0.25(65) + 0.25(68) + 0.15(45) + 0.15(50) = 57.5. This maps to Major Revision. |

**Strengths, recorded for balance:**

- **S1. Measurement rigor.** The split-call pitfall (Sec. 6.1), recurrence-preserving timing, and the value-versus-placement and warm-versus-cold-L2 interventions (Supp. Tables S8, S17, S18) are exemplary and of methodological value to TACO readers.
- **S2. Specialization keys.** The observation that a single Triton specialization bit (length divisible by 16) changes the compiled kernel and its optimal chunk (l. 435-438, 553-565) is a concrete, transferable code-generation insight.
- **S3. Candor.** Negative and limiting results are reported openly: the model fails on TFLA, on the 32-thread reference scan and under graphs (l. 1107-1111, 1166, 1168-1176), and Section 9 is a thorough limitations section.
- **S4. Reproducibility documentation.** Supp. Table S1 (per-experiment protocol), the platform table, seeds and validation against an fp32 reference are well above average.

---

## 4. Venue compliance

| Item | Status |
|---|---|
| Template: acmart `acmsmall`, `review`, `anonymous`, `screen`; `\acmJournal{TACO}`; supplement also acmart | OK (l. 6-7). |
| **Page limit: 20 pages including text and figures, excluding references** | **MUST FIX.** In `main.pdf` the References heading starts on **page 21**. Pages 1-20 hold the body through the Conclusion. The "Disclosure of AI Tool Use" (l. 1459-1463, about 4 lines) sits on page 21 above References, so non-reference text runs about 4 lines over. Trim at least 6 lines so the disclosure ends on page 20, or confirm with the editorial office that it is exempt. Strict desk checks count it. |
| "New Paper, Not an Extension of a Conference Paper" first-page footnote | OK (l. 90; visible on p. 1). |
| Double-anonymous: author block, self-citations, PDF metadata | OK in both PDFs: Anonymous Author(s), no Author metadata field, no self-identifying citations, no code URLs. |
| Anonymity and venue residue in source files | **MUST FIX if LaTeX sources or an artifact are uploaded for review; otherwise SHOULD FIX.** `appendix.tex` l. 1-5 says "Supplementary Material for the IEEE TPDS submission" and "loads IEEEtran". All figure files are named `tpds_*.pdf` (e.g., l. 783, 1005). The result files that would make up an artifact archive contain host names, absolute user paths with the project name, and git commit hashes in their metadata blocks (seen in the graph-analysis result file). Scrub all of this from anything shared with referees. |
| CCS concepts and keywords | Present and acceptable (l. 71-86). Optional: "Parallel computing methodologies" is generic. A more specific architecture or performance concept would help TACO routing, e.g., "Computer systems organization -> Parallel architectures" or "Software and its engineering -> Software performance"; check the latter against CCS 2012. |
| Citation style | OK: ACM-Reference-Format, author-year (acmsmall default). |
| Running head | Minor: the long title wraps the odd-page running head onto two lines. Supply `\title[Short title]{...}`. |
| AI-use disclosure (ACM authorship policy) | Present. ACM expects it in the Acknowledgments; it can stay visible because it does not identify the authors. It must fit within the page limit (see above). |
| Artifact or data availability statement | **SHOULD FIX.** None in the paper (grep finds no availability, artifact or code statement). The study depends on a custom Triton kernel and a patched CUDA scan. Add an anonymized artifact statement and link; TACO supports ACM artifact badging. |
| Supplementary material | **SHOULD FIX.** 48 pages with 47 tables and 7 figures, against a 20-page article. TACO referees are not obliged to read supplements, and the main article is not self-contained (E6). Reduce the supplement substantially and remove the "legacy" material (E9). |
| Length for a revision | A revision may run to 25 pages including references; the current 23 leaves room, which helps E6 and E8. |

---

## 5. Major concerns

### E1. The paper is framed around a strawman; the TACO-relevant contribution is buried

**Location:** Title (l. 39-40); abstract (l. 63-68); Intro (l. 117-125: "to our knowledge, no published runtime or paper proposes value-dependent dispatch for dense chunked kernels"); Sec. 3 (l. 311-360); Sec. 4 (l. 362-446; "The bound is elementary", l. 421); Sec. 7.1, 7.3, 7.6, 7.7 (l. 710-1334); contribution bullet 3 (l. 168-178).

**Why it matters:**

- **Space.** About a third of the article, the title and half the abstract are spent refuting an entropy-driven scheduler that no one has proposed. Its rule is defined only in the supplement (Supp. Eq. S2).
- **The bound.** Proposition 4.1(a) is a one-line inequality. Part (b)'s hypotheses (data-oblivious streams, fixed buffer addresses, operand-independent instruction timing) make its conclusion nearly definitional, and Sec. 7.1 and Supp. Sec. S1.3 show the hypotheses fail in practice anyway (placement effects up to 43%).
- **Audience.** A TACO reader expects the center of gravity to be the model, the code-variant selection and the auto-tuning economics.

**Required action:**

- Reframe the paper around chunk/tile selection for chunked recurrence kernels: a compile-informed model, specialization-aware tuning keys, and when to model versus profile, across eager and graph execution.
- Demote the bound to a short Remark plus one table: shape plus entropy versus shape, and the decision cost.
- Move the routed study, histogram-cost break-even and entropy-statistics material to the supplement, or cut it.
- Retitle accordingly.
- If the authors want to keep the negative result central, cite a concrete prior proposal or practice of value-dependent dispatch for dense kernels that it refutes.

**Effort:** (a) text only.

### E2. The cost model's generality is undemonstrated; the "framework" is validated on one kernel, the authors' own

**Location:** Sec. 5 (l. 448-515), Table 2 (l. 517-537), "Applying the method to a new kernel" (l. 539-551); Sec. 6.2 (l. 583-591: the Triton kernel "written for this study", 1.23x the reference latency); TFLA paragraph (l. 1168-1176); reference scan at 32 threads (l. 1107-1111); graph mode (l. 1165-1166); Discussion recipe (l. 1385-1409).

**Why it matters:**

- **What was actually tested.** The compile-informed part (registers, spills, tile layout feeding occupancy and waves) was only ever applied to the authors' Triton scan.
- **Other kernels.** The reference-scan instance has a fixed register count and hand-chosen exchange terms, and loses 49.5% once the thread pairing changes. The SSD and Gated DeltaNet "instances" use no compile facts at all (Table 2: "none").
- **TFLA.** On TFLA, an independent Triton kernel with a wide chunk range, the paper fits a five-term model *without* compile facts. It loses 7.5% against the default's 1.4%, so the compile-informed method was never tested on an independent Triton kernel.
- **Chunk sensitivity.** The large sensitivity (8.5x spread, 19-28% static regret) that motivates the model is largely a property of this one kernel's register and spill behaviour. Since the kernel is slower than the production reference, this weakens the significance claim.

**Required action:**

- Apply steps (1)-(5) of l. 540-551 unchanged (compile facts, waves and occupancy, calibration on about 12 keys, form chosen without target calls) to at least two independent Triton kernels with a non-trivial tile/chunk space. Candidates: TFLA's mLSTM forward kernel (chunks 32-2048, already timed on the RTX 3090), FLA chunked kernels (e.g., GLA/simple-GLA with block size BT, BK, BV and num_warps), or the Triton sub-kernels of the Mamba-2 SSD pipeline at their block sizes.
- Report regret against the default, the best static chunk, a batch/length-keyed table and the simple rankers of Table 6.
- If generality cannot be shown, recast Sec. 5 as a case study and remove the framework and recipe claims (Table 2 title, l. 539-551, l. 1385-1409).

**Effort:** (b) for compile facts and refits on existing TFLA timings; (c) calibration and evaluation sweeps, about 0.5-1 GPU-hour per kernel per GPU, **about 3-6 GPU-hours** for two kernels on three GPUs.

### E3. Model selection on test data and session-dependent headline numbers

**Location:** l. 493-509 (terms chosen on the captured calls of the RTX PRO 6000 and RTX 3090; clean re-selection 1.3-7.3%, 95th percentile up to 63%); Table 6 (l. 943-976, launch-inclusive); l. 660-665 and 1354-1357 ("the launch-inclusive values of Table 6 hold for one session"); Supp. l. 1025-1029 (second session: shape tree 4.7%, cost model 4.9%, against 0.7% and 0.88%); abstract l. 57-59.

**Why it matters:**

- **Leakage.** The most-cited numbers (0.9% and 3.2%) are in-sample with respect to model-form selection.
- **Instability.** The form-blind re-selection is unstable across GPUs and criteria (Supp. Table S23), and only one GPU (H800, 3.5%) is genuinely form-blind.
- **Reproducibility.** The primary convention (launch-inclusive, single block) is not reproducible near the launch floor. A fivefold change in the cost model's regret between sessions on the same GPU undermines every launch-inclusive comparison in Table 6 and Fig. 3.

**Required action:**

1. Fix one selection protocol a priori (e.g., leave-one-GPU-out with a stated tie rule) and report it as the primary result in the main table.
2. Adopt one reproducible primary convention for every main table: device time with three shuffled rounds, or CUDA-graph replay. Show launch-inclusive results only as a sensitivity analysis, together with between-session variation.
3. Strongly recommended: one further, genuinely form-blind GPU (e.g., A100, H100 SXM, L40S or an AMD part via Triton) with the form frozen before measurement.

**Effort:** (b) for items 1-2, since device-time results for all selectors and all three GPUs already exist (Supp. Table S24); (c) for item 3: calibration about 15 min, plus the 2360 captured calls x 6 chunks x 3 rounds and the TP sweep, **about 3-5 GPU-hours**.

### E4. The deployment recipe is supported by a one-sided comparison

**Location:** Table 6 bottom rows (l. 971-974), Fig. 3 (l. 1003-1013), l. 994-1001, recipe "Dispatch" line (l. 1398-1401), Supp. Table S25.

**Why it matters:**

- **What the recipe claims.** It prescribes ranking chunks by Eq. 3 and profiling the best two, but top-k is reported only for the cost model.
- **What the results show.** In the authors' result files, other rankers under the same top-2 budget match or beat it:
  - RTX 3090, launch-inclusive: three-parameter restriction 1.75% against 2.30% for Eq. 3.
  - RTX PRO 6000: structural gradient boosting 0.53% and a length tree 0.67% against 0.52%.
  - Device time: several learned rankers reach at or below 0.2% where Eq. 3 does not.
- **Consequence.** The specific 8-parameter model is not shown to be needed for the recommended deployment.

**Required action:** Add top-2 and top-3 rows for every ranker to Table 6 or Fig. 3 (or a compact companion table), in the primary convention. Revise the recipe and the claim at l. 999-1001 to whatever the data support, e.g., "any calibrated ranker plus top-2 profiling".

**Effort:** (b) offline analysis of existing results.

### E5. Clarity, density and residue from the source venue

**Location:**

- Abstract l. 47-69: about 30 numbers in 248 words; qualifiers such as "(device time)", "(one GPU)", "(three criteria)", "(launch-inclusive; 95th percentile up to 63%)", "(synthetic inputs)".
- Contributions l. 139-187.
- Six timing conventions: launch-inclusive single block, launch-inclusive three rounds, device time single block, device time three rounds, CUDA graph, flushed L2.
- Mixed conventions inside single tables:
  - Table 4: Mamba-1 in device time, Mamba-2 launch-inclusive (l. 809-811).
  - Table 7: launch-inclusive reference scan next to device-time Triton columns (l. 1048-1050).
  - Table 8: mixed (l. 1248-1250).
- Notation collisions:
  - *P*: number of programs (l. 456) versus one-time profiling cost (l. 1198).
  - *N*: state size (l. 458) versus number of calls (l. 1199).
  - *k*: resident programs per SM (l. 466) versus key *k* (l. 1288) versus top-*k*.
  - *t*: TP degree (l. 560) versus *t_0*.
  - *pi_s(s)*: the shape key used as both subscript and argument (l. 378).
- Contradictory-looking statements across conventions: the best single batch-1 chunk on the RTX PRO 6000 is 256 in Table 4 but "512" in l. 1212 and l. 1342-1343, without saying the latter is launch-inclusive.
- ML-systems framing throughout ("requests of 48 layers", LongBench, serving-trace replay as a main-text result).

**Why it matters:** A TACO referee needs to extract the model, its inputs, its accuracy and its cost in one reading. At present every number needs three qualifiers to interpret, and adjacent tables are not comparable.

**Required action:**

- Rewrite the abstract to state the problem, the method and three to four key numbers in a single convention, ending with the takeaway.
- Cut the contributions to three bullets of at most 60 words each.
- Add a notation table and a short "metrics and conventions" box. Use one convention in all main tables (see E3).
- Remove the notation collisions.
- Rewrite the introduction for a code-optimization audience: what the compiler sees (tile, registers, spills, specializations), why the parameter is hard to tune (unbounded key space under TP and chunked prefill), and what the model buys relative to Triton's autotuner and `perf_model`/`prune_configs_by`.

**Effort:** (a) text only.

### E6. The main article is not self-contained

**Location:**

- 46 `\Sref` references to 38 distinct supplementary items.
- The data-dependent policy's rule is used in Table 4 (l. 807: "Supplementary Eq. (S2)") but defined only in the supplement.
- Evidence for abstract and conclusion claims lives only in the supplement:
  - CUDA-graph losses of 16-22%: l. 1160-1166 has two sentences; the table is Supp. Table S39.
  - Shard-width regrets of 2.9-6.5% and 65%: l. 1124-1141 against Supp. Table S31.
  - TFLA 7.5% and 1.4%: Supp. Table S34.
  - "Equivalent within 0.5 points on 15 of 17 data sets": Supp. Table S10.
  - Clean-selection results: Supp. Table S23.
- The supplement runs to 48 pages and 47 tables, including "legacy" material (Supp. Sec. S15.2, l. 2961-2975; Supp. Sec. S16).

**Why it matters:** A TACO referee should be able to verify every abstract claim from the article alone, and a 48-page supplement is an unreasonable burden.

**Required action:**

- Every claim in the abstract and conclusion must point to a main-text table or figure.
- Bring in, compactly, a graph-mode table, a TP-width table, a TFLA/generalization table (E2) and the clean-selection result (E3). Free the space by cutting the entropy-scheduler material (E1).
- Define the evaluated policy in the main text.
- Reduce the supplement to what reproduction needs, ideally at most about 20 pages: drop legacy and superseded data, and consolidate replication tables.

**Effort:** (a) and (b).

### E7. Several claims are stronger than the evidence

**Location and required rewording:**

1. **"In eager execution the defaults suffice"** (abstract l. 49-53, l. 1380-1383, Conclusion l. 1444-1445). For SSD 2.7B and Gated DeltaNet this rests on 397 of 2360 and 155 of 1770 calls (17% and 9%) above the launch floor. The remaining calls are discarded because device-time timing fails there (CV 64-74%; Supp. Table S33). At 370M the best static Mamba-2 chunk loses 1.0-2.0% (Table 4). State the covered fractions in the abstract, or measure all calls in graph mode (E8).
2. **"Needs no retuning" at shard widths** (abstract l. 60-61, l. 159-161, l. 1374-1375, Conclusion l. 1451-1452). The model loses 0.7-15% per shard on the RTX 3090 (95th percentile 31%), and on the H800 a table tuned at the calibration width does better (2.2% against 2.9%; l. 1137-1138). "The unsharded table loses up to 65%" compares against a baseline the paper itself calls unrealistic (l. 1132-1134). Reword to "avoids retuning at a mean cost of 2.9-6.5%" and compare with retuning at the shard width.
3. **"Activation statistics add nothing to the shape"** (abstract l. 67-68, Conclusion l. 1456-1457) is phrased universally. Scope it to the three dense kernels, three GPUs and data sets measured; two of 17 sets are inconclusive (l. 175-176).
4. **"The chunk moves latency by up to 8.5x"** (l. 147-148, 757) compares chunk 32 against chunk 512 at L=8192. Lead with the regret against the default or best static chunk, which is the decision-relevant figure.
5. **Measurement-guidance contribution** (l. 183-186). The split-call pitfall is presented as a contribution without any cited instance of it in the literature. Either cite the practice or present it as protocol rather than as a contribution.

**Effort:** (a) text only, except item 1, which is (c) via E8.

### E8. The practically most relevant regime, CUDA graphs, is under-measured

**Location:** l. 1160-1166 (RTX 3090 only), l. 666 ("On the RTX 3090 we also time CUDA-graph replays"), Limitations l. 1417-1419, Supp. Sec. S10.

**Why it matters:**

- **Relevance.** Serving engines replay captured CUDA graphs (the bibliography already holds a vLLM CUDA-graph reference, cited only in the supplement).
- **The finding.** Under graphs the production defaults lose 16-22% and the work-term model fails (12-22%). This is the finding with the clearest practical impact for TACO readers, yet it rests on one modified consumer GPU, is not replicated on the data-center H800, and the Triton-scan model was not refit in graph mode.
- **Methodology.** Graph replay also removes the measurement problem that forced the paper to discard most production-kernel calls (E7.1).

**Required action:**

- Replicate the graph-mode measurements on the RTX PRO 6000 and the H800 for SSD 370M and 2.7B, Gated DeltaNet 1.4B and the Triton Mamba-1 scan.
- Refit the cost and work-term models on graph-mode calibration and report them.
- Motivate in the main text why graph mode is the relevant regime for prefill in production.
- Consider making graph replay the primary convention (E3).

**Effort:** (c), about 2360 calls x 5-6 chunks x 3 rounds x 60 replays per model; with graph-capture overhead, about 1-2 GPU-hours per model per GPU. **About 10-20 GPU-hours** for four models on two GPUs, plus about 1 GPU-hour of calibration per GPU.

### E9. Platform validity, data provenance and internal consistency

**Location and issues:**

- **Platform:**
  - Clocks cannot be locked (l. 666-667).
  - The main replication GPU is a non-stock 48 GB RTX 3090 board (l. 606; Supp. l. 53-54).
  - The primary captured-call data (Tables 4-6) were collected without the exclusive-GPU guard (Supp. Table S1; l. 669-670: only "the later runs" use it).
  - Placement attribution rests on intervention because hardware counters were unavailable (l. 1425-1427).
- **Provenance:** Supp. Sec. S15.2 "Legacy measurements" (l. 2961-2975) describes a build "whose source was not retained" and session records that were "not retained either". Supp. Table S2 lists three "legacy" platforms.
- **Inconsistencies:**
  - Supp. l. 1452 ("Gated DeltaNet, for which no calls were captured") contradicts main l. 632-634 (1770 captured GDN calls).
  - Limitations l. 1422-1423 ("the reference CUDA scan was timed at one fixed thread count (32)") contradicts Sec. 7.5, which times the stock 32-128-thread pairing *and* a 32-thread build.
  - Supp. l. 725-726 says "the main paper summarizes the comparison in Table S19", but that table is in the supplement.

**Why it matters:** A TACO archival article must rest on retained, reproducible data on representative hardware.

**Required action:**

- Remove all measurements whose source or records were not retained, and all "legacy" sections.
- Reconcile the contradictions above.
- Report clock and power statistics where they were logged, and quantify their spread for the primary data.
- Justify the modified RTX 3090 board (memory clock and bandwidth relative to stock) or replace it.
- State explicitly which main-table data were collected without the guard, and re-time them under the guard if feasible.

**Effort:** (a) for removal and reconciliation; (c) optional re-timing under the guard or on a stock board, **about 5-10 GPU-hours**.

### E10. Positioning against the TACO, CGO and autotuning literature

**Location:** Sec. 2 "Kernel auto-tuning" (l. 249-267) and "Performance models" (l. 282-296); Sec. 7.4 baselines.

**Why it matters:**

- **Missing related work.** The paper cites ML-compiler cost models and classical analytical GPU models, but not the autotuning and code-variant-selection work most TACO referees will expect to see positioned. Examples: CLTune; the Kernel Tuning Toolkit (KTT); Kernel Tuner's Bayesian-optimization strategies; BaCO; portable GPU kernel-time models such as Braun et al. (ACM TACO); and runtime kernel selection (e.g., Seer, CGO 2024). Please verify these exact references.
- **Missing comparison.** Triton's `perf_model`/`prune_configs_by` mechanism is cited (l. 251-253, 998-1001) but never exercised as a baseline.

**Required action:**

- Add this positioning.
- If feasible, compare the model against budgeted search (random-k, nearest-key top-k, BO with k samples) at equal profiling budget. This overlaps with E4.

**Effort:** (a); (b) for budgeted-search comparisons on existing per-chunk timings.

### E11. Cost/benefit case for the model is weak as stated

**Location:** Sec. 7.4 auto-tuning (l. 1015-1026: profiling six chunks takes 1.2 s for 12 keys); key churn (l. 1303-1312: the model's excess repays profiling after a median of 4564/3688 calls, about 95/77 requests, and on 22/17 of 33 keys the model already picks the table's chunk); recipe l. 1406-1409.

**Why it matters:** For a six-value parameter, profiling is nearly free. The model's value is confined to keys that cannot be enumerated or timed, and the paper does not quantify how often those occur or what they cost.

**Required action:**

- Using the existing serving replay, report the number of distinct keys and the cold-start latency lost with and without the model, under realistic TP degrees and chunked-prefill settings.
- Also report the end-to-end effect relative to always profiling on first sight.
- If the benefit is marginal, say so and position the model mainly as a performance-understanding tool, i.e., what the compile facts explain.

**Effort:** (b) offline analysis of existing results.

---

## 6. Minor concerns

1. **Fig. 1 (l. 330-360):** the "launch" edge label overlaps the border of the "Read-back + chunk rule" box. Increase node distance.
2. **Table 2 (l. 517-537):** the "Regret (%)" column mixes regrets with "opt. 61/85", which is an optimum chunk, not a regret. It also mixes in-sample (terms selected on these calls) and form-blind entries. Split the columns or footnote them.
3. **Table 5 (l. 890):** "Exact-shape table, held out 0.1 / 0.0" has a 95th percentile below the mean. Correct, but explain it (a heavy single tail).
4. **Ties:** CI overlap is lenient (Supp. Table S3: up to 98% tied against 42% within 2%), yet "Tied" columns appear in Tables 3, 5 and 7. Report the share of calls within 2% of the oracle instead, or alongside.
5. **Tails:** report the maximum regret next to the 95th percentile for the cost model (e.g., 116% on the RTX PRO 6000 per the result files and Supp. Table S25). Tails matter for latency SLOs.
6. **Proposition 4.1(b):** call it a Remark, or say explicitly that its conclusion follows from its hypotheses. Rename `pi_s` to avoid using *s* as both index and argument.
7. **Eq. (4) (l. 1202-1205):** define *P* and *P_online* without colliding with the program count *P* (l. 456).
8. **l. 1210-1214:** two definitions of "best single chunk" (lowest mean latency: 512; lowest mean regret: 256) in adjacent clauses. Keep one.
9. **l. 1348:** `\label{sec:res_datacentre}` sits mid-paragraph. Make the H800 a proper subsection or drop the label.
10. **Platform specifics:** give the exact RTX PRO 6000 variant (Workstation, Server or Max-Q; power limits differ), the memory configuration of the modified RTX 3090, and why mamba_ssm differs across platforms (2.2.2 local build against 2.3.2.post1; l. 608-610).
11. **Synthetic batch construction (l. 623-627):** a batch-16 call holds 2-10 distinct prompts and repeats prompts. Discuss whether this biases the information tests (repeated members raise within-call homogeneity).
12. **TP study (l. 1124-1128):** "we run no multi-GPU job" is acceptable for per-device shapes. Add one sentence on co-running communication kernels (overlapped all-reduce) that compete for SMs and L2 in real TP deployments.
13. **Bibliography:** several entries are arXiv-only where archival versions may exist (e.g., LASP-2). Check and update. Web references (XLA, nvFuser, FLA, Triton docs, Qwen3-Next) are acceptable but should carry access dates.
14. **Captions:** Tables 4-7 have 6-12-line captions that restate methodology. Move protocol detail to the conventions box (E5).
15. **Conclusion (l. 1439-1457):** it largely repeats the abstract's numbers. Answer the research questions directly and state the transferable lessons for kernel and autotuner designers.
16. **Supplement wording:** Supp. Sec. S1.1 (l. 43-45) notes measurements "collected before this protocol was fixed". Remove such history-revealing phrasing in a journal submission.
17. **Third-party checkpoint:** the supplement names a public third-party checkpoint by its hub identifier (Supp. l. 1869). That is acceptable. Confirm it does not inadvertently point to the authors' institution.

---

## 7. What would be needed for Accept

1. **Compliance:** fit the article within 20 pages excluding references; scrub venue and identity residue from sources, figure names and any artifact; add an anonymized artifact statement.
2. **Refocus (E1, E5):** make the paper about compile-informed chunk/tile selection and tuning economics for chunked recurrence kernels in eager and graph execution. Reduce the data-dependent no-gain material to a short remark, rewrite the abstract and introduction for TACO readers, and use one timing convention and consistent notation.
3. **Generality of the model (E2, E3):** show the unchanged method on at least two independent Triton kernels and at least one additional form-blind GPU, with the model form fixed a priori. Otherwise, recast it explicitly as a single-kernel case study with correspondingly modest claims.
4. **Graph mode (E8):** replicate on a data-center GPU and refit the models.
5. **Fair recipe evidence (E4, E10, E11):** top-k for all rankers, comparison with budgeted search, and a quantified cold-start benefit.
6. **Self-containment and provenance (E6, E9):** every headline claim backed by a main-text table or figure; a supplement of about 20 pages or less, with no legacy or unretained data; internal contradictions resolved.
7. **Proportionate claims (E7).**

With items 1, 2, 4 (or a clearly scoped limitation), 5, 6 and 7, plus either item 3 or an honest case-study framing, I would expect a Minor Revision outcome.

---

### Notes for the referees

- **Reviewer focused on performance modeling:** please assess whether Eq. 3 and its compile-fact inputs generalize beyond the authors' kernel (E2), and whether the selection protocol is leakage-free (E3).
- **Reviewer focused on GPU measurement:** please scrutinize the timing conventions, the launch-floor handling, the use of a modified board and unlocked clocks, and graph-mode representativeness (E3, E8, E9).
- **All referees:** please judge whether the main article stands without the supplement (E6).
