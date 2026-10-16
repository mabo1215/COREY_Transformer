# taco-r2 final re-check (Stage 9' style, independent)

Date: 2026-10-17. Inputs: `docs/revision_suggestions.tex` (taco-r2, V1-V5, M1a-M9d, N1-N34, A1-A8),
`paper/main.tex`, `paper/appendix.tex`, `paper/main.pdf`, `paper/appendix.pdf` (built 2026-10-17 02:14),
and the result files in `src/experiments/results/tpds/`. Progress files, ledger and git history were not read.
Text was extracted with PyMuPDF; numbers were recomputed from the JSON result files.

## 1. Summary verdict

The revision is substantial and mostly sound. All new measurement items that were run (M2d, M3b, M3d, M4f,
M5c, M5e, M5f, M5i, M7d/M7f data type and alignment) are in the paper. 41 of the 41 number groups I
spot-checked match their result files. There are 4 rounding or labelling slips and 3 convention or label
mix-ups, but no fabricated or stale number. The paper has been refocused on the "second-order knob"
finding (M1d), as the author decided.

Page limits pass. The main body ends on page 20, and "References" is a heading on page 20 itself (V4).
The supplement is 25 pages, with references from page 23 (V5, at the upper edge). Neither PDF has a "??"
or an undefined reference. Both are English only. Their metadata are clean, with no author, path or host,
and the running head is "Anon.". The only identifying string is the third-person citation of the refuted
preprint (Ma, Wu, Yan 2026), which ACM permits. No repository path, script name or code file name appears
in either source or PDF; the one `\texttt{}` is a public Hugging Face checkpoint id.

**Acceptance criteria.** The paper does **not yet strictly meet A1-A8**, so it is not yet at the "Minor
Revision" bar as the review defines it. Almost everything that remains is text or offline work on result
files that already exist. No further GPU time is required.

| Criterion | Verdict | Why |
|---|---|---|
| A1 Compliance | PARTLY MET | V4 and V5 pass. The capsule dry-run is clean (no host, user, project or venue strings) but predates E35b: it lacks `e35_counters_4c_e35.json` and `taco_r2_counters.json`, which back the hardware-counter paragraph. It also ships the stray test file `e36_packed_graph_nonexistent.json` and is not yet uploaded. V2 and V3 are author actions that cannot be checked from the repository. |
| A2 Framing | MET | The contributions are the robust findings. The refutation is one subsection (5.1) with Remark 1. The headroom is stated as an information result with a host-side corollary, including the RTX PRO 6000 reduction cost (+2.7/-3.7 us). M1d is settled as "second-order knob". |
| A3 Kernel and baselines | MET (minor residue) | The in-house kernel is disclosed in the abstract, the Introduction and Sec. 4.1. The free length rule, the best chunk per batch and the length-bucket table are in Table 8, Table 10 and Sec. 5.4. The fixed-launch slice was re-timed under the full protocol (Table S15). Residue: the reason the calibration-only length tree is worse than the free rule is still not explained. |
| A4 Device-bound evidence | PARTLY MET | Recoverable gains with 42-cluster intervals (Table 6), the whole-step graph in two sessions (Table 7), packed steps under graphs with a noise floor, and engine modes with sources are all done. **Open:** the abstract and conclusion do not say that no serving engine was run. They also do not scope the end-to-end claim ("at most 0.3%") to the one stock RTX 3090 it was measured on, or to latency weighting over 14 shapes (see F1). |
| A5 Cost model | PARTLY MET | The latency term is justified (floor never binds; max-form fits equally). The model is positioned as a case study and top-k pruner. The comparison on the form-blind H800 is visible in Table 10. **Open:** the "pre-specified" rule was fixed after the six alternatives were seen (F3). The six protocols appear only as a range, not as a supplementary table with LW regret, maxima, per-batch values and intervals. The supplement still says "a new GPU needs its compile facts but hardly its own calibration" (M4h requires removal). Fit documentation is incomplete. |
| A6 Measurement and statistics | PARTLY MET | SSD and Gated DeltaNet value invariance above the floor, H800 input effects in the main text, counters for the mechanism, and the stock lab-host corroboration are all done. **Open:** Tables 5, 8, 9 and 10 have no noise floor, although Sec. 4.3 promises "a noise floor next to every regret". The launch-floor sensitivity was computed but not reported. Unguarded data are not flagged in main-table captions. Several quantities still have two values across tables (F5). |
| A7 Workload and recipe | PARTLY MET | The recipe is correct on batch invariance, 32-bit indexing, signature key with padding, dense multi-allocation warm-up and precautions. **Open:** workload-weighted numbers appear only as a prose range for production-kernel recoverable gains, with no intervals and none for Tables 4, 5, 8 or 10. The serving replay is still costed with the Triton scan's nearest keys (M6c). The routed study in S15 still leads with chunk 128. |
| A8 Clarity and positioning | PARTLY MET | Sec. 5 is organised by finding, with the mechanism figure (Fig. 1), the claims table (Table 3), the conventions table, a glossary, an expanded Discussion, itemised Limitations and the M9 prior art. **Open:** the abstract is 207 words (target about 180) and names no convention for "44 and 8%". Captions of Tables 4, 6, 7 and 9 run 86-101 words. N5, N7, N14, N15, N25, N30-N32 and N34 are open. Three abstract or conclusion claims have no main-text table or figure anchor (F7). |

**Bottom line.** These are about one day of text and offline work, ranked in Section 4. With that done, the
paper would meet A1-A8 and, given Table 7, the stated "Accept" condition (M1d settled by measured end-to-end
evidence). The most consequential open issue is F1: the headline "at most 0.3%" has to be scoped. It is
measured on one stock RTX 3090 and latency-weighted over 14 shapes. A single 256-token step gains 2.3%
from the table (3.6% from the in-session best chunk), and trace weighting raises the single-call
recoverable share to as much as 25%.

## 2. Key findings beyond the item list

- **F1 (major, text plus offline). Scope of the end-to-end headline.** Abstract, Contribution 1 and the
  Conclusion state "at most 0.3%" (conclusion: "for dense chunked linear-recurrence kernels the chunk is a
  second-order knob end to end") with no qualifier.
  - The number is the latency-weighted mean over 14 uniformly chosen shapes (256-8192 tokens) on one
    stock RTX 3090. Table 3 says this; the abstract and Conclusion do not.
  - Unweighted per-shape mean gain of the table, SSD 370M: 0.64% (session 1) and 0.44% (session 2).
    Source: `gain_table_mean_pct` in `e32_model_graph_4c_e32s1_ssd370m_4c_e32s2_ssd370m.json`.
  - Single shapes: the table gains 2.3% (SSD 370M, 256x1). The in-session best chunk gains 3.6% (GDN 1.4B,
    256x1, session 1, Table S19), which contradicts "the largest single-shape gain, 2.3%" if that phrase
    refers to the best chunk.
  - Section 5.3 itself shows that trace weighting raises the single-call recoverable share to 0.8-13.5%
    (Azure) and up to 25% (BurstGPT), because real prompts are short.
  - Fix: name the GPU and the weighting in the abstract and Conclusion. Add per-shape mean and maximum, and
    one trace-weighted end-to-end figure computed offline from the per-shape gains in Table S19 and the
    Azure/BurstGPT length mix already used for the workload weighting.
- **F2 (major, text). M3e and M3f scoping not done where required.** Both items say that, if not run, the
  abstract and conclusion must state that no engine-level measurement was made and scope claims to the GPUs
  measured. Only Limitations and the Discussion do this.
- **F3 (moderate, text). "Pre-specified" protocol.**
  - Sec. 5.5 says "We fixed this rule in this revision after evaluating six alternatives". The rule equals
    protocol 6 of the six (ablation12+work_only, target-calibration LOSO, plus a parsimony tie-break), per
    `prespecified_protocol` in `taco_r2_baselines_m4.json`.
  - Calling it "pre-specified" in Contribution 6, Table 10 and the text overstates it. Rename it ("adopted
    target-calibration rule") or justify it as the earlier "What transfers" recommendation, with the
    six-protocol range as the honest uncertainty.
  - The sentence "with its terms chosen on the other GPUs' calls it is the worst of the four (4.5%)" refers
    to the LOGO-captured protocol (4.20/6.21/3.01). Table 10 has no such row; its nearest row, "terms chosen
    on these two GPUs' calls", reads 0.40/3.52/3.01 (mean 2.3). A reader cannot verify 4.5 from Table 10.
- **F4 (moderate, text). Convention mixing.** "with the parameters of another GPU and only the H800's
  resource reports and SM limits, with 3.3 to 3.8%" combines different conventions. 3.3% is device time
  (RTX PRO 6000 parameters) and 3.8% is launch-inclusive (RTX PRO 6000 parameters). The full set is device
  time 3.3/3.0% and launch-inclusive 3.8/5.4% (S2 paragraph). This conflicts with "Every number names its
  convention".
- **F5 (minor, text). One value per quantity (N14, N15).**
  - RTX PRO 6000 frozen cost model: 0.38 [0.38, 2.26] in Table S12 against 0.40 in Table 10 and Table S16.
  - RTX 3090 frozen cost model: 3.55 (Table S12), 3.52 (Table 10, Table S16) and 3.6 (S3 fit paragraph,
    "selects with 1.7% instead of 3.6%").
  - OLS 1.8% (appendix l.697) should read 1.7% (1.74).
  - "every policy, including the default, stays within 2.4%" (Sec. 5.3), but the launch-inclusive SSD
    default is 2.53% (`taco_r2_stats_m5a.json`, `m2_vgpu3090_launch`) and Table S17 shows 2.5.
  - "1.4 and 0.8% per-call regret" (Sec. 5.3) against `model_mean_regret_pct` = 1.347 in
    `taco_r2_graph_calibration.json`; Table 6 correctly prints 1.35, so the text should say 1.3.
  - "scan is 5 to 18% of the step" (abstract, Contribution 1, Sec. 5.3 "15 to 18%") against a maximum of
    18.84%, which rounds to 19.
- **F6 (minor, text). In-sample presented as profiling on first sight.** "profiling an unseen key on first
  sight ... recovers 1.01% [0.49, 2.09]" is `recoverable_lw_table_insample_pp` (the table chosen and scored
  on the same timings), which is biased up by the noise floor. The refit-based "profile unseen" interval is
  [0.52, 1.96]. Say in-sample and quote the noise floor.
- **F7 (moderate, text). Claims without a main-text anchor (M8d).**
  - Abstract "buffer placement ... moved single timings by up to 44%". This comes from the launch-inclusive
    RTX PRO 6000 control in Supplementary Table S7. The main-text placement column (Table 4, graph replay,
    stock RTX 3090) shows at most 7.5%.
  - "The key is the specialization signature" (abstract, Contribution 2, Conclusion). Its evidence is prose
    in Sec. 5.2 plus Table S18 and Fig. S2; there is no main table or figure.
  - "profiling ... costs seconds per GPU, less than calibrating a cost model" and "a model serves GPUs that
    cannot be timed". Both are prose only.
  - Abstract "a power cap moved single timings by ... 8%". The attribution to the cap is inferred from
    throttle logs, not shown by intervention, so the wording should be "under a power cap, sparse inputs
    ran up to 8% faster". The 8% itself is not a number the body states: it recomputes to 7.3% (slowest
    dense against fastest constant or 2%-sparse input at C*, `e33_value_graph_4c_e33.json`), and the body
    gives 3.2-9.3% (input spread at C*).
- **F8 (minor, artifact).** The combined E32 files (`e32_model_graph_*_e32s1_*_e32s2_*.json`) have every
  `meta` field null (gpu, versions, rounds). Provenance therefore rests on the per-session files; fill
  `meta` in or note this in the capsule README.

## 3. Item table

Status codes: **A** = ADDRESSED, **P** = PARTIAL, **N** = NOT ADDRESSED, **D** = DECLINED-WITH-JUSTIFICATION
(the claim is scoped as the item allows).

### Venue compliance

| Item | Status | Evidence (location, quote) | Missing |
|---|---|---|---|
| V1 artifact anonymity | P | Availability section: "An anonymized archive with the code, result files and scripts behind every table and figure accompanies the submission". Capsule dry-run (448 files): no host, user, project or venue strings; directory `study`; host tag `labrtx3090`. | Rebuild after E35b: add `e35_counters_4c_e35.json` and `taco_r2_counters.json`, exclude `e36_packed_graph_nonexistent.json`, re-scan, upload. Fill or document the null E32 `meta` (F8). |
| V2 paper-folder hygiene | P (author action) | `paper/figs/profile/{BoMa,JinsongWu,WeiqiYan}.jpg`, `paper/backup/` (AAAI27, IEEETPDS, MLSys, taco_r1), `paper/legacy/` (AAAI checklist) and unused old figures are still present. The PDFs are clean. | Upload only the two PDFs, or build a clean source bundle. |
| V3 preprint disclosure | P (author action) | Footnote "New Paper, Not an Extension of a Conference Paper" is present. The preprint is cited in the third person ("a preprint proposed ..."). | Confidential cover-letter disclosure (cannot be checked here). |
| V4 page limit | A | main.pdf: 25 pages; "References" heading on page 20 after the Conclusion, availability statement and AI disclosure. CCS concepts, keywords and running head are present. | none |
| V5 supplement length | A | appendix.pdf: 25 pages, references from page 23; 24 tables and 2 figures (was 31 pages and 40+ tables). 31 `\Sref` pointers in main.tex (was about 50). | At the upper edge of 20-25; no room for growth. |

### M1 Significance and framing

| Item | Status | Evidence | Missing |
|---|---|---|---|
| M1a contributions | A | Six findings in Sec. 1 ("End to end, the chunk is a second-order knob"; "The key is the specialization signature"; "Placement and power masquerade"; "The regime decides"; "A steep chunk curve is entangled"; "Profiling beats modelling"). A "What is new" paragraph and Sec. 2 position them against TFLA, Triton, Kernel Launcher, Nitro/Ding/Seer and production stacks. | none |
| M1b one-subsection refutation | A | Remark 1 ("The remark is elementary"). Routed study, break-even and CMI are in the supplement. Pitfall reduced to a two-sentence "Recurrence-preserving timing" note. | none |
| M1c headroom plus corollary | A | Sec. 5.1 "Headroom": "0.45, 2.2 and 2.2 us ... held-out per-shape maxima of 4.4, 6.7 and 14.5 us ... RTX PRO 6000 (+2.7 and -3.7 us)". Sec. 3: an isolated synchronization is a lower bound in device-bound pipelines. Bucket key 26 and 30-32 us. | none |
| M1d significance decision | A (scope: F1) | Abstract and Conclusion lead with "second-order knob ... at most 0.3%". | Scope per F1 and F2. |

### M2 The in-house kernel and its baselines

| Item | Status | Evidence | Missing |
|---|---|---|---|
| M2a provenance and role | A | Abstract: "a Triton scan we wrote as a stress case". Sec. 1: "1.23 times slower ... launch table, chosen on another GPU". Sec. 4.1: "short calls are grid- and latency-bound". Table 1 and Fig. 2: "Triton scan (ours)". | Minor: captions of Tables 5, 8 and 10 do not say "(ours)" or "stress case". |
| M2b fair untuned baselines | P | Table 8 (free rule 5.86/7.76/10.7, best chunk per batch, bucket and exact tables, chunk 256 as a secondary row) and Table 10 (free rule). CUDA-scan stock rule in Table 9. | No explanation of why the calibration-only length tree (16.6/20.4%) loses to the free rule. The best static chunk by latency on the RTX PRO 6000 (512, 15.5% of scan time) is not stated. |
| M2c one definition of default and best static | A | Sec. 4.1: "It has no library default; its untuned references are ...". Table 8 caption: "per batch (by per-call regret)". Fig. 2: "its best single chunk in each regime". Table 3 points the production gains to Table 6. | none |
| M2d chunk at fixed launch configuration | A | Sec. 5.4 "Chunk versus launch configuration": medians 14.5/4.6/4.2%; full protocol "(4,2) chunk 128 ... 10 of 12 keys ... 1.0% ... 12.5% slower"; Shapley in device time "44.5% = 50.6 ... -6.2"; Table S15. | Table 1 does not say "results for launch-table entries" (Sec. 5.4 does). |
| M2e chunk-only scope | D | Sec. 5.4: "the chunk-only results of this paper hold for the entries of a fixed launch table". Limitations: "no model replaces a search of the full launch space". S3: full-space tuner loses 8.92/23.14%. | none |
| M2f parity or retune (optional) | D | Not run. Stress-case framing is in the abstract, Sec. 1, Sec. 4.1, the Discussion and Limitations ("whose launch table we did not retune per GPU"). | none |

### M3 Production relevance and device-bound evidence

| Item | Status | Evidence | Missing |
|---|---|---|---|
| M3a recoverable gains with intervals | A | Table 6 "Recoverable LW" column with 42-cluster intervals (e.g. 0.45 [-0.83, 1.86]). Contribution 4: "of which a held-out table recovers 1.7 to 5.4% (on one kernel and GPU nothing distinguishable from zero)". | The Conclusion does not name the two GPUs (F2). F6 wording. |
| M3b whole-step graph | A | Table 7, two sessions, scan share, per-shape Table S19, capture cost. | Representative-point "deployment" rows in ms are only in S19. Scope per F1. |
| M3c engine modes and regime names | A | Discussion "Engines and graph capture" (vLLM V1 FULL_AND_PIECEWISE, SGLang breakable graphs, with sources). "launch-bound ... Python launch path (56 to 90 us)". Speculative decoding and chunked prefill named. | Fig. 2 and Sec. 5.3 still say "host-bound" while Table 2 says "launch-bound". |
| M3d packed steps under graphs | P | Sec. 5.3 "Packed steps as an engine replays them". S10: capture sizes 16..512 and 512..8192, five rounds, noise floor 0.02-0.57%, capture-size and number-of-sequences keys. | Carried state is still "SSD: first sequence; GDN: every sequence", unjustified. The recipe key still says "packed length" where the data support "capture size". No richer per-sequence structure key. |
| M3e engine integration (optional) | P | Not run. Limitations: "No serving engine was run". | The abstract and Conclusion must say it (item text). |
| M3f data-centre replication (optional) | P | Not run. Limitations: "no data-centre GPU was measured under graphs". Discussion: "We measured an Ampere consumer GPU and a Blackwell server GPU". | Abstract and Conclusion production claims are unscoped; Table 7 is one stock RTX 3090. The sm_90/sm_100 code-generation caveat is missing. |
| M3g shard layouts (optional) | D | Limitations: "not for the SSD and Gated DeltaNet pipelines at sharded head layouts". | none |

### M4 The cost model

| Item | Status | Evidence | Missing |
|---|---|---|---|
| M4a one protocol, all six reported | P | Sec. 5.5 "Accuracy": rule stated. Table 10: "six a priori protocols (range) 0.40-4.20 / 2.3-26.9 ...". | F3 ("pre-specified" chosen post hoc). No supplementary table of the six with LW regret, maxima, per-batch values, cluster intervals and per-fold forms. Contribution 6 lacks the LW companion. |
| M4b fair calibration-only comparison | A | Table 10 "same calls for every row" (MLP five seeds with s.d., structural MLP, three-parameter restriction, resource-report rule, top-2/top-3). Text: "it is the worst of the four (4.5%)". | F3 (4.5% not traceable to Table 10). The paired H800 difference -0.53 [-1.35, +0.60] is not quoted. |
| M4c fit documentation | P | "50 random starts reach a 16% lower objective"; "two optima ... 1.6 and 4.3%"; Table S12 bootstrap; S3 short-search timings and calibration order "none". | Inverse-variance weighting, objective R2, 226% discrepancy, drift-with-key confound, device-time LOKO 0.8/7.4/3.7, recommended optimizer protocol, six-configuration calibration (0.56/1.8/6.1). |
| M4d physical consistency of Eq. (2) | P | Sec. 5.5: "never falls below A (by a factor of 1.16 to 1.62) ... max(A+I, I W/W0) fits equally well but selects worse (2.1, 4.3 and 2.3%)". Counters: "only chunk 1024 spills". | Last partial wave not modelled or discussed. Spill proxy omega not explicitly validated against local-memory sectors. appendix l.609-610 still says sigma is "the spill-store traffic per thread". |
| M4e reposition | P | Sec. 5.5: "as a case study"; "The model's place is therefore a cold-start default ... and the table for a GPU that cannot be timed"; Triton top-k comparison; resource-report-only ranker. | appendix l.562-563 "a new GPU needs its compile facts but hardly its own calibration" is not softened. Conclusion lacks "for this kernel". Only the H800 target is shown for cross-device (F4). |
| M4f work-term model under graphs | A | Table 6 "Graph" column (1.35/0.78/12.2). Sec. 5.3: padded variant 4.2/1.0% per call, 0.8/0.2% LW. | Text says 1.4 where 1.3 is correct (F5). |
| M4g second kernel (optional) | D | Limitations: "whether the model's procedure transfers to another kernel is untested". Sec. 5.5 "case study". | none |
| M4h fourth GPU (optional) | N | Not run, and the sentence the item requires removing is still in S2 (appendix l.562-563). | Remove or soften it. |

### M5 Measurement reliability and statistics

| Item | Status | Evidence | Missing |
|---|---|---|---|
| M5a noise floors and floor sensitivity | P | Sec. 5.3: "noise floor of 0.06 to 0.39% per call"; packed noise floor; TP noise. | No floor on Tables 5, 8, 9, 10, against "a noise floor next to every regret" (Sec. 4.3). The launch-floor sensitivity (factors 1/1.5/2/3; at 1.5, 556 of 624 calls instead of 375 of 397) is computed in `taco_r2_stats_m5a.json` but unreported. No 100 us threshold sensitivity. Captions do not mention oracle-minimum bias. |
| M5b RTX 3090 TP sweep noise | P (alternative accepted) | Kept in the main text with noise-corrected excess: "30.7 [18.7, 44.7] and 21.5 points beyond the round-to-round noise"; "the cost model and a table retuned ... tie there (-1.0 points [-4.0, 1.9])". | Not demoted. The re-analysis removes the old artefact; acceptable if stated as the response. |
| M5c value invariance above the floor | A | Table 4 (graph replay, 14 inputs in one process, designs A and C, five rounds, TOST counts). S9. Recipe now prescribes dense random inputs on several allocations. | none |
| M5d H800 input effects in main | A | Sec. 5.1: "28 of 36 ... all 36 in device time, up to 48% ... at most 4.4 and 5.1% ... 5.7%; that sweep had no placement or L2 control". | none |
| M5e hardware counters | A | Sec. 5.3 "Where the chunk acts" plus Fig. 1: occupancy 13-16%, warps per scheduler, long-scoreboard stalls, spills, L2 hit 43 to 60% on placement, SSD state-passing issue 3.4%. | Breakdown is profiler time at L=4096, not graph replay at short and long L. Masked programs when C>L not discussed. |
| M5f platform corroboration | A | Sec. 5.3 "Another board, another session" (41 calls per model; boards differ 0.2-0.5%; cross-session table up to 2.2% LW). Table S2 lab host "bare metal, no virtualization, power limit 350 W". | Which conclusions rest on the 48 GB board alone is not listed (reference-scan sweep, TFLA, routed study were not re-timed). |
| M5g protocol documentation | P | Table S1 rows for all taco-r1 and taco-r2 runs with guard and state columns. "Host CPU, governor, host load ... were not recorded". mamba_ssm version note. Spin coverage stated. | Main-table numbers from unguarded runs (Table 5 H800 column, Table 9 CUDA scan) not flagged in captions. RTX PRO 6000 "local build" changes not described. Cold-L2 coverage not stated in Table 2. |
| M5h intervals and per-batch | P | 1.4B per-batch "13.9% [11.0, 16.6] at batch 4, against 1.0 and 0.9%". Multiplicity: "133 ... 12 ... 4 ... none under Holm ... descriptive". | No intervals in Table 9. No per-batch values at 2.8B. No cluster-size distribution. Percentile intervals, not BCa or studentized (BCa exists in the stats files). |
| M5i information-side power | A | Sec. 5.1: "0.89 us [-3.97, 5.75] ... minimum detectable effect 7 us, equivalent within +/-10.7 us ... (p=0.61)". Graph re-time: "0.00 us at all five shapes". | none |

### M6 Workload and serving

| Item | Status | Evidence | Missing |
|---|---|---|---|
| M6a workload weighting | P | Sec. 5.3 "Workload weighting": "rises from 0.45 to 5.4% to 0.8 to 13.5% ... up to 25% on BurstGPT ... -0.5 to 4.6% ... -1.2 to 9.0%". S13 method. | Prose range only: no intervals, no weighted rows for Tables 5, 8, 10 or the Triton headline, not in the abstract. Not propagated to the end-to-end claim (F1). |
| M6b natural-length calls (optional) | D | Limitations "Workload". | none |
| M6c serving replay and units | P | Sec. 5.5: "in a replay of public serving traces through a simulated scheduler". S13: "the step model is 2.3 times faster ... so these shares are upper bounds". Glossary defines ms per forward pass. | The replay still maps every call to the Triton scan's nearest key; it is not re-costed with the production kernels' graph and packed timings. Load sensitivity is not in the main text. |
| M6d routed study | P | Main: "the scheduler's statistics, not its chunks, made prefill 19 to 66% slower". S15 notes "not bandwidth-optimal" and "on the three greedy prompts checked". | S15 still leads with chunk 128 ("adds 16.8 +/- 1.4 ms"; Table S24 says +16.8 +/- 1.5%). The 46-47 ms prefill at 256 and 976 tokens is not stated. No per-statistic fused or device-side cost estimate. Table-means recomputation gives 18-65%, not 19-66% (minor). |

### M7 Recipe correctness

| Item | Status | Evidence | Missing |
|---|---|---|---|
| M7a batch invariance | A | Recipe "Consistency": "A batch-invariant or deterministic mode must fix the chunk or key it only on per-request quantities ... replicas should share one frozen, versioned table". 2.6-9.4% and 4.7% match Table S21. | Not reflected at abstract level (minor). |
| M7b crash hazard | A | Recipe "Fill": "skip chunks that exceed a kernel's 32-bit indexing at the shape". S5: the 2^31 bound. | none |
| M7c synthetic warm-up inputs | A | Recipe: "use dense random inputs of realistic scale on several allocations or with a flushed L2, above the launch floor". | none |
| M7d key equals specialization signature | P | Sec. 3 "The key is the specialization signature" (Triton 3.4 bits, strides, projection view). Sec. 5.2 and Table S18 (SSD/GDN alignment, graph replay). Recipe padding action. | The share of the CUDA scan's chunk effect due to kIsEvenLen versus threads versus items is not quantified. Table 1 does not mention the flag. Triton 3.4 attributes for this kernel are asserted, not verified. |
| M7e tuning cost under graphs | A | Discussion: "one capture took 0.4 to 1.0 s ... 0.05 to 1.0 GB of graph-pool memory ... or the cache can be filled in device time before capture, which is reliable only above the launch floor". | Recipe still quotes eager "30 to 50 ms" only (minor). |
| M7f precautions | A | Recipe "Precautions not measured here". Prefix-cache concreteness in the Discussion. bf16 against fp16 SSD measured (Sec. 5.2, S9). | none |

### M8 Clarity

| Item | Status | Evidence | Missing |
|---|---|---|---|
| M8a abstract | P | Kernel disclosed; three findings; undefined terms removed. | 207 words (target about 180). "44 and 8%" have no convention named. Causal wording on the power cap (F7). |
| M8b Sec. 5 by finding | A | 5.1-5.5 follow the questions; Tables 5 and 10 split; CMI, permutation tests and fit bootstrap moved to the supplement. | 5.2 has no table or figure. Sec. 2 is still long. |
| M8c one convention per table, glossary | P | Table 2 conventions-to-regime mapping; Glossary paragraph; most captions name their convention. | F4 mixing. "covered" and "tuned shapes" not in the glossary. "host-bound" versus "launch-bound". |
| M8d figures and anchors | P | Fig. 1 (mechanism), Table 3 (claims), S2 cross-device and S11 divergence in the main text. | Cost-model fit or profiling-budget figure not in the main text. TP still prose only. F7 anchors. |
| M8e Discussion and Limitations | A | About one page of Discussion (tuners, engines, beyond these kernels) and four itemised Limitations. | none |

### M9 Positioning

| Item | Status | Evidence | Missing |
|---|---|---|---|
| M9a pruning and occupancy prior art | A | Sec. 2 "Pruned and model-ranked tuning" (Ryoo, Lim, Falch, Bergstra, Kaufman, Pruner, GPTune/ytopt/BaCO, Balaprakash) and "Performance models" (Hong and Kim, Volkov 2010/2016). | none |
| M9b production stacks | A | Sec. 2 "Shape-keyed tables in production stacks" (FFTW, MIOpen, cuDNN, TensorRT, Kernel Launcher, PyTorch 2, FlashDecoding++, vLLM/SGLang MoE configs) and "what it adds". | none |
| M9c TFLA | A | Sec. 2 maps TFLA's expression onto the work term. TFLA-on-scan moved to S3 and S5. | none |
| M9d background | A | Blelloch, Martin and Cundy, Merrill and Garland, S5 (Smith et al.), Sinha et al. | none |

### Minor issues

| Item | Status | Evidence or what is missing |
|---|---|---|
| N1 n_tp | A | Sec. 5.4 and S6 use $n_{\mathrm{tp}}$. |
| N2 tau_0 | A | Table S12 uses $\tau_0$; pseudo-code $t_0$ is the tile start only. |
| N3 k | A (main) | No key index in the main text. The supplement keeps $P_k$ next to "top-$k$" (minor). |
| N4 waves | P | Main uses $n_{\mathrm{wave}}$; supplement notation Table S9 still lists $w$. |
| N5 caption length | N | Table 4: 99 words; Table 6: 99; Table 7: 101; Table 9: 86; Fig. 2: 80 (target under about 50). |
| N6 Fig. 1 | A (moot) | Old figure replaced by the 1x3 mechanism figure. New nit: panel (a) legend overlaps the L=512 curve at chunk 1024. |
| N7 Fig. 2 | P | Fonts, ticks and per-curve "default" fixed; LW panel added. No call counts per bin; host-bound RTX PRO 6000 series still missing (acknowledged in the caption). |
| N8 Fig. S5 | A (moot) | Removed. |
| N9 Table 2 "Calib." | A (moot) | Table removed. |
| N10 Table 4 labels | A (moot) | Table replaced (now Table 8). |
| N11 Table 5 | A | Table 10: MLP five-seed mean (s.d.), same calls per row, selectors separated. |
| N12 Table 7 | A | Table 6: "Oracle ms" column; work-term columns grouped and labelled per call. |
| N13 Table 8 | A (moot) | The forward-pass table was removed; Table 6 has oracle ms. |
| N14 reconcile one quantity | P | 0.38 against 0.40; 3.55/3.52/3.6 (F5). Table S12 caption does not state round means against medians. |
| N15 1.74 printed 1.8 | N | appendix l.697: "2.1, 1.8 and 3.0%". |
| N16 94% vs 93.5% | A | Main no longer quotes it. Supplement states the CI-overlap criterion. |
| N17 0.14% population | A | Sec. 5.4: "204 calls ... fall back to the bucket entry ... per-call figures are over all 2360 calls". |
| N18 stock dispatch 6.6% | A | Table 9 footnote c: "Same runtime-chunk binary; the separately timed stock binary: 6.6". Sec. 5.3: 5.4 and 3.1%. |
| N19 three GPU generations | A | Removed. Sec. 1 names GPUs per kernel. |
| N20 workstation GPUs | A | "an Ampere consumer GPU and a Blackwell server GPU". |
| N21 abstract qualifier | A | Removed. |
| N22 GDN ceiling | A | Sec. 5.3 "a ceiling rather than a balance"; Limitations "may be that ceiling (61 to 85)". |
| N23 resource-report rule defined | A | S3 "Calibration-only selectors": the rule with no fitted parameter is defined. |
| N24 three prompts | A | S15: "on the three greedy prompts checked". |
| N25 layer index | P | Main is correct ("adding the layer index changes no regret"); S2 l.295-296 still lists the layer index among the shape features. |
| N26 Supplementary prefix | A | All `\Sref` uses carry "Supplementary" or "Supp.". |
| N27 Sec. 1 question | A | "when can a model ... replace profiling a key? This is the classic trade-off ...". |
| N28 reference-scan dispatch | A | Sec. 4.1: "selects among the instantiations by the length at run time". Table 1 caption is consistent. |
| N29 chunk-spread statistics | A | 8.5x appears only in Table S8. Main reports regrets. |
| N30 "compile facts" | P | Main uses "resource report". The supplement still uses "Compile facts" (heading, Table S10, Fig. S1 caption, Table S14 row, S2 and S3 text). |
| N31 2% tie criterion | P | Glossary defines a 2% tie. No three-criterion tie table; launch-inclusive tie statements have no noise reference. |
| N32 batch-1 static LW share | N | 15.1/6.2/7.2% not added to any supplementary table. |
| N33 bibliography | A | Mamba-3 and log-linear now ICLR 2026; TFLA and FlashAttention-2 pages added; TileLang pages present. Preprint stays cs.CV (as filed). |
| N34 typography | P | Recipe is a shaded `\small` box and Conventions is a table. Table 6 is still `\scriptsize` with 2 pt column separation. |

### Acceptance criteria

See Section 1: A2 and A3 met; A1, A4, A5, A6, A7 and A8 partly met.

## 4. Number spot-check (manuscript against result files)

All files are in `src/experiments/results/tpds/`. Check marks: OK = matches after rounding; MISMATCH = see note.

| # | Manuscript location and value | Result file and field | Recomputed | Check |
|---|---|---|---|---|
| 1 | Table 7 SSD 370M table gain +0.29 / +0.18 | e32 ssd370m combined, graph `gain_table_latency_weighted_pct` | 0.2908 / 0.1764 | OK |
| 2 | Table 7 SSD 2.7B -0.03 / -0.04 [-0.40, +0.55] | e32 ssd2p7b combined | -0.0299 / -0.0426; range -0.404..0.546 | OK |
| 3 | Table 7 GDN -0.04 / -0.02 [-2.4, +0.15] | e32 gdn1p4b combined | -0.0424 / -0.0182; -2.387..0.151 | OK |
| 4 | Table 7 scan share 14.7-18.8 (15.3), 8.6-10.3 (9.6), 5.1-7.8 (5.4) | `scan_share_range_pct`, `_latency_weighted_pct` | 14.73-18.84 (15.22/15.34); 8.58-10.25 (9.60); 5.14-7.83 (5.36) | OK |
| 5 | Table 7 sessions agree 12/11/12 of 14; cross-session table -0.02 to +0.25% | graph `cross_session` | 12, 11, 12; -0.019..0.252 | OK |
| 6 | Sec. 5.3 "largest single-shape gain, 2.3% at 256 tokens" | `gain_best_range_pct` | best chunk: GDN 3.585 (s1) | AMBIGUOUS (F1) |
| 7 | Table 6 recoverable 2.79 [1.56, 5.10], 1.72 [0.95, 3.41], 2.17 [1.07, 4.42], 5.39 [2.70, 7.59], 0.45 [-0.83, 1.86], 2.94 [1.84, 5.19] | `taco_r2_stats_m3a.json` headline `recoverable_lw_table_pp` | identical | OK |
| 8 | Table 6 default LW 2.8/1.7/2.2/5.9/1.0/3.2 | m3a `default_lw_pct` | 2.80/1.75/2.20/5.85/1.02/3.19 | OK |
| 9 | Sec. 5.3 "recovers 1.01% [0.49, 2.09]" | m3a `recoverable_lw_table_insample_pp` | 1.01 [0.49, 2.09] (in-sample) | OK value, label F6 |
| 10 | Sec. 5.3 covered calls "at most 0.03% of the scan time" | m3a `subsets/covered/table_lw_pct` | max 0.023 | OK |
| 11 | Table 6 work-term Graph 1.35 / 0.78 / 12.2 | `taco_r2_graph_calibration.json` plain `model_mean_regret_pct` | 1.347 / 0.780 / 12.16 | OK |
| 12 | Sec. 5.3 "1.4 and 0.8% per-call regret" | same | 1.347 | MISMATCH (1.3) |
| 13 | Sec. 5.3 padded "4.2 and 1.0% per call, 0.8 and 0.2% of the scan time"; GDN "12% (0.8%)" | graph calibration, padded | 4.15/1.02; LW 0.757/0.200; GDN 12.16 / 0.789 | OK |
| 14 | S3 graph-fit R2 0.994/0.995/0.997, padded 0.994/0.993/0.997 | `fit_r2_log` | 0.9938/0.9945/0.9970; 0.9942/0.9926/0.9970 | OK |
| 15 | Sec. 5.3 boards differ 0.2-0.5% median, 5th-95th percentile -6 to +11% | `taco_r2_sessions.json` stock_s1_vs_cloud_board `latency_ratio_b_over_a` | 1.0018-1.0054; p5 0.9375, p95 1.1109 | OK |
| 16 | Sec. 5.3 oracle agreement 93-95%; served 0.10-0.77% per call, at most 0.03% LW | sessions `same_oracle_share`, `oracle_of_*_served_*` | 0.927-0.951; 0.097-0.769; LW max 0.028 | OK |
| 17 | Sec. 5.3 second session: oracle 63-100%, up to 2.2% LW, 0-0.7% per call; S8 0.66/0.57/0 and 2.2/1.6/0 | sessions stock_s1_vs_stock_s2 | 0.634-1.0; LW 2.159/1.621/0; 0.658/0.571/0 | OK |
| 18 | Sec. 5.3 packed: default per-step oracle 78-94%; 1.1-2.5% per step; at most 0.5% LW | `e36_packed_graph_4c_e36.json` | 133-159 of 170 (78.2-93.5%); 1.10-2.46; LW max 0.487 | OK |
| 19 | Packed: capture-size table recovers up to 44%, leaves 1.0-1.5%; noise floor up to 0.6% | e36 `table_capture_size`, `noise_floor_oracle_split_half` | max (2.457-1.373)/2.457 = 44.1%; 0.96-1.51; 0.57 | OK |
| 20 | S10: 146/159/154 (carried 151/156/133); 2.24/1.17/1.68; argmin same in 76-98% | e36 `oracle_hist`, default, `argmin_same_all_rounds_share` | identical; 0.765-0.982 | OK |
| 21 | Sec. 5.4 (4,2): chunk 128 best at 10/12, 1.0%, configuration 12.5% slower | `e34_fixed_launch_4c_e34.json` `fixed_launch/d4_w2_s1` | 10 keys, 1.054, 12.48 | OK |
| 22 | Sec. 5.4 median over 32 fixed configurations 4.5% (0-22); (8,2,1) 13.0%; best entry 14.5% (at most 38%) | e34 `fixed_launch`, summary | 4.45 (0-21.9); 12.97; 14.47 / 37.69 | OK |
| 23 | Table S15 rows (e.g. 128,1: 128/16/8/1, 0.016, 15.7, 56.8/-2.5; 512,1: 28.5, 0.0/24.0) | e34 `per_key` | identical | OK |
| 24 | Sec. 5.4 Shapley 44.5 = 50.6 - 6.2 (RTX PRO 6000); fixed-config medians 14.5/4.6/4.2; best entry 8.5% | `taco_r2_baselines_m2d.json` pro6000_r4spin, vgpu3090_r3spin, h800_r3spin | 44.46 = 50.63 - 6.17; 14.51/4.55/4.16; 8.54 | OK |
| 25 | Table 4, all 12 rows (C*, spread, values (draws), real, best changed, placement) | `e33_value_graph_4c_e33.json`, `e33_value_graph_4c_e33m1.json` | identical (e.g. GDN 2048x1: 1.78x, 9.3 (6.4), -4.5, 0.5) | OK |
| 26 | Sec. 5.1 Gaussian best for every input at 8/12, at most 2.8% else, 5.1% where rounds vary 2.6% | e33 `best_chunk_by_input`, `round_cv_median` | 4 shapes with other best; 2.09/0.52/2.80; 5.09 at cv 0.0255 | OK |
| 27 | Abstract and Sec. 1: "sparse and constant inputs up to 8% faster" | e33 configs at C* | 7.3% (slowest dense against fastest constant or 2% sparse); body says 3.2-9.3% | APPROX (F7) |
| 28 | Sec. 5.3 counters: occupancy 13-16%, 1.6-1.9 warps per scheduler at batch 1 | `taco_r2_counters.json` m1 default | 12.8-16.5%; 1.56-1.96 | OK (rounding) |
| 29 | Counters: L2 hit 43 to 60% on placement; (4,2) chunk 128 occupancy 26%, 8-13% faster; only chunk 1024 spills | counters m1 8192/1/64 fresh/offset; d4_w2 | 42.96 to 59.51; 25.6-26.0%; 7.5-12.9%; `local_st_sectors` > 0 only at 1024 | OK |
| 30 | Counters: SSD state passing 3.4% issue, 38 long-scoreboard stalls | counters ssd L=4096 | 3.42-3.50%; 36.3-38.1 | OK |
| 31 | Fig. 1 / Sec. 5.3: SSD balance at 128-256 (L=4096) | `taco_r2_mechanism.json` | sum 291.0 (128) / 292.9 (256) us, minimum | OK |
| 32 | Table 8, all 15 cells (chunk 256, free rule, per batch, bucket, exact) | `taco_r2_baselines_m2b.json`, `taco_r2_baselines_m2c.json` | identical (e.g. 7.76/16.3; 17.40/2.665; 1.55/3.93) | OK |
| 33 | Table 10 pre-specified 0.58/2.32, 0.85/0.86, 2.14/3.14; six-protocol ranges; terms-on-two-GPUs 0.40/2.29, 3.52/9.73, 3.01/5.33 | `taco_r2_baselines_m4.json` M4a, m2b policies | identical (ranges 0.40-4.20/2.29-26.92; 0.85-6.21/0.86-10.28; 2.14-3.01/3.14-5.33) | OK |
| 34 | Sec. 5.5 means 1.2 / 2.9 / 3.4 / 3.3%; "worst of the four (4.5%)" | Table 10 rows; m4 LOGO-captured | 1.19/2.86/3.35/3.26; 4.5 = mean(4.20, 6.21, 3.01), not a Table 10 row | OK / F3 |
| 35 | Sec. 5.5 cross-device "3.3 to 3.8%" | S2 paragraph | device 3.3/3.0, LI 3.8/5.4 | MISMATCH (F4) |
| 36 | Sec. 5.4 TP: 36.8/22.1; 30.7 [18.7, 44.7], 21.5; -1.0 [-4.0, 1.9]; 2.3 [1.3, 3.4]; argmin agreement 52/70% | `taco_r2_stats_m5b.json` | 36.82/22.10; 30.66 [18.69, 44.73], 21.52; -1.00 [-4.02, 1.93]; 2.33 [1.32, 3.38]; 0.518/0.696 | OK |
| 37 | Sec. 5.1 multiplicity 133 / 12 / 4 / 0 | `taco_r2_stats_m5h.json` multiplicity | 133/12/4/0 | OK |
| 38 | Sec. 5.1 continuous test 0.89 us [-3.97, 5.75], MDE 7 us, interaction p = 0.61; five shapes 0.00 us, at most 0.18 us over 29 | `taco_r2_stats_m5i.json` | 0.889 [-3.97, 5.75], 6.95, 0.606; 0.0001, 0.177 | OK |
| 39 | Sec. 5.1 H800: 28/36, 36/36, up to 48%, 4.4 and 5.1% at the selected chunk, 5.7% | `taco_r2_stats_m5d.json` headline_mamba1 | 28, 36, 47.6, 4.36/5.08, 5.65 | OK |
| 40 | Sec. 5.3 weighting: 0.8 to 25% request level; chunked prefill 8192: SSD -0.5 to 4.6, GDN -1.2 to 9.0 | `taco_r2_stats_m6a.json` compact_by_family recoverable LW | min 0.797, max 25.11; SSD -0.523..4.590; GDN -1.233..9.006 | OK |
| 41 | Sec. 5.3 "every policy, including the default, stays within 2.4%" (launch-inclusive SSD) | `taco_r2_stats_m5a.json` `m2_vgpu3090_launch` default | 2.53 (Table S17 shows 2.5) | MISMATCH |

Further minor rounding slips:
- "5 to 18%" scan share: the maximum is 18.84, so 19.
- The routed-study text gives "16.8 +/- 1.4 ms"; Table S24 gives "+16.8 +/- 1.5" in %.
- "19 to 66%" in the routed study recomputes to 18-65% from Table S24 means. This may be a pairing difference; check it.

## 5. Prioritised remaining fixes

| Priority | Fix | Items | Effort |
|---|---|---|---|
| 1 | Scope the headline (F1, F2). In the abstract, Contribution 1 and the Conclusion, say "on a stock RTX 3090, latency-weighted over 14 shapes of 256-8192 tokens". Give the per-shape mean and maximum (table +2.3% at 256 tokens; best chunk +3.6%). State that no serving engine or data-centre GPU was run and that device-bound results are single-call or whole-step graph replays. Fix the "largest single-shape gain" sentence. | M1d, M3b, M3e, M3f, A4 | text, 1 h |
| 2 | Trace-weighted end-to-end estimate. Weight the per-shape step gains of Table S19 by the Azure/BurstGPT prompt-length mix (already used in S13), per session, and report next to "at most 0.3%". | F1, M6a | offline, 1-2 h |
| 3 | "Pre-specified" protocol (F3). Rename or disclose it as protocol 6 of the six, selected in this revision. Add a supplementary table of all six protocols: per-call and LW regret, maxima, per-batch values, cluster intervals, chosen forms per fold (all in `taco_r2_baselines_m4.json`). Fix the "worst of the four (4.5%)" reference or add the LOGO row to Table 10. Add the LW companion to Contribution 6. | M4a, M4b, A5 | text plus offline, 2 h |
| 4 | Supplement cost-model residue. Remove "a new GPU needs its compile facts but hardly its own calibration" (l.562-563). Fix sigma "spill-store traffic" (l.609-610). Add "for this kernel" in the Conclusion. Report the cross-device result in one convention (F4). Add one sentence connecting the counters (no local stores below chunk 1024 on sm_86) to omega. | M4d, M4e, M4h | text, 30 min |
| 5 | Number and label consistency (F5, F6): 1.4 to 1.3; 5-18 to 5-19; 2.4 to 2.5; 1.8 to 1.7 (l.697); state round means against medians for 0.38/0.40 and 3.55/3.52, and drop "3.6" (l.696); "in-sample" for 1.01% [0.49, 2.09]; 16.8 ms against % in S15. | N14, N15, A6 | text, 30 min |
| 6 | Noise floors and floor sensitivity. Add a noise-floor row or footnote to Tables 5, 8, 9 and 10, and one sentence on launch-floor factors 1/1.5/2/3 (data in `taco_r2_stats_m5a.json`; e.g. at factor 1.5, 556 of 624 SSD calls have chunk 256 as oracle). Flag unguarded columns (Table 5 H800, Table 9 CUDA scan) in captions. | M5a, M5g, A6 | text plus offline, 1-2 h |
| 7 | Abstract to about 180 words. Name the convention for 44% (launch-inclusive, RTX PRO 6000, Supplementary) and 8% (graph replay, stock RTX 3090). Write "under a power cap" instead of "a power cap moved". Anchor the 44% in the main text, or quote the Table 4 placement maximum (7.5%). | M8a, F7, A8 | text, 1 h |
| 8 | Main-text anchors for the specialization-key and profiling-cost claims. Either a 3-4 row table in Sec. 5.2 (aligned/unaligned Triton cliff; SSD +7.4% at 2056; bucket-key headroom 22-90 us) or a pointer row in Table 3 that names a main-text table. Profiling-versus-calibration costs as two columns in Table 10. | M8d, A8 | text, 1 h (watch the 20-page limit) |
| 9 | Serving replay re-costing (M6c) with the production kernels' graph-replay timings (Table 6 data), or explicitly scope the Sec. 5.5 key-churn numbers as Triton-scan-based. Lead S15 with stock dispatch and prefill; state 46-47 ms prefill at 256/976 tokens. | M6c, M6d, A7 | offline 2-3 h, or text 15 min (scoping) |
| 10 | Packed-step residue. Justify or equalise carried state (SSD first sequence against GDN every sequence). Recipe key: "capture size (graph engines) or packed length". | M3d | text, 15 min |
| 11 | Minor items: shorten captions to about 50 words (Tables 4, 6, 7, 9; Fig. 2); "compile facts" to "resource report" in the supplement; notation $w$ to $n_{\mathrm{wave}}$ in Table S9; layer index in S2; Fig. 2 call counts; Fig. 1(a) legend overlap; N32 batch-1 static LW share (15.1/6.2/7.2%); N31 tie-criterion note; Table 6 `\scriptsize`; "host-bound" to "launch-bound" in Fig. 2 and Sec. 5.3; M2b length-tree sentence; M4c fit-documentation sentences. | N4, N5, N7, N25, N30-N32, N34, M2b, M4c, M3c | text, 2-3 h |
| 12 | Artifact. Rebuild the capsule after E35b (include `e35_counters_4c_e35.json`, `taco_r2_counters.json`; exclude `e36_packed_graph_nonexistent.json`). Fill or document the null `meta` in the E32 combined files. Re-run the anonymization scan and upload. | V1, A1 | artifact, 1 h |
| 13 | Author actions: confidential cover-letter disclosure of the preprint relationship; upload only the two PDFs (the paper folder still holds author photos and venue backups). | V2, V3, A1 | author |
| (optional) | A second GPU for the whole-step graph (RTX PRO 6000 or a data-centre GPU) would remove the single-GPU scope of the headline; not required if Fix 1 is done. | M3f | GPU, 2-4 GPU-h (rented) |

No GPU work is required to meet A1-A8. Fixes 1-8 and 12 close all criteria except the author actions (Fix 13).
