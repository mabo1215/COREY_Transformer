# tpds-r4 pre-submission check (2026-10-09)

Paper commit `740abe4` (submodule pinned by main `b26c24a`). Three parts:
(1) independent re-review by a fresh agent against `docs/revision_suggestions.tex` (tpds-r4) and
the editorial decision, without access to the ledger, progress notes or design notes;
(2) a number trace of every quantitative claim in `main.tex` against `src/experiments/results/tpds/`;
(3) build, template and anonymity checks (TeX Live 2023, bash port of `build.bat`).

Overall verdict of the independent reviewer: **Major Revision (short; no new GPU time needed)**.
The data for every fix exist; the work is text and offline re-analysis, but it changes the framing
of the abstract, Contribution 2 and the deployment section.

## 1. Build, template, anonymity

| Check | Result |
|---|---|
| Build | main 14 pp, supplement 25 pp (was 16); exit 0; BibTeX clean |
| Undefined references / citations | 0 / 0 in both; no `??` in the PDFs |
| Overfull boxes | none; 3 underfull warnings (one \vbox on p. 8, N3) |
| Page limit | exactly 14 pp, references end on p. 14: no slack for added text |
| Abstract length | ~264 words (IEEE asks 150-250) |
| Fonts | all embedded |
| PDF metadata | no Title/Author; date and trailer ID suppressed |
| Figures | 9 matplotlib PDFs, no author/path metadata |
| Leftover markers | no PENDING/TODO/red text |
| Template | `\documentclass[journal]{IEEEtran}`; TPDS is a Computer Society journal, usually `compsoc`: **verify against the current TPDS author guidelines** (could not be fetched from the session) |
| Anonymity, text | "Anonymous Author(s)"; no names, affiliations, host paths or "COREY" in main or supplement |
| **Anonymity, availability link (V1)** | **main.tex:1544 still `codeocean.com/capsule/3457679/tree`** (editorial letter C-A: blocks re-review; asks for an anonymized review link, DOI at camera-ready, logged-out check). The capsule was republished in `b26c24a`. |
| Anonymity, artifacts (V2) | 101 result JSON files embed `COREY_Transformer` paths; some carry host names and commits (e.g. `e20_cold_analysis_pro6000.json` meta: `bobsteam1215`, `/mnt/c/source/COREY_Transformer/...`). Scrub before they ship in a review package. |
| V4 | `Idiap/gated-deltanet-swa-1.4B-30B` still named (appendix.tex:1668) |
| Cover letter | discloses arXiv:2604.10597; the manuscript does not mention it (correct) |

## 2. Review items not fully addressed

Verdicts of the independent reviewer (all other M*, N*, V3, V5, V6 and A2, A5-A7, A9 ADDRESSED or
DROPPED-BUT-DISCLOSED for the optional items M4b-d, M7e, M10a, M10d, M11c).

| Item | Verdict | Missing / fix |
|---|---|---|
| V1 | NOT ADDRESSED | anonymized review link (see above) |
| V4 | NOT ADDRESSED | cite the GDN checkpoint by reference in the review version |
| N5 | PARTIAL | `src/experiments/tpds/README.md`:178-201 still lists E18-E21 as "not run yet" and r4 analyses as "not in the manuscript"; unmapped: `e20_*`, `e22_*`, `e23_*`, `e24_*`, `refscan_baselines_*`, `tables_iv_v_ci`, `r4_rounds_analysis`, `r4_thinkey_*` |
| M2b | PARTIAL | Mamba-2 rows of Tables II-III still one block, launch-inclusive: say so in the captions or re-derive |
| M3b | PARTIAL | graph results only in one paragraph and Table S-graph; not in Tables II-III or the deployment table |
| M3d | PARTIAL | one sentence on which L2 regime a tuning cache should use (warm fine for the chunk, cold differs at 7/12 keys for the full launch config) |
| M5b / A4 | PARTIAL | baseline table launch-inclusive only; add device-time columns and a cross-GPU block; rewrite main.tex:979-993 (see new problem 1) |
| M5c | PARTIAL | label "0.4 to 3.5% in device time" and give its p95 |
| M6a / A8 | PARTIAL | deployment table pooled per checkpoint; add per-(L, B) rows (`deployment_units.json` `by_bucket_batch`), share of prefill, graph rows |
| M7d | PARTIAL | bootstrap CIs on the Table IV regret columns |
| M8a | PARTIAL | device-side decision cost (7.3-42 us) next to the 20 us host sync in the abstract |
| M8c | PARTIAL | abstract lacks shard p95 (31%), ref-scan width insensitivity (2.2%), any trace result, the RTX 3090 device-time contrast (tree 0.4% vs model 3.5%) |
| M9a | PARTIAL | report the 32-thread model failure (49.5%, Table S-ref-threads) in the main text near main.tex:1103-1108 |
| M12a | PARTIAL | VII-D first and baseline paragraphs carry ~26 numbers each; move LOKO/top-k/ablation numbers to the table or supplement |
| N3 | PARTIAL | underfull \vbox p. 8; underfull \hbox main.tex:1173-1181 |
| A1 | PARTIAL | title still promises a general cost model (see new problem 5) |
| A3 | PARTIAL | abstract "0.8 to 3.8%" untabulated (see new problem 3) |
| A10 | NOT MET | V1, V4, N5 |

## 3. New problems (independent reviewer)

1. **[major] main.tex:989 "no learned regressor beats the model on any GPU" is false in device time.**
   `cost_model_baselines_r4_summary.json` captured_calls: RTX 3090 (`vgpu3090_r3spin`) cost model
   3.55/20.9, MLP-basic 0.47/3.6; H800 (`h800_r3spin`) cost model 3.01/13.7, MLP-structural
   0.57/5.3. No single learner wins on both (MLP-basic 7.99% on H800, MLP-structural 5.09% on 3090),
   so a defensible statement is "no single learned regressor consistently beats the model".
   (Verified in this check.)
2. **[major] CUDA-graph replays undercut "defaults suffice" and the deployment conclusion.** SSD
   default costs 0.571 ms (370M) and 1.417 ms (2.7B) per forward pass vs 0.014 / 0.050 ms for a
   held-out table (`e22_graph_analysis_vgpu3090.json`); main.tex:1288-1289 and the deployment table
   are eager-only, and the abstract leads with the eager result.
3. **[major] Abstract "0.8 to 3.8% over a length-bucket and batch table ... against about 20 us per
   host synchronization"** (main.tex:67-70, 156, 1530-1531): 3.8% is the H800 device-time headroom,
   19.8 us/call, i.e. about one host sync (H800 sync not measured); 1.4B/2.8B leave 17.1/20.4 us.
   The "sync exceeds headroom" argument holds only for exact-shape tables. The value is in no table
   (recomputed from the E2 CSVs with `e5_breakeven._cv_shape_table`).
4. **[major] PRO 6000 launch-inclusive headlines rest on a session that did not reproduce**
   (supplement: tree 0.7 -> 4.7%, model 0.9 -> 4.9%, headroom 0.45 -> 3.9 us; first session had no
   exclusive-GPU guard): Table baselines, Table framework, Table replication, Fig.
   calibration-budget, the abstract's 2.7%, "the analytical form wins on the RTX PRO 6000"
   (main.tex:979-981). Restate in device time.
5. **[major] The cost model's failures are absent from abstract, contributions and conclusion**:
   TFLA (only independent public kernel) 7.5% vs default 1.4% and batch table 0.9%; under graphs the
   work-term model loses 12-22% (main.tex:1170).
6. [minor] Abstract main.tex:57-58 and Contribution 1 (132-134) include the reference CUDA scan in
   "under CUDA graphs ... defaults lose 16 to 22%"; it was never graph-replayed.
7. [minor] Ref.-scan learned rows in Table baselines labelled "struct." are the basic learners
   (`refscan_baselines_vgpu3090.json` gbr_basic 16.53, mlp_basic 6.99).
8. [minor] Denominators: "0.24% above the launch floor" is over 397/2360 calls; "154 of 155" is out
   of 1770 GDN calls.
9. [minor] "Activation statistics add nothing to the shape" is absolute; Mamba-2 device-time CI
   [-0.92, -0.07] excludes zero (Table S-information); qualify ("on Mamba-1").
10. [minor] the second launch-inclusive session appears only in the supplement; give both
   sessions side by side in the main text.

## 4. Number trace (main.tex)

~760 values checked (every cell of 8 numeric tables, ~380 in-text/caption numbers): ~735 MATCH,
~20 DERIVED-OK, 4 MISMATCH (5 values), 2 UNTRACEABLE.

Fully verified: `tab:value_invariance` (37/37), `tab:oracle_regret` (42/42), `tab:predictors`
(69/69 apart from the Mamba-2 batch-only "--"), `tab:baselines` (80/80), `tab:reference` (54/54).
Mostly verified: `tab:framework` (17/19), `tab:deployment` (29/30), `tab:replication` (52/53).

### Mismatches

| main.tex | Paper | Correct | Source |
|---|---|---|---|
| 523 `tab:framework` Mamba-2 SSD | 1.5/2.5 | **1.6/2.4** | `cost_model_{pro6000,vgpu3090}.json` V4_pipelines.mamba2.e2.mean_regret_pct (1.561/2.361); 1.5/2.5 is the old tie rule; lines 889 and 1040 already use 1.6/2.4 (verified) |
| 1029 | 353 s of compilation | **363 s** | `e5b_pro6000.json` tuning_scale.compile_total_s = 362.9 (352.8 is the first key only) (verified) |
| 1272 `tab:deployment` Triton 370M RTX 3090, Table column | 0.24 | **0.23** | `deployment_units.json` kernels.mamba1_triton_370m_vgpu3090.pooled.all.exact_table_heldout.ms_per_forward_pass = 0.2349 (double rounding) |
| 1405 `tab:replication` chunk spread RTX 3090 | 2.0x | **2.2x** | `e1_analysis_vgpu3090.json` decision[].between_chunk_max_over_min = 2.226 (GDN, L=8192); 2.045 is Mamba-1 only, but the row sits above the "Mamba-1 only" rule |

### Untraceable

| main.tex | Claim | Note |
|---|---|---|
| 621 | top-token agreement 100% | computed in `capture_traces.py` (`argmax_agreement`) but stored only in the uncommitted trace manifest |
| 1295 | step model 2.4x faster than eager prefill | not stored; recomputed 2.34 with the stated step model (2.45 only without the 2 ms per-step cost) |

Derived values with no stored key (reproduced from CSVs): 0.8-3.8% bucket-table headroom and
2.2 us / 0.8% (64, 156, 1229, 1531), <1% convention difference above 100 us (651), median oracle
60/98 us (652), launch cost 56-90 us (646; 90 only in the supplement), 48.8%/14.1% (1360-1361),
4.3% (1116), best chunk 64-1024 (791-792). Consider writing these into a result JSON so that every
number has a key.

### Inconsistencies across places

1. Mamba-2 work-term regret 1.5/2.5 (523) vs 1.6/2.4 (889, 1040).
2. "A table tuned on another GPU loses 14 to 49%" (66, 147, 1521) is the batch-1 single chunk
   (48.8/14.1, 1360-1361); a per-batch table loses 47.8/13.9; H800 pairings excluded (PRO 6000
   table on the H800: 27.1%).
3. "At the selected chunk": 767 (8.6%) counts significant effects only; 1373 (10.6%, H800) is a
   non-significant effect (BH p = 0.52); significant-only would be 4.4%.
4. CUDA-graph claims in abstract (57-58) and intro (132-133) include the reference scan; 705-707
   and 1516-1518 scope it correctly.
5. "Unsharded table loses up to 63%" (65, 149, 699, 1520) is the RTX 3090; H800 shards reach 65.0%
   (2.8B, t=8, `e12_analysis_h800.json`); the reference scan's per-shard maximum is 11.2%, not the
   pooled 2.2% it is contrasted with.
6. "SSD chunk 256 at most 0.24%" is the largest per-batch mean; worst single call 14.1% (supplement only).
7. 1342 "packed ... at most 2.2 and 9.1%" holds for the colocated scheduler only (prefill-only: 4.7/13.3%).
8. `fig:chunk_curves` (784): the plotted batch-1 Gaussian data have their minimum at chunk 512 for
   all three L; the 64-1024 range of 790-792 comes from E4 keys at batch 4/16.
9. 879 Mamba-2 "Batch size only = --": files give 1.43/5.53/77%, below the "1.7% to 2.2%" range of 915.
10. 1001 "maximum does not drop" with two chunks profiled: drops 116.3 -> 101.6% on the PRO 6000.
11. "154 of 155 calls" (56, 704) omits "above the launch floor" (1770 GDN calls).
12. 906 "at most 0.013 bits on any data set" excludes the superseded E11 1.4B run (0.045 bits, p = 0.001).
13. 1202 "1 to 5% in total time": minimum 0.76%.

## 5. Suggested order of work (no GPU)

1. V1/V2/V4 anonymity (blocker): anonymized review link, scrub result-file metadata in the review
   package, cite the GDN checkpoint by reference.
2. Four mismatches and the 13 inconsistencies (mechanical).
3. Device-time baseline columns and the corrected main.tex:989 sentence (new problem 1); device-time
   restatement of the PRO 6000 headlines (4).
4. Abstract and contributions: graph caveat (2), 3.8% scoping (3), model failures (5), reference-scan
   graph scope (6), M8a/M8c additions, trim to <= 250 words, keep 14 pages.
5. Deployment table per (L, B) with graph rows (M6a), Table IV CIs (M7d), README mapping (N5).
6. Fresh Stage 9 review after the fixes; confirm the template option (`compsoc`) against the TPDS guidelines.
