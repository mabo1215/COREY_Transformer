# TACO re-check of cycle taco-r1 (fresh reviewer, 2026-10-16)

Scope: every item of `docs/revision_suggestions.tex` (V1-V4, M1a-M10d, N1-N19) and acceptance criteria A1-A8, judged only from the review file, `paper/main.tex`, `paper/appendix.tex`, the built `paper/main.pdf` (25 pp.) and `paper/appendix.pdf` (31 pp.), `paper/ref.bib`, and spot checks of `src/experiments/results/tpds/`. Venue: ACM TACO regular paper, acmsmall, at most 20 pages excluding references, double-anonymous review.

Number spot checks (all matched unless noted): graph-replay table, RTX PRO 6000 rows (`taco_e26_analysis_pro6000.json`); packed calls (`e27_varlen_pro6000_taco.json`); autotuner state (`e30_autotune_state_4c_e30.json`); output divergence (`e31_divergence_4c_*.json`); selector CIs and latency-weighted shares (`taco_report_numbers.json`); a priori form selection, launch-table Shapley split, calibration-fit checks, top-k and learning curve (`taco_model_baselines.json`); 1.4B/2.8B transfer (`cost_model_transfer_e2_mamba1_vgpu3090_*_r3.json`); TP pooled numbers. One mismatch: the OLS regret on the RTX 3090 is 1.74%, but the paper prints 1.8% (appendix.tex:847).

## 1. Summary counts

| Verdict | Count | Items |
|---|---|---|
| ADDRESSED | 47 | V1, V3, V4; M1a-e; M2b, M2c; M3a, M3b, M3d; M4a, M4c, M4d, M4f; M5a, M5b, M5c; M6b, M6c; M7a; M8a, M8b; M9d; M10a, M10b, M10d; N1-N8, N10-N19 |
| PARTIAL | 11 | V2; M3c; M4b; M4e; M6a; M9a; M9b; M9c; M9e; M10c; N9 |
| NOT ADDRESSED | 0 | none |
| DECLINED-WITH-JUSTIFICATION | 8 | M2a, M2d, M3e, M4g, M5d, M6d, M6e, M7b |
| **Total** | **66** | |

Acceptance criteria: **met** A2, A3 (by scoping), A4 (with minor residuals), A6, A7 (by scoping); **partial** A1, A5, A8.

Stage 11 gate: page limit PASS; PDF anonymity PASS; source anonymity PASS for the three paper files, with a WARNING for the surrounding folder; artifact anonymity **FAIL (archive not ready)**; undefined references PASS; cite keys PASS; repository paths and script names PASS.

## 2. All items

| ID | Verdict | Evidence | Gap (short) |
|---|---|---|---|
| V1 | ADDRESSED | Conclusion and disclosure end on p.20; "References" heading at top of p.21 (main.pdf) | none |
| V2 | PARTIAL | Supplement compiled with acmart TACO (build/appendix_only.tex); figures renamed `fig_*`; PDF metadata "Anonymous Author(s)" | Artifact and folder residue (see 3) |
| V3 | ADDRESSED | `\title[Shape Sets the Chunk]` main.tex:39; running head on odd pages | none |
| V4 | ADDRESSED | Availability statement main.tex:1455-1461 | Archive itself must exist (see V2) |
| M1a | ADDRESSED | Title main.tex:39-40; Lemma main.tex:364-380; entropy scheduler as control main.tex:136-139, 341-348; Sec. 3 condensed (main.tex:329-404); routed study is one paragraph main.tex:1177-1194 | none |
| M1b | ADDRESSED | Third-person preprint cite main.tex:117-120, 276-280 | Advisory on self-citation (see 3) |
| M1c | ADDRESSED | Assumptions "to check" main.tex:386-397; scoped exclusion against bucket vs exact table main.tex:1211-1226 | none |
| M1d | ADDRESSED | Protocol note in methodology main.tex:523-532; not a contribution | none |
| M1e | ADDRESSED | "Placement, not values" main.tex:731-744 | none |
| M2a | DECLINED | Not run; M2c fallback applied | see 3 |
| M2b | ADDRESSED | Reference-scan compile facts main.tex:1042-1053; appendix.tex:1167-1207; Table S (ref_threads) appendix.tex:1209-1245 (49.5% falls to 1.1%) | none |
| M2c | ADDRESSED | "Case Study" heading main.tex:406-414; "What transfers ... untested" main.tex:501-506; model removed from title; Contribution 3 main.tex:156-161; Limitations main.tex:1419-1421 | Minor over-generalizations (see 3) |
| M2d | DECLINED | Optional; claims limited to chunked recurrence kernels | none |
| M3a | ADDRESSED | Coupling and provenance main.tex:421-423, appendix.tex:212-215, pseudo-code table appendix.tex:238; flat registers main.tex:968-970; Shapley split main.tex:970-977 (matches data: 32.7 = 29.3 + 3.4 etc.); Table 1 wording main.tex:210 | Split given only launch-inclusive (secondary convention) |
| M3b | ADDRESSED | Static spill size as proxy main.tex:426-428; shared memory justified main.tex:431; bandwidth floor stated main.tex:445-447 (data: never binds) | appendix.tex:793 still calls sigma "spill-store traffic" |
| M3c | PARTIAL | OLS appendix.tex:846-848; 50 starts main.tex:932-935; bootstrap CIs Table S (params_ci) appendix.tex:858-880 | No full-protocol or variance-weighted refit reported; no objective R^2; 1.8 vs 1.74 |
| M3d | ADDRESSED | Best static per batch as untuned baseline in Table 8 (main.tex:1242-1269) and Fig. S (calibration budget); per-GPU launch share main.tex:974-977 | none |
| M3e | DECLINED | Optional; Limitations main.tex:1425 ("no model replaces a search of the full launch space"); chunk effect owned by one kernel main.tex:1419 | none |
| M4a | ADDRESSED | Device time primary main.tex:611-615, 650-651; LI secondary tables S (conventions, oracle_li) appendix.tex:1643-1717 with both sessions; MLP and three-parameter wins reported main.tex:941-950 | Production-kernel rows remain LI where no device time exists (labelled) |
| M4b | PARTIAL | LW and ms in Tables 4, 7 and 8; LW for the frozen model main.tex:920-921 | No LW for the a priori headline (PRO: 26.9% of scan time); per-GPU LW of batch-1 static chunk missing |
| M4c | ADDRESSED | LOGO with fewest-parameters rule main.tex:923-930; paired CIs Table S (topk_all) appendix.tex:897-923; MLP five seeds main.tex:943-949 | Advisory: protocol sensitivity (see 3) |
| M4d | ADDRESSED | 12 of 57 shapes, 773 calls, off-calibration 0.46/3.3/3.0, per-shape 1.8/5.2/6.1, leave-one-key-out main.tex:935-939 | Abstract keeps "without timing the target calls" (acceptable given text) |
| M4e | PARTIAL | Tables S (tables_ci, ci) appendix.tex:1719-1796; per-batch CIs for Table 4 | 1.4B batch-4 13.9% not reported; no CIs for Table 6 or Table 8 |
| M4f | ADDRESSED | Box main.tex:611-620 (operational criterion: host overhead below kernel time) | none |
| M4g | DECLINED | Optional; Limitations main.tex:1421-1422 ("batch sizes 1, 4 and 16 and one form-blind GPU only") | none |
| M5a | ADDRESSED | Table S (topk_all) appendix.tex:897-923 (all rankers top-1/2/3, zero-parameter compile-facts rule, nearest key, random-k); recipe main.tex:1333-1336 | "Compile-facts rule (no parameters)" is never defined in the paper |
| M5b | ADDRESSED | Learning curve Table S (learning) appendix.tex:925-947; cold/warm compile cost main.tex:1346-1356; trace replays; use case stated | none |
| M5c | ADDRESSED | TFLA marked "other kernel class" main.tex:867; fitted on TFLA's own kernel main.tex:1173-1175 | none |
| M5d | DECLINED | Optional BO or online tuner; profiling claim concerns six chunks, launch-space claim scoped main.tex:1425 | none |
| M6a | PARTIAL | RTX PRO 6000 graph replays in Table 7 main.tex:1096-1124 (verified); scoped in abstract main.tex:61 and Limitations main.tex:1425-1426 | No data-centre GPU; graph refit uses replay folds, not graph-timed calibration keys; conclusion not scoped |
| M6b | ADDRESSED | Packed calls main.tex:1152-1162; Table S (packed) appendix.tex:1991-2045 (verified) | One GPU only (disclosed main.tex:1426) |
| M6c | ADDRESSED | Autotuner state appendix.tex:1934-1989; main.tex:1432-1434 | none |
| M6d | DECLINED | Limitations main.tex:1426-1432; main.tex:742-744; H800 sync "nominal" appendix.tex:2128-2129 | Main text omits H800 input effects (see 3) |
| M6e | DECLINED | Optional; claims stated as scan time | Add one Limitations clause (see 3) |
| M7a | ADDRESSED | Pooled vs pooled, retune reference, H800 table beats model, worst 15.2%, synthetic shard shapes main.tex:1062-1077; SP dropped from claims; collectives main.tex:1435-1436 | none |
| M7b | DECLINED | Optional; Limitations main.tex:1435-1436 | none |
| M8a | ADDRESSED | Recipe box main.tex:1320-1344 (warm-up or capture, persistence, invalidation keys, carried state, TP broadcast, prefix-cache, batch invariance) | Minor tension on "synthetic inputs suffice" for SSD/GDN (see 3) |
| M8b | ADDRESSED | Divergence appendix.tex:2047-2096 (192 prompts; verified); main.tex:1410-1412 | none |
| M9a | PARTIAL | Abstract main.tex:47-70 (266 words, about 6 percentages); contributions main.tex:141-162 | Two conventions mixed; undefined terms |
| M9b | PARTIAL | Box main.tex:608-642; notation table appendix.tex:754-780; n_p, N_s, r, pi_sh adopted | Leftover slips main.tex:516, appendix.tex:868; k overloaded |
| M9c | PARTIAL | Abstract and conclusion claims point to Tables 5-8 and Fig. 2; policy defined main.tex:341-348 | Supplement 31 pp. (target about 20); some abstract claims rest on the supplement only |
| M9d | ADDRESSED | "Triton Mamba-1 scan" vs "reference CUDA scan"; Fig. 2 regime map main.tex:1126-1135; practitioner Table S appendix.tex:1909-1932 | none |
| M9e | PARTIAL | Within-2% ties main.tex:637-640; conclusion answers RQs main.tex:1445-1453 | Captions of Tables 5 and 6 still >100 words |
| M10a | ADDRESSED | No legacy sections; supplement 48 to 31 pp.; "Every number was measured" appendix.tex:28-31 | none |
| M10b | ADDRESSED | Full-histogram range main.tex:1189-1192; seven GDN inputs main.tex:670-671, 6/12 df appendix.tex:332-334; pairwise TOST main.tex:725-727; GDN captured calls appendix.tex:1466-1479; Limitations no longer implies 32 threads only | none |
| M10c | PARTIAL | Telemetry appendix.tex:59-71; guard appendix.tex:87-95; memory clock 9501 MHz; mamba_ssm difference Table S2 caption | Table S1/S2 not updated for four newer runs |
| M10d | ADDRESSED | Pseudo-code appendix.tex:218-248; library patch appendix.tex:583-596 | none |
| N1 | ADDRESSED | Table 2 regret "--" for GDN main.tex:496; extrapolation main.tex:989-990; in-sample asterisk main.tex:485 | none |
| N2 | ADDRESSED | "no practically relevant information" main.tex:57, 146; normalized CMI main.tex:886-889; 0.5-point margin Table S (information) | Noise-floor filtering is implicit only |
| N3 | ADDRESSED | 17% and 9% main.tex:1088-1092 | none |
| N4 | ADDRESSED | Regret-led results intro main.tex:653-660; 8.5x only in Sec. 6.1 | none |
| N5 | ADDRESSED | Abstract no longer attributes it; graph floor explanation appendix.tex:1896-1900 | none |
| N6 | ADDRESSED | Stock dispatch as reference, +3.1% total main.tex:1185-1193 | Supplement table still reports vs. post-hoc best static (labelled) |
| N7 | ADDRESSED | Within-2% in main tables; CI-overlap only where stated main.tex:637-642 | none |
| N8 | ADDRESSED | Maxima 18, 31 and 50% main.tex:919-920 | Not given for the a priori forms (53/27/50%) |
| N9 | PARTIAL | Capped default tested main.tex:1147-1149, appendix.tex:1904-1906 | Padding term not added (disclosed main.tex:1424) |
| N10 | ADDRESSED | appendix.tex:208-212, 231 | none |
| N11 | ADDRESSED | Table 1 caption main.tex:197-202 | none |
| N12 | ADDRESSED | Step and decode model, 88 settings, prefill denominator appendix.tex:2206-2220 | none |
| N13 | ADDRESSED | Proportionality main.tex:1306-1308 | none |
| N14 | ADDRESSED | 0.0156 main.tex:1035-1037; appendix.tex:592-596 | none |
| N15 | ADDRESSED | Virtualization disclosed main.tex:559-561; Server Edition and power limit "not recorded" Table S2 | none |
| N16 | ADDRESSED | All listed works cited main.tex:233-307 (tritonBLAS, Roller, Pruner, Yotov, CLTune, KTT, Kernel Tuner BO, ATF, Seer, Kernel Launcher, NeuSight, CDMPP, CRAT, Kayiran, Stream-K, FFTW, OSKI, cuBLASLt, FA-2/3, Lightning-2, TileLang, fused SSD) | 20 entries spot-checked; see 4 |
| N17 | ADDRESSED | appendix.tex:40-48 | none |
| N18 | ADDRESSED | main.tex:586-587; decode share appendix.tex:2213-2215 | none |
| N19 | ADDRESSED | No "before this protocol" phrasing found | none |

## 3. Details for non-ADDRESSED items (and residuals that matter)

**V2 (PARTIAL): venue and identity residue in artifacts and folders.** The two PDFs and the three paper source files are clean. The folder and the artifact are not:
- **Author photos:** `paper/figs/profile/` contains `BoMa.jpg`, `JinsongWu.jpg` and `WeiqiYan.jpg`, the same three names as the cited preprint.
- **Unused old figures:** `paper/figs/` also holds `entropy_gain*`, `pipeline.jpg`, `workflow.png` and `performance_summary.jpg`.
- **Prior-venue folders:** `paper/backup/` (AAAI27, IEEETPDS, MLSys2027) and `paper/legacy/` (AAAI checklist).
- **Figure timezone:** the figure PDFs carry `+13'00'` creation dates. This is not propagated into main.pdf.
- **Raw result files:** in `src/experiments/results/tpds/`, 122 files contain `/root/` (autodl) paths, 148 contain git hashes, 146 contain host names, 166 contain "COREY" and 6 contain "mabo". The directory name itself is the previous venue.
- **Code Ocean capsule:** the latest build (`codeocean/`, 2026-10-10) has removed the user names. It still has `tpds` directory names and a real hostname "NZA1LT32KT324" in 2 files. It predates the taco-r1 runs (E26-E31), so it does not back Tables 7, S-packed, S-autotune_state or S-divergence.

Fix:
1. Upload only the two PDFs for review.
2. Never upload the `paper/` folder as is.
3. Rebuild the anonymized archive the paper promises (main.tex:1456):
   - strip host names, paths and hashes;
   - rename the `tpds` directories;
   - include the taco-r1 result files;
   - re-run an anonymization scan on it.

**M3c (PARTIAL): calibration fit quality.** The analysis file shows four results the paper does not report:
- A refit on the full-protocol timings alone selects with 0.73 / 4.28 / 6.05% (device time; the H800 doubles from 3.0%).
- Inverse-variance weighting leaves the fit unchanged.
- The objective R^2 is 0.89 / 0.93 / 0.94.
- The search-vs-full-protocol discrepancy reaches 226% (launch-inclusive RTX 3090).

The printed RTX 3090 OLS value 1.8% should be 1.7% (data 1.74; appendix.tex:847).

Fix: add one sentence and a row to Table S (params_ci) covering:
- the full-protocol-only and variance-weighted refits;
- the objective R^2;
- the maximum search-vs-full discrepancy;
- correct 1.8 to 1.7.

**M4b (PARTIAL): latency-weighted shares next to per-call headlines.** The abstract (main.tex:64-65) and conclusion (main.tex:1450-1451) give "3.0 to 6.2%" per call as the method's accuracy, with no latency-weighted companion. On the RTX PRO 6000 the a priori-selected form is the three-parameter restriction, which loses **26.9% of scan time latency-weighted** (taco_report_numbers.json, `work_only`, pro6000_device; CI 23.5-29.3). That is no better than the best chunk per batch (27.0%). The paper reports latency-weighted shares only for the frozen form (2.3 / 9.7 / 5.3%). Also missing: the latency-weighted share of the batch-1 static chunk per GPU (requested 15.1 / 6.2 / 7.2%). Table 4 gives the share of chunk 512, not of the per-call-best chunk.

Fix:
1. Add the latency-weighted share and ms per forward pass of the a priori forms next to "3.0 to 6.2%" in Sec. 6.4.
2. Qualify the abstract and conclusion (e.g., "3.0 to 6.2% per call, up to 27% of scan time on one GPU").
3. Add the latency-weighted shares to Table S (static_bucket).

**M4e (PARTIAL): intervals and per-batch breakdowns.** The pooled 1.4B transfer regret of 3.7% (Table 6, main.tex:1022, 1056-1058) still hides **13.9% at batch 4** (1.0 / 13.9 / 0.9 by batch; cost_model_transfer_e2_mamba1_vgpu3090_1p4b_r3.json), which is exactly the example the review named. There are no cluster-bootstrap intervals for Table 6 (reference scan and larger checkpoints) or Table 8 (deployment).

Fix:
1. Give the per-batch values for the 1.4B and 2.8B cost-model and table rows (text or a Table S column).
2. Add intervals for Tables 6 and 8 to Table S (ci).

**M6a (PARTIAL; remainder scoped).**
- Done: the RTX PRO 6000 graph replays (Table 7 rows "PRO", verified).
- Not done: a data-centre GPU, and calibration of the work-term model on graph-timed calibration keys. The refit uses replay folds (disclosed, main.tex:1424).
- Scoping: the abstract scopes the claim correctly to "two workstation GPUs" (main.tex:61), but the conclusion "The chunk matters where execution is device-bound ... 1.0 to 5.9%" (main.tex:1447-1449) is unscoped.
- Wording: Table S2 lists the RTX PRO 6000 as "Blackwell Server Edition", so calling it a "workstation GPU" is inaccurate.

Fix:
1. Scope the conclusion ("on the two GPUs measured").
2. Say "an Ampere consumer GPU and a Blackwell server GPU".
3. State in Limitations that no data-centre-class (HBM) GPU was replayed under graphs.

**M9a (PARTIAL, minor).** The abstract mixes the scan-time share (1.0 to 5.9%) with per-call means (3.0 to 6.2%, 0.4 to 3.5%). It also uses undefined terms ("recurrence-preserving launch", "three-parameter ranker", "residual regret").

Fix:
1. Label the convention of each number.
2. Define or drop the three terms.

**M9b (PARTIAL, minor).** Three notation slips remain:
- main.tex:516 "Tensor parallelism over $t$ devices runs the scan at width $d/n_{\mathrm{tp}}$" should use $n_{\mathrm{tp}}$ throughout.
- appendix.tex:868 labels the first parameter "$t_0$", but it is $\tau_0$; $t_0$ is also the tile start in the pseudo-code.
- $k$ denotes both a shape key (Eq. 5, main.tex:1296-1299) and top-$k$.

Fix: correct the first two, and rename the key index (e.g., $\kappa$) or write "top-$m$".

**M9c (PARTIAL).**
- The supplement is 31 pages, against the target of about 20.
- The tensor-parallel results appear in main-text prose only; the table is in the supplement.
- The abstract claim "a three-parameter ranker matches it once three chunks are profiled" rests only on Table S (topk_all).

Fix:
1. Trim the supplement further, e.g. merge Tables S (conventions, oracle_li, tables_ci, ci).
2. Move the top-3 result into Table 5 (one row: "three-parameter restriction, top-3") so that the abstract claim has a main-text anchor.

**M9e (PARTIAL, minor).** The captions of Table 5 (103 words, main.tex:827-837) and Table 6 (105 words, main.tex:996-1005) remain long, as does Table 3 (82 words).

Fix: move definitions into the conventions box or the text, and keep the captions under about 50 words.

**M10c (PARTIAL, minor).** Table S1 (protocol, appendix.tex:97-132) has no rows for four runs:
- RTX PRO 6000 CUDA-graph replays;
- packed calls (RTX PRO 6000);
- the autotuner-state re-timing;
- the output-divergence runs.

Table S2 (platforms) lacks the "stock 24 GB RTX 3090 board" used for the last two (appendix.tex:1952-1953, 2063). As a result, guard and telemetry status for these data sets is undocumented.

Fix: add the four rows and a platform column or footnote for the 24 GB board (driver and version).

**N9 (PARTIAL).** The capped-default rule was tested. The padding term for $L<C$ was not added, which is disclosed (main.tex:1424). This is an offline (class b) refit.

Fix: add $\max(L,C)$ padding to the work term and report the eager and graph regrets; otherwise leave it as disclosed.

**Declined optional or GPU items, and their scoping:**
- **M2a:** not run. Scoping is adequate: case-study framing (main.tex:406-414, 501-506), title without the model, and Limitations main.tex:1419-1421.
  - Residual over-generalizations to soften:
    - appendix.tex:742 "so a new GPU needs its compile facts but hardly its own calibration" (one kernel, three GPUs).
    - main.tex:1449-1451 "A model of the compiled kernel pays only for ..." (add "for this kernel").
- **M2d:** claims never extend to non-recurrent kernels. Adequate.
- **M3e:** the claims attribute the large chunk effect to "one kernel" and the reference CUDA scan shows its own 5.4x spread. Adequate.
- **M4g:** Limitations names batch sizes 1/4/16 and one form-blind GPU. Adequate.
- **M5d:** adequate. "No model replaces a search of the full launch space" covers the BO case.
- **M6d:** disclosed in Limitations (main.tex:1426-1427, 1431-1432). However, the main-text replication paragraph (main.tex:1386-1388) says "The role of the activation values does not change", citing only the RTX 3090. The H800 shows 28 significant input effects of up to 48% (Table S replication, appendix.tex:710) with no placement control.
  - Fix: add "on the H800, 28 of 36 configurations show input effects up to 48%, which we could not attribute to placement there".
- **M6e:** acceptable, because claims are stated as scan time.
  - Fix: add "no whole-model graph or serving-engine measurement" to Limitations.
- **M7b:** disclosed (main.tex:1435-1436). Adequate.

**Further observations (not tied to a single item):**
- **Routed study vs divergence study.** "Every configuration produced the greedy tokens of the unmodified model" (main.tex:1194) refers to three prompts. Section S11 shows that 2.6-4.7% of 192 prompts diverge for fixed alternative Mamba-1 chunks in fp16. Add "on these three prompts".
- **Recipe vs Limitations.** The recipe's "synthetic inputs suffice, as latency is value-invariant" (main.tex:1330) conflicts with Limitations, which states that SSD and GDN invariance above the launch floor was not measured. Qualify the recipe sentence for those kernels.
- **Sensitivity of the a priori protocol (advisory).** In taco_model_baselines.json, alternative candidate sets and criteria select other forms on the RTX PRO 6000 (2.24% without the three-parameter restriction in the set, 0.38% with the calibration-LOSO criterion, against the reported 4.16%). Reporting the least favourable variant is fine, but one supplement sentence disclosing the range would pre-empt a "garden of forking paths" objection.
- **Convention of the Shapley split.** The chunk-versus-launch split (main.tex:970-974) is given launch-inclusive only. In device time (the primary convention) the RTX PRO 6000 split is 50.6 points chunk and -6.2 launch. Give both, or say which.
- **Self-citation (advisory, allowed).** The refuted preprint (arXiv:2604.10597; Ma, Wu, Yan) is cited in the third person, as ACM permits. Before submission, the authors should:
  - make sure any public code repository linked from that preprint does not contain this manuscript or its result files during review;
  - tell the editor in the confidential cover letter that the preprint is their own;
  - check that the footnote "New Paper, Not an Extension of a Conference Paper" (main.tex:91) remains accurate.

## 4. Stage 11 gate

| Check | Result | Evidence |
|---|---|---|
| Page limit (20 pp. excluding references) | PASS | main.pdf 25 pp.; Discussion, Limitations, Conclusion and the "Artifact Availability and Disclosure of AI Tool Use" section end on p.20; "References" is the first heading of p.21 |
| Anonymity, PDF content | PASS | "Anonymous Author(s)", "Anonymous Institution"; no names, URLs of own work, hosts or paths in the text of either PDF |
| Anonymity, PDF metadata | PASS | Info and XMP: author "Anonymous Author(s)", no CreationDate/ModDate, no PTEX.FileName (`\pdfsuppressptexinfo=-1`), no embedded files; no matplotlib metadata or timezone leaks. The only "corey" strings are hyperref anchors `cite.corey2026preprint`, which match the printed third-person reference |
| Anonymity, paper sources (main.tex, appendix.tex, ref.bib, figure file names) | PASS | No author names except the third-person preprint entry (ref.bib:670-678); ref.bib comments are topical only; figure files `fig_*.pdf` |
| Anonymity, surrounding folder and artifact | **WARN / FAIL** | `paper/figs/profile/` author photos; `paper/backup/`, `paper/legacy/`; raw result files and the Oct-10 capsule carry host names, paths, hashes and `tpds` names (see V2). Must be cleaned before any source or artifact upload |
| Undefined references | PASS | 0 occurrences of "??" in either PDF; no undefined or multiply-defined warnings in build/main.log or build/appendix_only.log; 0 overfull boxes |
| Every \cite key in ref.bib | PASS | 92 keys (main) + 14 (supplement), all among the 103 bib keys; no duplicates |
| No repository paths or script names | PASS | No `src/`, `.py`, `.json`, result tags or host labels in either source. Only third-party ID: `Idiap/gated-deltanet-swa-1.4B-30B` (appendix.tex:1471) |
| TACO formalities | PASS | acmsmall review/anonymous; CCS concepts; keywords; running head; new-paper footnote; AI-use disclosure |
| Bibliography hygiene (non-blocking) | WARN | BibTeX: empty publisher/address on many entries (cosmetic in ACM style); no pages for beck2025tfla and dao2024flashattention2. The page range 73099-73134 on an ICLR entry (wang2026tilelang) is unusual; verify it |

## 5. Judgement on the taco-r1 acceptance criteria

Five criteria are met and three are partly met.
- **Met:**
  - A2 (framing): the bound is a lemma, the entropy scheduler is a cited control, and the title is about selecting and keying the chunk.
  - A3: met by explicitly scoping the cost model as a single-kernel case study after declining the GPU-heavy M2a.
  - A4: coupling disclosed and decomposed; Eq. (3) consistent with the implementation; bootstrap and multi-start uncertainty.
  - A6: top-k for every ranker; learning curve; calibration versus profiling every key.
  - A7: graph replay on two GPUs and packed calls on one, with claims scoped to them.
- **Partly met:**
  - **A5:** device time is now primary and the a priori selection carries paired intervals. However, the method's accuracy headline ("3.0 to 6.2%") is per call only, and its latency-weighted counterpart, about 27% of scan time on the RTX PRO 6000 under the a priori form, is not reported. The pooled 1.4B transfer still hides 13.9% at batch 4.
  - **A8:** parallel-deployment claims are proportionate and the recipe is runtime-grade, but the supplement is 31 pages rather than about 20, its protocol and platform tables omit four new runs, and a few notation slips remain.
  - **A1:** the PDFs comply (page limit, running head, availability statement, anonymity), but the anonymized artifact the paper promises does not yet exist in a clean, current form.

None of the residual gaps needs new GPU time. All are text or offline re-analysis of stored files, plus the archive rebuild. With them fixed, the manuscript would satisfy the taco-r1 acceptance criteria as a characterization study. As it stands, it is close to submission-ready but not yet complete.
