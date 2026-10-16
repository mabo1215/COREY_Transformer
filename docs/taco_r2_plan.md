# taco-r2 working plan (author chose option B on 2026-10-16: text + offline + key GPU runs on 4c)

## GPU work on 4c (stock RTX 3090 24 GB, physical GPUs 0 and 1; lanes in src/experiments/remote/taco_lane_4c_r2.sh)
| ID | Items | What | Lane |
|---|---|---|---|
| E32 | M3b, M7e, M1d | full-model prefill as one CUDA graph: default vs held-out shape table vs best vs no-scan; SSD 370M, GDN 1.4B, SSD 2.7B; session s1, then s2 | G0 |
| E33 | M5c, M7d | SSD/GDN value invariance above the floor in graph replay, designs A (values) and C (placement), captured activations in-process; E33b aligned vs unaligned lengths | G1 |
| E34 | M2d, M2e | Triton Mamba-1 whole launch space (184 configs) at the 12 keys, full protocol | G1 |
| E35 | M5e | per-kernel breakdown (profiler: time, grid, registers, est. occupancy) | G1b |
| E35b | M5e | Nsight Compute counters (needs the owner's sudo; question 9) | manual |
Still to design if time allows: M3d packed steps under graphs at padded capture sizes; M4f graph-timed calibration keys for the work-term model; M5f re-time a stratified subset of captured calls on the stock board in two sessions; M5i re-time the five Mamba-2 shapes in graph mode; M7f bf16 vs fp16 SSD (question 11).

## Offline agents
- stats (M3a, M5a, M5h, M5b, M5i, M6a, M5d) -> docs/review_reports/taco_r2_offline_stats.md
- baselines / cost model (M2b, M2c, M2d offline, M4a-d, M7b) -> docs/review_reports/taco_r2_offline_baselines.md
- related work (M9a-d, M3c sources; ref.bib) -> docs/review_reports/taco_r2_related_work.md

## Manuscript restructure (M1, M8; one writer at a time on main.tex)
New outline (20 pages max, references excluded):
1. Introduction: three questions; disclose the in-house Triton scan as a stress case; contributions = robust findings (M1a): key = specialization signature; regime decides; placement/L2 masquerade; eager-fitted models fail under graphs; profiling vs modelling economics; "what was not known before" paragraph.
2. Background and related work: compressed lists; add M9 prior art (static pruning, shape-keyed production caches, TFLA).
3. Shape keys and headroom: lemma -> observation/remark (M1b, M1c); key = compiler specialization signature (M7d).
4. Kernels, platforms, methodology: kernel names "Triton scan (ours)" vs "CUDA scan (mamba_ssm)" (M2a); platforms incl. the stock lab-host RTX 3090; conventions table mapping convention -> regime -> tables + glossary (M8c).
5. Results by finding (M8b), each with one table/figure and a takeaway:
   5.1 values do not matter, placement does (value invariance incl. E33; placement/L2; profiled-selector information table; H800 effects M5d)
   5.2 the regime decides (host- vs device-bound; Table 7 with recoverable gains + CIs (M3a); E32 full-model graph prefill deployment table (M3b); packed steps)
   5.3 the in-house kernel as a stress case: chunk vs launch configuration (E34, Shapley; fair baselines incl. free length rule, M2b); reference CUDA scan; TP widths (demote 3090, M5b)
   5.4 modelling vs profiling: calibration-only selectors table (split from Table 5), one pre-specified a-priori protocol + all six (M4a), costs (calibration vs profiling), key churn conclusion only
   Moved to the supplement: routed study, break-even/key-churn derivation, CMI/permutation/depth-6 detail, bootstrap of the fit.
6. Discussion (~1 page, M8e): Triton autotuner / do_bench_cudagraph integration, engine graph capture modes (M3c), generality, data-centre expectations, where data-dependent dispatch can still help; corrected recipe box (M7a-c, M7f).
7. Limitations (itemized; scope to the GPUs measured; no engine/data-centre measurement, M3e/M3f).
8. Conclusion (M1d decision after E32).

## Not done this cycle (author: no rented GPU) -> scoped in text
M3e serving-engine integration, M3f data-centre replication, M4h fourth GPU, M5f second RTX PRO 6000 session.
