# Handoff - cycle taco-r1 (ACM TACO), 2026-10-15

Written when the local Windows machine (NZA1LT32KT324) was shut down mid-cycle, so that any session
(local or claude.ai/code cloud) can resume. Read this file, then `docs/revision_suggestions.tex`,
`docs/progress.md` (status block and `# 遗留问题`) and `python src/revloop/revstate.py status`.

## 1. Where the cycle stands

- Target venue: ACM TACO (new paper; acmsmall; initial submission <= 20 pages excluding references;
  double-anonymous; first-page footnote "New Paper, Not an Extension of a Conference Paper").
- Review: `docs/revision_suggestions.tex` (5-reviewer panel, Major Revision; reports
  `docs/review_reports/taco_r1_*.md`). 67 items registered in the ledger (cycle taco-r1).
- Author decisions (2026-10-15):
  1. Narrative A: characterization of chunk selection and tuning economics; the cost model is a
     single-kernel case study; the bound is a lemma; title without "No-Gain Bound".
     Current title (proposed, author may change): "Shape Sets the Chunk: Selecting and Keying the
     Chunk of Chunked Linear-Recurrence GPU Kernels".
  2. Cite arXiv:2604.10597 in the third person (key `corey2026preprint`).
  3. GPU work only on the RTX PRO 6000 (no data-centre GPU rented) and on the 4c lab host (two GPUs).
  4. The TACO cover letter must not mention the earlier TPDS rejection (done in `docs/cover_letter.txt`).
  5. Code is submitted separately as an anonymized archive (no link in the paper).

## 2. Manuscript rewrite (paper/main.tex) - status

Done (this session):
- Title, abstract (3 numbers), CCS/keywords, introduction (three questions), three contributions.
- Sections 3+4 merged into "Shape Keys and Data-Dependent Dispatch" (labels sec:method, sec:why kept);
  Proposition -> Lemma "Headroom bound" (label thm:nogain kept), former part (b) restated as assumptions
  to check; Fig. 1 (system path) removed.
- Section 4 rewritten as "Case Study: A Compile-Informed Cost Model": launch-table coupling disclosed,
  spill input defined as the compiler's static spill size, bandwidth floor disclosed, "What transfers"
  replaces the method recipe, GDN extrapolated optimum removed from Table 2, sequence-parallel claim dropped.
- Methodology: kernels text corrected, platforms text, new "Conventions" box (device time primary,
  launch-inclusive secondary = host-bound, graph replay, warm L2, regret summaries, ties within 2%).
- Discussion: data-dependent dispatch paragraph scoped; runtime-grade recipe (M8a).
- Conclusion answers the three questions.
- New cite keys used in main.tex (must exist in ref.bib): corey2026preprint, yotov2003comparison,
  xie2015crat, kayiran2013neither, swann2025tritonblas.

Remaining (in this order):
1. Results restructure with the numbers of the two analysis scripts (section 4 below):
   - results intro: host-bound vs device-bound framing, latency-weighted figures next to per-call ones;
   - Tables oracle_regret / predictors / baselines in device time (round medians) as primary, launch-inclusive
     moved to the supplement; "Tied" columns -> share within 2% of the oracle;
   - sec:res_model: a priori (leave-one-GPU-out) selection as the method's accuracy; top-k for every ranker;
     learning curve and cost comparison (M5a, M5b, M4c); paired CIs;
   - sec:res_autotune: launch-table decomposition at fixed (D_b, n_w) (M3a); "most of the gain from the chunk"
     per GPU;
   - sec:res_reference: TFLA as sanity check only (M5c); TP pooled-vs-pooled, retune-at-shard reference, H800
     calibration-width result, per-shard maxima (M7a); CUDA-graph table in the main text with RTX 3090 and,
     when E26 is in, RTX PRO 6000 (M6a, M9c); packed varlen (E27, M6b); E28 TP with collectives (M7b);
   - sec:res_breakeven: economics table; deployment units with best static chunk per GPU as baseline (M3d);
   - sec:res_invariance: placement finding promoted (M1e); H800 scoped (no placement control);
   - Limitations: update for E26/E27/E28 and the new conventions.
2. Related work: add the positioning paragraph (tritonBLAS, Roller, Pruner, Yotov, CLTune, KTT, Kernel Tuner
   BO, ATF, Seer, Kernel Launcher, NeuSight, CRAT/Kayiran, Stream-K, FFTW/OSKI/ATLAS, cuBLASLt heuristics,
   FlashAttention-2, Lightning Attention-2, fused Mamba-2 SSD blog) with the keys the bib agent added; fix the
   "Input-aware variant selection" paragraph (prior entropy-scheduler preprint).
3. Every minor item N1-N19 (see revision_suggestions.tex) and V1 (all non-reference text within 20 pages),
   V4 (an availability sentence: code and data as an anonymized supplementary archive).
4. Build (Windows: `cmd /c paper\build.bat`; Linux/cloud: `bash paper/build.sh` after installing TeX Live:
   `apt-get install -y texlive-latex-extra texlive-publishers texlive-science latexmk`); check page limit,
   undefined refs, overfull boxes; then the Stage 9' re-check with a fresh agent and the Stage 11 gate.

## 3. Experiments

| ID | Host | What | State | How to continue |
|---|---|---|---|---|
| E26a-d | pro6000 | CUDA-graph replays (3 rounds) of SSD 370M/2.7B, GDN 1.4B, Triton Mamba-1 370M on the RTX PRO 6000, same traces as the RTX 3090 E25 (`/autodl-fs/data/corey_r4_traces_vgpu3090`) | prepped, host in no-GPU mode | when the author switches the PRO 6000 to GPU mode: `wsl bash src/revloop/remote.sh sync pro6000`, `remote.sh launch pro6000 taco-r1 src/experiments/remote/taco_lane_pro6000.sh`, then `remote.sh watch pro6000 E26c E26a E26b E27 E26d` (no --shutdown: shared with HKVLM). Host git push may fail (token); fetch with `remote.sh fetch pro6000 <IDs>` |
| E27 | pro6000 | packed variable-length SSD/GDN calls, 170 steps from Azure 2023 traces (`packed_steps_taco.json`), fresh and carried state | prepped | same lane (step E27) |
| M2b | pro6000 | sm_86 compile facts of the runtime-chunk reference scan | killed by the no-GPU memory limit | in GPU mode: `remote.sh exec pro6000 src/experiments/remote/taco_r1_prep.sh` (TACO_REF_FACTS=1) -> `refscan_compile_facts_sm86.json`; then refit the reference-scan instance offline |
| E28 | 4c | two-GPU TP=2 with overlapping NCCL all-reduce (`e28_tp_collectives.py`) | lane `taco_lane_4c.sh` running in screen `taco-4c` on 4c since 2026-10-15 15:06 NZDT: waits (up to 72 h) for a free NVLink pair (2-3, then 0-1; all GPUs busy with user kjp), then smoke test, then full run | results stay on 4c (no GitHub credentials there); fetch from a machine on the lab network: `wsl bash src/revloop/remote.sh fetch 4c E28`; code updates: `wsl bash src/experiments/remote/taco_push_4c.sh` |

A cloud session cannot reach the PRO 6000 (no ssh in the cloud container) or 4c (lab-internal address);
GPU orchestration must run from a machine with WSL access (this laptop or another lab machine).

## 4. Background agents of 2026-10-15 (local session) - tasks and status

The four agents worked on disjoint files. Their final reports are summarized below; to resume an
unfinished agent, give a new agent the task text and the status line.

### A. Offline analyses A - reporting conventions (`src/experiments/tpds/taco_report_numbers.py` -> `taco_report_numbers.json`)
Task: one script computing (1) latency-weighted figures next to every per-call headline (CUDA-graph defaults
on the RTX 3090 for SSD 370M/2.7B and GDN 1.4B, incl. ms per forward pass and share of forward-pass scan
time; Triton Mamba-1 best single chunk per batch in device time on the three GPUs; cost model device time);
(2) device-time primary selector table for the 370M Triton Mamba-1 calls on three GPUs (best single chunk,
entropy rule, entropy only, shape tree, shape+entropy, exact-shape table held-out, cost model, three-parameter
restriction, nearest key, interpolation, GBR/MLP) with mean/p95/max regret and within-2% share; (3) M4d
distinct shapes, on/off calibration-key regret, per-shape unweighted regret with shape bootstrap CIs; (4)
cluster-bootstrap CIs and per-batch breakdowns, paired CIs model minus baselines; (5) best static chunk per
GPU (and per batch) as the untuned baseline for the deployment-units table and the calibration-budget figure;
(6) TP pooled-vs-pooled, per-shard maxima, retuning cost; (7) GDN invariance df and pairwise TOST bounds;
(8) normalized CMI, coverage above the launch floor for SSD 2.7B/GDN, max regret next to p95 for the cost
model. Run with WSL `python3` (has torch/triton/pandas/scipy/sklearn): `PYTHONPATH=. python3 -m
src.experiments.tpds.taco_report_numbers`.
Status: (filled in below from the agent's final report)

### B. Offline analyses B - cost model and baselines (`src/experiments/tpds/taco_model_baselines.py` -> `taco_model_baselines.json`)
Task: (1) top-1/2/3 for every ranker (cost model, three-parameter, work-term, nearest key, log-linear
interpolation, length/shape trees, GBR/MLP, random-k, a new zero-parameter compile-facts rule); (2) a priori
model-form selection (leave-one-GPU-out with a fewer-parameters tie rule; leave-one-calibration-shape-out);
(3) calibration fit quality (full-protocol timings only, OLS vs robust, 50 multistarts, bootstrap over the 12
keys for parameter and regret CIs, R^2 of the minimized objective); (4) bandwidth floor binding and every
difference between the implemented predictor and Eq. (3); (5) launch-table coupling: chunk effect at fixed
(D_b, n_w, num_stages) from the e4 184-configuration sweeps, registers/spills per default entry; (6)
economics: learning curve over k calibration keys, total cost calibration vs profile-every-key (captured
workload and serving-trace key counts, with and without compile cache), payback key count, cold-start regret;
(7) TFLA's own runtime expression on TFLA's own kernel timings. Checks: reproduce 0.9/3.2/3.5 (LI) and
0.38/3.55/3.01 (device time).
Status: (filled in below)

### C. Supplement cleanup (`paper/appendix.tex` only)
Task: remove legacy measurements and platforms (M10a); fix inconsistencies (GDN calls captured, Table S..
"summarized in main", GDN invariance F(6,12), TOST pooled vs pairwise) (M10b); platform facts from metadata
(M10c); pseudo-code of the Triton kernel incl. the launch table, placed in sec:supp_kernel_experiments with
label supp_kernel_pseudocode, and the runtime-chunk reference-scan patch (M10d); reduce the supplement from 48
to about 30 pages without deleting evidence cited by main.tex or any label main.tex references through \Sref
(M9c); remove history-revealing phrasing; define the serving replay's step/decode model and denominators;
state whether warm-ups fall inside the spin window; rename "Proposition ... (b)" references to
"Lemma~\Mref{thm:nogain}" / "the assumptions stated after Lemma ...".
Status: (filled in below)

### D. Related-work BibTeX (`paper/ref.bib` only)
Task: verify online and append entries for: corey2026preprint (arXiv:2604.10597), tritonBLAS
(arXiv:2512.04226), Yotov PLDI 2003, Pruner ASPLOS 2025, Seer CGO 2024, Kernel Launcher iWAPT 2023, KTT FGCS
2020, Kernel Tuner BO PMBS 2021, ATF TACO 2021, CRAT MICRO 2015, Kayiran PACT 2013, NeuSight ASPLOS 2025,
FFTW3 Proc. IEEE 2005, OSKI 2005, Stream-K PPoPP 2023, FlashAttention-2 ICLR 2024, CLTune MCSoC 2015,
Lightning Attention-2 arXiv 2024, PyTorch blog "Accelerating Mamba2 with Kernel Fusion" (2026), cuBLASLt
heuristics doc, ATLAS SC 1998.
Status: (filled in below)

## 5. Agent final reports
### D. Related-work BibTeX - DONE
All 21 entries verified and appended to paper/ref.bib (99 entries; balanced; ASCII). Keys:
corey2026preprint, swann2025tritonblas, yotov2003comparison, qiao2025pruner, swann2024seer,
heldens2023kernellauncher, petrovic2020benchmark, willemsen2021bayesian, rasch2021atf, xie2015crat,
kayiran2013neither, lee2025neusight, frigo2005fftw3, vuduc2005oski, osama2023streamk (PPoPP poster; full
paper arXiv:2301.03598), dao2024flashattention2, nugteren2015cltune, qin2024lightning2, astra2026mamba2fusion
(PyTorch blog, Astra, Dao, Hoque, Feb 2026), nvidia2026cublaslt, whaley1998atlas.
Notes for the prose: the published titles of xie2015crat and lee2025neusight do not contain "CRAT" /
"NeuSight" - name the systems in the text; IEEE metadata of the PACT 2013 paper (kayiran2013neither) is
scrambled (DOI shows another title) - drop the doi field if it confuses. Not yet test-compiled with BibTeX.

### B. Offline analyses B - PARTIAL (1 of 7 items done)
Script src/experiments/tpds/taco_model_baselines.py runs end to end in WSL (PYTHONPATH=. python3 -m
src.experiments.tpds.taco_model_baselines, < 1 min): fast numpy copy of the cost-model predictor and fit
(identical to cost_model.predict; refits reproduce the stored parameters exactly), checks pass (device time
0.383/3.554/3.008, launch-inclusive 0.878/3.249/3.543 for PRO 6000/RTX 3090/H800; tags pro6000_r4spin,
vgpu3090_r3spin, h800_r3spin).
Done - M3b: the bandwidth floor never binds (0/2298 calibration points, 0/14160 captured-call candidates, 0
chunk choices changed); four extra terms (tau_f, tau_d, tau_l, tau_x) are fixed at 0 in every fit. Code vs
Eq. (3): floor max(T, bytes/bandwidth), bytes = 2*B*d*L*4 + 2*2*B*16*L; omega = (spill store + spill load
bytes)/4; mu uses spill store bytes x 32 x m x n_w against the nominal L1 (no shared-memory carveout);
occupancy ignores shared memory, registers rounded up to 8 per thread; unaligned lengths use unaligned compile
facts; e = e_D*e_C with clamping; soft-l1 fit, f_scale 0.1, three restarts (scales 1, 0.3, 3).
Remaining (plans in the script's PLANNED dict): M5a top-k for every ranker (reuse
cost_model_baselines_r4.fit_selectors; ~5 min per tag; existing r4 top-2 on RTX 3090 LI: three-parameter 1.75%
vs cost model 2.30%); M4c a priori selection (leave-one-GPU-out, fewest parameters within 0.1 points; known
exact tie on PRO 6000 LI: dropping tau_e or tau_p gives 0.878% = full); M3c refit variants, OLS, 50
multistarts, 200-replicate bootstrap over the 12 keys, R^2 of the minimized objective (the 12 calibration
clusters cover all 2298 points); M3a launch-table decomposition - registers per default entry on sm_120
(aligned) 64/126/202/202/255/255 for C = 32..1024 (flat from 128 to 256: "registers grow with C" refuted;
spills start at C = 512); table provenance per kernels.py: "a short offline search on the RTX 3070 at
d=2048, L in {512, 2048, 8192}"; M5b learning curve/cost table/cold start (H800 has no e5/e5b/serving data);
M5c TFLA expression on TFLA timings (RTX 3090 only, e24_tfla_vgpu3090_r4spinb/_r4plainb, 63 shapes; TFLA's
expression is proportional to (L/C)*g(C), so it can only pick one static chunk; library default = best static
= chunk 128 at 1.36% device time).