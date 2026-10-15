# Supplement numbering: reviewed version, pre-compression version, current version

Cycle taco-r1, written 2026-10-16. The supplement was renumbered twice after the Stage 9 review: it was
condensed from 48 to 35 pages on 2026-10-15 and compressed from 39 to 29 pages on 2026-10-16 (item M9c).
Supplement numbers written by hand in the review reports, `docs/progress.md` and the experiment ledger
refer to the version of their date. Use this file to find the item in the current PDF.

How the tables below were made: each version of the supplement was compiled from its own source and the
labels in its `.aux` file were matched by LaTeX label (tables, figures) or by heading (sections).

| Version | Source | Supplement |
|---|---|---|
| Reviewed (Stage 9 review reports `taco_r1_*.md`) | paper commit `1428895` (2026-10-15) | 48 pages; Sections S1-S17, Tables S1-S47, Figs. S1-S7 |
| Before compression (before M9c) | `paper/backup/taco_r1/appendix_before_m9c.tex` | 39 pages; Sections S1-S15, Tables S1-S42, Figs. S1-S6 |
| Current | `paper/appendix.tex` (build of 2026-10-16) | 29 pages; Sections S1-S15, Tables S1-S37, Figs. S1-S6 |

The main text cites the supplement through LaTeX cross-references, so `main.pdf` always carries the current
numbers (checked 2026-10-16: 42 distinct supplement labels, all resolved, no `??`). The Code Ocean README
maps result files to paper items by label and is regenerated from the build at every capsule build.

## Current supplement

| Current | Caption (start) |
|---|---|
| Table S1 | Measurement protocol per experiment. GPUs: PRO = RTX PRO 6000, 3090 = RTX 3090; al |
| Table S2 | Platforms (memory and SM clock as reported by the driver). All three run Python3 |
| Table S3 | Tie shares under three criteria (%): CI overlap / within 2% / within 5% of th |
| Table S4 | Output error against a sequential fp32 reference scan ( L=300 , not a multiple o |
| Table S5 | TOST equivalence of value invariance (alpha = 0.05): passing configurations at margin |
| Table S6 | Does entropy add information beyond the shape? M1: Triton Mamba-1 scan (370M unl |
| Table S7 | Auto-tuning per shape key on the RTX PRO 6000 and the RTX 3090: latency (ms) of th |
| Table S8 | Isolated per-call decision cost of the entropy scheduler on captured Mamba-1 act |
| Table S9 | Triton Mamba-1 kernel at its best chunk versus the reference CUDA selective scan |
| Table S10 | Value versus placement (L in {2048, 8192}, batch 1, both modes, five rounds; Trito |
| Table S11 | Kernel-level results on three GPUs (same software except mamba\_ssm; launch-incl |
| Table S12 | Notation of the main paper. |
| Table S13 | Compile facts of the default launch table (chunk C, channel block D_b , warps n |
| Table S14 | Cost-model parameters per GPU (device time): the stored fit and the 95% interva |
| Table S15 | Calibration-only rankers with the k best-ranked chunks profiled per exact shape  |
| Table S16 | Learning curve of the cost model over the number of calibration keys (device tim |
| Table S17 | Cost-model parameters and validation (launch-inclusive fit; per-call relative re |
| Table S18 | Ablation of the cost model: explained variance of log latency on the fitting dat |
| Table S19 | Calibration-only chunk selectors (fitted on the calibration timings of each GPU, |
| Table S20 | Work-term model of the multi-kernel pipelines: fitted parameters ( F in ms, a an |
| Table S21 | Reference CUDA scan with the stock thread pairing and at a fixed 32 threads per  |
| Table S22 | Triton Mamba-1 scan, 370M checkpoint (48 layers), captured calls per length buck |
| Table S23 | Best single chunk per batch size (mean per-call relative regret in %, in parent |
| Table S24 | Mean per-call relative regret (%) at tensor-parallel widths on the RTX 3090 and  |
| Table S25 | Synthetic sweep of the production kernels at larger widths (RTX 3090, launch-incl |
| Table S26 | Captured calls of the production kernels at larger widths (RTX 3090, device time) |
| Table S27 | TFLA mLSTM forward kernel: regret of single chunks, tables and a fitted model ov |
| Table S28 | Both timing conventions on the captured Mamba-1 prefill calls (batch 1, 4 and 16 |
| Table S29 | The Mamba-1 rows of main-text Tables (n) and (n) in |
| Table S30 | 95% prompt-cluster bootstrap intervals (2000 replicates, percentile intervals)  |
| Table S31 | 95% prompt-cluster bootstrap intervals (2000 replicates) of the mean per-call r |
| Table S32 | CUDA-graph replays against eager (launch-inclusive) and device-time timings of t |
| Table S33 | Practitioner summary: mean per-call regret (%) of the default chunk (SSD 256, G |
| Table S34 | Packed variable-length prefill steps (170 per row, RTX PRO 6000, device time, rou |
| Table S35 | Absolute oracle headroom over an exact-shape table (us per call; seed 20260930,  |
| Table S36 | Replayed serving traces (RTX PRO 6000 break-evens). Keys: distinct bucket keys (l |
| Table S37 | Routed end-to-end study (RTX 3090, Mamba-1 370M, one session, 30 rounds, 32 greed |
| Fig. S1 | Pseudo-code of the Triton Mamba-1 scan (forward pass, one program). B, C : the  |
| Fig. S2 | Kernel latency at L=8192 , batch 1, for every chunk and eight input distribution |
| Fig. S3 | Measured latency of the captured Mamba-1 calls (markers: median over the calls o |
| Fig. S4 | Regret of the Triton Mamba-1 scan on the 2360 captured calls of each GPU (per-ca |
| Fig. S5 | Break-even number of calls per shape key, Eq.( (n) ) of the main paper |
| Fig. S6 | Activation entropy (K=256) of each captured call versus its per-call oracle ch |

Equation numbers (S1, S2, ...) are the same in all three versions.

## Before compression (39 pages) to current

Sections and figures did not change, except that Subsections S13.1 (Estimation protocol) and S13.2
(Variants) were merged into Section S13. Tables:

| Before compression | Current | Caption (start) |
|---|---|---|
| Table S1 | Table S1 | Measurement protocol per experiment. GPUs: PRO = RTX PRO 6000, 3090 = R |
| Table S2 | Table S2 | Platforms (memory and SM clock as reported by the driver). All three r |
| Table S3 | Table S3 | Tie shares under three criteria (%): CI overlap / within 2% / within |
| Table S4 | Table S4 | Output error against a sequential fp32 reference scan ( L=300 , batch  |
| Table S5 | Table S5 | TOST equivalence of value invariance (alpha = 0.05): passing configurations |
| Table S6 | removed; counts in Table S10 and the text of Section S1.3 | Fixed-chunk configurations with a significant input effect on latency  |
| Table S7 | removed (per-chunk latency at L=8192: Fig. S2) | Kernel latency (ms) per chunk at batch 1, Gaussian input, act mode, on |
| Table S8 | Table S6 | Does entropy add information beyond the shape? Per data set (M1: Trito |
| Table S9 | Table S7 | Auto-tuning per shape key on the RTX PRO 6000 and the RTX 3090: latency  |
| Table S10 | Table S8 | Isolated per-call decision cost of the entropy scheduler on captured M |
| Table S11 | Table S9 | Triton Mamba-1 kernel at its best chunk versus the reference CUDA sele |
| Table S12 | removed; paragraph "Redraws and placement" in Section S1.3 | Control run (RTX PRO 6000, ten rounds): between-distribution range (Dis |
| Table S13 | Table S10 | Value versus placement (L in {2048, 8192}, batch 1, act and all modes, f |
| Table S14 | Table S11 | Kernel-level results on three GPUs (same software except mamba\_ssm; l |
| Table S15 | Table S12 | Notation of the main paper. |
| Table S16 | Table S13 | Compile facts of the default launch table (chunk C, channel block D_b |
| Table S17 | Table S14 | Cost-model parameters per GPU (device time): the stored fit and the 95 |
| Table S18 | Table S15 | Calibration-only rankers with the k best-ranked chunks profiled per ex |
| Table S19 | Table S16 | Learning curve of the cost model over the number of calibration keys ( |
| Table S20 | Table S17 | Cost-model parameters and validation (per-call relative regrets in per |
| Table S21 | Table S18 | Ablation of the cost model: explained variance of log latency on the f |
| Table S22 | removed | Selection of the terms of Eq.( (n) ) without the target GP |
| Table S23 | Table S19 | Calibration-only chunk selectors (fitted on the calibration timings of |
| Table S24 | removed; calibration-only rankers with k profiled chunks: Table S15 | Cost model with the k best-ranked chunks profiled per exact shape: per |
| Table S25 | Table S20 | Work-term model of the multi-kernel pipelines: fitted parameters ( F i |
| Table S26 | Table S21 | Reference CUDA scan with the stock thread pairing and at a fixed 32 th |
| Table S27 | Table S22 | Triton Mamba-1 scan, 370M checkpoint (48 layers), captured calls per l |
| Table S28 | Table S23 | Best single chunk per batch size (mean per-call relative regret in %, |
| Table S29 | Table S24 | Mean per-call relative regret (%) at tensor-parallel widths on the RT |
| Table S30 | Table S25 | Synthetic sweep of the production kernels at larger widths (RTX 3090, l |
| Table S31 | Table S26 | Captured calls of the production kernels at larger widths (RTX 3090, de |
| Table S32 | Table S27 | TFLA mLSTM forward kernel: regret of single chunks, tables and a fitte |
| Table S33 | Table S28 | Both timing conventions on the RTX 3090 and, in the last three rows, th |
| Table S34 | Table S29 | The Mamba-1 rows of main-text Tables (n) and (n) |
| Table S35 | Table S30 | 95% prompt-cluster bootstrap intervals (2000 replicates, percentile i |
| Table S36 | Table S31 | 95% prompt-cluster bootstrap intervals (2000 replicates) of the mean  |
| Table S37 | Table S32 | CUDA-graph replays against eager (launch-inclusive) and device-time ti |
| Table S38 | Table S33 | Practitioner summary: mean per-call regret (%) of the default chunk ( |
| Table S39 | Table S34 | Packed variable-length prefill steps (170 per row, RTX PRO 6000, device |
| Table S40 | Table S35 | Absolute oracle headroom over an exact-shape table (us per call; seed  |
| Table S41 | Table S36 | Replayed serving traces (RTX PRO 6000 break-evens). Keys: distinct shap |
| Table S42 | Table S37 | Routed end-to-end study (RTX 3090, Mamba-1 370M, one session, 30 rounds |

## Reviewed version (48 pages) to current

Sections (matched by heading):

| Reviewed | Current |
|---|---|
| S1 Experimental Details; S1.1-S1.3 | S1; S1.1-S1.3 (same) |
| S2 Replication on Other GPUs | S2 |
| S3 A Data-Center GPU | merged into S2 |
| S4 Cost Model of the Chunked Scan: Details | S3 |
| S5 Cumulative Cost of Tuning versus Online Scheduling | S4 |
| S6 Tensor-Parallel Widths | S5 |
| S7 Production Kernels at Larger Widths | S6 |
| S8 A Public Kernel with a Wide Chunk Range | S7 |
| S9 Timing Conventions | S8 |
| S10 CUDA-Graph Timing | S9 |
| (none) | S10 Packed Variable-Length Calls (new) |
| S11 Oracle Headroom and Timing Noise | S11 |
| S12 Key Churn in Serving Traces | S12 |
| S13 Entropy Estimator; S13.1-S13.3 | S13 (subsections merged; S13.3 bin-count sensitivity removed) |
| S14 Cost of Computing the Statistic; S14.1-S14.2 | removed |
| S15 Routed End-to-End Study: Additional Results; S15.1-S15.2 | S14 (S15.1 decision cost and S15.2 legacy measurements removed) |
| S16 Activation-Entropy Statistics of Real Checkpoints; S16.1-S16.3 | removed |
| S17 Further Related Work | S15 |

Tables and figures (caption of the reviewed version):

| Reviewed | Current | Caption (start) |
|---|---|---|
| Table S1 | Table S1 | Measurement protocol per experiment. LI: launch-inclusive; dev.: devic |
| Table S2 | Table S2 | Platforms and software stacks. "Routed" denotes the patched selectiv |
| Table S3 | Table S3 | Tie shares under three criteria (%): CI overlap / within 2% / within |
| Table S4 | Table S4 | Output error against a sequential fp32 reference scan ( L=300 , batch  |
| Table S5 | removed; counts in Table S10 and the text of Section S1.3 | Fixed-chunk configurations with a significant input effect on latency  |
| Table S6 | removed; paragraph "Redraws and placement" in Section S1.3 | Control run: between-distribution range (spread of the per-input mean  |
| Table S7 | Table S5 | TOST equivalence of value invariance (alpha = 0.05): passing configurations |
| Table S8 | Table S10 | Value versus placement (RTX 3090): configurations with a significant in |
| Table S9 | removed (per-chunk latency at L=8192: Fig. S2) | Kernel latency (ms) per chunk at batch 1, Gaussian input, act mode: me |
| Table S10 | Table S6 | Does entropy add information beyond the shape? Per data set (M1: Trito |
| Table S11 | Table S7 | Auto-tuning per shape key (RTX PRO 6000): latency (ms) of the tuned ker |
| Table S12 | Table S8 | Isolated per-call decision cost of the entropy scheduler on captured M |
| Table S13 | Table S9 | Triton Mamba-1 kernel at its best chunk versus the reference CUDA sele |
| Table S14 | removed | RTX 3090: kernel latency (ms) per chunk at batch 1, Gaussian input, act |
| Table S15 | removed; counts in Table S10 and the text of Section S1.3 | RTX 3090: fixed-chunk configurations with a significant input effect on |
| Table S16 | merged into Table S7 | RTX 3090: auto-tuning per shape key, as in Table tab:supp_autotune (glo |
| Table S17 | merged into Table S10 | Value versus placement on the RTX PRO 6000 (Triton kernel, L\ 2048,8192\ |
| Table S18 | removed; paragraph "A warm and a cold L2" in Section S1.3 | Placement effect with a warm and a cold L2 cache (RTX PRO 6000, Triton k |
| Table S19 | Table S11 | Kernel-level results on three GPUs (same software except mamba\_ssm; l |
| Table S20 | Table S13 | Compile facts of the default launch table (chunk C, channel block D_b |
| Table S21 | Table S17 | Cost-model parameters and validation (per-call relative regrets in per |
| Table S22 | Table S18 | Ablation of the cost model: explained variance of log latency on the f |
| Table S23 | removed | Selection of the terms of Eq.( (n) ) without the target GP |
| Table S24 | Table S19 | Calibration-only chunk selectors (fitted on the calibration timings of |
| Table S25 | removed; calibration-only rankers with k profiled chunks: Table S15 | Cost model with the k best-ranked chunks profiled per exact shape: per |
| Table S26 | Table S20 | Work-term model of the multi-kernel pipelines: fitted parameters ( F i |
| Table S27 | Table S21 | Reference CUDA scan with the stock thread pairing and at a fixed 32 th |
| Table S28 | Table S22 | Triton Mamba-1 scan, 370M checkpoint (48 layers), captured calls per l |
| Table S29 | Table S23 | Best single chunk per batch size (mean per-call relative regret in %, |
| Table S30 | merged into the last rows of Table S11 | Batch-1 chunk tuned on one GPU (rows; the chunk in parentheses) and it |
| Table S31 | Table S24 | Mean per-call relative regret (%) at tensor-parallel widths on the RT |
| Table S32 | Table S25 | Synthetic sweep of the production kernels at larger widths (RTX 3090, l |
| Table S33 | Table S26 | Captured calls of the production kernels at larger widths (RTX 3090, de |
| Table S34 | Table S27 | TFLA mLSTM forward kernel: regret of single chunks, tables and a fitte |
| Table S35 | Table S28 | Both timing conventions on the RTX 3090 and, in the last three rows, th |
| Table S36 | Table S29 | The Mamba-1 rows of main-text Tables (n) and (n) |
| Table S37 | Table S30 | 95% prompt-cluster bootstrap intervals (2000 replicates, percentile i |
| Table S38 | removed; paragraph "A cold L2" in Section S8 | Warm versus cold L2 at the twelve calibration keys (RTX PRO 6000, Triton |
| Table S39 | Table S32 | CUDA-graph replays against eager (launch-inclusive) and device-time ti |
| Table S40 | Table S35 | Absolute oracle headroom over an exact-shape table (us per call; seed  |
| Table S41 | Table S36 | Replayed serving traces (RTX PRO 6000 break-evens). Keys: distinct shap |
| Table S42 | removed with Section S13.3 | Bin-count sensitivity of the normalized entropy signal ( 10^6 CPU samp |
| Table S43 | removed with Section S14 | Decision cost of the inline entropy hook on all 48 layers of the Mamba |
| Table S44 | removed with Section S14 | Passive entropy hook on real checkpoints (512-token prompt, one sample |
| Table S45 | Table S37 | Full routed end-to-end study of Section (n) of the main p |
| Table S46 | removed with Section S16 | Per-prompt entropy (bits, K=256 ) of the token-embedding output of the |
| Table S47 | removed with Section S16 | Per-layer entropy (bits, K=256 ) of hand-constructed and generated pro |

| Reviewed | Current | Caption (start) |
|---|---|---|
| Fig. S1 | Fig. S2 | Kernel latency at L=8192 , batch 1, for every chunk and eight input di |
| Fig. S2 | removed | Activation entropy (K=256) of the captured scan inputs per workload  |
| Fig. S3 | removed | Cumulative distribution of the held-out per-call relative regret of ea |
| Fig. S4 | Fig. S3 | Measured latency of the captured Mamba-1 calls (markers: median over t |
| Fig. S5 | removed | Cumulative cost of each policy (one-time cost P plus N calls), relativ |
| Fig. S6 | Fig. S5 | Break-even number of calls per shape key, Eq.( (n) ) of the  |
| Fig. S7 | Fig. S6 | Activation entropy (K=256) of each captured call versus its per-call |

