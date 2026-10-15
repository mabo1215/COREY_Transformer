# Devil's Advocate Review: "Shape Sets the Chunk" (ACM TACO, new submission, round 1)

Reviewed jointly: main article (`paper/main.tex`, `main.pdf`, 20 body pages + references) and Supplementary Material (`paper/appendix.tex`, `appendix.pdf`, 48 pages). Section, table and page numbers refer to the compiled PDFs (main: Sec. 1-10, Tables 1-8, Figs. 1-3; supplement: Sec. S1-S17, Tables S1-S47). Several claims were re-derived from the released result files with read-only scripts; those checks are marked **[verified]**.

## What the paper does well

The measurement work is unusually careful. It uses recurrence-preserving chunk sweeps, both timing conventions, three-round re-timings, prompt-cluster bootstraps, TOST equivalence, placement and cold-L2 interventions and CUDA-graph replays. The abstract also states the cost model's transfer failures. A skeptical reviewer cannot say that the authors hid negative results. The attack below is mostly about what the results mean and how they are framed, not about whether they were measured honestly.

## Summary of the strongest case against the paper

The title advertises two contributions, and neither survives scrutiny as a TACO-level result. The "no-gain bound" (Proposition 1) is a one-line inequality: no policy beats the per-input oracle. Its part (b) is true by definition: a data-oblivious kernel has value-independent latency. The paper's own data show that the idealized premise of (b) is false at the few-percent level. The bound is tested against an entropy scheduler that, in the authors' words, "no published runtime or paper proposes". That scheduler pays a host synchronization and runs a histogram costing 3.4 to 9.5 times the scan, so it was always going to lose. The "compile-informed cost model" is compile-informed for exactly one kernel: a Triton Mamba-1 scan that the authors wrote, which is 1.23 times slower than the production CUDA scan. Its compile facts mainly capture register spills in the authors' own launch table. Its 8-term form was selected on the evaluation calls of two of the three GPUs. Selecting the form honestly is unstable (p95 up to 63%) and sometimes loses to bilinear interpolation over the 12 calibration keys. On every independent kernel the model is worse than the library default or than a trivial table: TFLA 7.5% against a 1.4% default, the reference scan at fixed threads 49.5%, and production pipelines under CUDA graphs 12 to 22%. Its 8-to-14-minute calibration is a full auto-tuning sweep. That costs more than profiling every shape key of the evaluated workload (2.5 to 5 s) or of any replayed serving trace. For production kernels, the one regime where the chunk "matters" (CUDA graphs, one non-stock GPU) amounts to 16 to 22% only under a per-call relative mean. Weighted by latency, the defaults lose 1.0 to 2.7% of scan time **[verified]**. The parsimonious reading of the paper is therefore a null result: production chunk defaults are fine, activations carry no information about a data-oblivious kernel's speed, and a shape-keyed table suffices. This is what a TACO reader would have assumed. It is wrapped in a cost model whose only demonstrated use case is a slower kernel that the authors wrote themselves.

## Offline checks run for this review (read-only, on released result files)

| Check | Result |
|---|---|
| CUDA-graph replays, library default, per-call relative vs. latency-weighted regret (Table S39 data, round medians) | SSD 370M: 19.1% vs **2.7%** (11.7 µs/call). SSD 2.7B: 22.3% vs **2.1%** (22.4 µs/call). GDN 1.4B: 16.2% vs **1.0%** (8.5 µs/call). At batch 16: 26.0 / 28.3 / 5.7% vs 1.9 / 1.8 / 0.2%. |
| Triton Mamba-1, best static chunk at batch 1, device time, per-call vs. latency-weighted | PRO 6000 18.8% vs 15.1%. RTX 3090 26.1% vs **6.2%**. H800 28.3% vs **7.2%**. |
| Captured calls whose exact (L, B, width) equals a calibration key | 773 of 2360 (**32.8%**). Only **57 distinct exact shapes** among 2360 calls. |
| Cost-model regret (LI) on calls at a calibration key vs. off-key | PRO 6000 0.76 vs 0.94. RTX 3090 2.90 vs 3.42. H800 3.55 vs 3.54. The overlap does **not** flatter the model materially. |
| Launch configuration vs. chunk on the RTX 3090 (Table S16) | Best static chunk at the default launch configuration vs. the tuned kernel: geomean about +7%, up to **+36%** (L=2048, B=1). |

## Issue overview

| ID | Severity | Dimension | Location |
|---|---|---|---|
| DA1 | CRITICAL | So-what / overgeneralization | Title; abstract; contribution 2; Sec. 5; Table 2; Sec. 7.4-7.5; Sec. S4, S8, S10 |
| DA2 | CRITICAL | Core thesis / strawman | Title; Sec. 1 (p. 2); Sec. 3; Proposition 1 (Sec. 4, p. 5); Sec. 7.6; S14-S15 |
| DA3 | MAJOR | Data-conclusion mismatch (metric) | Abstract; contribution 1; Sec. 7.5 "Replayed as CUDA graphs" (p. 16); Sec. 10; Table S39 |
| DA4 | MAJOR | Cherry-picking (timing convention) | Sec. 7.4 p. 13 ("3.2 against 1.9%"); Tables 4-6; abstract; Table S24 |
| DA5 | MAJOR | Evidence provenance / reproducibility | Sec. 6.4; Sec. 7.8; Table S1; Sec. S9 (p. 29); Table S35 |
| DA6 | MAJOR | Confirmation bias (model selection) | Sec. 5 "Model form and data"; recipe step (4); Tables S22-S24; Table 2 (ref. scan, GDN) |
| DA7 | MAJOR | Alternative path (profiling) dominates | Sec. 7.4 "Auto-tuning baseline"; Sec. 7.7 "Key churn"; Sec. 8 recipe; Sec. S5 |
| DA8 | MAJOR | Strawman baseline / mismatched statistics | Abstract; contribution 2; Sec. 7.5 "Tensor-parallel widths"; Table S31 |
| DA9 | MAJOR | Overgeneralization (one GPU) | Abstract; contribution 1; Sec. 8 recipe; Sec. 10; Sec. S10 |
| DA10 | MAJOR | Evidence gap (premise of Prop. 1(b)) | Sec. 7.1; Sec. S3 (p. 15); Table S19 |
| DA11 | MAJOR | Alternative explanation (self-inflicted effect) | Sec. 6.2; Sec. 7.2; Sec. 7.4 "Auto-tuning baseline"; Tables S13, S16, S20; Table 8; Fig. 3 |
| DA12 | MINOR | Evaluation leakage (shape overlap) | Sec. 6.3; Sec. 7.4; Table 6 |
| DA13 | MINOR | Overclaim ("add nothing") | Abstract; contribution 3; Sec. 7.3; Table S10; Sec. S10 |
| DA14 | MINOR | Overfitting / unsupported extrapolation | Table 2 (GDN row); Sec. 6.2; Sec. 7.4 "Production pipelines"; Table S26 |
| DA15 | MINOR | Internal inconsistency | Sec. 9 vs. Sec. 7.5, Table 1 |
| DA16 | MINOR | Clarity → overclaim risk | Abstract; contribution 1; Sec. 7 intro |
| DA17 | MINOR | Strawman (pitfall) | Contribution 5; Sec. 6.1 |
| DA18 | MINOR | Weak evidence promoted to contribution | Contribution 4; Sec. 7.7 "Serving traces"; Sec. S12 |
| DA19 | MINOR | Confounds in replication | Sec. 6.3; Table S2; Sec. S1.1 |

---

## CRITICAL

### DA1 - The positive contribution has no demonstrated use outside the authors' own non-production kernel

- **Location:** Title ("A Compile-Informed Cost Model"); abstract lines 7-17; contribution 2 (p. 2-3); Sec. 5 and Table 2 (p. 6-7); Sec. 7.4-7.5 (p. 13-16); Sec. S4 (Tables S20-S24), S8 (Table S34), Table S27, Sec. S10 (Table S39).
- **Attack:**
  - **Compile facts matter for one kernel only.** The "compile-informed" part enters only for the Triton Mamba-1 scan of this study. Table 2 lists "fixed registers, stock threads" for the reference CUDA scan and "none" for Mamba-2 SSD and Gated DeltaNet. The TFLA fit (Sec. S8) is a five-term work model with no compile facts at all.
  - **What the compile facts capture is a defect of the authors' launch table.** The most important term is the spill term: without it the regret rises from 0.88 to 4.80% and from 3.25 to 12.72%, and R² falls to 0.79 and 0.70 (Table S22). Spills occur in the large-chunk and unaligned entries of the authors' default launch table (255 registers with 12 to 72 B of spills; Table S20). The production reference scan avoids spills with a fixed 168-register design (Sec. S4).
  - **The kernel is slower than the production kernel it replaces.** At its best chunk it takes 0.81 to 1.54 times the reference scan's latency, 1.23 times on geometric mean (Table S13). The reference scan's own stock length rule is within 6.6% LI (3.1% device time, Table S27) of its oracle. A practitioner would deploy the reference scan and would not need the model.
  - **Every independent test fails.** On every kernel the authors did not write, the model or its form loses to the library default or to a trivial table:
    - TFLA: 7.5% against a 1.4% default and a 0.9% batch table (Table S34).
    - Reference scan at 32 threads: 49.5% against a 0.06% length-only tree (Table S27).
    - SSD and GDN under CUDA graphs: 12 to 22% against 0.2 to 0.4% for a shape tree (Table S39).
  - **Conclusion of the attack.** The evidence supports "an 8-parameter model fits one in-house kernel on the GPUs where its form was chosen". It does not support a cost-model contribution in the title of a TACO paper.
- **Current defense:** It is partial. The abstract and conclusion state the transfer failures openly ("It does not transfer…", p. 1; Sec. 10). Sec. 5 calls the novelty "narrow". The model does select well on two unseen widths (1.4B 3.7%, 2.8B 0.8%; Table 7) and on the H800 (3.5%). Openness, however, is not a contribution. The narrowness is admitted, but the title and contribution list still frame the model as a general method ("Applying the method to a new kernel", Sec. 5), and that method fails on every new kernel tried.
- **Action:** Choose one of the following.
  - (a) **Text.** Reframe the paper as a characterization study: where the chunk matters, what predicts it, and how to measure it. Present the cost model as a case study of one kernel. Remove it from the title, and drop or qualify "Applying the method to a new kernel".
  - (c) **New GPU measurement.** Show the full method working, with calibration-only fitting and form selection without the target, on at least one production kernel where the chunk does matter. The natural candidates are SSD and GDN under CUDA graphs: calibrate on graph replays at about 12 keys on the RTX 3090 and the RTX PRO 6000, and evaluate on the captured calls. Also apply it to a second independent wide-chunk kernel with real compile facts (for example FLA chunk GLA or TFLA with registers and spills read). Estimate: about 8-12 GPU-hours (graph calibration and replays on two GPUs), plus 4-6 GPU-hours for the second kernel.

### DA2 - The "no-gain bound" is elementary and is validated against a self-built strawman

- **Location:** Title; Sec. 1, p. 2 ("to our knowledge, no published runtime or paper proposes value-dependent dispatch for dense chunked kernels. To close this design question … we study one such policy"); Sec. 3 and Fig. 1; Proposition 1 and the text after it (Sec. 4, p. 5: "The bound is elementary"); contribution 3; Sec. 7.6; Sec. S14-S15 (Tables S12, S43, S45).
- **Attack:**
  - **Part (a) is a tautology.** It says that $T(s,\pi(\cdot);Z)\ge\min_C T(s,C;Z)$, followed by a conditional expectation.
  - **Part (b) is true by definition.** Data-oblivious streams, fixed addresses and value-independent instruction timing imply value-independent latency.
  - **The premise of (b) is empirically false for the kernels studied.** With placement fixed, values still move latency by a median of 1.8% and up to 6.7% (Sec. S1.3, Table S8). Placement alone moves it by up to 43.6% (Table S17). On the H800 all 36 of 36 device-time configurations show a significant input effect (Sec. S3). The empirical work therefore rests entirely on (a), which needs no proposition.
  - **No TACO reader would expect the opposite result.** Activation entropy does not predict the speed of a dense kernel whose control flow is independent of the data.
  - **The tested policy is a strawman.** The policy is the authors' own construction, which no published system proposes (p. 2). It was implemented with a device-to-host synchronization and a histogram that is "not bandwidth-optimal" (Sec. 9). That histogram costs 3.4 to 9.5 times the scan it schedules (Sec. S15.1) and adds 24 to 71% to prefill (Table S45). These outcomes follow from the design. The routed study (Sec. 7.6) measures the implementation, not the idea.
  - **The cheap variants that could have tested the cost side were not measured.** Fused or device-side statistics are excluded (Sec. 4; Sec. 9).
- **Current defense:** It is partial. Sec. 4 concedes that the bound is elementary and argues that its use is to reduce the question "to measurable quantities per shape key, for every statistic at once". The information-side tests do cover all statistics through the measured headroom (Tables 5 and S40). That is a reasonable methodological point, but it does not make a title-level contribution. The title and contribution 3 present it as one.
- **Action:**
  - (a) **Text.** Demote Proposition 1 to a remark or lemma that motivates the headroom measurement, and drop "No-Gain Bound" from the title. State up front that value-dependent dispatch for dense kernels has no proponents, and present the entropy scheduler as a control rather than as a design question. Move the placement and L2 finding (up to 43% at a fixed chunk from buffer addresses alone, which disappears with a cold L2; Tables S17-S18) into the main text. That is the genuinely architecture-level result.
  - (b) **Offline.** Bound the cost side by the measured cost of a single device-side reduction (7.3 µs at L=2048) rather than by the unoptimized histogram.
  - Optional (c): one fused-statistic microbenchmark, about 1 GPU-hour.

---

## MAJOR

### DA3 - Per-call relative regret inflates the production-kernel result by about 7 to 16 times

- **Location:** Abstract ("Replayed as CUDA graphs (one GPU), the SSD and Gated DeltaNet defaults lose 16 to 22%"); contribution 1 ("defaults: 16 to 22%"); Sec. 7.5, p. 16; Sec. 10 ("a shape-keyed table recovers the 16 to 22% their defaults lose"); Table S39. Also the Triton batch-1 claim "one chunk loses 19 to 28% at batch 1" (abstract; contribution 1).
- **Attack:**
  - **The production numbers.** The 16 to 22% figures are means over calls of $T(c)/T(c^\star)-1$. Short calls near the 45 to 83 µs graph floor dominate them. Weighted by latency, the SSD 370M, SSD 2.7B and GDN 1.4B defaults lose **2.7, 2.1 and 1.0%** of scan time (11.7, 22.4 and 8.5 µs per call) **[verified]**. At batch 16 the per-call means are 26.0, 28.3 and 5.7%, but the latency-weighted shares are 1.9, 1.8 and 0.2%.
  - **The conclusion misstates the result.** Its sentence "recovers the 16 to 22% their defaults lose" will be read as a latency fraction, and as a latency fraction it is wrong by an order of magnitude.
  - **The same pattern holds for the in-house kernel, more mildly.** The batch-1 loss of "19 to 28%" is 15.1, 6.2 and 7.2% latency-weighted on the PRO 6000, RTX 3090 and H800 **[verified]**.
  - **Consequence for the paper's first question.** Once weighted by latency, the answer to "where does the chunk matter" for production kernels becomes "by at most about 3% of scan time, under graphs, on one GPU".
- **Current defense:** It is partial. Sec. 6.4 defines three regret summaries. Sec. 7.7 "Deployment units" gives 0.56 and 1.44 ms per forward pass for SSD under graphs, but never relates this to the forward-pass scan time (about 20 and 65 ms). The abstract and conclusion use only the per-call figure.
- **Action:** (b) **Offline.**
  - Report the latency-weighted share next to every headline regret in the abstract, contributions, Sec. 7.5 and the conclusion.
  - Add a "share of scan time per forward pass" column to Table 8.
  - Rewrite the conclusion sentence accordingly.

### DA4 - The timing convention changes from claim to claim, each time in the direction that favors the claim

- **Location:** Sec. 7.4, p. 13: "0.4% in device time on the RTX PRO 6000 … and 3.2 against 1.9% on the RTX 3090 and 3.5 against 2.6% on the H800". Tables 4 and 5 (Mamba-1 in device time), Table 6 (LI), Sec. 7.7 (headroom in device time), abstract; Tables S10, S24 and S35.
- **Attack:** The convention lines up with the conclusion each claim needs:

  | Claim | Convention used | Why it favors the claim |
  |---|---|---|
  | The chunk matters | Device time | Batch-1 static loss 18.8% vs 1.7% LI on the PRO 6000 |
  | The cost model is good | LI | Rivals look worse in LI; in device time an MLP beats the model on the PRO 6000 (0.35 vs 0.38) and the RTX 3090 (0.47 vs 3.55), and a structural MLP on the H800 (0.57 vs 3.01) (Table S24) |
  | The headroom is negligible | Device time | Headroom is smallest in device time |

  The quoted sentence in Sec. 7.4 mixes conventions inside one comparison. It uses device time for the PRO 6000, where the model is near the shape tree, and LI for the other two GPUs, where the gap looks small (3.2 vs 1.9). In device time the RTX 3090 gap is **3.55 vs 0.38 (9.3 times)**, and on the H800 it is 3.01 vs 1.63 (Tables S10 and S24). The device-time ranking of selectors is the opposite of the LI ranking that Table 6 shows in the main text.
- **Current defense:** It is partial. Sec. 6.4 explains why headroom is stated in device time. Table S24 gives both conventions. Sec. 7.4 admits that "in device time no selector wins everywhere" and claims worst-case robustness ("stays within 3.6% on every GPU"). That robustness margin over the 3-parameter restriction is thin: 3.55 vs 4.16.
- **Action:** (b) **Offline.**
  - Fix one primary convention for all headline claims. Device time with three-round medians is the reproducible one (see DA5).
  - Put the LI equivalents in a single secondary table.
  - Rewrite the Sec. 7.4 sentence so that each comparison uses one convention.
  - Show the device-time version of Table 6 in the main text.

### DA5 - The headline LI numbers come from an unguarded single-block session that did not reproduce

- **Location:** Sec. 6.4 ("tables are first-session launch-inclusive single blocks unless stated"); Sec. 7.8 last sentences; Table S1 (rows "Captured 370M calls": Guard *no*, State *no*, 1 round); Sec. S9, p. 29; Table S35 (row "LI, 3 rounds†").
- **Attack:**
  - **The guarded second session diverged from the first.** On the RTX PRO 6000 the second, guarded LI session gives a shape tree of 4.7% (first session 0.7%), a cost model of 4.9% (0.9%) and a headroom of 3.93 µs (0.45 µs). The median oracle latency is 98 µs instead of 60 µs, and for 26% of calls the first round is more than 5% slower.
  - **The headline still rests on the first session.** The abstract's "1.3 to 7.3%" lower bound (PRO 6000, 1.34%), Table 6's bold 0.88 and Fig. 3 all rest on that first session. That session ran without the exclusive-GPU guard or clock logging, on a rented container where host-load contamination did occur in another sweep (the 1.4B and 2.8B batch-1 sweep "timed under host load", Sec. S11).
  - **The reproducibility statement does not cover re-measurement.** "Reproduced exactly by the same code (80 of 80 published values)" is a re-analysis check, not a re-measurement.
- **Current defense:** It is partial. The paper discloses the non-reproduction (Sec. 7.8; Sec. S9) and states that "the conclusions hold in every convention". The non-reproducing first-session numbers nevertheless remain as headline values.
- **Action:**
  - (b) **Offline.** Restate the headline cost-model and clean-selection numbers on guarded three-round device-time data. These exist for all three GPUs: 0.38, 3.55 and 3.01 (Table S24). Rerun the clean-selection analysis of Table S23 in device time.
  - (c) If LI is kept as a headline, re-time the 2360 calls in a third guarded three-round LI session on the PRO 6000 to settle which session is representative. Estimate: about 3 GPU-hours.

### DA6 - The cost model behaves like curve fitting, and its selection procedure is not validated

- **Location:** Sec. 5 "Model form and data" (p. 6); recipe step (4) (Sec. 5) and the recipe box (Sec. 8); Tables S21-S24; Table 2 rows for the reference scan (8 parameters on 60 timings) and GDN (5 parameters on 9 timings).
- **Attack:**
  - **The form was chosen on the test data.** The terms were selected on the captured (test) calls of the PRO 6000 and the RTX 3090. The candidate set of 12 forms was itself built while looking at those calls.
  - **Honest re-selection is unstable.**
    - On the H800, leave-one-shape-out selection gives **7.25% mean and 62.5% p95** (Table S23).
    - The PRO 6000 held-out-log-error criterion gives 4.82%, worse than bilinear interpolation over the 12 keys (3.88%; Table S24).
    - The RTX 3090 "two other GPUs" criterion gives 4.52%, worse than interpolation (4.16%).
    - Selecting on the H800 alone gives 12.72% (p95 53%).
  - **The advantage over trivial interpolation therefore does not survive honest model selection on two of the three GPUs.**
  - **Several terms are redundant.** Dropping $\tau_e$ or $\tau_p$ leaves the PRO 6000 regret unchanged at 0.88 (Table S22).
  - **A smaller model wins on one GPU.** The 3-parameter restriction beats the full model on the RTX 3090 (2.49 vs 3.25 LI; 0.88 vs 3.55 in device time).
  - **The recipe is not validated as a procedure.** Its step (4) ("choose the terms on held-out calibration keys or on other GPUs") is exactly the procedure that produced 7.25% / 62.5% on the H800. Readers who follow the recipe get the unstable procedure, while the paper reports the frozen form.
- **Current defense:** It is partial. The paper is open about this: Sec. 5 says "Re-selecting the terms without them is unstable", and Sec. 9 lists it. It calls the H800 and shard widths "form-blind evidence", and the H800 frozen-form 3.5% is genuinely clean. One clean GPU is thin support for an 8-parameter method.
- **Action:**
  - (b) **Offline.** Put clean-selected forms and interpolation or nearest-key baselines side by side in one main-text table. Fix and state a single selection rule, and report only its results as the method's accuracy. Drop $\tau_e$ and $\tau_p$ if they do not earn their keep.
  - (c) **New GPU.** Add a fourth GPU (for example an A100, L40S or H100 SXM), measured after both the form and the selection rule are frozen: about 15 minutes of calibration plus about 2-3 GPU-hours for the captured calls (6 chunks × 3 rounds). Estimate: 4-6 GPU-hours.

### DA7 - Calibration costs more than profiling every key the paper ever evaluates

- **Location:** Sec. 7.4 "Auto-tuning baseline" (493 s of tuning, 1.2 s to profile six chunks at 12 keys); Table 2 ("2298; 8-14 min"); Sec. 7.7 "Key churn"; the recipe in Sec. 8 ("Dispatch: … profile the best two"; "enumerable keys are cheaper to profile directly"); Sec. S5 (profiling all 33 keys costs 2.5 and 5.0 s); Table S41 (11 to 852 bucket keys per trace).
- **Attack:**
  - **The calibration is a full auto-tuning sweep.** It covers 184 configurations at 12 keys and takes 493 to 843 s, mostly compilation.
  - **Direct profiling is cheaper.** Profiling the six chunks at every bucket key of the evaluated workload costs 2.5 to 5 s, plus about 10 s of compilation.
  - **The recipe profiles anyway.** It still profiles the model's top two chunks on first sight of a key. The model therefore saves about 0.02 to 0.03 s per key, and calibration breaks even only after roughly 16,000 to 25,000 keys. No replayed trace produces more than 852 bucket keys.
  - **The cold-start niche is small.** The latency argument (no stall at first sight) is the only remaining niche. The paper's own key-churn analysis says the model's residual is repaid by profiling after a median of about 4,600 and 3,700 calls, roughly 77 to 95 requests.
- **Current defense:** It is partial. The recipe concedes that "enumerable keys are cheaper to profile directly" and limits the model to "keys that cannot be enumerated or timed before serving". The paper never quantifies how often such keys occur, or whether a per-GPU-model table shipped once, like the calibration, would cover them.
- **Action:**
  - (b) **Offline.** Add a cost table: calibration wall-clock time vs. compile-inclusive profiling of all keys for each workload and trace. Report the key count at which the model pays back in throughput terms, and the stall it avoids in latency terms.
  - (a) **Text.** Narrow the claimed use case to what that table supports.

### DA8 - Shard widths: a strawman baseline and mismatched statistics in the abstract

- **Location:** Abstract ("needs no retuning (2.9 to 6.5%), where the unsharded table loses up to 65%"); contribution 2; Sec. 7.5 "Tensor-parallel widths" (p. 16); Table S31; Sec. S3 (H800).
- **Attack:**
  - **The statistics do not match.** "2.9 to 6.5%" is the model's pooled mean. "Up to 65%" is the worst single shard of the baseline (H800, 2.8B, t=8). The pooled baseline figures are 36.8% on the RTX 3090 and 22.1% on the H800.
  - **The baseline is one nobody would use.** The unsharded table is not a realistic alternative: t is fixed at deployment-configuration time. The paper itself names the realistic alternative ("retune at the shard width, at 0.03 to 0.05 s … per key").
  - **A table needing no retuning does better on one GPU.** The calibration-width table beats the model on the H800 (2.2 vs 2.9%). On the RTX 3090 the model's per-shard regret reaches 15.2%.
  - **The evidence is thin.** All shapes are synthetic, timed on one GPU, with no multi-GPU run.
- **Current defense:** It is partial. Sec. 7.5 mentions the retuning cost and the H800 result, but the abstract and contribution 2 keep the mismatched pair.
- **Action:** (a) **Text.**
  - Compare pooled with pooled, or worst with worst.
  - Add "retune at the shard width (seconds)" as the reference baseline.
  - State the H800 calibration-width result in the abstract or drop the claim from it.

### DA9 - The regime where production kernels need tuning was measured on one non-stock consumer GPU

- **Location:** Abstract; contribution 1 ("(one GPU)"); Sec. 7.5 p. 16; Sec. 9 ("CUDA-graph timings exist for the RTX 3090 only"); Sec. 8 recipe ("under CUDA graphs only a shape-keyed table does", no qualifier); Sec. 10 (no qualifier); Sec. S10; Table S2 (48 GB non-stock RTX 3090 board).
- **Attack:**
  - **The evidence base is narrow.** CUDA-graph execution is how serving engines actually run, so it is the deployment-relevant regime. Every graph result comes from one consumer GPU, a modified 48 GB RTX 3090 board in a rented container.
  - **The general statements drop the qualifier.** The conclusion and the recipe state the graph findings without the "one GPU" qualifier that the abstract keeps.
  - **No calibration-only graph fit exists.** The work-term model's graph failure (12 to 22%) is "fixed" only by refitting on the training folds of the replays (1.3 and 0.8% for SSD; Sec. S10). A model fitted on graph-timed calibration keys was never tried.
- **Current defense:** It is partial. "(one GPU)" appears in the abstract and contribution 1, and the limitation is listed in Sec. 9.
- **Action:**
  - (a) **Text.** Qualify the statements in the conclusion and the recipe.
  - (c) **New GPU measurement.** Replicate the graph replays for SSD, GDN and the Triton scan on the RTX PRO 6000 and the H800 (2360 + 2360 + 1770 calls × 3 to 6 chunks × 3 rounds). Calibrate the work-term model on graph-timed calibration keys. Estimate: about 3-4 GPU-hours per GPU, 6-8 GPU-hours in total.

### DA10 - Value invariance on the data-center GPU lacks the control that rescued it elsewhere

- **Location:** Sec. 7.1 ("the input distribution does not change which chunk is best"); Sec. 9 ("The H800 … without a placement control"); Sec. S3, p. 15; Table S19.
- **Attack:**
  - **The H800 data look like a violation of the premise.** All 36 of 36 device-time configurations have a significant input effect (largest 40%; 48% LI).
  - **At L=2048 the input-agnostic chunk often loses its tie.** It is tied with each input's best in only **54 to 71%** of cells, with the inputs' own best chunks "split between 256 and 1024" (regret up to 5.7%).
  - **The rescue argument was never run on this GPU.** On the other two GPUs the paper attributes such effects to placement through an intervention (Tables S8 and S17-S18). The intervention was not run on the H800. As far as the evidence goes, the "value-invariance" premise of Proposition 1(b) is unverified on the only data-center part, and the data there look like a violation.
- **Current defense:** It is partial. The gap is disclosed, and Proposition 1(a) still bounds the gain.
- **Action:**
  - (c) Run the placement and cold-L2 control on the H800 (designs A and C, L ∈ {2048, 8192}, 5 rounds). Estimate: about 1-2 GPU-hours.
  - (a) Otherwise, restrict the Sec. 7.1 claim to the two GPUs where it was controlled.

### DA11 - The large chunk effect is plausibly self-inflicted, and "default" is the authors' own choice

- **Location:** Sec. 6.2 (kernel (i)); Sec. 7.2 ("up to 8.5×"); Sec. 7.4 "Auto-tuning baseline" ("most of the gain comes from the chunk", PRO 6000 only); Tables S13, S16 and S20; Table 8 and Fig. 3 ("default chunk 256").
- **Attack:**
  - **The "chunk" is a whole launch-table entry.** In the Triton kernel each chunk is a (chunk, channel block, warps) entry of an author-defined launch table, and large-chunk entries spill (Table S20).
  - **The headline spread comes from configurations no one would use.** The 8.5× spread is a PRO 6000 maximum driven by chunk 32 at L=8192. The same spread is 2.0× on the RTX 3090 and 4.1× on the H800 (Table S19).
  - **On the RTX 3090 the launch configuration matters as much as the chunk.** The best chunk at the default launch configuration is about 7% off the tuned kernel on geometric mean and up to **36%** off (L=2048, B=1: 0.270 vs 0.198 ms; Table S16). This contradicts the main-text generalization "most of the gain comes from the chunk", which is true only on the PRO 6000. The model also fails as a full auto-tuner (23.1% mean on the RTX 3090; Table S21).
  - **The "default" is not a library default.** The Triton kernel has no library default. Chunk 256 was borrowed from Mamba-2 (Sec. S5), so the losses quoted in Table 8 (4.1 to 25 ms per pass) and Fig. 3 (23.4, 33.9 and 20.1%) are measured against a baseline the authors picked. The best static chunk per GPU (for example 128 at batch 1 on the RTX 3090) is the fair untuned baseline.
- **Current defense:** It is partial. Sec. 9 says that the model "does not replace a search of the full launch space". Table 4 reports the best static chunk.
- **Action:**
  - (b) **Offline.** Restate the auto-tuning conclusion per GPU. Use the best single chunk per GPU (or per GPU and batch) as the "default" in Table 8 and Fig. 3. Report the chunk spread over reasonable chunks (for example within 4× of the length-rule chunk).
  - Optional (c): re-sweep the chunks with the best launch configuration per chunk on the captured calls, to separate chunk effects from launch-table artifacts. Estimate: about 3 GPU-hours per GPU.

---

## MINOR

### DA12 - A third of the "untimed" target calls share their exact shape with calibration keys

- **Location:** Sec. 6.3 (truncation to 64, 128, 256, 512 and 1024 tokens); Sec. 7.4 ("2298 timings, none of them a captured call"); Table 6.
- **Attack:** 773 of 2360 captured calls (32.8%) have exactly the (L, B, width) of a calibration key, and the 2360 calls contain only 57 distinct exact shapes **[verified]**. Under the paper's own thesis that latency is value-invariant, timing a calibration key is timing those calls.
- **Current defense:** None in the paper. My offline split shows that the overlap does not inflate the model's regret: off-key vs. at-key is 0.94 vs 0.76 (PRO 6000), 3.42 vs 2.90 (RTX 3090) and 3.54 vs 3.55 (H800).
- **Action:** (b) Report the off-key subset and the effective number of distinct shapes, and soften "without timing target shapes" to "without timing the captured calls".

### DA13 - "Activation statistics add nothing" is stated too absolutely

- **Location:** Abstract (last sentence); contribution 3 ("equivalent within 0.5 points on 15 of 17 data sets"); Sec. 7.3; Table S10; Sec. S10.
- **Attack:**
  - **Some datasets show small but significant information.**
    - Entropy carries significant information given the exact key on two data sets (0.012 bits, p = 0.001, RTX 3090 device time, three rounds).
    - It lowers the Mamba-2 RTX 3090 device-time tree by 0.47 points.
    - It lowers the Mamba-1 graph tree by 0.10 points.
  - **The shape key looks incomplete.** The information disappears "given key and layer", and the layer itself carries 0.054 bits on the superseded 1.4B sweep. Layer identity is not part of the shape key, so the key that "sets the chunk" is incomplete, probably through placement or cache context.
  - **"15 of 17" counts uninformative sets.** Several of the 17 equivalence verdicts are Mamba-2 sets where the whole chunk effect is within timing noise, so equivalence there is uninformative.
- **Action:** (a) Write "no practically relevant information (at most 0.5 points)". Count only the data sets whose chunk effect exceeds the noise floor. Discuss the layer effect as a missing context feature.

### DA14 - The GDN work-term model is fitted on 9 points and extrapolated

- **Location:** Table 2 (GDN: 5 parameters, 9 calibration timings, "opt. 61/85" in the *Regret* column); Sec. 6.2 ("puts the RTX 3090 optimum near 85"); Sec. 7.4 "Production pipelines"; Table S26.
- **Attack:** Five parameters fitted to nine timings, with R² = 0.976 and 0.990, cannot support an optimum outside the tested range (16 to 64). Reporting an optimum chunk in a regret column is a category error.
- **Action:** (a) Remove the extrapolated optimum, or mark it as an untested extrapolation, and fill the column with a regret or with "--".

### DA15 - Contradiction in the Limitations section

- **Location:** Sec. 9 ("the reference CUDA scan was timed at one fixed thread count (32)") vs. Sec. 7.5 (stock pairing with 32 to 128 threads, plus a fixed 32-thread build) and Table 1 ("32-128 threads by length").
- **Attack:** As written, Sec. 9 says the reference scan was timed only at 32 threads, which contradicts the stock-pairing sweep.
- **Action:** (a) Rewrite as "besides the stock pairing, only one fixed thread count (32) was timed".

### DA16 - Density of numbers hides what is claimed

- **Location:** Abstract (more than 30 numbers across four conventions or metrics); contribution 1; Sec. 7 intro ("up to 8.5×").
- **Attack:** A reader cannot tell which number is the headline, which convention it uses, or whether it is a mean, a p95 or a maximum. Max-picked figures such as 8.5× and 65% sit next to means. This makes the overclaims in DA3, DA4 and DA8 easy to miss and hard to audit.
- **Action:** (a) Cut the abstract to one number per claim, in one convention with one statistic, and push the rest into a "claims and evidence" table.

### DA17 - The measurement pitfall is presented as a contribution without evidence that anyone makes the mistake

- **Location:** Contribution 5; Sec. 6.1.
- **Attack:** No citation shows split-call chunk timing in prior work or tools, so the guidance refutes a hypothetical practice.
- **Action:** (a) Cite instances of the practice or demote the point to methodology text.

### DA18 - The serving replay is too approximate for contribution-level claims

- **Location:** Contribution 4; Sec. 7.7 "Serving traces" and "Deployment units"; Sec. S12; Table S41.
- **Attack:**
  - **The replay is approximate at every step.** The step model is 2.3 times faster than the measured prefill. Packed calls are costed as batch-1 scans, unmeasured shapes go to the nearest key, and token counts come from foreign tokenizers.
  - **One sparse key dominates.** On the RTX 3090 a single key with 11 captured calls drives the padded-budget results.
- **Action:** (a) Present the replay as illustrative and remove its percentages from the contribution list. Optional (c): time variable-length packed calls, about 2 GPU-hours.

### DA19 - Confounds in the cross-GPU replication

- **Location:** Sec. 6.3; Table S2; Sec. S1.1.
- **Attack:** Several factors besides the GPU differ between platforms, and they confound GPU-to-GPU comparisons, especially for SSD:
  - The Mamba-2 SSD kernel comes from mamba_ssm 2.2.2 (local build) on one GPU and 2.3.2.post1 on another.
  - The RTX 3090 is a non-stock 48 GB board.
  - Clocks are unlocked on rented containers.
- **Action:** (a) State these confounds where cross-GPU differences are interpreted. Optional (c): rerun the SSD sweep with one package version, under 1 GPU-hour.

---

## Ignored alternative explanations and paths

1. **Kernel quality explains the chunk sensitivity.** The parsimonious explanation for "the chunk matters only on the Triton scan" is that its launch table spills registers at large chunks and lacks a length rule. It is not a property of chunked recurrences. The reference scan, with a fixed register budget and a stock length rule, is within 3 to 7% of its oracle (DA1, DA11).
2. **Making the kernel chunk-robust is an unexamined design path.** Examples are register-aware tile selection inside the kernel, persistent kernels, or a two-entry length rule. Such a kernel may make prediction unnecessary.
3. **A shipped per-GPU-model table is the natural competitor to a shipped per-GPU parameter set.** Both are built once per GPU type. With 11 to 852 bucket keys per trace, the table is cheap (DA7).
4. **L2 set or partition effects are an alternative explanation for the 43% placement effect.** They are not distinguished from "which part of an L2-sized working set remains cached". No counters were collected, and the evidence is one intervention.

## Missing stakeholder perspectives

- Maintainers of serving engines that run CUDA graphs and variable-length (packed) kernels. Neither regime is measured across GPUs (DA9, DA18).
- Kernel-library maintainers who choose defaults (mamba_ssm, FLA, TFLA). The paper's evidence mostly vindicates their defaults, and this is not framed as the main practical message.

## Unexamined premise

Every regret and headroom in the paper is defined against a micro-benchmark oracle. That oracle is the chunk that is fastest when the call is repeated back to back, in isolation, on a warm L2 holding its own inputs, with a fixed launch table. The paper's own data show that this context is not inert:

- Buffer placement alone moves fixed-chunk latency by up to 43% on a warm L2 (Tables S17-S18).
- The timing convention changes the oracle for about a quarter of short calls (Sec. S9).
- End to end, host dispatch hides chunk differences entirely at 256 and 976 tokens (Table S45).
- The layer index carries information beyond the shape (DA13).

The title's "shape sets the chunk" is therefore true only for a fixed execution context and memory state. That context is the micro-benchmark's, not a model's. The cold-L2 test (PRO 6000) and the routed study (RTX 3090, batch 1, three lengths) are partial checks. An in-model measurement of the per-call oracle, with the preceding kernels running, is missing.

## Observations (non-defects)

- The disclosure discipline is exemplary: legacy data are flagged, the second session is reported, transfer failures are listed in the abstract, and per-claim conventions are stated. This makes the paper auditable, which is what made this review possible.
- The most TACO-relevant finding is the warm-L2 placement effect (Sec. S1.3, Tables S17-S18), together with the observation that compiler specialization bits change the optimal configuration (Sec. 5, Sec. S11 "Key completeness"). Both are buried in the supplement.
- The 48-page supplement is 2.4 times the article. Several headline claims, including CUDA graphs, TFLA and clean selection, are evidenced only there.

## Recommendation

**Reject** in the current framing, with moderate confidence (about 65%). The measurements are sound and honest. The case for rejection is significance and framing:

- the title's bound is elementary and aimed at a strawman (DA2);
- the title's cost model works only on the authors' own slower kernel and loses to defaults or trivial tables everywhere else (DA1);
- the one production-relevant effect shrinks to 1 to 3% of scan time when weighted by latency (DA3).

A version reframed as a characterization and measurement-methodology study would plausibly reach **Major Revision**. That version would put device-time and latency-weighted headlines first (DA3-DA5), report honest model selection (DA6) and the cost trade-off against profiling (DA7), and add multi-GPU CUDA-graph evidence (DA9).

## Scores (0-100)

| Criterion | Score | Rationale |
|---|---|---|
| Novelty | 30 | Elementary bound; a negative result few would contest; a narrow single-kernel cost model in a well-trodden family of analytical models. |
| Methodology | 62 | Measurement rigor is high. Model-form selection on test data, convention switching and an unguarded headline session cost points. |
| Experiments | 60 | Broad (3 GPUs, 4 kernel families, graphs, TFLA, TP widths), but the decisive regimes (graphs, the data-center GPU) are thin, and much evidence sits in the supplement. |
| Clarity | 35 | The abstract and contributions are number-dense, with mixed conventions and statistics. The key caveats are accurate but hard to find. |
| Significance | 25 | Practical message: production defaults are fine in eager mode, and under graphs (one GPU) a shape table recovers 1 to 3% of scan time. The constructive tool applies to one in-house kernel. |
