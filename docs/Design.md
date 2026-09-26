# Experiment Design — Cycle tpds-r1 (IEEE TPDS)

Review file: `docs/revision_suggestions.tex` (items C1–C8).
Narrative: negative-result characterization. The question is whether per-call
activation statistics carry scheduling information for chunked
linear-recurrence kernels beyond shape, sequence length and hardware.

Common rules for every timing experiment:
- Recurrence-preserving kernels only. The chunk must be a real tiling
  parameter of one kernel call; never split a sequence into independent calls
  (that resets the recurrent state and only measures launch count).
- CUDA-event timing, warmup >= 10, timed repeats >= 50, report mean, std,
  95% CI; L2 flush between repeats is optional but recorded.
- Correctness check of every kernel/chunk against a reference scan
  (max abs error recorded in the result file).
- Results are JSON/CSV under `src/experiments/results/tpds/`, one file per
  (experiment, host) with full environment metadata (GPU, driver, CUDA,
  torch, triton, library versions).
- Scripts are resumable: one row per completed configuration, flushed and
  fsynced; completed keys are skipped on restart.
- Hosts: local RTX 3070 8 GB (smoke, small shapes), RTX PRO 6000 96 GB
  Blackwell (`pro6000`, full runs).

## Kernels

| Family | Kernel | Chunk parameter |
|---|---|---|
| Mamba-1 | Triton chunked selective scan written for this study: intra-chunk associative scan, state carried across chunks inside one launch | `BLOCK_T` in {32..1024} |
| Mamba-1 | stock `mamba_ssm` CUDA selective scan (if it builds) | none (fixed internal chunk); reference point only |
| Mamba-2 | `mamba_ssm` SSD `mamba_chunk_scan_combined` (Triton) | `chunk_size` in {32..512} |
| Gated DeltaNet | `fla` `chunk_gated_delta_rule` (Triton) | chunk size if exposed, else fixed 64 (value-invariance only) |

## E1 — Value invariance of kernel latency (item C1, B4, C6)
Purpose: show that, for fixed shape/dtype/chunk, latency does not depend on
activation values, so entropy cannot carry information beyond shape.
Setup: for each kernel family, shapes (batch 1, L in {512, 2048, 8192}, model
widths of the 370M-class checkpoints), every candidate chunk, inputs drawn
from uniform, Gaussian, Laplace, sparse (50%, 10%, 2% nonzero), constant, and
real captured activations (from E2). Report latency per (chunk, input) and
the between-input coefficient of variation versus the between-chunk spread.
Runtime: ~1 h on PRO 6000. Peak VRAM < 10 GB.

## E1b — E1 control: value effects vs device state (items C1, B4)
Why: E1 on PRO 6000 (3 rounds) found that the chunk decision is input-independent
within noise (tie-aware agreement 94%; regret of the input-agnostic chunk <= 1.4%
for Mamba-1/GDN, <= 5.8% for Mamba-2), but 12/84 fixed-chunk tests show a
significant input effect on latency (BH-adjusted), almost all Mamba-1, with no
monotone relation to entropy, up to 43% at non-optimal chunks at L=8192, and
bimodal per-round latencies (~1.0 vs ~1.4 ms) even for the constant input. With
three rounds these cannot be separated from clock/power state.
Setup: E1 with Mamba-1 (all chunks) and Mamba-2 (control), L in {512, 2048, 8192},
act/all modes, 8 distributions + 2 extra random draws of each synthetic
distribution (null spread for inputs that differ only in values), 10 rounds with
shuffled chunk and input order, SM clock / power / temperature / throttle reasons
recorded before and after every timing. GPU must be exclusive (lane waits;
foreign GPU processes are logged every 30 s).
Analysis (e1_analysis): tie-aware decision agreement; two-way input x round
decomposition with BH correction; between-distribution vs between-redraw range;
latency-vs-SM-clock correlation and clock-adjusted input test.
Decision rule: if between-distribution effects are within the redraw null and/or
vanish after clock adjustment, report latency value invariance with that
evidence; otherwise report the residual effect honestly as a limitation of
assumption 2 of the proposition (decision-level result unaffected).
Runtime: ~0.5 h on PRO 6000 (13k timings). Peak VRAM < 2 GB.

## E1c — E1b rerun with an in-run exclusive-GPU guard (items C1, B4)
Why: E1b (2026-09-26 11:38-12:05 host time) started when the GPU was free, but
another project's jobs claimed it 30 s later (three processes, about 50 GB) and
ran for the whole measurement. Its tie-aware agreement of 99.75% is an artifact
of the inflated confidence intervals and the between-chunk spreads changed
(Mamba-1 L=512: 4.37x vs 2.50x in E1), so E1b cannot separate value effects
from device state and is not used.
Setup: identical to E1b, plus a guard inside the timing loop: before every
timed cell the run waits while any other compute process is on the GPU
(polling every 30 s, extra untimed warmup after a pause), and a cell during
which another process appeared (checked before and after the timing) is
discarded and re-timed. Every row records the guard result. A deadline
(`--guard-until`) stops a paused run cleanly so completed rows can be pushed;
the run is resumable.
Analysis and decision rule: as E1b.
Runtime: ~0.5 h of exclusive GPU time on PRO 6000. Peak VRAM ~4 GB.

## E7 — Triton Mamba-1 scan vs the reference CUDA selective scan (items C1, C4; final-review limitation)
Why: the kernel-level Mamba-1 results use a Triton scan written for this study
(chunk = time tile). A reviewer can ask whether the chunk sensitivity (up to
8.5x) and the per-call oracle variation are artefacts of a slow kernel. The
reference CUDA selective scan of mamba_ssm has no sm_120 build on the RTX PRO
6000, but the stock wheel on the RTX 3090 has sm_86 kernels.
Setup: RTX 3090; the twelve E4 keys (L in {128, 512, 2048, 8192} x batch
{1, 4, 16}), captured Mamba-1 activations as in E4; time the reference
`selective_scan_fn` (with z and D, softplus delta) and the Triton kernel at
every chunk (default launch) and at the E4-tuned configuration of the key;
CUDA events, 10 warmup, 50 repeats; exclusive-GPU guard per key; output check
(max |y_ref - y_triton|).
Metrics: best Triton static chunk / reference and tuned / reference per key
(geometric mean, min, max).
Decision rule: report the ratio in the paper (Limitations or Sec. V-B). If the
Triton kernel at its best chunk is within ~1.5x of the reference, state that
the chunk effects are measured on a competitive kernel; if it is much slower,
say so and scope the kernel-level magnitudes accordingly.
Runtime: ~10-15 min exclusive GPU. Peak VRAM ~3 GB.

## E2 — Per-call oracle variation on real traces (items C2, C6, C7)
Purpose: measure whether the optimal chunk changes from call to call on real
workloads, and how much a single static chunk loses (regret).
Setup: capture the scan inputs of every layer during prefill for
Mamba-1 370M and Mamba-2 370M checkpoints over the mixed-regime corpus
(8 regimes) plus long-context LongBench prompts; sequence lengths from ~20
to 8192 tokens. Per captured call record entropy (K=256), variance, kurtosis,
sparsity, L, layer, width. Time each call at every chunk; batch size
{1, 4, 16} by stacking prompts of equal length bucket.
Outputs: per-call table (features, latency per chunk, oracle chunk), regret
of each static chunk, entropy distribution per regime.
Runtime: ~3–4 h on PRO 6000. Peak VRAM ~30 GB (batch 16, L 8192).

## E3 — Feature-to-oracle prediction (item C3)
Purpose: answer reviewer Q2 — does entropy predict the oracle beyond L/shape?
Setup: CPU analysis of E2. Predictors: global static; L-only (decision tree
depth <= 3); shape (L, batch, width, layer); entropy-only; activation moments;
all features. Grouped (by prompt) 5-fold cross-validation. Metrics: oracle
accuracy, mean and p95 regret (% latency above per-call oracle), and mutual
information between entropy and oracle chunk conditional on L bucket.
Runtime: minutes, local.

## E4 — Auto-tuning baseline (item C4)
Purpose: compare against the strongest offline tuning practice under identical
conditions; report tuning cost.
Setup: Triton-autotune-style exhaustive search over chunk x num_warps
(x num_stages) keyed on (L bucket, batch, width) for the Mamba-1 Triton kernel
and chunk_size for Mamba-2 SSD. Record wall-clock tuning cost per key and
total, and the tuned latency; compare with best single static chunk, the
per-call oracle (E2) and the entropy rule (Eq. 2, C_max=512) plus its
measured runtime overhead.
Runtime: ~1 h on PRO 6000.

## E5 — Break-even amortization (item C5)
Purpose: place "replace offline profiling" on an operational scale.
Model: policy cost over N calls = one-time tuning cost + N x (mean latency +
per-call decision overhead). Inputs: E4 tuning cost, E2/E3 regret, measured
entropy decision overhead (existing routed ablation and a fresh isolated
measurement on PRO 6000). Output: break-even N for entropy-online vs static
vs autotuned, and the minimum per-call gain that online adaptation would
need to break even.
Runtime: minutes, local, plus the overhead microbenchmark in E4.

## Figures (item C7)
- latency vs chunk per (family, L), with CI bands;
- value-invariance plot (latency by input distribution at fixed chunk);
- entropy distribution per regime and entropy vs oracle chunk scatter;
- regret CDF per policy; break-even curves.
