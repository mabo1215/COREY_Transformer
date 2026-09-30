# Experiment Design — Cycle tpds-r3 (2026-09-30): GPU experiments selected by the author

The author selected two of the five GPU experiments of the tpds-r3 review
(`docs/revision_suggestions.tex`): M1c (tensor-parallel widths) and M4c (a
data-centre GPU). M2b, M6b and M6c are not run; the manuscript states their
absence in Limitations. Lane: `src/experiments/remote/tpds_lane_r3.sh`
(`TPDS_R3_ONLY` selects E12 or E13).

## E14 — Re-timed larger checkpoints (M6c; author 2026-10-01)

Purpose: the E11 sweeps of the 1.4B and 2.8B checkpoints ran at batch 1 only and, for the
2.8B model, while the host CPU was heavily loaded (median within-block CV 129%), so the
paper could not decide the per-shape headroom there. Re-time both checkpoints at batch 1,
4 and 16 with a GPU-side pre-spin in front of every timing block (the host queues all
repeats while the GPU is busy, so host load cannot insert gaps between a start event and
its kernel), and log the host load average every 30 s.

Setup: download the checkpoints again (aria2, hf-mirror; `tpds_r3_e14_download.sh`),
capture the traces of 8 layers each (stride 6 of 48 for 1.4B, 8 of 64 for 2.8B; same 192
prompts), then E2 (`e2_oracle_sweep.py --pre-spin-ms 20`) at batch 1, 4, 16 (2360 calls per
model). Metrics: E3 predictors, cost-model transfer (3090 parameters, no refit), exact-shape
floor, held-out and split-half headroom, within-block CV compared with E11.

Host: vgpu3090. ~1 h download, ~2-3 h GPU. Peak VRAM ~12 GB (2.8B capture). Disk ~110 GB of
traces (475 GB free). Result files: `e2_mamba1_vgpu3090_{1p4b,2p8b}_r3.{csv,json}`.

## E15 — Both timing conventions on the RTX 3090 (author 2026-10-01)

Finding from the E14 preview: back-to-back CUDA-event timing (the protocol of E1-E11) is floored
by the host launch cost for calls shorter than about 100 us (median old/new ratio 5.0 below 20 us,
2.9 at 20-40 us, 1.5 at 60-100 us, 1.00 above 100 us on the 1.4B batch-1 calls; within-block CV
5.9% -> 0.5%); the per-call oracle differs for 30% of the batch-1 calls. The launch-inclusive
latency is the relevant cost in eager execution, device time the relevant cost under CUDA graphs.
E15 measures both so that the paper can state whether its conclusions hold under either: the 1.4B
and 2.8B calls launch-inclusive (tags `_1p4b_r3plain`, `_2p8b_r3plain`), and the 370M cost-model
calibration (E4 Mamba-1, E1 Mamba-1) plus the 2360 captured 370M calls in device time
(tag `vgpu3090_r3spin`). Lane `tpds_lane_r3_e15.sh`, starts after E14. ~3 h on vgpu3090.

## E12 — Chunk sweep at tensor-parallel widths (M1c)

Purpose: tensor parallelism over t devices splits the d_inner channels, so each
device runs the scan at width d/t. Test whether the frozen cost model, calibrated
once at d = 2048, predicts the chunk at unseen widths, and how much a chunk table
tuned at the unsharded width loses after sharding.

Setup: widths {5120, 4096, 2560, 2048, 1280, 1024, 640, 512} (full d_inner of the
2.8B, 1.4B and 370M checkpoints and their shards t = 2, 4, 8); L in {128, 256, 512,
1024, 2048, 4096, 8192}; batch {1, 4, 16}: 168 shapes. Every chunk of the Triton
Mamba-1 scan (32..1024, default launch of each chunk) and, when the runtime-chunk
module is available, of the reference CUDA scan (128..2048). Synthetic Gaussian
inputs (the kernels are data-oblivious; E1/E9 bound value effects at the selected
chunk to a few percent), one draw per shape. Three rounds with shapes and chunks
shuffled per round (block noise shows up between rounds); 10 warmup, 50 repeats;
exclusive-GPU guard; NVML state per shape.

Metrics (`e12_analysis.py`, CPU): per shape the oracle over round means and its tie
set (overlapping 95% CIs over rounds); per model and TP degree the mean / p95 regret
and tied share of (a) the frozen cost model (Triton: Eq. (5) with the RTX 3090
parameters; reference: the refscan instance), (b) the unsharded-width table,
(c) the d = 2048 table, (d) the nearest calibration key, (e) the best static
chunk; share of shapes whose oracle moves beyond the tie set after sharding;
agreement of the per-round argmin.

Host: vgpu3090 (RTX 3090 48 GB). Runtime ~1 h. Peak VRAM ~6 GB (b16, d5120, L8192,
fp16). Result files: `e12_tp_widths_vgpu3090.{jsonl,json}`, `e12_analysis_vgpu3090.json`.

## E13 — Frozen cost model on a data-centre GPU (M4c)

Purpose: both GPUs so far are consumer-lineage dies. Test the frozen model form on
a data-centre GPU (A100/A800, sm_80, or H800/H100/H20, sm_90), refitting only the
eight per-GPU parameters on the same calibration protocol, and evaluate chunk
selection on the 2360 captured Mamba-1 calls timed on that GPU.

Setup (same protocol and software versions as the other GPUs: torch 2.8.0 cu128,
Triton 3.4.0): kernel_stats for the native architecture; capture the 370M Mamba-1
traces (8 layers, 192 prompts); E4 Mamba-1 launch sweep at the twelve keys; E1
Mamba-1 value invariance at L in {512, 2048, 8192}, batch 1; E2 per-call sweep of
the 2360 captured calls; then E12 on the same GPU. Offline: add the GPU to
`cost_model.HARDWARE` (SM count from the run metadata), fit with
`cost_model.py --hosts <dc>` (form frozen), report V1/V2/LOKO/top-k as for the other
GPUs, cross-GPU transfer from the RTX PRO 6000 and RTX 3090 parameters, and the
Table III rows (cost model, nearest calibration key, shape tree) on that GPU.

Host: to be rented by the author (AutoDL). Runtime ~3 h GPU (prep in no-GPU mode:
clone, venv, checkpoint download). Peak VRAM ~20 GB. Disk: ~35 GB of traces.
Result files: `e2_mamba1_<dc>.csv`, `e4_<dc>.jsonl`, `e1_<dc>.jsonl`,
`kernel_stats_sm<arch>.json`, `cost_model_<dc>.json`, `e12_*_<dc>.*`.

# Experiment Design — Cycle tpds-r2 (proposed 2026-09-29): from negative result to predictive model

Status: PROPOSAL, written after the post-tpds-r1 assessment. Not yet
registered in the ledger (`docs/state/`). Register it as cycle `tpds-r2`
through the paper-revise ledger before the first result is pushed, so that
several terminals can work on it without colliding. The tpds-r1 design is
kept unchanged below this section for reference.

## 0. Why another cycle

All AAAI-27 review items are answered in the tpds-r1 manuscript (27/27
verified). What remains is a contribution problem, not a compliance problem.
A TPDS reviewer is likely to raise, in this order:

1. Proposition 1 (value invariance) is close to a tautology. The paper spends
   its length refuting a hypothesis (COREY) that the authors themselves
   proposed. A negative result needs a transferable positive deliverable.
2. The large chunk effect (8.5x) exists only on the Triton Mamba-1 kernel
   written for this study. The production kernels are flat (Mamba-2 SSD up to
   L=2048), monotone (Gated DeltaNet always at its largest chunk), or do not
   expose the chunk at all (reference CUDA scan). So on production kernels the
   scheduling question may be moot at these sizes.
3. The 43% fixed-chunk input effects on the RTX PRO 6000 (12/84 configs) are
   attributed to buffer placement, but the paper says "which we did not test
   directly". This is the soft spot of the central claim.
4. The routed end-to-end study is low-resolution: one prompt length, one
   GPU, separate sessions with 887-952 ms passive drift, no prefill-only
   timing, per-run records of two sessions not retained.
5. Coverage: 370M checkpoints only, prefill only, batch <= 16, GDN synthetic
   only, KDA absent.
6. Venue fit: single-GPU kernel characterization with no parallel or
   distributed dimension.

This cycle turns the paper into: "a shape-and-hardware cost model predicts
the optimal chunk of chunked linear-recurrence kernels; a no-gain theorem
shows that no activation statistic can improve on it by more than a
headroom that is below the minimum decision cost; a key-churn condition
states when any runtime chunk adaptation pays". The entropy result becomes
the motivating counter-example instead of the whole paper.

Work packages T1-T4 need no GPU and can start immediately from the result
files already in `src/experiments/results/tpds/`. E8-E11 need the RTX 3090
(`vgpu3090`) and, for E8/E10, a prerequisite P1 on the H800 in no-GPU mode.

## 1. Offline work packages (no GPU)

### T1 — Analytical chunk cost model (main deliverable)

Purpose: replace the tautological Proposition 1 with a predictive model
T(s, C) that is fitted per GPU from a handful of hardware parameters and
reproduces the measured chunk curves, the argmin, and the cross-GPU shift of
the optimal chunk. If it selects chunks with ~1-2% regret without any
profiling, it is the paper's positive contribution.

Kernel facts to model (from `src/experiments/tpds/kernels.py`): the Triton
Mamba-1 scan launches grid = (ceil(d / BLOCK_D), B) programs; each program
walks the sequence in time tiles of BLOCK_T = C steps; inside a tile it loops
over the N = 16 state indices and runs `tl.associative_scan` over the C time
steps of a [BLOCK_D, C] tile; the state h[BLOCK_D, N] stays in registers
across tiles. The default launch (BLOCK_D, num_warps) depends on C
(`DEFAULT_MAMBA1_LAUNCH`); E4 additionally sweeps BLOCK_D, num_warps and
num_stages (184 configurations per key).

Model form (starting point; refine if the fit demands it):

    T(L, B, d, C; hw) = t_0 + waves(B, d, C) * [ (L/C) * (tau_f + tau_s * log2 C) + L * tau_e ]

    waves(B, d, C)    = ceil( ceil(d / BLOCK_D) * B / (n_SM * progs_per_SM(C)) )
    progs_per_SM(C)   = min( floor(regs_per_SM / (regs_per_thread(C) * 32 * num_warps)),
                             floor(max_warps_per_SM / num_warps) )
    regs_per_thread(C)= min(255, r_0 + r_1 * C * BLOCK_D / (32 * num_warps))

Terms: t_0 launch and ramp cost; tau_f per-tile fixed cost (state carry,
loop control, pipeline fill); tau_s per-level latency of the associative
scan (depth log2 C); tau_e per-element work; r_0, r_1 register demand of the
live tiles (a, bx, y, acum are [BLOCK_D, C] each). Fitted per GPU: t_0,
tau_f, tau_s, tau_e, r_0, r_1 (six parameters). Known per GPU, read from
`torch.cuda.get_device_properties` (multi_processor_count,
regs_per_multiprocessor, max_threads_per_multi_processor) and recorded in
the result file rather than hard-coded: n_SM, regs_per_SM, max_warps_per_SM.

Qualitative predictions the model must reproduce (all already measured):
- C* grows with L until the register/occupancy knee, then stops (Mamba-1
  batch 1: 256 at L=512, 512 at L=2048 and 8192 on the PRO 6000).
- At batch 4 and 16 the grid saturates the GPU, so the occupancy loss of a
  large tile costs more and C* moves down (64 on the PRO 6000, 32 on the
  RTX 3090).
- The Ampere GPU has the knee at a smaller C (128 vs 512 at batch 1).
- Mamba-2 SSD: the multi-kernel pipeline is launch-bound at small L
  (measured floor ~0.46 ms on the PRO 6000, ~0.78 ms on the RTX 3090), so
  T_SSD(L, C) ~ n_kernels * t_launch + intra-chunk matmul term (L*C*d) +
  inter-chunk term (L/C * d * N); flat up to L=2048, chunk 32 penalized at
  L=8192.
- Gated DeltaNet: monotone in {16, 32, 64} (chunk 64 is below the knee).

Data (all local, no GPU):
- `e1_pro6000.jsonl`, `e1_vgpu3090.jsonl`: latency per (kernel, L, chunk,
  input, round) at batch 1, three lengths; use only the value-independent
  part (medians over inputs).
- `e2_mamba{1,2}_{pro6000,vgpu3090}.csv`: 2360 calls per model per GPU with
  `lat_c{32..1024}_mean/std/ci` per chunk, plus L, batch, width, d_state,
  layer. This is the main fitting and validation set.
- `e4_{pro6000,vgpu3090}.jsonl`: 184 launch configurations per key at 12
  keys; the only data that vary BLOCK_D / num_warps / num_stages at fixed
  chunk, hence the best data for the occupancy term.
- Supplement Tables S4/S7 (per-chunk latency at batch 1) as a sanity check.

Fitting and validation protocol:
1. Fit the six parameters per GPU by least squares on log latency over the
   E4 sweep plus the E1 medians (batch 1). Report the parameter values and
   their physical plausibility (tau_e vs. memory bandwidth, t_0 vs. measured
   launch latency ~0.05-0.2 ms).
2. V1 curve fit: predicted vs measured latency for every (kernel, L, chunk)
   of Tables S4/S7 and for the 12 E4 keys; report R^2 on log latency and the
   maximum relative error per chunk. Target: < 15% relative error at the
   chunks near the optimum, correct ordering of chunks.
3. V2 argmin prediction with held-out keys: leave-one-key-out over the 12
   E4 keys (fit on 11, predict C* of the 12th); and a "tiny calibration"
   protocol that fits on the three E1 lengths only and predicts all 2360 E2
   calls. Add a row "Analytical model (no profiling)" to the predictor
   table (Table III) with mean / p95 regret and the tied share, computed
   exactly as in `e3_predictors.py`. Target: <= 2% mean regret on Mamba-1 on
   both GPUs (the shape tree reaches 0.7% / 1.9% with profiling).
4. V3 cross-GPU: with per-GPU parameters, does the model reproduce the C*
   shift 512 -> 128 (batch 1) and 64 -> 32 (batch 4/16)? Also test the
   naive transfer: PRO 6000 parameters with RTX 3090 SM counts, to show
   which parameter carries the difference.
5. V4 production kernels: fit the SSD launch-floor model and the GDN model
   on the E1/E2 data and show that they predict the flatness and the
   monotone behavior. This is what answers reviewer point 2 without new
   hardware: the model says at which L, B, d the chunk starts to matter.
6. Optional refinement (needs a GPU for compilation only, ~10 min on any
   host, can piggyback on E8): read `kernel.n_regs` and `kernel.n_spills`
   of the compiled Triton kernel for every (C, BLOCK_D, num_warps) and use
   the measured register count instead of the linear r_0 + r_1 model.

Decision rule: if V2 reaches <= 2% regret on both GPUs, promote the model to
the main contribution (new Section IV "A shape-and-hardware cost model",
Proposition 1 becomes its corollary, abstract and title change). If 2-5%,
present it as a zero-profiling default that a few profiles refine. If worse,
keep it as the explanatory model behind Fig. 3 and say so.

Deliverables: `src/experiments/tpds/cost_model.py` (new; fit, predict,
validation tables, all figures from result files), result files
`cost_model_{pro6000,vgpu3090}.json`, figure "measured vs predicted latency
per chunk" (two GPUs, three kernels), new Table III row, text for Section IV.
Runtime: analysis and writing, 2-3 days; no GPU.

### T2 — No-gain theorem for activation-dependent policies

Purpose: state once, formally, what Sections VI-C and VI-G show variant by
variant: no policy that reads the activations can beat the shape-keyed
policy by more than the oracle headroom, and the headroom is below the
minimum decision cost.

Statement. Let s be the call shape, Z the input, T(s, C; Z) the latency at
chunk C. The shape-keyed policy is pi_s(s) = argmin_C E[T(s, C; Z) | s] and
its headroom is h(s) = E[T(s, pi_s(s); Z) - min_C T(s, C; Z) | s] >= 0. Let
pi be any policy pi(s, phi(Z)) with per-call decision cost o_pi >= o_min.
Then

    E[T(s, pi(s, phi(Z)); Z)] + o_pi - E[T(s, pi_s(s); Z)] >= o_min - h(s).

Proof: E[T(s, pi; Z)] >= E[min_C T(s, C; Z)] = E[T(s, pi_s(s); Z)] - h(s);
add o_pi >= o_min. Under Proposition 1 (data-oblivious kernel, value-
independent instruction timing) h(s) = 0 exactly. The bound holds for every
statistic phi, so it covers all COREY variants and any future one.

Instantiation (numbers already in the paper, from E5): h = 0.0025 ms per
call on the PRO 6000 and 0.0083 ms on the RTX 3090 (oracle over the
shape-keyed table); o_min = one host synchronization, 0.020 / 0.021 ms, for
any host-side decision. For a device-side decision o_min is the launch of
one extra reduction kernel over the scan input; measure it in E9 (a trivial
kernel over the same tensor, CUDA-event timed) so that the theorem also
excludes the "compute the statistic on the device" escape.

Where: Section IV as Theorem 1 right after Proposition 1; Section VI-G
instantiates it; the contributions list and abstract cite it. Also express
the per-call loss as a fraction of the scan latency at the median call.
Runtime: half a day of writing; no GPU.

### T3 — Key-churn break-even for any online policy

Purpose: generalize Eq. (3) from "entropy vs profiling" to "any online policy
vs a shape-keyed tuning cache", which is the operational question behind
reviewer Q1/Q5.

Model. Shape keys arrive in a stream. Tuning a new key costs P_k once
(profiling six chunks: 1.17 s / 12 keys ~ 0.1 s per key on the PRO 6000;
full auto-tuning ~41 s per key excluding compilation, from E4). Each key
is served N_k calls in the workload's lifetime. The cache policy has per-call
regret r_c (shape-tree row of Table III); an online policy has overhead o
and regret r_on >= r_c - h (T2). The online policy has the lower total cost
iff

    N_k * (o + r_on - r_c) < P_k,   hence necessarily   N_k < P_k / (o - h).

Compute N_k* for o in {host sync only 0.020 ms (ideal), sampled histogram
0.19 ms, full histogram 4.43 ms} and P_k in {profiling 0.1 s, auto-tuning
41 s}, on both GPUs. Then estimate N_k in the paper's own workloads: number
of distinct (L bucket, batch) keys over the 2360 captured calls, and per
regime. Expected result: online scheduling of any kind pays only when a
distinct shape key is served fewer than a few hundred to a few thousand
calls, i.e. tens of requests of 48 layers, which is extreme shape churn.

Deliverables: extension of `e5_breakeven.py` (or a new
`e5b_key_churn.py`), one figure (N_k* versus o for the two P_k values, both
GPUs), one paragraph in Section VI-G, and a formula-based answer to Q1/Q5 in
the Discussion. Runtime: one day; no GPU.

### T4 — Equivalence testing for the invariance claim

Purpose: "value invariance" is an equivalence claim; a non-rejected null is
not evidence for it, and the paper's own NHST already rejects 12/84
configurations. Use two one-sided tests (TOST) with a pre-stated margin.

Setup: on `e1_*.jsonl` (and E9 when available), for every (kernel, L, mode)
at the input-agnostic best chunk C*, test whether each input's mean latency
lies within +/- delta of the pooled mean, delta = 2% (the tie margin used in
the regret tables) and also 5%; report the share of configurations that pass
at C* and at every chunk. Report alongside the existing BH-corrected ANOVA.
Deliverable: a paragraph in Section VI-A, a supplement table, code in
`e1_analysis.py`. Runtime: half a day; no GPU.

## 2. Prerequisite P1 — recover the runtime-chunk CUDA scan patch (H800, no-GPU mode)

Why: E8 and E10 need the reference Mamba-1 CUDA selective scan with the
chunk as a runtime argument. That patch exists only as a built extension on
the H800 (`selective_scan_cuda.fwd_with_chunk_size`, used by
`src/corey_selective_scan_dispatch.py`); its source is not in this
repository, and the stock wheel on the RTX 3090 has only `fwd`
(`docs/progress.md`, 3090 smoke note).

Steps (H800 in no-GPU mode is enough):
1. Start the H800 in no-GPU mode; find the repo root and the patched build:
   `python -c "import selective_scan_cuda, inspect; print(selective_scan_cuda.__file__)"`,
   then the source tree it was built from (look for a modified
   `csrc/selective_scan/selective_scan_fwd_kernel.cuh` and `selective_scan.cpp`
   under the mamba_ssm checkout or a `mamba` source directory).
2. Diff it against the stock source (a stock copy is at
   `Quamba/3rdparty/mamba/csrc/selective_scan/`; match the mamba_ssm
   version first). Save the diff as `src/kernels/mamba1_runtime_chunk.patch`
   (new directory) with a `README.md` stating the base version, build
   command, and the chunks it supports ({128, 256, 512, 1024, 2048}).
3. Record which chunks the H800 build validated (only 1024 was checked
   against the stock output in the routed study) so E8 can validate all.
4. Commit; shut the H800 down (one `shutdown -h now`).
If the source cannot be found: re-implement. In the stock kernel the chunk is
the compile-time `kChunkSize = kNThreads * kNItems`; the patch must template
the forward over the chunk set and dispatch on the runtime argument. Budget
one day; validate against stock `fwd` at every chunk on L not a multiple of
any chunk.

Runtime: 1-2 h if the source is on the H800; no GPU.

## 3. GPU experiments (RTX 3090 `vgpu3090`; optional PRO 6000 hour)

Common: exclusive-GPU guard as in E1c/E7; CUDA events; resumable jsonl with
fsync; env metadata; check `df -h` and `df -i` before starting; verify
`src/outputs/tpds_traces` still exists on the host, otherwise re-capture
with `capture_traces.py` (~20 min). Never edit tracked files on the host:
local edit, commit, push, `source /etc/network_turbo && git pull`.

### E8 — Chunk sweep of the production kernel (reviewer point 2; highest value)

Purpose: decide whether the chunk sensitivity and the per-call oracle
variation are properties of chunked scans or artefacts of the Triton kernel.

Setup: RTX 3090 (sm_86; E7 showed the stock wheel works there). Build the
patched extension from P1. Inputs: the same 2360 captured Mamba-1 calls as
E2 (batch 1, 4, 16) and the 12 E4 keys. Time the patched reference scan at
every supported chunk {128, 256, 512, 1024, 2048} and the stock `fwd` (its
fixed internal chunk), with the E2 protocol (10 warmup, 50 repeats, CI). Output
check of every chunk against stock `fwd` on every call (max |dy|) — this also
closes the "validated only at 1024" limitation. Record entropy and moments of
each call as E2 does (reuse the E2 feature columns by call key).

Analysis: run the E2/E3 pipeline on the new per-call table: chunk spread per
call, per-call oracle, static regret, tied share, predictors (shape vs
entropy vs both), and the T1 cost model refitted for this kernel (same
form: sequential chunk loop, kNThreads x kNItems tiles).

Decision rule: (a) spread < ~1.3x at every key: state in the paper that on
the production kernel the chunk matters little at these sizes, the Triton
kernel is the stress case, and the cost model says where it would start to
matter; (b) spread large: the Triton results generalize and the paper says
so; (c) either way the predictor table gets a "reference kernel" block.
Runtime: ~2-3 h exclusive (2360 calls x 6 variants x 60 timings, plus
batches). Peak VRAM ~10 GB. Result files: `e8_refscan_vgpu3090.{jsonl,csv}`.

### E9 — In-place overwrite control for the fixed-chunk input effects (reviewer point 3)

Purpose: separate "value" from "placement" directly instead of inferring it
from redraws.

Setup: Triton Mamba-1 kernel; L in {2048, 8192}, batch 1; all six chunks;
`act` and `all` modes. Three designs, same distributions as E1 (eight, with
three draws each):
- A, same buffers: allocate one set of input/output buffers once; for each
  (distribution, draw) copy the values into these buffers in place; time.
- B, fresh buffers: allocate a new buffer set per (distribution, draw), as
  E1 did; time.
- C, same values, different placement: one Gaussian draw copied into ten
  freshly allocated buffer sets (record the device pointers and their
  alignment mod 256 B, 4 KB, 2 MB); time.
Five rounds, shuffled order, exclusive guard, SM clock / power / throttle
reasons logged as in E1b/E1c. Also time the device-side "statistic" cost
needed by T2: one trivial reduction kernel over the scan input (sum), CUDA
events, 500 repeats, per (L, batch).
If E8's build is available, repeat design A and C on the patched reference
scan at its best chunk.

Analysis: two-way ANOVA (input x round) with BH correction per design; TOST
(T4) per design; correlation of latency with pointer alignment in design C;
clock-adjusted test.
Decision rule: A shows no input effect and C shows a placement effect ->
Section VI-A states that the residual effects are placement, tested
directly; A shows effects -> report them as a real limitation of assumption
2 and quantify them at C*. The 43% effects were seen on the PRO 6000
(Blackwell) and are <= 4.3% on the RTX 3090, so the RTX 3090 run establishes
the method; rent the PRO 6000 for one hour to run design A and C at L=8192
if the budget allows — that is where the claim is contested.
Runtime: 30-45 min on the RTX 3090; ~1 h on the PRO 6000 (optional). Peak
VRAM < 4 GB. Result files: `e9_placement_{host}.jsonl`.

### E10 — Single-session interleaved routed end-to-end rerun (reviewer point 4)

Prerequisite: E8's patched build on the RTX 3090 (or the H800 with GPU).

Setup: Mamba-1 370M; one process, one session; configurations interleaved
round-robin within each round: passive, static {128, 256, 512, 1024, 2048},
midpoint, random, full / sampled / token-level histogram, moment / variance
/ kurtosis proxies, guarded, learned table (15 configurations); 30 rounds; 3
warmup rounds; prompt lengths {256, 976, 4096} tokens; 32 greedy tokens.
Record per run: prefill time (CUDA events around the prefill forward), decode
time, total, and the routed chunk counts. Keep every per-run record.

Analysis: paired differences to the best static chunk within each round
(no session drift by construction), mean and 95% CI; prefill-only chunk
effect compared with the kernel-level prediction of T1/E8 (48 scans of ~1000
tokens should be a few ms); decision cost of every statistic-bearing variant
in ms and in percent of prefill, not only of end-to-end.
Decision rule: replaces Table IV if the resolution is better than the H800
table (expected: CI half-widths of ~1% or below at n=30 paired); otherwise
added as a supplement table with the H800 table kept.
Runtime: 15 configs x 33 rounds x 3 lengths x ~1-2 s ~ 1-1.5 h. Peak VRAM
~4 GB. Result file: `e10_routed_vgpu3090.jsonl`.

### E11 — Larger checkpoints (optional; reviewer point 5 and a test of T1's width dependence)

Setup: Mamba-1 1.4B (`state-spaces/mamba-1.4b-hf`, d_inner 4096) and 2.8B
(`state-spaces/mamba-2.8b-hf`, d_inner 5120); capture batch-1 traces of 8
layers over the 192 prompts (`capture_traces.py` with a new model id);
E2-style chunk sweep with the Triton kernel; predict C* with the T1 model
using the new d without refitting the hardware parameters; report the regret
of the model-selected chunk and of the 370M-tuned chunk.
Runtime: ~1.5-2 h on the RTX 3090 (fp16 weights ~5.6 GB for 2.8B). Result
files: `e11_{model}_vgpu3090.{jsonl,csv}`.

## 4. Manuscript changes implied

- Section IV: cost model (T1) + Theorem 1 (T2); Proposition 1 becomes a
  corollary. Title and abstract to be revised by the author once V2 is in
  (candidate direction: "shape, not values, sets the chunk").
- Section VI-A: TOST (T4) and E9 results; remove "which we did not test
  directly".
- Sections VI-B/C: reference-kernel block from E8; "Analytical model" row
  in Table III; measured-vs-predicted figure.
- Section VI-F: E10 replaces or supplements Table IV; prefill-only numbers.
- Section VI-G and Discussion: key-churn condition (T3) as the answer to
  Q1/Q5; T2 instantiated.
- Limitations: shrink accordingly; add E11 if run.
- Related work: position the cost model against TFLA (arithmetic-intensity
  model) and Triton/TVM/Ansor (search-based) explicitly.

## 5. Execution order and terminals

- Terminal A (local, no GPU, start now): T1 -> T2 -> T3 -> T4. T1's V4 and
  the reference-kernel refit wait for E8 but everything else does not.
- Terminal B (H800 no-GPU mode, start now): P1. Then on `vgpu3090`:
  E8 -> E9 -> E10 -> E11 (optional). Total GPU budget ~6-8 h on the RTX 3090
  plus an optional 1 h on the PRO 6000 for E9.
- Integration (after E8 and T1 V2): paper sections in 4., then a fresh
  Stage 9' review against this design and the TPDS standard.
- Dependencies: E8 and E10 need P1; T1 V4 (reference kernel) needs E8; T2's
  device-side o_min needs E9; everything else is independent.

---

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
