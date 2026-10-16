# taco-r2 offline statistics (M3a, M5a, M5b, M5d, M5h, M5i offline part, M6a)

All numbers below were computed on the CPU from stored result files; nothing was timed. Every script was run
twice and the JSON outputs were identical apart from run-time fields. Scripts live in `src/experiments/tpds/`,
results in `src/experiments/results/tpds/`. Shared helpers: `taco_r2_stats_common.py` (prompt-cluster bootstrap
with the replicate matrix of the existing analyses, BCa from the cluster jackknife, graph-table loading).

Conventions (as in the manuscript): per-call regret T(c)/T(c*) - 1 against the per-call oracle of the view;
LW = latency-weighted share = 100 x sum(excess) / sum(oracle latency) ("% of scan time", the LW column of
Table 7); intervals are 95% prompt-cluster bootstrap intervals (42 clusters, 2000 replicates) unless stated;
"BCa" = bias-corrected and accelerated interval from the same replicates.

Naming note: the other agent of this cycle overwrote `taco_r2_common.py` with its own helpers. The helpers of
these items therefore live in `taco_r2_stats_common.py`, and all scripts of these items are named
`taco_r2_stats_*`.

---

## M3a: recoverable gains under CUDA-graph replay (Table 7, both GPUs)

**Script:** `taco_r2_stats_m3a_recoverable.py`. **Result:** `taco_r2_stats_m3a.json`.

**What was computed.** For the eight three-round CUDA-graph tables of Table 7 the script computes:
- the default's LW (library default; Triton: best single chunk) and the LW of the held-out exact-shape table
  (capture-time prompt folds, bucket and best-chunk fallback; `deployment_units.cv_exact_pick`);
- the recoverable share = LW(default) - LW(table), in pp of oracle scan time, also per call, in us per call and
  in ms per forward pass;
- the held-out shape tree, and the in-sample (every key profiled) table;
- all of the above over all calls, above and below the graph launch floor, on covered calls only, and per batch.

Three interval types are given:
- percentile intervals, using the same replicates as the stored Table 7 intervals (reproduced bit-exactly);
- BCa intervals;
- a refit-in-bootstrap interval (1000 replicates). In each replicate the table is re-chosen with random folds per
  resampled cluster copy. Two variants: the paper's fallback, and "profile-unseen", where a key without training
  calls is profiled on first sight.

In addition, 50 random reassignments of clusters to folds were run (repeated grouped CV).

**Reproduction.** 54/54 checks pass:
- default, table and tree LW equal the stored values;
- the default LW percentile intervals equal the stored ones;
- the default LW matches Table 7 as printed (2.8 / 1.7 / 2.2 / 5.9 / 1.0 / 3.2 / 17.9 / 32.3).

R1's point estimates are reproduced to two decimals (2.79, 1.72, 2.17, 5.39, 0.45, 2.94). R1's intervals used
another bootstrap seed, so the endpoints differ by Monte Carlo error only (e.g. 2.79 [1.56, 5.10] here against
R1's [1.51, 5.30]); ours use the paper's own replicates.

**Results (recoverable LW = default LW - held-out table LW, pp of scan time):**

| Kernel, GPU | Default LW | Table LW | Recoverable [pct CI] | BCa | Refit (paper fallback) | Refit (profile unseen) | ms per forward pass recovered | Oracle scan ms / pass |
|---|---|---|---|---|---|---|---|---|
| SSD 370M, RTX 3090 | 2.80 [1.57, 5.11] | 0.011 | **2.79 [1.56, 5.10]** | [1.51, 4.91] | [1.55, 4.80] | [1.68, 5.18] | 0.561 [0.427, 0.689] | 20.09 |
| SSD 370M, RTX PRO 6000 | 1.75 [0.96, 3.46] | 0.027 | **1.72 [0.95, 3.41]** | [0.89, 3.16] | [-4.55, 2.93] | [0.93, 3.50] | 0.122 [0.087, 0.156] | 7.07 |
| SSD 2.7B, RTX 3090 | 2.20 [1.10, 4.47] | 0.032 | **2.17 [1.07, 4.42]** | [1.04, 4.31] | [0.98, 4.15] | [1.13, 4.58] | 1.415 [0.927, 1.950] | 65.36 |
| SSD 2.7B, RTX PRO 6000 | 5.85 [3.60, 7.77] | 0.462 | **5.39 [2.70, 7.59]** | [2.74, 7.61] | [2.19, 7.02] | [3.50, 7.89] | 1.189 [0.433, 2.048] | 22.05 |
| GDN 1.4B, RTX 3090 | 1.02 [0.49, 2.09] | 0.569 | **0.45 [-0.83, 1.86]** | [-2.01, 1.37] | [-10.49, 1.40] | [0.52, 1.96] | 0.045 [-0.091, 0.131] | 10.01 |
| GDN 1.4B, RTX PRO 6000 | 3.19 [2.13, 5.46] | 0.245 | **2.94 [1.84, 5.19]** | [1.75, 4.80] | [-0.17, 5.02] | [2.15, 5.58] | 0.175 [0.144, 0.193] | 5.95 |
| Triton M1 370M, RTX 3090 | 17.94 [14.14, 23.30] | 2.296 | 15.64 [13.42, 18.87] | [12.96, 18.23] | [11.88, 22.24] | [14.06, 22.86] | 4.63 [3.17, 6.40] | 29.63 |
| Triton M1 370M, RTX PRO 6000 | 32.25 [25.12, 38.89] | 0.118 | 32.13 [25.05, 38.77] | [24.34, 38.38] | [17.77, 38.00] | [24.46, 38.90] | 4.18 [2.33, 6.68] | 13.01 |

Other variants of the recoverable share:

| Kernel, GPU | Covered calls only | In-sample table (every key profiled) | Shape tree | Batch 1 / 4 / 16 | Above / below graph floor |
|---|---|---|---|---|---|
| SSD 370M, 3090 | 2.85 [1.54, 5.49] | 2.79 [1.56, 5.10] | 2.35 [1.15, 4.67] | 6.05 / 2.88 / 1.91 | 1.99 [1.16, 3.80] / 36.4 |
| SSD 370M, PRO | 1.84 [0.99, 3.72] | 1.74 [0.96, 3.45] | 1.69 [0.92, 3.36] | 3.41 / 1.81 / 1.24 | 0.56 [0.23, 1.30] / 20.2 |
| SSD 2.7B, 3090 | 2.27 [1.15, 4.68] | 2.17 [1.07, 4.43] | 2.01 [0.95, 4.21] | 3.53 / 2.27 / 1.82 | 1.82 [0.93, 3.84] / 50.3 |
| SSD 2.7B, PRO | 6.05 [3.77, 8.02] | 5.85 [3.59, 7.76] | 3.32 [1.63, 5.94] | 6.64 / 2.43 / 5.91 | 4.91 [2.14, 7.09] / 28.6 |
| GDN 1.4B, 3090 | 1.08 [0.52, 2.36] | 1.01 [0.49, 2.09] | 0.92 [0.42, 1.95] | 3.79 / 0.75 / -0.72 | -0.38 [-1.93, 0.33] / 36.0 |
| GDN 1.4B, PRO | 3.14 [2.08, 5.43] | 3.19 [2.13, 5.46] | 2.96 [1.88, 5.27] | 8.43 / 5.31 / 0.07 | 1.85 [1.11, 3.10] / 42.4 |

Key findings:
- **Gated DeltaNet, RTX 3090, is the only cell whose interval includes zero.** The cause is one untuned key:
  - The six calls of shape (L = 2210, batch 16) all lie in one prompt fold, so the held-out table has no exact
    entry for them.
  - The bucket fallback picks chunk 32 where 64 is the oracle, at 1.34 to 1.41 ms extra per call.
  - These six calls are 97.5% of the table's whole excess; without them the table's LW is 0.014%.
  - When unseen keys are profiled (the recipe's warm-up rule), the recoverable share is 1.01 [0.49, 2.09] in
    sample and [0.52, 1.96] with refit.
- **Uncovered keys drive most of the table's loss everywhere.** The 158 to 204 calls whose exact key is absent
  from the training folds carry 32 to 99.7% of the held-out table's excess in every cell. On covered calls the
  table loses at most 0.03% of scan time. Two examples:
  - SSD 2.7B, RTX PRO 6000: 96% of the table's 0.46% comes from 8 calls at (2382, 16), 0.45 ms each.
  - The negative lower ends of the paper-fallback refit intervals come from the same mechanism.
- **"Recovers to within 0.7%"** refers to per-call means (held-out tree and table 0.06 to 0.62% per call). In
  scan time the table is within 0.01 to 0.57% (SSD and Gated DeltaNet) and 0.12 / 2.30% (Triton).
- **Cluster structure of these tables:** 42 clusters, 32 to 112 calls per cluster (median 48; Gated DeltaNet 24
  to 84), 2 to 10 prompts per cluster; the largest cluster holds 4.7% of the calls.

**Suggested text (paper-ready):**
- "Measured as the share of graph-replayed scan time that a shape-keyed table chosen on held-out prompts
  actually recovers from the library default, the gain is 2.8% (95% interval 1.6 to 5.1) and 2.2% (1.1 to 4.4)
  for SSD 370M and 2.7B on the RTX 3090, and 1.7% (1.0 to 3.4), 5.4% (2.7 to 7.6) and 2.9% (1.8 to 5.2) for SSD
  370M, SSD 2.7B and Gated DeltaNet 1.4B on the RTX PRO 6000, i.e. 0.12 to 1.4 ms per forward pass."
- "For Gated DeltaNet on the RTX 3090 the held-out table recovers only 0.45% (-0.8 to 1.9), because one shape
  (L = 2210, batch 16) occurs in a single prompt fold. Its bucket fallback selects chunk 32 instead of 64 and
  loses 1.4 ms per call, 97% of the table's loss. Once that key is profiled, as the recipe prescribes, the
  recoverable share is 1.0% (0.5 to 2.1)."

---

## M5a: noise floors next to every regret; floor and threshold sensitivity

**Script:** `taco_r2_stats_m5a_noise_floors.py`. **Result:** `taco_r2_stats_m5a.json` (`table_map` gives the floor
next to every data set of Tables 4 to 8).

**What was computed.** The noise floor is the regret of a selector that picks the best chunk on one set of
timings, scored on an independent set against that set's own oracle. It is given per call (best chunk of each
call) and per shape (best chunk of each exact (L, batch) shape, the "cross-round regret of the per-shape best
chunk"). Three sources of independent timings are used:
- **cross-round** (three-round sweeps): choose on two rounds, score on the third;
- **cross-session**, wherever two sessions of the same calls and convention exist:
  - RTX 3090 device time;
  - launch-inclusive RTX PRO 6000 and RTX 3090;
  - 1.4B and 2.8B launch-inclusive;
  - RTX 3090 graph single round against three rounds;
- **split-half** of each 50-repeat block, used for single-block tables. It captures within-block noise only and
  is therefore a lower bound.

Each floor is reported in % per call, LW % and us per call. Also computed:
- the factor 1 / 1.5 / 2 / 3 sensitivity of the launch floor, and the 0.5 to 3 x sensitivity of the 100 us
  convention threshold;
- for the 19 information data sets, whether the chunk effect (held-out best single chunk) exceeds twice the
  per-shape floor. 19/19 recomputed effects match the stored analyses.

**Floors (per-shape floor, % per call, with method; next to the paper-view default and held-out table):**

| Data set (tables) | Floor per shape (per call) | Method | Default | Held-out table |
|---|---|---|---|---|
| Triton 370M, PRO 6000, device (T4, T5, T8) | 0.054% (0.070%), 0.14 us | cross-round | 48.6% | 0.14% |
| Triton 370M, RTX 3090, device (T5, T8) | 0.018% (0.041%); cross-round 0.084% | cross-session | 96.4% | 0.35% |
| Triton 370M, H800, device (T5, T8) | 0.17% (0.004%), 0.30 us | split-half (lower bound) | 50.9% | 0.20% |
| Triton 1.4B / 2.8B, device (T6, T8) | 0.073% / 0.052% | split-half | 123% / 131% | 0.24% / 0.20% |
| Triton 370M launch-incl., PRO / 3090 / H800 | 2.55% / 1.34% / 0.93% | cross-session / cross-session / split-half | 23.4 / 33.9 / 20.1% | 0.62 / 1.94 / 0.98% |
| Triton 1.4B / 2.8B launch-incl. | 3.12% / 12.5% | cross-session | 52.2% / 60.1% | 2.18% / 3.97% |
| Reference scan, launch-incl. (T6, T8) | 2.48% (0.09%), 2.1 us | split-half | stock dispatch 6.58% (runtime build at stock chunk 5.35%) | 3.01% (shape tree 3.00%) |
| SSD 370M launch-incl., PRO (T4, T8) | 1.19% (0.44%), 5.4 us | split-half | 1.69% | 1.30% |
| SSD 370M launch-incl., RTX 3090 (T8) | 2.03% (0.48%), 16.5 us | split-half | 2.53% | 2.28% |
| SSD 2.7B device, all / above floor (T8) | 16.3% / 0.44% | split-half | 13.7% / 0.15% | 11.4% / 0.21% |
| GDN 1.4B device, all / above floor (T8) | 5.4% / 0.13% | split-half | 3.17% / 0.03% | 3.17% / 0.03% |
| Graph SSD 370M, 3090 / PRO (T7) | 0.30% / 0.31% | cross-session / cross-round | 19.1% / 10.1% | 0.06% / 0.10% |
| Graph SSD 2.7B, 3090 / PRO (T7) | 0.13% / 0.25% | cross-session / cross-round | 22.3% / 19.2% | 0.07% / 0.17% |
| Graph GDN 1.4B, 3090 / PRO (T7) | 0.10% / 0.06% | cross-session / cross-round | 16.2% / 23.9% | 0.13% / 0.06% |
| Graph Triton 370M, 3090 / PRO (T7) | 0.39% / 0.12% | cross-session / cross-round | 31.8% / 46.0% | 0.35% / 0.13% |

How to read the floors:
- **Single-round scale for graph and three-round floors.** These floors are scored against a single round's
  oracle, which is noisier than the round medians of the paper. On that same scale the default loses 10.4 to 46%
  per call, so the default is about 25 to 400 times its floor.
- **A tighter bound for the round-median view.** In the round-median view the held-out tables' own regrets
  bound the oracle bias from above: 0.06 to 0.35% per call in graph mode, and 0.14 / 0.35 / 0.20% in device time.
- **Launch-inclusive and SSD data sets.** Here the floors (1.2 to 3.1% per call, 12.5% for the 2.8B sessions,
  which pick the same per-call oracle for only 44% of calls) are of the order of the effects claimed:
  - the default and held-out-table regrets of the launch-inclusive SSD 370M rows (1.3 to 2.5%) lie at the floor;
  - Table 8's SSD 370M default-minus-table gap (0.09 ms per pass) is below the floor (5.4 and 16.5 us per call,
    i.e. 0.26 and 0.79 ms per pass).

**Information data sets (M5a, last sentence).** 12 of 19 resolve the chunk effect (effect > 2 x per-shape floor):
- **Resolving:** every primary-session Triton Mamba-1 data set (3 GPUs, both conventions, 1.4B and 2.8B), plus
  the reference scan.
- **Not resolving** (ratio 0.4 to 1.5):
  - Gated DeltaNet;
  - Mamba-2 launch-inclusive on both GPUs;
  - Mamba-2 2.7B and 370M device time;
  - the two early launch-inclusive 1.4B / 2.8B sessions.

The "borderline" case of Sec. 6.3, Mamba-2 in RTX 3090 device time (entropy lowers the tree by 0.47 points),
is a data set whose chunk effect (5.7%) is below its floor (16.1%). Above the launch floor, the SSD 2.7B and
Gated DeltaNet subsets still do not resolve (ratio 0.3 and 0.2).

**Launch-floor factor (published factor 2):**

| Factor | 1 | 1.5 | 2 | 3 |
|---|---|---|---|---|
| SSD 2.7B device: calls above | 1750 (74%) | 624 (26%) | 397 (17%) | 368 (16%) |
| chunk 256 is the oracle | 786 (45%) | 556 (89%) | 375 (94%) | 356 (97%) |
| chunk 256 max per-batch mean regret | 13.6% | 1.95% | 0.24% | 0.08% |
| held-out headroom above floor (us) | 38.5 | 7.1 | 0.90 | 0.26 |
| GDN 1.4B device: calls above | 1043 (59%) | 311 (18%) | 155 (9%) | 150 (8%) |
| chunk 64 is the oracle | 733 (70%) | 299 (96%) | 154 (99%) | 150 (100%) |
| GDN max per-batch mean regret / headroom (us) | 5.49% / 40.8 | 0.98% / 6.8 | 0.05% / 0.0 | 0.00% / 0.0 |

The graph tables' share above the floor changes as follows from factor 1 to 3:
- share above: 82 to 85% at factor 1, 54 to 77% at factor 2, 38 to 68% at factor 3;
- the default's LW above the floor changes little for SSD 2.7B and Triton (e.g. SSD 2.7B PRO 5.79 / 5.58 /
  5.37 / 5.20%);
- the default's LW above the floor falls for SSD 370M PRO (1.68 to 0.30%) and GDN 3090 (0.62 to 0.08%).

**100 us threshold** (launch-inclusive against device time, captured Mamba-1 calls; reproduces the appendix):

| GPU | Calls below 100 us | Median / p95 difference above 100 us (per call, chunk cell) |
|---|---|---|
| RTX 3090 | 45.7% | 0.88% / 5.0% |
| RTX PRO 6000 | 64.2% | 1.20% / 7.3% (three-round medians) |
| H800 | 48.5% | 0.22% / 1.2% |

Further RTX 3090 detail:
- at 150 / 200 / 300 us the median stays at 1.07 to 1.09% and the 95th percentile at 5.0 to 5.2%, while the
  share below rises to 59 / 62 / 67%;
- at 50 us no call lies below, and the 95th percentile above explodes (354%);
- below 100 us the conventions differ by up to 6.9 times, and the oracle differs for 27% of all calls.

**Suggested text:**
- "Next to every regret we give the noise floor of its data set, the regret of the per-shape best chunk chosen
  on one set of timings and scored on an independent one. It is 0.02 to 0.17% per call for the Triton scan in
  device time and 0.06 to 0.39% under CUDA graphs (scored on single rounds), against default regrets of 10 to
  130%. For the launch-inclusive Mamba-2 SSD calls it is 1.2 to 2.0%, so their default and table regrets (1.3 to
  2.5%) lie at the noise floor."
- "Regrets against the per-call oracle are biased upward because the oracle is the minimum of noisy means. The
  held-out exact-shape tables bound this bias at 0.06 to 0.35% in device time and under CUDA graphs."
- "The above-floor counts depend on the floor's factor. With 1.5 instead of 2 times the median batch-1 latency,
  chunk 256 is the oracle of 89% of the SSD 2.7B calls above the floor (94% at factor 2), and Gated DeltaNet's
  chunk 64 of 96% (99%). Of the 19 information data sets, the 12 Triton and reference-scan data sets have a
  chunk effect more than twice their noise floor, the Mamba-2 and Gated DeltaNet data sets do not, so the
  information tests on the latter are uninformative rather than negative."

---

## M5h: intervals, per-batch breakdowns, cluster structure, refit, multiplicity

**Script:** `taco_r2_stats_m5h_intervals.py`. **Result:** `taco_r2_stats_m5h.json`.

**Checks: 98/98 pass.**
- Table 5: every point and percentile interval equals the stored analysis (same replicates).
- Table 6 and Table 8: every point equals the printed value at its printed precision.

**Cluster structure.**
- Every captured-call table has 42 prompt clusters over 192 prompts, nested in the capture folds.
- Calls per cluster: 32 to 112 (median 48; Gated DeltaNet 24 to 84).
- Prompts per cluster: 2 to 10 (median 4).
- The largest cluster holds 4.7% of the calls.
- Sizes (calls): 112, 104, 88, 88, 80 x4, 72 x5, 64 x4, 48 x11, 40 x7, 32 x7.

**Table 5 with BCa** (device time; mean per-call regret, percentile and BCa intervals):
- BCa shifts the small regrets' intervals up slightly:
  - shape tree, RTX PRO 6000: 0.46 [0.23, 0.74], BCa [0.26, 0.78];
  - exact table, RTX 3090: 0.35 [0.11, 0.71], BCa [0.16, 0.86];
  - cost model, H800: 3.01 [1.90, 4.36], BCa [2.02, 4.70].
- Paired differences, in points and microseconds:
  - shape + entropy minus shape: RTX 3090 -0.03 [-0.22, 0.10] pp = -1.9 [-6.6, 0.07] us; 0 on the other two GPUs.
  - cost model minus exact table: +0.27 [0.12, 0.48], +3.17 [2.52, 3.77] and +2.81 [1.66, 4.18] pp on the three
    GPUs.
  - cost model minus best chunk per batch (held out), RTX 3090: -13.9 [-26.2, -4.0] pp per call but
    **+43.9 [20.8, 73.9] us per call**. The model wins per call and loses in absolute time, the reversal behind
    "9.7% latency-weighted against 3.5% per call".

**Table 6 with intervals** (mean per-call regret %, all calls; per batch in the JSON):

| Row | Reference 370M (launch-incl.) | Triton 1.4B (device) | Triton 2.8B (device) |
|---|---|---|---|
| Best single chunk (held out) | 30.2 [24.3, 36.3] | 15.4 [12.7, 17.9] | 15.5 [13.1, 17.7] |
| Entropy rule | 185.5 [133.6, 235.9] | 119.2 [70.3, 174.0] | 128.7 [77.9, 186.3] |
| Stock dispatch (separate binary) | 6.58 [4.94, 8.60] | -- | -- |
| Stock chunk, runtime-chunk build | 5.35 [3.93, 7.22] | -- | -- |
| Shape | 3.00 [2.41, 3.62] | 2.11 [1.11, 3.22] | 2.90 [1.41, 4.78] |
| Entropy only | 28.9 [24.2, 33.8] | 13.4 [10.9, 15.6] | 14.3 [10.1, 20.5] |
| Shape + entropy | 3.13 [2.51, 3.77] | 2.11 [1.11, 3.22] | 2.90 [1.41, 4.78] |
| Cost model | 4.89 [4.03, 5.77] | 3.65 [2.85, 4.40] | 0.79 [0.36, 1.25] |
| Table tuned on 370M | -- | 5.31 [3.81, 7.02] | 6.81 [5.42, 8.17] |
| Exact-shape floor (in sample) | 2.46 [1.74, 3.19] | 0.07 [0.04, 0.11] | 0.05 [0.03, 0.07] |
| Held-out headroom, covered (us; max shape) | 2.16 (14.5) | 0.18 (4.7) | 0.32 (4.5) |

- **Per-batch transfer rows.** The pooled 3.7% of the 1.4B row is the Triton Mamba-1 1.4B cost-model transfer,
  not Gated DeltaNet (the task brief's wording).

  | Row | Batch 1 | Batch 4 | Batch 16 |
  |---|---|---|---|
  | 1.4B cost model | 1.00 [0.49, 1.61] | **13.88 [11.04, 16.59]** | 0.93 [0.61, 1.28] |
  | 2.8B cost model | 0.95 [0.37, 1.64] | 0.24 [0.15, 0.34] | 0.85 [0.55, 1.14] |
  | 1.4B table tuned on 370M | 7.66 | 1.01 | 0.80 |
  | 2.8B table tuned on 370M | 9.79 | 0.87 | 1.82 |

- **Reference scan.** Shape + entropy minus shape is +0.13 [0.03, 0.25] pp (fixed predictions), so entropy makes
  the tree worse.

**Table 8 with intervals** (ms per forward pass, percentile; BCa in the JSON):

| Row | Default | Best static | Model | Table | Oracle scan ms / pass |
|---|---|---|---|---|---|
| Triton 370M, PRO 6000 | 4.42 [2.33, 6.91] | 2.03 [1.35, 2.73] | 0.30 [0.09, 0.57] | 0.018 [0.006, 0.031] | 13.1 |
| Triton 370M, RTX 3090 | 6.40 [4.65, 8.52] | 0.79 [0.48, 1.23] | 2.90 [1.68, 4.35] | 0.70 [0.03, 2.09] | 29.8 |
| Triton 370M, H800 | 4.43 [2.72, 6.66] | 1.76 [1.12, 2.65] | 1.29 [0.57, 2.42] | 0.016 [0.008, 0.025] | 24.2 |
| Triton 1.4B, RTX 3090 | 14.5 [10.1, 19.7] | 2.91 [2.01, 4.01] | 2.66 [1.65, 3.94] | 0.30 [0.05, 0.77] | 57.2 |
| Triton 2.8B, RTX 3090 | 25.2 [17.5, 35.0] | 4.04 [2.63, 5.81] | 1.53 [0.90, 2.18] | 0.53 [0.05, 1.47] | 90.5 |
| Reference scan 370M | 1.75 [0.72, 3.30] | -- | 0.66 [0.36, 1.10] | 0.30 [0.13, 0.60] | 21.9 |
| SSD 370M, PRO 6000 | 0.37 [0.33, 0.41] | -- | 0.34 [0.31, 0.38] | 0.28 [0.25, 0.32] | 25.2 |
| SSD 370M, RTX 3090 | 0.98 [0.93, 1.04] | -- | 0.92 [0.86, 0.98] | 0.89 [0.81, 1.01] | 48.2 |
| SSD 2.7B (above floor) | 0.24 [0.04, 0.61] | -- | 0.26 [0.05, 0.64] | 0.15 [0.06, 0.30] | 325 |
| GDN 1.4B (above floor) | 0.006 [0.000, 0.021] | -- | same | same | 77.2 |

The oracle scan ms per forward pass column also answers N13: the identical 4.42 ms on the RTX PRO 6000 and the
H800 sit on oracles of 13.1 and 24.2 ms.

**Refit inside the bootstrap** (shape + entropy minus shape, pp; 200 replicates; trees re-trained on resampled
clusters with random folds):

| Data set | Refit interval | Fixed-prediction interval |
|---|---|---|
| RTX PRO 6000 | [0.00, 0.00] | [0.00, 0.00] |
| RTX 3090 | [-0.04, 0.07] | [-0.20, 0.09] |
| H800 | [-0.03, 0.50] | [0.00, 0.00] |
| Mamba-2 RTX 3090 | [-0.12, 0.24] | [-0.08, 0.10] |
| Reference scan | **[-0.17, 0.25]** | **[0.03, 0.25]** |

- With training variability, the reference scan's "entropy hurts" becomes non-significant, and the H800's exact
  zero becomes an interval up to +0.5 points.
- The stored 50-fold-assignment repeated CV gives mean differences of -0.19 to +0.20 pp on all 19 data sets.

**Multiplicity** over all 19 data sets:
- **Family E (entropy), 133 tests.** These are the exact-key permutation tests in three strata, plus the CMI
  given the exact key and given the exact key and layer, at two discretizations.
  - 12 have p < 0.05; 4 survive Benjamini-Hochberg; 0 survive Holm.
  - The four survivors are the early noisy 1.4B launch-inclusive session (two tests), the 2.8B launch-inclusive
    session, and RTX 3090 device time. Their normalized CMI excess is 0.02 to 0.11 of H(oracle | key).
- **Family A (all statistics), 741 tests:** 88 raw, 16 BH, 0 Holm.
- With 1000 permutations the smallest attainable p is 0.001, so Holm over 133 tests cannot reject. The tests
  should be labelled descriptive, or BH reported.

**Suggested text:**
- "The 2360 captured calls form 42 prompt clusters of 32 to 112 calls (2 to 10 prompts; median 48 calls and 4
  prompts), nested in the cross-validation folds. All intervals resample these clusters; for the selectors of
  Table 5 with regrets below 5%, BCa intervals lie at most 0.37 points from the percentile intervals."
- "The 1.4B cost-model regret of 3.7% is 1.0% (0.5 to 1.6) at batch 1, 13.9% (11.0 to 16.6) at batch 4 and 0.9%
  (0.6 to 1.3) at batch 16; on the 2.8B checkpoint it is 0.95, 0.24 and 0.85%."
- "When the trees are re-trained inside every bootstrap replicate, the effect of adding entropy to the shape
  tree lies within -0.17 to +0.50 points on all five data sets re-fitted (three GPUs, Mamba-2, reference scan). Across the 19 data sets, 12 of 133 entropy tests are
  nominally significant, 4 survive a Benjamini-Hochberg correction and none a Holm correction; we report these
  tests as descriptive."

---

## M5b: the RTX 3090 tensor-parallel sweep is noise-dominated

**Script:** `taco_r2_stats_m5b_tp_noise.py`. **Result:** `taco_r2_stats_m5b.json`.

**Reproduction: 28/28 stored values match exactly**, including the 2000-replicate shape-bootstrap intervals:
- retuned-at-shard table 14.87% [11.13, 19.05];
- cost model minus retuned -8.35 [-12.38, -4.59] (Triton) and -4.35 [-7.06, -1.86] (reference);
- H800 retuned 0.75%, cost model minus retuned +2.16 [1.16, 3.22];
- unsharded table 36.8% / 22.1%.

The "52% per-round argmin agreement" is the share of all 168 RTX 3090 Triton shapes whose argmin is identical
in all three rounds: 51.8%. The other cells are 62.5% (RTX 3090 reference), 69.6% (H800 Triton) and 88.7%
(H800 reference).

**Round-to-round variation (shard shapes):**

| | RTX 3090 Triton | RTX 3090 reference | H800 Triton | H800 reference |
|---|---|---|---|---|
| Between-round relative SD, median / p90 | 4.2 / 24.6% | 1.4 / 16.7% | 0.61 / 2.9% | 0.43 / 1.9% |
| abs(relative difference) between two rounds, median / p90 | 2.4 / 34.7% | 1.2 / 20.3% | 0.67 / 3.6% | 0.49 / 2.3% |
| Argmin agrees in all three rounds | 45% | 59% | 68% | 87% |

- On the RTX 3090 Triton kernel, one round deviates from the three-round median by more than 10% in 26% of
  (shape, chunk) cells.
- The noise floor there is 14.9% pooled but heavy-tailed: median shape 0.66%, 90th percentile 49.8%.

**Key recomputation: every selector scored the same way** (each held-out round against its own oracle, mean over
three rounds):

| Comparison | Cost model | Retuned table | Difference [CI] |
|---|---|---|---|
| RTX 3090, Triton | 13.86% [10.67, 17.44] | 14.87% | -1.00 [-4.02, +1.93], not significant |
| RTX 3090, reference | -- | -- | -2.20 [-4.11, -0.35] |
| H800, Triton | -- | -- | +2.33 [1.32, 3.38] |

- The stored -8.35 came from scoring the model against the three-round-mean oracle and the retuned table
  against a single noisier round.
- The remaining reference-scan advantage of the model tracks per-shape noise: Spearman rho = 0.43 between a
  shape's round-to-round noise and the model's advantage there (H800: 0.06).
- **Shard effect net of noise** (unsharded table minus retuned, both cross-round):

  | | Triton | Reference |
  |---|---|---|
  | RTX 3090 | +30.7 [18.7, 44.7] pp; 75 of 126 shard shapes above their own floor | -1.73 [-3.83, 0.20] |
  | H800 | +21.5 [12.2, 32.5] | +6.16 [3.36, 9.51] |

**Suggested text:**
- "On the RTX 3090 three rounds of the identical kernel differ by a median of 2.4% and a 90th percentile of 35%,
  and the per-round best chunk agrees in all three rounds for only 45% of the shard shapes. A table retuned at
  the exact shard shape and scored on another round therefore loses 14.9%, the noise floor of that sweep."
- "Scored the same way, the cost model and the retuned table are indistinguishable on the RTX 3090 (difference
  -1.0 points, interval -4.0 to 1.9). On the H800, whose floor is 0.75%, the model is 2.3 points (1.3 to 3.4)
  behind, so we base the shard-width results on the H800."
- "The shard effect itself exceeds the noise: a table tuned at the unsharded width loses 22% on the H800."

---

## M5i (offline part): continuous-outcome test where the headroom exceeds one synchronization

**Script:** `taco_r2_stats_m5i_continuous.py`. **Result:** `taco_r2_stats_m5i.json`.

**Reproduction.**
- H(oracle | exact key) = 1.968 bits (review 1.97).
- CMI excess -0.00028 bits, p = 0.51 (stored -0.00028, p = 0.50).
- Five covered shapes exceed 21.4 us: (8192, 16) 43.0, (4096, 16) 31.2, (128, 16) 27.0, (4096, 1) 24.3 and
  (256, 1) 21.5 us.
- R1's Spearman check is reproduced exactly: rho -0.056 to 0.315, minimum p 0.134, 24 to 224 calls.
- Correction to the review: "3.2 chunks within 2%" is the RTX PRO 6000 value. On this RTX 3090 data set it is
  2.48.

**Tests.** A linear mixed model with crossed random intercepts for layer and prompt cluster (REML via Woodbury;
it reproduces OLS exactly at zero variance components).
- **Test A:** top-2 log-latency contrast on within-key entropy, with key fixed effects.
- **Test B:** per-chunk log latency, with a chunk x entropy interaction and the call effect removed by
  within-call contrasts.

**Mamba-2 SSD 370M, launch-inclusive, RTX 3090, all 2156 covered calls (29 keys):**
- **Test A.** The entropy effect across the within-key 5th to 95th percentile (2.3 bits) is 0.89 us per call.
  - Interval: Wald [-3.97, 5.75], cluster-bootstrap refit [-3.04, 5.13]; p = 0.72.
  - Minimum detectable effect: 6.95 us.
  - TOST at half a synchronization (+-10.7 us): equivalent.
  - Implied gain of an entropy-aware switch: 0.00 us (upper bound 0.09 us).
- **Test B.** The interaction is not significant: chi2(4) = 2.72, p = 0.61. The largest slope difference is
  -5.0 us [-11.2, 1.1] (MDE 8.7 us).
- **The five above-sync shapes (328 calls)** are under-powered:
  - Test A: 13.4 us [-12.1, 38.9], MDE 36.4 us, inconclusive.
  - Only (8192, 16) is nominally significant (n = 24, p = 0.049; 0.25 after Holm): 176 us [1, 350] on a 10.8 ms
    call, implied gain 8.6 us (upper bound 28 us).
- **Same calls under CUDA-graph replay** (three rounds, same board):
  - Test A: 0.21 us [-0.46, 0.87], MDE 0.95 us, equivalent.
  - All five shapes have **0.00 us** headroom in sample, held out and cross-round. Over all 29 covered shapes the
    held-out maximum is 0.18 us, and no shape exceeds one synchronization.
  - So the offline part of the GPU re-timing is already answered by existing data; a lab-host run would only add
    a cross-instance check.
- **Equivalence margin.**
  - The 0.5-point margin exceeds the shape tree's whole residual (0.465 / 0.383% on the two Mamba-1 device-time
    sets), so it cannot fail in the beneficial direction there.
  - In absolute terms, shape + entropy minus shape is -1.90 us [-6.12, 0.06] (Mamba-1, RTX 3090) and -0.03 us
    [-0.82, 0.70] (Mamba-2, RTX 3090).
  - All five data sets checked (Mamba-1 on three GPUs, Mamba-2 on two) pass TOST at +-10.7 us and at half the
    tree's residual.

**Suggested text:**
- "Where the exact-shape headroom exceeds one host synchronization (Mamba-2, launch-inclusive), we tested a
  continuous outcome. A mixed model of the log-latency ratio of the two fastest chunks on entropy within each
  exact shape, with layer and prompt-cluster random effects, gives an entropy effect of 0.9 us per call across
  the 5th to 95th entropy percentiles (95% interval -4.0 to 5.8 us; minimum detectable effect 7.0 us). This is
  equivalent to zero within half a synchronization (+-10.7 us), and an entropy-aware switch would gain at most
  0.1 us per call."
- "Replayed as CUDA graphs in three rounds, the same calls show no headroom at these five shapes (0.00 us held
  out and cross-round; at most 0.18 us over all 29 covered shapes), so their launch-inclusive headroom is timing
  noise."
- "Because a 0.5-point margin exceeds the shape tree's entire residual regret (0.38 to 0.46%), we state
  equivalence in absolute terms: adding entropy changes the tree's regret by -1.9 to 0.0 us per call, inside
  half a synchronization on each of the five data sets checked."

---

## M6a: workload-weighted headline numbers

**Script:** `taco_r2_stats_m6a_workload_weights.py`. **Result:** `taco_r2_stats_m6a.json`.

**Availability.** The distributions are available offline:
- the per-window prompt-length bucket histograms of all ten trace windows are stored in the serving-replay
  result, and were re-derived from the raw traces (all ten match);
- the batched (chunked-prefill) step distributions were re-simulated with the replay's own scheduler, and their
  step counts and sequences per step match the stored replay in all ten settings.

What could not be produced:
- packed-kernel weights, because the captured calls are padded batches, so no matching cell exists;
- batched weights for the 24-hour and whole-week windows, which were not attempted.

**Checks: 97/97.**
- Table 5 rows reproduced.
- The free length rule clip(nextpow2(L), 32, 1024): per call 5.86 / 7.76 / 10.66%, LW 5.96 / 16.31 / 13.94%
  (RTX PRO 6000 / RTX 3090 / H800), exactly the review's values.
- Table 7 defaults and their stored intervals reproduced.

**Weightings.**
- **W1, request-level prefill** (batch 1, exact prompt length; all ten windows): 0.5 to 4.8% of the trace mass
  is remapped to the nearest captured bucket; BurstGPT's busiest day 16.9%.
- **W2, chunked prefill** (colocated, load 0.5, padded kernel, T = 2048 / 8192; five windows): 1.4 to 4.7
  sequences per step, and 26 to 74% of the mass remapped to the nearest captured batch in log2.

Post-stratified estimates are recomputed in every replicate. Effective clusters are 10.5 to 26 (W1) and 5.8 to
20 (W2), against 37 unweighted.

**Table 7, recoverable LW (default minus held-out table, pp), point estimates as ranges over windows:**

| Weighting | SSD 370M 3090 | SSD 370M PRO | SSD 2.7B 3090 | SSD 2.7B PRO | GDN 3090 | GDN PRO |
|---|---|---|---|---|---|---|
| Unweighted | 2.79 [1.56, 5.10] | 1.72 [0.95, 3.41] | 2.17 [1.07, 4.42] | 5.39 [2.70, 7.59] | 0.45 [-0.83, 1.86] | 2.94 [1.84, 5.19] |
| W1, Azure windows | 4.66-8.23 | 1.89-6.11 | 1.76-3.43 | 6.22-11.70 | 0.80-3.25 | 5.13-13.50 |
| W1, incl. BurstGPT | 4.66-17.14 | 1.89-8.63 | 1.76-20.04 | 6.21-16.81 | 0.80-22.92 | 5.13-25.11 |
| W2, T = 2048 | 3.46-7.03 | 0.17-3.55 | 1.03-5.53 | 3.74-6.74 | -0.18-1.64 | 7.09-11.52 |
| W2, T = 8192 | 0.14-4.03 | 0.05-2.59 | 0.04-3.56 | -0.52-4.59 | -1.23-1.18 | 3.58-9.01 |

Example with intervals (Azure 2023 code, first hour, W1): SSD 370M 3090 5.49 [4.91, 5.96]; SSD 2.7B PRO 7.25
[6.76, 7.62]; GDN PRO 6.65 [5.92, 7.34].

**Triton Mamba-1, device time** (per-call % / LW %; ranges over windows):

| GPU, policy | Unweighted | W1 | W2 |
|---|---|---|---|
| PRO 6000, fixed 256 | 48.6 / 33.8 | 9.2-28.8 / 8.1-15.7 | 11.5-70.0 / 10.3-24.9 |
| PRO 6000, free length rule | 5.86 / 5.96 | 2.2-11.5 / 12.0-16.8 | 5.5-10.1 / 6.3-14.0 |
| PRO 6000, exact table | 0.14 / 0.13 | 0.06-0.26 / 0.40-0.68 | 0.11-0.58 / 0.06-0.93 |
| PRO 6000, cost model | 0.41 / 2.29 | 0.09-0.37 / 0.60-0.84 | 0.39-2.15 / 0.85-3.50 |
| RTX 3090, best chunk per batch | 17.4 / 2.67 | 3.8-42.8 / 6.0-13.3 | 3.6-11.4 / 2.4-5.6 |
| RTX 3090, free length rule | 7.76 / 16.31 | 2.1-9.9 / 16.1-18.6 | 6.7-11.3 / 12.9-23.0 |
| RTX 3090, cost model | 3.52 / 9.73 | 0.42-2.44 / 2.70-4.70 | 1.44-4.13 / 2.04-7.69 |
| H800, free length rule | 10.66 / 13.94 | 4.7-24.8 / 32.3-39.1 | 7.9-21.3 / 9.3-26.9 |
| H800, exact table | 0.20 / 0.07 | 0.03-0.11 / 0.08-0.13 | 0.02-0.22 / 0.00-0.22 |

Caveats:
- Each captured (bucket, batch) cell comes from only 3 to 5 of the 42 clusters, and no captured length falls in
  bucket 32. The weighted intervals hold the trace mix fixed and are probably too narrow, so read them as ranges.
- The alignment-aware variant is not supported by the data. Captured calls are 79.5% aligned against 5 to 7% in
  the traces; it needs 23 to 75% fallback mass and gives implausible flips. It is in the JSON as a sensitivity
  check only.

**Suggested text:**
- "Re-weighted to the length mix of request-level prefill in ten windows of the Azure 2023/2024 and BurstGPT
  traces, the share of graph-replayed scan time that a held-out shape-keyed table recovers from the SSD and Gated
  DeltaNet defaults rises from 0.45 to 5.4% on the captured mix to 0.8 to 13.5% on the Azure windows and up to
  25% on BurstGPT, whose prompts are short. Under chunked prefill with an 8192-token budget it is -0.5 to 4.6% on
  SSD and -1.2 to 9.0% on Gated DeltaNet."
- "Under the same request-level weights the free length rule loses 2.1 to 24.8% per call and 12 to 39% of the
  Triton scan's time in device time, while a held-out exact-shape table stays within 0.5% per call and 1.3% of
  scan time on every GPU and trace window."
- "Each (length bucket, batch) cell of the captured calls comes from only three to five of the 42 prompt
  clusters, so these workload-weighted estimates rest on 6 to 26 effective clusters and are conditional on the
  trace mix."

---

## M5d: H800 input effects, numbers for the main text

**Script:** `taco_r2_stats_m5d_h800_inputs.py`. **Result:** `taco_r2_stats_m5d.json`.

Everything is recomputed from the raw rows with the existing value-invariance functions. 20/20 stored values
match; the RTX 3090 device-time tag has no stored analysis.

**H800, Mamba-1, 36 fixed-chunk configurations:**

| | Launch-inclusive | Device time |
|---|---|---|
| Significant input effects (BH 0.05) | 28 of 36 | 36 of 36 |
| Largest input range | 47.6% (L = 2048, chunk 32) | 39.9% (L = 2048, chunk 32) |
| Largest significant range at the input-agnostic chunk | 4.4% | 5.1% |
| Median input range | 3.9% | 2.6% |
| Largest regret of the input-agnostic chunk | 5.56% | 5.65% |
| TOST at the input-agnostic chunk: median bound; equivalent at 2% / 5% | 1.9%; 3 / 5 of 6 | 0.48%; 4 / 6 of 6 |

Notes on the table:
- In the 47.6% launch-inclusive configuration the slowest input is the captured activations.
- The input-agnostic chunk is 256 at L = 512 and 1024 at L = 2048 and 8192.
- The per-input best chunk moves between 256 and 1024 at L = 2048 only (3 to 4 of 8 inputs pick 256).

Tie share of the input-agnostic chunk at L = 2048:
- CI-overlap rule: 54 to 71%;
- 2% rule: 70.8% (minimum over modes, both conventions);
- L = 512 and L = 8192: 96 to 100%.

**Counterparts on the other GPUs:**
- RTX PRO 6000, launch-inclusive, all kernels: 12 of 84 significant, up to 43.0%, 8.6% at the selected chunk.
  Device-time Mamba-1: 15 of 36, up to 24.2%.
- RTX 3090: 24 of 84, at most 4.3%. Device-time Mamba-1: 33 of 36, at most 4.1%.

No placement or L2 control exists on the H800.

**Suggested text:**
- "On the H800, 28 of 36 fixed-chunk configurations show a significant input effect launch-inclusive and all 36
  in device time, with ranges up to 47.6% and 39.9% at non-selected chunks (L = 2048) but at most 5.1% at the
  selected chunk."
- "At L = 2048 the per-input best chunk moves between 256 and 1024, so the input-agnostic chunk is tied with an
  input's own best in only 54 to 71% of the cells and costs up to 5.7%; at L = 512 and 8192 it is the best chunk
  of every input."
- "No placement or L2 control was run on the H800, so these effects cannot be attributed to buffer placement
  rather than to the values."
