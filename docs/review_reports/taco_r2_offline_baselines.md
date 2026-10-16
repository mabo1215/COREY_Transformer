# taco-r2 offline re-analyses: fair baselines, launch coupling, cost-model protocols, index limit

Cycle taco-r2, 2026-10-16. Items M2b, M2c, M2d (offline part), M4a, M4b, M4c, M4d, M7b (offline part), and the
minor items N15, N17, N18, N23. CPU only: every number below was recomputed from the stored result files by the
scripts named per item; nothing was timed. No manuscript, ledger or existing source file was changed.

## Conventions used throughout

- **Per call**: mean of the per-call relative regret lat(chosen chunk) / lat(per-call oracle) - 1, in percent.
- **LW (latency-weighted)**: 100 x sum of excess latency / sum of oracle latency, i.e. the share of the scan time at
  the per-call oracle that a policy adds. This is the convention of Tables 4 and 7 (e.g. chunk 256 on the RTX PRO
  6000: 33.8%). The share of the policy's own time (lw_share_of_policy_pct) is also stored in every JSON.
- **Intervals**: 95% percentile intervals over 2000 prompt-cluster bootstrap replicates (42 clusters on every data
  set), with the same seeds as the stored taco-r1 analyses, so intervals of the same quantity reproduce exactly.
- **Held out**: chosen on the training prompt folds of the grouped five-fold split and scored on the held-out fold.
- **Data sets (Triton Mamba-1 scan, device time, primary)**: RTX PRO 6000 and RTX 3090 = medians of three shuffled
  rounds (the data set of Table 5); H800 = one block per chunk. The stored cost-model analyses use two other views
  of the same calls (RTX PRO 6000: means pooled over the three rounds; RTX 3090: the single-block session); both
  views are reported where values differ ("own table" below).

## Result files and scripts

| Item | Script | Result |
|---|---|---|
| shared helpers | `src/experiments/tpds/taco_r2_baselines_common.py` | -- |
| M2b, M2c | `src/experiments/tpds/taco_r2_untuned_baselines.py` | `src/experiments/results/tpds/taco_r2_baselines_m2b.json`, `..._m2c.json` |
| M2d | `src/experiments/tpds/taco_r2_fixed_launch_offline.py` | `src/experiments/results/tpds/taco_r2_baselines_m2d.json` |
| M4a, M4b, M4c, N15 | `src/experiments/tpds/taco_r2_cost_model_protocols.py` | `src/experiments/results/tpds/taco_r2_baselines_m4.json` |
| M4d | `src/experiments/tpds/taco_r2_latency_form.py` | `src/experiments/results/tpds/taco_r2_baselines_m4d.json` |
| M7b | `src/experiments/tpds/taco_r2_index_limit.py` | `src/experiments/results/tpds/taco_r2_baselines_m7b.json` |
| N17, N18, N23 | `src/experiments/tpds/taco_r2_minor_checks.py` | `src/experiments/results/tpds/taco_r2_baselines_minor.json` |

Run order: `PYTHONPATH=. python3 -m src.experiments.tpds.taco_r2_cost_model_protocols` first (about 3 min on 6
processes; M2b reads its per-call choices), then the others (each 0.5 to 2 min). Every JSON carries a `checks`
block; 213 of 215 reproduction checks match, the two exceptions are explained under M4c.

## Reproduction of the values quoted in the review

| Quoted value | Recomputed | Note |
|---|---|---|
| free length rule per call 5.86 / 7.76 / 10.66%, LW 5.96 / 16.31 / 13.94% | 5.86 / 7.76 / 10.66; 5.96 / 16.31 / 13.94 | exact (primary data sets) |
| chunk 256: 48.6 / 96.4 / 50.9% | 48.59 / 96.35 / 50.89 | exact |
| best chunk per batch 20.0 / 17.4 / 24.7% | 19.99 / 17.40 / 24.70 | in sample = held out, as R1 found |
| best static chunk by latency (512) loses 15.5% of scan time | 15.51% | exact |
| "calibration buys 1.7 / 1.5 / 7.7 points per call" | 1.66 / 1.55 / 7.65 on the same data set | the review subtracted the a-priori model of the single-block RTX 3090 session (6.23) from the rule on the round-median set (7.76); on one data set the result is the same to 0.04 points |
| calibration-only length tree 16.6 / 20.4 / 38.4% | 16.59 / 20.43 / 38.36 | exact |
| length-bucket table 2-20 us per call (Table S23) | RTX PRO 6000 device 2.15, RTX 3090 launch-inclusive 8.30, H800 device 19.76 us | Table S23's RTX PRO 6000 and RTX 3090 device-time rows use pooled round means although labelled round medians (2.22 and 6.19 us on round medians, against 2.2 and 5.6 printed) |
| six protocols PRO 2.24, 0.38, 0.56, 4.16, 0.38, 0.56; 3090 6.23, 2.49, 1.74, 6.23, 2.49, 0.88; H800 2.14-3.01 | identical on the own tables | on the primary data sets: 2.26, 0.40, 0.58, 4.20, 0.40, 0.58 and 6.21, 2.46, 1.76, 6.21, 2.46, 0.85 |
| a-priori form LW 26.9% (CI 23.5-29.3) on the RTX PRO 6000 | 26.92% [23.5, 29.3] | exact |
| frozen form LW 2.3 / 9.7 / 5.3% | 2.29 / 9.73 / 5.33 | exact |
| per batch: RTX 3090 15.9% at batch 16, 1.5% at 1 and 4; H800 6.3% at batch 4 | 15.92 / 1.52 / 1.44 (own table); 6.26 | exact |
| aligned/unaligned 0.22/1.01, 3.89/2.25, 2.69/4.24 | 0.22/1.01, 3.89/2.25, 2.69/4.24 | own tables |
| restriction 4.20 / 0.85 / 3.53; MLP five seeds 1.59 / 1.31 / 7.13; structural 3.75 / 4.90 / 1.11 | 4.20 / 0.85 / 3.53; 1.59 / 1.31 / 7.13; 3.75 / 4.90 / 1.11 | MLP values on the own tables; primary sets 1.61 / 1.31 / 7.13 and 3.77 / 4.90 / 1.11 |
| paired full - restriction: H800 -0.53 [-1.35, +0.60], PRO -3.80 [-4.78, -2.89], 3090 +2.66 [+2.12, +3.17] | identical | |
| H800 add_l2_miss = frozen form (3.0077%) | 3.00769% both | identical choices (fitted tau_d = 0.0071) |
| full-protocol refit 0.73 / 4.28 / 6.05% | 0.73 / **1.55** / 6.05 | **differs on the RTX 3090**, see M4c |
| R^2 of the objective 0.89 / 0.93 / 0.94 | 0.895 / 0.926 / 0.938 | |
| search vs full protocol, largest 226% (launch-inclusive RTX 3090) | 225.8% | device time: 7.0 / 15.8 / 1.3% |
| 72 static re-measurements alone 0.56 / 1.8 / 6.1% | 0.56 / **1.55** / 6.05 | RTX 3090 differs, see M4c |
| 12-key bootstrap [0.38, 2.26] and [1.74, 6.09] | identical | |
| leave-one-key-out 0.8 / 7.4 / 3.7% | 0.76 / 7.43 / 3.67 | |
| 50 random starts: 16% lower objective, 1.7% regret (RTX 3090) | 15.9%, 1.74% | |
| Shapley "32.7 = 29.3 + 3.4" (launch-inclusive PRO) and device time "50.6 chunk, -6.2 launch" | 32.65 = 29.28 + 3.37; 44.46 = 50.63 - 6.17 | |
| R2: RTX 3090 fixed (4, 2): chunk 128 at 11 of 12 keys, ~0%, 17% along the table; H800 fixed (8, 2): 11 of 12, 1.2% vs 21% | 11/12, 0.03%, 17.1%; 11/12, 1.17%, 20.7% | true for one and two stages |
| SSD chunk 32 at width 5120, batch 16, L > 6528; TFLA chunks 32 and 64 at the largest shapes | confirmed; see M7b | |
| OLS 1.8% (appendix) | 1.7447% -> 1.7% | N15 |
| N17: 0.22% with a plain length-bucket fallback; 0.14 vs 0.37 us | 0.224%; 0.136 vs 0.366 us | |
| N18: 5.35%, 6.6%, 3.1%, ratio 0.86-1.13 | 5.35%, 6.58%, 3.09%, 0.864-1.134 | |

---

## M2b. Fair untuned baselines

**Computed.** On every per-call data set of the Triton Mamba-1 scan (device time on three GPUs, launch-inclusive,
the 1.4B and 2.8B widths, CUDA-graph replay on two GPUs), of the reference CUDA scan and of the production kernels
(SSD 370M / 2.7B and Gated DeltaNet 1.4B under graph replay on two GPUs, plus their host-bound and device-time
tables): the free length rule C = clip(nextpow2(L), C_min, C_max); for kernels with a library default also the
capped default max(C_min, min(C_default, nextpow2(L))); one chunk for all calls and one per batch size, each chosen by
per-call regret or by total latency, in sample and held out; held-out tables keyed on (length bucket, batch) with and
without the alignment bit; the held-out exact-shape table; the cost model (frozen fit and the two a-priori forms of
M4a). Per call, LW, us per call, ms per forward pass, per batch and paired differences against the free rule, all
with cluster intervals.

**Triton Mamba-1 scan, device time (per call % [95% CI] / LW %; ms per forward pass).**

| Policy | RTX PRO 6000 | RTX 3090 | H800 |
|---|---|---|---|
| chunk 256 (borrowed SSD default) | 48.6 [32.9, 65.4] / 33.8; 4.42 | 96.4 [55.7, 142.3] / 21.5; 6.40 | 50.9 [29.7, 74.1] / 18.3; 4.42 |
| **free length rule** | **5.86 [4.25, 7.57] / 5.96 [3.0, 10.8]; 0.78** | **7.76 [5.98, 9.69] / 16.31 [13.9, 19.9]; 4.86** | **10.66 [7.54, 14.18] / 13.94 [9.5, 21.4]; 3.37** |
| one chunk, held out, by total latency | 98.6 / 15.5 (512); 2.03 | 36.9 / 19.8 (64 or 128); 5.89 | 50.9 / 18.3 (256); 4.42 |
| one chunk, held out, by per-call regret | 48.6 / 33.8 (256) | 33.2 / 18.0 (64) | 50.9 / 18.3 (256) |
| one chunk per batch, held out, by per-call regret (Table 5) | 20.0 / 27.0; 3.53 | 17.4 / 2.66; 0.79 | 24.7 / 20.8; 5.03 |
| one chunk per batch, held out, by total latency | 98.6 / 15.5; 2.03 | 17.4 / 2.66; 0.79 | 41.2 / 7.30; 1.76 |
| length-bucket table, held out | 0.25 / 0.81; 0.11 (2.2 us) | 0.91 / 1.00; 0.30 (6.2 us) | 1.55 / 3.93; 0.95 (19.8 us) |
| length-bucket + alignment table, held out | 0.20 / 0.65 | 0.39 / 2.38 | 0.20 / 0.07 |
| exact-shape table, held out | 0.14 / 0.13; 0.02 | 0.35 / 2.35; 0.70 | 0.20 / 0.07; 0.02 |
| cost model, frozen form | 0.40 / 2.29; 0.30 | 3.52 / 9.73; 2.90 | 3.01 / 5.33; 1.29 |
| cost model, a priori (paper's protocol) | 4.20 / 26.9; 3.52 | 6.21 / 10.28; 3.06 | 3.01 / 5.33; 1.29 |
| cost model, a priori (pre-specified, M4a) | 0.58 / 2.32; 0.30 | 0.85 / 0.86; 0.26 | 2.14 / 3.14; 0.76 |

Free rule per batch (per call / LW): RTX PRO 6000 5.8/12.6, 4.8/3.7, 7.6/3.7 at batch 1/4/16; RTX 3090 5.2/14.9,
10.4/11.0, 15.8/18.8; H800 12.8/25.0, 6.6/5.9, 6.9/12.3. Other conventions (per call / LW): launch-inclusive 5.22/5.42,
8.56/15.09, 8.39/13.09; graph replay 6.02/6.04 (RTX PRO 6000), 7.66/16.26 (RTX 3090); RTX 3090 1.4B 11.27/19.24 and
2.8B 16.35/24.35 (at the larger widths small chunks win and the rule is poor by LW: 11.0 and 22.0 ms per pass).

**What the calibrated model buys over the free rule (same data set, points; positive = better than the rule).**

| | RTX PRO 6000 | RTX 3090 | H800 |
|---|---|---|---|
| frozen form, per call | 5.46 [3.95, 7.04] | 4.25 [2.75, 5.97] | 7.65 [4.98, 10.60] |
| frozen form, LW | 3.67 [0.67, 8.21] | 6.58 [3.15, 11.98] | 8.61 [5.55, 13.17] |
| a priori (paper's protocol), per call | 1.66 [0.16, 3.15] | 1.55 [0.21, 3.07] | 7.65 [4.98, 10.60] |
| a priori (paper's protocol), LW | **-20.96 [-25.65, -14.06]** | 6.03 [2.74, 11.04] | 8.61 [5.59, 13.13] |
| a priori (pre-specified), per call / LW | 5.28 / 3.64 | 6.91 / 15.46 | 8.52 / 10.80 |
| length-bucket table, per call / LW | 5.61 / 5.15 | 6.86 / 15.32 | 9.11 / 10.01 |

**Why the calibration-only length tree (16.6 / 20.4 / 38.4%) loses to the free rule.** The twelve calibration keys
have four lengths (128 to 8192), so the tree has no shape below L = 128, while 584 of the 2360 calls (24.7%) are
shorter. The tree extends the label of L = 128 to them (chunk 64, 64 and 128 on the three GPUs), whereas their oracle
is 32 or 64; on these calls the tree loses 14.0, 47.7 and 106.3% per call against 0.0, 0.16 and 1.8% for the rule,
which picks the next power of two of the length. Its leaves are also labelled with the chunk of lowest summed
normalized latency over batches 1, 4 and 16 at each calibrated length, a compromise that is wrong at every batch where
the per-batch optima differ (e.g. RTX 3090, L = 512: per-batch optima 128 / 32 / 32, tree label 512). The rule encodes
a priori what the tree cannot learn from four lengths: the optimal chunk grows with L and never exceeds it.

**Production kernels (per call / LW, CUDA-graph replay, three rounds).**

| Kernel, GPU | library default | capped default | free length rule | per batch, held out, latency | bucket table, held out |
|---|---|---|---|---|---|
| SSD 370M, RTX 3090 | 19.09 / 2.80 | **4.37 / 0.83** | 19.70 / 23.19 | 9.82 / 2.14 | 0.06 / 0.01 |
| SSD 370M, RTX PRO 6000 | 10.11 / 1.75 | **6.19 / 0.81** | 22.80 / 21.81 | 10.11 / 1.75 | 0.16 / 0.05 |
| SSD 2.7B, RTX 3090 | 22.31 / 2.20 | **1.15 / 0.24** | 14.62 / 21.52 | 12.86 / 2.01 | 0.07 / 0.03 |
| SSD 2.7B, RTX PRO 6000 | 19.21 / 5.85 | **6.24 / 4.30** | 22.91 / 22.07 | 19.21 / 5.85 | 0.09 / 0.00 |
| GDN 1.4B, RTX 3090 | 16.17 / 1.02 | 12.16 / 0.79 (= free rule) | 12.16 / 0.79 | 16.17 / 1.02 | 0.04 / 0.00 |
| GDN 1.4B, RTX PRO 6000 | 23.91 / 3.19 | 19.78 / 2.94 (= free rule) | 19.78 / 2.94 | 2.99 / 0.45 | 0.01 / 0.02 |

The uncapped free rule is wrong for SSD (it serves chunk 512 to long calls, LW 21.5-23.2%); the capped default
reproduces the stored N9 values exactly. Host-bound (launch-inclusive SSD 370M) every policy lies within 1.3-2.5% per
call (default 2.53 / 1.69%, bucket table 2.19 / 1.28%).

**Reference CUDA scan (RTX 3090).** Its stock dispatch is exactly clip(nextpow2(L), 128, 2048) on all 2360 calls.
Same binary: 5.35% [3.92, 7.25] per call, LW 7.99% launch-inclusive; 3.09% [1.47, 5.08], LW 8.12% in device time.
Held-out bucket table: 3.04 / 0.92% launch-inclusive, 0.66 / 0.58% in device time (details under N18).

**Suggested text.**
- "Our Triton scan has no library default. As its untuned reference we use the stock dispatch of the reference CUDA
  scan transplanted to its chunk range, C = clip(nextpow2(L), 32, 1024), which needs no timing: in device time it
  loses 5.9, 7.8 and 10.7% per call (6.0, 16.3 and 13.9% of the scan time) on the RTX PRO 6000, RTX 3090 and H800,
  against 48.6, 96.4 and 50.9% (33.8, 21.5 and 18.3%) for chunk 256 borrowed from Mamba-2. A held-out table keyed on
  the length bucket and the batch size, profiled in seconds, comes within 0.25, 0.91 and 1.55% per call (0.8, 1.0 and
  3.9% of the scan time)."
- "Measured against this rule, the cost model with terms chosen a priori gains 1.7, 1.6 and 7.7 points per call, and
  on the RTX PRO 6000 it adds 21 points of scan time that the rule does not lose."
- "For the production kernels the analogous zero-cost rule caps the library default at the next power of two of the
  length: under CUDA-graph replay it lowers the SSD default's per-call regret from 19.1 and 22.3% to 4.4 and 1.2% on
  the RTX 3090 (370M and 2.7B; 2.8 and 2.2% to 0.8 and 0.2% of the scan time) and from 10.1 and 19.2% to 6.2 and 6.2%
  on the RTX PRO 6000, but helps Gated DeltaNet little (16.2 and 23.9% to 12.2 and 19.8%)."

---

## M2c. One definition of "default" and "best static"

**Inventory (manuscript as it stands).**

| Where | "Default" | "Best static" |
|---|---|---|
| Sec. 6 opening (main.tex:653-658) | Triton: chunk 256 borrowed from Mamba-2 | best chunk per batch, per-call criterion, quoted with its LW in parentheses ("20% (27%)") |
| Table 4 | -- | "Best static chunk / Static regret": lowest mean per-call regret per batch, in sample; the "LW" column is a different chunk (least total latency, 512) |
| Table 5 | "Default chunk 256, entropy rule" (one row) | "Best single chunk, held out" (per-call criterion); "Best chunk per batch, held out" (per-call criterion, computed in sample in the stored analysis; values identical) |
| Table 5, a-priori row | -- | the "terms chosen a priori" row is evaluated on other data sets (RTX PRO 6000 pooled means: 4.16; RTX 3090 single-block session: 6.23) than every other row (round medians: 4.20 and 6.21); on the RTX PRO 6000 it is the same form as the "Three-parameter restriction" row, which therefore shows 4.20 next to 4.16 |
| Table 6 | reference scan: "Stock length dispatch" = the installed binary timed in its own block against the runtime-chunk oracle (6.6%) | "Best single chunk" (per-call criterion, in sample) |
| Table 7, Fig. 2 | SSD 256, Gated DeltaNet 64; Triton: the best single chunk of the regime (per-call criterion, in sample) | -- |
| Table 8 | Triton 256; reference scan stock dispatch; SSD 256; GDN 64 | best chunk per batch by total latency, in sample; none for the production kernels |
| Table S23 (static/bucket) | -- | per batch, per-call criterion, in sample; bucket table by mean latency on pooled round means while the row is labelled round medians |
| calibration-budget figure | chunk 256 | per batch, held out, per-call criterion |
| autotune table | default launch table | best static chunk per key at its default launch; "single global static chunk" by mean ratio to the tuned kernel |
| graph and practitioner tables | as Table 7 | best single chunk, in sample |
| packed-step and large-width tables | library defaults | best single chunk, in sample |
| deployment-bucket and serving tables | Triton 256 | -- |
| TFLA table | library default 128 | best single chunk and batch-keyed table, in sample |

**Proposal (one definition per kernel).**
- Default: SSD 256, Gated DeltaNet 64, TFLA 128 (library defaults, unchanged); reference CUDA scan: its stock length
  dispatch clip(nextpow2(L), 128, 2048), timed in the same runtime-chunk binary and session as the other chunks;
  Triton scan (ours): no library default exists, so the free length rule clip(nextpow2(L), 32, 1024), with chunk 256
  kept as a secondary row labelled "borrowed SSD default". For SSD and Gated DeltaNet the capped default is a zero-cost
  variant worth one extra row.
- Best static: one chunk per data set (and, as a separate row, one per batch size) with the lowest total latency on
  the training prompt folds, scored on the held-out fold, reported per call and LW. Rationale: total latency is what a
  deployed static chunk minimizes and the latency-weighted share is the paper's headline convention; held out removes
  the Table 5 label problem; the rule applies unchanged to every kernel and convention. Where a per-call table wants
  the per-call-optimal static chunk, label it "best static by per-call regret".

**Recomputed under the proposal (device time unless stated; per call / LW; ms per forward pass).**

| Table entry | RTX PRO 6000 | RTX 3090 | H800 |
|---|---|---|---|
| Table 5 / Sec. 6: default = free rule | 5.86 / 5.96; 0.78 | 7.76 / 16.31; 4.86 | 10.66 / 13.94; 3.37 |
| Table 5: one chunk, held out, total latency | 98.6 / 15.5 (512) | 36.9 / 19.8 (64/128) | 50.9 / 18.3 (256) |
| Table 5: per batch, held out, total latency | 98.6 / 15.5 (512 at every batch) | 17.4 / 2.66 (128; 32; 32) | 41.2 / 7.30 (256; 256; 128) |
| current Table 5 per batch (per-call criterion) | 20.0 / 27.0 | 17.4 / 2.66 | 24.7 / 20.8 |
| Table 8: free rule | 0.78 ms | 4.86 ms | 3.37 ms |
| Table 8: per batch, held out, total latency | 2.03 ms (= current) | 0.79 ms (= current) | 1.76 ms (= current) |

Table 4 (RTX PRO 6000, by batch 1/4/16): proposed per-batch chunk = 512 at every batch, per call 45.3 / 171.4 /
236.2%, LW 8.8 / 18.3 / 17.6% (exactly the current "LW" column); the current per-call-criterion chunks 256 / 64 / 64
give 18.8 / 20.5 / 24.6% per call but LW 17.8 / 38.2 / 27.2%; the free rule gives 5.8 / 4.8 / 7.6% and 12.6 / 3.7 / 3.7%.
Table 7, Triton rows: default = free rule 7.66 / 16.26%, 4.82 ms (RTX 3090) and 6.02 / 6.04%, 0.79 ms (RTX PRO 6000),
against the current best single chunk 31.8 / 17.9%, 5.31 ms and 46.0 / 32.3%, 4.20 ms. Larger checkpoints (Table 8):
free rule 11.0 ms (1.4B) and 22.0 ms (2.8B); per batch by total latency held out 2.91 ms (= current) and 4.51 ms
(current in sample: 4.04 ms). Reference scan (Table 6/8): stock rule in the same binary 5.35% (LW 7.99%, 1.75 ms per
pass) launch-inclusive, 3.09% (8.12%, 1.73 ms) in device time.

**Suggested text.** "Throughout, the default of a kernel is its library default (SSD 256, Gated DeltaNet 64, TFLA 128)
or its stock length dispatch (reference CUDA scan); our Triton scan has none, and we use the same length rule,
clip(nextpow2(L), 32, 1024). The best static chunk is the single chunk, or one chunk per batch size, with the least
total latency on the training prompts, scored on held-out prompts."

---

## M2d (offline part). The chunk effect at a fixed launch configuration

**Computed** from the stored 184-configuration sweep at the twelve keys (search timings: 3 warm-ups, median of 10
repeats, one pass in key order). Per GPU and convention: per fixed (BLOCK_D, num_warps, stages), the per-key best chunk
and its counts, the number of distinct optima, the spread over chunks, the best single chunk across keys and its
regret against the per-key best at that launch, and how far that launch lies above each key's overall optimum; the
same along the default table (with search and with full-protocol timings); the chunk curve when every chunk gets its
own best launch at each key; a launch table retuned on each GPU; Shapley splits. E34's analysis code was also applied
to the search timings (`e34_analysis_on_search_timings`) so that the full-protocol re-timing can be compared one to
one. Noise of the search timings at the 72 default entries against the full protocol: median |difference| 0.38 /
0.61 / 0.18%, 90th percentile 3.5 / 7.8 / 1.4%, maximum 10.2 / 15.8 / 5.9% (device time); launch-inclusive RTX 3090
up to 226%.

**Device time (RTX PRO 6000 / RTX 3090 / H800).**

| | RTX PRO 6000 | RTX 3090 | H800 |
|---|---|---|---|
| along the default table: best single chunk, regret | 256, 44.5% (full protocol 42.7%) | 64, 17.1% (full protocol 128, 15.1%) | 128, 20.7% (21.4%) |
| default table: distinct optima over 12 keys | 4 | 4 | 4 |
| fixed (8, 2, 1): optimum counts; single chunk regret; launch above key optimum | 64 x8, 128 x3, 512 x1; 16.1%; +55.6% | 32 x7, 128 x4, 64 x1; 16.8%; +12.1% | **128 x11**, 32 x1; **1.2%**; +43.4% |
| fixed (4, 2, 1) | 512 x9, 128 x2, 64 x1; 36.7%; +6.8% | **128 x11**, 64 x1; **0.03%**; +11.1% | 512 x4, 256 x3, 128 x3, 64 x2; 5.8%; +19.8% |
| fixed (4, 4, 1) | 1024 x5, 256 x4, 128 x3; 18.9%; +9.0% | 256 x6, 128 x4, 1024 x2; 7.0%; +20.5% | 256 x8, 128 x3, 1024 x1; 13.9%; +2.4% |
| all 32 fixed launches: single-chunk regret, median [range] | 14.5 [1.2, 37.0] | 4.6 [0.0, 21.9] | 4.2 [0.0, 19.4] |
| launches with single-chunk regret < 2% / < 5% | 4 / 8 of 32 | 8 / 18 | 12 / 16 |
| median distinct optima per fixed launch | 3 | 2 | 3 |
| every chunk at its own best launch per key: best single chunk, regret; optima | 128, 13.2%; 5 distinct | 128, 7.1%; 4 | 256, 16.0%; 3 |
| launch table retuned on this GPU: best single chunk, regret | 128, 13.2% | 128, 8.2% | 256, 14.4% |

The launches where one chunk wins everywhere are slow ones: RTX 3090 (4, 2) is on average 11.1% and H800 (8, 2) 43.4%
above the key's optimum. At the five fastest fixed launches the optimum still moves (e.g. RTX PRO 6000 (8, 8, 1): 128 at
9 keys, single chunk 6.3%; RTX 3090 (16, 8, 1): 2.6-3.0%; H800 (4, 4, 1): 13.9%). Launch-inclusive the medians are 9.7 /
8.3 / 3.9% and with a flushed L2 (RTX PRO 6000) 14.2%.

**Shapley split** (two players, chunk and launch; average of the two orders; mean over the 12 keys, % of the "to"
configuration, i.e. "x% slower than"; interaction = the chunk gain at the target launch minus that at the default
launch):

| Pair | RTX PRO 6000 | RTX 3090 | H800 |
|---|---|---|---|
| chunk 256 -> best default-table entry (the paper's split) | 44.5 = 50.6 chunk - 6.2 launch (interaction 23.8) | 43.3 = 36.4 + 6.9 (67.9) | 27.5 = 33.9 - 6.4 (34.7) |
| same, launch-inclusive | 32.7 = 29.3 + 3.4 (8.7) | 44.2 = 39.8 + 4.4 (59.0) | 18.4 = 18.8 - 0.4 (23.8) |
| chunk 256 -> overall optimum of the 184 configurations | 55.4 = 39.8 + 15.7 (10.7) | 57.6 = 71.2 - 13.6 (97.9) | 49.1 = 31.4 + 17.6 (10.2) |
| best default-table entry -> overall optimum | 8.5 = 5.8 + 2.7 | 11.8 = 14.0 - 2.2 | 18.8 = 14.7 + 4.0 |
| in % of the default time (E34 convention), 256 -> optimum | 32.2 = 20.6 + 11.5 | 32.4 = 43.7 - 11.3 | 29.9 = 16.0 + 13.9 |

The interaction is as large as the parts (RTX 3090 up to 98 points), so the split is order dependent: on the RTX 3090
changing the chunk first gains only 2.5 points at the default launch but 70 at the target launch. The split from the
free-rule chunk to the optimum is not meaningful (cross corners such as chunk 1024 at small channel blocks are
extremely slow; RTX 3090 mean parts +433 / -407) and should not be reported.

**Suggested text.**
- "The default launch table couples each chunk to a channel block and a warp count. Holding the launch configuration
  fixed, the chunk effect depends on the configuration: at the table's (4, 2) entry on the RTX 3090 and its (8, 2) entry
  on the H800 one chunk is fastest at 11 of the 12 keys (single-chunk regret 0.03 and 1.2%), but these configurations
  are 11 and 43% slower than each key's optimum; over all 32 fixed configurations a single chunk loses a median 14.5,
  4.6 and 4.2% (RTX PRO 6000, RTX 3090, H800), and with every chunk at its own best configuration the optimum still
  moves with the shape (single chunk 13.2, 7.1 and 16.0%). These are short search timings; the full-protocol re-timing
  is in the supplement."
- "A two-player Shapley split of the gap between chunk 256 and each key's best table entry gives, in device time,
  44.5% = 50.6 points from the chunk and -6.2 from the launch configuration on the RTX PRO 6000 (RTX 3090: 43.3 = 36.4 +
  6.9; H800: 27.5 = 33.9 - 6.4); the two orders differ by up to 68 points, so the attribution is order dependent and
  the chunk results are results for launch-table entries."

---

## M4a. One pre-specified a-priori protocol; all six reported

**Computed.** The 13 candidate forms were refitted per GPU with their leave-one-calibration-shape-out (LOSO) criteria
(the stored round-4 forms and selections are reproduced exactly: every selected form of the stored analysis matches).
For each target GPU, both candidate sets and the three criteria, the stored tie rule and an a-priori deterministic
tie-break (lowest LOSO held-out log-RMSE among exact ties); per-call and LW regret with intervals, p95, maximum, per
batch and aligned/unaligned, on the primary data set and on the own table; the full candidate table per fold
(`candidate_table`: each form's parameter count, three criterion values, held-out RMSE and target regret).

**Device time, per call % (LW %) on the primary data sets** (protocol order: ablation12 with LOGO-calls, LOGO-LOSO,
target-LOSO; then ablation12+restriction with the same three):

| GPU | six protocols, per call | six protocols, LW | selected forms |
|---|---|---|---|
| RTX PRO 6000 | 2.26, 0.40, 0.58, 4.20, 0.40, 0.58 | 14.1, 2.29, 2.32, 26.9, 2.29, 2.32 | drop_issue_sharing, add_l2_miss, drop_scan_work, work_only, add_l2_miss, drop_scan_work |
| RTX 3090 | 6.21, 2.46, 1.76, 6.21, 2.46, 0.85 | 10.3, 7.51, 2.57, 10.3, 7.51, 0.86 | drop_shuffle, drop_spill_l1_miss, drop_issue_sharing, drop_shuffle, drop_spill_l1_miss, work_only |
| H800 | 3.01, 3.01, 2.14, 3.01, 3.01, 2.14 | 5.33, 5.33, 3.14, 5.33, 5.33, 3.14 | add_l2_miss, add_l2_miss, drop_spill, add_l2_miss, add_l2_miss, drop_spill |

Own tables (the stored analysis): PRO 2.24, 0.38, 0.56, 4.16, 0.38, 0.56; RTX 3090 6.23, 2.49, 1.74, 6.23, 2.49, 0.88;
H800 as above. Ranges over the six protocols: per call 0.40-4.20 / 0.85-6.21 / 2.14-3.01%; LW 2.3-26.9 / 0.9-10.3 /
3.1-5.3%. Maxima of the paper's protocol: 53.2 / 30.9 / 50.2% (own tables 53.3 / 27.2 / 50.2). Exact ties: on the RTX
PRO 6000 target-LOSO ties drop_scan_work (0.58%), drop_shuffle (1.04%) and drop_spill_l1_miss (0.58%) at a criterion
of 0.756; LOGO-LOSO ties add_l2_miss and add_cross_warp (both 0.40%); the RMSE tie-break keeps drop_scan_work. Paper
protocol per batch (per call): RTX PRO 6000 0.19 / 0.59 / 27.8% (LW at batch 16: 47.7%); RTX 3090 1.50 / 16.2 / 13.2%;
H800 2.27 / 6.26 / 1.67%. Launch-inclusive the six give 0.88-4.48 / 2.49-4.52 / 3.54-7.25% per call.

**Pre-specified protocol (recommended).** Candidate set ablation12 + three-parameter restriction; criterion: the
target GPU's own calibration LOSO chunk regret; tie rule: fewest free parameters within 0.1 points of the lowest
criterion, then the lowest criterion, then the lowest LOSO held-out log-RMSE. It needs no captured call and no other
GPU, so a practitioner can run it on a new GPU, and it matches the "What transfers" recommendation. It selects
drop_scan_work (7 parameters), the three-parameter restriction, and drop_spill (6) and gives **0.58 [0.37, 0.84] /
0.85 [0.51, 1.22] / 2.14 [1.44, 2.97]% per call and 2.32 [1.0, 3.5] / 0.86 [0.6, 1.3] / 3.14 [2.5, 3.7]% LW**
(per batch 0.19/1.68/0.78, 0.84/1.41/0.10, 1.35/5.47/0.88). Caveat to state in the paper: this protocol was fixed in
this revision, after all six had been computed; all six are reported.

**Suggested text.** "We fix the term selection a priori as the form with the fewest parameters whose leave-one-shape-out
chunk regret on the target GPU's own twelve calibration keys is within 0.1 points of the best (ties: lower held-out
error); it needs no captured call. In device time it selects with 0.6, 0.9 and 2.1% mean per-call regret (2.3, 0.9
and 3.1% of the scan time) on the RTX PRO 6000, RTX 3090 and H800. Across the six selection protocols we evaluated, the
per-call regret ranges over 0.4-4.2, 0.9-6.2 and 2.1-3.0% and the latency-weighted share over 2.3-26.9, 0.9-10.3 and
3.1-5.3% (Supplementary Table S?)."

---

## M4b. Fair comparison of calibration-only selectors

**Computed.** Every calibration-only ranker of the stored analysis (refitted, the learned regressors with the stored
cross-validated hyperparameters, MLPs over seeds 0-4) plus the two a-priori forms, on the same (primary) data set per
GPU: top-1, top-2, top-3 (profile the k best-ranked chunks per exact shape), per call, LW, per batch, intervals; GPU
mean and worst; paired differences.

**Device time, top-1 per call % on RTX PRO 6000 / RTX 3090 / H800 (GPU mean, worst) and LW %.**

| Selector (parameters) | per call | mean, worst | LW | top-2 per call | top-3 per call |
|---|---|---|---|---|---|
| Eq. (3), frozen form (8) | 0.40 / 3.52 / 3.01 | 2.31, 3.52 | 2.29 / 9.73 / 5.33 | 0.23 / 0.72 / 0.17 | 0.04 / 0.55 / 0.17 |
| Eq. (3), a priori, paper's protocol | 4.20 / 6.21 / 3.01 | 4.47, 6.21 | 26.9 / 10.3 / 5.33 | 1.89 / 3.02 / 0.17 | 0.05 / 2.01 / 0.17 |
| **Eq. (3), a priori, pre-specified** | **0.58 / 0.85 / 2.14** | **1.19, 2.14** | **2.32 / 0.86 / 3.14** | 0.32 / 0.15 / 1.16 | 0.04 / 0.03 / 1.16 |
| three-parameter restriction (3) | 4.20 / 0.85 / 3.53 | 2.86, 4.20 | 26.9 / 0.86 / 11.2 | 1.89 / 0.15 / 1.16 | 0.05 / 0.03 / 0.17 |
| MLP, five seeds, mean (s.d.) | 1.61 (1.18) / 1.31 (1.05) / 7.13 (4.55) | 3.35, 7.13 | 2.31 / 1.44 / 6.45 | 0.24 / 0.29 / 3.84 | 0.04 / 0.13 / 0.24 |
| MLP, seed range | 0.28-2.65 / 0.49-3.06 / 0.44-13.2 | | | | |
| MLP structural, five seeds | 3.77 (0.36) / 4.90 (0.60) / 1.11 (0.47) | 3.26, 4.90 | 3.24 / 6.67 / 2.07 | 0.42 / 3.17 / 0.23 | 0.04 / 0.64 / 0.17 |
| gradient boosting, structural | 3.20 / 2.52 / 7.42 | 4.38, 7.42 | 1.68 / 2.87 / 10.1 | 0.62 / 0.88 / 4.06 | 0.15 / 0.57 / 0.95 |
| gradient boosting, basic | 6.36 / 2.91 / 16.7 | 8.67, 16.7 | 4.72 / 3.03 / 10.2 | | |
| work-term model | 2.11 / 21.5 / 4.16 | 9.24, 21.5 | 2.78 / 17.0 / 13.8 | | |
| log-linear interpolation | 8.90 / 16.9 / 23.4 | 16.4, 23.4 | 2.96 / 1.98 / 8.80 | | 0.45 / 0.61 / 4.19 |
| nearest calibration key | 14.1 / 16.9 / 28.9 | 19.9, 28.9 | 3.48 / 2.02 / 10.1 | | |
| length tree (calibration) | 16.6 / 20.4 / 38.4 | 25.1, 38.4 | 3.42 / 15.0 / 16.8 | | |
| compile-facts rule (0) | 29.1 / 21.5 / 29.5 | 26.7, 29.5 | 41.0 / 10.1 / 21.8 | 22.0 / 4.75 / 17.5 | 17.2 / 4.04 / 8.18 |
| TFLA expression (sanity check) | 13.3 / 6.85 / 12.9 | 11.0, 13.3 | 13.5 / 13.0 / 14.3 | | |
| (free length rule, no calibration) | 5.86 / 7.76 / 10.66 | 8.09, 10.66 | 5.96 / 16.3 / 13.9 | | |

Paired (frozen full form minus restriction, points): RTX PRO 6000 -3.80 [-4.78, -2.89], RTX 3090 +2.66 [+2.12, +3.17],
H800 -0.53 [-1.35, +0.60]; pre-specified minus restriction: -3.62 [-4.63, -2.65], 0, -1.40 [-1.77, -1.03]. Frozen
minus MLP (seed 0): +0.04 [-0.11, 0.18], +3.03 [+2.35, +3.69], -4.98 [-7.82, -2.58]. The H800 nine-parameter
variant (add_l2_miss) makes identical choices to the frozen form (3.0077%). Under the paper's protocol the
eight-parameter model has the worst GPU mean of the four main competitors (4.47 against 2.86, 3.35 and 3.26); under the
pre-specified protocol it has the best (1.19, worst 2.14). The zero-parameter compile-facts rule loses to the free
length rule on every GPU.

**Suggested text.** "With every calibration-only selector evaluated on the same calls (Table 5), the pre-specified
form of Eq. (3) has the lowest mean over the three GPUs (1.2% per call, at most 2.1%), ahead of the three-parameter
restriction (2.9%, at most 4.2%), an MLP over five seeds (3.4%; 1.6, 1.3 and 7.1% with seed-to-seed standard
deviations of 1.2, 1.1 and 4.6 points) and the structural MLP (3.3%). With the terms chosen on the other GPUs' calls
instead, the same model is the worst of the four (4.5%), and on the H800, where the form was chosen blind, it is not
better than the restriction (-0.5 points [-1.4, +0.6]). A rule that uses the compiler's resource report alone, without
fitted parameters, loses 21-30% per call."

---

## M4c. Fit documentation (and N15)

**Computed.** Per GPU: refits on the 90 full-protocol timings only, on the 72 static re-measurements only, on all 162
timings of the six default-table configurations, without the first-key search timings, inverse-variance weighted,
OLS, and the best of 50 random starts; for each the R^2 of log latency and of the minimized objective, and the per-call
and LW regret on the primary and own tables; 50 random single starts for each small refit; the 12-key bootstrap,
leave-one-key-out, search-versus-full-protocol discrepancy, timing sources and sweep metadata.

**Device time (per call % / LW % on the own table; RTX PRO 6000 / RTX 3090 / H800).**

| Fit | points | RTX PRO 6000 | RTX 3090 | H800 | R^2 objective |
|---|---|---|---|---|---|
| stored (soft-l1, 2298 timings) | 2208 search + 72 static + 18 E1 | 0.38 / 2.26 | 3.55 / 9.81 | 3.01 / 5.33 | 0.895 / 0.926 / 0.938 |
| full-protocol timings only | 90 | 0.73 / 2.48 | **1.55 / 2.68** (taco-r1: 4.28) | 6.05 / 8.88 | 0.959 / 0.971 / 0.966 |
| 72 static re-measurements only | 72 | 0.56 / 2.30 | **1.55 / 2.68** (taco-r1: 1.76) | 6.05 / 8.88 | 0.963 / 0.970 / 0.971 |
| six default-table configurations, all sources | 162 | 0.56 / 2.30 | 1.55 / 2.68 | 6.05 / 8.88 | 0.961 / 0.972 / 0.969 |
| without first-key search timings | 2114 | 0.38 / 2.26 | 1.74 / 2.66 | 3.01 / 5.33 | |
| inverse-variance weighted | 2298 | 0.38 (unchanged) | 3.55 (unchanged) | 3.01 (unchanged) | 0.897 / 0.938 / 0.938 |
| OLS | 2298 | 2.09 / 13.9 | **1.74** / 2.66 | 3.01 / 5.33 | 0.973 / 0.990 / 0.989 |
| best of 50 random starts | 2298 | 0.38 | 1.74 / 2.66 (objective -15.9%) | 3.01 | |

Other facts: R^2 of log latency of the stored fit 0.963 / 0.986 / 0.989; 12-key bootstrap of the mean regret
[0.38, 2.26] / [1.74, 6.09] / [2.41, 3.04] (the RTX PRO 6000 point estimate is the lower edge; the RTX 3090 median
replicate is 1.74); leave-one-key-out chunk regret 0.76 / 7.43 / 3.67% (max 4.8 / 23.6 / 13.8%); search vs full
protocol at the 72 default entries: median |difference| 0.38 / 0.61 / 0.18%, largest 10.2 / 15.8 / 5.9% in device
time, 225.8% launch-inclusive on the RTX 3090; timing sources 2208 search (3/10) + 72 static (10/100) + 18 E1 medians
on every GPU; the sweep runs key by key (L128 b1, b4, b16, L512 ...) with one fixed configuration order and no shuffle
or seed argument; the exclusive-GPU guard was on only for the RTX PRO 6000 device-time sweep (off for the RTX 3090 and
H800 device-time sweeps). Launch-inclusive, the inverse-variance weighting is not neutral: it moves the RTX 3090 fit from
3.25 to 7.40% (as does dropping the first-key search timings).

**Discrepancy (the two failing checks).** On the RTX 3090 the 90-point full-protocol refit has two optima within
0.5% of each other in objective: the taco-r1 run reached cost 0.32578 (4.28%), the re-run with the same code and data
(SciPy 1.17.1, NumPy 2.4.4) reaches 0.32410 (1.55%), mainly by moving tau_s from 4.4e-6 to 5e-8; the 72-point refit
behaves the same way (1.76 vs 1.55%). None of 50 random single starts improved on the three-restart result for any
small refit (best random start 1.53% on the RTX 3090). The full-protocol refit is therefore numerically fragile on the
RTX 3090; it should be quoted as 1.6% (or 1.6-4.3% across optima) with this remark, not as "the H800 doubles while the
RTX 3090 rises to 4.3%".

**N15.** The OLS regret on the RTX 3090 is 1.7447% and rounds to 1.7%, not 1.8% (appendix.tex:847); the other two are
2.09 (2.1) and 3.01 (3.0). The appendix's "1.7% instead of 3.6%" for the 50-start search is right (3.554 rounds to 3.6).

**Suggested text.** "Fitted on the 90 full-protocol timings alone, the model selects with 0.7, 1.6 and 6.1% mean
regret (on the RTX 3090 a second optimum within 0.5% of the objective gives 4.3%); weighting the timing sources by
inverse variance leaves the device-time fits unchanged. The fit explains 89.5, 92.6 and 93.8% of the robust objective
(R^2 against the best constant) and 96-99% of the variance of log latency; 2208 of the 2298 calibration timings are
short search timings (median of 10 repeats), which differ from full-protocol timings of the same configuration by a
median of 0.2-0.6% and at most 16% in device time. With twelve calibration keys the honest uncertainty is the
leave-one-key-out regret, 0.8, 7.4 and 3.7%."

---

## M4d. Physical consistency of Eq. (3)

**Computed.** With the stored fits, how often the latency term is hidden (W > W1) and how often the predicted time per
tile falls below the serial latency A; refits (same data, loss, bounds, three-restart protocol, warm-started from the
stored fit as well) of Eq. (3) floored at A, of max(A + I, I W / W0), of max(A, I max(1, W / W0)), and of A + I max(1,
W / W0) (no hiding); fit quality, LOSO chunk regret, per-call and LW regret with intervals, share of calls whose chunk
changes; programs per SM at batch 1.

**Critical path.** In device time the stored fits never predict a tile time below A for any captured-call candidate
(minimum ratio of tile time to A 1.41, 1.62 and 1.16), although the latency term is hidden for 12, 30 and 67% of them
(W1 = 10.5, 8.0 and 6.6 warps), because the issue term keeps the sum above A; on the calibration timings only the H800
has 1.7% below A (minimum 0.83). Launch-inclusive, the RTX 3090 fit (W1 = 4.4) predicts below the critical path for
16% of the call candidates (minimum 0.81) and the H800 for 3% (0.95). Programs per SM at batch 1: 1.36 / 2.72 (RTX PRO
6000, channel blocks 8 / 4), 3.12 / 6.24 (RTX 3090), 2.25 / 4.49 (H800); the "1.4-2.7" of the review holds for the RTX
PRO 6000 only.

**Device time, per call % [CI] / LW % on the primary data sets (parameters; R^2 objective; LOSO %; calls with another
chunk).**

| Form | RTX PRO 6000 | RTX 3090 | H800 |
|---|---|---|---|
| Eq. (3) (8) | 0.40 [0.22, 0.64] / 2.29; 0.895; 0.76 | 3.52 [2.85, 4.15] / 9.73; 0.926; 7.43 | 3.01 [1.90, 4.36] / 5.33; 0.938; 3.67 |
| Eq. (3) floored at A (8) | identical (0% of calls change) | identical | identical choices; objective 0.939 |
| max(A + I, I W/W0) (7) | 2.11 [1.47, 2.81] / 14.0; 0.893; 6.08; 7% | 4.30 [3.16, 5.43] / 5.19; 0.929; 8.89; 24% | 2.25 [1.51, 3.11] / 4.36; 0.939; 3.67; 2% |
| max(A, I max(1, W/W0)) (7) | 10.4 / 29.8; 0.888; 22.9; 45% | 6.25 / 12.4; 0.921; 11.3; 39% | 2.41 / 3.16; 0.930; 3.67; 4% |
| A + I max(1, W/W0), no hiding (7) | 1.06 / 2.59; 0.888; 0.76; 4% | 6.21 / 10.3; 0.922; 12.5; 22% | 2.41 / 3.16; 0.925; 4.80; 4% |

Paired (max(A + I, I W/W0) minus Eq. (3)): +1.71 [1.18, 2.25], +0.78 [0.06, 1.45], -0.76 [-1.61, -0.14] points per call.

**Suggested text.** "The latency term of Eq. (3) can in principle predict a time per tile below the serial latency
of the state loop; with the fitted parameters it never does in device time (the predicted time per tile is at least
1.16 times that latency for every candidate), and flooring it at the critical path changes no selection. A
max(latency, throughput) form with one parameter fewer fits as well (R^2 of the objective 0.893, 0.929 and 0.939
against 0.895, 0.926 and 0.938) but selects worse on the RTX PRO 6000 and the RTX 3090 (2.1 and 4.3% against 0.4 and
3.5% per call) and better on the H800 (2.3 against 3.0%), so the data do not distinguish the two forms."

---

## M7b (offline part). Where 32-bit indexing faults

**Computed.** Every per-call table (38 files) and the synthetic sweeps were scanned for chunks recorded as
unavailable; the 2^31 rule was applied independently and compared cell by cell; per kernel instance the largest batch x
length before overflow.

**Recorded cells (all exactly predicted by the rule).**
- SSD (mamba_ssm 2.3.2, Mamba-2 2.7B width: d_inner 5120 = 80 heads x 64, state 128): chunk 32 at batch 16, L = 8192:
  24 of the 2360 captured calls in each of the five SSD 2.7B tables (RTX 3090 device time, graph, graph three rounds,
  graph cold; RTX PRO 6000 graph three rounds), state tensor 2.68e9 elements; the largest available length at batch 16
  and chunk 32 among the captured calls is 4096. Synthetic sweep: d = 5120, L = 8192, batch 16, chunk 32 in all three
  rounds. d = 4096 at the same shape is exactly 2^31 elements and ran, so the limit is "more than 2^31".
- TFLA mLSTM (state batch x heads x (L/C + 1) x 256 x 512): 9 cells, chunk 32 at seven and chunk 64 at two shapes, all at
  batch 16: 1.4B and 2.8B layouts at L = 4096, 6144, 8192 (chunk 32) and L = 8192 (chunk 64), 370M layout at L = 8192
  (chunk 32); identical in the device-time and launch-inclusive sweeps.
- No unavailable cell in any Mamba-1, Mamba-2 370M or Gated DeltaNet table.

**Candidate-set rule (largest length at batch 1 / 4 / 16 before the state tensor exceeds 2^31 elements).** SSD 2.7B:
chunk 32: 104,832 / 26,208 / 6,528; chunk 64: 209,664 / 52,416 / 13,056; chunk 128: twice that. SSD 370M (32 heads):
chunk 32: 262,144 / 65,536 / 16,384. In tokens per call (batch x L), SSD 2.7B allows at most 3276 x C tokens (104,832
at chunk 32), SSD 370M 8192 x C. TFLA 2.8B layout: chunk 32: 52,384 / 13,056 / 3,232; chunk 64: 104,768 / 26,112 / 6,464.

**Suggested text.** "Small chunks overflow the 32-bit indexing of the chunk-state tensor: the SSD kernel faults with an
illegal memory access when batch x ceil(L/C) x heads x head dimension x state exceeds 2^31 elements (at the 2.7B width
with chunk 32: beyond 104,832 tokens per call, e.g. batch 16 at L > 6528, which excluded chunk 32 for 24 of our 2360
calls), and the TFLA kernel at 2^31 elements of its saved chunk states (chunk 32 at seven and chunk 64 at two of our
63 shapes). Because the fault poisons the CUDA context, a warm-up profiler must drop such chunks from the candidate set
for the largest batch and length a key admits."

---

## N17. Population and fallback of the held-out exact-shape table (Triton scan, device time)

**Computed** from the same data sets as Table 5. The table is evaluated on **all 2360 calls**; for each prompt fold it
is built on the other four folds. A call is "covered" when its exact key (length, batch, width, alignment) has
training calls: 2156 calls (91.4%) on every GPU. The **204** others fall back to the (length bucket, batch, alignment)
entry (195 calls) or, if that is empty too, to the single chunk with the lowest per-call regret on the training folds
(9 calls); entries use the lowest mean per-call regret.

| | RTX PRO 6000 | RTX 3090 | H800 |
|---|---|---|---|
| Table 5 value, all calls (fallback bucket + alignment, then global) | 0.138% (0.37 us) | 0.350% (14.6 us) | 0.200% (0.34 us) |
| covered calls (2156) | 0.046% (0.14 us) | 0.030% (0.05 us) | 0.219% (0.37 us) |
| fallback calls (204) | 1.105% (2.79 us) | 3.724% (168 us) | 0.000% |
| plain length-bucket fallback (no alignment bit), all calls | **0.224%** (2.14 us) | 0.217% (4.0 us) | 0.599% (10.5 us) |
| global-chunk fallback only, all calls | 0.381% | 3.990% | 0.200% |
| entries by mean latency (deployment-table estimator), all calls | 0.176% | 0.337% | 0.200% |

The "0.14 us per call" of main.tex:814-815 is the covered-call headroom; over all calls it is 0.37 us on the RTX PRO
6000. On the RTX 3090 the alignment-keyed fallback is costly (168 us per call on the 204 fallback calls, against 46 us
with the plain bucket fallback), so the RTX 3090 all-call value depends strongly on the fallback rule.

**Suggested text.** "Over all 2360 calls, with calls whose exact shape has no training entry (204) served by the
(length bucket, batch, alignment) entry and otherwise by the best single chunk, the held-out exact-shape table is within
0.14, 0.35 and 0.20% of the per-call oracle (0.37, 14.6 and 0.34 us per call); on the 2156 calls with an entry it is
within 0.05, 0.03 and 0.22% (0.14, 0.05 and 0.37 us)."

---

## N18. Reference-scan stock dispatch and the shape tree on one basis (RTX 3090, 2360 calls)

| Basis | launch-inclusive | device time |
|---|---|---|
| stock pairing in the runtime-chunk build (same binary, same blocks as the oracle) | **5.35% [3.92, 7.25]**, LW 7.99%, 36.5 us | **3.09% [1.47, 5.08]**, LW 8.12%, 36.1 us |
| rebuilt module's own stock dispatch, timed as a separate call | 6.37%, LW 7.97% | 3.12%, LW 8.08% |
| installed stock binary, timed in its own block (Table 6) | 6.58%, LW 7.98% | 3.56%, LW 8.09% |
| ratio installed binary / runtime build at the stock chunk: p10, median, p90 (min, max) | 0.864, 1.002, 1.134 (0.26, 4.30) | 0.981, 1.001, 1.038 (0.87, 1.18) |
| ratio rebuilt separate call / runtime build at the stock chunk: p10, p90 | 0.874, 1.105 | 0.982, 1.017 |
| shape tree, held out | **3.00% [2.44, 3.61]**, LW 1.17% | **0.60% [0.31, 0.90]**, LW 0.81% |
| shape tree minus stock pairing (paired, points) | -2.35 [-4.37, -0.93] | -2.49 [-4.33, -1.03] |
| reduction by the shape tree | 44% (vs 6.58: 55%) | 81% |
| length-only tree; exact-shape table | 3.55%; 2.91% | 1.55%; 0.73% |

The gap between 6.6 and 5.35% is block-to-block timing noise: the same binary's stock dispatch timed as a separate
call already gives 6.37%, and the median cross-binary ratio is 1.00. "A shape tree halves the stock rule" is supported
only against the separately timed binary (55%); on the same basis it is a 44% reduction launch-inclusive and 81% in
device time. Recommended support: the device-time, same-binary pair, 3.1% to 0.6% per call (8.1 to 0.8% of the scan
time).

**Suggested text.** "Timed in the same binary and blocks, the reference scan's stock length dispatch loses 3.1% per
call in device time (8.1% of the scan time) and 5.4% launch-inclusive; a shape tree chosen on held-out prompts reduces
this to 0.6% (0.8%) and 3.0%, respectively."

---

## N23. The zero-parameter compile-facts rule (definition from the code)

Candidates: the six chunks of the default launch table, each at its default (BLOCK_D, num_warps) and one stage. For
each candidate the kernel is compiled, not run, for the target architecture and for the call's alignment
specialization (L mod 16 = 0 or not; aligned facts if the unaligned variant is missing); the compiler's resource
report gives registers per thread and spill store/load bytes. From these and the SM limits the rule computes the
resident programs per SM k = max(1, min(floor(65536 / (ceil(regs/8) x 8 x 32 x num_warps)), floor(max warps per SM /
num_warps), max blocks per SM)), the programs P = ceil(d / BLOCK_D) x B and the waves w = ceil(P / (n_SM x k)). The
score is S(C) = 10^9 x [spill bytes > 0] + w x ceil(L / C) - 0.001 x log2 C, and the rule serves the chunk with the lowest
score: any non-spilling chunk before any spilling one, then the fewest sequential wave-tiles, ties to the larger chunk.
No timing and no fitted parameter enter; the hardware inputs are the SM count and the CUDA occupancy limits. An
independent re-implementation from the raw compile facts agrees on 100% of the calls on all three GPUs. Spilling
default entries: RTX PRO 6000 512 and 1024 (aligned), 256-1024 (unaligned); RTX 3090 1024 (aligned), 128, 512 and
1024 (unaligned); H800 512 and 1024 (aligned), 256 (unaligned). It selects mostly 256 (RTX PRO 6000, H800) or 512 (RTX
3090) and loses 29.1, 21.5 and 29.5% per call (top-3: 17.2, 4.0 and 8.2%).

**Suggested text.** "The compile-facts rule uses no parameter: among the default-table chunks whose compiled kernel
reports no spilling, it serves the one with the fewest waves times time tiles, ceil(n_p / (n_SM r)) ceil(L / C), with r
the resident programs per SM implied by registers, warps and block limits; ties go to the larger chunk."

---

## Open points for the author

1. M4a: the pre-specified protocol is chosen after all six results were seen; the paper must say so (or adopt the
   stored paper protocol and report the range). The two give very different headlines (1.2% vs 4.5% GPU mean).
2. M4c: on the RTX 3090 the full-protocol and 72-point refits are fragile (1.55% here vs 4.28% / 1.76% in taco-r1);
   decide whether to quote a range or the better optimum.
3. M2c: adopting "best static by total latency" changes Table 5's per-batch row from 20.0 / 17.4 / 24.7% to 98.6 /
   17.4 / 41.2% per call (15.5 / 2.7 / 7.3% LW); keep the per-call variant as a labelled extra row if wanted.
4. M2d: the offline numbers are from short search timings; they should be replaced or confirmed by the full-protocol
   re-timing (E34) when it lands, using the identical analysis code.
