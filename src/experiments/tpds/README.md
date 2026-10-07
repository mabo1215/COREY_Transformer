# TPDS experiments (cycles tpds-r1 to tpds-r4)

Design: `docs/Design.md`, one section per cycle (tpds-r1: E1-E7; tpds-r2: offline packages T1-T4
and E8-E11; tpds-r3: E12-E17; tpds-r4: E18-E21 and the offline re-analyses). Review items:
`docs/revision_suggestions.tex`. Ledger with status, host and result files per experiment ID:
`docs/state/experiments/<ID>.json` (the cost model T1 is ledger ID `CM1`).

All timing uses recurrence-preserving kernels: one call processes the whole sequence and carries
the recurrent state across chunks; the chunk is a real tiling parameter of that call, and every
kernel and chunk is checked against a reference. The old chunk-splitting harnesses under
`src/experiments/run_w1_*`, `run_calibrated_chunk512_3090.py` and
`run_cuda_profile_three_policies.py` are not used.

**Paper locations** are given as LaTeX labels, not numbers, because numbers move. A plain label
(`tab:predictors`) is in `paper/main.tex`; a label with the prefix `S-` (`S-tab:supp_l2`) is in
`paper/appendix.tex` (the supplement; `main.tex` loads its labels through `xr` with that prefix
and cites them with `\Sref`). "Not in the manuscript" means that no table, figure or number of
the current `paper/` sources is taken from the file.

## Hosts

| Tag | GPU | Software notes |
|---|---|---|
| `pro6000` | NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB, 188 SMs, sm_120 | main kernel-level host; mamba_ssm 2.2.2 local build; no reference CUDA selective scan for sm_120 |
| `vgpu3090` | NVIDIA GeForce RTX 3090 on a 48 GB board, 82 SMs, sm_86 | replication, reference CUDA scan (stock and runtime chunk), routed study, larger checkpoints, tensor-parallel widths, E17; mamba_ssm 2.3.2.post1 wheel with its compiled kernels |
| `h800` | NVIDIA H800 PCIe, 80 GB, 114 SMs, sm_90 | data-centre GPU (E13, E13b) |
| (local) | RTX 3070, WSL2 | smoke tests only (`results/tpds/smoke/`) |

Common stack on the three timing hosts: torch 2.8.0 (cu128), Triton 3.4.0, transformers 5.6.2;
FLA 0.5.2 on the RTX PRO 6000 and the RTX 3090 (`S-tab:supp_platforms`).

## Kernels

| Family | Implementation | Chunks |
|---|---|---|
| Mamba-1, Triton | `kernels.mamba1_scan`, written for this study: the chunk is the time tile `BLOCK_T`; tunable `BLOCK_D`, `num_warps`, `num_stages`; default launch per chunk in `kernels.DEFAULT_MAMBA1_LAUNCH` | 32..1024 |
| Mamba-2 SSD | `mamba_chunk_scan_combined` of mamba_ssm (Triton), `kernels.mamba2_ssd` | 32..512 |
| Gated DeltaNet | `chunk_gated_delta_rule` of FLA. `kernels.gdn_chunk`: plain entry (pre-normalized k, log-gate g, sigmoid beta; E1, E17c). `kernels.gdn_chunk_prod`: the entry the model layer uses, which normalizes and gates inside the kernel (E17b) | 16, 32, 64 (larger chunks are rejected) |
| Mamba-1 reference CUDA scan, stock | `selective_scan_cuda` of a mamba_ssm build with compiled kernels (on `vgpu3090` the stock 2.3.2.post1 wheel) | fixed, chosen from L by the stock dispatch |
| Mamba-1 reference CUDA scan, runtime chunk | module `selective_scan_chunk_cuda`, built from `src/kernels/mamba1_runtime_chunk` (see its README): the unchanged kernel, each chunk dispatched to the stock thread configuration of that length; `fwd_with_chunk_size` plus a same-binary stock `fwd` | 128..2048 |

The runtime-chunk module is built on the GPU host by `tpds_r2_prep.sh` (sm_86 by default;
`tpds_r3_e13_prep.sh` / `tpds_lane_r3_e13.sh` build it for sm_90, `tpds_r3_e16_prep.sh` attempts
sm_120). It is used by E8, E10, E12 (`--ref`) and E9 (`--ref`); `e8_refscan --validate-only` checks
every chunk against the stock forward.

## Scripts in this directory

| File | Purpose |
|---|---|
| `kernels.py` | kernel wrappers above, fp32 references, validation; `python -m src.experiments.tpds.kernels` runs the self-test |
| `common.py` | environment metadata, resumable JSONL/CSV writers (flush + fsync, skip completed keys), disk guard, CUDA-event timing (`cuda_time`: optional L2 flush, optional GPU pre-spin), Eq. (2) |
| `corpus.py` | mixed-regime corpus (extracted from `run_mixed_regime_discovery.py`) plus LongBench prompts (`src/data/longbench_push120`) |
| `capture_traces.py` | per-layer scan inputs during prefill of Mamba-1 (`state-spaces/mamba-370m-hf`, `-1.4b-hf`, `-2.8b-hf`) and Mamba-2 (`AntonV/mamba2-370m-hf`, `AntonV/mamba2-2.7b-hf`); `--check-only` verifies a capture without loading the model |
| `capture_gdn_traces.py` | Gated DeltaNet kernel inputs of `Idiap/gated-deltanet-swa-1.4B-30B` (E17b) |
| `e1_value_invariance.py`, `e1_analysis.py`, `e1_from_log.py` | E1/E1b/E1c timing, analysis (ANOVA with BH, TOST = T4, redraw null, clock), rebuild of E1c rows from the run log |
| `e2_oracle_sweep.py` | per-call sweep of captured calls at every chunk (E2, E11, E13-E15, E17a/b, E18-E21) |
| `e3_predictors.py` | E3 predictors of the per-call oracle (CPU) |
| `e4_autotune.py` | E4 launch-space auto-tuning, static re-measurement, decision overhead, host synchronization |
| `e5_breakeven.py`, `e5b_key_churn.py` | E5 break-even; T3 key-churn break-even (CPU) |
| `e7_reference_scan.py` | E7 Triton vs stock reference CUDA scan |
| `e8_refscan.py` | E8 runtime-chunk reference scan on the captured calls (`--analyze` for the CPU analysis) |
| `e9_placement.py` | E9 / E16 value-versus-placement control, device-side reduction cost |
| `e10_routed_interleaved.py` | E10 single-session routed end-to-end study |
| `e12_tp_widths.py`, `e12_analysis.py` | E12 tensor-parallel widths and its analysis |
| `e16_counters.py`, `e16c_l2_placement.py` | E16b Nsight Compute counters (not run), E16c warm/cold L2 placement test |
| `e17_workbound.py`, `e17_analysis.py` | E17c synthetic sweep of the production kernels; E17 analysis |
| `cost_model.py` | T1 cost model: fit, validation V1-V4, ablation, reference-scan variant, transfer to other checkpoints |
| `kernel_stats.py` | compile facts (registers, spills, tile layout) per launch configuration and architecture |
| `cost_model_baselines.py`, `cost_model_baselines_r4.py` | simpler selectors against the cost model (tpds-r3); clean form selection, calibration-only and learned baselines, TFLA, top-k (tpds-r4) |
| `e3b_information.py` | tpds-r4 bootstrap, permutation and conditional-information tests |
| `entropy_band.py` | entropy band of real scan inputs and the chunks Eq. (2) maps it to |
| `headroom_noise.py`, `headroom_r4.py` | oracle headroom under timing noise; tpds-r4 headroom rows and N1 checks |
| `serving_key_churn.py`, `deployment_units.py` | serving-trace replay of the key-churn break-even; regret in ms per forward pass / request |
| `analysis.py` | shared regret, mutual-information, held-out headroom and round-median helpers |
| `make_figures.py`, `make_tables.py` | paper figures (`tpds_*.pdf`) and LaTeX table bodies from the result files |

Remote lanes and prep scripts: `src/experiments/remote/` (section "Lanes").

## Install (repeat on every host)

For the Triton kernels only the pure-Python/Triton parts of the libraries are needed. `--no-deps`
keeps the host's torch/triton untouched.

```bash
PY=python3                       # or the host's interpreter
# If pip refuses (PEP 668 "externally managed"), either add --user --break-system-packages,
# or use a venv that sees the system torch:
#   python3 -m venv --system-site-packages ~/tpds_venv && PY=~/tpds_venv/bin/python
$PY -m pip install --no-deps flash-linear-attention==0.5.2 fla-core==0.5.2 "einops>=0.8"
$PY -m pip install -U "typing_extensions>=4.12"   # torch._inductor (imported by fla) fails with 4.10
MAMBA_SKIP_CUDA_BUILD=TRUE CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE MAMBA_FORCE_BUILD=FALSE \
  $PY -m pip install --no-deps --no-build-isolation mamba-ssm==2.3.2.post1
$PY -c "from huggingface_hub import snapshot_download as s; [print(s(r)) for r in ('state-spaces/mamba-370m-hf','AntonV/mamba2-370m-hf')]"
```

`mamba_ssm/__init__.py` imports the compiled `selective_scan_cuda`, which does not exist with
`MAMBA_SKIP_CUDA_BUILD=TRUE`. `kernels._load_ssd()` therefore registers a namespace stub for the
top-level `mamba_ssm` package and imports only `mamba_ssm.ops.triton.ssd_combined` (pure Triton).
A full install with the CUDA extension is used unchanged.

The reference CUDA selective scan is **not** part of this Triton-only install, but it is used in
the study: E7, E8 and E10 need a mamba_ssm build with its compiled `selective_scan_cuda` (on the
RTX 3090 host the stock 2.3.2.post1 wheel, which ships sm_86 kernels), and E8, E10 and the `--ref`
options of E9/E12 additionally need the runtime-chunk module `selective_scan_chunk_cuda` from
`src/kernels/mamba1_runtime_chunk` (build: `tpds_r2_prep.sh`, or the commands in that directory's
README). No sm_120 build of the reference scan was used, so all reference-scan results come from
the RTX 3090 (and the E12 rows on the H800).

Further environments: `tpds_r3_e13_prep.sh` builds a fresh venv with the pinned versions above
(data-centre host); `tpds_r3_e17_prep.sh` checks FLA and downloads the E17 checkpoints.

Local smoke host (WSL2, RTX 3070): system python 3.12 is PEP-668 managed, so a venv with
`--system-site-packages` was used; the commands above were run with that venv's python.
`tpds_prep.sh` performs the same steps (`PIP_EXTRA` passes extra pip flags) plus the checkpoint
download and static checks.

## Timing conventions and result-file tags

Result files are `src/experiments/results/tpds/<experiment>_[<model>_]<tag>.<ext>`, with
`<model>` in `mamba1`, `mamba2`, `gdn` for per-call tables. The tag is the host, optionally
followed by a checkpoint and a convention suffix.

| Tag / suffix | Meaning |
|---|---|
| `<host>` (`pro6000`, `vgpu3090`, `h800`) | **launch-inclusive**: repeats issued back to back from Python, CUDA events around each call, no pre-spin; a call shorter than its host launch cost includes the wait for its launch (the cost in eager execution). Warm L2 (no flush), one round. Protocol of E1-E13 |
| `<host>_r3spin` | **device time**: a 20 ms GPU-side spin (`--pre-spin-ms 20`, or `TPDS_PRE_SPIN_MS=20` for E1/E4) before every timing block lets the host queue all repeats, so only GPU execution is timed (approximates a CUDA graph). E13b (`h800_r3spin`), E15 / E15b (`vgpu3090_r3spin`) |
| `<host>_1p4b_r3`, `<host>_2p8b_r3` | Mamba-1 1.4B / 2.8B checkpoints, batch 1/4/16, **device time** (E14; `_r3` here means device time) |
| `<host>_1p4b_r3plain`, `<host>_2p8b_r3plain` | the same calls, **launch-inclusive** (E15) |
| `<host>_1p4b`, `<host>_2p8b` | E11 batch-1 sweep of the same checkpoints, launch-inclusive, timed under host load; **superseded** by E14/E15 |
| `<host>_2p7b_r3` (`mamba2`), `<host>_1p4b_r3` (`gdn`) | Mamba-2 2.7B and Gated DeltaNet 1.4B captured calls, device time (E17a, E17b) |
| `<host>_r4spin` | device time, **three rounds** with the chunk order shuffled per call and per round (E18, E19; E1/E4 calibration of E18) |
| `<host>_r4plain` | launch-inclusive, three shuffled rounds (E18, E19) |
| `<host>_r4cold` | **cold L2** (a 128 MiB buffer overwritten before every timed repeat, outside the timed interval, as Triton's `do_bench`) plus device time, three rounds (E20) |
| `<host>_r4spin_u2048b4`, `<host>_r4plain_u2048b4` | E21: 64 synthesized calls of the key (length bucket 2048, batch 4, L mod 16 != 0), both conventions |
| `pro6000_e1b`, `pro6000_e1c` | E1 control runs (E1b contaminated, not used; E1c) |
| `pro6000_r3` (`e9_placement_*`) | E16: the E9 placement control on the RTX PRO 6000 |
| `vgpu3090full` (`e17_*`) | E17c full rerun (the first `vgpu3090` file lost rows) |
| `smoke-*` | smoke runs, in `results/tpds/smoke/`, never used |

Multi-round tables (`--rounds N > 1`) keep every single-round column (pooled over N x repeats
samples) and add `lat_c<C>_med`, `lat_c<C>_rsd`, `lat_c<C>_rounds`, `oracle_chunk_med`,
`oracle_med_ms` and `tied_chunks_med`; per-round blocks are in `e2_<model>_<tag>_rounds.jsonl`.
`analysis.round_median_view(df)` returns the table with the round-median latencies and oracle in
the single-round columns, so every per-call analysis can read it unchanged.

## Experiments: script, result files, paper location

Experiment IDs follow `docs/Design.md`. No experiment ran under the ID E6 (real Gated DeltaNet
traces, proposed in tpds-r1, were captured later as E17b).

| ID | What | Host | Script (lane) | Result files | Paper location |
|---|---|---|---|---|---|
| E1 | value invariance: 3 kernels, 8 inputs, act/all modes, L 512/2048/8192, batch 1, 3 rounds | pro6000, vgpu3090 | `e1_value_invariance`, `e1_analysis` (`tpds_lane_a.sh`) | `e1_{pro6000,vgpu3090}.{json,jsonl}`, `e1_analysis_{pro6000,vgpu3090}.json` | `sec:res_invariance` (`fig:value_invariance`, `tab:value_invariance`), `fig:chunk_curves`, `tab:replication`; `S-tab:supp_validation`, `S-tab:supp_latency`, `S-tab:supp_effects`, `S-tab:supp_tost`, `S-tab:supp_latency_3090`, `S-tab:supp_effects_3090`, `S-sec:supp_invariance_details`. Medians also calibrate the cost model |
| E1b | E1 control, 10 rounds, extra draws, GPU state | pro6000 | `tpds_lane_e1b.sh` | `e1_pro6000_e1b.{json,jsonl}`, `e1_analysis_pro6000_e1b.json` | not used (another job held the GPU; cancelled) |
| E1c | E1b rerun with in-run exclusive-GPU guard | pro6000 | `tpds_lane_e1c.sh`; rows rebuilt by `e1_from_log` | `e1_pro6000_e1c.{json,jsonl}`, `e1_pro6000_e1c.log` (raw run log), `e1_analysis_pro6000_e1c.json` | `S-tab:supp_redraw`, `S-sec:supp_kernel_experiments` ("Control run with independent redraws") |
| E2 | per-call oracle on 2360 captured 370M calls per model, batch 1/4/16 | pro6000, vgpu3090 | `capture_traces`, `e2_oracle_sweep` (`tpds_lane_b.sh`) | `e2_mamba{1,2}_{pro6000,vgpu3090}.{csv,json}` | `sec:res_oracle`, `tab:oracle_regret` (pro6000), `tab:replication` (vgpu3090); `S-fig:supp_entropy_regime`, `S-fig:entropy_vs_oracle`. Input of every per-call analysis |
| E3 | predictors of the per-call oracle (CPU) | - | `e3_predictors` | `e3_mamba{1,2}_{pro6000,vgpu3090}.json` | `sec:res_predictors`, `tab:predictors` (all rows except "Batch size only" and the last three; conditional mutual information), `tab:replication`; `S-fig:supp_regret_cdf` |
| E4 | auto-tuning baseline (Mamba-1 184 configs, Mamba-2 chunk), decision overhead, host sync | pro6000, vgpu3090 | `e4_autotune` (`tpds_lane_a.sh`) | `e4_{pro6000,vgpu3090}.{json,jsonl}` | `sec:res_autotune`, `S-tab:supp_autotune`, `S-tab:supp_autotune_3090`, `S-tab:supp_overhead`, `S-sec:supp_isolated_cost`; host synchronization in `tab:replication`. Launch sweep and static re-measurements calibrate the cost model |
| E5 | break-even of tuning vs online scheduling (CPU) | - | `e5_breakeven` | `e5_{pro6000,vgpu3090}.json` | `sec:res_breakeven` (headroom against a single profiled chunk, o = 4.43 ms), `S-sec:supp_breakeven`, `S-fig:supp_breakeven` (pro6000) |
| E7 | Triton scan vs stock reference CUDA scan, 12 E4 keys | vgpu3090 | `e7_reference_scan` (`tpds_lane_e7.sh`) | `e7_vgpu3090.{json,jsonl}` | `sec:methodology` "Kernels" (0.81-1.54x), `S-tab:supp_reference` |
| E8 | runtime-chunk reference scan on the 2360 Mamba-1 calls and the 12 E4 keys | vgpu3090 | `e8_refscan` (`tpds_lane_r2.sh`; prep `tpds_r2_prep.sh`) | `e8_refscan_vgpu3090.{csv,json}`, `e8_analysis_vgpu3090.json` | `sec:res_reference`, `tab:reference` ("Reference, 370M"); reference-scan row of `S-tab:supp_headroom_noise`. E4-key rows feed `cost_model --reference` |
| E9 | value vs placement (designs A/B/C), Triton and reference scan; device-side reduction cost | vgpu3090 | `e9_placement` (`tpds_lane_r2.sh`) | `e9_placement_vgpu3090.{jsonl,json}` | `sec:res_invariance` (placement control), `S-tab:supp_placement`, `S-sec:supp_kernel_experiments` "Value versus placement"; 7.3 / 42 us reduction cost in `sec:res_breakeven` |
| E10 | single-session interleaved routed end-to-end study, 25 configurations x 3 lengths x 30 rounds | vgpu3090 | `e10_routed_interleaved` (`tpds_lane_r2.sh`) | `e10_routed_vgpu3090.{jsonl,json}` | `sec:res_routed`, `tab:routed_ablation`, `S-tab:supp_routed_full` |
| E11 | Mamba-1 1.4B / 2.8B, batch 1 | vgpu3090 | `capture_traces`, `e2_oracle_sweep` (`tpds_lane_r2.sh` with `TPDS_R2_ONLY=E11`, `tpds_lane_r2_e11*.sh`) | `e2_mamba1_vgpu3090_{1p4b,2p8b}.{csv,json}`, `e3_mamba1_vgpu3090_{1p4b,2p8b}.json`, `cost_model_transfer_e2_mamba1_vgpu3090_{1p4b,2p8b}.json` | superseded by E14/E15; only the rows marked superseded in `S-tab:supp_headroom_noise` and the last paragraph of `S-sec:supp_headroom_noise`. Traces deleted (`tpds_r2_cleanup_e11.sh`, author-approved) |
| E12 | chunk sweep at tensor-parallel widths 512-5120, 168 shapes, Triton and reference scan, 3 rounds, synthetic inputs | vgpu3090 | `e12_tp_widths --ref`, `e12_analysis` (`tpds_lane_r3.sh`, `TPDS_R3_ONLY=E12`) | `e12_tp_widths_vgpu3090.{jsonl,json}`, `e12_analysis_vgpu3090.json` | `sec:res_reference` "Tensor-parallel widths", `S-sec:supp_tp`, `S-tab:supp_tp` |
| E13 | data-centre GPU: compile facts, 370M capture, E4/E1 calibration, 2360 calls, E12 | h800 | `tpds_lane_r3_e13.sh` (prep `tpds_r3_e13_prep.sh`) | `kernel_stats_sm90.{json,jsonl}`, `e4_h800.*`, `e1_h800.*`, `e1_analysis_h800.json`, `e2_mamba1_h800.{csv,json}`, `e3_mamba1_h800.json`, `e12_tp_widths_h800.*`, `e12_analysis_h800.json`; offline `cost_model_h800.json`, `cost_model_curves_h800.csv`, `cost_model_baselines_h800.json`, `cost_model_transfer_h800.json` | `sec:res_datacentre` ("One H800 PCIe"), `tab:replication` (H800 column), `S-sec:supp_h800` |
| E13b | E13 calibration and 2360 calls in device time | h800 | `tpds_lane_r3_e13.sh` (step 2) | `*_h800_r3spin.*` (e4, e1, e1_analysis, e2_mamba1, e3_mamba1, cost_model, cost_model_curves, cost_model_baselines) | `sec:res_datacentre` (device time), `S-sec:supp_h800` |
| E14 | 1.4B / 2.8B re-timed at batch 1/4/16, device time, host load logged | vgpu3090 | `capture_traces`, `e2_oracle_sweep --pre-spin-ms 20` (`tpds_lane_r3.sh`, `TPDS_R3_ONLY=E14`; download `tpds_r3_e14_download.sh`) | `e2_mamba1_vgpu3090_{1p4b,2p8b}_r3.{csv,json}`, `e3_mamba1_vgpu3090_{1p4b,2p8b}_r3.json`, `cost_model_transfer_e2_mamba1_vgpu3090_{1p4b,2p8b}_r3.json`, `e14_loadavg_vgpu3090.tsv` (host load average every 30 s; not in the manuscript) | `sec:res_reference`, `tab:reference` ("Triton, 1.4B / 2.8B"), device-time rows of `S-tab:supp_conventions` |
| E15 | both timing conventions on the RTX 3090: 1.4B/2.8B launch-inclusive; 370M calibration and 2360 calls in device time | vgpu3090 | `tpds_lane_r3_e15.sh` | `e2_mamba1_vgpu3090_{1p4b,2p8b}_r3plain.{csv,json}`, `e3_..._r3plain.json`, `cost_model_transfer_..._r3plain.json`; `e4_vgpu3090_r3spin.*`, `e1_vgpu3090_r3spin.*`, `e2_mamba1_vgpu3090_r3spin.{csv,json}`, `e3_mamba1_vgpu3090_r3spin.json`, offline `cost_model_vgpu3090_r3spin.json`, `cost_model_curves_vgpu3090_r3spin.csv`, `cost_model_baselines_vgpu3090_r3spin.json` | `sec:res_replication` "Device time", `S-sec:supp_conventions`, `S-tab:supp_conventions`; the `vgpu3090_r3spin` fit is the cost model applied without refitting to the 1.4B/2.8B calls in `tab:reference` |
| E15b | 2360 Mamba-2 370M calls in device time | vgpu3090 | `tpds_lane_r3_e15b.sh` | `e2_mamba2_vgpu3090_r3spin.{csv,json}`, `e3_mamba2_vgpu3090_r3spin.json` | not used for results: `S-sec:supp_conventions` states that these timings mix two regimes and are not used |
| E16 | E9 placement control on the RTX PRO 6000 (Triton kernel), 5 rounds | pro6000 | `e9_placement --tag pro6000_r3` (`tpds_lane_r3_e16.sh`; prep `tpds_r3_e16_prep.sh`) | `e9_placement_pro6000_r3.{jsonl,json}` | `S-tab:supp_placement_pro6000`, `S-sec:supp_invariance_details`, `sec:res_invariance` |
| E16b | Nsight Compute counters of the E9 cells | pro6000 | `e16_counters` (`tpds_lane_r3_e16.sh`) | none: cancelled, the container denies counter access (ERR_NVGPUCTRPERM) | - |
| E16c | placement with a warm and a cold L2 (counter-free substitute for E16b) | pro6000 | `e16c_l2_placement` (`tpds_lane_r3_e16c.sh`) | `e16c_l2_placement_pro6000.{jsonl,json}` | `S-tab:supp_l2`, `S-sec:supp_invariance_details`, `sec:res_invariance` (L2 overwritten before every repeat) |
| E17a | Mamba-2 2.7B (`AntonV/mamba2-2.7b-hf`, 8 of 64 layers), 2360 captured calls, device time | vgpu3090 (design planned pro6000) | `capture_traces --model mamba2`, `e2_oracle_sweep --model mamba2` (`tpds_lane_r3_e17.sh`; prep `tpds_r3_e17_prep.sh`) | `e2_mamba2_vgpu3090_2p7b_r3.{csv,json}`, `e3_mamba2_vgpu3090_2p7b_r3.json` (the E3 file is not in the manuscript) | `sec:res_reference` "Production kernels at larger widths", `S-sec:supp_largewidth`, `S-tab:supp_largewidth_calls` (through `e17_analysis_vgpu3090full.json` and `headroom_noise_e17.json`) |
| E17b | Gated DeltaNet 1.4B hybrid (`Idiap/gated-deltanet-swa-1.4B-30B`, 6 of its 12 GDN layers), 1770 captured calls (1152/366/252 at batch 1/4/16), production kernel entry, device time | vgpu3090 | `capture_gdn_traces`, `e2_oracle_sweep --model gdn` (`tpds_lane_r3_e17.sh`) | `e2_gdn_vgpu3090_1p4b_r3.{csv,json}`, `e3_gdn_vgpu3090_1p4b_r3.json` (the E3 file is not in the manuscript) | as E17a; the 1770-call count in `sec:methodology` |
| E17c | synthetic sweep: SSD at widths 2048/4096/5120, GDN at five head layouts, L 512-8192, batch 1/4/16, 3 rounds, launch-inclusive | vgpu3090 | `e17_workbound`, `e17_analysis` (`tpds_lane_r3_e17.sh`) | `e17_workbound_vgpu3090full.{jsonl,json}`, `e17_analysis_vgpu3090full.json`; `e17_workbound_vgpu3090.{jsonl,json}` is the first run, which lost rows to a git sync (superseded) | `S-tab:supp_largewidth`, `sec:res_reference` "Production kernels at larger widths" |
| E18 | RTX PRO 6000 device-time calibration (E4, E1) and the 2360 Mamba-1 calls in three rounds, both conventions (M2a, M2b) | pro6000 | `tpds_lane_r4_pro6000.sh` | planned: `e4_pro6000_r4spin.*`, `e1_pro6000_r4spin.*`, `e2_mamba1_pro6000_r4{spin,plain}.{csv,json}`, `..._rounds.jsonl` | not run yet (smoke-tested); no paper location |
| E19 | RTX 3090 three-round sweep of the 2360 Mamba-1 calls, both conventions (M2b) | vgpu3090 | `tpds_lane_r4_3090.sh` | planned: `e2_mamba1_vgpu3090_r4{spin,plain}.{csv,json}`, `..._rounds.jsonl` | not run yet |
| E20 | cold-L2 re-timing: E4 keys, E1 medians, calls with L >= 2048 (544 of 2360), three rounds (M3a) | pro6000 | `tpds_lane_r4_pro6000.sh` | planned: `e4_pro6000_r4cold.*`, `e1_pro6000_r4cold.*`, `e2_mamba1_pro6000_r4cold.*` | not run yet |
| E21 | 64 synthesized calls of the thin serving key (bucket 2048, batch 4, unaligned), three rounds, both conventions (M10b) | vgpu3090 | `tpds_lane_r4_3090.sh` | planned: `e2_mamba1_vgpu3090_r4{spin,plain}_u2048b4.*` | not run yet |

## Offline analyses (CPU only; read the result files above)

| Analysis | Script | Output | Paper location |
|---|---|---|---|
| T1 cost model (ledger CM1): fit, V1 fit quality, V2 selection and leave-one-key-out, V3 cross-GPU transfer, V4 work-term model of Mamba-2 SSD and Gated DeltaNet, ablation, top-k | `cost_model` | `cost_model_{pro6000,vgpu3090,h800,h800_r3spin,vgpu3090_r3spin}.json`, `cost_model_curves_<tag>.csv` | `sec:cost_model` (`eq:cost_model`), `sec:res_model`, `fig:cost_model`, "Cost model" row of `tab:predictors` (Mamba-2: work-term model), `tab:replication`; `S-sec:supp_cost_model` (`S-tab:supp_model_params`, `S-tab:supp_model_ablation`, `S-tab:supp_pipeline_model`), `S-sec:supp_h800`, `S-tab:supp_conventions` |
| Cost model of the reference CUDA scan | `cost_model --reference` | `cost_model_refscan_vgpu3090.json` | "Cost model" row of `tab:reference`, `S-sec:supp_cost_model` "Reference CUDA scan"; reference rows of `S-tab:supp_tp` |
| Cost model applied to other checkpoints (no refit) | `cost_model --transfer-calls` | `cost_model_transfer_e2_mamba1_vgpu3090_*.json` | `_r3`: `tab:reference` ("Cost model", "Table tuned on 370M"); `_r3plain`: `S-tab:supp_conventions`; without suffix: superseded (E11) |
| Cross-GPU transfer onto the H800 | `cost_model` V3 | `cost_model_transfer_h800.json` | `sec:res_datacentre`, `S-sec:supp_h800` |
| Compile facts | `kernel_stats` | `kernel_stats_sm{120,86,90}.{json,jsonl}` | `S-tab:supp_model_compile` (sm_120, sm_86); input of every cost-model fit |
| Cost-model baselines (tpds-r3, M4a/N3) | `cost_model_baselines` | `cost_model_baselines_{pro6000,vgpu3090,h800,h800_r3spin,vgpu3090_r3spin}.json` | rows "Batch size only", "Launch and work terms only" and "Nearest calibration key" of `tab:predictors`, `sec:res_model`; Kendall rank correlation in `S-sec:supp_cost_model`; `S-sec:supp_h800`; nearest-key number in `sec:res_replication` "Device time" |
| Cost-model baselines, tpds-r4 (M5a, M5b, M6d) | `cost_model_baselines_r4` | `cost_model_baselines_r4_<tag>.json`, `cost_model_baselines_r4_summary.json` | not in the manuscript yet (tpds-r4 results rewrite) |
| E1 analysis incl. TOST (T4) | `e1_analysis` | `e1_analysis_<tag>.json` | see E1, E1c, E13 |
| Information tests, tpds-r4 (M7d, M2c) | `e3b_information` | `e3b_information_<model>_<tag>.json` (16 per-call tables), `e3b_information_refscan_vgpu3090.json`, `e3b_information_summary.{json,csv}` | not in the manuscript yet |
| Entropy band | `entropy_band` | `entropy_band_{pro6000,vgpu3090}.json` | `sec:res_predictors` (entropy band of real calls, width within a layer) |
| Key churn (T3) | `e5b_key_churn` | `e5b_{pro6000,vgpu3090}.json` | `sec:res_breakeven` "Key churn" (`eq:keychurn`), `fig:keychurn`, held-out headroom and host synchronization in `tab:replication`, `S-sec:supp_breakeven` "Per-key tuning costs" |
| Serving-trace replay (M1b) | `serving_key_churn` | `serving_key_churn.json` | `sec:res_breakeven` "Serving traces", `S-sec:supp_serving`, `S-tab:supp_serving` |
| Oracle headroom under timing noise (M3b, M3c) | `headroom_noise` | `headroom_noise.json` (370M, reference scan, superseded E11 rows), `headroom_noise_r3.json` (E14/E15 data sets), `headroom_noise_e17.json` (E17a/b) | `S-sec:supp_headroom_noise`, `S-tab:supp_headroom_noise`, `sec:res_breakeven`; headroom row of `tab:reference`, headroom column of `S-tab:supp_conventions`, headroom row of `S-tab:supp_largewidth_calls` |
| Headroom rows and N1 checks, tpds-r4 (M8a, M3c, N1) | `headroom_r4` | `headroom_r4.json` | not in the manuscript yet |
| Deployment units, tpds-r4 (M6a, M6b, M10b, N1) | `deployment_units` | `deployment_units.json` | not in the manuscript yet |
| T2 no-gain bound | no code | - | `sec:why`, `thm:nogain`; instantiated with E4/E5b/E9 numbers in `sec:res_breakeven` |

Notes on the tpds-r4 analyses: `docs/tmp/tpds_r4_analysis_notes.md` summarizes their numbers.

## Regenerating tables and figures

Every command runs from the repository root with `PYTHONPATH=.` and needs only the CPU (except
`kernel_stats` for the native architecture, which needs a CUDA device; other architectures are
cross-compiled with `--archs`). Outputs are rewritten in place in `src/experiments/results/tpds/`.

```bash
export PYTHONPATH=.
M=src.experiments.tpds
R=src/experiments/results/tpds
# per-experiment analyses
python -m $M.e1_analysis --tag pro6000                 # also vgpu3090, pro6000_e1c, h800, h800_r3spin
python -m $M.e3_predictors                             # every e2_<model>_<tag>.csv of mamba1/mamba2
python -m $M.e3_predictors --models mamba2,gdn --hosts vgpu3090_2p7b_r3,vgpu3090_1p4b_r3
python -m $M.entropy_band --host pro6000               # and --host vgpu3090
python -m $M.e8_refscan --analyze --tag vgpu3090
python -m $M.e9_placement --analyze --tag vgpu3090     # E16: --tag pro6000_r3
python -m $M.e16c_l2_placement --analyze --tag pro6000
python -m $M.e10_routed_interleaved --analyze --tag vgpu3090
python -m $M.e12_analysis --host vgpu3090              # and --host h800
python -m $M.e17_analysis --tag vgpu3090full --per-call e2_mamba2_vgpu3090_2p7b_r3.csv,e2_gdn_vgpu3090_1p4b_r3.csv
# cost model (T1), its variants and baselines
python -m $M.kernel_stats --archs 120,86               # compile facts; sm_90 on the H800 host
python -m $M.cost_model --ablation                     # pro6000 and vgpu3090 (with V3 between them)
python -m $M.cost_model --hosts h800                   # one tag per call: h800_r3spin, vgpu3090_r3spin likewise
python -m $M.cost_model --reference                    # cost_model_refscan_vgpu3090.json
python -m $M.cost_model --hosts vgpu3090_r3spin --transfer-calls $R/e2_mamba1_vgpu3090_1p4b_r3.csv \
    --transfer-ref $R/e2_mamba1_vgpu3090_r3spin.csv    # 2p8b likewise; _r3plain: --hosts vgpu3090,
                                                       # --transfer-ref $R/e2_mamba1_vgpu3090.csv
python -m $M.cost_model_baselines --hosts pro6000,vgpu3090,h800,h800_r3spin,vgpu3090_r3spin
python -m $M.cost_model_baselines_r4
# break-even, key churn, serving, headroom, tpds-r4 re-analyses
python -m $M.e5_breakeven --host pro6000               # and --host vgpu3090
python -m $M.e5b_key_churn
python -m $M.serving_key_churn                         # traces in src/data/serving_traces (git-ignored; URLs in the script)
python -m $M.headroom_noise                            # headroom_noise.json
python -m $M.headroom_noise --datasets mamba1_1p4b_r3_vgpu3090,mamba1_2p8b_r3_vgpu3090,mamba1_1p4b_r3plain_vgpu3090,mamba1_2p8b_r3plain_vgpu3090,mamba1_r3spin_vgpu3090 \
    --out $R/headroom_noise_r3.json
python -m $M.headroom_noise --datasets mamba2_2p7b_r3_vgpu3090,gdn_1p4b_r3_vgpu3090 --out $R/headroom_noise_e17.json
python -m $M.e3b_information
python -m $M.headroom_r4
python -m $M.deployment_units
```

`cost_model_transfer_h800.json` collects the V3 transfers onto the H800 and was computed with
`cost_model --hosts h800,<source> --no-loko` (and the same for `h800_r3spin`); that call also
rewrites `cost_model_<source>.json` without the leave-one-key-out block, so re-run the source's
full fit afterwards.

**Figures** (`paper/figs/`, a git submodule): `make_figures --host pro6000 --out-dir paper/figs`
writes all eight; the two cross-GPU figures use `--cross-hosts pro6000,vgpu3090`.

| Figure file | Data | Paper label |
|---|---|---|
| `tpds_value_invariance.pdf` | `e1_pro6000.json` | `fig:value_invariance` |
| `tpds_latency_vs_chunk.pdf` | `e1_pro6000.json` | `fig:chunk_curves` |
| `tpds_cost_model.pdf` | `cost_model_curves_{pro6000,vgpu3090}.csv` | `fig:cost_model` |
| `tpds_key_churn.pdf` | `e5b_{pro6000,vgpu3090}.json` | `fig:keychurn` |
| `tpds_entropy_by_regime.pdf` | `e2_mamba{1,2}_pro6000.csv` | `S-fig:supp_entropy_regime` |
| `tpds_entropy_vs_oracle.pdf` | `e2_mamba{1,2}_pro6000.csv` | `S-fig:entropy_vs_oracle` |
| `tpds_regret_cdf.pdf` | `e3_mamba{1,2}_pro6000.json` | `S-fig:supp_regret_cdf` |
| `tpds_breakeven.pdf` | `e5_pro6000.json` | `S-fig:supp_breakeven` |

**Tables**: `make_tables --host <tag> <table>...` prints LaTeX table bodies (rows only) to stdout.

| `make_tables` name | Host | Data | Paper label |
|---|---|---|---|
| `validation` | pro6000 | E1 validation rows | `S-tab:supp_validation` |
| `latency` | pro6000 / vgpu3090 | `e1_<host>.json` | `S-tab:supp_latency` / `S-tab:supp_latency_3090` |
| `effects` | pro6000 / vgpu3090 | `e1_analysis_<host>.json` | `S-tab:supp_effects` / `S-tab:supp_effects_3090` |
| `tost` | pro6000, vgpu3090 (one block each) | `e1_analysis_<host>.json` | `S-tab:supp_tost` |
| `redraw` | pro6000 | `e1_pro6000{,_e1c}.json`, `e1_analysis_pro6000_e1c.json` | `S-tab:supp_redraw` |
| `autotune` | pro6000 / vgpu3090 | `e4_<host>.json` | `S-tab:supp_autotune` / `S-tab:supp_autotune_3090` |
| `overhead` | pro6000 | `e4_pro6000.json` | `S-tab:supp_overhead` |
| `reference` | vgpu3090 | `e7_vgpu3090.json` | `S-tab:supp_reference` |
| `model_params`, `model_ablation` | (both GPUs) | `cost_model_{pro6000,vgpu3090}.json` | `S-tab:supp_model_params`, `S-tab:supp_model_ablation` |
| `model_compile` | (sm_120, sm_86) | `kernel_stats_sm{120,86}.json` | `S-tab:supp_model_compile` |
| `routed_e10` | vgpu3090 | `e10_routed_vgpu3090.json` | `S-tab:supp_routed_full`; selected rows in `tab:routed_ablation` |

The other tables are transcribed from the JSON summaries named in the two tables above:
`tab:value_invariance` (`e1_analysis_pro6000.json`), `tab:oracle_regret` (`e2_*_pro6000.json`,
`e3_*_pro6000.json`), `tab:predictors` (`e3_*_pro6000.json`, `cost_model_pro6000.json`,
`cost_model_baselines_pro6000.json`), `tab:reference` (`e8_analysis_vgpu3090.json`,
`cost_model_refscan_vgpu3090.json`, `e3_mamba1_vgpu3090_{1p4b,2p8b}_r3.json`,
`cost_model_transfer_*_r3.json`, `headroom_noise.json`, `headroom_noise_r3.json`),
`tab:replication`, `S-tab:supp_placement` / `S-tab:supp_placement_pro6000` (`e9_placement_*.json`),
`S-tab:supp_l2` (`e16c_l2_placement_pro6000.json`), `S-tab:supp_pipeline_model` (`V4_pipelines` of
`cost_model_{pro6000,vgpu3090}.json`), `S-tab:supp_tp` (`e12_analysis_vgpu3090.json`),
`S-tab:supp_largewidth` / `S-tab:supp_largewidth_calls` (`e17_analysis_vgpu3090full.json`,
`headroom_noise_e17.json`), `S-tab:supp_conventions`, `S-tab:supp_headroom_noise`
(`headroom_noise*.json`) and `S-tab:supp_serving` (`serving_key_churn.json`).

Not produced from this directory: the supplement sections on the entropy estimator
(`S-sec:supp_estimator`), the cost of computing the statistic (`S-sec:supp_cost`), the legacy
measurements (`S-sec:supp_legacy`) and the entropy statistics of real checkpoints
(`S-sec:supp_real_entropy`); they come from earlier harnesses and are used only to bound the cost
of the statistic, never to rank chunks.

## Running experiments

From the repository root (`PYTHONPATH=.`), single scripts:

```bash
python -m src.experiments.tpds.kernels                       # correctness self-test
python -m src.experiments.tpds.capture_traces --model mamba1 # then --model mamba2
python -m src.experiments.tpds.e1_value_invariance
python -m src.experiments.tpds.e2_oracle_sweep --model mamba1 # and --model mamba2
python -m src.experiments.tpds.e4_autotune
python -m src.experiments.tpds.e3_predictors
python -m src.experiments.tpds.e5_breakeven --host <tag>
python -m src.experiments.tpds.make_figures --host <tag> --out-dir paper/figs
```

Useful options of the timing scripts: `--tag` (result-file tag), `--pre-spin-ms` (device time;
E1 and E4 read `TPDS_PRE_SPIN_MS`), `--flush-l2` (cold L2; E1, E2, E4), `--exclusive-guard
--guard-until <epoch>` (pause while another process holds the GPU, re-time what it overlapped),
and for `e2_oracle_sweep`: `--rounds N`, `--min-length L`, `--synth-calls N --synth-bucket B
--synth-batch b --synth-align {aligned,unaligned}`, `--max-rows`.

### Lanes

GPU work runs through the lanes in `src/experiments/remote/`, launched with `src/revloop/remote.sh`
(through WSL from Windows):

```bash
wsl bash src/revloop/remote.sh exec   pro6000 src/experiments/remote/tpds_prep.sh
wsl bash src/revloop/remote.sh launch pro6000 tpds-a src/experiments/remote/tpds_lane_a.sh
wsl bash src/revloop/remote.sh launch pro6000 tpds-b src/experiments/remote/tpds_lane_b.sh
wsl bash src/revloop/remote.sh watch  pro6000 E1 E2 E3 E4 E5
```

| Cycle | Lane / prep | Experiments |
|---|---|---|
| tpds-r1 | `tpds_prep.sh` (CPU), `tpds_lane_a.sh`, `tpds_lane_b.sh`, `tpds_lane_e1b.sh`, `tpds_lane_e1c.sh`, `tpds_lane_e7.sh` | E1, E4 / capture, E2, E3, E5 / E1b / E1c / E7. On the RTX 3090 the tpds-r1 runs used `TPDS_ID_PREFIX=3090-` (ledger IDs `3090-E1` ...) |
| tpds-r2 | `tpds_r2_prep.sh` (runtime-chunk build), `tpds_lane_r2.sh` (`TPDS_R2_ONLY`), `tpds_lane_r2_e11.sh`, `tpds_lane_r2_e11_after_download.sh`, `tpds_r2_cleanup_e11.sh` (deletes the E11 traces and checkpoints; run only with the author's confirmation) | E8, E9, E10, E11 |
| tpds-r3 | `tpds_lane_r3.sh` (`TPDS_R3_ONLY` = E12, E13, E14), `tpds_r3_e13_prep.sh`, `tpds_lane_r3_e13.sh`, `tpds_r3_e14_download.sh`, `tpds_lane_r3_e15.sh`, `tpds_lane_r3_e15b.sh`, `tpds_r3_e16_prep.sh`, `tpds_lane_r3_e16.sh`, `tpds_lane_r3_e16c.sh`, `tpds_r3_e17_prep.sh`, `tpds_lane_r3_e17.sh` (`TPDS_E17_ONLY`) | E12-E17 |
| tpds-r4 | `tpds_lane_r4_pro6000.sh`, `tpds_lane_r4_3090.sh` | E18-E21 |

Shared settings are in `tpds_common.sh`: `PY`, `TPDS_TAG` (default `$REVLOOP_HOST`, else the
hostname), `TPDS_TRACE_DIR` (default `src/outputs/tpds_traces`), `TPDS_SMOKE=1` (tiny run, outputs
in `results/tpds/smoke/`, markers prefixed `smoke-`), `TPDS_NO_GPU_LOCK=1` (debugging only),
`TPDS_ID_PREFIX`. Untracked host-local settings go in `.tpds_env` at the repository root. Every
GPU step holds an exclusive `flock` (`src/logs/tpds/gpu_timing.lock`), so no kernel timing
overlaps another lane. Lanes check free bytes and inodes before writing.

### Cycle tpds-r4 lanes (review M2a, M2b, M3a, M10b)

| Lane | Host | Steps (default `TPDS_R4_ONLY`) | Estimate (lane header) |
|---|---|---|---|
| `tpds_lane_r4_pro6000.sh` | RTX PRO 6000 (`pro6000`) | E18: E4 Mamba-1 launch sweep and E1 Mamba-1 medians in device time (`<host>_r4spin`), then the 2360 captured 370M Mamba-1 calls in three rounds in device time (`_r4spin`) and launch-inclusive (`_r4plain`). E20: cold L2 (`--flush-l2`) plus device time: E4 Mamba-1, E1 Mamba-1, and the captured calls with L >= 2048 in three rounds (`_r4cold`) | E18 ~1.2 GPU-h, E20 ~0.6 GPU-h |
| `tpds_lane_r4_3090.sh` | RTX 3090 (`vgpu3090`) | E19: the 2360 calls in three rounds, `_r4spin` and `_r4plain` (the device-time calibration exists from E15). E21: 64 synthesized calls of the key (bucket 2048, batch 4, L mod 16 != 0), three rounds, `_r4spin_u2048b4` and `_r4plain_u2048b4` | E19 ~1.4 GPU-h, E21 ~5 min |

```bash
wsl bash src/revloop/remote.sh launch pro6000  tpds-r4 src/experiments/remote/tpds_lane_r4_pro6000.sh
wsl bash src/revloop/remote.sh launch vgpu3090 tpds-r4 src/experiments/remote/tpds_lane_r4_3090.sh
```

Options (environment variables):

| Variable | Default | Effect |
|---|---|---|
| `TPDS_R4_ONLY` | `E18 E20` / `E19 E21` | steps to run |
| `TPDS_R4_ROUNDS` | 3 | `e2_oracle_sweep --rounds`: every call is timed in N rounds, chunk order shuffled per call and per round (seeded, so a restart reproduces it); the per-call oracle is defined on round medians, ties on the between-round CI. N = 1 is the unchanged single-round sweep |
| `TPDS_R4_SPIN_MS` | 20 (smoke 5) | GPU pre-spin of the device-time steps; the `_r4plain` sweeps use 0 |
| `TPDS_R4_GUARD` | 1 | exclusive-GPU guard of the E2 sweeps (`--exclusive-guard --guard-until now + TPDS_R4_GUARD_H`): the sweep pauses while another process holds the GPU and re-times the calls it overlapped. 0 disables it and saves ~0.2 s of `nvidia-smi` queries per call (about 8 min per 2360-call sweep). The E1/E4 steps are not guarded |
| `TPDS_R4_GUARD_H` | 12 | guard deadline in hours; a paused run then stops cleanly and resumes on relaunch |
| `TPDS_R4_E21_CALLS` | 64 (smoke 6) | 3090 lane: number of synthesized E21 calls |
| `TPDS_TRACE_NEED_GB` | 30 | free space required before a re-capture |

Both lanes first run `capture_traces --check-only` (no model load) and re-capture the 370M Mamba-1
traces (~22 GB; needs `state-spaces/mamba-370m-hf` in the HF cache or via `HF_ENDPOINT`) only if
they are missing or incomplete. Markers: `src/logs/revloop/{CAPTURE-r4,E18,E20}` and
`{CAPTURE-r4,E19,E21}` `.{START,OK,FAIL}`. Every step is resumable (completed rows and rounds are
skipped), and a failed step does not stop the others. The analysis of E18-E21 reruns the per-call
analyses on `analysis.round_median_view` of the new tables.

## Outputs and logs

Result files: `src/experiments/results/tpds/` (the `.json` files of E1/E4 contain every row, so the
ledger globs capture the complete data; `.jsonl` files are the resumable raw logs). Captured traces:
`src/outputs/tpds_traces/` (370M), `src/outputs/tpds_traces_<ckpt>/` (larger checkpoints;
git-ignored, size-guarded by `--max-gb`). Logs: `src/logs/tpds/`; lane markers:
`src/logs/revloop/<ID>.{START,OK,FAIL}`. Smoke outputs: `src/experiments/results/tpds/smoke/`.
Never sync (`git pull`) a host while a lane appends to a tracked result file: a sync once replaced
`e17_workbound_vgpu3090.jsonl` mid-run.

## Notes

- Mamba-1 launch configuration when only the chunk varies (E1, E2, E7-E21):
  `kernels.DEFAULT_MAMBA1_LAUNCH` (per-chunk BLOCK_D, num_warps). E4 searches the full joint
  space (184 configurations per key).
- Capture replaces each mixer's prefill forward by an equivalent forward that calls the model's own
  submodules plus the validated scan kernel; the logits are checked against the stock transformers
  forward before capture (recorded in the manifest).
- Mamba-1 traces store u, delta (pre-softplus, dt_proj bias included), B, C and the gate z; Mamba-2
  traces store x, dt (pre-bias), B, C. All fp16. 370M captures take every sixth layer (8 of 48),
  the 1.4B every sixth (8 of 48), the 2.8B and Mamba-2 2.7B every eighth (8 of 64), over the same
  192 prompts.
- Gated DeltaNet: E1 times it with synthetic inputs only, at a 370M-class layout (4 heads,
  d_k = 256, d_v = 512), through the plain kernel entry. Real Gated DeltaNet calls come from E17b:
  `capture_gdn_traces` stores the inputs the layers of the public 1.4B hybrid
  `Idiap/gated-deltanet-swa-1.4B-30B` hand to the FLA kernel (raw q, k, v, the `a_proj` output g,
  the `b_proj` output beta, plus A_log and dt_bias), for 6 of its 12 Gated DeltaNet layers over the
  192 prompts; `e2_oracle_sweep --model gdn` times the resulting 1770 calls at chunks 16/32/64
  through the production entry (`kernels.gdn_chunk_prod`). A stored-input round trip reproduces the
  live kernel output (relative difference 4e-5). E17c adds synthetic sweeps at five head layouts.
- Mamba-2 SSD indexes its chunk-state tensor with 32-bit offsets: at width 5120 and batch 16,
  chunk 32 is unavailable for L > 6528 (24 of the 2360 Mamba-2 2.7B calls carry no chunk-32 timing).
