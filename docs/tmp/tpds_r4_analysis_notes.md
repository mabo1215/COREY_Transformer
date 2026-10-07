# tpds-r4 offline analysis notes (input for the results-section rewrite)

All numbers below come from the r4 scripts' result files in src/experiments/results/tpds/.
Paper text must cite them through the result files, never by script name.

## E3b information tests (M7d, M2c) - e3b_information_<tag>.json, e3b_information_summary.{json,csv}
- Settings: 2000 bootstrap replicates over 42 connected prompt clusters (batched calls mix prompts;
  CIs over 611 request groups also stored), 1000 permutations, 1000 CMI permutations, 50 CV re-splits.
- Reproduction: all E3 rows match (max diff 0.0003 pp); batch-only 7.19/47.4; in-sample floor 0.36;
  CMI 0.046 / 1.228 bits; cost model 0.88%, 90% tied.
- DEPTH MISMATCH: e3_predictors.py uses max_depth=4 (L_only uses 3); paper says "depth 3".
  True depth-3 shape trees: 0.91 (PRO 6000), 2.70 (RTX 3090), 3.11 (H800), 3.61 (3090 device time,
  vs 0.38 at depth 4). Entropy adds 0.000 at depth 3 as well. -> paper must say depth 4.
- Shape vs shape+entropy (mean regret %, delta pp [95% CI], equivalence at 0.5/1 pp):
  M1 PRO6000 0.73 [0.50,1.04] vs 0.74, +0.005 [-0.005,+0.019] E/E; M1 3090 1.93 vs 1.93, -0.001 [-0.03,+0.02];
  M1 H800 2.56 vs 2.59, +0.03 [-0.01,+0.10]; M1 3090 device 0.38 vs 0.34, -0.04 [-0.22,+0.08];
  1.4B/2.8B device 2.11/2.90, delta 0.000; reference scan 3.00 vs 3.13, +0.13 [+0.03,+0.25];
  M2 PRO6000 2.21 vs 2.21; M2 3090 device 6.02 vs 5.56, -0.47 [-0.92,-0.07] (I at 0.5, E at 1 pp).
- Exact-key permutation tests: M1 PRO6000 entropy -0.02 pp (p 0.82), any moment -0.03 (p 0.79);
  3090 entropy p 0.14; H800 p 0.75; M2 3090 device any moment +0.20 (p 0.033, Holm 0.10);
  1.4B device variance p 0.003 (Holm 0.018) but effect 0.004 pp.
- CMI (excess bits, p): PRO6000 M1 given bucket 0.043 (0.001); given exact key 0.005 (0.18);
  moment vector 0.001 (0.44); H(oracle|key) 0.70 bits; exact length within bucket adds 0.031 (0.001).
  3090/H800 given exact key 0.001. All 17 tags: entropy | exact key <= 0.013 bits (exception: superseded
  batch-1 1.4B 0.045, p 0.001, vanishes given key x layer: layer carries 0.054 bits).
- Higher capacity (delta for +entropy): M1 PRO6000 depth6 +0.003, GB classifier -0.04, GB regressor
  +0.10, all equivalent; H800 GB regressor -0.19 [-0.42,-0.03] and M2 PRO6000 depth6 -0.15 still
  equivalent at 0.5 pp. Extra features often hurt: ref scan +0.78, GDN +0.55..+1.1, batch-1 2.8B +1.6..+4.0.
- M2 3090 device: entropy lowers the tree regret by 0.47 pp (better in 90% of re-splits) but only to the
  exact-key table level (5.64%); CMI given key 0.003 (p 0.38): compensates tree capacity, no information
  beyond shape.
- Tie shares (CI overlap / within 2% / within 5%): M1 PRO6000 static 42/39/50, shape 93/94/97, model
  90/90/96; M1 3090 static 37/31/43, shape 86/82/89, model 69/63/78; M1 H800 static 51/47/53, shape
  86/85/86, model 79/75/79; M2 PRO6000 shape 79/80/94. Noisy tables: M2 3090 device shape 98/42/54,
  M2 2.7B 92/48/56, GDN 97/81/86, ref scan stock 72/55/72 -> CI overlap is NOT conservative there.
  Mean oracle tie-set size: M1 2.36/2.08/2.76; M2 3.06/3.24/4.27.
- 24 Mamba-2 2.7B calls lack chunk 32 timings (inf): capped only when fitting.

## Deployment units (M6a, M6b, M10b, N1.5/6/8/9) - deployment_units.json
- Kernel level, pooled regret to per-call oracle, us/call -> ms per forward pass (layers):
  Triton M1 370M PRO6000 (48): default 89.6->4.30, model 7.1->0.34, held-out exact table 0.65->0.03
  Triton M1 370M 3090: 119->5.71, 14.9->0.71, 4.9->0.24; H800: 86.0->4.13, 27.3->1.31, 1.05->0.05
  Ref CUDA 3090 stock: 36.4->1.75, model 7.0->0.34, table 6.2->0.30
  Triton 1.4B/2.8B (48/64): 302/393->14.5/25.2, 55/24->2.7/1.5, 6.2/8.3->0.30/0.53
  SSD 370M PRO6000/3090: 7.7/20.5->0.37/0.98, model 6.6/20.1, table 5.9/18.6->0.28/0.89
  SSD 2.7B above floor (64): 3.7->0.24, model 4.5->0.29, table 2.4->0.15
  GDN 1.4B above floor (12): 0.47->0.006, same for all. SSD table-minus-default ~2 us/call
  (<=0.09 ms/pass, noise floor); GDN exactly 0.
- Routed study (ref scan, 3090, batch 1) at 4096 tokens: stock 0.48 ms (0.48% of 100.4 ms prefill,
  0.035% of 1354 ms total); model 0.44 ms; held-out table 0.016 ms; best single chunk 3.36 ms (3.3%).
  At 976 tokens all policies <= 0.8%; at 256 ~0.4% (oracle noise).
- Sanity: 48 x (c128 - c512) at L=4096 = 17.37 ms predicted vs 16.81 +/- 1.41 ms measured;
  chunk 256: 3.34 vs 3.13 +/- 1.25 ms.
- Serving (step model, 10 windows, per-request rows): PRO6000 default 0.13-0.64 ms/request (2.1-2.6% of
  prefill), model 0.04-0.13, table 0.04-0.13 (<=0.93%); 3090 default 0.60-1.51 (5.5-10%), model
  0.40-1.58 (5.3-10.5%), table 0.20-0.46 (1.5-3.7%). All 88 primary rows, table <= 0.23/1.09 ms/request
  (<=0.96/3.7% of prefill; <=0.81/3.1% of total GPU time). Caveat: step model (10 us/token) is 2.4x
  faster than measured eager prefill at 4096 tokens; oracle includes noise bias -> upper bounds.
- M6b: P_k median 0.033 / 0.054 s per key; all 33 keys 2.5 / 5.0 s vs calibration 493 / 843 s.
  N*_model median 4564 / 3688 calls (95 / 77 requests); unbounded on 22 / 17 keys (model = table chunk).
  Keys with model regret above one sync (5 / 10): profiling pays after 19-91 requests (PRO6000),
  30-258 (3090; one unbounded). Replay: model default cheaper than profiling on 83-100% of calls
  (PRO6000) but only 31-87% (3090); on 3090 profiling every key costs 0.20-0.47 ms/request vs model
  0.40-1.58. Startup: profiling every bucket key 0.3-1.3 s per request-level window; padded budget
  rows 308 / 797 s (BurstGPT, 852 keys).
  Tension: model regret 7.1 / 14.9 us/call = 15.6x / 6.9x the held-out exact headroom (0.45 / 2.16 us)
  and 2.9x / 1.8x the bucket-table headroom (2.47 / 8.30 us). The headroom bounds online dispatch
  against a profiled table, not against the model; the model is a cold-start default whose excess
  (6.5 / 12.0 us/call) profiling recovers within N*.
- M10b: per-request, fixed-slice and packed rows: 0% of calls on keys with < 30 captured calls; padded
  budget rows up to 84.8%. Thin keys: (1024,4,unaligned;1), (2048,16,unaligned;8), (2048,4,unaligned;11).
  The "93%": Azure code 1 h, budget 8192, padded, colocated, rho 0.85 (sensitivity run) 92.8%; 84.8% of
  its calls on thin keys, 73.1% on (2048,4,unaligned,11 calls). Table S-serving rho 0.5 row: 90.8 / 76.5 / 65.7%.
- N1.5: 23.4% = chunk 256 per-call relative mean (e3 predictors global_static); 34 us (10.6%) = chunk 512,
  (mean-oracle)/mean (e5 view_e2_traces). Cross: chunk 256 = 89.6 us (23.8% weighted); chunk 512 = 38.3% relative.
- N1.6: o = 4.43 ms = call-weighted mean over 2360 calls of full-histogram time at nearest measured
  tensor size (e5_pro6000 view_e2_traces.mamba1.overhead_ms); by batch 0.93/4.46/20.4 ms. 188.8/48 = 3.93 ms
  is the 3090 in-model hook at 4096 tokens; 1.27-5.06 ms are batch-1 rows at L = 2048/8192 only.
- N1.8: static 512 prefill 100.41 +/- 0.85 ms; static 128 117.22 ms; paired diff 16.81 +/- 1.41 ms =
  16.79 +/- 1.47%: give absolute prefill, keep +/-1.4 ms next to +/-1.5%.
- N1.9: median N* 1901 / 2662 calls = 39.6 / 55.5 requests: write "1900 and 2660 calls".

## Headroom rows, N1.1/1.2/1.7, M3c - headroom_r4.json (seed 20260930, 200 draws)
- Reproduction: 12,409 stored values in headroom_noise*.json identical; e5b/e8 checks 21/21; 13/14 printed
  table entries match; superseded 2.8B split-half upper bound printed 10.6, true 10.548 -> 10.5.
- M8a replacement rows (us/call; cov = tuned shapes; counts naive/split-half; H800 sync = 20 us assumed):
  row | in-sample (max) | held-out all/cov | held-out max (cov) | split-half cov [95%] | >sync | >7.3 us
  370M PRO6000 LI   | 0.34 (4.4) | 0.65/0.45 | 4.4  | 0.13 [0.09,0.17] | 0/0 | 0/0
  370M 3090 LI      | 1.86 (6.7) | 4.89/2.16 | 6.7  | 1.78 [1.72,1.83] | 0/0 | 0/0
  370M 3090 device  | 0.03 (0.5) | 14.8/0.03 | 0.5  | 0.02 [0.01,0.03] | 0/0 | 0/0
  370M H800 LI      | 0.79 (8.5) | 1.05/1.15 | 11.6 | 0.69 [0.63,0.74] | 0/0 | 1/1
  370M H800 device  | 0.29 (8.2) | 0.34/0.37 | 10.8 | 0.37 [0.36,0.37] | 0/0 | 1/1
  Ref scan 3090 LI  | 2.07 (14.5)| 6.21/2.16 | 14.5 | 2.05 [1.99,2.10] | 0/0 | 3/3
  1.4B 3090 LI      | 2.14 (11.7)| 31.6/2.59 | 11.7 | 2.11 [2.02,2.19] | 0/0 | 3/2
  1.4B 3090 device  | 0.17 (4.7) | 6.15/0.18 | 4.7  | 0.11 [0.06,0.15] | 0/0 | 0/0
  2.8B 3090 LI      | 4.14 (11.0)| 41.8/4.80 | 11.0 | 4.49 [4.40,4.67] | 0/0 | 7/6
  2.8B 3090 device  | 0.29 (4.5) | 8.31/0.32 | 4.5  | 0.23 [0.18,0.28] | 0/0 | 0/0
  No Mamba-1 shape above 42 us. Size-interpolated device cost: ref 0, 1.4B LI 1/0, 2.8B LI 5/5.
  Held-out max (cov) for other rows: Mamba-2 PRO6000 13.7, Mamba-2 3090 43.0, Mamba-2 2.7B 80.0, GDN 52.5.
- 4.7 us = held-out per-shape max of 1.4B device time (4.69 at L=256, b16; headroom_noise_r3.json);
  2.8B 4.47; launch-inclusive maxima 11.7 / 11.0.
- 4.4/6.7/14.5 us: held-out maxima equal the in-sample ones (same largest shape kept in every fold);
  main text is correct; table needs the held-out-max column. With untuned fallback shapes: 16.4/328/784.
- N1.1: 0.985/0.992 = R^2 of log latency on captured calls (11,800 latencies); 0.983/0.998 = R^2 on the
  75 fitting timings. Name the data set for each.
- N1.2: pooled-shard p95 = 30.52% -> main text should say 31% (mean 6.51%).
- N1.7: LOKO, model as full auto-tuner over 184 configs: mean 8.92% (max 53.4%) PRO6000, 23.14% (max 95.3%)
  RTX 3090; reproduced by refit. Add to Table S-model-params.
- M3c: model predicts exact ties for all calls <= 512 tokens (1096/2360), ties default to chunk 32.
  Sending ties to 256: 370M PRO6000 LI 1.45 -> 1.56%; 370M 3090 LI 2.48 -> 2.36%; 370M 3090 device
  5.72 -> 5.61%; 2.7B 3090 device all 11.40 -> 12.74%; 2.7B above floor (397) 0.24 -> 0.19%; synthetic
  3 widths 1.46 -> 1.63%. Premise does not hold: tied calls lie below the floor and chunk 32 is never
  clearly worst there. Work-term-only ranking: 1.52, 2.49, 5.62, 11.68, 1.65%.
  -> Paper: state the tie rule explicitly and report both defaults; no change of headline.
- Launch-bound boundary F/(a+2 sqrt(bg)) at a 320 us floor: ~3,900 tokens on the RTX 3090 (3,878, from
  9,513; per batch ~3,880/970/240); PRO 6000 ~11,800 (from 17,321) but 320 us was measured on the 3090 only.

## Cost-model baselines and clean selection (M5a, M5b, M6d) - cost_model_baselines_r4_<tag>.json
- Reproduction exact: model 0.878 / 3.249 / 3.543%, all 11 ablation forms, three-parameter restriction,
  work-term, nearest key, top-k, stored LOKO 0.98/2.04/3.95, E12 pooled model 6.51/2.91, width-2048
  table 16.46/2.20, V3 transfers 5.13, 2.54, 3.81, 5.44. Deterministic (two runs identical).
- Captured calls, calibration-only selectors (mean/p95 %, tied share) PRO6000 | 3090 | H800:
  cost model 0.88/4.0 (.90) | 3.25/15.4 (.69) | 3.54/14.1 (.79)
  three-param restriction 4.48/50.1 | 2.49/13.1 (.78) | 4.13/17.4
  nearest key 5.24/16.0 | 4.54/20.5 | 14.38/78.3
  log-linear interpolation (log L, log B) 3.88/12.8 | 4.16/19.4 | 14.34/78.3
  length tree 2.67/10.7 | 9.87/53.0 | 19.92/78.3
  gradient boosting basic/structural 3.05/1.26 | 6.45/6.18 | 9.28/6.26
  MLP basic/structural 9.63/3.95 | 5.69/4.15 | 8.89/4.79
  TFLA fitted 12.70/88.1 | 7.62/31.8 | 10.58/67.4; TFLA literal 106.0 | 17.7 | 152.1
  ("structural" = learner also gets Eq. (5) derived quantities: occupancy, waves, ...)
- TFLA: App. G.1 no-overlap runtime with App. F per-chunk byte/FLOP counts; every channel a head
  (d_hv=1), d_qk=16, fp16 activations, fp32 state, F_causal=0.5, ceil(L/C) chunks; alpha, beta fitted on
  the 90 default-launch calibration timings; LS optimum alpha -> inf on all GPUs (data reject the C^2
  FLOP term; memory term only). Datasheet ratios 64-256 FLOP/byte pick chunk 32 always (148/29/76%).
- M5a clean selection over 12 forms (target GPU mean/p95): calibration-only leave-one-shape-out:
  PRO6000 picks drop-shuffle 1.34/5.7 (two exactly tied alternatives score 0.88); 3090 picks
  drop-L1-miss 3.61/19.0 (tied alternative 3.25); H800 picks drop-scan-work 7.25/62.5. Adding the
  three-param restriction as candidate changes only the 3090 pick (2.49).
  Selected on another GPU's calls: PRO6000 0.88 (chosen on 3090), 4.80 (on H800); 3090 3.72 (on PRO6000),
  12.72 (on H800); H800 3.74 (on PRO6000), 3.54 (on 3090); chosen on the other two together:
  2.68 / 4.52 / 3.54.
  -> calibration-only selection is unstable; leave-one-shape-out often cannot separate forms.
- M6d top-k (mean/p95/p99): top-2 0.52/1.79/10.8 | 2.30/13.4/28.7 | 0.73/1.90/13.3;
  top-3 0.36/1.21/5.0 | 2.06/12.7/27.9 | 0.72/1.85/13.3. Max regret does not drop.
- Shard widths (126 pooled shard shapes; 3090 | H800): model 6.51/30.5 | 2.91/13.1; three-param
  6.14 | 6.02; GB structural 6.53 | 3.41; TFLA 12.92 | 3.32; interpolation 21.8 | 3.53; nearest key
  22.2 | 3.66; width-2048 table 16.46 | 2.20; top-2 3.00/15.9/25.2 | 0.22/0.60/5.5; top-3 1.92/12.4/18.9 |
  0.03/0.01/0.73.
- Cross-GPU transfer of the model: H800->PRO6000 0.87, H800->3090 4.70; three-param restriction to
  3090 2.48; learned regressors 3.5-23.
- Learned regressors swing 0.47-9.6% across tags; none wins consistently. MLP trained with Adam,
  300 epochs (L-BFGS too slow).
