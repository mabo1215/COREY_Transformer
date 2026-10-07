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
