# taco-r2: positioning and missing related work (M9a--M9d, M3c sources)

Prepared 2026-10-16 for revision cycle taco-r2.

- **Files changed.** Only `paper/ref.bib` was edited: 30 new entries were
  appended at the end of the file, in three commented blocks. No existing
  entry was changed, and `main.tex` and `appendix.tex` are untouched.
- **Syntax check.** BibTeX with `ACM-Reference-Format`, citing all 30 new
  keys, ran without errors. The only warnings are "empty publisher/address"
  and "page numbers missing" (ICLR), the same kinds the existing entries
  already produce.
- **Citation commands.** The drop-in LaTeX uses `\cite` and `\citet`, as the
  manuscript does (acmart, `\citestyle{acmauthoryear}`).
- **Line references.** `main.tex` was being edited by another agent while
  this note was written, so locations are given by the quoted anchor text.
  Line numbers are approximate, as of the afternoon of 2026-10-16.

## 1. Added bibliography keys (all verified)

Sources: Crossref DOI records, publisher and proceedings pages, OpenReview,
arXiv, official documentation, and source code at a pinned commit or release.
dblp blocked automated access and was not used.

### M9a: static and predictive pruning; occupancy and latency modelling

| Key | Verified source |
|---|---|
| `ryoo2008pruning` | Crossref 10.1145/1356058.1356084; CGO 2008, pp. 195--204 |
| `lim2017autotuning` | Crossref 10.1109/ICPP.2017.61; ICPP 2017, pp. 523--532 (middle initials from arXiv 1701.08547) |
| `bergstra2012predictive` | Crossref 10.1109/InPar.2012.6339587; InPar 2012, pp. 1--9 (IEEE metadata; the author PDF has 6 pages) |
| `falch2017mlautotuning` | Crossref 10.1002/cpe.4029; CCPE 29(8), e4029, 2017 |
| `balaprakash2018autotuning` | Crossref 10.1109/JPROC.2018.2841200; Proc. IEEE 106(11), 2068--2083 |
| `kaufman2021learned` | proceedings.mlsys.org page and PDF; MLSys vol. 3, pp. 387--400 (MLSys has no DOI, so the entry carries a `url`) |
| `volkov2010lower` | NVIDIA-hosted GTC 2010 slides, title slide dated September 22, 2010 |
| `hong2010integrated` | Crossref 10.1145/1815961.1815998; ISCA 2010, pp. 280--289 |
| `liu2021gptune` | Crossref 10.1145/3437801.3441621; PPoPP 2021, pp. 234--246 |
| `wu2020ytopt` | Crossref 10.1109/PMBS51919.2020.00012; PMBS 2020, pp. 61--70. This is the first peer-reviewed paper describing ytopt (listed in the ytopt README; no CITATION file) |
| `hellsten2023baco` | Crossref 10.1145/3623278.3624770; ASPLOS 2023 vol. 4, pp. 19--42 |

### M9b: shape-keyed tuning in production stacks

| Key | Verified source |
|---|---|
| `ansel2024pytorch2` | Crossref 10.1145/3620665.3640366; ASPLOS 2024 vol. 2, pp. 929--947, 49 authors |
| `hong2024flashdecodingpp` | proceedings.mlsys.org; MLSys vol. 6, pp. 148--161. The proceedings title differs from the arXiv title ("... with Asynchronization, Flat GEMM Optimization, and Heuristics"). Sec. 5 profiles three GEMM implementations offline per weight shape and picks one at run time by the token count M |
| `khan2019miopen` | arXiv 1910.00078 (15 authors). The paper describes a Find step whose result "may be used in subsequent invocations" and tuned parameters "serialized ... for future retrieval". The names FindDb/PerfDb appear only in the MIOpen documentation, so the drop-in text does not use them |
| `zheng2024sglang` | proceedings.neurips.cc BibTeX and Crossref; NeurIPS 37, pp. 62557--62583, DOI 10.52202/079017-2000 |
| `pytorchcompiledocs` | PyTorch 2.14 `torch.compile` page: "max-autotune ... leverages Triton or template based matrix multiplications ... It enables CUDA graphs by default on GPU"; `max_autotune` "will profile to pick the best matmul configuration" |
| `pytorchdynamicshapes` | PyTorch 2.14 dynamic-shapes page: automatic dynamic recompiles with the changed dimensions marked dynamic; "we specialize automatically on inputs with sizes 0 or 1" |
| `pytorchcudnnbenchmark` | PyTorch 2.14 reproducibility notes: "When a cuDNN convolution is called with a new set of size parameters ... benchmarking them to find the fastest one ... used consistently ... for the corresponding set of size parameters" |
| `tensorrttimingcache` | TensorRT 11.4.0 documentation, "Timing Cache": the cache is "specific to the targeted device, CUDA, TensorRT versions, and BuilderConfig parameters" and "can be serialized and deserialized" |
| `vllmfusedmoeconfigs` | vLLM repository at commit 4ac1f8d: per-device JSON files keyed by token count M; `try_get_optimal_moe_config()` picks the nearest key |
| `sglangfusedmoeconfigs` | SGLang repository at commit a2981ab: same scheme, per Triton version, nearest-M lookup. The directory moved; the old path `fused_moe_triton/configs` exists only up to about v0.5.5 |

### M9d: background

| Key | Verified source |
|---|---|
| `blelloch1990prefix` | CMU catalogue page; CMU-CS-90-190, November 1990 |
| `martin2018parallelizing` | OpenReview HyUNwulC-; ICLR 2018 (poster) |
| `merrill2016singlepass` | NVIDIA Research page and PDF; NVR-2016-002, March 2016 |
| `smith2023s5` | OpenReview Ai8Hw3AXqks; ICLR 2023 |
| `sinha2022notall` | Crossref 10.1109/SC41404.2022.00070; SC22, pp. 1--15. The abstract reports "8% (max 22%) average performance variation" across same-SKU GPUs, attributed to manufacturing variability and power management |

### M3c: engine execution modes (`@misc`, accessed 2026-10-16)

| Key | Verified source |
|---|---|
| `vllmcompilationconfig` | vLLM v0.31.0 `CompilationConfig` docs. `FULL_AND_PIECEWISE` is the V1 default ("full cudagraph for decode batches and piecewise cudagraph for prefill and mixed prefill-decode batches"). `splitting_ops` defaults to the attention ops, which include `vllm::mamba_mixer2`, `vllm::mamba_mixer`, `vllm::short_conv`, `vllm::linear_attention` and `vllm::qwen_gdn_attention_core`. `max_cudagraph_capture_size` "is capped at 512 by default, or 1024 on data center Blackwell GPUs" |
| `vllmschedulerconfig` | vLLM v0.31.0 `SchedulerConfig.async_scheduling`: "Async scheduling helps to avoid gaps in GPU utilization". The v0.31.0 source enables it by default for generative models when no incompatible option is set. The docs do not say this, so the drop-in text does not claim it is the default |
| `sglang2026cudagraph` | LMSYS Org blog, "Advanced CUDA Graph Techniques in SGLang", SGLang Team, 2026-08-17: "For prefill, Breakable CUDA Graph is now SGLang's default"; prefill is padded to captured token buckets, and marked operators run eagerly between segments |
| `sglang2026release` | SGLang release v0.5.21 (2026-10-02), source code. On CUDA the prefill default is the breakable backend, whose docstring says it "break[s] at attention / mamba boundaries". The linear-attention layer (used by Qwen3-Next's Gated DeltaNet) and the Nemotron-H Mamba-2 mixer are wrapped as graph breaks |

The existing entry `vllmcudagraphs` (vLLM design document "CUDA Graphs") was
re-checked and is reused. It states:
- `FULL_AND_PIECEWISE` is the default;
- "Mamba attention" supports CUDA graphs only for `UNIFORM_SINGLE_TOKEN_DECODE`;
- without a matching captured key the dispatcher will "default to return
  `NONE` mode for eager execution".

### Unverified or not added

- **Unverified:** none. Every requested work was verified and added.
- **Sinha et al.:** the SC22 article number (possibly 65) could not be
  confirmed, so the entry gives pages 1--15 only.
- **MIOpen docs:** not added as separate entries; the drop-in text is
  worded to match the 2019 paper.
- **Hong and Kim 2010:** this is the integrated *power* and performance
  model. The occupancy/latency model the reviewer means for Eq. (3) is
  their 2009 paper, already cited as `hong2009analytical`. Cite the 2010
  paper only next to it.
- **For the bibliography minor item (not applied, because existing entries
  were left unchanged):** `beck2025tfla` now has proceedings data: NeurIPS
  38, DOI 10.52202/085713-2522. Pages conflict between sources:
  proceedings.neurips.cc gives 75093--75148, while Crossref gives
  83630--83685 for the same DOI. Add the DOI only, or the proceedings pages.

## 2. Drop-in paragraphs

Line estimates assume acmsmall at about 92 characters per line, with
author-year citations counted at their printed length.

### (a) Static and predictive pruning of GPU autotuning (M9a; M4 "position the model as a pruning case study")

**Where:** Section 2, a new paragraph directly after "Kernel auto-tuning and
tuning caches". It replaces the "Positioning" paragraph (about main.tex:343,
"Shape-keyed caches, measured verification of model-ranked candidates ...").
In Section 6 "Modelling versus profiling", extend the analytical-model
citation ("The model belongs to the analytical GPU models that combine
occupancy, latency hiding and issue throughput~\cite{...}", about
main.tex:1059) to
`\cite{hong2009analytical,volkov2010lower,volkov2016understanding,williams2009roofline}`.
That also gives M4d its references for a max(latency, throughput) form.
**Adds:** about 11 lines; net about +7.5 after the Positioning paragraph goes.

```latex
\paragraph*{Pruned and model-ranked tuning}
Our hybrid (compile every candidate without running it, read the compiler's
resource report, rank with a model, time the best $k$) has close
precedents. \citet{ryoo2008pruning} timed only the configurations on the
Pareto curve of efficiency and utilization metrics computed from the
compiled code of CUDA kernels; \citet{lim2017autotuning} prune an
autotuner's search from the instruction mix and occupancy of compiled,
unexecuted variants; \citet{falch2017mlautotuning} time only the
configurations that a learned model ranks best,
\citet{bergstra2012predictive} search a boosted-tree model instead of the
hardware, and \citet{kaufman2021learned} rank tile sizes and fusions with a
learned model inside a compiler's autotuner; Bayesian tuners such as GPTune,
ytopt and BaCO reduce the number of
timings~\cite{liu2021gptune,wu2020ytopt,hellsten2023baco}, and
\citet{balaprakash2018autotuning} survey the field. Our cost model is a
ranking model for this kind of top-$k$ pruning, which Triton's autotuner
also offers~\cite{tritondocs}, not a new tuning method
(Section~\ref{sec:res_model}).
```

**What the sources support:**
- Ryoo et al. compute "Efficiency" and "Utilization" from the PTX and cubin
  resource usage and time only the Pareto set. Their abstract reports up to
  98% pruning.
- Lim et al. use PTX, `ptxas -v` and the occupancy calculator to prune
  Orio's search, which "does not require any program runs".
- Falch and Elster time the configurations with the lowest predicted time.
- Bergstra et al. hill-climb on the model itself, not on the hardware.
- Kaufman et al. integrate a GNN model into an XLA autotuner for TPU
  tile-size selection and fusion.

The "what is specific" sentence is not repeated here: Section 6 already says
that "the chunk enters only through the compiler's resource report and the
tile layout, so an untimed shape can be predicted".

### (b) Shape-keyed tuning in production stacks (M9b; DA C17)

**Where:** Section 2, a new paragraph after (a). Remove the sentence "FFTW
stores the plans it estimated or measured as wisdom ... All of them key the
result on shape and device, not on data values." from "Kernel auto-tuning and
tuning caches" (about main.tex:274--277), because (b) re-cites both works.
The last sentence of (b) leans on the Introduction paragraph "What was not
known before ..." (about main.tex:173), which already states the new findings.
**Adds:** about 13.5 lines; net about +10.

```latex
\paragraph*{Shape-keyed tables in production stacks}
Deployed stacks already keep such tables: FFTW stores measured plans as
wisdom~\cite{frigo2005fftw3}, MIOpen stores the result of benchmarking its
kernels for a problem configuration for later calls~\cite{khan2019miopen},
cuDNN's benchmark mode times the convolution algorithms for each new set of
sizes~\cite{pytorchcudnnbenchmark}, TensorRT keeps a serializable layer-timing
cache specific to the device and software versions~\cite{tensorrttimingcache},
and Kernel Launcher serves tuned configurations by device and problem
size~\cite{heldens2023kernellauncher}. In LLM stacks, PyTorch~2 specializes
sizes 0 and 1, recompiles with dynamic dimensions when shapes change, and in
its max-autotune mode profiles matrix-multiplication configurations and
enables CUDA graphs~\cite{ansel2024pytorch2,pytorchcompiledocs,pytorchdynamicshapes};
FlashDecoding++ selects among GEMM implementations by the token count from an
offline-profiled table~\cite{hong2024flashdecodingpp}; and vLLM and SGLang
ship per-GPU Triton configurations of their fused mixture-of-experts kernels,
keyed on the token count~\cite{vllmfusedmoeconfigs,sglangfusedmoeconfigs}.
All of them key on shape and device, not on values. Our recipe applies this
practice to the chunk; our findings say what its key must contain and where
it pays.
```

**Notes:**
- The reviewer's phrase "per-shape CUDA graphs" for TorchInductor was not
  verified in the documentation and is not used.
- vLLM's fused-MoE lookup takes the nearest token-count key. This is the
  production twin of the length-bucket key the paper criticizes; it could be
  mentioned in the Discussion if space allows.

### (c) TFLA: what the paper adds (M9c)

**Where:** Section 2, "How chunk size is chosen". Replace "Tiled Flash Linear
Attention (TFLA)~\cite{beck2025tfla}, the closest analysis to ours, derives
the chunk ... rather than derive it to design one." (about main.tex:241--248)
with the paragraph below. Keep the opening clause on the per-kernel defaults
and the closing sentence on the fused Mamba-2 pipeline.
**Adds:** about 12.5 lines in place of about 7; net about +5.5.

**Companion change (M9c):** the TFLA-on-scan baseline goes to the
supplement. Delete "TFLA expression" from the row list of the Section 6
selector table (`tab:baselines`, the TODO row at about main.tex:1083); it
stays in Supplementary Table `tab:supp_baselines`.

```latex
Tiled Flash Linear Attention (TFLA)~\cite{beck2025tfla}, the closest analysis
to ours, models a chunkwise-parallel kernel as
$N_{\mathrm{batch}}N_{\mathrm{head}}\,(L/C)\,\bigl(\mathrm{Bytes}(C)/\beta+\mathrm{FLOPs}(C)/\alpha\bigr)$
(our notation; peak rates $\alpha$ and $\beta$, no overlap of computation and
memory traffic, bytes linear and FLOPs quadratic in $C$), whose optimum grows
with the square root of the head dimension and with the arithmetic intensity
of the accelerator. Multiplied out, this is the work term
$BL(a+bC)+B\lceil L/C\rceil g$ of our model of the Mamba-2 SSD and Gated
DeltaNet pipelines (Section~\ref{sec:res_regime}), with coefficients from
operation counts and peak rates instead of per-GPU fits, with $L/C$ instead of
$\lceil L/C\rceil$ chunks, so that its optimum $\sqrt{g/b}$ does not depend on
the length, and without the cost of launching its several kernels, which in
our model is the floor below which every chunk ties. Beyond TFLA we measure
the regime, the movement of the optimum with length and batch (on TFLA's own
mLSTM kernel from 128 at batch 1 to 32 at most batched shapes, which its
expression does not follow even when fitted; Supplementary
Section~\Sref{sec:supp_tfla}), and a scan whose chunk acts through registers,
spills and occupancy, outside its model class.
```

**Verification and algebra:**
- Checked against arXiv 2503.14376v3, accepted at NeurIPS 2025.
  - Appendix G.1, Eqs. 104--108: runtime = N_batch N_head (T/L)
    (Bytes/beta + FLOPs/alpha). The paper notes that "we assume that memory
    operations are not overlapped with computation".
  - Section 5.3: "The runtime-optimal chunk size scales proportionally with
    the square root of the head dimension"; it rises on newer GPUs.
  - Appendix G.1: "we do not model the fact that our computation consists of
    multiple kernels. Hence, we do not include the delay when multiple
    kernels are launched in sequence."
- Algebra. Take Bytes = b1 C + b0 and FLOPs = f2 C^2 + f1 C + f0. Then
  (L/C)(Bytes/beta + FLOPs/alpha)
  = L (f2/alpha) C + L (b1/beta + f1/alpha) + (L/C)(b0/beta + f0/alpha).
  This is BL(a + bC) + B(L/C)g with a = b1/beta + f1/alpha, b = f2/alpha and
  g = b0/beta + f0/alpha, with the heads absorbed into the coefficients.
  The comparison M9c asks for is therefore a statement of identity plus the
  three differences, which needs no new numbers.
- Optional numbers. If wanted, one sentence can report the regret of the
  fitted TFLA form (literal L/C, no floor) on the captured SSD and Gated
  DeltaNet calls, eager and under graphs, from an offline re-fit of the
  existing calibration timings. That re-fit has not been run; do not add
  numbers without it.
- The mLSTM numbers in the paragraph are those of the earlier main text
  ("chunk 128 at every batch-1 shape, 32 at 30 of the 42 batched ones";
  "TFLA's own runtime expression, fitted on these timings ($R^2=0.998$),
  selects chunk 128 for every shape"). That passage now lives in
  Supplementary Section `sec:supp_tfla`.

### (d) Background fixes (M9d)

**(d1) Parallel scans.**
- **Where:** Section 2, "Chunked linear-recurrence kernels", directly after
  the first sentence ("... in a fused kernel that walks the sequence in
  chunks and carries the state between them.", about main.tex:193).
- **Adds:** about 6 lines.

```latex
A first-order linear recurrence is a prefix scan under an associative
operator and can be evaluated in logarithmic depth~\cite{blelloch1990prefix},
which parallelizes linear RNNs over the sequence~\cite{martin2018parallelizing}
and the S5 layer~\cite{smith2023s5}. GPU scans process the sequence in tiles
and pass each tile's aggregate to its successor, between thread blocks in
single-pass scans with decoupled look-back~\cite{merrill2016singlepass}; the
kernels studied here carry the state from chunk to chunk inside one program
or through a separate state-passing kernel (Table~\ref{tab:taxonomy}).
```

**(d2) Measurement variability.**
- **Where:** Section 2, "Performance models and measurement", last sentence
  (about main.tex:326--328). Replace the final "... are why we test value
  invariance with placement controls rather than assume it." with the text
  below, which continues the same sentence.
- **Adds:** about 2.5 lines.

```latex
... are why we test value invariance with placement controls rather than
assume it; GPUs of one model also differ by 8\% on average, and up to 22\%, in
large clusters under power management~\cite{sinha2022notall}, one reason to
key a table by device and to invalidate it when the power limit changes.
```

**(d3) Occupancy and latency modelling.**
- No new sentence is needed; see the citation extension under (a).
- `hong2010integrated` was added because the review lists it. Cite it next
  to `hong2009analytical` only where power is discussed (e.g. the
  power-limit invalidation in the recipe), or leave it uncited.
- An uncited entry does not appear in the reference list, so leaving it
  uncited is harmless.

### (e) Engine execution modes (M3c)

**Main location:** Discussion, "Engines and graph capture" (about
main.tex:1137). Replace the TODO block "\textbf{[TODO: M3c: cited description
of vLLM V1 ... launch cost.]}" with the paragraph below, which follows the
sentence "Whether a deployment is in the device-bound regime depends on the
engine and its mode."
**Adds:** about 13 lines in place of the TODO.

**Introduction sentence:** "given that library defaults exist and that
engines run prefill either eagerly, with a host-side launch cost per kernel,
or captured as CUDA graphs?" (about main.tex:114) still states the general
claim without a source. Replace it with "given that library defaults exist
and that engines launch the recurrent mixer of a prefill step from the host,
between or outside captured CUDA
graphs~\cite{vllmcudagraphs,sglang2026cudagraph}?". This takes the same
space.

```latex
In vLLM's V1 engine the default CUDA-graph mode captures full graphs only for
uniform decode batches and piecewise graphs for prefill and mixed batches; the
operators between the pieces, which by default include the Mamba, Mamba-2,
linear-attention and Gated DeltaNet mixers, run eagerly, the Mamba backend
supports full graphs only for uniform single-token decode, and batches above
the largest captured size (at most 512 tokens by default, 1024 on data-centre
Blackwell GPUs) run without graphs~\cite{vllmcudagraphs,vllmcompilationconfig}.
SGLang~\cite{zheng2024sglang} now captures prefill by default as breakable
graphs padded to token buckets, with marked operators run eagerly between the
captured segments~\cite{sglang2026cudagraph}; its Gated DeltaNet
(linear-attention) and Nemotron-H Mamba-2 layers are such
breaks~\cite{sglang2026release}. In both engines the mixer of a prefill step
therefore runs outside the captured graph with its kernels launched from the
host, and it is device-bound only when the host runs ahead of the GPU, which
the captured segments around it and asynchronous
scheduling~\cite{vllmschedulerconfig} favour. Short mixer calls arise from
short prompts, chunked-prefill slices and speculative-decoding verification.
```

**Notes:**
- **What the sources say.**
  - vLLM `FULL_DECODE_ONLY` runs mixed prefill-decode batches "without
    cudagraphs".
  - The vLLM capture sizes are `[1, 2, 4] + range(8, 256, 8) +
    range(256, max+1, 16)`.
  - Speculative-decoding verification batches (1 + k tokens per request)
    count as "uniform decode" in vLLM. They are full-graph only for
    backends with `UNIFORM_BATCH` support, which excludes the Mamba backend.
- **SGLang changes fast.**
  - The blog's timeline dates breakable prefill graphs as the default to
    2026-04-24 (PR #22218).
  - In v0.5.21 the linear-attention break is unconditional.
  - On main of 2026-10-16 a backend can instead keep linear-attention
    prefill inside the graph.
  - Hence the paragraph cites the release, not main. It names only the two
    layer types verified as breaks; other SGLang Mamba-2 models were not
    checked.
- **Not drafted here.**
  - The regime rename, the harness's per-call launch overhead and the "C++
    launch paths" point belong to the Conventions table and Section 3 and
    need the paper's own measurements.
  - Example clause, if the author wants it: "kernels launched from C++, such
    as the CUDA selective scan, pay a smaller host-side cost per call". Use
    it only with the paper's own floor numbers as evidence.

## 3. Page budget: what to shorten

**Additions.**

| Item | Lines |
|---|---|
| (a) | +7.5 |
| (b) | +10 |
| (c) | +5.5 |
| (d1) | +6 |
| (d2) | +2.5 |
| (e) | +13 |
| **Total** | **about +45 (about one page)** |

The Introduction change is space-neutral.

**Suggested compensation.** The reviewers asked for a compressed Section 2
(M8b, "compress the citation lists of Sec. 2"); every moved citation stays in
the reference list through the supplement, which uses the same `ref.bib`.

| # | Cut | Saves |
|---|---|---|
| 1 | "Chunked linear-recurrence kernels": move "The family keeps growing (RWKV-7, DeltaProduct, Mamba-3, log-linear memory) and runs in production hybrids (MiniMax-01, Nemotron-H, Jamba, Qwen3-Next); ThunderKittens and FlashRNN tile recurrences on GPUs, and LASP-2 parallelizes linear attention by sequence" to Supplementary "Further Related Work", keeping the pointer | about 6 lines |
| 2 | "Kernel auto-tuning and tuning caches": compress the tool enumeration (ATLAS, CLTune, Kernel Tuner, ATF, OpenTuner, Active Harmony, KTT) into one clause, e.g. "offline and online empirical autotuners~\cite{whaley1998atlas,nugteren2015cltune,vanwerkhoven2019kerneltuner,ansel2014opentuner,tapus2002activeharmony}" (survey now in (a)). Move the TenSet/TLP, Transfer-Tuning, Nimble/DietCode and XLA/nvFuser/FlashAttention/TileLang sentences to the supplement | about 10 lines |
| 3 | "Performance models and measurement": keep Hong and Kim, Volkov, roofline and the compile-informed models that Section 6 contrasts (Sim et al., GPUMech, Habitat, tritonBLAS). Move Baghsorkhi, Zhang, Konstantinidis, CDMPP, NeuSight, Roller, Stream-K, CRAT and Kayiran et al. to the supplement | about 8 lines |
| 4 | Introduction "What was not known before ..." (12 lines): (a)--(c) now carry the positioning, so cut it to its claims (activations carry no information; the key holds the specialization bits; the effect is a property of the regime; profiling versus modelling) without the Triton/Kernel Launcher/FFTW/TFLA preamble | about 6 lines |
| 5 | "Input-aware variant selection" (12 lines): merge the Nitro / input-sensitive / input-aware autotuning clauses into one cited clause | about 4 lines |
| 6 | Companion cut of (c): TFLA row out of `tab:baselines` | about 1 line |

Cuts 1--6 save about 35 lines, so the net growth of the main text is about
+10 lines.

**If more space is needed:**
- Move (d1) to the supplement's "Further Related Work" with the four scan
  citations (the reviewer asked for background, not for main-text space).
  Saves 6 lines.
- Drop the last sentence of (e). Saves 1.5 lines.

With both, the net change is about +2 lines.
