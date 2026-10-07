# Field Analysis and Reviewer Configuration (independent review round tpds-r4, 2026-10-07)

Manuscript: "Shape Sets the Chunk: A Compile-Informed Cost Model and a No-Gain Bound for
Data-Dependent Dispatch of Chunked Linear-Recurrence Kernels" (main body 14 pp + supplement 16 pp,
built 2026-10-07 from paper/main.tex and paper/appendix.tex).

Target venue: IEEE Transactions on Parallel and Distributed Systems (regular paper, IEEEtran
journal class, 14 formatted pages, double-anonymous review, supplemental material permitted).

Primary discipline: GPU systems / performance modelling of ML kernels (auto-tuning, kernel dispatch).
Secondary discipline: sequence-model kernels (selective state space models, linear attention) and
LLM serving (continuous batching, chunked prefill, tensor parallelism).
Research paradigm: empirical systems measurement plus an analytical cost model and a formal bound.
Methodology type: kernel micro-benchmarks on three GPUs, captured real workloads, regression-based
cost model, statistical tests, trace replay.
Paper maturity: resubmission-quality (third internal revision cycle complete).

Reviewer panel (five independent reviewers, no shared notes):

| Role | Persona | Focus |
|---|---|---|
| EIC | TPDS associate editor with an HPC/GPU-systems background | Journal fit (parallel/distributed dimension), significance, positioning, venue compliance (MUST FIX items) |
| R1 Methodology | GPU performance-measurement and statistical-modelling expert | Measurement protocol, statistical validity, cost-model fitting/validation protocol, soundness of the no-gain bound, number consistency, reproducibility |
| R2 Domain | Researcher on SSM / linear-attention kernels and auto-tuning | Novelty versus prior GPU cost models and input-aware tuning, correctness of claims about production kernels, missing references, incremental contribution |
| R3 Perspective | LLM-serving / distributed-inference systems practitioner | Practical value for serving systems and heterogeneous fleets, deployability, generality beyond Mamba-1, broader implications |
| Devil's Advocate | Adversarial reviewer | Strongest counter-arguments: triviality of the bound, over-fitting of the cost model, whether the chunk matters on production kernels, cherry-picking, over-generalisation, "so what" |

Each reviewer scores Novelty (20%), Methodology (25%), Experiments (25%), Clarity (15%),
Significance (15%) on 0-100 and gives an independent recommendation. Reports are written to
docs/review_reports/tpds_r4_<role>.md and synthesised afterwards.
