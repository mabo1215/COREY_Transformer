# TPDS revision experiments (E1-E5)

Design: `docs/Design.md` (cycle tpds-r1). Review items: `docs/revision_suggestions.tex` section C.

All timing uses recurrence-preserving kernels: one call processes the whole
sequence and carries the recurrent state across chunks; the chunk is a real
tiling parameter of that call. The old chunk-splitting harnesses under
`src/experiments/run_w1_*`, `run_calibrated_chunk512_3090.py` and
`run_cuda_profile_three_policies.py` are not used.

| File | Purpose |
|---|---|
| `kernels.py` | Triton Mamba-1 chunked selective scan (chunk = time tile BLOCK_T; tunable BLOCK_D, num_warps, num_stages), wrappers for Mamba-2 SSD `mamba_chunk_scan_combined` (chunk_size) and Gated DeltaNet `chunk_gated_delta_rule` (chunk_size in {16, 32, 64}), fp32 references, validation. `python -m src.experiments.tpds.kernels` runs the self-test. |
| `common.py` | env metadata, resumable JSONL/CSV writers (flush + fsync, skip completed keys), disk guard, CUDA-event timing, Eq. (2) |
| `corpus.py` | mixed-regime corpus (extracted from `run_mixed_regime_discovery.py`) + LongBench prompts (`src/data/longbench_push120`) |
| `capture_traces.py` | per-layer scan inputs during prefill of `state-spaces/mamba-370m-hf` and `AntonV/mamba2-370m-hf` |
| `e1_value_invariance.py` | E1 |
| `e2_oracle_sweep.py` | E2 (per-call CSV) |
| `e3_predictors.py` | E3 (CPU) |
| `e4_autotune.py` | E4 incl. isolated entropy-decision overhead |
| `e5_breakeven.py` | E5 (CPU) |
| `analysis.py` | shared regret / mutual-information helpers |
| `make_figures.py` | `tpds_*.pdf` figures (IEEE column width), works on partial results |

Remote lanes: `src/experiments/remote/tpds_prep.sh` (CPU prep), `tpds_lane_a.sh`
(E1 + E4), `tpds_lane_b.sh` (capture + E2, then E3/E5/figure previews).

## Install (repeat on every host)

Only pure-Python/Triton parts of the libraries are needed. `--no-deps` keeps
the host's torch/triton untouched.

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

`mamba_ssm/__init__.py` imports the compiled `selective_scan_cuda`, which does
not exist with `MAMBA_SKIP_CUDA_BUILD=TRUE`. `kernels._load_ssd()` therefore
registers a namespace stub for the top-level `mamba_ssm` package and imports
only `mamba_ssm.ops.triton.ssd_combined` (pure Triton). A full install with
the CUDA extension is used unchanged. The stock mamba_ssm CUDA selective scan
(the design's optional reference point) is not built.

Local smoke host (WSL2, RTX 3070): system python 3.12 is PEP-668 managed, so a
venv with `--system-site-packages` was used; the commands above were run with
that venv's python. `tpds_prep.sh` performs the same steps (`PIP_EXTRA` passes
extra pip flags) plus the checkpoint download and static checks.

## Running

From the repository root (`PYTHONPATH=.`):

```bash
python -m src.experiments.tpds.kernels                       # correctness self-test
python -m src.experiments.tpds.capture_traces --model mamba1  # then --model mamba2
python -m src.experiments.tpds.e1_value_invariance
python -m src.experiments.tpds.e2_oracle_sweep --model mamba1 # and --model mamba2
python -m src.experiments.tpds.e4_autotune
python -m src.experiments.tpds.e3_predictors
python -m src.experiments.tpds.e5_breakeven --host <tag>
python -m src.experiments.tpds.make_figures --host <tag> --out-dir paper/figs
```

On the PRO 6000 use the lanes:

```bash
wsl bash src/revloop/remote.sh exec   pro6000 src/experiments/remote/tpds_prep.sh
wsl bash src/revloop/remote.sh launch pro6000 tpds-a src/experiments/remote/tpds_lane_a.sh
wsl bash src/revloop/remote.sh launch pro6000 tpds-b src/experiments/remote/tpds_lane_b.sh
wsl bash src/revloop/remote.sh watch  pro6000 E1 E2 E3 E4 E5
```

The two lanes run concurrently but every GPU step holds an exclusive `flock`
(`src/logs/tpds/gpu_timing.lock`), so no kernel timing ever overlaps another
GPU job; set `TPDS_NO_GPU_LOCK=1` only for debugging.

## Outputs

`src/experiments/results/tpds/`: `e1_<host>.json(l)`, `e2_<model>_<host>.csv/json`,
`e3_<model>_<host>.json`, `e4_<host>.json(l)`, `e5_<host>.json`. The `.json`
files contain every row, so the ledger globs (`e1_*.json`, `e4_*.json`) capture
the complete data; `.jsonl` files are the resumable raw logs. Captured traces
go to `src/outputs/tpds_traces/` (git-ignored, size-guarded by `--max-gb`).
Logs: `src/logs/tpds/`; lane markers: `src/logs/revloop/<ID>.{START,OK,FAIL}`.
Smoke outputs: `src/experiments/results/tpds/smoke/`.

## Notes

- Mamba-1 launch configuration when only the chunk varies (E1/E2):
  `kernels.DEFAULT_MAMBA1_LAUNCH` (per-chunk BLOCK_D, num_warps). E4 searches
  the full joint space.
- Capture replaces each mixer's prefill forward by an equivalent forward that
  calls the model's own submodules plus the validated scan kernel; the logits
  are checked against the stock transformers forward before capture
  (recorded in the manifest).
- Mamba-1 traces store u, delta (pre-softplus, dt_proj bias included), B, C and
  the gate z; Mamba-2 traces store x, dt (pre-bias), B, C. All fp16.
- Gated DeltaNet has no checkpoint in this study; it appears in E1 with
  synthetic inputs only (370M-class shape: 4 heads, d_k=256, d_v=512).
