"""Build the Code Ocean capsule (submodule ``codeocean/``) from this repository.

The capsule gets only what the paper needs: the entry scripts listed in ENTRIES, every
local module they import (resolved from the source, so the package is complete), the
TPDS result files listed in RESULTS_MANIFEST (no legacy summaries, raw per-round logs or LongBench
prompt subsets). Everything else in the capsule is replaced. The package is written for
anonymous review: the license holder is anonymized, host names and paths in result metadata
are neutralized (SANITIZE), the project's former name is removed from file names, code and
text (RENAME), and a scan for identifying strings must pass before anything is committed.
The README maps every table and figure of the paper to its result files and scripts
(PAPER_MAP); the numbers are read from the built paper (paper/build/*.aux), so build the
paper first. The capsule metadata (title, authors) and environment must be edited in the
Code Ocean UI; this script never touches metadata/ or environment/ (scan hits there are
only reported).

Usage (repository root):
    python src/revloop/capsule_build.py            # rebuild codeocean/ in place, scan, no commit
    python src/revloop/capsule_build.py --commit   # also commit inside codeocean/ (anonymous identity)
    python src/revloop/capsule_build.py --push     # commit and push to the capsule's git remote

The Code Ocean environment (``environment/``) is left as it is; the capsule's run script
recomputes analyses, tables and figures on the CPU from the stored measurements and needs
NumPy, pandas, SciPy, Matplotlib and an importable PyTorch/Triton (several analyses import the
kernel modules); scikit-learn is optional (faster, exact predictor trees).
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAPSULE = ROOT / "codeocean"
CREDENTIAL_HELPER = ROOT / "src" / "revloop" / "codeocean_credential.sh"

TITLE = ("Shape Sets the Chunk: A Compile-Informed Cost Model and a No-Gain Bound for "
         "Data-Dependent Dispatch of Chunked Linear-Recurrence Kernels")
DESCRIPTION = ("Code, measurements and analysis scripts for a study of chunk selection in chunked "
               "linear-recurrence GPU kernels (Mamba-1 selective scan, Mamba-2 SSD, Gated DeltaNet): "
               "a compile-informed cost model, a no-gain bound for data-dependent dispatch, and "
               "key-churn analysis. The reproducible run recomputes analyses, tables and figures "
               "from the stored measurements on the CPU.")

# Entry scripts of the experiments reported in the paper and its supplement.
ENTRIES = [
    "src/experiments/tpds/*.py",                       # kernel-level study, analyses, tables, figures
    "src/corey_selective_scan_dispatch.py",            # loaded by name by the routed study (e10)
]
# Code shipped with the entries besides their imports.
EXTRA = [
    "src/kernels/mamba1_runtime_chunk/**/*",          # reference CUDA scan with a runtime chunk (source)
]
# Result files: only those listed in RESULTS_MANIFEST (one path or glob per line, relative to the
# repository root): the files the capsule's run reads and the stored analyses the paper's numbers
# come from. Raw per-round logs, legacy summaries and benchmark prompt data are not packaged.
RESULTS_MANIFEST = ROOT / "src" / "revloop" / "capsule_results.txt"
EXCLUDE = ["*_e1b.*", "*/smoke/*", "*localsmoke*", "*/__pycache__/*", "src/outputs/*",
           "*/e17_workbound_vgpu3090.*",
           # per-round logs the run does not read (r4_rounds_analysis reads the RTX PRO 6000 ones
           # for the single-block view; the others are recovered from the per-call tables)
           "*_r4cold_rounds.jsonl", "*_u2048b4_rounds.jsonl"]
LEGACY: list[str] = []
LEGACY_DEST = "src/experiments/results/legacy/"
# Files larger than this are not packaged (reported by the build).
MAX_FILE_BYTES = 50 * 2**20

# The project's former name must not appear anywhere in the package: file and module names,
# identifiers, option values, paths recorded in result metadata, prose. These rules rewrite the
# shipped copies (paths and text) only; the repository keeps its own names. Order matters:
# specific rules first, the bare lower-case name (identifiers, option values) last.
RENAME = [(re.compile(p, f), r) for p, r, f in [
    (r"corey_selective_scan_dispatch", "selective_scan_dispatch", 0),  # module and file name
    (r"COREY_SELECTIVE_SCAN_", "SELECTIVE_SCAN_", 0),                   # environment variables
    (r"(?<=/)corey_transformer\w*", "repo", re.I),                     # checkout directory in recorded paths
    (r"(?<=/)corey_", "", re.I),                                         # trace directories in recorded paths
    (r"\+corey\.", "+", 0),                                              # local version label of the patched mamba_ssm
    (r"\bCOREY_Transformer ", "", 0),
    (r"COREY_VS_STATIC", "ENTROPY_VS_STATIC", 0),
    (r"\bCOREY selective-scan\b", "selective-scan", 0),
    (r"\bCOREY entropy estimator\b", "entropy estimator", 0),
    (r"\bCOREY scheduler\b", "entropy scheduler", 0),
    (r"\bCOREY routed\b", "entropy-routed", 0),
    (r"\ba COREY chunk\b", "an entropy-rule chunk", 0),
    (r"\bCOREY chunk\b", "entropy-rule chunk", 0),
    (r"\bCOREY's\b", "the entropy scheduler's", 0),
    (r"(?:^|(?<=\. )|(?<=\.  ))COREY\b", "The entropy scheduler", re.M),  # sentence start
    (r"\bCOREY\b", "the entropy scheduler", 0),
    (r"corey", "entropy", 0),                                            # identifiers, option values
]]
FORMER_NAME = re.compile(r"corey", re.I)
# Text formats the RENAME rules are applied to (everything else is copied unchanged).
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".log", ".tsv", ".sh", ".toml", ".cfg",
                 ".yml", ".yaml", ".cu", ".cuh", ".cpp", ".h", ".c", ".in", ""}

# Identifying strings that must not appear in the package (checked case-insensitively, in file
# contents and in file names).
IDENTIFYING = [r"mabo", r"bo\s+ma\b", r"weiqi", r"jinsong", r"aut\.ac\.nz", r"\brcn\d{3,}", r"auckland",
               r"bobsteam", r"c:[\\/]+source", r"/mnt/c/", r"autodl", r"seetacloud", r"westb",
               r"ssh\s+-p", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", r"hkvlm", r"github\.com/mabo",
               r"corey", r"/root/"]
IP_ALLOW = {"0.0.0.0", "127.0.0.1"}
# Public benchmark text under src/data/ is checked for personal identifiers only, not for
# place names or numeric patterns that occur in ordinary prose.
DATA_ONLY_PATTERNS = {r"mabo", r"bo\s+ma\b", r"weiqi", r"jinsong", r"aut\.ac\.nz", r"\brcn\d{3,}", r"bobsteam",
                      r"corey"}
# Capsule folders owned by the Code Ocean UI (never written by this script): scan hits there are
# reported for a manual fix in the UI but do not block the commit of the code folder.
UI_OWNED = {"environment", "metadata"}


def expand(patterns):
    out = set()
    for pat in patterns:
        out.update(p for p in ROOT.glob(pat) if p.is_file())
    return {p for p in out if not any(fnmatch.fnmatch(str(p.relative_to(ROOT)), e) for e in EXCLUDE)}


def local_imports(path: Path) -> set[Path]:
    """Files of this repository imported by ``path`` (``src.*`` absolute and relative imports)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    mods = set()
    pkg = path.relative_to(ROOT).with_suffix("").parts[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names if a.name.split(".")[0] == "src")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(pkg[: len(pkg) - node.level + 1])
                name = ".".join(base + ([node.module] if node.module else []))
            else:
                name = node.module or ""
            if name.split(".")[0] != "src":
                continue
            mods.add(name)
            mods.update(f"{name}.{a.name}" for a in node.names)  # "from pkg import module"
    files = set()
    for m in mods:
        parts = m.split(".")
        for cand in (ROOT.joinpath(*parts).with_suffix(".py"), ROOT.joinpath(*parts, "__init__.py")):
            if cand.is_file():
                files.add(cand)
        for i in range(1, len(parts)):  # package __init__ files on the way
            init = ROOT.joinpath(*parts[:i], "__init__.py")
            if init.is_file():
                files.add(init)
    return files


def closure(entries: set[Path]) -> set[Path]:
    todo, seen = list(entries), set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        if p.suffix == ".py":
            todo.extend(local_imports(p) - seen)
    return seen


def scan(root: Path) -> list[str]:
    hits = []
    rx = [re.compile(p, re.I) for p in IDENTIFYING]
    for f in sorted(root.rglob("*")):
        rel = f.relative_to(root)
        if ".git" in rel.parts:
            continue
        hits.extend(f"{rel}: file name matches /{r.pattern}/" for r in rx if r.search(str(rel)))
        if not f.is_file():
            continue
        if f.suffix.lower() in {".png", ".pdf", ".jpg", ".zip", ".gz", ".pt", ".npz", ".npy"}:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        in_data = "data" in f.relative_to(root).parts
        for n, line in enumerate(text.splitlines(), 1):
            for r in rx:
                if in_data and r.pattern not in DATA_ONLY_PATTERNS:
                    continue
                for m in r.finditer(line):
                    if m.group(0) in IP_ALLOW:
                        continue
                    hits.append(f"{f.relative_to(root)}:{n}: matches /{r.pattern}/")
    return hits


REQUIREMENTS = """# CPU run (the capsule's run script)
numpy
pandas
scipy
matplotlib
scikit-learn  # optional: without it the predictor trees use the built-in CART fallback
# GPU experiments (not needed by the run): Python 3.12, torch==2.8.0, triton==3.4.0,
# transformers==5.6.2, flash-linear-attention==0.5.2, mamba_ssm 2.2.2 / 2.3.2.post1
"""

RUN_SCRIPT = r"""#!/usr/bin/env bash
# Reproducible run: recompute the analyses, tables and figures of the paper from the stored
# measurements, on the CPU (no GPU is used). README.md maps every table and figure of the paper
# to its result files and scripts, lists the analyses this run does not repeat, and describes the
# GPU experiments that produced the measurements.
set -ex
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" MPLBACKEND=Agg
PY=$(command -v python3 || command -v python)
M=src.experiments.tpds
OUT="${RESULTS_DIR:-../results}"
mkdir -p "$OUT/analysis" "$OUT/figures" "$OUT/tables"
"$PY" -c "import sklearn" 2>/dev/null || echo "scikit-learn not installed: the predictor trees use the built-in CART fallback (see README.md)"
# The analyses read and rewrite a scratch copy of the stored measurements. At the end, every file
# they rewrote is copied to $OUT/analysis under its stored name, for comparison with
# src/experiments/results/tpds/.
W=$(mktemp -d)
cp -p src/experiments/results/tpds/*.json src/experiments/results/tpds/*.jsonl src/experiments/results/tpds/*.csv "$W"/
touch "$W/.start"
sleep 1

# Compile-informed cost model: fit, validation, drop-one ablation, reference CUDA scan, the
# device-time and cold-L2 calibrations, and the 1.4B / 2.8B checkpoints without refitting
"$PY" -u -m $M.cost_model --in-dir "$W" --ablation
"$PY" -u -m $M.cost_model --in-dir "$W" --hosts h800,h800_r3spin
for tag in vgpu3090_r3spin pro6000_r4spin pro6000_r4cold; do
  "$PY" -u -m $M.cost_model --in-dir "$W" --hosts "$tag"
done
"$PY" -u -m $M.cost_model --in-dir "$W" --reference
for ck in 1p4b 2p8b; do
  "$PY" -u -m $M.cost_model --in-dir "$W" --hosts vgpu3090_r3spin \
    --transfer-calls "$W/e2_mamba1_vgpu3090_${ck}_r3.csv" --transfer-ref "$W/e2_mamba1_vgpu3090_r3spin.csv"
  "$PY" -u -m $M.cost_model --in-dir "$W" --hosts vgpu3090 \
    --transfer-calls "$W/e2_mamba1_vgpu3090_${ck}_r3plain.csv" --transfer-ref "$W/e2_mamba1_vgpu3090.csv"
done
"$PY" -u -m $M.cost_model_baselines --in-dir "$W" --hosts pro6000,vgpu3090,h800,h800_r3spin,vgpu3090_r3spin

# Value invariance, predictors of the per-call oracle, break-even, key churn, entropy band
for tag in pro6000 vgpu3090 pro6000_e1c h800 h800_r3spin; do
  "$PY" -u -m $M.e1_analysis --in-dir "$W" --tag "$tag"
done
"$PY" -u -m $M.e3_predictors --in-dir "$W" --models mamba1 \
  --hosts pro6000,vgpu3090,h800,h800_r3spin,vgpu3090_r3spin,vgpu3090_1p4b_r3,vgpu3090_2p8b_r3,vgpu3090_1p4b_r3plain,vgpu3090_2p8b_r3plain
"$PY" -u -m $M.e3_predictors --in-dir "$W" --models mamba2 --hosts pro6000,vgpu3090
for host in pro6000 vgpu3090; do
  "$PY" -u -m $M.e5_breakeven --in-dir "$W" --host "$host"
  "$PY" -u -m $M.entropy_band --in-dir "$W" --host "$host"
done
"$PY" -u -m $M.e5b_key_churn --in-dir "$W"

# Placement controls, reference CUDA scan, tensor-parallel widths, production kernels at larger
# widths, routed end-to-end study
"$PY" -u -m $M.e9_placement --analyze --out-dir "$W" --tag vgpu3090
"$PY" -u -m $M.e9_placement --analyze --out-dir "$W" --tag pro6000_r3
"$PY" -u -m $M.e16c_l2_placement --analyze --out-dir "$W" --tag pro6000
"$PY" -u -m $M.e8_refscan --analyze --out-dir "$W" --tag vgpu3090
"$PY" -u -m $M.e23_fixed_threads_analysis --in-dir "$W"
for host in vgpu3090 h800; do
  "$PY" -u -m $M.e12_analysis --host "$host" --in-dir "$W" --params-dir "$W" --out-dir "$W"
done
"$PY" -u -m $M.e17_analysis --in-dir "$W" --tag vgpu3090full \
  --per-call e2_mamba2_vgpu3090_2p7b_r3.csv,e2_gdn_vgpu3090_1p4b_r3.csv
"$PY" -u -m $M.e10_routed_interleaved --analyze --out-dir "$W" --tag vgpu3090

# Oracle headroom under timing noise
"$PY" -u -m $M.headroom_noise --in-dir "$W" \
  --datasets mamba1_pro6000,mamba1_vgpu3090,mamba2_pro6000,mamba2_vgpu3090,refscan_vgpu3090,mamba1_1p4b_vgpu3090,mamba1_2p8b_vgpu3090
"$PY" -u -m $M.headroom_noise --in-dir "$W" --out "$W/headroom_noise_r3.json" \
  --datasets mamba1_1p4b_r3_vgpu3090,mamba1_2p8b_r3_vgpu3090,mamba1_1p4b_r3plain_vgpu3090,mamba1_2p8b_r3plain_vgpu3090,mamba1_r3spin_vgpu3090
"$PY" -u -m $M.headroom_noise --in-dir "$W" --out "$W/headroom_noise_e17.json" \
  --datasets mamba2_2p7b_r3_vgpu3090,gdn_1p4b_r3_vgpu3090

# Repeated shuffled rounds in both timing conventions, bootstrap intervals of Tables IV and V,
# cold L2, CUDA-graph timing, TFLA mLSTM kernel
"$PY" -u -m $M.r4_rounds_analysis --in-dir "$W" --tags vgpu3090_r4spin,vgpu3090_r4plain,pro6000_r4spin,pro6000_r4plain
"$PY" -u -m $M.tables_iv_v_ci --in-dir "$W"
"$PY" -u -m $M.e20_cold_analysis --in-dir "$W"
"$PY" -u -m $M.e22_graph_analysis --in-dir "$W"
for tag in vgpu3090_r4spinb vgpu3090_r4plainb; do
  "$PY" -u -m $M.e24_tfla_chunk --analyze --out-dir "$W" --tag "$tag"
done

# Tables (LaTeX rows) and figures
for host in pro6000 vgpu3090; do
  "$PY" -u -m $M.make_tables --in-dir "$W" --host "$host" validation latency effects autotune overhead > "$OUT/tables/tables_$host.tex"
  "$PY" -u -m $M.make_figures --in-dir "$W" --host "$host" --out-dir "$OUT/figures/$host"
done
"$PY" -u -m $M.make_tables --in-dir "$W" --host pro6000 redraw tost > "$OUT/tables/tables_e1_controls.tex"
"$PY" -u -m $M.make_tables --in-dir "$W" --host vgpu3090 reference routed_e10 > "$OUT/tables/tables_reference_routed.tex"
"$PY" -u -m $M.make_tables --in-dir "$W" model_params model_ablation model_compile > "$OUT/tables/tables_cost_model.tex"

find "$W" -maxdepth 1 -type f -newer "$W/.start" -exec cp -p {} "$OUT/analysis/" \;
rm -rf "$W"
"""

README = """# {title}

Code and measurements for the paper. The reproducible run of this capsule recomputes the
analyses, tables and figures from the stored measurements on the CPU; it does not need a GPU.

## Layout

- `src/experiments/tpds/` - kernels, GPU experiments and CPU analyses of the paper (the files are
  listed with the paper items below), with the modules they import (`src/algorithms/`,
  `src/experiments/run_*.py`).
- `src/kernels/mamba1_runtime_chunk/` and `src/selective_scan_dispatch.py` - the reference
  Mamba-1 CUDA selective scan rebuilt with the chunk as a runtime argument, for the routed
  end-to-end study.
- `src/experiments/results/tpds/` - the stored measurements the run reads and the stored analyses
  the paper's numbers come from. Raw per-round logs, benchmark prompts and the summaries of the
  earlier harnesses behind a few supplementary tables are not included.

## Result files

`src/experiments/results/tpds/<experiment>_[<model>_]<tag>.<ext>`, with `<model>` = `mamba1`,
`mamba2` or `gdn` for per-call tables. `.jsonl` files are the raw, resumable logs; `.json` and
`.csv` files are tables and analyses. The tag names the GPU, the checkpoint and the timing
convention:

| Tag | Meaning |
|---|---|
| `pro6000`, `vgpu3090`, `h800` | RTX PRO 6000 Blackwell, RTX 3090, H800 PCIe. Without a suffix: launch-inclusive timing (repeats issued back to back from Python), warm L2, one round |
| `_1p4b`, `_2p8b`, `_2p7b` | Mamba-1 1.4B / 2.8B and Mamba-2 2.7B checkpoints (`gdn_*_1p4b`: Gated DeltaNet 1.4B); 370M otherwise. `_1p4b` / `_2p8b` without a further suffix: first batch-1 sweep, superseded |
| `_r3spin`, `_r3` | device time: a GPU-side spin before every timing block lets the host queue all repeats, so only GPU execution is timed |
| `_r3plain` | the `_r3` calls, launch-inclusive |
| `_r4spin`, `_r4plain` | three rounds with the chunk order shuffled per call and round, device time / launch-inclusive |
| `_r4cold` | cold L2 (a 128 MiB buffer overwritten before every timed repeat) plus device time, three rounds |
| `_r4graph`, `_r4graphcold` | CUDA-graph replays, warm / flushed L2 |
| `_r4t32` | reference CUDA scan at 32 threads per block |
| `_u2048b4` | 64 synthesized calls of one serving key (length bucket 2048, batch 4, unaligned length) |
| `_r4spinb`, `_r4plainb` (`e24_tfla_*`) | TFLA mLSTM chunk sweep, device time / launch-inclusive; `_r4spin`, `_r4plain`: first attempt, stopped after one shape |
| `_rmed` (`e3b_information_*`) | information tests on the round-median oracle |
| `pro6000_e1c` | control run of the value-invariance experiment with independent redraws |
| `pro6000_r3` (`e9_placement_*`) | placement control on the RTX PRO 6000 |
| `vgpu3090full` (`e17_*`) | synthetic sweep of the production kernels at larger widths |

Host names, absolute paths and local build labels recorded in the result metadata are replaced
by neutral placeholders (`gpu-host`, `workstation`, `/data`, `/shared`, `/repo`).

## Reproducible run (CPU)

`run` copies the stored measurements to a scratch folder, recomputes from them every analysis
listed below, and writes into `../results/`:

- `analysis/` - every analysis file it rewrote, under the name of the stored file in
  `src/experiments/results/tpds/`, for direct comparison: cost-model fits, validation and
  ablation (`cost_model_*.json`, `cost_model_curves_*.csv`) and simpler selectors
  (`cost_model_baselines_*.json`); value invariance (`e1_analysis_*.json`); predictors of the
  per-call oracle (`e3_*.json`); break-even, key churn and entropy band (`e5_*.json`,
  `e5b_*.json`, `entropy_band_*.json`); placement controls (`e9_placement_*.json`,
  `e16c_l2_placement_pro6000.json`); reference CUDA scan (`e8_analysis_vgpu3090.json`,
  `e23_fixed_threads_vgpu3090_r4t32.json`); tensor-parallel widths (`e12_analysis_*.json`);
  production kernels at larger widths (`e17_analysis_vgpu3090full.json`); routed end-to-end
  study (`e10_routed_vgpu3090.json`); headroom under timing noise (`headroom_noise*.json`);
  repeated rounds (`r4_rounds_analysis.json`), bootstrap intervals (`tables_iv_v_ci.json`),
  cold L2 (`e20_cold_analysis_pro6000.json`), CUDA graphs (`e22_graph_analysis_vgpu3090.json`),
  TFLA (`e24_tfla_*_analysis.json`);
- `tables/` - LaTeX rows of the supplementary tables generated by `make_tables.py`;
- `figures/` - the paper's figures (`pro6000/`: Figs. 2-3 and the supplementary figures;
  `vgpu3090/`: the same figures for the RTX 3090).

{runtime}

The predictor trees (`e3_predictors.py`, reused by `cost_model_baselines.py`,
`r4_rounds_analysis.py`, `tables_iv_v_ci.py`, `e22_graph_analysis.py` and
`e23_fixed_threads_analysis.py`) use scikit-learn's decision tree when it is installed and an
equivalent built-in CART tree otherwise. The capsule environment has no scikit-learn: the run then
takes longer, and a few per-batch-size values of the tree predictors can differ slightly where two
splits tie (pooled values agree). With scikit-learn installed the recomputed files match the stored
ones.

## Paper items and the files they come from

Result files are in `src/experiments/results/tpds/`; scripts are in `src/experiments/tpds/`
unless a path is given.
"Run" marks the items whose analysis `run` recomputes; "partly" means that some of the listed
analyses are not repeated (next section).

{paper_map}

## Analyses not repeated by `run`

These analyses read only stored files as well, but need packages or data that the capsule does
not include, or take long; their outputs are stored in `src/experiments/results/tpds/`.

| Script | Output | Why not in `run` |
|---|---|---|
| `cost_model_baselines_r4.py` | `cost_model_baselines_r4_*.json` | needs scikit-learn (gradient-boosted and MLP regressors) |
| `e3b_information.py` (`--round-median --tags mamba1_pro6000_r4spin,mamba1_vgpu3090_r4spin` for `_rmed`) | `e3b_information_*.json` | needs scikit-learn; long bootstrap and permutation tests |
| `headroom_r4.py` | `headroom_r4.json` | about 10 min on a 16-core workstation (repeated refits of the cost model) |
| `serving_key_churn.py`, `deployment_units.py`, `r4_thinkey_replay.py` | `serving_key_churn.json`, `deployment_units.json`, `r4_thinkey_vgpu3090.json` | replay public serving traces (Azure LLM inference traces 2023/2024, BurstGPT, Mooncake; about 2 GB, download URLs in `serving_key_churn.py`); `deployment_units.py --skip-serving` and `r4_thinkey_replay.py --skip-replay` run without them |
| `cost_model.py --hosts h800,<source> --no-loko` | `cost_model_transfer_h800.json` | transfer onto the H800; rewrites the source fit without its leave-one-key-out block, so the source fit is re-run afterwards |
| `kernel_stats.py`, `e1_from_log.py` | `kernel_stats_sm*.json*`, `e1_pro6000_e1c.json` | compile facts need the Triton compiler and a GPU of the target architecture (others are cross-compiled with `--archs`); `e1_from_log.py` rebuilds the control-run rows from its raw run log (not included) |

## GPU experiments

The measurements were taken with Python 3.12, PyTorch 2.8, Triton 3.4, Transformers 5.6.2,
FLA 0.5.2 and mamba_ssm 2.2/2.3 on an RTX PRO 6000 Blackwell, an RTX 3090 and an H800 PCIe.
Kernel timing needs a GPU without other processes; `--exclusive-guard` waits for one.

```bash
export PYTHONPATH=$PWD
M=src.experiments.tpds
# captured prefill traces (Mamba-1 370M/1.4B/2.8B, Mamba-2 370M/2.7B) and Gated DeltaNet 1.4B inputs
python -m $M.capture_traces --help
python -m $M.capture_gdn_traces --help
# value invariance (control run: --rounds 10 --extra-seeds 2 --record-gpu-state)
python -m $M.e1_value_invariance --families mamba1,mamba2,gdn --exclusive-guard
# per-call chunk sweep: one round (launch-inclusive), --pre-spin-ms 20 (device time),
# --rounds 3 (shuffled rounds), --flush-l2 (cold L2), --cuda-graph (CUDA-graph replays),
# --synth-calls 64 --synth-bucket 2048 --synth-batch 4 (one serving key)
python -m $M.e2_oracle_sweep --model mamba1 --exclusive-guard
python -m $M.e4_autotune --help                      # launch-space auto-tuning, decision overhead
python -m $M.e7_reference_scan --exclusive-guard     # needs the mamba_ssm CUDA kernels
python -m $M.e8_refscan --help                       # runtime-chunk reference scan (--thread-families 32)
python -m $M.e9_placement --help                     # value versus placement
python -m $M.e12_tp_widths --ref                     # tensor-parallel widths
python -m $M.e17_workbound --help                    # production kernels at larger widths
python -m $M.e24_tfla_chunk --pre-spin-ms 20 --tag <tag>   # needs mlstm_kernels
python -m $M.e10_routed_interleaved --help           # routed end-to-end study
```

The routed end-to-end study needs a mamba_ssm build whose selective-scan kernel accepts the
chunk as a runtime argument (`src/kernels/mamba1_runtime_chunk/`, loaded through
`src/selective_scan_dispatch.py`).
"""

# Paper items -> (section, result files, scripts, recomputed by run). Labels are the LaTeX labels of
# the manuscript; "S-" marks the supplement. Numbers are rendered from the built paper's .aux files.
PAPER_MAP = [
    ("tab:taxonomy", "sec:related", "none (comparison of related work)", "-", "-"),
    ("fig:system_path", "sec:method", "none (schematic)", "-", "-"),
    ("tab:framework", "sec:cost_model",
     "`cost_model_{pro6000,vgpu3090,h800}.json`, `cost_model_refscan_vgpu3090.json`, `e4_{pro6000,vgpu3090}.json`",
     "`cost_model.py` (also `--reference`); tuning times from `e4_autotune.py` (GPU)", "yes"),
    ("sec:methodology", None,
     "`e7_vgpu3090.json` (Triton vs reference scan), `e20_cold_analysis_pro6000.json` (cold L2); launch floor "
     "and timing-convention statistics from `e2_mamba1_{pro6000,vgpu3090}.csv` and `e2_mamba1_pro6000_r4plain.csv`",
     "`e7_reference_scan.py` (GPU), `e20_cold_analysis.py`", "partly"),
    ("tab:value_invariance", "sec:res_invariance", "`e1_analysis_pro6000.json` (from `e1_pro6000.json`)",
     "`e1_analysis.py`", "yes"),
    ("sec:res_invariance", None,
     "`e1_analysis_{pro6000,vgpu3090,pro6000_e1c}.json`, `e9_placement_{vgpu3090,pro6000_r3}.json`, "
     "`e16c_l2_placement_pro6000.json`",
     "`e1_analysis.py`, `e9_placement.py --analyze`, `e16c_l2_placement.py --analyze`", "yes"),
    ("fig:chunk_curves", "sec:res_oracle", "`e1_pro6000.json`", "`make_figures.py`", "yes"),
    ("tab:oracle_regret", "sec:res_oracle",
     "Mamba-1: `r4_rounds_analysis.json` (from `e2_mamba1_pro6000_r4spin.csv`); Mamba-2: `e2_mamba2_pro6000.{csv,json}`",
     "`r4_rounds_analysis.py`; summary written by `e2_oracle_sweep.py`", "yes"),
    ("tab:predictors", "sec:res_predictors",
     "Mamba-1: `r4_rounds_analysis.json`, `cost_model_pro6000_r4spin.json`, "
     "`e3b_information_mamba1_pro6000_r4spin_rmed.json`; Mamba-2: `e3_mamba2_pro6000.json`, "
     "`e3b_information_mamba2_pro6000.json`, `cost_model_pro6000.json`",
     "`r4_rounds_analysis.py`, `e3_predictors.py`, `cost_model.py`, `e3b_information.py`", "partly"),
    ("sec:res_predictors", None,
     "`e3b_information_*.json`, `r4_rounds_analysis.json`, `entropy_band_pro6000.json`",
     "`e3b_information.py`, `r4_rounds_analysis.py`, `entropy_band.py`", "partly"),
    ("tab:baselines", "sec:res_model",
     "`cost_model_baselines_r4_{pro6000,vgpu3090,h800}.json`, `cost_model_baselines_r4_summary.json`",
     "`cost_model_baselines_r4.py`", "no"),
    ("fig:calibration_budget", "sec:res_model",
     "`cost_model_baselines_r4_summary.json`; default chunk from `e2_mamba1_{pro6000,vgpu3090,h800}.csv`",
     "`make_figures.py`", "yes (figure)"),
    ("sec:res_model", None,
     "`cost_model_{pro6000,vgpu3090,h800}.json`, `e3_mamba1_{pro6000,vgpu3090,h800}.json`, `e4_pro6000.json`, "
     "`r4_rounds_analysis.json`, `headroom_r4.json`",
     "`cost_model.py`, `e3_predictors.py`, `r4_rounds_analysis.py`, `headroom_r4.py`", "partly"),
    ("tab:reference", "sec:res_reference",
     "`e8_analysis_vgpu3090.json`, `cost_model_refscan_vgpu3090.json`, `e3_mamba1_vgpu3090_{1p4b,2p8b}_r3.json`, "
     "`cost_model_transfer_e2_mamba1_vgpu3090_{1p4b,2p8b}_r3.json`, "
     "`e3b_information_mamba1_vgpu3090_{1p4b,2p8b}_r3.json`, `headroom_r4.json`; spread from "
     "`e2_mamba1_vgpu3090_{1p4b,2p8b}_r3.csv`",
     "`e8_refscan.py --analyze`, `cost_model.py`, `e3_predictors.py`, `e3b_information.py`, `headroom_r4.py`",
     "partly"),
    ("sec:res_reference", None,
     "`e23_fixed_threads_vgpu3090_r4t32.json`, `e12_analysis_{vgpu3090,h800}.json`, "
     "`e17_analysis_vgpu3090full.json`, `e22_graph_analysis_vgpu3090.json`, "
     "`e24_tfla_vgpu3090_r4spinb_analysis.json`",
     "`e23_fixed_threads_analysis.py`, `e12_analysis.py`, `e17_analysis.py`, `e22_graph_analysis.py`, "
     "`e24_tfla_chunk.py --analyze`", "yes"),
    ("sec:res_routed", None, "`e10_routed_vgpu3090.json` (from `.jsonl`); predicted cost: `deployment_units.json`",
     "`e10_routed_interleaved.py --analyze`; `deployment_units.py`", "partly"),
    ("tab:deployment", "sec:res_breakeven", "`deployment_units.json`", "`deployment_units.py`", "no"),
    ("sec:res_breakeven", None,
     "`e5_pro6000.json`, `e4_pro6000.json`, `e5b_{pro6000,vgpu3090}.json`, `e9_placement_vgpu3090.json`, "
     "`headroom_r4.json`, `serving_key_churn.json`, `r4_thinkey_vgpu3090.json`, `deployment_units.json`",
     "`e5_breakeven.py`, `e5b_key_churn.py`, `e9_placement.py`, `headroom_r4.py`, `serving_key_churn.py`, "
     "`r4_thinkey_replay.py`, `deployment_units.py`", "partly"),
    ("tab:replication", "sec:res_replication",
     "`e1_analysis_{pro6000,vgpu3090,h800}.json`, `e2_mamba1_{pro6000,vgpu3090,h800}.json`, "
     "`e3_mamba1_{pro6000,vgpu3090,h800}.json`, `cost_model_{pro6000,vgpu3090,h800}.json`, `headroom_r4.json`, "
     "`r4_rounds_analysis.json`, `e5b_{pro6000,vgpu3090}.json`",
     "`e1_analysis.py`, `e3_predictors.py`, `cost_model.py`, `headroom_r4.py`, `r4_rounds_analysis.py`, "
     "`e5b_key_churn.py`", "partly"),
    ("sec:res_replication", None,
     "`r4_rounds_analysis.json`, `cost_model_h800{,_r3spin}.json`, `cost_model_transfer_h800.json`, "
     "`e1_analysis_h800.json`, `e2_mamba1_h800_r3spin.{csv,json}`",
     "`r4_rounds_analysis.py`, `cost_model.py`, `e1_analysis.py`", "partly"),
    ("sec:discussion", None,
     "`cost_model_baselines_r4_summary.json`, `cost_model_{pro6000,vgpu3090}.json`, `cost_model_transfer_h800.json`, "
     "`e5b_{pro6000,vgpu3090}.json`",
     "`cost_model_baselines_r4.py`, `cost_model.py`, `e5b_key_churn.py`", "partly"),
    # supplement
    ("S-tab:supp_protocol", "S-sec:supp_timing", "settings recorded in every result file (`meta`, `args`)",
     "`common.py`", "-"),
    ("S-tab:supp_platforms", "S-sec:supp_timing", "software versions recorded in the result-file metadata", "-", "-"),
    ("S-tab:supp_ties", "S-sec:supp_timing", "`e3b_information_*.json` (tie margins), `r4_rounds_analysis.json`",
     "`e3b_information.py`, `r4_rounds_analysis.py`", "partly"),
    ("S-tab:supp_validation", "S-sec:supp_kernel_experiments", "`e1_pro6000.json`", "`make_tables.py` (validation)",
     "yes"),
    ("S-fig:value_invariance", "S-sec:supp_kernel_experiments", "`e1_pro6000.json`", "`make_figures.py`", "yes"),
    ("S-tab:supp_effects", "S-sec:supp_kernel_experiments", "`e1_analysis_pro6000.json`",
     "`e1_analysis.py`, `make_tables.py` (effects)", "yes"),
    ("S-tab:supp_redraw", "S-sec:supp_kernel_experiments",
     "`e1_pro6000.json`, `e1_pro6000_e1c.json`, `e1_analysis_pro6000_e1c.json`",
     "`e1_analysis.py`, `make_tables.py` (redraw); rows of the control run rebuilt by `e1_from_log.py`", "yes"),
    ("S-tab:supp_tost", "S-sec:supp_kernel_experiments", "`e1_analysis_{pro6000,vgpu3090}.json`",
     "`e1_analysis.py`, `make_tables.py` (tost)", "yes"),
    ("S-tab:supp_placement", "S-sec:supp_kernel_experiments", "`e9_placement_vgpu3090.json` (from `.jsonl`)",
     "`e9_placement.py --analyze`", "yes"),
    ("S-fig:supp_entropy_regime", "S-sec:supp_kernel_experiments", "`e2_mamba{1,2}_pro6000.csv`",
     "`make_figures.py`", "yes"),
    ("S-tab:supp_latency", "S-sec:supp_kernel_experiments", "`e1_pro6000.json`", "`make_tables.py` (latency)", "yes"),
    ("S-tab:supp_information", "S-sec:supp_kernel_experiments",
     "`e3b_information_<model>_<tag>.json` (17 data sets; `_rmed`: round medians)", "`e3b_information.py`", "no"),
    ("S-fig:supp_regret_cdf", "S-sec:supp_kernel_experiments", "`e3_mamba{1,2}_pro6000.json`",
     "`e3_predictors.py`, `make_figures.py`", "yes"),
    ("S-tab:supp_autotune", "S-sec:supp_kernel_experiments", "`e4_pro6000.json`", "`make_tables.py` (autotune)",
     "yes"),
    ("S-tab:supp_overhead", "S-sec:supp_kernel_experiments", "`e4_pro6000.json`", "`make_tables.py` (overhead)",
     "yes"),
    ("S-tab:supp_reference", "S-sec:supp_kernel_experiments", "`e7_vgpu3090.json`", "`make_tables.py` (reference)",
     "yes"),
    ("S-tab:supp_latency_3090", "S-sec:supp_kernel_experiments", "`e1_vgpu3090.json`", "`make_tables.py` (latency)",
     "yes"),
    ("S-tab:supp_effects_3090", "S-sec:supp_kernel_experiments", "`e1_analysis_vgpu3090.json`",
     "`e1_analysis.py`, `make_tables.py` (effects)", "yes"),
    ("S-tab:supp_autotune_3090", "S-sec:supp_kernel_experiments", "`e4_vgpu3090.json`",
     "`make_tables.py` (autotune)", "yes"),
    ("S-tab:supp_placement_pro6000", "S-sec:supp_invariance_details", "`e9_placement_pro6000_r3.json`",
     "`e9_placement.py --analyze`", "yes"),
    ("S-tab:supp_l2", "S-sec:supp_invariance_details", "`e16c_l2_placement_pro6000.json`",
     "`e16c_l2_placement.py --analyze`", "yes"),
    ("S-sec:supp_h800", None,
     "`cost_model_h800{,_r3spin}.json`, `cost_model_transfer_h800.json`, `cost_model_baselines_h800{,_r3spin}.json`, "
     "`e3_mamba1_h800{,_r3spin}.json`, `e1_analysis_h800{,_r3spin}.json`, `e12_analysis_h800.json`",
     "`cost_model.py`, `cost_model_baselines.py`, `e3_predictors.py`, `e1_analysis.py`, `e12_analysis.py`", "partly"),
    ("S-tab:supp_model_compile", "S-sec:supp_cost_model", "`kernel_stats_sm{120,86}.json`",
     "`kernel_stats.py` (compiler), `make_tables.py` (model_compile)", "yes (table)"),
    ("S-fig:cost_model", "S-sec:supp_cost_model", "`cost_model_curves_{pro6000,vgpu3090}.csv`",
     "`cost_model.py`, `make_figures.py`", "yes"),
    ("S-tab:supp_model_params", "S-sec:supp_cost_model",
     "`cost_model_{pro6000,vgpu3090}.json`; auto-tuner leave-one-key-out row: `headroom_r4.json`",
     "`cost_model.py`, `make_tables.py` (model_params); `headroom_r4.py`", "partly"),
    ("S-tab:supp_model_ablation", "S-sec:supp_cost_model", "`cost_model_{pro6000,vgpu3090}.json`",
     "`cost_model.py --ablation`, `make_tables.py` (model_ablation)", "yes"),
    ("S-tab:supp_clean_selection", "S-sec:supp_cost_model", "`cost_model_baselines_r4_summary.json`",
     "`cost_model_baselines_r4.py`", "no"),
    ("S-tab:supp_baselines", "S-sec:supp_cost_model",
     "`cost_model_baselines_r4_{pro6000,vgpu3090,h800}.json`, `cost_model_baselines_r4_summary.json`",
     "`cost_model_baselines_r4.py`", "no"),
    ("S-tab:supp_topk", "S-sec:supp_cost_model", "`cost_model_baselines_r4_summary.json`",
     "`cost_model_baselines_r4.py`", "no"),
    ("S-tab:supp_pipeline_model", "S-sec:supp_cost_model",
     "`cost_model_{pro6000,vgpu3090}.json` (work-term model), `headroom_r4.json`, `e22_graph_analysis_vgpu3090.json`",
     "`cost_model.py`, `headroom_r4.py`, `e22_graph_analysis.py`", "partly"),
    ("S-tab:supp_ref_threads", "S-sec:supp_cost_model",
     "`e23_fixed_threads_vgpu3090_r4t32.json` (from `e8_refscan_vgpu3090_r4t32.csv`)",
     "`e23_fixed_threads_analysis.py`", "yes"),
    ("S-fig:supp_breakeven", "S-sec:supp_breakeven", "`e5_pro6000.json`", "`e5_breakeven.py`, `make_figures.py`",
     "yes"),
    ("S-fig:keychurn", "S-sec:supp_breakeven", "`e5b_{pro6000,vgpu3090}.json`; per-key costs: `deployment_units.json`",
     "`e5b_key_churn.py`, `make_figures.py`; `deployment_units.py`", "yes (figure)"),
    ("S-tab:supp_tp", "S-sec:supp_tp", "`e12_analysis_vgpu3090.json` (from `e12_tp_widths_vgpu3090.jsonl`)",
     "`e12_analysis.py`", "yes"),
    ("S-tab:supp_largewidth", "S-sec:supp_largewidth",
     "`e17_analysis_vgpu3090full.json` (from `e17_workbound_vgpu3090full.jsonl`)", "`e17_analysis.py`", "yes"),
    ("S-tab:supp_largewidth_calls", "S-sec:supp_largewidth",
     "`e22_graph_analysis_vgpu3090.json` (from `e2_mamba2_vgpu3090_2p7b_r3.csv`, `e2_gdn_vgpu3090_1p4b_r3.csv`)",
     "`e22_graph_analysis.py`", "yes"),
    ("S-tab:supp_tfla", "S-sec:supp_tfla", "`e24_tfla_vgpu3090_r4{spin,plain}b_analysis.json` (from the `.jsonl` files)",
     "`e24_tfla_chunk.py --analyze`", "yes"),
    ("S-tab:supp_conventions", "S-sec:supp_conventions",
     "`e3_mamba1_vgpu3090{,_r3spin}.json`, `e3_mamba1_vgpu3090_{1p4b,2p8b}_{r3,r3plain}.json`, "
     "`e3_mamba2_vgpu3090.json`, `cost_model_vgpu3090{,_r3spin}.json`, `cost_model_transfer_*.json`, "
     "`headroom_r4.json`, `r4_rounds_analysis.json`, `e22_graph_analysis_vgpu3090.json`, `e3_mamba1_pro6000.json`, "
     "`cost_model_pro6000.json`",
     "`e3_predictors.py`, `cost_model.py`, `headroom_r4.py`, `r4_rounds_analysis.py`, `e22_graph_analysis.py`",
     "partly"),
    ("S-tab:supp_oracle_li", "S-sec:supp_conventions",
     "`r4_rounds_analysis.json` (collects `e2_mamba1_pro6000.csv`, `e3_mamba1_pro6000.json`, "
     "`e3b_information_mamba1_pro6000.json`, `cost_model_pro6000.json`)", "`r4_rounds_analysis.py`", "yes"),
    ("S-tab:supp_tables_ci", "S-sec:supp_conventions", "`tables_iv_v_ci.json`", "`tables_iv_v_ci.py`", "yes"),
    ("S-tab:supp_cold", "S-sec:supp_conventions",
     "`e20_cold_analysis_pro6000.json` (from `e4_`, `e1_`, `e2_mamba1_pro6000_r4{spin,cold}.*`, "
     "`cost_model_pro6000_r4{spin,cold}.json`)", "`e20_cold_analysis.py`", "yes"),
    ("S-tab:supp_graph", "S-sec:supp_graph", "`e22_graph_analysis_vgpu3090.json` (from `e2_*_r4graph*.csv`)",
     "`e22_graph_analysis.py`", "yes"),
    ("S-tab:supp_headroom_noise", "S-sec:supp_headroom_noise",
     "`headroom_r4.json`, `headroom_noise{,_r3,_e17}.json`; three-round rows: `r4_rounds_analysis.json`",
     "`headroom_r4.py`, `headroom_noise.py`, `r4_rounds_analysis.py`", "partly"),
    ("S-tab:supp_serving", "S-sec:supp_serving", "`serving_key_churn.json`, `r4_thinkey_vgpu3090.json`",
     "`serving_key_churn.py`, `r4_thinkey_replay.py`", "no"),
    ("S-fig:entropy_vs_oracle", "S-sec:supp_entropy_protocol", "`e2_mamba{1,2}_pro6000.csv`", "`make_figures.py`", "yes"),
    ("S-tab:bin_count_sensitivity", "S-sec:supp_bin_count", "not included (earlier harness)", "-", "no"),
    ("S-tab:active_integration", "S-sec:supp_active_overhead", "not included (earlier harness)", "-", "no"),
    ("S-tab:hook_micro", "S-sec:supp_hook_micro", "not included (earlier harness)", "-", "no"),
    ("S-tab:supp_routed_full", "S-sec:supp_routed", "`e10_routed_vgpu3090.json` (from `.jsonl`)",
     "`e10_routed_interleaved.py --analyze`, `make_tables.py` (routed_e10)", "yes"),
    ("S-sec:supp_legacy", None, "not included (earlier harness)", "-", "no"),
    ("S-tab:entropy_variance_real", "S-sec:entropy_variance_real", "not included (earlier harness)", "-", "no"),
    ("S-sec:supp_per_layer", None, "not included (earlier harness)", "-", "no"),
    ("S-tab:h800_corpora", "S-sec:supp_corpora", "not retained (earlier H800 corpus runs)", "-", "no"),
]
PAPER_AUX = [(ROOT / "paper" / "build" / "main.aux", ""), (ROOT / "paper" / "build" / "appendix_only.aux", "S-")]
AUX_LABEL = re.compile(r"\\newlabel\{([^}@]+)\}\{\{(?:\\mbox\s*\{)?([^{}]*)\}?\}")


def paper_labels() -> dict[str, str]:
    """LaTeX label -> rendered number, from the built main text and supplement ("S-" prefix)."""
    labels = {}
    for aux, prefix in PAPER_AUX:
        if not aux.is_file() or aux.stat().st_size == 0:
            raise SystemExit(f"{aux.relative_to(ROOT)} is missing or empty: build the paper first "
                             "(the README maps paper items by their rendered numbers)")
        labels.update((prefix + k, v) for k, v in AUX_LABEL.findall(aux.read_text(encoding="utf-8", errors="ignore")))
    return labels


def render_item(label: str, labels: dict[str, str]) -> str:
    supp = label.startswith("S-")
    kind = label[2:].split(":")[0] if supp else label.split(":")[0]
    num = ("S" if supp else "") + labels[label]
    name = {"tab": "Table", "fig": "Fig.", "sec": "Sec."}[kind]
    return ("Supplementary " if supp and kind != "sec" else "") + f"{name} {num}"


def paper_map_markdown() -> str:
    labels = paper_labels()
    rows = ["| Paper item | Result files | Script | Run |", "|---|---|---|---|"]
    for label, sec, files, script, in_run in PAPER_MAP:
        if label not in labels or (sec and sec not in labels):
            print(f"WARNING: paper label {label if label not in labels else sec} not found; row left out")
            continue
        item = render_item(label, labels) + (" text" if label.split(":")[0].endswith("sec") else "")
        if sec:
            item += f" ({render_item(sec, labels)})"
        rows.append(f"| {item} | {files} | {script} | {in_run} |")
    mapped = {lab for lab, *_ in PAPER_MAP}
    for lab in sorted(labels):
        kind = lab[2:].split(":")[0] if lab.startswith("S-") else lab.split(":")[0]
        if kind in {"tab", "fig"} and lab not in mapped:
            print(f"WARNING: {render_item(lab, labels)} ({lab}) has no row in PAPER_MAP")
    return "\n".join(rows)


# Run time of `run`, stated in the README (measured with the build's smoke test).
RUNTIME_NOTE = ("On a 16-core workstation with scikit-learn installed the run takes about 7 minutes. Without "
                "scikit-learn (the capsule environment as configured) the predictor trees run single-threaded "
                "in pure Python and the run takes about 70 minutes on the same workstation (longer on slower "
                "cores); adding scikit-learn to the environment brings it back to a few minutes.")


# Host names and absolute host paths recorded in result metadata are replaced in the copies.
SANITIZE = [(re.compile(r"autodl-container-[A-Za-z0-9-]+"), "gpu-host"),
            (re.compile(r"/root/autodl-tmp"), "/data"),
            (re.compile(r"/autodl-fs/data"), "/shared"),
            # analyses run on the authors' workstation record its host name, checkout path and kernel
            (re.compile(r"bobsteam\d*", re.I), "workstation"),
            (re.compile(r"/mnt/c/source/[A-Za-z_]+", re.I), "/repo"),
            (re.compile(r"Linux-\d+\.\d+\.\d+(?:\.\d+)?[A-Za-z0-9_.+-]*"), "Linux")]


def write_text(path: Path, text: str, tries: int = 5) -> None:
    """Write UTF-8 text; retry briefly, since a file just written on a Windows-mounted drive can
    be held by another process (EINVAL / EACCES) for a moment."""
    for i in range(tries):
        try:
            path.write_text(text, encoding="utf-8")
            return
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(0.5 * (i + 1))


def sanitize_results(code: Path) -> int:
    n = 0
    roots = [code / "src" / "experiments" / "results", code / "src" / "outputs"]
    for f in (p for r in roots if r.exists() for p in r.rglob("*")):
        if f.is_file() and f.suffix in {".json", ".jsonl", ".csv", ".log", ".txt", ".md", ".tsv"}:
            text = f.read_text(encoding="utf-8", errors="ignore")
            new = text
            for rx, rep in SANITIZE:
                new = rx.sub(rep, new)
            if new != text:
                write_text(f, new)
                n += 1
    return n


def renamed(text: str) -> str:
    for rx, rep in RENAME:
        text = rx.sub(rep, text)
    return text


def rename_text(code: Path) -> int:
    """Apply RENAME to every shipped text file outside src/data/ (benchmark text is not rewritten)."""
    n = 0
    data = code / "src" / "data"
    for f in code.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in TEXT_SUFFIXES or data in f.parents:
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new = renamed(text)
        if new != text:
            write_text(f, new)
            n += 1
    return n


def filter_data_records(code: Path) -> list[str]:
    """Drop the records of public benchmark subsets (JSONL) whose text contains the former name.

    Benchmark text is never rewritten; a matching record is left out of the shipped subset
    instead. The scripts read only the first records of each task, far below the dropped ones.
    """
    report = []
    rx = re.compile(FORMER_NAME.pattern.encode(), re.I)  # bytes: some subsets are not valid UTF-8
    for f in sorted((code / "src" / "data").rglob("*.jsonl")):
        lines = f.read_bytes().splitlines(keepends=True)
        drop = [i for i, ln in enumerate(lines) if rx.search(ln)]
        if drop:
            f.write_bytes(b"".join(ln for i, ln in enumerate(lines) if i not in drop))
            report.append(f"{f.relative_to(code)}: dropped record(s) {', '.join(map(str, drop))} "
                          f"of {len(lines)}")
    return report


LICENSE_HOLDER = re.compile(r"^(Copyright \(c\) \d{4}) .*$", re.M)


def build(args) -> int:
    if not (CAPSULE / ".git").exists():
        print("codeocean/ is not initialised (git submodule update --init codeocean)")
        return 2
    paper_map = paper_map_markdown()  # before anything is replaced: needs the built paper
    manifest = [ln.split("#", 1)[0].strip() for ln in RESULTS_MANIFEST.read_text().splitlines()]
    results = expand([m for m in manifest if m])
    missing = [m for m in manifest if m and not any(ROOT.glob(m))]
    for m in missing:
        print(f"manifest entry without a file: {m}")
    files = closure(expand(ENTRIES)) | expand(EXTRA) | results
    too_big = sorted(f for f in files if f.stat().st_size > MAX_FILE_BYTES)
    files -= set(too_big)
    for f in too_big:
        print(f"skipped (larger than {MAX_FILE_BYTES >> 20} MB): {f.relative_to(ROOT)} "
              f"({f.stat().st_size / 2**20:.1f} MB)")
    code = CAPSULE / "code"
    # clean replacement: keep the capsule's git data, environment and license
    for item in CAPSULE.iterdir():
        if item.name in {".git", "environment", "metadata", "code", ".gitignore"}:
            continue
        shutil.rmtree(item) if item.is_dir() else item.unlink()
    for item in code.iterdir() if code.exists() else []:
        if item.name == "LICENSE":
            continue
        shutil.rmtree(item) if item.is_dir() else item.unlink()
    code.mkdir(exist_ok=True)
    for f in sorted(files):
        rel = f.relative_to(ROOT).as_posix()
        out = renamed(rel.replace("src/outputs/", LEGACY_DEST, 1))
        dst = code / out
        if renamed(rel) != rel:
            print(f"renamed: {rel} -> {out}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
    for pkg in {p.parent for p in files if p.suffix == ".py"}:  # every package directory importable
        d = code / renamed(pkg.relative_to(ROOT).as_posix())
        while d != code:
            (d / "__init__.py").touch(exist_ok=True)
            d = d.parent
    print(f"sanitized host names and paths in {sanitize_results(code)} result files")
    print(f"removed the former project name from {rename_text(code)} files")
    for line in filter_data_records(code):
        print("benchmark subset:", line)
    (code / "src" / "requirements.txt").write_text(REQUIREMENTS)
    (code / "run").write_text(RUN_SCRIPT)
    (code / "run").chmod(0o755)
    (code / "README.md").write_text(README.replace("{title}", TITLE).replace("{runtime}", RUNTIME_NOTE)
                                     .replace("{paper_map}", paper_map))
    lic = code / "LICENSE"
    if lic.exists():
        lic.write_text(LICENSE_HOLDER.sub(r"\1 Anonymous Authors", lic.read_text()))
    # metadata/metadata.yml is owned by the Code Ocean metadata UI: Code Ocean rejects the
    # capsule ("Error Parsing Files") if the file is edited through git, so it is left as is.
    n_py = sum(1 for f in files if f.suffix == ".py")
    print(f"copied {len(files)} files ({n_py} Python modules) into codeocean/code")

    bad = [f for f in code.rglob("*.py") if not _py38_ok(f)]
    if bad:
        print("Python 3.8 syntax problems (the capsule environment runs Python 3.8):")
        for f in bad:
            print("  ", f.relative_to(CAPSULE))
    hits = scan(CAPSULE)
    ui_hits = [h for h in hits if Path(h.split(":", 1)[0]).parts[0] in UI_OWNED]
    hits = [h for h in hits if h not in ui_hits]
    if ui_hits:
        print(f"anonymization scan: {len(ui_hits)} hit(s) in folders owned by the Code Ocean UI "
              f"({', '.join(sorted(UI_OWNED))}); this script never writes them, fix them in the UI:")
        for h in ui_hits:
            print("  ", h)
    if hits:
        print(f"anonymization scan: {len(hits)} hit(s); nothing committed")
        for h in hits[:200]:
            print("  ", h)
        return 1
    print("anonymization scan: code folder clean")
    if not (args.commit or args.push):
        return 0
    git = ["git", "-C", str(CAPSULE)]
    subprocess.run(git + ["add", "-A"], check=True)
    if subprocess.run(git + ["diff", "--cached", "--quiet"]).returncode == 0:
        print("capsule unchanged")
    else:
        subprocess.run(git + ["-c", "user.name=Anonymous Authors", "-c", "user.email=anonymous@example.com",
                              "commit", "-q", "-m", "Update code and measurements for the TPDS submission"],
                       check=True)
    if args.push:
        subprocess.run(git + ["-c", "credential.helper=", "-c",
                              f"credential.https://git.codeocean.com.helper={CREDENTIAL_HELPER}",
                              "push", "-q", "origin", "HEAD:main"], check=True)
        print("pushed to the capsule")
    return 0


def _py38_ok(path: Path) -> bool:
    try:
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 8))
        return True
    except SyntaxError:
        return False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--commit", action="store_true")
    p.add_argument("--push", action="store_true")
    sys.exit(build(p.parse_args(argv)))


if __name__ == "__main__":
    main()
