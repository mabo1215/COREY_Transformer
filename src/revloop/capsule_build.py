"""Build the Code Ocean capsule (submodule ``codeocean/``) from this repository.

The capsule gets only what the paper needs: the entry scripts listed in ENTRIES, every
local module they import (resolved from the source, so the package is complete), the
TPDS result files, and the small LongBench prompt subsets. Everything else in the
capsule is replaced. The package is written for anonymous review: the metadata authors
license holder is anonymized, and a scan for identifying strings must pass before
anything is committed. The capsule metadata (title, authors) must be edited in the Code
Ocean metadata UI; this script never touches metadata/metadata.yml.

Usage (repository root):
    python src/revloop/capsule_build.py            # rebuild codeocean/ in place, scan, no commit
    python src/revloop/capsule_build.py --commit   # also commit inside codeocean/ (anonymous identity)
    python src/revloop/capsule_build.py --push     # commit and push to the capsule's git remote

The Code Ocean environment (``environment/``) is left as it is; the capsule's run script
only needs NumPy, pandas and Matplotlib (it recomputes analyses, tables and figures on the
CPU from the stored measurements).
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import re
import shutil
import subprocess
import sys
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
    "src/experiments/tpds/*.py",                       # kernel-level study E1-E7, analyses, tables, figures
    "src/corey_selective_scan_dispatch.py",            # imported by name by the routed study
    "src/experiments/run_integrated_end_to_end.py",    # routed end-to-end study (patched live kernel)
    "src/experiments/run_scheduler_ablation_matrix.py",
    "src/experiments/run_routed_quality_check.py",     # output agreement on LongBench
    "src/experiments/run_active_hook_integration.py",  # inline-hook overhead
    "src/experiments/run_active_hook_real_benchmark.py",
    "src/experiments/run_longbench_inference.py",      # passive hook, per-prompt entropy
    "src/experiments/bin_count_sensitivity.py",
    "src/experiments/run_real_activation_sinkhorn.py", # Hadamard analysis (supplement)
    "src/experiments/run_heterogeneous_corpus.py",     # prompt-regime corpora (supplement)
    "src/experiments/run_real_workload_diversity_h800.py",
    "src/experiments/summarize_entropy_distribution.py",
]
# Data and results shipped with the code (small files only).
EXTRA = [
    "src/experiments/results/tpds/*.json",
    "src/experiments/results/tpds/*.jsonl",
    "src/experiments/results/tpds/*.csv",
    "src/experiments/results/tpds/*.log",
    "src/data/longbench_subset/**/*",
    "src/data/longbench_subset_remote_min/**/*",
    "src/data/longbench_push120/**/*",
    "src/requirements.txt",
    "src/kernels/mamba1_runtime_chunk/**/*",          # reference CUDA scan with a runtime chunk (source)
]
# Generated run outputs (src/outputs/, the capsule's own results) are never packaged.
# e17_workbound_vgpu3090.*: first E17c run, rows lost to a git autostash; superseded by *_vgpu3090full.*
EXCLUDE = ["*_e1b.*", "*/smoke/*", "*localsmoke*", "*/__pycache__/*", "src/outputs/*",
           "*/e17_workbound_vgpu3090.*"]

# Identifying strings that must not appear in the package (checked case-insensitively).
IDENTIFYING = [r"mabo", r"bo\s+ma\b", r"weiqi", r"jinsong", r"aut\.ac\.nz", r"\brcn\d{3,}", r"auckland",
               r"bobsteam", r"c:[\\/]+source", r"/mnt/c/", r"autodl", r"seetacloud", r"westb",
               r"ssh\s+-p", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", r"hkvlm", r"github\.com/mabo"]
IP_ALLOW = {"0.0.0.0", "127.0.0.1"}
# Public benchmark text under src/data/ is checked for personal identifiers only, not for
# place names or numeric patterns that occur in ordinary prose.
DATA_ONLY_PATTERNS = {r"mabo", r"bo\s+ma\b", r"weiqi", r"jinsong", r"aut\.ac\.nz", r"\brcn\d{3,}", r"bobsteam"}


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
        if not f.is_file() or ".git" in f.relative_to(root).parts:
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


RUN_SCRIPT = """#!/usr/bin/env bash
# Reproducible run: recompute the analyses, tables and figures of the paper from the stored
# measurements (CPU only). The GPU experiments that produced the measurements are described
# in README.md.
set -ex
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY=$(command -v python3 || command -v python)
OUT="${RESULTS_DIR:-../results}"
WORK="$OUT/analysis"
mkdir -p "$WORK" "$OUT/figures" "$OUT/tables"
cp src/experiments/results/tpds/*.json src/experiments/results/tpds/*.csv "$WORK"/
cp src/experiments/results/tpds/e17_workbound_vgpu3090full.jsonl "$WORK"/
for tag in pro6000 vgpu3090 pro6000_e1c; do
  "$PY" -u -m src.experiments.tpds.e1_analysis --in-dir "$WORK" --tag "$tag"
done
for host in pro6000 vgpu3090; do
  "$PY" -u -m src.experiments.tpds.e5_breakeven --in-dir "$WORK" --host "$host"
  "$PY" -u -m src.experiments.tpds.entropy_band --in-dir "$WORK" --host "$host"
  "$PY" -u -m src.experiments.tpds.make_tables --in-dir "$WORK" --host "$host" validation latency effects autotune overhead > "$OUT/tables/tables_$host.tex"
  "$PY" -u -m src.experiments.tpds.make_figures --in-dir "$WORK" --host "$host" --out-dir "$OUT/figures/$host"
done
"$PY" -u -m src.experiments.tpds.make_tables --in-dir "$WORK" --host pro6000 redraw > "$OUT/tables/table_redraw_pro6000.tex"
"$PY" -u -m src.experiments.tpds.make_tables --in-dir "$WORK" --host vgpu3090 reference > "$OUT/tables/table_reference_vgpu3090.tex"
# Production kernels at larger widths (E17): synthetic sweep and captured Mamba-2 2.7B / Gated DeltaNet 1.4B calls
"$PY" -u -m src.experiments.tpds.e17_analysis --in-dir "$WORK" --tag vgpu3090full \
  --per-call e2_mamba2_vgpu3090_2p7b_r3.csv,e2_gdn_vgpu3090_1p4b_r3.csv
"""

README = """# {title}

Code and measurements for the paper. The reproducible run of this capsule recomputes the
analyses, tables and figures from the stored measurements on the CPU; it does not need a GPU.

## Layout

- `src/experiments/tpds/` - kernels and experiments: Triton Mamba-1 selective scan and kernel
  wrappers (`kernels.py`); value invariance and placement controls (`e1_value_invariance.py`,
  `e1_analysis.py`, `e9_placement.py`, `e16_counters.py`); captured prefill traces
  (`capture_traces.py`, `capture_gdn_traces.py`) and per-call chunk sweeps (`e2_oracle_sweep.py`);
  predictors (`e3_predictors.py`); auto-tuning (`e4_autotune.py`); compile facts and the cost
  model (`kernel_stats.py`, `cost_model.py`, `cost_model_baselines.py`); break-even and key
  churn (`e5_breakeven.py`, `e5b_key_churn.py`, `serving_key_churn.py`); the reference CUDA scan
  (`e7_reference_scan.py`, `e8_refscan.py`); routed end-to-end study (`e10_routed_interleaved.py`);
  tensor-parallel widths (`e12_tp_widths.py`, `e12_analysis.py`); production kernels at larger
  widths (`e17_workbound.py`, `e17_analysis.py`); oracle headroom and timing noise
  (`headroom_noise.py`); tables and figures (`make_tables.py`, `make_figures.py`).
- `src/kernels/mamba1_runtime_chunk/` - the reference Mamba-1 CUDA selective scan rebuilt with the
  chunk as a runtime argument.
- `src/experiments/results/tpds/` - measurements and analyses, tagged by GPU: `*_pro6000*`
  (RTX PRO 6000), `*_vgpu3090*` (RTX 3090), `*_h800*` (H800); `_r3spin` and `_r3` tags are
  device-time re-timings.
- `src/experiments/run_*.py`, `src/corey_selective_scan_dispatch.py`, `src/algorithms/` -
  earlier routed end-to-end study, output agreement on LongBench, inline-hook overhead, and
  supplementary analyses.
- `src/data/` - LongBench prompt subsets used by the trace capture and the end-to-end runs.

## Reproducible run (CPU)

`run` recomputes, into `../results/`: the value-invariance analyses
(`analysis/e1_analysis_*.json`), the break-even model (`analysis/e5_*.json`), the entropy band
(`analysis/entropy_band_*.json`), the analysis of the production kernels at larger widths
(`analysis/e17_analysis_vgpu3090full.json`), the supplementary tables (`tables/`) and the
figures (`figures/`).

## GPU experiments

The measurements were taken with Python 3.12, PyTorch 2.8, Triton 3.4, Transformers 5.6.2,
FLA 0.5.2 and mamba_ssm 2.2/2.3 on an RTX PRO 6000 (Blackwell) and an RTX 3090 (Ampere).
Kernel timing needs a GPU without other processes; `--exclusive-guard` waits for one.

```bash
export PYTHONPATH=$PWD
# captured prefill traces of the 370M Mamba-1 and Mamba-2 checkpoints
python -m src.experiments.tpds.capture_traces --help
# E1 value invariance (and the control run: --rounds 10 --extra-seeds 2 --record-gpu-state)
python -m src.experiments.tpds.e1_value_invariance --families mamba1,mamba2,gdn --exclusive-guard
python -m src.experiments.tpds.e2_oracle_sweep --help
python -m src.experiments.tpds.e3_predictors --help
python -m src.experiments.tpds.e4_autotune --help
python -m src.experiments.tpds.e5_breakeven --host <tag>
python -m src.experiments.tpds.e7_reference_scan --exclusive-guard   # needs mamba_ssm CUDA kernels
```

The routed end-to-end study needs a mamba_ssm build whose selective-scan kernel accepts the
chunk as a runtime argument (see `src/corey_selective_scan_dispatch.py`).
"""

# Host names and absolute host paths recorded in result metadata are replaced in the copies.
SANITIZE = [(re.compile(r"autodl-container-[A-Za-z0-9-]+"), "gpu-host"),
            (re.compile(r"/root/autodl-tmp"), "/data"),
            (re.compile(r"/autodl-fs/data"), "/shared"),
            # analyses run on the authors' workstation record its host name, checkout path and kernel
            (re.compile(r"bobsteam\d*", re.I), "workstation"),
            (re.compile(r"/mnt/c/source/[A-Za-z_]+", re.I), "/repo"),
            (re.compile(r"Linux-\d+\.\d+\.\d+(?:\.\d+)?[A-Za-z0-9_.+-]*"), "Linux")]


def sanitize_results(code: Path) -> int:
    n = 0
    roots = [code / "src" / "experiments" / "results", code / "src" / "outputs"]
    for f in (p for r in roots if r.exists() for p in r.rglob("*")):
        if f.is_file() and f.suffix in {".json", ".jsonl", ".csv", ".log"}:
            text = f.read_text(encoding="utf-8", errors="ignore")
            new = text
            for rx, rep in SANITIZE:
                new = rx.sub(rep, new)
            if new != text:
                f.write_text(new, encoding="utf-8")
                n += 1
    return n


LICENSE_HOLDER = re.compile(r"^(Copyright \(c\) \d{4}) .*$", re.M)


def build(args) -> int:
    if not (CAPSULE / ".git").exists():
        print("codeocean/ is not initialised (git submodule update --init codeocean)")
        return 2
    files = closure(expand(ENTRIES)) | expand(EXTRA)
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
        dst = code / f.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
    for pkg in {p.parent for p in files if p.suffix == ".py"}:  # every package directory importable
        d = code / pkg.relative_to(ROOT)
        while d != code:
            (d / "__init__.py").touch(exist_ok=True)
            d = d.parent
    print(f"sanitized host names and paths in {sanitize_results(code)} result files")
    (code / "run").write_text(RUN_SCRIPT)
    (code / "run").chmod(0o755)
    (code / "README.md").write_text(README.format(title=TITLE))
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
    if hits:
        print(f"anonymization scan: {len(hits)} hit(s); nothing committed")
        for h in hits[:200]:
            print("  ", h)
        return 1
    print("anonymization scan: clean")
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
