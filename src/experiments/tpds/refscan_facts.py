"""Per-instantiation compile facts of the runtime-chunk reference CUDA scan (review item M2b).

Reads the facts from the binary that was timed, not from a rebuild:

  registers, stack, local memory (spills) and static shared memory per kernel instantiation
      ``cuobjdump --dump-resource-usage`` of the extension module, names demangled with ``c++filt``;
  dynamic shared memory per block
      the stock launcher passes ``Ktraits::kSmemSize + MAX_DSTATE * sizeof(scan_t)`` bytes, which the
      binary does not record; ``smem_sizes.cu`` (next to the extension sources) prints it for every
      instantiation and is compiled here with nvcc against the same mamba_ssm headers (host code only).

Every instantiation is keyed on its kernel traits (threads, items per thread, rows, even length,
variable B, variable C, has z, input type, weight type). CPU only; needs cuobjdump, c++filt and, for the
dynamic shared memory, nvcc and the mamba_ssm source tree the extension was built from (``--mamba-csrc``).

Output: ``refscan_compile_facts_<arch>.json`` in the results directory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# No torch import (and none of the modules that import it): the no-GPU container of the host has a 2 GB
# memory limit, which torch plus nvcc exceed.
ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "src" / "experiments" / "results" / "tpds"
KERNEL_DIR = ROOT / "src" / "kernels" / "mamba1_runtime_chunk"
TRAITS = re.compile(r"Selective_Scan_fwd_kernel_traits<(\d+), (\d+), (\d+), (true|false), (true|false), (true|false), "
                    r"(true|false), ([\w:]+), ([\w:]+)>")
USAGE = re.compile(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")


def torch_include_dir() -> Path:
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("torch not found; pass --torch-include")
    return Path(list(spec.submodule_search_locations)[0]) / "include"


def find_module(explicit: Path | None) -> Path:
    if explicit:
        return explicit
    hits = sorted(KERNEL_DIR.glob("build/lib*/selective_scan_chunk_cuda*.so"))
    if not hits:
        raise SystemExit("runtime-chunk extension module not found; pass --module")
    return hits[0]


def tool(name: str, explicit: str | None = None) -> str:
    for cand in ([explicit] if explicit else []) + [shutil.which(name), f"/usr/local/cuda/bin/{name}"]:
        if cand and Path(cand).exists():
            return cand
    raise SystemExit(f"{name} not found")


def resource_usage(module: Path, cuobjdump: str) -> tuple[list[dict], list[str]]:
    out = subprocess.run([cuobjdump, "--dump-resource-usage", str(module)], capture_output=True, text=True,
                         check=True).stdout
    archs = sorted(set(re.findall(r"arch = (sm_\d+)", out)))
    rows, name = [], None
    for line in out.splitlines():
        m = re.match(r"\s*Function (\S+):\s*$", line)
        if m:
            name = m.group(1)
            continue
        m = USAGE.search(line)
        if m and name is not None:
            rows.append({"mangled": name, "registers": int(m.group(1)), "stack": int(m.group(2)),
                         "static_smem": int(m.group(3)), "local": int(m.group(4))})
            name = None
    demangled = subprocess.run(["c++filt"], input="\n".join(r["mangled"] for r in rows), capture_output=True,
                               text=True, check=True).stdout.splitlines()
    facts = []
    for r, n in zip(rows, demangled):
        t = TRAITS.search(n)
        if not t or "selective_scan_fwd_kernel" not in n:
            continue
        threads, items, nrows = int(t.group(1)), int(t.group(2)), int(t.group(3))
        facts.append({**r, "name": n, "threads": threads, "items": items, "rows": nrows, "chunk": threads * items,
                      "even_len": t.group(4) == "true", "variable_b": t.group(5) == "true",
                      "variable_c": t.group(6) == "true", "has_z": t.group(7) == "true",
                      "input_t": t.group(8), "weight_t": t.group(9)})
    return facts, archs


def dynamic_smem(mamba_csrc: Path, nvcc: str, torch_include: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "smem_sizes"
        cmd = [nvcc, "-std=c++17", "-O0", f"-I{mamba_csrc}", f"-I{torch_include}",
               f"-I{torch_include / 'torch' / 'csrc' / 'api' / 'include'}", "--expt-relaxed-constexpr",
               "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
               "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", str(KERNEL_DIR / "smem_sizes.cu"), "-o", str(exe)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    sizes = {}
    for line in out.splitlines():
        thr, items, even, vb, vc, z, smem = map(int, line.split("\t"))
        sizes[(thr, items, bool(even), bool(vb), bool(vc), bool(z))] = smem
    return sizes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--module", type=Path, default=None, help="extension module (default: the in-place build)")
    p.add_argument("--mamba-csrc", type=Path, default=None,
                   help="mamba_ssm csrc/selective_scan directory (needed for the dynamic shared memory)")
    p.add_argument("--torch-include", type=Path, default=None, help="torch include directory (default: installed torch)")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv)
    module = find_module(args.module)
    facts, archs = resource_usage(module, tool("cuobjdump"))
    res = {"experiment": "M2b reference-scan compile facts", "created_utc": utc_now(), "module": module.name,
           "archs": archs, "n_instantiations": len(facts), "source": "cuobjdump --dump-resource-usage"}
    if args.mamba_csrc:
        inc = args.torch_include or torch_include_dir()
        sizes = dynamic_smem(args.mamba_csrc, tool("nvcc"), inc)
        for f in facts:
            f["dynamic_smem"] = sizes.get((f["threads"], f["items"], f["even_len"], f["variable_b"], f["variable_c"],
                                           f["has_z"]))
        res["dynamic_smem_source"] = "smem_sizes.cu (Ktraits::kSmemSize + MAX_DSTATE * sizeof(scan_t))"
    res["instantiations"] = facts
    arch = archs[0] if len(archs) == 1 else "multi"
    out = args.out_dir / f"refscan_compile_facts_{arch.replace('sm_', 'sm')}.json"
    write_json(out, res)
    sel = [f for f in facts if f["input_t"].endswith("Half") and f["variable_b"] and f["variable_c"] and f["has_z"]]
    for f in sorted(sel, key=lambda f: (f["threads"], f["items"], f["even_len"])):
        print(f"threads {f['threads']:4d} items {f['items']:3d} chunk {f['chunk']:5d} even {int(f['even_len'])}: "
              f"{f['registers']} regs, local {f['local']} B, stack {f['stack']} B, static smem {f['static_smem']} B, "
              f"dynamic smem {f.get('dynamic_smem')} B")
    print(f"{len(facts)} instantiations ({archs}) -> {out}")


if __name__ == "__main__":
    main()
