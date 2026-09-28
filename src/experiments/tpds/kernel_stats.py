"""Compile-time statistics of the Triton Mamba-1 scan kernel (input to the T1 cost model).

For every launch configuration of the E4 search space (BLOCK_T x BLOCK_D x
num_warps x num_stages, BLOCK_T * BLOCK_D <= 16384) and for aligned and
unaligned sequence lengths, the kernel is compiled for a target GPU
architecture and ptxas reports the register count, the spill traffic and the
static shared memory. The distributed layout of the [BLOCK_D, BLOCK_T] tiles
is read from the TTGIR. Nothing is launched: the native architecture needs a
CUDA device only to build the specialization, and other architectures are
cross-compiled (for example sm_120 on an sm_86 workstation).

Alignment: Triton specializes integer arguments that are divisible by 16.
The row stride of the (b, d, L) scan inputs is L, so a sequence length that
is not a multiple of 16 compiles to a different kernel (no vectorized loads).
Both variants are recorded (``aligned`` = L % 16 == 0).

Output: ``kernel_stats_sm<arch>.json`` in the results directory (and the resumable
per-configuration log ``kernel_stats_sm<arch>.jsonl``).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.runtime import driver

from src.experiments.tpds import kernels as K
from src.experiments.tpds.common import (MAMBA1_CHUNKS, RESULTS_DIR, JsonlWriter, dedupe_by_key, env_metadata,
                                         read_jsonl, write_json)

BLOCK_DS = (4, 8, 16, 32)
NUM_WARPS = (1, 2, 4, 8)
NUM_STAGES = (1, 2)
MAX_TILE = 16384
_PTXAS_RE = re.compile(r"Used (\d+) registers")
_SPILL_RE = re.compile(r"(\d+) bytes spill stores, (\d+) bytes spill loads")
_STACK_RE = re.compile(r"(\d+) bytes stack frame")
_SMEM_RE = re.compile(r"(\d+) bytes smem")
_BLOCKED_RE = re.compile(r"#blocked\d* = #ttg\.blocked<\{sizePerThread = \[([\d, ]+)\], threadsPerWarp = \[([\d, ]+)\], "
                         r"warpsPerCTA = \[([\d, ]+)\]")


def search_space() -> list[tuple[int, int, int, int]]:
    return [(c, bd, nw, ns) for c in MAMBA1_CHUNKS for bd in BLOCK_DS for nw in NUM_WARPS for ns in NUM_STAGES
            if c * bd <= MAX_TILE]


def _ptxas_path(arch: int) -> str:
    base = Path(triton.__file__).parent / "backends" / "nvidia" / "bin"
    if arch >= 100 and (base / "ptxas-blackwell").exists():
        return str(base / "ptxas-blackwell")
    return str(base / "ptxas")


def ptxas_stats(ptx: str, arch: int) -> dict:
    gpu = f"sm_{arch}a" if arch >= 90 else f"sm_{arch}"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "k.ptx"
        src.write_text(ptx)
        proc = subprocess.run([_ptxas_path(arch), "-v", f"--gpu-name={gpu}", str(src), "-o", str(Path(tmp) / "k.cubin")],
                              capture_output=True, text=True)
    txt = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return {"ptxas_error": txt[-500:]}
    regs = _PTXAS_RE.search(txt)
    spill = _SPILL_RE.search(txt)
    stack = _STACK_RE.search(txt)
    smem = _SMEM_RE.search(txt)
    return {"regs": int(regs.group(1)) if regs else None,
            "spill_store_bytes": int(spill.group(1)) if spill else 0,
            "spill_load_bytes": int(spill.group(2)) if spill else 0,
            "stack_bytes": int(stack.group(1)) if stack else 0,
            "ptxas_smem_bytes": int(smem.group(1)) if smem else 0}


def tile_layout(ttgir: str) -> dict | None:
    """First two-dimensional blocked layout of the TTGIR (the [BLOCK_D, BLOCK_T] tiles)."""
    for m in _BLOCKED_RE.finditer(ttgir):
        spt, tpw, wpc = ([int(x) for x in g.split(",")] for g in m.groups())
        if len(spt) == 2:
            return {"size_per_thread": spt, "threads_per_warp": tpw, "warps_per_cta": wpc}
    return None


def compile_config(arch: int, native: bool, L: int, chunk: int, block_d: int, num_warps: int, num_stages: int,
                   dim: int = 64, n: int = 16):
    inp = K.make_mamba1_inputs(1, dim, n, L, device="cuda")
    u, dt, A, B, C, D, z = (inp[k] for k in ("u", "delta", "A", "B", "C", "D", "z"))
    out = torch.empty_like(u)
    target = GPUTarget("cuda", arch, 32)
    orig = driver.active.get_current_target
    driver.active.get_current_target = lambda: target
    try:
        ck = K._mamba1_chunked_scan_kernel.warmup(
            u, dt, A, B, C, D, z, u, out, L, dim,
            *u.stride(), *dt.stride(), *A.stride(), *B.stride(), *C.stride(), *z.stride(), *out.stride(),
            N=n, BLOCK_N=triton.next_power_of_2(n), BLOCK_T=chunk, BLOCK_D=block_d, HAS_D=True, HAS_Z=True,
            HAS_DBIAS=False, SOFTPLUS=True, num_warps=num_warps, num_stages=num_stages, grid=(1,))
    finally:
        driver.active.get_current_target = orig
    rec = ptxas_stats(ck.asm["ptx"], arch)
    rec["shared_bytes"] = int(getattr(ck.metadata, "shared", 0) or 0)
    rec["layout"] = tile_layout(ck.asm["ttgir"])
    if native:
        ck._init_handles()
        rec["driver_regs"] = int(ck.n_regs)
        rec["driver_spills"] = int(ck.n_spills)
    return rec


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archs", default=None, help="comma list of compute capabilities, e.g. 86,120 (default: native)")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    return p.parse_args(argv)


def _key(aligned, chunk, bd, nw, ns) -> str:
    return f"{'al' if aligned else 'un'}|C{chunk}|BD{bd}|nw{nw}|ns{ns}"


def main(argv=None):
    args = parse_args(argv)
    native = driver.active.get_current_target().arch
    archs = [int(a) for a in args.archs.split(",")] if args.archs else [native]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for arch in archs:
        out = args.out_dir / f"kernel_stats_sm{arch}.json"
        part = args.out_dir / f"kernel_stats_sm{arch}.jsonl"
        meta = env_metadata()
        meta.update({"experiment": "kernel_stats", "target_arch": arch, "native_arch": native,
                     "compile_only": True, "dim_for_compile": 64, "d_state": 16})
        writer = JsonlWriter(part, meta=meta)   # resumable: completed keys are skipped on restart
        rows = [r for r in read_jsonl(part, kind="config")]
        for aligned, L in ((True, 2048), (False, 2047)):
            for chunk, bd, nw, ns in search_space():
                key = _key(aligned, chunk, bd, nw, ns)
                if key in writer.done:
                    continue
                rec = compile_config(arch, arch == native, L, chunk, bd, nw, ns)
                rec.update({"kind": "config", "key": key, "block_t": chunk, "block_d": bd, "num_warps": nw,
                            "num_stages": ns, "aligned": aligned})
                writer.write(rec)
                rows.append(rec)
                print(f"sm{arch} {key} regs={rec.get('regs')} spill={rec.get('spill_store_bytes')} "
                      f"layout={rec.get('layout')}", flush=True)
        writer.close()
        rows = dedupe_by_key(rows)
        write_json(out, {"meta": meta, "configs": rows})
        print(f"wrote {out} ({len(rows)} configurations)")


if __name__ == "__main__":
    main()
