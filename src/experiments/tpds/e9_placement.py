"""E9 - value versus placement: in-place overwrite control for the fixed-chunk input effects (reviewer point 3).

E1 found input effects of up to 43% at fixed shape and chunk that were not ordered by the
input distribution and were as large between two draws of one distribution as between
distributions (E1c). The manuscript attributed them to the placement of the input buffers,
without testing it. This experiment separates the two directly, on the Triton Mamba-1 scan:

  A  same_buffers    one set of input/output buffers is allocated once; for every
                     (distribution, draw) the values are copied into these buffers in place.
                     Only the values change.
  B  fresh_buffers   every (distribution, draw) lives in its own freshly allocated buffers,
                     all held at once, as in E1. Values and placement change.
  C  placements      one Gaussian draw copied into ten buffer sets: half freshly allocated, half
                     placed at fixed byte offsets inside larger allocations (multiples of 256 B,
                     so the 16-byte alignment that Triton specializes on is unchanged). Only the
                     placement changes. Device pointers, offsets and their residues modulo
                     256 B, 4 KiB and 2 MiB are recorded.

Inputs are the E1 distributions with three draws each (``--draws``), in the ``act`` and ``all``
perturbation modes; lengths {2048, 8192}, batch 1, all six chunks. Every cell is timed in
``--rounds`` rounds with chunks and inputs shuffled per round; SM clock, power and throttle
reasons are recorded before and after every cell, and the exclusive-GPU guard re-times a cell
if another process used the GPU during it. With ``--ref`` the runtime-chunk reference CUDA scan
(``selective_scan_chunk_cuda``) is timed as well (designs A and C).

Also measured, for Theorem 1: the cost of the cheapest device-side decision, one reduction
kernel over the scan input (``sum``), timed alone and as the marginal cost when it precedes
the scan in the same stream, and the launch floor of an empty kernel (CUDA events, 500 repeats).

``--analyze`` reads the JSONL and writes the per-design analysis (two-way ANOVA with rounds as
blocks and Benjamini-Hochberg correction, TOST equivalence, the input-agnostic chunk decision,
the correlation of latency with pointer residues in design C, clock-adjusted tests).

Output: ``e9_placement_<tag>.jsonl`` (resumable), ``e9_placement_<tag>.json`` (analysis).
"""
from __future__ import annotations

import argparse
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.experiments.tpds import kernels as K
from src.experiments.tpds.common import (DEFAULT_TRACE_DIR, MAMBA1_CHUNKS, RESULTS_DIR, JsonlWriter, Logger,
                                         all_gpu_pids, check_disk, cuda_time, env_metadata, foreign_gpu_pids,
                                         gpu_state, host_name, peak_vram_gb, read_jsonl, register_own_gpu_pids,
                                         summarize, wait_for_exclusive_gpu, write_json)
from src.experiments.tpds.e1_analysis import decision_analysis, effect_analysis, equivalence_analysis
from src.experiments.tpds.e1_value_invariance import DISTS, input_names, mamba1_inputs, split_input_name

REF_CHUNKS = (128, 256, 512, 1024, 2048)
TENSORS = ("u", "delta", "A", "B", "C", "D", "z")
RESIDUES = (256, 4096, 2 * 1024 * 1024)
OFFSETS = (256, 4096 + 256, 65536 + 512, (1 << 20) + 1024, (1 << 21) + 4096)   # bytes, multiples of 256


# --------------------------------------------------------------------------- runners
def triton_runner(inp, out, chunk):
    return lambda: K.mamba1_scan(inp["u"], inp["delta"], inp["A"], inp["B"], inp["C"], inp["D"], inp["z"], None,
                                 True, chunk=chunk, out=out)


def ref_runner(inp, chunk):
    import selective_scan_chunk_cuda  # type: ignore

    B = inp["B"].unsqueeze(1)
    C = inp["C"].unsqueeze(1)
    return lambda: selective_scan_chunk_cuda.fwd_with_chunk_size(inp["u"], inp["delta"], inp["A"], B, C, inp["D"],
                                                                 inp["z"], None, True, chunk)[2]


def clone_set(inp):
    """Fresh buffers with the same values."""
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inp.items()}


def offset_set(inp, offset_bytes):
    """Same values, each tensor placed ``offset_bytes`` into a larger allocation."""
    out = {}
    for k, v in inp.items():
        if torch.is_tensor(v):
            n = v.numel() * v.element_size()
            raw = torch.empty(n + offset_bytes, dtype=torch.uint8, device=v.device)
            t = raw[offset_bytes:offset_bytes + n].view(v.dtype).view(v.shape)
            t.copy_(v)
            out[k] = t
        else:
            out[k] = v
    return out


def pointer_info(inp):
    return {k: {"ptr": int(inp[k].data_ptr()), **{f"mod{r}": int(inp[k].data_ptr() % r) for r in RESIDUES}}
            for k in ("u", "delta", "B", "C", "z") if torch.is_tensor(inp.get(k))}


# --------------------------------------------------------------------------- decision-cost probes
@torch.no_grad()
def decision_cost(L, bsz, chunk, repeats, log):
    inp = K.make_mamba1_inputs(bsz, 2048, 16, L, dtype=torch.float16, seed=7)
    out = torch.empty_like(inp["u"])
    scan = triton_runner(inp, out, chunk)
    acc = torch.empty(1, device="cuda", dtype=torch.float32)
    tiny = torch.zeros(1, device="cuda")

    def red():
        torch.sum(inp["u"].view(1, -1), dim=1, dtype=torch.float32, out=acc)

    def red_then_scan():
        red()
        scan()

    res = {"L": L, "batch": bsz, "chunk": chunk, "numel": int(inp["u"].numel()),
           "empty_kernel": summarize(cuda_time(lambda: tiny.add_(0.0), 50, repeats)),
           "reduction": summarize(cuda_time(red, 50, repeats)),
           "scan": summarize(cuda_time(scan, 20, repeats)),
           "reduction_then_scan": summarize(cuda_time(red_then_scan, 20, repeats))}
    res["marginal_ms"] = res["reduction_then_scan"]["mean_ms"] - res["scan"]["mean_ms"]
    log(f"decision cost L={L} b={bsz}: empty kernel {res['empty_kernel']['mean_ms'] * 1e3:.1f} us, reduction "
        f"{res['reduction']['mean_ms'] * 1e3:.1f} us, marginal before the scan {res['marginal_ms'] * 1e3:.1f} us")
    return res


# --------------------------------------------------------------------------- timing
def time_cell(fn, args, log):
    y = fn()
    finite = bool(torch.isfinite(y.float()).all())
    retimed = 0
    while True:
        if args.exclusive_guard:
            waited = wait_for_exclusive_gpu(args.guard_until, log)
            if waited is None:
                return None
            if waited > 0:
                cuda_time(fn, 5 * args.warmup, 1)
        before = foreign_gpu_pids() if args.exclusive_guard else None
        s0 = gpu_state()
        samples = cuda_time(fn, args.warmup, args.repeats)
        s1 = gpu_state()
        after = foreign_gpu_pids() if args.exclusive_guard else None
        if not (before or after):
            return {**summarize(samples), "output_finite": finite, "gpu_state_before": s0, "gpu_state_after": s1,
                    "retimed": retimed}
        retimed += 1
        log(f"foreign GPU processes {before or after}; re-timing")


def build_inputs(dists, mode, L, bsz, seed, trace_dir, log):
    out = {}
    for d in dists:
        dist, s = split_input_name(d, seed)
        inp = mamba1_inputs(dist, mode, L, bsz, s, trace_dir)
        if inp is None:
            log(f"input {d}/{mode}: unavailable (no captured traces); skipped")
            continue
        out[d] = inp
    return out


@torch.no_grad()
def run(args):
    tag = args.tag or host_name()
    log = Logger(f"e9_placement_{tag}")
    gpu_baseline = all_gpu_pids()
    check_disk(args.out_dir, need_gb=0.5, log=log)
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    own = register_own_gpu_pids(gpu_baseline, log=log) if args.exclusive_guard else set()
    meta = env_metadata()
    meta.update({"experiment": "E9", "own_gpu_pids": sorted(own), "args": {k: str(v) for k, v in vars(args).items()}})
    w = JsonlWriter(args.out_dir / f"e9_placement_{tag}.jsonl", meta=meta)
    rng = random.Random(args.seed)
    dists = input_names([d for d in args.inputs.split(",") if d], args.draws - 1)
    families = ["mamba1"] + (["ref"] if args.ref else [])
    for L in [int(x) for x in args.lengths.split(",")]:
        key = f"e9|decision_cost|L{L}|b{args.batch}"
        if key not in w.done:
            w.write({"kind": "decision_cost", "key": key, **decision_cost(L, args.batch, 512, args.cost_repeats, log)})
        for mode in [m for m in args.modes.split(",") if m]:
            fresh = build_inputs(dists, mode, L, args.batch, args.seed, args.trace_dir, log)
            if not fresh:
                continue
            base = next(iter(fresh.values()))
            shared = clone_set(base)                         # design A: one buffer set, values copied in place
            gauss = fresh.get("gaussian", base)
            n_off = min(len(OFFSETS), args.placements // 2)
            offsets = [0] * (args.placements - n_off) + list(OFFSETS[:n_off])
            placements = [clone_set(gauss) if o == 0 else offset_set(gauss, o) for o in offsets]   # design C
            outs = {"A": torch.empty_like(shared["u"]), "C": [torch.empty_like(p["u"]) for p in placements]}
            fresh_outs = {d: torch.empty_like(v["u"]) for d, v in fresh.items()}
            ptrs = {f"p{i}": {"offset_bytes": o, **pointer_info(p)} for i, (o, p) in enumerate(zip(offsets, placements))}
            for fam in families:
                chunks = list(MAMBA1_CHUNKS if fam == "mamba1" else REF_CHUNKS)
                designs = ("A", "B", "C") if fam == "mamba1" else ("A", "C")
                for r in range(args.rounds):
                    order_c = list(chunks)
                    rng.shuffle(order_c)
                    for c in order_c:
                        cells = [(dsg, name) for dsg in designs
                                 for name in (list(fresh) if dsg in "AB" else [f"p{i}" for i in range(len(placements))])]
                        rng.shuffle(cells)
                        for dsg, name in cells:
                            key = f"e9|{fam}|{dsg}|L{L}|b{args.batch}|c{c}|{name}|{mode}|r{r}"
                            if key in w.done:
                                continue
                            if dsg == "A":
                                for k in TENSORS:
                                    if torch.is_tensor(shared.get(k)):
                                        shared[k].copy_(fresh[name][k])
                                inp, out = shared, outs["A"]
                            elif dsg == "B":
                                inp, out = fresh[name], fresh_outs[name]
                            else:
                                i = int(name[1:])
                                inp, out = placements[i], outs["C"][i]
                            fn = triton_runner(inp, out, c) if fam == "mamba1" else ref_runner(inp, c)
                            st = time_cell(fn, args, log)
                            if st is None:
                                w.close()
                                raise SystemExit(3)
                            row = {"kind": "timing", "key": key, "family": f"{fam}:{dsg}", "kernel": fam,
                                   "design": dsg, "L": L, "batch": args.batch, "chunk": c, "input": name,
                                   "mode": mode, "round": r, **st, "peak_vram_gb": peak_vram_gb()}
                            if dsg == "C":
                                row["pointers"] = ptrs[name]
                            elif dsg == "B":
                                row["pointers"] = pointer_info(inp)
                            w.write(row)
                            log(f"{fam}:{dsg} L={L} c={c} {name:12s} {mode} r{r}: {st['mean_ms']:.4f} ms")
            del fresh, shared, placements, outs, fresh_outs
            torch.cuda.empty_cache()
    w.close()
    analyze(args)


# --------------------------------------------------------------------------- analysis
def placement_correlation(rows):
    """Design C: correlation of the per-placement mean latency with the pointer residues of u."""
    out = []
    groups = defaultdict(list)
    for r in rows:
        if r.get("design") == "C" and r.get("pointers"):
            groups[(r["kernel"], r["L"], r["mode"], r["chunk"])].append(r)
    for key, rs in sorted(groups.items()):
        by_p = defaultdict(list)
        for r in rs:
            by_p[r["input"]].append(r["mean_ms"])
        names = sorted(by_p)
        lat = np.array([statistics.fmean(by_p[n]) for n in names])
        rec = {"kernel": key[0], "L": key[1], "mode": key[2], "chunk": key[3], "n_placements": len(names),
               "range_pct": float(100 * (lat.max() - lat.min()) / lat.mean())}
        ptr = {n: next(r["pointers"] for r in rs if r["input"] == n) for n in names}
        rec["offsets_bytes"] = {n: ptr[n].get("offset_bytes", 0) for n in names}
        rec["latency_ms"] = {n: float(v) for n, v in zip(names, lat)}
        for res in RESIDUES:
            x = np.array([ptr[n]["u"][f"mod{res}"] for n in names], dtype=float)
            rec[f"corr_u_mod{res}"] = float(np.corrcoef(x, lat)[0, 1]) if x.std() > 0 and lat.std() > 0 else None
        out.append(rec)
    return out


def analyze(args):
    tag = args.tag or host_name()
    rows = [r for r in read_jsonl(args.out_dir / f"e9_placement_{tag}.jsonl") if r.get("kind") == "timing"]
    costs = [r for r in read_jsonl(args.out_dir / f"e9_placement_{tag}.jsonl") if r.get("kind") == "decision_cost"]
    res = {"experiment": "E9-analysis", "tag": tag, "n_timings": len(rows), "decision_cost": costs, "designs": {}}
    for fam in sorted({r["family"] for r in rows}):
        sel = [r for r in rows if r["family"] == fam]
        eff = effect_analysis(sel)
        dec = decision_analysis(sel)
        eq = equivalence_analysis(sel, dec)
        n_sig = sum(1 for e in eff if e.get("p_input_bh") is not None and e["p_input_bh"] < 0.05)
        n_sig_clock = sum(1 for e in eff if e.get("p_input_clock_adjusted") is not None
                          and e["p_input_clock_adjusted"] < 0.05)
        res["designs"][fam] = {
            "n_configs": len(eff), "n_significant_input_effects_bh": n_sig,
            "n_significant_clock_adjusted_unc": n_sig_clock,
            "max_input_range_pct": max((e["input_range_pct"] for e in eff), default=None),
            "median_input_range_pct": float(np.median([e["input_range_pct"] for e in eff])) if eff else None,
            "tie_aware_agreement": (sum(d["tie_aware_agreement"] * d["n_cells"] for d in dec)
                                    / max(1, sum(d["n_cells"] for d in dec))),
            "equivalence": eq["summary"], "effects": eff, "decision": dec}
    res["placement_correlation"] = placement_correlation(rows)
    write_json(args.out_dir / f"e9_placement_{tag}.json", res)
    for fam, d in res["designs"].items():
        rng = d["max_input_range_pct"]
        print(f"{fam}: {d['n_significant_input_effects_bh']}/{d['n_configs']} significant input effects (BH), "
              f"max range {'n/a' if rng is None else f'{rng:.1f}%'}, tie-aware agreement {d['tie_aware_agreement']:.3f}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lengths", default="2048,8192")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--inputs", default=",".join(DISTS))
    p.add_argument("--draws", type=int, default=3, help="draws per synthetic distribution")
    p.add_argument("--placements", type=int, default=10)
    p.add_argument("--modes", default="act,all")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--cost-repeats", type=int, default=500)
    p.add_argument("--ref", action="store_true", help="also time the runtime-chunk reference CUDA scan (A and C)")
    p.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    p.add_argument("--exclusive-guard", action="store_true")
    p.add_argument("--guard-until", type=float, default=None)
    p.add_argument("--analyze", action="store_true", help="only (re)write the analysis from the JSONL")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--tag", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.analyze:
        analyze(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
