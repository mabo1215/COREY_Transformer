"""E16c - counter-free test of the placement explanation on the RTX PRO 6000 (review M6b).

Nsight Compute counters are not accessible in the rented container (ERR_NVGPUCTRPERM), so E16b
could not measure L2 hit rates directly. This experiment tests the same hypothesis by
intervention instead of observation: if the fixed-chunk latency differences between buffer
placements come from which lines of the working set stay resident in L2 between repeats, they
must change when L2 is flushed before every repeat, and they must be absent when the working
set fits in L2.

Design (Triton Mamba-1 scan, batch 1, all six chunks, ``act`` mode):
  A  same buffers   the eight E1 distributions copied in place into one buffer set (values only)
  C  placements     one Gaussian draw in ``--placements`` buffer sets: half freshly allocated,
                    half at fixed byte offsets (multiples of 256 B, up to 64 MiB) inside larger
                    allocations (placement only). Pointers and residues are recorded.
Each cell is timed with a warm L2 (back-to-back repeats, as in E1/E9/E16) and with a cold L2
(a 128 MiB overwrite before every repeat, outside the timed region). Lengths {8192, 2048}: at
L=8192 the scan's working set (u, delta, z, out: 4 x 32 MiB in fp16 at d=2048) is about the size
of the 128 MiB L2 of GB202; at L=2048 it is a quarter of it. Rounds are shuffled; the
exclusive-GPU guard and SM clock/power/throttle reasons per cell are as in E9.

Analysis per (design, L2 state, L, chunk): two-way ANOVA (inputs or placements x rounds, BH),
range in percent, the correlation of per-placement latency with pointer residues, the stability of
the per-placement ranking across rounds (Spearman), and the warm-to-cold change of the
placement range. Expected under the placement explanation: C shows a stable placement effect with
a warm L2 at L=8192 that shrinks or reorders with a cold L2 and at L=2048; A shows no effect.

Output: ``e16c_l2_placement_<tag>.jsonl`` (resumable) and ``.json`` (analysis).
"""
from __future__ import annotations

import argparse
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.experiments.tpds.common import (DEFAULT_TRACE_DIR, MAMBA1_CHUNKS, RESULTS_DIR, JsonlWriter, Logger,
                                         all_gpu_pids, check_disk, env_metadata, host_name, peak_vram_gb,
                                         read_jsonl, register_own_gpu_pids, write_json)
from src.experiments.tpds.e1_analysis import effect_analysis

OFFSETS = (256, 4096 + 256, 65536 + 512, (1 << 20) + 1024, (1 << 21) + 4096, (1 << 22) + 256,
           (1 << 23) + 768, (1 << 24) + 256, (1 << 25) + 512, (1 << 26) + 256)   # bytes, multiples of 256
RESIDUES = (256, 4096, 1 << 21, 1 << 25)


def pointer_info(inp):
    return {k: {"ptr": int(inp[k].data_ptr()), **{f"mod{r}": int(inp[k].data_ptr() % r) for r in RESIDUES}}
            for k in ("u", "delta", "B", "C", "z") if hasattr(inp.get(k), "data_ptr")}


def run(args):
    import torch

    from src.experiments.tpds import e9_placement as E9
    from src.experiments.tpds.e1_value_invariance import input_names

    tag = args.tag or host_name()
    log = Logger(f"e16c_l2_placement_{tag}")
    gpu_baseline = all_gpu_pids()
    check_disk(args.out_dir, need_gb=0.5, log=log)
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    own = register_own_gpu_pids(gpu_baseline, log=log) if args.exclusive_guard else set()
    meta = env_metadata()
    meta.update({"experiment": "E16c", "own_gpu_pids": sorted(own), "l2_bytes": _l2_bytes(),
                 "args": {k: str(v) for k, v in vars(args).items()}})
    w = JsonlWriter(args.out_dir / f"e16c_l2_placement_{tag}.jsonl", meta=meta)
    rng = random.Random(args.seed)
    dists = input_names([d for d in args.inputs.split(",") if d], 0)
    l2_states = [s for s in args.l2.split(",") if s]
    chunks = [int(c) for c in args.chunks.split(",")]
    with torch.no_grad():
        for L in [int(x) for x in args.lengths.split(",")]:
            for mode in [m for m in args.modes.split(",") if m]:
                fresh = E9.build_inputs(dists, mode, L, args.batch, args.seed, args.trace_dir, log)
                if not fresh:
                    continue
                shared = E9.clone_set(next(iter(fresh.values())))
                gauss = fresh.get("gaussian", next(iter(fresh.values())))
                n_off = min(len(OFFSETS), args.placements // 2)
                offsets = [0] * (args.placements - n_off) + list(OFFSETS[:n_off])
                placements = [E9.clone_set(gauss) if o == 0 else E9.offset_set(gauss, o) for o in offsets]
                out_a = torch.empty_like(shared["u"])
                out_c = [torch.empty_like(p["u"]) for p in placements]
                ptrs = {f"p{i}": {"offset_bytes": o, **pointer_info(p)}
                        for i, (o, p) in enumerate(zip(offsets, placements))}
                for r in range(args.rounds):
                    order = [(c, s) for c in chunks for s in l2_states]
                    rng.shuffle(order)
                    for c, l2 in order:
                        cells = [("A", n) for n in fresh] + [("C", f"p{i}") for i in range(len(placements))]
                        rng.shuffle(cells)
                        for dsg, name in cells:
                            key = f"e16c|{dsg}|{l2}|L{L}|b{args.batch}|c{c}|{name}|{mode}|r{r}"
                            if key in w.done:
                                continue
                            if dsg == "A":
                                for k in E9.TENSORS:
                                    if torch.is_tensor(shared.get(k)):
                                        shared[k].copy_(fresh[name][k])
                                inp, out = shared, out_a
                            else:
                                inp, out = placements[int(name[1:])], out_c[int(name[1:])]
                            fn = E9.triton_runner(inp, out, c)
                            targs = argparse.Namespace(**{**vars(args), "cold": l2 == "cold"})
                            st = time_cell(fn, targs, log)
                            if st is None:
                                w.close()
                                raise SystemExit(3)
                            row = {"kind": "timing", "key": key, "family": f"mamba1:{dsg}:{l2}", "design": dsg,
                                   "l2": l2, "L": L, "batch": args.batch, "chunk": c, "input": name, "mode": mode,
                                   "round": r, **st, "peak_vram_gb": peak_vram_gb()}
                            if dsg == "C":
                                row["pointers"] = ptrs[name]
                            w.write(row)
                        log(f"L={L} {mode} r{r} c={c} L2 {l2}: done")
                del fresh, shared, placements, out_a, out_c
                torch.cuda.empty_cache()
    w.close()
    analyze(args)


def _l2_bytes():
    try:
        import torch

        return int(getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0)) or None
    except Exception:  # noqa: BLE001
        return None


def time_cell(fn, args, log):
    """E9's guarded timing, with an L2 overwrite before every repeat when ``args.cold``."""
    from src.experiments.tpds.common import (cuda_time, foreign_gpu_pids, gpu_state, summarize,
                                             wait_for_exclusive_gpu)
    import torch

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
        samples = cuda_time(fn, args.warmup, args.repeats, flush_l2=args.cold)
        s1 = gpu_state()
        after = foreign_gpu_pids() if args.exclusive_guard else None
        if not (before or after):
            return {**summarize(samples), "output_finite": finite, "gpu_state_before": s0, "gpu_state_after": s1,
                    "retimed": retimed}
        retimed += 1
        log(f"foreign GPU processes {before or after}; re-timing")


# --------------------------------------------------------------------------- analysis
def _spearman(a, b):
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def placement_analysis(rows):
    """Design C per (l2, L, mode, chunk): per-placement means, residue correlations, round-to-round rank stability."""
    groups = defaultdict(list)
    for r in rows:
        if r["design"] == "C":
            groups[(r["l2"], r["L"], r["mode"], r["chunk"])].append(r)
    out = []
    for (l2, L, mode, c), rs in sorted(groups.items()):
        by_p = defaultdict(dict)
        for r in rs:
            by_p[r["input"]][r["round"]] = r
        names = sorted(by_p, key=lambda n: int(n[1:]))
        rounds = sorted(set.intersection(*[set(v) for v in by_p.values()]))
        if len(names) < 3 or not rounds:
            continue
        X = np.array([[by_p[n][k]["mean_ms"] for k in rounds] for n in names])
        lat = X.mean(axis=1)
        rec = {"l2": l2, "L": L, "mode": mode, "chunk": c, "n_placements": len(names), "n_rounds": len(rounds),
               "range_pct": float(100 * (lat.max() - lat.min()) / lat.mean()),
               "latency_ms": {n: float(v) for n, v in zip(names, lat)},
               "offsets_bytes": {n: by_p[n][rounds[0]]["pointers"].get("offset_bytes", 0) for n in names}}
        rho = [_spearman(X[:, i], X[:, j]) for i in range(len(rounds)) for j in range(i + 1, len(rounds))]
        rho = [v for v in rho if v is not None]
        rec["round_rank_spearman_mean"] = float(np.mean(rho)) if rho else None
        for res in RESIDUES:
            x = np.array([by_p[n][rounds[0]]["pointers"]["u"][f"mod{res}"] for n in names], dtype=float)
            rec[f"corr_u_mod{res}"] = float(np.corrcoef(x, lat)[0, 1]) if x.std() > 0 and lat.std() > 0 else None
        out.append(rec)
    return out


def warm_cold_comparison(place):
    """Per (L, mode, chunk): placement range warm vs cold and whether the per-placement ordering survives."""
    idx = {(p["l2"], p["L"], p["mode"], p["chunk"]): p for p in place}
    out = []
    for (l2, L, mode, c), warm in sorted(idx.items()):
        cold = idx.get(("cold", L, mode, c))
        if l2 != "warm" or cold is None:
            continue
        names = sorted(set(warm["latency_ms"]) & set(cold["latency_ms"]), key=lambda n: int(n[1:]))
        out.append({"L": L, "mode": mode, "chunk": c, "range_warm_pct": warm["range_pct"],
                    "range_cold_pct": cold["range_pct"],
                    "warm_cold_rank_spearman": _spearman([warm["latency_ms"][n] for n in names],
                                                         [cold["latency_ms"][n] for n in names])})
    return out


def analyze(args):
    tag = args.tag or host_name()
    rows = [r for r in read_jsonl(args.out_dir / f"e16c_l2_placement_{tag}.jsonl") if r.get("kind") == "timing"]
    eff = effect_analysis(rows)
    place = placement_analysis(rows)
    res = {"experiment": "E16c-analysis", "tag": tag, "n_timings": len(rows), "families": {},
           "placements": place, "warm_vs_cold": warm_cold_comparison(place), "effects": eff}
    for fam in sorted({e["family"] for e in eff}):
        for L in sorted({e["L"] for e in eff if e["family"] == fam}):
            sel = [e for e in eff if e["family"] == fam and e["L"] == L]
            res["families"][f"{fam}|L{L}"] = {
                "n_configs": len(sel),
                "n_significant_bh": sum(1 for e in sel if e.get("p_input_bh") is not None and e["p_input_bh"] < 0.05),
                "median_range_pct": float(np.median([e["input_range_pct"] for e in sel])),
                "max_range_pct": float(max(e["input_range_pct"] for e in sel))}
    write_json(args.out_dir / f"e16c_l2_placement_{tag}.json", res)
    for k, d in res["families"].items():
        print(f"{k}: {d['n_significant_bh']}/{d['n_configs']} significant (BH), median range "
              f"{d['median_range_pct']:.1f}%, max {d['max_range_pct']:.1f}%")
    for d in res["warm_vs_cold"]:
        rho = d["warm_cold_rank_spearman"]
        print(f"C L={d['L']} c={d['chunk']}: placement range warm {d['range_warm_pct']:.1f}% cold "
              f"{d['range_cold_pct']:.1f}%, rank rho {'n/a' if rho is None else f'{rho:.2f}'}")
    return res


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lengths", default="8192,2048")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--inputs", default="uniform,gaussian,laplace,sparse50,sparse10,sparse02,constant,real")
    p.add_argument("--placements", type=int, default=20)
    p.add_argument("--modes", default="act")
    p.add_argument("--chunks", default=",".join(str(c) for c in MAMBA1_CHUNKS))
    p.add_argument("--l2", default="warm,cold", help="L2 states to time: warm, cold")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    p.add_argument("--exclusive-guard", action="store_true")
    p.add_argument("--guard-until", type=float, default=None)
    p.add_argument("--analyze", action="store_true")
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
