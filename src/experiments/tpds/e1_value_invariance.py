"""E1 - value invariance of chunked-scan latency (items C1, B4, C6).

For fixed shape, dtype and chunk, time the recurrence-preserving scan kernels
on inputs with very different value distributions (and hence very different
activation entropy): uniform, Gaussian, Laplace, sparse (50/10/2 % nonzero),
constant, and real captured activations. Families: Mamba-1 selective scan
(Triton, chunk = time tile), Mamba-2 SSD (chunk_size), Gated DeltaNet
(chunk_size in {16, 32, 64}).

Two perturbation modes:
  act - only the activation tensor that the entropy scheduler reads (Mamba-1 u,
        Mamba-2 x, GDN v) follows the distribution; other inputs are Gaussian.
  all - every data input follows the distribution (mapped to its valid range).

Controls (optional): ``--extra-seeds K`` adds K further random draws of every
synthetic distribution (inputs named ``<dist>#s<k>``), so the spread between two
draws of the same distribution gives a null baseline for input effects;
``--record-gpu-state`` stores SM clock, power, temperature and throttle reasons
before and after every timing, to separate device-state drift from value effects.

Output: ``<out-dir>/e1_<host>.jsonl`` (resumable raw rows) and
``<out-dir>/e1_<host>.json`` (metadata, validation, all rows, summary).
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
from pathlib import Path

import torch

from src.algorithms.torch_entropy import activation_entropy
from src.experiments.tpds import kernels as K
from src.experiments.tpds.common import (DEFAULT_TRACE_DIR, GDN_CHUNKS, MAMBA1_CHUNKS, MAMBA2_CHUNKS, RESULTS_DIR,
                                         JsonlWriter, Logger, check_disk, cuda_time, dedupe_by_key, env_metadata,
                                         gpu_state, host_name, peak_vram_gb, read_jsonl, summarize, write_json)

DISTS = ("uniform", "gaussian", "laplace", "sparse50", "sparse10", "sparse02", "constant", "real")
MAMBA1_DIM, MAMBA1_N = 2048, 16                       # mamba-370m
MAMBA2_H, MAMBA2_P, MAMBA2_G, MAMBA2_N = 32, 64, 1, 128  # mamba2-370m
GDN_H, GDN_DK, GDN_DV = 4, 256, 512                   # 370M-class Gated DeltaNet


# --------------------------------------------------------------------------- input generation
def sample(dist: str, shape, gen: torch.Generator, device="cuda") -> torch.Tensor:
    """Zero-mean, unit-variance samples of the named distribution (constant: all ones)."""
    if dist == "uniform":
        return (torch.rand(shape, device=device, generator=gen) * 2 - 1) * math.sqrt(3.0)
    if dist == "gaussian":
        return torch.randn(shape, device=device, generator=gen)
    if dist == "laplace":
        u = torch.rand(shape, device=device, generator=gen).clamp(1e-7, 1 - 1e-7) - 0.5
        return -torch.sign(u) * torch.log1p(-2 * u.abs()) / math.sqrt(2.0)
    if dist.startswith("sparse"):
        keep = int(dist[6:]) / 100.0
        mask = torch.rand(shape, device=device, generator=gen) < keep
        return torch.randn(shape, device=device, generator=gen) * mask / math.sqrt(keep)
    if dist == "constant":
        return torch.ones(shape, device=device)
    raise ValueError(dist)


def _fit_time(t: torch.Tensor, L: int, dim: int) -> torch.Tensor:
    """Crop or tile a captured tensor along its time dimension to length L."""
    n = t.shape[dim]
    if n >= L:
        return t.narrow(dim, 0, L).contiguous()
    reps = math.ceil(L / n)
    return torch.cat([t] * reps, dim=dim).narrow(dim, 0, L).contiguous()


def _longest_call(trace_dir: Path, kind: str):
    from src.experiments.tpds.capture_traces import load_manifest

    rows = load_manifest(trace_dir, kind)
    if not rows:
        return None
    best = max(rows, key=lambda r: r["L"])
    layers = best["layers"]
    return best["prompt_id"], layers[len(layers) // 2]


def mamba1_inputs(dist, mode, L, bsz, seed, trace_dir):
    base = K.make_mamba1_inputs(bsz, MAMBA1_DIM, MAMBA1_N, L, dtype=torch.float16, seed=seed)
    if dist == "real":
        from src.experiments.tpds.capture_traces import load_call, load_params

        found = _longest_call(trace_dir, "mamba1")
        if found is None:
            return None
        call = load_call(trace_dir, "mamba1", *found)
        prm = load_params(trace_dir, "mamba1", found[1])
        real = {"u": _fit_time(call["u"], L, 2), "delta": _fit_time(call["delta"], L, 2),
                "B": _fit_time(call["B"], L, 2), "C": _fit_time(call["C"], L, 2)}
        real = {k: v.expand(bsz, *v.shape[1:]).contiguous() for k, v in real.items()}
        if mode == "act":
            base["u"] = real["u"]
        else:
            base.update(real)
            base["A"] = prm["A"].contiguous()
            base["D"] = prm["D"].float()
        return base
    gen = torch.Generator(device="cuda").manual_seed(seed + 101)
    shp = (bsz, MAMBA1_DIM, L)
    base["u"] = sample(dist, shp, gen).half()
    if mode == "all":
        base["delta"] = (-3.5 + 0.5 * sample(dist, shp, gen)).half()
        base["B"] = sample(dist, (bsz, MAMBA1_N, L), gen).half()
        base["C"] = sample(dist, (bsz, MAMBA1_N, L), gen).half()
    return base


def mamba2_inputs(dist, mode, L, bsz, seed, trace_dir):
    base = K.make_ssd_inputs(bsz, L, MAMBA2_H, MAMBA2_P, MAMBA2_G, MAMBA2_N, dtype=torch.float16, seed=seed)
    if dist == "real":
        from src.experiments.tpds.capture_traces import load_call, load_params

        found = _longest_call(trace_dir, "mamba2")
        if found is None:
            return None
        call = load_call(trace_dir, "mamba2", *found)
        prm = load_params(trace_dir, "mamba2", found[1])
        real = {k: _fit_time(call[k], L, 1) for k in ("x", "dt", "B", "C")}
        real = {k: v.expand(bsz, *v.shape[1:]).contiguous() for k, v in real.items()}
        if mode == "act":
            base["x"] = real["x"]
        else:
            base.update(real)
            base.update({"A": prm["A"], "D": prm["D"].float(), "dt_bias": prm["dt_bias"].float()})
        return base
    gen = torch.Generator(device="cuda").manual_seed(seed + 202)
    base["x"] = sample(dist, (bsz, L, MAMBA2_H, MAMBA2_P), gen).half()
    if mode == "all":
        base["dt"] = (-3.5 + 0.5 * sample(dist, (bsz, L, MAMBA2_H), gen)).half()
        base["B"] = sample(dist, (bsz, L, MAMBA2_G, MAMBA2_N), gen).half()
        base["C"] = sample(dist, (bsz, L, MAMBA2_G, MAMBA2_N), gen).half()
    return base


def gdn_inputs(dist, mode, L, bsz, seed, trace_dir):
    base = K.make_gdn_inputs(bsz, L, GDN_H, GDN_DK, GDN_DV, seed=seed)
    if dist == "real":
        return None  # no Gated DeltaNet checkpoint is part of this study
    gen = torch.Generator(device="cuda").manual_seed(seed + 303)
    base["v"] = sample(dist, (bsz, L, GDN_H, GDN_DV), gen).bfloat16()
    if mode == "all":
        base["q"] = sample(dist, (bsz, L, GDN_H, GDN_DK), gen).bfloat16()
        k = sample(dist, (bsz, L, GDN_H, GDN_DK), gen)
        base["k"] = torch.nn.functional.normalize(k + 1e-3, p=2, dim=-1).bfloat16()
        base["g"] = torch.nn.functional.logsigmoid(sample(dist, (bsz, L, GDN_H), gen) + 3.0).float()
        base["beta"] = torch.sigmoid(sample(dist, (bsz, L, GDN_H), gen)).bfloat16()
    return base


def runner(family, inp, chunk):
    if family == "mamba1":
        out = torch.empty_like(inp["u"])
        return lambda: K.mamba1_scan(inp["u"], inp["delta"], inp["A"], inp["B"], inp["C"], inp["D"], inp["z"],
                                     None, True, chunk=chunk, out=out)
    if family == "mamba2":
        return lambda: K.mamba2_ssd(inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"], chunk, D=inp["D"],
                                    dt_bias=inp["dt_bias"])
    return lambda: K.gdn_chunk(inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], chunk_size=chunk)


ACT_KEY = {"mamba1": "u", "mamba2": "x", "gdn": "v"}
BUILDERS = {"mamba1": mamba1_inputs, "mamba2": mamba2_inputs, "gdn": gdn_inputs}
CHUNKS = {"mamba1": MAMBA1_CHUNKS, "mamba2": MAMBA2_CHUNKS, "gdn": GDN_CHUNKS}


# --------------------------------------------------------------------------- summary
def summarize_rows(rows):
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["family"], r["L"], r["batch"], r["mode"]), []).append(r)
    out = []
    for (fam, L, b, mode), rs in sorted(groups.items()):
        # mean latency per (chunk, input) averaged over rounds
        cell: dict[tuple, list] = {}
        for r in rs:
            cell.setdefault((r["chunk"], r["input"]), []).append(r["mean_ms"])
        chunks = sorted({c for c, _ in cell})
        inputs = sorted({i for _, i in cell})
        per_chunk = []
        for c in chunks:
            means = [statistics.fmean(cell[(c, i)]) for i in inputs if (c, i) in cell]
            if len(means) < 2:
                continue
            m = statistics.fmean(means)
            per_chunk.append({"chunk": c, "mean_ms": m, "cv_between_inputs": statistics.pstdev(means) / m,
                              "max_over_min_inputs": max(means) / min(means), "n_inputs": len(means)})
        chunk_means = [p["mean_ms"] for p in per_chunk]
        noise = [r["std_ms"] / r["mean_ms"] for r in rs if r["mean_ms"] > 0]
        out.append({
            "family": fam, "L": L, "batch": b, "mode": mode, "per_chunk": per_chunk,
            "median_cv_between_inputs": statistics.median([p["cv_between_inputs"] for p in per_chunk]) if per_chunk else None,
            "max_cv_between_inputs": max([p["cv_between_inputs"] for p in per_chunk]) if per_chunk else None,
            "between_chunk_max_over_min": (max(chunk_means) / min(chunk_means)) if chunk_means else None,
            "best_chunk": per_chunk[chunk_means.index(min(chunk_means))]["chunk"] if chunk_means else None,
            "median_within_repeat_cv": statistics.median(noise) if noise else None,
        })
    return out


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--families", default="mamba1,mamba2,gdn")
    p.add_argument("--lengths", default="512,2048,8192")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--inputs", default=",".join(DISTS))
    p.add_argument("--modes", default="act,all")
    p.add_argument("--chunks-mamba1", default=",".join(map(str, MAMBA1_CHUNKS)))
    p.add_argument("--chunks-mamba2", default=",".join(map(str, MAMBA2_CHUNKS)))
    p.add_argument("--chunks-gdn", default=",".join(map(str, GDN_CHUNKS)))
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--rounds", type=int, default=2, help="interleaved repetitions of the full input set")
    p.add_argument("--flush-l2", action="store_true")
    p.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    p.add_argument("--skip-real", action="store_true", help="do not wait for / use captured activations")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--tag", default=None, help="file tag (default: host name)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--extra-seeds", type=int, default=0,
                   help="additional random draws per synthetic distribution (null control for input effects)")
    p.add_argument("--record-gpu-state", action="store_true",
                   help="store SM clock, power, temperature and throttle reasons around every timing")
    return p.parse_args(argv)


def input_names(dists, extra_seeds):
    """Input names: every distribution, plus ``<dist>#s<k>`` redraws of the synthetic ones."""
    names = list(dists)
    for k in range(1, extra_seeds + 1):
        names += [f"{d}#s{k}" for d in dists if d not in ("real", "constant")]
    return names


def split_input_name(name, seed):
    """``gaussian#s2`` -> (``gaussian``, seed + 2000); plain names keep the base seed."""
    dist, _, k = name.partition("#s")
    return dist, seed + 1000 * int(k) if k else seed


def main(argv=None):
    args = parse_args(argv)
    tag = args.tag or host_name()
    log = Logger(f"e1_{tag}")
    check_disk(args.out_dir, need_gb=0.5, log=log)
    jsonl = args.out_dir / f"e1_{tag}.jsonl"
    meta = env_metadata()
    meta.update({"experiment": "E1", "args": vars(args)})
    w = JsonlWriter(jsonl, meta=meta)
    families = [f for f in args.families.split(",") if f]
    avail = {"mamba1": True, "mamba2": K.ssd_available(), "gdn": K.gdn_available()}
    chunk_arg = {"mamba1": args.chunks_mamba1, "mamba2": args.chunks_mamba2, "gdn": args.chunks_gdn}

    # Correctness of every kernel/chunk against the fp32 reference (small shape, once per file).
    for fam in families:
        if not avail[fam]:
            log(f"family {fam} unavailable; skipped")
            continue
        chunks = [int(c) for c in chunk_arg[fam].split(",")]
        todo = [c for c in chunks if f"validation|{fam}|{c}" not in w.done]
        if not todo:
            continue
        vfn = {"mamba1": K.validate_mamba1, "mamba2": K.validate_ssd, "gdn": K.validate_gdn}[fam]
        for v in vfn(todo):
            w.write({"kind": "validation", "key": f"validation|{fam}|{v['chunk']}", **v})
            log(f"validation {fam} chunk {v['chunk']}: max_abs {v['max_abs_err']:.3e} rel {v['max_rel_err']:.3e} "
                f"passed={v['passed']}")

    rng = random.Random(args.seed)
    dists = input_names([d for d in args.inputs.split(",") if d], args.extra_seeds)
    modes = [m for m in args.modes.split(",") if m]
    for fam in families:
        if not avail[fam]:
            continue
        chunks = [int(c) for c in chunk_arg[fam].split(",")]
        for L in [int(x) for x in args.lengths.split(",")]:
            inputs = {}
            for mode in modes:
                for d in dists:
                    if d == "real" and args.skip_real:
                        continue
                    if all(f"{fam}|L{L}|b{args.batch}|c{c}|{d}|{mode}|r{r}" in w.done
                           for c in chunks for r in range(args.rounds)):
                        continue
                    dist, seed = split_input_name(d, args.seed)
                    inp = BUILDERS[fam](dist, mode, L, args.batch, seed, args.trace_dir)
                    if inp is None:
                        log(f"{fam} input {d}/{mode}: not available (no captured traces); skipped")
                        continue
                    inputs[(d, mode)] = inp
            if not inputs:
                continue
            ent = {k: float(activation_entropy(v[ACT_KEY[fam]].float(), num_bins=256)) for k, v in inputs.items()}
            for r in range(args.rounds):
                order_c = list(chunks)
                rng.shuffle(order_c)
                for c in order_c:
                    ref_out = None
                    keys = list(inputs)
                    rng.shuffle(keys)
                    for (d, mode) in keys:
                        key = f"{fam}|L{L}|b{args.batch}|c{c}|{d}|{mode}|r{r}"
                        if key in w.done:
                            continue
                        fn = runner(fam, inputs[(d, mode)], c)
                        y = fn()
                        finite = bool(torch.isfinite(y.float()).all())
                        state_before = gpu_state() if args.record_gpu_state else None
                        samples = cuda_time(fn, args.warmup, args.repeats, args.flush_l2)
                        state_after = gpu_state() if args.record_gpu_state else None
                        st = summarize(samples)
                        row = {"kind": "timing", "key": key, "family": fam, "L": L, "batch": args.batch, "chunk": c,
                               "input": d, "mode": mode, "round": r, "entropy_bits_k256": ent[(d, mode)],
                               "entropy_norm": ent[(d, mode)] / 8.0, "output_finite": finite,
                               "flush_l2": args.flush_l2, **st, "peak_vram_gb": peak_vram_gb()}
                        if args.record_gpu_state:
                            row["gpu_state_before"], row["gpu_state_after"] = state_before, state_after
                        w.write(row)
                        log(f"{fam} L={L} c={c} {d:9s} {mode} r{r}: {st['mean_ms']:.4f} +- {st['std_ms']:.4f} ms "
                            f"(H={ent[(d, mode)]:.2f} bits)")
            del inputs
            torch.cuda.empty_cache()
            _write_summary(args, tag, w)
    _write_summary(args, tag, w)
    w.close()
    log(f"E1 done; peak VRAM {peak_vram_gb():.2f} GB")


def _write_summary(args, tag, w):
    rows = dedupe_by_key(read_jsonl(args.out_dir / f"e1_{tag}.jsonl"))
    timing = [r for r in rows if r.get("kind") == "timing"]
    val = [r for r in rows if r.get("kind") == "validation"]
    metas = read_jsonl(args.out_dir / f"e1_{tag}.jsonl", kind="meta")
    write_json(args.out_dir / f"e1_{tag}.json", {
        "experiment": "E1", "meta": metas[-1] if metas else {}, "validation": val,
        "all_validation_passed": all(v["passed"] for v in val) if val else None,
        "summary": summarize_rows(timing), "rows": timing})


if __name__ == "__main__":
    main()
