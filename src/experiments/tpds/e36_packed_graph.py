"""E36 - packed variable-length prefill steps under CUDA graphs at padded capture sizes (review taco-r2 M3d).

E27 timed the packed prefill steps sampled from the Azure 2023 traces (several prompt chunks concatenated at
batch 1, fresh or with a carried state) in device time, one exact composition at a time. Engines that capture
their prefill step as a CUDA graph capture it at a few padded token counts (capture sizes) and replay the same
graph for every step that fits, with the step's composition passed as data (sequence indices or offsets).
This experiment times the production kernels that way, so that the regret of a chunk table keyed on the capture
size alone, and of richer keys, can be measured in the regime that matters.

Families:
  ssd370m / ssd2p7b  ``mamba_chunk_scan_combined`` with ``seq_idx`` (Mamba-2 370M / 2.7B shapes, fp16). For
                     every capture size P that the steps need, one graph per (chunk, variant) is captured on
                     static buffers of P tokens; for each step the sequence-index buffer is rewritten with the
                     step's composition (the padding tokens form one extra sequence) and the graph is replayed.
  gdn1p4b            ``chunk_gated_delta_rule`` with ``cu_seqlens`` (Gated DeltaNet 1.4B shapes, bf16). The
                     library derives per-chunk indices from the offsets with host-side code, so a graph cannot be
                     reused across compositions: one graph per (step, chunk, variant) is captured at the padded
                     size P with that step's offsets (the padding forms one extra sequence).
Capture sizes: multiples of 16 up to 512 and multiples of 512 up to 8192 (``--capture-sizes`` to change).
Variants: ``fresh`` and ``carried`` (a carried state for the first packed sequence; for Gated DeltaNet the state
tensor holds one entry per sequence, zero except the first).
Values: one Gaussian draw of the model's shapes (the latency of these kernels does not depend on the values at
fixed placement, E33), with the layer parameters of a captured call where available.

Timing: ``--rounds`` rounds; per round the steps and, per step, the chunks run in a seeded random order; every
(step, chunk, variant) is ``--warmup`` replays and ``--repeats`` replays timed with CUDA events after a
GPU-side pre-spin; the cell value is the 10 % trimmed mean. Exclusive-GPU guard as in the other experiments.

Resumable: one fsynced JSONL row per (family, step, variant, round) with the latency of every chunk.
``--analyze`` writes ``e36_packed_graph_<tag>.json``: per family and variant the default's per-step and
latency-weighted regret, the best single chunk, held-out tables keyed on the capture size and on (capture size,
number of sequences), the noise floor (an oracle chosen on some rounds evaluated on the others), and the share
of steps whose per-round argmin is the same in every round.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.experiments.tpds.common import (GDN_CHUNKS, LOG_DIR, MAMBA2_CHUNKS, RESULTS_DIR, JsonlWriter, Logger, check_disk,
                                         cuda_time, env_metadata, foreign_gpu_pids, host_name, read_jsonl, utc_now,
                                         wait_for_exclusive_gpu, write_json)

FAMILIES = {"ssd370m": {"kind": "mamba2", "H": 32, "P": 64, "N": 128, "default": 256, "chunks": MAMBA2_CHUNKS},
            "ssd2p7b": {"kind": "mamba2", "H": 80, "P": 64, "N": 128, "default": 256, "chunks": MAMBA2_CHUNKS},
            "gdn1p4b": {"kind": "gdn", "default": 64, "chunks": GDN_CHUNKS}}
VARIANTS = ("fresh", "carried")
TIE = 0.02


def default_capture_sizes() -> list[int]:
    return [16 * k for k in range(1, 33)] + [512 * k for k in range(2, 17)]


def capture_size(T: int, sizes: list[int]) -> int:
    for s in sizes:
        if s >= T:
            return s
    raise ValueError(f"no capture size >= {T}")


def _g(x, d=6):
    return None if x is None else float(f"{float(x):.{d}g}")


def tmean(ts: list[float]) -> float:
    s = sorted(ts)
    c = len(s) // 10
    return statistics.fmean(s[c: len(s) - c] or s)


def load_layer_params(trace_dir: str | None, kind: str):
    if not trace_dir:
        return None
    try:
        from src.experiments.tpds.capture_traces import load_manifest, load_params

        rows = load_manifest(Path(trace_dir), kind)
        for r in rows:
            for layer in r.get("layers", []):
                try:
                    return load_params(Path(trace_dir), kind, layer), layer
                except FileNotFoundError:
                    continue
    except Exception:
        return None
    return None


def gdn_dims(trace_dir: str | None) -> dict[str, int]:
    from src.experiments.tpds.e35_mechanism import gdn_dims as _dims

    return _dims(argparse.Namespace(gdn_trace_dir=trace_dir))


# --------------------------------------------------------------------------- run
def run(args, log) -> None:
    import torch

    from src.experiments.tpds import kernels as K

    steps = json.loads(Path(args.steps).read_text())["steps"]
    if args.max_steps:
        steps = steps[: args.max_steps]
    sizes = [int(x) for x in args.capture_sizes.split(",")] if args.capture_sizes else default_capture_sizes()
    for i, st in enumerate(steps):
        st["idx"] = i
        st["P"] = capture_size(int(st["tokens"]), sizes)
        st["sid"] = f"{st['trace']}|T{st['budget']}|s{st['step']}"
    check_disk(args.out_dir, need_gb=0.2, need_inodes=100, log=log)
    meta = env_metadata()
    meta.update({"experiment": "E36", "steps_file": str(args.steps), "capture_sizes": sizes, "rounds": args.rounds,
                 "warmup": args.warmup, "repeats": args.repeats, "pre_spin_ms": args.pre_spin_ms,
                 "guard_gpu": args.guard_gpu, "args": {k: str(v) for k, v in vars(args).items()}})
    w = JsonlWriter(args.out_dir / f"e36_packed_graph_{args.tag}.jsonl", meta=meta)
    traces = {"ssd370m": args.ssd370m_traces, "ssd2p7b": args.ssd2p7b_traces, "gdn1p4b": args.gdn_traces}
    for fam in [f for f in args.families.split(",") if f]:
        spec = FAMILIES[fam]
        prm = load_layer_params(traces.get(fam), spec["kind"])
        log(f"{fam}: {len(steps)} steps, capture sizes used {sorted({s['P'] for s in steps})}; "
            f"parameters from {'layer ' + str(prm[1]) if prm else 'synthetic'}")
        if spec["kind"] == "mamba2":
            run_ssd(args, log, w, fam, spec, steps, prm)
        else:
            run_gdn(args, log, w, fam, spec, steps, prm)
    w.close()
    try:
        analyze(args)
    except Exception as exc:
        log(f"analysis failed: {exc!r}")


def _guarded_time(args, fn) -> tuple[float, bool]:
    if args.exclusive_guard:
        if wait_for_exclusive_gpu(args.guard_until, None, index=args.guard_gpu) is None:
            raise SystemExit("guard deadline passed while another process held the GPU")
    t = tmean(cuda_time(fn, warmup=args.warmup, repeats=args.repeats, pre_spin_ms=args.pre_spin_ms))
    foreign = bool(args.exclusive_guard and foreign_gpu_pids(args.guard_gpu))
    if foreign:
        wait_for_exclusive_gpu(args.guard_until, None, index=args.guard_gpu)
        t = tmean(cuda_time(fn, warmup=args.warmup, repeats=args.repeats, pre_spin_ms=args.pre_spin_ms))
    return t, foreign


def _capture(fn):
    import torch

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def _seq_idx(lens: list[int], P: int):
    import torch

    idx = [i for i, n in enumerate(lens) for _ in range(n)]
    idx += [len(lens)] * (P - len(idx))
    return torch.tensor(idx, dtype=torch.int32, device="cuda")[None]


def run_ssd(args, log, w, fam, spec, steps, prm) -> None:
    import torch

    from src.experiments.tpds import kernels as K

    H, Pd, N = spec["H"], spec["P"], spec["N"]
    byP = defaultdict(list)
    for st in steps:
        byP[st["P"]].append(st)
    for P, sts in sorted(byP.items()):
        keys_left = [(st, v, r) for st in sts for v in VARIANTS for r in range(args.rounds)
                     if f"e36|{fam}|{st['sid']}|{v}|r{r}" not in w.done]
        if not keys_left:
            continue
        inp = K.make_ssd_inputs(1, P, H, Pd, 1, N, dtype=torch.float16, seed=0)
        if prm is not None:
            p = prm[0]
            inp.update({"A": p["A"].float(), "D": p["D"].float(), "dt_bias": p["dt_bias"].float()})
        seq_buf = torch.zeros(1, P, dtype=torch.int32, device="cuda")
        state = 0.1 * torch.randn(1, H, Pd, N, device="cuda")
        chunks = [c for c in spec["chunks"] if K.ssd_state_elems(1, P, H, Pd, N, c) <= K.SSD_INDEX_LIMIT]
        graphs = {}
        seq_buf.copy_(_seq_idx(sts[0]["lens"], P))
        for v in VARIANTS:
            for c in chunks:
                st_arg = state if v == "carried" else None
                fn = (lambda c=c, st_arg=st_arg: K.mamba2_ssd_varlen(inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"],
                                                                      c, seq_buf, D=inp["D"], dt_bias=inp["dt_bias"],
                                                                      initial_states=st_arg))
                graphs[(v, c)] = _capture(fn)
        for r in range(args.rounds):
            order = list(sts)
            random.Random(f"e36|{fam}|P{P}|r{r}").shuffle(order)
            for st in order:
                seq_buf.copy_(_seq_idx(st["lens"], P))
                for v in VARIANTS:
                    key = f"e36|{fam}|{st['sid']}|{v}|r{r}"
                    if key in w.done:
                        continue
                    corder = list(chunks)
                    random.Random(f"{key}|order").shuffle(corder)
                    lat, foreign = {}, False
                    for c in corder:
                        t, f = _guarded_time(args, graphs[(v, c)].replay)
                        lat[str(c)] = _g(t, 6)
                        foreign |= f
                    w.write({"kind": "timing", "key": key, "family": fam, "sid": st["sid"], "trace": st["trace"],
                             "budget": st["budget"], "step": st["step"], "variant": v, "round": r,
                             "tokens": st["tokens"], "P": P, "nseq": st["nseq"], "lens": st["lens"],
                             "lat_ms": lat, "retimed": foreign, "mode": "padded_graph_reused", "utc": utc_now()})
            log(f"{fam} P={P} round {r}: {len(sts)} steps")
        del graphs, inp, seq_buf, state
        torch.cuda.empty_cache()


def run_gdn(args, log, w, fam, spec, steps, prm) -> None:
    import numpy as np
    import torch

    from src.experiments.tpds import kernels as K

    d = gdn_dims(args.gdn_traces)
    g = torch.Generator(device="cuda").manual_seed(0)
    Pmax = max(st["P"] for st in steps)
    base = {"q": torch.randn(1, Pmax, d["Hq"], d["K"], device="cuda", generator=g).bfloat16(),
            "k": torch.randn(1, Pmax, d["Hq"], d["K"], device="cuda", generator=g).bfloat16(),
            "v": torch.randn(1, Pmax, d["Hv"], d["V"], device="cuda", generator=g).bfloat16(),
            "g": torch.randn(1, Pmax, d["Hv"], device="cuda", generator=g).bfloat16(),
            "beta": torch.randn(1, Pmax, d["Hv"], device="cuda", generator=g).bfloat16()}
    if prm is not None:
        A_log, dt_bias = prm[0]["A_log"], prm[0]["dt_bias"]
        neg = bool(prm[0]["allow_neg_eigval"].item()) if "allow_neg_eigval" in prm[0] else False
    else:
        A_log = torch.log(torch.rand(d["Hv"], device="cuda", generator=g) * 15 + 1)
        dt_bias = torch.randn(d["Hv"], device="cuda", generator=g) * 0.1
        neg = False
    for r in range(args.rounds):
        order = list(steps)
        random.Random(f"e36|{fam}|r{r}").shuffle(order)
        for st in order:
            P = st["P"]
            todo = [v for v in VARIANTS if f"e36|{fam}|{st['sid']}|{v}|r{r}" not in w.done]
            if not todo:
                continue
            lens = list(st["lens"]) + ([P - st["tokens"]] if P > st["tokens"] else [])
            cu = torch.tensor([0, *np.cumsum(lens).tolist()], dtype=torch.long, device="cuda")
            inp = {k: v[:, :P].contiguous() for k, v in base.items()}
            for v in todo:
                key = f"e36|{fam}|{st['sid']}|{v}|r{r}"
                corder = list(spec["chunks"])
                random.Random(f"{key}|order").shuffle(corder)
                lat, err, foreign = {}, {}, False
                state = None
                if v == "carried":
                    state = torch.zeros(len(lens), d["Hv"], d["V"], d["K"], device="cuda", dtype=torch.float32)
                    state[0].normal_(0, 0.1)
                for c in corder:
                    fn = (lambda c=c: K.gdn_chunk_prod_varlen(inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"],
                                                              A_log, dt_bias, cu, c, neg, initial_state=state))
                    try:
                        gr = _capture(fn)
                    except Exception as exc:  # noqa: BLE001 - recorded; the cell is timed eagerly instead
                        err[str(c)] = f"capture: {type(exc).__name__}: {exc}"[:200]
                        torch.cuda.synchronize()
                        gr = None
                    t, f = _guarded_time(args, gr.replay if gr is not None else fn)
                    lat[str(c)] = _g(t, 6)
                    foreign |= f
                    del gr
                w.write({"kind": "timing", "key": key, "family": fam, "sid": st["sid"], "trace": st["trace"],
                         "budget": st["budget"], "step": st["step"], "variant": v, "round": r,
                         "tokens": st["tokens"], "P": P, "nseq": st["nseq"], "lens": st["lens"], "lat_ms": lat,
                         "errors": err, "retimed": foreign, "mode": "padded_graph_per_step", "utc": utc_now()})
            del inp
        log(f"{fam} round {r}: {len(steps)} steps")
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- analysis
def _nseq_class(n: int) -> str:
    return "1" if n == 1 else "2-3" if n <= 3 else "4-7" if n <= 7 else "8+"


def _items(rows, rounds=None):
    by = defaultdict(list)
    for r in rows:
        if rounds is None or r["round"] in rounds:
            by[r["sid"]].append(r)
    items = []
    for sid, rs in by.items():
        chunks = sorted({int(c) for x in rs for c in x["lat_ms"]})
        chunks = [c for c in chunks if all(str(c) in x["lat_ms"] for x in rs)]
        lat = {c: statistics.median(x["lat_ms"][str(c)] for x in rs) for c in chunks}
        o = min(chunks, key=lambda c: (lat[c], c))
        items.append({"id": sid, "P": rs[0]["P"], "nseq": rs[0]["nseq"], "tokens": rs[0]["tokens"], "lat": lat,
                      "oracle": o, "rel": {c: lat[c] / lat[o] for c in chunks}, "chunks": chunks,
                      "argmins": [min(x["lat_ms"], key=lambda c: (x["lat_ms"][c], int(c))) for x in rs]})
    return items


def _summ(pairs):
    reg = [it["rel"][c] - 1 for it, c in pairs]
    lat = sum(it["lat"][c] for it, c in pairs)
    opt = sum(it["lat"][it["oracle"]] for it, _ in pairs)
    return {"n": len(pairs), "mean_regret_pct": _g(100 * statistics.fmean(reg), 4),
            "latency_weighted_pct": _g(100 * (lat / opt - 1), 4),
            "share_within_2pct": _g(sum(x <= TIE for x in reg) / len(reg), 4)}


def _table(items, keyfn, chunks, folds=5, seed=0):
    ids = sorted({it["id"] for it in items})
    random.Random(seed).shuffle(ids)
    fold = {s: i % folds for i, s in enumerate(ids)}
    out = []
    for f in range(folds):
        agg, glob = defaultdict(lambda: defaultdict(float)), defaultdict(float)
        for it in items:
            if fold[it["id"]] != f:
                for c in chunks:
                    agg[keyfn(it)][c] += it["rel"][c]
                    glob[c] += it["rel"][c]
        bg = min(chunks, key=lambda c: glob[c])
        for it in items:
            if fold[it["id"]] == f:
                k = keyfn(it)
                out.append((it, min(chunks, key=lambda c: agg[k][c]) if k in agg else bg))
    return out


def analyze(args) -> dict[str, Any]:
    tags = [t for t in str(getattr(args, "input_tags", None) or args.tag).split(",") if t]
    rows = []
    for t in tags:
        rows += [r for r in read_jsonl(args.out_dir / f"e36_packed_graph_{t}.jsonl") if r.get("kind") == "timing"]
    res: dict[str, Any] = {"experiment": "E36", "tags": tags, "created_utc": utc_now(), "families": {}}
    for fam in sorted({r["family"] for r in rows}):
        default = FAMILIES[fam]["default"]
        out = {}
        for v in VARIANTS:
            rs = [r for r in rows if r["family"] == fam and r["variant"] == v]
            if not rs:
                continue
            rounds = sorted({r["round"] for r in rs})
            items = _items(rs)
            common = [c for c in FAMILIES[fam]["chunks"] if all(c in it["chunks"] for it in items)]
            single = min(common, key=lambda c: sum(it["rel"][c] for it in items))
            o = {"n_steps": len(items), "rounds": rounds,
                 "oracle_hist": {str(c): sum(it["oracle"] == c for it in items) for c in common},
                 "default": {"chunk": default, **_summ([(it, default) for it in items])},
                 "best_single": {"chunk": single, **_summ([(it, single) for it in items])},
                 "table_capture_size": _summ(_table(items, lambda it: it["P"], common)),
                 "table_capture_size_nseq": _summ(_table(items, lambda it: (it["P"], _nseq_class(it["nseq"])), common)),
                 "argmin_same_all_rounds_share": _g(statistics.fmean(len(set(it["argmins"])) == 1 for it in items), 4)}
            if len(rounds) >= 4:
                half_a, half_b = rounds[::2], rounds[1::2]
                ia = {it["id"]: it for it in _items(rs, set(half_a))}
                ib = {it["id"]: it for it in _items(rs, set(half_b))}
                pairs = [(ib[s], ia[s]["oracle"]) for s in ia if s in ib and ia[s]["oracle"] in ib[s]["chunks"]]
                o["noise_floor_oracle_split_half"] = _summ(pairs)
            out[v] = o
        res["families"][fam] = out
        for v, o in out.items():
            print(f"{fam}/{v}: {o['n_steps']} steps; default {o['default']['mean_regret_pct']}% "
                  f"(LW {o['default']['latency_weighted_pct']}%); capture-size table "
                  f"{o['table_capture_size']['mean_regret_pct']}%; noise floor "
                  f"{o.get('noise_floor_oracle_split_half', {}).get('mean_regret_pct')}%; argmin stable "
                  f"{o['argmin_same_all_rounds_share']}")
    out_tag = tags[0] if len(tags) == 1 else "_".join(tags)
    write_json(args.out_dir / f"e36_packed_graph_{out_tag}.json", res)
    return res


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=Path, default=RESULTS_DIR / "packed_steps_taco.json")
    p.add_argument("--families", default="ssd2p7b,ssd370m,gdn1p4b")
    p.add_argument("--ssd370m-traces", default=None)
    p.add_argument("--ssd2p7b-traces", default=None)
    p.add_argument("--gdn-traces", default=None)
    p.add_argument("--capture-sizes", default=None)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--pre-spin-ms", type=float, default=50.0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--exclusive-guard", action="store_true")
    p.add_argument("--guard-until", type=float, default=None)
    p.add_argument("--guard-gpu", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--log-dir", type=Path, default=LOG_DIR)
    p.add_argument("--tag", default=None)
    p.add_argument("--input-tags", default=None)
    p.add_argument("--analyze", action="store_true")
    a = p.parse_args(argv)
    a.tag = a.tag or f"{host_name()}_e36"
    if a.guard_gpu is None:
        import os

        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        a.guard_gpu = int(vis) if vis.isdigit() else 0
    return a


def main(argv=None):
    args = parse_args(argv)
    if args.analyze:
        analyze(args)
        return
    run(args, Logger(f"e36_packed_graph_{args.tag}", args.log_dir))


if __name__ == "__main__":
    main()
