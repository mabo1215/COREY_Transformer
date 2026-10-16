"""E33 - value invariance of the production SSD and Gated DeltaNet scans above their launch floor (taco-r2 M5).

The value-invariance base of the paper times the production scans launch-inclusive, where short calls
are dominated by the launch cost, and loads captured activations in a separate process from the
synthetic inputs, so that values and buffer placement change together. This experiment times every
cell as a CUDA-graph replay (device time of the whole multi-kernel scan, no host launch cost) at shapes
well above the launch floor, with all inputs in one process, and separates values from placement:

  A  same_buffers   one set of input buffers per (family, L, B); one graph per chunk is captured on these
                    buffers once; for every input the values are copied into the buffers in place and the
                    graph is replayed. Only the values change.
  C  placements     one Gaussian draw copied into ``--placements`` buffer sets (half freshly allocated, half
                    at 256-byte multiples inside larger allocations, so Triton's 16-byte alignment
                    specialization is unchanged); one graph per buffer set and chunk. Only the placement
                    changes.

Families (production kernels, shapes of the checkpoints):
  ssd370m  Mamba-2 370M SSD (``mamba_chunk_scan_combined``): 32 heads x 64, 1 group, state 128, fp16
  ssd2p7b  Mamba-2 2.7B SSD: 80 heads x 64, 1 group, state 128, fp16
  gdn1p4b  Gated DeltaNet 1.4B on the production path (in-kernel L2 norm, gate and beta), bf16; head
           counts and widths are read from the captured call.
  m1triton the Triton Mamba-1 scan of the study (370M width 2048, state 16, fp16, default launch table per
           chunk); captured Mamba-1 activations (u, delta, B, C, z) and the layer's A and D for ``real``.

Inputs: the E1 distributions (uniform, Gaussian, Laplace, sparse 50/10/2 %, constant) in ``--draws``
independent draws each (draw-to-draw spread = null baseline), applied to every data input (``all`` mode,
each mapped to its valid range as in E1), and ``real``: the captured activations of the longest stored
call of that model (all data inputs and the layer's parameters), tiled along time to L. Synthetic cells
use the real call's parameters (A, D, dt_bias / A_log, dt_bias) as well, so that only the values of the
data inputs differ.

Protocol: ``--rounds`` rounds; per round the chunk order and, inside a chunk, the input order (design A)
or placement order (design C) are shuffled; every cell is ``--warmup`` replays and ``--repeats`` replays
timed with CUDA events after a GPU-side pre-spin. SM clock, power and throttle reasons before and after
every cell; the exclusive-GPU guard (``--guard-gpu``: physical index) pauses while another process holds
the GPU and a cell is re-timed once if one appeared during it.

Resumable: one fsynced JSONL row per (family, L, B, design, round, chunk, input/placement).
``--ssd-dtype bfloat16`` runs the SSD families in bf16 instead of fp16 (E38: does the best chunk depend on the
data type that the cache key records?). With ``--inputs gaussian#0 --designs A`` the script is a plain CUDA-graph timing grid of the production
scans over shapes and chunks (used for the graph-timed calibration keys of the work-term model, E37).

``--analyze`` (CPU) writes ``e33_value_graph_<tag>.json``: per (family, L, B, chunk) the round-median of the
per-cell 10 % trimmed mean latency (CUDA events resolve about 0.5 us, which quantizes per-replay medians) per input, the spread across distributions and between draws of one distribution, a Friedman
test across inputs with rounds as blocks, TOST equivalence of every input against the first Gaussian
draw at +-1 % and +-2 %, whether the best chunk changes with the input, and the spread across placements.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.experiments.tpds.common import (GDN_CHUNKS, LOG_DIR, MAMBA1_CHUNKS, MAMBA2_CHUNKS, RESULTS_DIR, JsonlWriter, Logger,
                                         check_disk, cuda_time, dedupe_by_key, env_metadata, foreign_gpu_pids,
                                         gpu_state, host_name, read_jsonl, utc_now, wait_for_exclusive_gpu,
                                         write_json)

SYNTH = ("uniform", "gaussian", "laplace", "sparse50", "sparse10", "sparse02", "constant")
FAMILIES = {
    "ssd370m": {"kind": "mamba2", "H": 32, "P": 64, "G": 1, "N": 128, "chunks": MAMBA2_CHUNKS},
    "ssd2p7b": {"kind": "mamba2", "H": 80, "P": 64, "G": 1, "N": 128, "chunks": MAMBA2_CHUNKS},
    "gdn1p4b": {"kind": "gdn", "chunks": GDN_CHUNKS},
    "m1triton": {"kind": "mamba1", "d": 2048, "n": 16, "chunks": MAMBA1_CHUNKS},
}


def _g(x: Any, digits: int = 6) -> Any:
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return x
    return float(f"{x:.{digits}g}")


def _parse_shapes(raw: str) -> list[tuple[int, int]]:
    out = []
    for tok in str(raw).split(","):
        if tok.strip():
            L, B = tok.lower().split("x")
            out.append((int(L), int(B)))
    return out


# --------------------------------------------------------------------------- inputs
def _fit_time(t, L: int, dim: int):
    import torch

    n = t.shape[dim]
    if n >= L:
        return t.narrow(dim, 0, L).contiguous()
    reps = math.ceil(L / n)
    return torch.cat([t] * reps, dim=dim).narrow(dim, 0, L).contiguous()


def load_real(trace_dir: Path, kind: str, log) -> tuple[dict[str, Any], dict[str, Any], str] | None:
    """Captured inputs and parameters of the longest stored call whose file is present."""
    from src.experiments.tpds.capture_traces import load_call, load_manifest, load_params

    rows = sorted(load_manifest(trace_dir, kind), key=lambda r: -r["L"])
    for r in rows:
        layers = r["layers"]
        for layer in [layers[len(layers) // 2]] + [x for x in layers if x != layers[len(layers) // 2]]:
            try:
                call = load_call(trace_dir, kind, r["prompt_id"], layer)
                prm = load_params(trace_dir, kind, layer)
            except FileNotFoundError:
                continue
            return call, prm, f"{r['prompt_id']}:layer{layer}:L{r['L']}"
    log(f"no stored {kind} call found under {trace_dir}")
    return None


def sample(dist: str, shape, gen, device="cuda"):
    from src.experiments.tpds.e1_value_invariance import sample as e1_sample

    return e1_sample(dist, shape, gen, device)


class Family:
    """Buffers and kernel call of one (family, L, B)."""

    def __init__(self, name: str, L: int, B: int, real: tuple | None, log, ssd_dtype: str = "float16"):
        import torch

        self.name, self.L, self.B = name, L, B
        spec = FAMILIES[name]
        self.kind = spec["kind"]
        self.chunks = list(spec["chunks"])
        self.real_src = None
        if self.kind == "mamba2":
            H, P, G, N = spec["H"], spec["P"], spec["G"], spec["N"]
            self.shapes = {"x": (B, L, H, P), "dt": (B, L, H), "B": (B, L, G, N), "C": (B, L, G, N)}
            dt = getattr(torch, ssd_dtype)
            self.dtypes = {"x": dt, "dt": dt, "B": dt, "C": dt}
            self.params = {"A": -torch.exp(torch.rand(H, device="cuda") * math.log(16.0)),
                           "D": torch.randn(H, device="cuda"), "dt_bias": torch.randn(H, device="cuda") * 0.1}
            self.real = None
            if real is not None:
                call, prm, src = real
                if tuple(call["x"].shape[2:]) == (H, P) and tuple(call["B"].shape[2:]) == (G, N):
                    self.real = {k: _fit_time(call[k], L, 1) for k in ("x", "dt", "B", "C")}
                    self.real = {k: v.expand(B, *v.shape[1:]).contiguous().to(self.dtypes[k]) for k, v in self.real.items()}
                    self.params = {"A": prm["A"].float(), "D": prm["D"].float(), "dt_bias": prm["dt_bias"].float()}
                    self.real_src = src
                else:
                    log(f"{name}: captured call shape {tuple(call['x'].shape)} does not match; no real input")
        elif self.kind == "mamba1":
            d, n = spec["d"], spec["n"]
            self.shapes = {"u": (B, d, L), "delta": (B, d, L), "B": (B, n, L), "C": (B, n, L), "z": (B, d, L)}
            self.dtypes = {k: torch.float16 for k in self.shapes}
            self.params = {"A": -torch.exp(torch.rand(d, n, device="cuda") * math.log(16.0)),
                           "D": torch.randn(d, device="cuda")}
            self.real = None
            if real is not None:
                call, prm, src = real
                if tuple(call["u"].shape[1:2]) == (d,) and "z" in call:
                    self.real = {k: _fit_time(call[k], L, 2) for k in self.shapes}
                    self.real = {k: v.expand(B, *v.shape[1:]).contiguous().to(self.dtypes[k])
                                 for k, v in self.real.items()}
                    self.params = {"A": prm["A"].float().contiguous(), "D": prm["D"].float()}
                    self.real_src = src
                else:
                    log(f"{name}: captured call shape {tuple(call['u'].shape)} does not match; no real input")
        else:
            if real is None:
                raise SystemExit("gdn1p4b needs a captured call (head counts and widths are read from it)")
            call, prm, src = real
            _, _, Hq, K = call["q"].shape
            Hv, V = call["v"].shape[2], call["v"].shape[3]
            self.shapes = {"q": (B, L, Hq, K), "k": (B, L, Hq, K), "v": (B, L, Hv, V), "g": (B, L, Hv),
                           "beta": (B, L, Hv)}
            self.dtypes = {k: torch.bfloat16 for k in self.shapes}
            self.real = {k: _fit_time(call[k], L, 1) for k in self.shapes}
            self.real = {k: v.expand(B, *v.shape[1:]).contiguous().to(self.dtypes[k]) for k, v in self.real.items()}
            self.params = {"A_log": prm["A_log"], "dt_bias": prm["dt_bias"]}
            self.real_src = src
        self.inputs_desc = {k: list(v) for k, v in self.shapes.items()}

    # values ---------------------------------------------------------------
    def values(self, inp: str, seed: int) -> dict[str, Any] | None:
        """Tensors (on the GPU, final dtypes) of one input: ``real`` or ``<dist>#<draw>``."""
        import torch

        if inp == "real":
            return self.real
        dist, draw = inp.split("#")
        gen = torch.Generator(device="cuda").manual_seed(seed * 1000 + 17 * int(draw) + 7 * SYNTH.index(dist) + 1)
        out = {}
        if self.kind == "mamba1":
            out["u"] = sample(dist, self.shapes["u"], gen)
            out["delta"] = -3.5 + 0.5 * sample(dist, self.shapes["delta"], gen)
            out["B"] = sample(dist, self.shapes["B"], gen)
            out["C"] = sample(dist, self.shapes["C"], gen)
            out["z"] = sample(dist, self.shapes["z"], gen)
        elif self.kind == "mamba2":
            out["x"] = sample(dist, self.shapes["x"], gen)
            out["dt"] = -3.5 + 0.5 * sample(dist, self.shapes["dt"], gen)
            out["B"] = sample(dist, self.shapes["B"], gen)
            out["C"] = sample(dist, self.shapes["C"], gen)
        else:
            # raw projections as the production layer passes them (L2 norm, gate and sigmoid inside the kernel)
            out["q"] = sample(dist, self.shapes["q"], gen)
            out["k"] = sample(dist, self.shapes["k"], gen) + 1e-3
            out["v"] = sample(dist, self.shapes["v"], gen)
            out["g"] = sample(dist, self.shapes["g"], gen)
            out["beta"] = sample(dist, self.shapes["beta"], gen)
        return {k: v.to(self.dtypes[k]).contiguous() for k, v in out.items()}

    # buffers --------------------------------------------------------------
    def alloc(self, offset_bytes: int = 0) -> tuple[dict[str, Any], list[Any]]:
        """One buffer set; with an offset, every tensor is a view at that byte offset inside a larger allocation."""
        import torch

        bufs, owners = {}, []
        for k, shp in self.shapes.items():
            dt = self.dtypes[k]
            n = math.prod(shp)
            if offset_bytes:
                es = torch.tensor([], dtype=dt).element_size()
                off = offset_bytes // es
                base = torch.empty(n + off, dtype=dt, device="cuda")
                owners.append(base)
                bufs[k] = base[off: off + n].view(shp)
            else:
                bufs[k] = torch.empty(shp, dtype=dt, device="cuda")
        return bufs, owners

    def call(self, bufs: dict[str, Any], chunk: int):
        import torch

        from src.experiments.tpds import kernels as K

        if self.kind == "mamba1":
            p = self.params
            out = torch.empty_like(bufs["u"])
            return lambda: K.mamba1_scan(bufs["u"], bufs["delta"], p["A"], bufs["B"], bufs["C"], p["D"], bufs["z"],
                                         None, True, chunk=chunk, out=out)
        if self.kind == "mamba2":
            p = self.params
            return lambda: K.mamba2_ssd(bufs["x"], bufs["dt"], p["A"], bufs["B"], bufs["C"], chunk, D=p["D"],
                                        dt_bias=p["dt_bias"])
        p = self.params
        return lambda: K.gdn_chunk_prod(bufs["q"], bufs["k"], bufs["v"], bufs["g"], bufs["beta"], p["A_log"],
                                        p["dt_bias"], chunk_size=chunk)

    def feasible(self, chunk: int) -> bool:
        from src.experiments.tpds import kernels as K

        if self.kind != "mamba2":
            return True
        H, P = self.shapes["x"][2], self.shapes["x"][3]
        N = self.shapes["B"][3]
        return K.ssd_state_elems(self.B, self.L, H, P, N, chunk) <= K.SSD_INDEX_LIMIT


def capture(fn):
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


@contextlib.contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise SystemExit(f"another E33 process holds {path}")
    try:
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def row_key(fam: str, L: int, B: int, design: str, rnd: int, chunk: int, unit: str) -> str:
    return f"e33|{fam}|L{L}|B{B}|{design}|r{rnd}|c{chunk}|{unit}"


# --------------------------------------------------------------------------- run
def timed(args, fn_replay) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if args.exclusive_guard:
        waited = wait_for_exclusive_gpu(args.guard_until, None, index=args.guard_gpu)
        if waited is None:
            raise SystemExit("guard deadline passed while another process held the GPU")
        out["guard_wait_s"] = _g(waited, 4)
    if args.record_gpu_state:
        out["gpu_before"] = gpu_state(args.guard_gpu)
    ts = cuda_time(fn_replay, warmup=args.warmup, repeats=args.repeats, pre_spin_ms=args.pre_spin_ms)
    out["median_ms"] = _g(statistics.median(ts))
    out["mean_ms"] = _g(statistics.fmean(ts))
    # CUDA events resolve about 0.5 us, so the median of short replays is quantized; the 10 % trimmed mean is not
    srt = sorted(ts)
    cut = len(srt) // 10
    out["tmean_ms"] = _g(statistics.fmean(srt[cut: len(srt) - cut] or srt))
    out["std_ms"] = _g(statistics.pstdev(ts))
    out["min_ms"] = _g(min(ts))
    if args.record_gpu_state:
        out["gpu_after"] = gpu_state(args.guard_gpu)
    if args.exclusive_guard:
        out["foreign_after"] = bool(foreign_gpu_pids(args.guard_gpu))
    return out


def run(args, log) -> None:
    import torch

    jsonl = args.out_dir / f"e33_value_graph_{args.tag}.jsonl"
    check_disk(args.out_dir, need_gb=0.2, need_inodes=100, log=log)
    trace_dirs = dict(t.split("=", 1) for t in args.trace_dirs.split(",") if "=" in t)
    fams = [f for f in args.families.split(",") if f]
    shapes = _parse_shapes(args.shapes)
    with exclusive_lock(Path(args.log_dir) / f"e33_value_graph_{args.tag}.lock"):
        meta = env_metadata()
        meta.update({"experiment": "E33", "ssd_dtype": args.ssd_dtype, "families": fams, "shapes": shapes, "designs": args.designs,
                     "draws": args.draws, "placements": args.placements, "rounds": args.rounds,
                     "warmup": args.warmup, "repeats": args.repeats, "pre_spin_ms": args.pre_spin_ms,
                     "guard_gpu": args.guard_gpu, "args": {k: str(v) for k, v in vars(args).items()}})
        w = JsonlWriter(jsonl, meta=meta)
        inputs = [f"{d}#{k}" for d in SYNTH for k in range(args.draws if d != "constant" else 1)] + ["real"]
        if args.inputs:
            inputs = [i for i in inputs if i in set(args.inputs.split(","))]
        for fam in fams:
            kind = FAMILIES[fam]["kind"]
            td = trace_dirs.get(fam)
            real = load_real(Path(td), kind, log) if td else None
            for (L, B) in shapes:
                F = Family(fam, L, B, real, log, ssd_dtype=args.ssd_dtype)
                cell_inputs = [i for i in inputs if i != "real" or F.real is not None]
                chunks = [c for c in F.chunks if F.feasible(c)]
                log(f"E33 {fam} L={L} B={B}: chunks {chunks}, inputs {len(cell_inputs)} (real: {F.real_src}), "
                    f"designs {args.designs}")
                w.write({"kind": "cell", "key": f"e33cell|{fam}|L{L}|B{B}", "family": fam, "L": L, "batch": B,
                         "chunks": chunks, "inputs": cell_inputs, "real_source": F.real_src,
                         "tensor_shapes": F.inputs_desc, "utc": utc_now()}, track=False)
                with torch.inference_mode():
                    if "A" in args.designs:
                        design_a(args, log, w, F, chunks, cell_inputs)
                    if "C" in args.designs:
                        design_c(args, log, w, F, chunks)
                del F
                gc.collect()
                torch.cuda.empty_cache()
        w.close()
    try:
        analyze(argparse.Namespace(**{**vars(args), "input_tags": args.tag}))
    except Exception as exc:
        log(f"analysis failed: {exc!r}")


def design_a(args, log, w, F: Family, chunks: list[int], inputs: list[str]) -> None:
    import torch

    bufs, owners = F.alloc()
    vals = {i: F.values(i, args.seed) for i in inputs}           # all inputs held at once (values only)
    graphs = {}
    for c in chunks:
        for k, v in vals[inputs[0]].items():
            bufs[k].copy_(v)
        try:
            graphs[c] = capture(F.call(bufs, c))
        except torch.cuda.OutOfMemoryError as exc:  # e.g. Gated DeltaNet chunk 16 at 8192 x 16 (17 GB of chunk states)
            log(f"A {F.name} L={F.L} B={F.B} c{c}: out of memory, chunk skipped ({str(exc)[:80]})")
            w.write({"kind": "oom", "key": f"e33oom|{F.name}|L{F.L}|B{F.B}|c{c}", "family": F.name, "L": F.L,
                     "batch": F.B, "chunk": c, "utc": utc_now()}, track=False)
            gc.collect()
            torch.cuda.empty_cache()
    chunks = [c for c in chunks if c in graphs]
    for rnd in range(args.rounds):
        corder = list(chunks)
        random.Random(f"A|{F.name}|{F.L}|{F.B}|{rnd}").shuffle(corder)
        for c in corder:
            iorder = list(inputs)
            random.Random(f"A|{F.name}|{F.L}|{F.B}|{rnd}|{c}").shuffle(iorder)
            for inp in iorder:
                key = row_key(F.name, F.L, F.B, "A", rnd, c, inp)
                if key in w.done:
                    continue
                for k, v in vals[inp].items():
                    bufs[k].copy_(v)
                torch.cuda.synchronize()
                row = timed(args, graphs[c].replay)
                if args.exclusive_guard and row.get("foreign_after"):
                    row = {**timed(args, graphs[c].replay), "retimed": True}
                row.update({"key": key, "kind": "timing", "family": F.name, "L": F.L, "batch": F.B, "design": "A",
                            "round": rnd, "chunk": c, "input": inp, "dist": inp.split("#")[0], "utc": utc_now()})
                w.write(row)
            log(f"A {F.name} L={F.L} B={F.B} r{rnd} c{c}: {len(iorder)} inputs")
    del graphs, vals, bufs, owners


def design_c(args, log, w, F: Family, chunks: list[int]) -> None:
    import torch

    v0 = F.values("gaussian#0", args.seed)
    n = args.placements
    for rnd in range(args.rounds):
        corder = list(chunks)
        random.Random(f"C|{F.name}|{F.L}|{F.B}|{rnd}").shuffle(corder)
        for c in corder:
            if all(row_key(F.name, F.L, F.B, "C", rnd, c, f"p{j}") in w.done for j in range(n)):
                continue
            sets = []
            for j in range(n):
                off = 0 if j < n // 2 else 256 * (j - n // 2 + 1) * 37
                bufs, owners = F.alloc(off)
                for k, v in v0.items():
                    bufs[k].copy_(v)
                ptr = {k: int(t.data_ptr()) for k, t in bufs.items()}
                sets.append((j, off, bufs, owners, capture(F.call(bufs, c)), ptr))
            order = list(range(n))
            random.Random(f"C|{F.name}|{F.L}|{F.B}|{rnd}|{c}").shuffle(order)
            for j in order:
                _, off, bufs, owners, g, ptr = sets[j]
                key = row_key(F.name, F.L, F.B, "C", rnd, c, f"p{j}")
                if key in w.done:
                    continue
                row = timed(args, g.replay)
                if args.exclusive_guard and row.get("foreign_after"):
                    row = {**timed(args, g.replay), "retimed": True}
                row.update({"key": key, "kind": "timing", "family": F.name, "L": F.L, "batch": F.B, "design": "C",
                            "round": rnd, "chunk": c, "input": "gaussian#0", "placement": j,
                            "placement_kind": "fresh" if off == 0 else "offset", "offset_bytes": off,
                            "ptr_mod_4k": {k: p % 4096 for k, p in ptr.items()},
                            "ptr_mod_2m": {k: p % (2 << 20) for k, p in ptr.items()}, "utc": utc_now()})
                w.write(row)
            del sets
            gc.collect()
            torch.cuda.empty_cache()
        log(f"C {F.name} L={F.L} B={F.B} r{rnd}: {len(chunks)} chunks x {n} placements")


# --------------------------------------------------------------------------- analysis
def _tost(a: list[float], b: list[float], margin: float) -> dict[str, Any]:
    """Paired (by round) TOST on log latency ratios; equivalence if the 90% CI lies within +-margin."""
    from scipy import stats

    d = [math.log(x / y) for x, y in zip(a, b)]
    if len(d) < 3:
        return {"n": len(d), "equivalent": None}
    m, s = statistics.fmean(d), statistics.stdev(d)
    se = s / math.sqrt(len(d)) if s > 0 else 1e-12
    tcrit = stats.t.ppf(0.95, len(d) - 1)
    lo, hi = m - tcrit * se, m + tcrit * se
    lim = math.log(1 + margin)
    return {"n": len(d), "ratio": _g(math.exp(m), 5), "ci90": [_g(math.exp(lo), 5), _g(math.exp(hi), 5)],
            "equivalent": bool(lo > -lim and hi < lim)}


def analyze(args) -> dict[str, Any]:
    from scipy import stats

    tags = [t for t in str(args.input_tags or args.tag).split(",") if t]
    rows = []
    for t in tags:
        rows += [r for r in read_jsonl(args.out_dir / f"e33_value_graph_{t}.jsonl") if r.get("kind") == "timing"]
    rows = dedupe_by_key(rows)
    res: dict[str, Any] = {"experiment": "E33", "tags": tags, "created_utc": utc_now(), "configs": [],
                           "placements": [], "summary": {}}
    A = defaultdict(lambda: defaultdict(dict))   # (fam, L, B, chunk) -> input -> round -> median
    C = defaultdict(lambda: defaultdict(dict))   # (fam, L, B, chunk) -> placement -> round -> median
    for r in rows:
        k = (r["family"], r["L"], r["batch"], r["chunk"])
        t = r.get("tmean_ms", r["median_ms"])
        if r["design"] == "A":
            A[k][r["input"]][r["round"]] = t
        else:
            C[k][r["placement"]][r["round"]] = t
    best_by_input = defaultdict(dict)
    for (fam, L, B, c), per in sorted(A.items()):
        rounds = sorted(set.intersection(*[set(v) for v in per.values()]))
        inputs = sorted(per)
        if len(rounds) < 2 or "gaussian#0" not in per:
            continue
        med = {i: statistics.median([per[i][r] for r in rounds]) for i in inputs}
        dists = defaultdict(list)
        for i in inputs:
            dists[i.split("#")[0]].append(med[i])
        dist_means = {d: statistics.fmean(v) for d, v in dists.items()}
        within = [max(v) / min(v) - 1 for d, v in dists.items() if len(v) > 1]
        mat = [[per[i][r] for i in inputs] for r in rounds]
        try:
            fr = stats.friedmanchisquare(*[[mat[ri][ii] for ri in range(len(rounds))] for ii in range(len(inputs))])
            friedman = {"stat": _g(fr.statistic, 4), "p": _g(fr.pvalue, 4)}
        except Exception as exc:
            friedman = {"error": repr(exc)[:80]}
        ref = [per["gaussian#0"][r] for r in rounds]
        tost = {i: {"m1": _tost([per[i][r] for r in rounds], ref, 0.01),
                    "m2": _tost([per[i][r] for r in rounds], ref, 0.02)} for i in inputs if i != "gaussian#0"}
        cfg = {"family": fam, "L": L, "batch": B, "chunk": c, "n_rounds": len(rounds), "n_inputs": len(inputs),
               "median_ms": {i: _g(v, 5) for i, v in med.items()},
               "spread_inputs_pct": _g(100 * (max(med.values()) / min(med.values()) - 1), 4),
               "spread_distributions_pct": _g(100 * (max(dist_means.values()) / min(dist_means.values()) - 1), 4),
               "spread_between_draws_pct_max": _g(100 * max(within), 4) if within else None,
               "real_vs_gaussian0_pct": _g(100 * (med["real"] / med["gaussian#0"] - 1), 4) if "real" in med else None,
               "round_cv_median": _g(statistics.median([statistics.pstdev([per[i][r] for r in rounds]) /
                                                        statistics.fmean([per[i][r] for r in rounds]) for i in inputs]), 4),
               "friedman": friedman,
               "tost_equivalent_1pct": sum(1 for v in tost.values() if v["m1"]["equivalent"]),
               "tost_equivalent_2pct": sum(1 for v in tost.values() if v["m2"]["equivalent"]),
               "n_tost": len(tost), "tost": tost}
        res["configs"].append(cfg)
        for i in inputs:
            best_by_input[(fam, L, B)].setdefault(i, {})[c] = med[i]
    flips = {}
    for (fam, L, B), per in sorted(best_by_input.items()):
        best = {i: min(v, key=lambda c: (v[c], c)) for i, v in per.items() if len(v) > 1}
        if "gaussian#0" not in best:
            continue
        b0 = best["gaussian#0"]
        # regret of using the Gaussian-input best chunk on every other input
        reg = {i: _g(100 * (per[i][b0] / min(per[i].values()) - 1), 4) for i in per if b0 in per[i]}
        flips[f"{fam}|L{L}|B{B}"] = {"best_gaussian0": b0, "best_by_input": best,
                                     "n_inputs_other_best": sum(1 for i, c in best.items() if c != b0),
                                     "regret_of_gaussian_choice_pct_max": max(reg.values()) if reg else None,
                                     "regret_of_gaussian_choice_pct": reg}
    res["best_chunk_by_input"] = flips
    for (fam, L, B, c), per in sorted(C.items()):
        rounds = sorted(set.intersection(*[set(v) for v in per.values()]))
        if len(rounds) < 2:
            continue
        med = {j: statistics.median([per[j][r] for r in rounds]) for j in per}
        res["placements"].append({"family": fam, "L": L, "batch": B, "chunk": c, "n_rounds": len(rounds),
                                  "n_placements": len(med),
                                  "spread_placements_pct": _g(100 * (max(med.values()) / min(med.values()) - 1), 4),
                                  "median_ms": {str(j): _g(v, 5) for j, v in sorted(med.items())}})
    for fam in sorted({c["family"] for c in res["configs"]}):
        cs = [c for c in res["configs"] if c["family"] == fam]
        ps = [p for p in res["placements"] if p["family"] == fam]
        fl = [v for k, v in flips.items() if k.startswith(fam + "|")]
        res["summary"][fam] = {
            "n_configs": len(cs),
            "spread_inputs_pct_median": _g(statistics.median([c["spread_inputs_pct"] for c in cs]), 4),
            "spread_inputs_pct_max": _g(max(c["spread_inputs_pct"] for c in cs), 4),
            "spread_distributions_pct_max": _g(max(c["spread_distributions_pct"] for c in cs), 4),
            "spread_between_draws_pct_max": _g(max((c["spread_between_draws_pct_max"] or 0) for c in cs), 4),
            "real_vs_gaussian0_pct_range": [_g(min(c["real_vs_gaussian0_pct"] for c in cs if c["real_vs_gaussian0_pct"] is not None), 4),
                                            _g(max(c["real_vs_gaussian0_pct"] for c in cs if c["real_vs_gaussian0_pct"] is not None), 4)]
            if any(c["real_vs_gaussian0_pct"] is not None for c in cs) else None,
            "friedman_p_below_0.05": sum(1 for c in cs if (c["friedman"].get("p") or 1) < 0.05),
            "tost_2pct_equivalent": f"{sum(c['tost_equivalent_2pct'] for c in cs)}/{sum(c['n_tost'] for c in cs)}",
            "tost_1pct_equivalent": f"{sum(c['tost_equivalent_1pct'] for c in cs)}/{sum(c['n_tost'] for c in cs)}",
            "round_cv_median": _g(statistics.median([c["round_cv_median"] for c in cs]), 4),
            "shapes_with_input_dependent_best": sum(1 for f in fl if f["n_inputs_other_best"] > 0),
            "n_shapes": len(fl),
            "regret_of_gaussian_choice_pct_max": _g(max((f["regret_of_gaussian_choice_pct_max"] or 0) for f in fl), 4) if fl else None,
            "spread_placements_pct_median": _g(statistics.median([p["spread_placements_pct"] for p in ps]), 4) if ps else None,
            "spread_placements_pct_max": _g(max(p["spread_placements_pct"] for p in ps), 4) if ps else None}
    out_tag = tags[0] if len(tags) == 1 else "_".join(tags)
    path = args.out_dir / f"e33_value_graph_{out_tag}.json"
    write_json(path, res)
    print(f"wrote {path}")
    for fam, s in res["summary"].items():
        print(fam, s)
    return res


# --------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--families", default="ssd370m,ssd2p7b,gdn1p4b")
    p.add_argument("--shapes", default="2048x1,16384x1,4096x4", help="LxB list")
    p.add_argument("--trace-dirs", default="", help="family=trace_dir,... (captured calls for the real input)")
    p.add_argument("--designs", default="AC")
    p.add_argument("--draws", type=int, default=2)
    p.add_argument("--ssd-dtype", default="float16", choices=("float16", "bfloat16"),
                   help="data type of the SSD inputs (the study's default is float16; bfloat16 for the dtype check)")
    p.add_argument("--inputs", default=None, help="restrict to these inputs (e.g. gaussian#0: a plain timing grid)")
    p.add_argument("--placements", type=int, default=6)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--pre-spin-ms", type=float, default=50.0)
    p.add_argument("--exclusive-guard", action="store_true")
    p.add_argument("--guard-until", type=float, default=None)
    p.add_argument("--guard-gpu", type=int, default=None)
    p.add_argument("--record-gpu-state", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--log-dir", type=Path, default=LOG_DIR)
    p.add_argument("--tag", default=None)
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--input-tags", default=None)
    a = p.parse_args(argv)
    a.tag = a.tag or f"{host_name()}_e33"
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
    run(args, Logger(f"e33_value_graph_{args.tag}", args.log_dir))


if __name__ == "__main__":
    main()
