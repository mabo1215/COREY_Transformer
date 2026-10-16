"""E30 - Triton autotuner state of the SSD and Gated DeltaNet kernels: pinned vs re-tuned per shape (M6c).

The Mamba-2 SSD kernels (``mamba_chunk_scan_combined``, mamba_ssm) and the Gated DeltaNet kernels
(``chunk_gated_delta_rule``, FLA) choose their inner tiles (block sizes, warps, stages) with Triton
autotuners. An autotuner benchmarks its configurations the first time it meets a tuning key and caches
the winner in memory for the rest of the process. The keys hold head, state and chunk dimensions and the
tensor dtypes, but (with one FLA exception, below) neither the sequence length nor the batch size, so in
the CUDA-graph sweeps (``e2_oracle_sweep.py --cuda-graph``) every call reused inner tiles that were tuned
on the first call that reached the key. This script documents that state and re-times a stratified subset
of the sweep's calls under two autotuner states:

* ``pinned`` (the study's state): one autotuner cache for the whole process, rebuilt by replaying the
  sweep's call sequence. The plan of ``e2_oracle_sweep`` (batches 1, 4, 16, same groups and seed) is
  walked in order; each new (batch, length) shape is run once eagerly at every chunk, in the chunk order
  of round 0 of that call in the three-round sweep. Every key is therefore tuned on the first call (and,
  for keys without the chunk, the first chunk) that reached it, exactly as in the sweep process, which ran
  without restarts. Shapes after the first are synthesized by tiling the first call's tensors (the kernels'
  latency does not depend on the values). With ``--pinned-disk`` (default ``auto``: on when the library
  enables Triton's persistent autotune cache, as FLA does) every key found in Triton's on-disk autotune
  cache takes the cached configuration instead, which is the configuration the earlier sweeps loaded.
* ``retuned``: for every chunk separately, all autotuner caches are cleared and the call itself is run
  once eagerly, so every kernel is benchmarked on this exact (shape, chunk).

Re-tuning is real: every in-memory cache is cleared, Triton's persistent autotune cache is switched off
per autotuner (``cache_results = False``; FLA turns it on by default through ``FLA_CACHE_RESULTS``), and
FLA's own config-directory lookup (``FLA_CACHE_MODE``) is forced to ``disabled``. Compiled kernels stay
cached; only the choice of configuration is redone.

Timing per call: ``--rounds`` rounds (default 3); each round times every (state, chunk) pair once, in an
order shuffled per call and round (seeded by subset seed, call key and round). A pair is timed by
restoring that state's cache, capturing the call into a CUDA graph (as the sweep's ``--cuda-graph`` mode)
and timing ``--repeats`` replays after ``--warmup`` (warm L2); ``--no-graph`` times eager calls instead.
Per (state, chunk, round) the row stores n, mean, median, std and min; it also stores the configuration of
every autotuner key the call uses in both states, whether it changed, the autotuner's own benchmark of the
chosen and the pinned configuration on this shape, the maximum output difference between the two states
at each chunk (they compute the same recurrence) and any key that had to be tuned while timing (none is
expected). SSD chunks beyond the kernel's 32-bit index limit are skipped as in the sweep.

Subset: about ``--n-calls`` calls (default 40) drawn evenly over the (batch, power-of-two length bucket)
cells of the full plan with a seeded RNG, plus the first call of the plan as an ``anchor`` row (its pinned
keys were tuned on that very call, so its configuration changes measure autotuning noise). Call keys are
those of the sweep (``<kind>|b<batch>|<group id>|layer<layer>``), so rows join with its CSVs.

Autotuners found on the RTX 3090 host (mamba_ssm 2.3.2.post1, fla 0.5.2, Triton 3.4.0; the run records
the exact list in its metadata):
  SSD forward: _chunk_cumsum_fwd_kernel [chunk_size, nheads] 7 configs; _chunk_state_fwd_kernel [hdim,
  dstate, chunk_size] 9; _state_passing_fwd_kernel [dim] 6 (no chunk in the key); _bmm_chunk_fwd_kernel
  [chunk_size, K, IS_CAUSAL] 9; _chunk_scan_fwd_kernel [chunk_size, hdim, dstate, IS_CAUSAL] 11; no
  persistent cache.
  GDN forward (FLA CachedAutotuner, cache_results=True): l2norm_fwd_kernel [D, NB] 25, where
  NB = ceil(batch * length * heads / 65536) depends on batch and length; gdn_gate_chunk_cumsum_scalar_kernel
  [H, BT, IS_VARLEN, REVERSE] 4; chunk_gated_delta_rule_fwd_kkt_solve_kernel [H, HV, K, BC] 6 (chunk 64);
  chunk_scaled_dot_kkt_fwd_kernel [H, HV, K, BT, IS_VARLEN] 27 with solve_tril_16x16_kernel [BT] 16 (chunk 16)
  or merge_16x16_to_32x32_inverse_kernel [H, BT, IS_VARLEN] 16 (chunk 32); recompute_w_u_fwd_kernel [H, HV, K,
  V, BT, BK, BV, IS_VARLEN] 9; chunk_gated_delta_rule_fwd_kernel_h_blockdim64 [H, HV, K, V, BT, STATE_V_FIRST]
  8 (the list depends on the GPU's shared memory); chunk_fwd_kernel_o [H, HV, K, V, BT, STATE_V_FIRST] 3.
  Every key also carries the dtypes of the tensor arguments.

Outputs (``--out-dir``, default the TPDS results directory):
  e30_autotune_state_<tag>.jsonl         one fsynced row per completed call (resumable: completed keys are
                                         skipped), plus meta rows and an ``autotune_doc`` row per start
  e30_autotune_state_<tag>_pinned.json   the pinned cache (reused on restart, so all rows share it), the
                                         call that tuned each key and the on-disk comparison
  e30_autotune_state_<tag>.json          ``--analyze`` (CPU only; several runs via ``--tags``/``--inputs``)

Other modes: ``--dry-run`` builds the plan and prints the stratified subset (manifest only; no kernels,
no GPU); ``--max-rows N`` stops after N new rows (smoke); ``--walk-max-shapes N`` truncates the pinned
walk (smoke only; the pinned file records it).
"""
from __future__ import annotations

import argparse
import os
import ast
import glob
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from src.experiments.tpds.common import (DEFAULT_TRACE_DIR, GDN_CHUNKS, MAMBA2_CHUNKS, RESULTS_DIR, JsonlWriter,
                                         Logger, check_disk, host_name, read_jsonl, write_json)

CHUNKS = {"mamba2": tuple(MAMBA2_CHUNKS), "gdn": tuple(GDN_CHUNKS)}
DEFAULT_CHUNK = {"mamba2": 256, "gdn": 64}  # library defaults (mamba_ssm Mamba2 layers, FLA chunk_gated_delta_rule)
LIB_PREFIXES = {"mamba2": ("mamba_ssm.ops.triton",), "gdn": ("fla.",)}
SEQ_TENSORS = {"mamba2": ("x", "dt", "B", "C"), "gdn": ("q", "k", "v", "g", "beta")}
STATES = ("pinned", "retuned")
TOL = 0.02  # relative tolerance for "faster / slower" and "agreement within 2%"


# =========================================================================== plan (no torch, no kernels)
def read_manifest(trace_dir: Path, kind: str) -> list[dict]:
    """The capture manifest, last row per key (as ``capture_traces.load_manifest``, without importing torch)."""
    seen: dict[str, dict] = {}
    for r in read_jsonl(Path(trace_dir) / kind / "manifest.jsonl"):
        seen[r["key"]] = r
    return list(seen.values())


def length_bucket(L: int) -> int:
    """Power-of-two length bucket (as ``e2_oracle_sweep.length_bucket``)."""
    return 1 << max(0, int(math.floor(math.log2(max(L, 1)))))


def build_groups(manifest, layers, batch, seed=0):
    """Copy of ``e2_oracle_sweep.build_groups`` (kept import-free for ``--dry-run``; the GPU path checks
    that both give the same plan)."""
    groups = []
    rng = random.Random(seed + batch)
    for layer in layers:
        if batch == 1:
            for r in sorted(manifest, key=lambda r: r["prompt_id"]):
                groups.append((layer, [r]))
            continue
        cells: dict[tuple, list] = {}
        for r in manifest:
            cells.setdefault((r["fold"], length_bucket(r["L"])), []).append(r)
        for (_, _), rs in sorted(cells.items()):
            rs = sorted(rs, key=lambda r: (r["L"], r["prompt_id"]))
            for i in range(0, len(rs), batch):
                members = rs[i:i + batch]
                if len(members) < batch:
                    pool = list(rs)
                    rng.shuffle(pool)
                    j = 0
                    while len(members) < batch:
                        members.append(pool[j % len(pool)])
                        j += 1
                groups.append((layer, members))
    out = []
    for layer, members in groups:
        ids = [m["prompt_id"] for m in members]
        gid = hashlib.sha1("|".join(ids).encode()).hexdigest()[:12]
        out.append((gid, layer, members))
    return out


def call_key(kind: str, batch: int, gid: str, layer: int) -> str:
    return f"{kind}|b{batch}|{gid}|layer{layer}"


def full_plan(manifest, kind, layers, batches, seed=0) -> list[dict]:
    """The sweep's call plan in sweep order (batches in the given order, groups as ``build_groups``)."""
    plan = []
    for b in batches:
        for gid, layer, members in build_groups(manifest, layers, b, seed):
            L = min(m["L"] for m in members)
            plan.append({"index": len(plan), "key": call_key(kind, b, gid, layer), "batch": b, "gid": gid,
                         "layer": layer, "members": members, "L": L, "bucket": length_bucket(L)})
    return plan


def stratified_subset(plan, n_calls, seed=0) -> list[dict]:
    """About ``n_calls`` calls spread evenly over the (batch, length bucket) cells; deterministic.

    Every cell gets the same quota (capped by its size); the remainder goes one call each to cells drawn
    with the seeded RNG; calls are drawn without replacement inside each cell. Returned in plan order.
    """
    cells: dict[tuple, list] = {}
    for p in plan:
        cells.setdefault((p["batch"], p["bucket"]), []).append(p)
    order = sorted(cells)
    rng = random.Random(f"e30-subset|{seed}|{n_calls}")
    quota = {s: 0 for s in order}
    remaining = min(n_calls, len(plan))
    while remaining > 0:
        open_cells = [s for s in order if quota[s] < len(cells[s])]
        if not open_cells:
            break
        if remaining >= len(open_cells):
            for s in open_cells:
                quota[s] += 1
            remaining -= len(open_cells)
        else:
            for s in sorted(rng.sample(open_cells, remaining)):
                quota[s] += 1
            remaining = 0
    chosen = []
    for s in order:
        chosen += rng.sample(cells[s], quota[s])
    return sorted(chosen, key=lambda p: p["index"])


def chunk_order_round0(chunks, key, seed=0) -> list[int]:
    """Chunk order of round 0 of a call in the multi-round sweep (``e2_oracle_sweep`` with ``--rounds`` > 1)."""
    order = list(chunks)
    random.Random(f"{seed}|{key}|r0").shuffle(order)
    return order


def walk_items(plan, chunks, seed=0, max_shapes=None) -> list[dict]:
    """First call of every distinct (batch, length) shape, in plan order, with its round-0 chunk order."""
    seen, out = set(), []
    for p in plan:
        shape = (p["batch"], p["L"])
        if shape in seen:
            continue
        seen.add(shape)
        out.append({"key": p["key"], "index": p["index"], "batch": p["batch"], "L": p["L"],
                    "order": chunk_order_round0(chunks, p["key"], seed)})
        if max_shapes is not None and len(out) >= max_shapes:
            break
    return out


def plan_for_args(args, manifest):
    layers = sorted({l for r in manifest for l in r["layers"]})
    if args.layers:
        keep = {int(x) for x in args.layers.split(",")}
        layers = [l for l in layers if l in keep]
    batches = [int(b) for b in args.batches.split(",")]
    plan = full_plan(manifest, args.model, layers, batches, args.seed)
    subset = stratified_subset(plan, args.n_calls, args.subset_seed)
    anchor = plan[0]
    rows = [dict(p, role="sample") for p in subset]
    if any(p["key"] == anchor["key"] for p in rows):
        for p in rows:
            if p["key"] == anchor["key"]:
                p["role"] = "sample+anchor"
    else:
        rows.insert(0, dict(anchor, role="anchor"))
    return layers, batches, plan, rows


def dry_run(args) -> None:
    manifest = read_manifest(args.trace_dir, args.model)
    if not manifest:
        raise SystemExit(f"no manifest for {args.model} under {args.trace_dir}")
    layers, batches, plan, rows = plan_for_args(args, manifest)
    chunks = parse_chunks(args)
    walk = walk_items(plan, chunks, args.seed, args.walk_max_shapes)
    print(f"{args.model}: {len(manifest)} prompts, layers {layers}, batches {batches}, {len(plan)} calls in the plan")
    cells: dict[tuple, list] = {}
    for p in plan:
        cells.setdefault((p["batch"], p["bucket"]), []).append(p)
    picked: dict[tuple, int] = {}
    for p in rows:
        if p["role"] != "anchor":
            picked[(p["batch"], p["bucket"])] = picked.get((p["batch"], p["bucket"]), 0) + 1
    print(f"{len(cells)} (batch, length bucket) cells; selected {sum(picked.values())} calls "
          f"(+ anchor = {len(rows)} rows)")
    print("  batch  bucket  plan_calls  selected")
    for s in sorted(cells):
        print(f"  {s[0]:>5}  {s[1]:>6}  {len(cells[s]):>10}  {picked.get(s, 0):>8}")
    print(f"pinned walk: {len(walk)} distinct (batch, length) shapes x {len(chunks)} chunks "
          f"(first: {walk[0]['key']} L={walk[0]['L']}, round-0 chunk order {walk[0]['order']})")
    print("rows (plan index, role, key, batch, L):")
    for p in rows:
        print(f"  {p['index']:>5}  {p['role']:<13}  {p['key']}  b={p['batch']}  L={p['L']}")


def parse_chunks(args) -> list[int]:
    chunks = [int(c) for c in args.chunks.split(",")] if args.chunks else list(CHUNKS[args.model])
    return chunks


# =========================================================================== autotuner control (GPU path)
def cfg_str(cfg) -> str | None:
    """Canonical string of a triton.Config (kwargs sorted, then warps, stages, CTAs, maxnreg)."""
    if cfg is None:
        return None
    kw = ",".join(f"{k}={cfg.kwargs[k]}" for k in sorted(cfg.kwargs))
    return (f"{kw}|warps={cfg.num_warps}|stages={cfg.num_stages}|ctas={getattr(cfg, 'num_ctas', 1)}"
            f"|maxnreg={getattr(cfg, 'maxnreg', None)}")


def cfg_to_dict(cfg) -> dict:
    return {"kwargs": dict(cfg.kwargs), "num_warps": cfg.num_warps, "num_stages": cfg.num_stages,
            "num_ctas": getattr(cfg, "num_ctas", 1), "maxnreg": getattr(cfg, "maxnreg", None)}


def cfg_dict_str(d: dict) -> str:
    """``cfg_str`` of a configuration stored with ``cfg_to_dict``."""
    kw = ",".join(f"{k}={d['kwargs'][k]}" for k in sorted(d["kwargs"]))
    return (f"{kw}|warps={d['num_warps']}|stages={d['num_stages']}|ctas={d.get('num_ctas', 1)}"
            f"|maxnreg={d.get('maxnreg')}")


def _bench_ms(t) -> float | None:
    """Median of one autotuner benchmark entry (Triton stores [median, p20, p80] or a float)."""
    if t is None:
        return None
    v = t[0] if isinstance(t, (list, tuple)) else t
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


class _NoBench(Exception):
    """Raised instead of benchmarking when only the on-disk autotune cache may be read."""


_MISSING = object()


def _follow_to_autotuner(obj, cls, prefixes, depth=8):
    """Follow wrapper chains (triton Heuristics and similar, via ``.fn``) to an Autotuner instance."""
    chain = []
    for _ in range(depth):
        if isinstance(obj, cls):
            return obj, chain
        mod = getattr(type(obj), "__module__", "") or ""
        if not (mod.startswith("triton") or mod.startswith(prefixes)):
            return None, chain  # only walk through Triton / library wrapper objects
        try:
            nxt = getattr(obj, "fn", None)
        except Exception:
            return None, chain
        if nxt is None or nxt is obj:
            return None, chain
        chain.append(type(obj).__name__)
        obj = nxt
    return None, chain


class AutotuneControl:
    """Find, document, clear, snapshot and restore every Triton autotuner of one kernel library."""

    def __init__(self, prefixes, log=print):
        self.prefixes = tuple(prefixes)
        self.log = log
        self.tuners: dict[str, Any] = {}
        self._by_id: dict[int, str] = {}
        self.doc: dict[str, dict] = {}
        self.used: set[str] = set()
        self.fla_cache_mode_original: str | None = None
        self.scan()

    def scan(self) -> list[str]:
        """Register autotuners not seen yet (modules may be imported lazily); returns their names."""
        from triton.runtime.autotuner import Autotuner

        new = []
        for mname in sorted(list(sys.modules)):
            if not mname.startswith(self.prefixes):
                continue
            mod = sys.modules.get(mname)
            if mod is None:
                continue
            try:
                items = sorted(vars(mod).items(), key=lambda kv: kv[0])
            except Exception:
                continue
            for attr, val in items:
                at, chain = _follow_to_autotuner(val, Autotuner, self.prefixes)
                if at is None:
                    continue
                if id(at) in self._by_id:
                    aliases = self.doc[self._by_id[id(at)]]["aliases"]
                    if f"{mname}.{attr}" not in aliases:
                        aliases.append(f"{mname}.{attr}")
                    continue
                base = getattr(at, "base_fn", None) or at.fn
                name = f"{getattr(base, '__module__', mname)}.{getattr(base, '__qualname__', attr)}"
                if name in self.tuners:
                    k = 2
                    while f"{name}#{k}" in self.tuners:
                        k += 1
                    name = f"{name}#{k}"
                self.tuners[name] = at
                self._by_id[id(at)] = name
                self.doc[name] = {
                    "class": f"{type(at).__module__}.{type(at).__name__}", "keys": list(at.keys),
                    "n_configs": len(at.configs), "wrappers": chain,
                    "cache_results_library": bool(getattr(at, "cache_results", False)),
                    "early_config_prune": bool(getattr(at, "early_config_prune", None)),
                    "perf_model": bool(getattr(at, "perf_model", None)),
                    "reset_to_zero": list(getattr(at, "reset_to_zero", []) or []),
                    "restore_value": list(getattr(at, "restore_value", []) or []),
                    "aliases": [f"{mname}.{attr}"]}
                if hasattr(at, "cache_results"):
                    at.cache_results = False  # never read or write Triton's on-disk autotune cache
                new.append(name)
        cm = sys.modules.get("fla.ops.utils.cache")  # FLA's own config-directory lookup
        if cm is not None and hasattr(cm, "FLA_CACHE_MODE") and hasattr(cm, "FlaCacheMode"):
            if self.fla_cache_mode_original is None:
                self.fla_cache_mode_original = str(getattr(cm.FLA_CACHE_MODE, "value", cm.FLA_CACHE_MODE))
            cm.FLA_CACHE_MODE = cm.FlaCacheMode.DISABLED
        return new

    def clear(self) -> None:
        for at in self.tuners.values():
            at.cache.clear()

    def snapshot(self) -> dict[str, dict]:
        snap = {n: dict(at.cache) for n, at in self.tuners.items() if at.cache}
        self.used.update(snap)
        return snap

    def restore(self, snap: dict[str, dict]) -> None:
        self.clear()
        for n, entries in snap.items():
            self.tuners[n].cache.update(entries)

    def signature(self) -> set[tuple[str, str]]:
        return {(n, repr(k)) for n, at in self.tuners.items() for k in at.cache}

    def reset_timings(self) -> None:
        for at in self.tuners.values():
            at.configs_timings = None

    def last_benchmarks(self) -> dict[str, tuple]:
        """(key, {config string: median ms}) of the last benchmark of each autotuner since reset_timings()."""
        out = {}
        for n, at in self.tuners.items():
            t = getattr(at, "configs_timings", None)
            if t and at.cache:
                out[n] = (list(at.cache)[-1], {cfg_str(c): _bench_ms(v) for c, v in t.items()})
        return out

    def config_from_dict(self, name: str, d: dict):
        """The autotuner's own Config object equal to ``d`` (keeps pre-hooks), else a new triton.Config."""
        import triton

        want = cfg_dict_str(d)
        for c in self.tuners[name].configs:
            if cfg_str(c) == want:
                return c
        try:
            return triton.Config(d["kwargs"], num_warps=d["num_warps"], num_stages=d["num_stages"],
                                 num_ctas=d.get("num_ctas", 1), maxnreg=d.get("maxnreg"))
        except TypeError:
            return triton.Config(d["kwargs"], num_warps=d["num_warps"], num_stages=d["num_stages"])

    def disk_config(self, name: str, key) -> tuple[Any, str]:
        """Configuration Triton's on-disk autotune cache holds for (autotuner, key); never benchmarks or writes.

        Calls the autotuner's own ``check_disk_cache`` (same cache key as Triton computes it) with a
        benchmark function that raises, so a miss leaves the disk untouched.
        """
        at = self.tuners[name]
        if not hasattr(at, "check_disk_cache"):
            return None, "unsupported (no check_disk_cache)"
        if getattr(at, "early_config_prune", None) or getattr(at, "perf_model", None):
            return None, "unsupported (pruned configs)"
        saved = at.cache.pop(key, _MISSING)
        saved_t = getattr(at, "configs_timings", None)

        def _no_bench():
            raise _NoBench()

        try:
            hit = bool(at.check_disk_cache(key, at.configs, _no_bench))
            status = "hit" if hit else "miss"
        except _NoBench:
            hit, status = False, "miss"
        except Exception as exc:  # pragma: no cover - depends on the Triton build
            hit, status = False, f"error: {exc!r}"
        cfg = at.cache.get(key) if hit else None
        at.cache.pop(key, None)
        if saved is not _MISSING:
            at.cache[key] = saved
        at.configs_timings = saved_t
        return cfg, status

    def documentation(self) -> dict:
        used = sorted(self.used)
        return {"n_autotuners_scanned": len(self.tuners), "library_prefixes": list(self.prefixes),
                "fla_cache_mode_original": self.fla_cache_mode_original,
                "used": {n: self.doc[n] for n in used},
                "note": "autotuners with one configuration never tune and are not listed as used"}


def serialize_snapshot(snap: dict[str, dict]) -> dict[str, list]:
    return {n: [[repr(k), cfg_to_dict(c)] for k, c in entries.items()] for n, entries in sorted(snap.items())}


def deserialize_snapshot(ctl: AutotuneControl, data: dict[str, list]) -> dict[str, dict]:
    missing = [n for n in data if n not in ctl.tuners]
    if missing:
        raise SystemExit(f"pinned file names autotuners this process does not have: {missing[:5]}")
    return {n: {ast.literal_eval(k): ctl.config_from_dict(n, d) for k, d in entries} for n, entries in data.items()}


# =========================================================================== GPU helpers
def synth_like(kind, tmpl, batch, L):
    """Inputs of shape (batch, L, ...) made by tiling the template call's tensors (params shared)."""
    out = dict(tmpl)
    for k in SEQ_TENSORS[kind]:
        t = tmpl[k]
        rb, rl = -(-batch // t.shape[0]), -(-L // t.shape[1])
        out[k] = t.repeat(rb, rl, *([1] * (t.dim() - 2)))[:batch, :L].contiguous()
    return out


def feasible_chunks(kind, inp, chunks, K):
    if kind != "mamba2":
        return list(chunks)
    return [c for c in chunks if K.ssd_chunk_feasible(inp["x"], inp["B"], c)]


def output_diff(y_a, y_b) -> tuple[float, bool]:
    """Maximum absolute difference (computed per batch row to bound memory) and whether any value is non-finite."""
    import torch

    diff, nonfinite = 0.0, False
    for i in range(y_a.shape[0]):
        a, b = y_a[i].float(), y_b[i].float()
        nonfinite = nonfinite or not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all())
        diff = max(diff, float((a - b).abs().nan_to_num(nan=math.inf).max()))
    return diff, nonfinite


class _Foreign(Exception):
    """A foreign GPU process appeared; the current unit of work is redone."""


# =========================================================================== GPU run
def run(args) -> None:
    import torch

    from src.experiments.tpds import e2_oracle_sweep as E2
    from src.experiments.tpds import kernels as K
    from src.experiments.tpds.capture_traces import load_manifest, load_params
    from src.experiments.tpds.common import (all_gpu_pids, cuda_time, env_metadata, foreign_gpu_pids, peak_vram_gb,
                                             register_own_gpu_pids, summarize, wait_for_exclusive_gpu)

    kind = args.model
    tag = args.tag or f"{host_name()}_{kind}"
    log = Logger(f"e30_autotune_state_{tag}")
    out_dir = Path(args.out_dir)
    check_disk(out_dir, need_gb=0.5, log=log)
    jsonl_path = out_dir / f"e30_autotune_state_{tag}.jsonl"
    pinned_path = out_dir / f"e30_autotune_state_{tag}_pinned.json"

    # the guard watches the physical GPU this process runs on (first entry of CUDA_VISIBLE_DEVICES)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    gidx = int(vis) if vis.isdigit() else 0
    gpu_baseline = all_gpu_pids(gidx)
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    own = register_own_gpu_pids(gpu_baseline, index=gidx, log=log) if args.exclusive_guard else set()

    manifest = load_manifest(args.trace_dir, kind)
    if not manifest:
        raise SystemExit(f"no captured traces for {kind} under {args.trace_dir}")
    layers, batches, plan, rows = plan_for_args(args, manifest)
    e2_keys = [call_key(kind, b, gid, layer) for b in batches
               for gid, layer, _ in E2.build_groups(manifest, layers, b, args.seed)]
    if e2_keys != [p["key"] for p in plan]:
        raise SystemExit("plan differs from e2_oracle_sweep.build_groups; refuse to run (keys would not join)")
    chunks = parse_chunks(args)
    if args.chunks and set(chunks) != set(CHUNKS[kind]):
        log(f"WARNING: chunk subset {chunks}")
    walk = walk_items(plan, chunks, args.seed, args.walk_max_shapes)
    log(f"{kind}: {len(manifest)} prompts, layers {layers}, {len(plan)} plan calls, {len(rows)} rows "
        f"({sum(r['role'] != 'anchor' for r in rows)} sampled + anchor), pinned walk over {len(walk)} shapes")

    if kind == "mamba2":
        K._load_ssd()
    else:
        K._load_gdn()
    ctl = AutotuneControl(LIB_PREFIXES[kind], log)
    log(f"{len(ctl.tuners)} autotuners under {ctl.prefixes}; FLA_CACHE_MODE was {ctl.fla_cache_mode_original}")
    params = {l: load_params(args.trace_dir, kind, l) for l in layers}
    anchor = plan[0]
    tmpl, _, _, _ = E2.stack_inputs(kind, args.trace_dir, anchor["members"], anchor["layer"], params[anchor["layer"]])

    def guard_wait():
        if args.exclusive_guard and wait_for_exclusive_gpu(args.guard_until, log, index=gidx) is None:
            log("GPU still busy at the guard deadline; stopping (restart resumes)")
            raise SystemExit(3)

    def guard_check(where: str):
        if args.exclusive_guard:
            pids = foreign_gpu_pids(gidx)
            if pids:
                log(f"foreign GPU processes {pids} during {where}; redoing it")
                raise _Foreign()

    # ------------------------------------------------------------------ pinned state
    signature = hashlib.sha1(json.dumps({"kind": kind, "chunks": chunks, "seed": args.seed,
                                         "walk": [[w["key"], w["batch"], w["L"], w["order"]] for w in walk]},
                                        sort_keys=True).encode()).hexdigest()[:16]
    if pinned_path.exists():
        saved = json.loads(pinned_path.read_text(encoding="utf-8"))
        if saved.get("signature") != signature:
            raise SystemExit(f"{pinned_path} was built for another plan/walk (signature {saved.get('signature')} "
                             f"!= {signature}); move it and the JSONL aside first")
        ctl.used.update(saved["snapshot"])
        pinned = deserialize_snapshot(ctl, saved["snapshot"])
        pinned_info = {k: saved[k] for k in ("origin", "disk", "pinned_disk", "walk_s", "built_utc") if k in saved}
        log(f"pinned state loaded from {pinned_path} ({sum(len(v) for v in pinned.values())} keys, on-disk "
            f"configurations {'used' if pinned_info.get('pinned_disk') else 'not used'}; --pinned-disk is ignored)")
    else:
        pinned, pinned_info = None, None
        for attempt in range(50):
            guard_wait()
            try:
                t0 = time.time()
                walk_snap, origin = None, {}
                for _rescan in range(3):
                    ctl.clear()
                    origin = {}
                    for wi, w in enumerate(walk):
                        inp = tmpl if wi == 0 else synth_like(kind, tmpl, w["batch"], w["L"])
                        ok = feasible_chunks(kind, inp, w["order"], K)
                        for c in w["order"]:
                            if c not in ok:
                                continue
                            y = E2.make_fn(kind, inp, c)()
                            torch.cuda.synchronize()
                            del y
                            for n, at in ctl.tuners.items():
                                for k in at.cache:
                                    origin.setdefault(n, {}).setdefault(
                                        repr(k), {"call": w["key"], "plan_index": w["index"], "batch": w["batch"],
                                                  "L": w["L"], "chunk": c})
                        del inp
                        if (wi + 1) % 50 == 0:
                            guard_check("the pinned walk")
                            log(f"pinned walk: {wi + 1}/{len(walk)} shapes ({time.time() - t0:.0f} s)")
                    walk_snap = ctl.snapshot()
                    if not ctl.scan():
                        break
                    log("new autotuners appeared during the walk (lazy import); repeating the walk")
                guard_check("the pinned walk")
            except _Foreign:
                continue
            break
        else:
            raise SystemExit("could not complete the pinned walk without foreign GPU processes")
        walk_s = time.time() - t0
        use_disk = args.pinned_disk == "on" or (
            args.pinned_disk == "auto" and any(ctl.doc[n]["cache_results_library"] for n in walk_snap))
        disk, pinned = {}, {n: dict(e) for n, e in walk_snap.items()}
        for n, entries in walk_snap.items():
            if not ctl.doc[n]["cache_results_library"]:
                continue
            for k, cfg in entries.items():
                dcfg, status = ctl.disk_config(n, k)
                same = None if dcfg is None else cfg_str(dcfg) == cfg_str(cfg)
                disk.setdefault(n, {})[repr(k)] = {"status": status, "config": cfg_str(dcfg),
                                                   "walk_config": cfg_str(cfg), "equals_walk": same}
                if use_disk and dcfg is not None:
                    pinned[n][k] = dcfg
        pinned_info = {"origin": origin, "disk": disk, "pinned_disk": use_disk, "walk_s": round(walk_s, 1),
                       "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        write_json(pinned_path, {"signature": signature, "kind": kind, "chunks": chunks, "n_walk_shapes": len(walk),
                                 "walk_max_shapes": args.walk_max_shapes, **pinned_info,
                                 "snapshot": serialize_snapshot(pinned), "walk_snapshot": serialize_snapshot(walk_snap),
                                 "documentation": ctl.documentation()})
        n_disk = sum(1 for d in disk.values() for v in d.values() if v["status"] == "hit")
        n_eq = sum(1 for d in disk.values() for v in d.values() if v["equals_walk"])
        log(f"pinned walk done in {walk_s:.0f} s: {sum(len(v) for v in pinned.values())} keys over "
            f"{len(pinned)} autotuners; on-disk cache hits {n_disk} (equal to the fresh walk: {n_eq}); "
            f"pinned state {'uses' if use_disk else 'ignores'} the on-disk configurations")
    origin = (pinned_info or {}).get("origin", {})

    meta = env_metadata()
    meta.update({"experiment": "E30", "model": kind, "label": args.label, "tag": tag, "args": vars(args),
                 "layers": layers, "batches": batches, "chunks": chunks, "n_plan_calls": len(plan),
                 "n_rows_planned": len(rows), "walk_shapes": len(walk), "pinned_signature": signature,
                 "own_gpu_pids": sorted(own), "default_chunk": DEFAULT_CHUNK[kind],
                 "ssd_index_limit": K.SSD_INDEX_LIMIT if kind == "mamba2" else None})
    w = JsonlWriter(jsonl_path, meta=meta)
    w.write({"kind": "autotune_doc", "tag": tag, "label": args.label, "model": kind,
             "documentation": ctl.documentation(), "pinned_disk": (pinned_info or {}).get("pinned_disk"),
             "disk": (pinned_info or {}).get("disk"), "pinned_keys": {n: sorted(repr(k) for k in e)
                                                                      for n, e in pinned.items()}}, track=False)

    settings = {"cuda_graph": not args.no_graph, "warmup": args.warmup, "repeats": args.repeats,
                "pre_spin_ms": float(args.pre_spin_ms), "rounds": int(args.rounds)}

    def time_pair(snap, inp, c):
        ctl.restore(snap)
        before = ctl.signature()
        fn = E2.make_fn(kind, inp, c)
        if args.no_graph:
            fn()
            timed, graph = fn, None
        else:
            timed, graph = E2.capture_graph(fn)
        st = summarize(cuda_time(timed, args.warmup, args.repeats, False, args.pre_spin_ms))
        del timed, graph
        new = sorted(ctl.signature() - before)
        return {k: st[k] for k in ("n", "mean_ms", "median_ms", "std_ms", "min_ms")}, new

    def process(p) -> dict:
        inp, act, _, Lmin = E2.stack_inputs(kind, args.trace_dir, p["members"], p["layer"], params[p["layer"]])
        ok = feasible_chunks(kind, inp, chunks, K)
        skipped = [c for c in chunks if c not in ok]
        torch.cuda.reset_peak_memory_stats()
        # 1. re-tune every kernel on this exact (shape, chunk)
        t0 = time.time()
        retuned, bench = {}, {}
        for c in ok:
            ctl.clear()
            ctl.reset_timings()
            y = E2.make_fn(kind, inp, c)()
            torch.cuda.synchronize()
            del y
            retuned[c] = ctl.snapshot()
            bench[c] = ctl.last_benchmarks()
        retune_s = time.time() - t0
        guard_check(f"{p['key']} (re-tuning)")
        # 2. configuration of every key the call uses, both states
        autotune, changed_any = {}, {}
        for c in ok:
            entries = []
            for n, cache in sorted(retuned[c].items()):
                for k, cfg in cache.items():
                    pc = pinned.get(n, {}).get(k)
                    e = {"tuner": n, "key": repr(k), "pinned": cfg_str(pc), "retuned": cfg_str(cfg),
                         "changed": None if pc is None else cfg_str(pc) != cfg_str(cfg),
                         "pinned_origin": origin.get(n, {}).get(repr(k))}
                    b = bench[c].get(n)
                    if b is not None and b[0] == k:
                        times = b[1]
                        finite = [t for t in times.values() if t is not None]
                        e.update({"n_benchmarked": len(times), "best_ms": min(finite) if finite else None,
                                  "retuned_cfg_ms": times.get(cfg_str(cfg)),
                                  "pinned_cfg_ms": times.get(cfg_str(pc)) if pc is not None else None})
                    entries.append(e)
            autotune[str(c)] = entries
            changed_any[str(c)] = any(e["changed"] for e in entries if e["changed"] is not None)
        # 3. output difference between the two states at each chunk
        out_diff, nonfinite, tuned_during = {}, {}, []
        for c in ok:
            ys = {}
            for s in STATES:
                snap = pinned if s == "pinned" else retuned[c]
                ctl.restore(snap)
                before = ctl.signature()
                ys[s] = E2.make_fn(kind, inp, c)()
                torch.cuda.synchronize()
                tuned_during += [[s, c, n, k, "output check"] for n, k in sorted(ctl.signature() - before)]
            out_diff[str(c)], nonfinite[str(c)] = output_diff(ys["pinned"], ys["retuned"])
            del ys
        # 4. timing rounds
        rounds = []
        for r in range(int(args.rounds)):
            pairs = [(s, c) for s in STATES for c in ok]
            random.Random(f"e30|{args.subset_seed}|{p['key']}|r{r}").shuffle(pairs)
            stats = {s: {} for s in STATES}
            for s, c in pairs:
                st, new = time_pair(pinned if s == "pinned" else retuned[c], inp, c)
                stats[s][str(c)] = st
                tuned_during += [[s, c, n, k, f"round {r}"] for n, k in new]
            rounds.append({"round": r, "order": [[s, c] for s, c in pairs], "stats": stats})
            guard_check(f"{p['key']} (round {r})")
        med = {s: {str(c): statistics.median(rd["stats"][s][str(c)]["mean_ms"] for rd in rounds) for c in ok}
               for s in STATES}
        rsd = {s: {str(c): (statistics.stdev([rd["stats"][s][str(c)]["mean_ms"] for rd in rounds])
                            if len(rounds) > 1 else 0.0) for c in ok} for s in STATES}
        width = act.shape[2] * act.shape[3]
        row = {"kind": "call", "key": p["key"], "tag": tag, "label": args.label, "model": kind, "role": p["role"],
               "plan_index": p["index"], "group_id": p["gid"],
               "prompt_ids": ";".join(m["prompt_id"] for m in p["members"]), "fold": p["members"][0]["fold"],
               "layer": p["layer"], "batch": p["batch"], "L": Lmin, "L_bucket": length_bucket(Lmin), "width": width,
               "d_state": inp["k"].shape[-1] if kind == "gdn" else inp["B"].shape[-1],
               "chunks": ok, "skipped_chunks": skipped, **settings,
               "pinned_disk": (pinned_info or {}).get("pinned_disk"),
               "autotune": autotune, "changed_any": changed_any, "max_out_diff": out_diff, "nonfinite": nonfinite,
               "tuned_during_timing": tuned_during, "rounds_data": rounds, "med_ms": med, "rsd_ms": rsd,
               "oracle": {s: min(ok, key=lambda c: med[s][str(c)]) for s in STATES} if ok else {},
               "retune_s": round(retune_s, 2), "peak_vram_gb_call": round(peak_vram_gb(), 3)}
        del inp, act
        return row

    new_rows, t_start = 0, time.time()
    for i, p in enumerate(rows):
        if p["key"] in w.done:
            continue
        if args.max_rows is not None and new_rows >= args.max_rows:
            break
        retimed = 0
        while True:
            guard_wait()
            try:
                row = process(p)
            except _Foreign:
                retimed += 1
                torch.cuda.empty_cache()
                continue
            break
        row["retimed"] = retimed
        w.write(row)
        new_rows += 1
        torch.cuda.empty_cache()
        ch = sum(row["changed_any"].values())
        med = row["med_ms"]
        ratios = [med["retuned"][str(c)] / med["pinned"][str(c)] for c in row["chunks"]] or [math.nan]
        log(f"[{i + 1}/{len(rows)}] {p['key']} ({p['role']}) b={p['batch']} L={row['L']} "
            f"oracle pinned={row['oracle'].get('pinned')} retuned={row['oracle'].get('retuned')} "
            f"configs changed at {ch}/{len(row['chunks'])} chunks, retuned/pinned "
            f"{min(ratios):.3f}-{max(ratios):.3f}, re-tune {row['retune_s']:.0f} s, "
            f"tuned while timing {len(row['tuned_during_timing'])} ({(time.time() - t_start) / 60:.1f} min, "
            f"peak {row['peak_vram_gb_call']} GB)")
    w.close()
    log(f"E30 {kind} done: {new_rows} new rows -> {jsonl_path}")


# =========================================================================== analysis (CPU only)
def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    pos = q * (len(xs) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _mean(xs):
    return statistics.fmean(xs) if xs else None


def _share(flags):
    flags = [bool(f) for f in flags if f is not None]
    return sum(flags) / len(flags) if flags else None


def round_medians(row: dict, block_stat: str = "mean") -> dict:
    """Per state and chunk, the median over rounds of the per-round block statistic (``mean`` as in the sweep's
    ``lat_c<C>_med`` columns, or ``median`` for a check that is robust to single slow repeats)."""
    field = f"{block_stat}_ms"
    return {s: {int(c): statistics.median(float(rd["stats"][s][str(c)][field]) for rd in row["rounds_data"])
                for c in row["chunks"]} for s in STATES}


def predicted_excess_ms(entries: list[dict]) -> float | None:
    """Time the pinned configurations lose against the best configurations on this shape, summed over the call's
    changed keys, as the autotuner's own (cold-L2, per-kernel) benchmark of this shape measured it. A lower bound
    when a kernel runs more than once per call. None if a changed key lacks either timing."""
    excess = 0.0
    for e in entries:
        if e.get("changed") is False:
            continue
        if not (e.get("changed") and e.get("pinned_cfg_ms") is not None and e.get("best_ms") is not None):
            return None  # pinned key missing, or a changed key without both benchmark timings
        excess += max(0.0, e["pinned_cfg_ms"] - e["best_ms"])
    return excess


def call_metrics(row: dict, block_stat: str = "mean") -> dict:
    """Per-call oracle, regret and ratio metrics from the round medians."""
    chunks = [int(c) for c in row["chunks"]]
    med = round_medians(row, block_stat)
    P, R = med["pinned"], med["retuned"]
    op, orr = min(chunks, key=P.get), min(chunks, key=R.get)
    d = DEFAULT_CHUNK.get(row["model"])
    out = {"oracle_pinned": op, "oracle_retuned": orr, "agree": op == orr,
           "agree_within_tol": R[op] <= (1 + TOL) * R[orr],
           "regret_pinned_oracle_under_retuned": R[op] / R[orr] - 1,
           "gain_retuned_oracle_vs_pinned_oracle": 1 - R[orr] / P[op],
           "ratios": {c: R[c] / P[c] for c in chunks},
           "changed_any": {c: bool(row["changed_any"].get(str(c))) for c in chunks},
           "excess_share": {c: (None if (x := predicted_excess_ms(row["autotune"].get(str(c), []))) is None
                                else x / P[c]) for c in chunks}}
    # material: re-tuning is predicted to save more than TOL of the pinned call latency
    out["material"] = {c: None if v is None else v > TOL for c, v in out["excess_share"].items()}
    if d in chunks:
        out["default_regret_pinned"] = P[d] / P[op] - 1
        out["default_regret_retuned"] = R[d] / R[orr] - 1
    return out


def summarize_rows(rows: list[dict], block_stat: str = "mean") -> dict:
    if not rows:
        return {"calls": 0}
    ms = [call_metrics(r, block_stat) for r in rows]
    ratios = [v for m in ms for v in m["ratios"].values()]
    changed = [v for m in ms for v in m["changed_any"].values()]
    material = [v for m in ms for v in m["material"].values()]
    ratio_changed = [m["ratios"][c] for m in ms for c in m["ratios"] if m["changed_any"][c]]
    ratio_material = [m["ratios"][c] for m in ms for c in m["ratios"] if m["material"][c]]
    excess = [v for m in ms for v in m["excess_share"].values() if v is not None]
    per_tuner: dict[str, dict] = {}
    for r in rows:
        for c, entries in r["autotune"].items():
            for e in entries:
                t = per_tuner.setdefault(e["tuner"], {"entries": 0, "changed": 0, "missing_pinned": 0, "gap": []})
                t["entries"] += 1
                if e["changed"] is None:
                    t["missing_pinned"] += 1
                elif e["changed"]:
                    t["changed"] += 1
                if e.get("pinned_cfg_ms") and e.get("best_ms"):
                    t["gap"].append(e["pinned_cfg_ms"] / e["best_ms"] - 1)
    tuners = {}
    for n, t in sorted(per_tuner.items()):
        known = t["entries"] - t["missing_pinned"]
        tuners[n] = {"entries": t["entries"], "share_changed": t["changed"] / known if known else None,
                     "missing_pinned": t["missing_pinned"],
                     "isolated_gap_pinned_cfg_median": quantile(t["gap"], 0.5),
                     "isolated_gap_pinned_cfg_p95": quantile(t["gap"], 0.95),
                     "isolated_gap_pinned_cfg_max": max(t["gap"]) if t["gap"] else None}
    regret = [m["regret_pinned_oracle_under_retuned"] for m in ms]
    out = {
        "calls": len(rows), "pairs": len(ratios), "block_stat": block_stat,
        "share_pairs_config_changed": _share(changed),
        "share_pairs_material_change": _share(material),
        "predicted_excess_share_of_call": {"median": quantile(excess, 0.5), "p95": quantile(excess, 0.95),
                                           "max": max(excess) if excess else None, "pairs": len(excess)},
        "per_autotuner": tuners,
        "ratio_retuned_over_pinned": {"median": quantile(ratios, 0.5), "p5": quantile(ratios, 0.05),
                                      "p95": quantile(ratios, 0.95), "min": min(ratios), "max": max(ratios),
                                      "share_retuned_faster_2pct": _share([x < 1 - TOL for x in ratios]),
                                      "share_retuned_slower_2pct": _share([x > 1 + TOL for x in ratios]),
                                      "median_where_config_changed": quantile(ratio_changed, 0.5),
                                      "median_where_material_change": quantile(ratio_material, 0.5)},
        "oracle_agreement": _share([m["agree"] for m in ms]),
        "oracle_agreement_within_2pct": _share([m["agree_within_tol"] for m in ms]),
        "regret_pinned_oracle_under_retuned_pct": {"mean": 100 * _mean(regret), "max": 100 * max(regret)},
        "gain_retuned_oracle_vs_pinned_oracle_pct": {
            "mean": 100 * _mean([m["gain_retuned_oracle_vs_pinned_oracle"] for m in ms]),
            "min": 100 * min(m["gain_retuned_oracle_vs_pinned_oracle"] for m in ms),
            "max": 100 * max(m["gain_retuned_oracle_vs_pinned_oracle"] for m in ms)},
        "oracle_pinned_distribution": _counts(m["oracle_pinned"] for m in ms),
        "oracle_retuned_distribution": _counts(m["oracle_retuned"] for m in ms),
        "max_out_diff": max(float(v) for r in rows for v in r["max_out_diff"].values()),
        "nonfinite_pairs": sum(bool(v) for r in rows for v in r["nonfinite"].values()),
        "keys_tuned_while_timing": sum(len(r.get("tuned_during_timing", [])) for r in rows),
        "retune_s_mean": _mean([float(r["retune_s"]) for r in rows]),
        "peak_vram_gb": max(float(r.get("peak_vram_gb_call", 0.0)) for r in rows),
    }
    dp = [m["default_regret_pinned"] for m in ms if "default_regret_pinned" in m]
    dr = [m["default_regret_retuned"] for m in ms if "default_regret_retuned" in m]
    if dp:
        out["default_chunk_regret_pct"] = {"pinned_mean": 100 * _mean(dp), "retuned_mean": 100 * _mean(dr),
                                           "pinned_max": 100 * max(dp), "retuned_max": 100 * max(dr),
                                           "calls": len(dp)}
    return out


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _counts(xs) -> dict:
    out: dict[int, int] = {}
    for x in xs:
        out[int(x)] = out.get(int(x), 0) + 1
    return dict(sorted(out.items()))


def analyze(args) -> None:
    out_dir = Path(args.out_dir)
    paths: list[Path] = []
    if args.inputs:
        for pat in args.inputs:
            paths += [Path(p) for p in sorted(glob.glob(pat))]
    else:
        tags = args.tags.split(",") if args.tags else ([args.tag] if args.tag else [])
        paths = [out_dir / f"e30_autotune_state_{t}.jsonl" for t in tags]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit("no E30 JSONL found (give --tag, --tags or --inputs)")
    models, all_rows = {}, []
    for path in paths:
        rows = read_jsonl(path)
        calls = list({r["key"]: r for r in rows if r.get("kind") == "call"}.values())  # last row per key
        docs = [r for r in rows if r.get("kind") == "autotune_doc"]
        metas = read_jsonl(path, kind="meta")
        if not calls:
            continue
        tag = calls[0].get("tag") or path.stem.replace("e30_autotune_state_", "")
        calls = [r for r in calls if r.get("chunks")]  # a call with no feasible chunk carries no comparison
        sample = [r for r in calls if r["role"] in ("sample", "sample+anchor")]
        anchor = [r for r in calls if r["role"] in ("anchor", "sample+anchor")]
        block = {"label": calls[0].get("label"), "model": calls[0]["model"], "source": str(path),
                 "summary": summarize_rows(sample, args.block_stat),
                 "by_batch": {int(b): summarize_rows([r for r in sample if int(r["batch"]) == b], args.block_stat)
                              for b in sorted({int(r["batch"]) for r in sample})},
                 "anchor": summarize_rows(anchor, args.block_stat) if anchor else None,
                 "documentation": docs[-1]["documentation"] if docs else None,
                 "pinned_disk": docs[-1].get("pinned_disk") if docs else None,
                 "disk_cache": _disk_summary(docs[-1].get("disk")) if docs else None,
                 "timing": {k: calls[0].get(k) for k in ("cuda_graph", "warmup", "repeats", "pre_spin_ms", "rounds")},
                 "env": {k: metas[-1].get(k) for k in ("gpu", "pkg_triton", "pkg_mamba_ssm", "pkg_fla-core",
                                                       "torch", "git_commit")} if metas else None}
        models[tag] = block
        all_rows += sample
    out_tag = args.tag or (list(models)[0] if len(models) == 1 else "combined")
    result = {"experiment": "E30", "inputs": [str(p) for p in paths], "models": models,
              "overall": summarize_rows(all_rows, args.block_stat), "tolerance": TOL}
    suffix = "" if args.block_stat == "mean" else "_blockmedian"
    out = out_dir / f"e30_autotune_state_{out_tag}{suffix}.json"
    write_json(out, result)
    print(f"E30 analysis -> {out}")
    for tag, b in list(models.items()) + [("overall", {"label": "all models", "summary": result["overall"]})]:
        s = b["summary"]
        if not s.get("calls"):
            continue
        rr, dr = s["ratio_retuned_over_pinned"], s.get("default_chunk_regret_pct", {})
        print(f"  {tag} ({b.get('label')}): {s['calls']} calls, {s['pairs']} (call, chunk) pairs; "
              f"configs changed in {100 * s['share_pairs_config_changed']:.0f}% of pairs "
              f"(material, predicted re-tuning saving >2% of the call: "
              f"{_pct(s['share_pairs_material_change'])}); "
              f"retuned/pinned median {rr['median']:.3f} [p5 {rr['p5']:.3f}, p95 {rr['p95']:.3f}], "
              f"faster>2% {100 * rr['share_retuned_faster_2pct']:.0f}%, slower>2% "
              f"{100 * rr['share_retuned_slower_2pct']:.0f}%")
        print(f"      oracle agreement {100 * s['oracle_agreement']:.0f}% (within 2%: "
              f"{100 * s['oracle_agreement_within_2pct']:.0f}%), pinned-oracle regret under retuned "
              f"{s['regret_pinned_oracle_under_retuned_pct']['mean']:.2f}% mean / "
              f"{s['regret_pinned_oracle_under_retuned_pct']['max']:.2f}% max, re-tuning gain "
              f"{s['gain_retuned_oracle_vs_pinned_oracle_pct']['mean']:.2f}% mean"
              + (f", default-chunk regret {dr['pinned_mean']:.1f}% pinned / {dr['retuned_mean']:.1f}% retuned"
                 if dr else "")
              + f", keys tuned while timing {s['keys_tuned_while_timing']}")
        if b.get("anchor") and b["anchor"].get("calls"):
            a = b["anchor"]
            print(f"      anchor call (noise control): configs changed in "
                  f"{100 * a['share_pairs_config_changed']:.0f}% of chunks, ratio median "
                  f"{a['ratio_retuned_over_pinned']['median']:.3f}")


def _disk_summary(disk: dict | None) -> dict | None:
    if not disk:
        return None
    vals = [v for d in disk.values() for v in d.values()]
    return {"keys": len(vals), "hits": sum(v["status"] == "hit" for v in vals),
            "hits_equal_to_fresh_walk": sum(bool(v.get("equals_walk")) for v in vals),
            "per_autotuner": {n: {"keys": len(d), "hits": sum(v["status"] == "hit" for v in d.values()),
                                  "equal": sum(bool(v.get("equals_walk")) for v in d.values())}
                              for n, d in sorted(disk.items())}}


# =========================================================================== main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("mamba2", "gdn"), help="kernel family (required unless --analyze)")
    p.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR,
                   help="trace root holding <model>/manifest.jsonl (as for e2_oracle_sweep)")
    p.add_argument("--label", default=None, help="model label stored in the rows (e.g. 'SSD 2.7B')")
    p.add_argument("--n-calls", type=int, default=40, help="stratified calls per model (plus the anchor call)")
    p.add_argument("--batches", default="1,4,16", help="batches of the sweep plan (must match the sweep)")
    p.add_argument("--layers", default=None, help="subset of captured layers (default: all, as the sweep)")
    p.add_argument("--chunks", default=None, help="chunk subset (default: the sweep's chunks)")
    p.add_argument("--seed", type=int, default=0, help="plan seed of the sweep (keep 0 so keys join)")
    p.add_argument("--subset-seed", type=int, default=0, help="seed of the stratified draw and timing order")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--no-graph", action="store_true", help="time eager calls instead of CUDA-graph replays")
    p.add_argument("--pre-spin-ms", type=float, default=0.0, help="GPU-side spin before each timed block (0 = off)")
    p.add_argument("--pinned-disk", choices=("auto", "on", "off"), default="auto",
                   help="take pinned configurations from Triton's on-disk autotune cache where present "
                        "(auto: when the library enables that cache, i.e. FLA)")
    p.add_argument("--walk-max-shapes", type=int, default=None, help="truncate the pinned walk (smoke only)")
    p.add_argument("--exclusive-guard", action="store_true",
                   help="pause while other processes use the GPU and redo work they overlapped")
    p.add_argument("--guard-until", type=float, default=None, help="epoch seconds after which the guard gives up")
    p.add_argument("--max-rows", type=int, default=None, help="stop after this many new rows (smoke)")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--tag", default=None, help="output tag (default <host>_<model>); with --analyze the output tag")
    p.add_argument("--dry-run", action="store_true", help="print the plan and the stratified subset (no GPU)")
    p.add_argument("--analyze", action="store_true", help="summarize E30 JSONL files (CPU only)")
    p.add_argument("--tags", default=None, help="--analyze: comma-separated run tags, one per model")
    p.add_argument("--inputs", nargs="*", default=None, help="--analyze: JSONL paths or globs")
    p.add_argument("--block-stat", choices=("mean", "median"), default="mean",
                   help="--analyze: per-round block statistic (mean as the sweep; median as a robustness check)")
    args = p.parse_args(argv)
    if not args.analyze and not args.model:
        p.error("--model is required unless --analyze")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.analyze:
        analyze(args)
    elif args.dry_run:
        dry_run(args)
    else:
        import torch

        with torch.no_grad():
            run(args)


if __name__ == "__main__":
    main()
