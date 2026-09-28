"""E10 - single-session interleaved routed end-to-end rerun (reviewer point 4).

The tpds-r1 routed study (Table IV) ran every configuration in its own child process, so the
passive path it timed drifted from 887 to 952 ms between sessions, more than most differences
between configurations, and only one prompt length and end-to-end latency were measured. This
experiment repeats that study with the drift removed by construction:

  * one process and one session: the Mamba-1 370M checkpoint is loaded once and the scheduler
    patch of the routed study is installed once; every configuration only changes the global
    routing state between runs;
  * blocks: a round runs one block per prompt length (length order shuffled per round) and each
    block runs every configuration once, in an order shuffled per (round, length) with a fixed
    seed; differences between configurations are formed within a block (paired over rounds);
  * prefill is timed on its own: CUDA events around the top-level forward call that consumes
    the prompt, separately from the decode steps.

Configurations (names and semantics of the routed ablation matrix, see
``run_scheduler_ablation_matrix`` and the patched forward of ``run_integrated_end_to_end``):

  passive                   the unmodified model (scheduler hook disabled, stock scan)
  static_chunk_{C}          routed, fixed chunk C in {128, 256, 512, 1024, 2048}, no statistic
  no_entropy_mid            routed, midpoint of the chunk rule without a statistic
  random_seed_{S}           routed, uniform random chunk per call (seeded per run, see below)
  hist                      routed, full 256-bin histogram entropy, Eq. (2)
  sampled_hist_s8           routed, histogram entropy on every 8th element
  token_hist_s8             routed, per-token histogram entropy averaged (stride 8)
  cheap_proxy               routed, moment proxy (mean |x| / RMS)
  variance_proxy            routed, variance proxy
  kurtosis_proxy            routed, kurtosis proxy
  guarded_sampled_hist_s8   routed, sampled histogram with the guard (fallback 512, 2 buckets)
  learned_table             routed, rule table on (layer, length, entropy); it computes the
                            sampled histogram as the routed study did
  hook_constant             hook enabled but not routed, no statistic (patched forward only)
  hook_<variant>            the "active only" condition of the routed study for every
                            statistic-bearing variant: the statistic is computed and the chunk
                            selected, but the scan runs the stock path. Paired against passive
                            this is the decision cost; against hook_constant it is the cost of
                            the statistic alone.

Optional extra configurations that the patched forward supports but the tpds-r1 matrix did
not run by default: guarded_variance_proxy, guarded_hist, guarded_kurtosis_proxy
(``--configs``). The adaptive range is [min, max] of the static chunks, i.e. [128, 2048], the
routed range of the manuscript. The learned table uses the policy of the tpds-r1 H800 run
(default 512, sequence length <= 50 -> 128; embedded below, ``--learned-policy-json`` overrides).
The random policy draws per scan call from a generator re-seeded per run with
crc32("<seed>|L<len>|r<round>"), so a restart reproduces the same draws.

Runtime-chunk scan: the routed configurations need the reference Mamba-1 CUDA scan with the chunk
as a runtime argument. ``src/corey_selective_scan_dispatch.py`` looks for
``selective_scan_cuda.fwd_with_chunk_size``; the rebuilt kernel (``src/kernels/mamba1_runtime_chunk``)
lives in the separate module ``selective_scan_chunk_cuda``. After mamba_ssm has bound the stock
extension, this script registers a stand-in ``selective_scan_cuda`` module in ``sys.modules``
that forwards every attribute to the stock extension and adds ``fwd_with_chunk_size`` from
``selective_scan_chunk_cuda``; mamba_ssm keeps its own reference to the stock module, and only the
dispatch module, which imports ``selective_scan_cuda`` on each call, sees the added entry point.
Every chunk is checked against the stock scan once before timing. Without the runtime-chunk
module the routed configurations are skipped with a warning; passive and the hook controls
still run.

Prompts: the routed study's prompt (``DEFAULT_PROMPT``, 122 tokens) repeated with single spaces
as in ``--prompt-repeat``, tokenized once, and the first L token ids taken for L in
{256, 976, 4096}; 976 tokens are exactly the 8-repeat prompt of Table IV. ``--prompt-file`` tiles
another text the same way. Greedy decoding of ``--new-tokens`` (32) tokens with the KV/SSM cache.

Per run: prefill_ms (CUDA events around the prefill forward), decode_ms (end of prefill to end
of generate: all decode steps plus sampling), pre_ms (generate set-up before the prefill),
total_ms (CUDA events around generate), decode_forward_ms (sum of the decode forwards), wall_ms,
the chunk each of the 48 prefill scan calls selected and the chunk the scan ran with (stock rule
for passive and hook controls), guard fallbacks, entropy summary, and the greedy token ids with
their hash (output equality against passive in the same block).

Resumable: one fsynced JSONL row per run with key ``e10|L<len>|r<round>|<config>``. A block is
complete when one attempt of it holds every configuration; complete blocks are skipped on
restart, and an incomplete block is re-run in full so that every block used in the analysis comes
from one session (at most one block is repeated per restart; the rows of the abandoned attempt
stay in the file and are ignored by the analysis). Warmup rounds (``--warmup-rounds``) run at
the start of every session and are recorded with kind "warmup" only. With ``--exclusive-guard``
every run waits until no other process holds the GPU; if another process appeared during a run
or the guard paused, the block is re-warmed and restarted.

``--analyze`` writes per length and configuration: mean prefill / decode / total with 95% CI;
paired differences (ms and %) to the best static chunk (lowest mean of the metric), to the
per-round fastest static chunk, and to passive, with t-distribution CIs over rounds; the
decision cost of every statistic-bearing variant (hook - passive, hook - hook_constant, and
routed - static at the same chunk when the variant routed one chunk throughout) in ms and in
percent of prefill; output equality versus passive; chunk-count distributions.

``--dry-run`` prints the schedule without importing torch or loading the model. ``--smoke``
runs a tiny version (1 warmup and 2 timed rounds, L=256, 4 new tokens, results in the smoke
directory) and allows the slow HF path when mamba_ssm is missing (passive only).

Output: ``e10_routed_<tag>.jsonl`` (resumable) and ``e10_routed_<tag>.json`` (analysis).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import time
import types
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.experiments.run_integrated_end_to_end import DEFAULT_PROMPT, MODEL_REGISTRY
from src.experiments.tpds.common import (LOG_DIR, RESULTS_DIR, SMOKE_DIR, JsonlWriter, Logger, all_gpu_pids,
                                         check_disk, env_metadata, foreign_gpu_pids, gpu_state, host_name,
                                         read_jsonl, register_own_gpu_pids, t_crit, utc_now,
                                         wait_for_exclusive_gpu, write_json)

STATIC_CHUNKS = (128, 256, 512, 1024, 2048)
LENGTHS = (256, 976, 4096)
DEFAULT_CONFIGS = ("passive,static,no_entropy,random,hist,sampled_hist,token_hist,cheap_proxy,variance_proxy,"
                   "kurtosis_proxy,guarded_sampled_hist,learned_table")
VALID_VARIANTS = ("passive", "static", "no_entropy", "random", "hist", "sampled_hist", "token_hist", "cheap_proxy",
                  "variance_proxy", "kurtosis_proxy", "guarded_sampled_hist", "guarded_variance_proxy",
                  "guarded_hist", "guarded_kurtosis_proxy", "learned_table")
STATISTIC_MODES = {"hist", "sampled_hist", "token_hist", "cheap_proxy", "variance_proxy", "kurtosis_proxy",
                   "guarded_sampled_hist", "guarded_variance_proxy", "guarded_hist", "guarded_kurtosis_proxy",
                   "learned_table"}
METRICS = ("prefill_ms", "decode_ms", "total_ms")
EXTRA_METRICS = ("pre_ms", "decode_forward_ms", "wall_ms")
FAST_PATH_NAMES = ("selective_scan_fn", "selective_state_update", "causal_conv1d_fn", "causal_conv1d_update",
                   "mamba_inner_fn")

# Policy of the tpds-r1 learned_table row (H800 unified run), derived from a static sweep of the
# mixed-regime corpus: short sequences (<= 50 tokens) -> 128, everything else -> 512.
DEFAULT_LEARNED_POLICY: dict[str, Any] = {
    "description": "tpds-r1 learned_table policy: seq_len <= 50 -> chunk 128, otherwise the global static "
                   "oracle 512 (derived from the mixed-regime static sweep on the RTX 3090).",
    "default_chunk": 512,
    "rules": [{"name": "short_seq_to_128", "chunk": 128, "max_seq_len": 50}],
}

LABELS = {"passive": "Passive (stock fast path)", "no_entropy_mid": "Midpoint, no statistic", "hist": "Full histogram",
          "cheap_proxy": "Moment proxy", "variance_proxy": "Variance proxy", "kurtosis_proxy": "Kurtosis proxy",
          "guarded_variance_proxy": "Guarded variance proxy", "guarded_hist": "Guarded full histogram",
          "guarded_kurtosis_proxy": "Guarded kurtosis proxy", "learned_table": "Learned table",
          "hook_constant": "Hook only, no statistic (not routed)"}


# --------------------------------------------------------------------------- configurations
@dataclass(frozen=True)
class Config:
    name: str
    kind: str                  # passive | static | no_entropy | random | adaptive | proxy | guarded | learned | hook
    label: str
    enabled: bool              # scheduler hook active (patched forward)
    route: bool                # selected chunk passed to the runtime-chunk scan
    scheduler_mode: str
    statistic: bool            # computes an activation statistic
    force_chunk: int | None = None
    chunk_min: int = 128
    chunk_max: int = 2048
    entropy_stride: int = 1
    base: str | None = None    # routed configuration that a hook control mirrors


def _label(name: str) -> str:
    if name in LABELS:
        return LABELS[name]
    if name.startswith("static_chunk_"):
        return f"Static {name[len('static_chunk_'):]}"
    if name.startswith("random_seed_"):
        return "Random"
    for prefix, text in (("guarded_sampled_hist", "Guarded sampled histogram"), ("sampled_hist", "Sampled histogram"),
                         ("token_hist", "Token-level histogram")):
        if name.startswith(prefix):
            return text
    if name.startswith("hook_"):
        return f"{_label(name[len('hook_'):])}, hook only (not routed)"
    return name


def _parse_ints(raw: str) -> list[int]:
    return [int(x) for x in str(raw).split(",") if x.strip()]


def _parse_variants(raw: str) -> list[str]:
    out = []
    for part in raw.split(","):
        v = part.strip()
        if not v:
            continue
        if v not in VALID_VARIANTS:
            raise SystemExit(f"unknown configuration {v!r}; expected a subset of {', '.join(VALID_VARIANTS)}")
        out.append(v)
    return list(dict.fromkeys(out))


def build_configs(args) -> list[Config]:
    """Configurations in the naming and parameterization of the routed ablation matrix."""
    chunks = sorted(set(args.static_chunks))
    lo = args.adaptive_chunk_min if args.adaptive_chunk_min is not None else min(chunks)
    hi = args.adaptive_chunk_max if args.adaptive_chunk_max is not None else max(chunks)
    s = max(args.entropy_stride, 1)
    cfgs: list[Config] = []

    def routed(name, kind, mode, stride=1):
        cfgs.append(Config(name, kind, _label(name), True, True, mode, mode in STATISTIC_MODES, None, lo, hi, stride))

    for v in _parse_variants(args.configs):
        if v == "passive":
            cfgs.append(Config("passive", "passive", _label("passive"), False, False, "passive", False, None, lo, hi))
        elif v == "static":
            for c in chunks:
                name = f"static_chunk_{c}"
                cfgs.append(Config(name, "static", _label(name), True, True, "constant", False, c, c, c))
        elif v == "no_entropy":
            routed("no_entropy_mid", "no_entropy", "no_entropy")
        elif v == "random":
            routed(f"random_seed_{args.random_seed}", "random", "random")
        elif v in ("sampled_hist", "token_hist"):
            routed(f"{v}_s{s}", "adaptive", v, s)
        elif v == "guarded_sampled_hist":
            routed(f"guarded_sampled_hist_s{s}", "guarded", v, s)
        elif v in ("guarded_variance_proxy", "guarded_hist", "guarded_kurtosis_proxy"):
            routed(v, "guarded", v)
        elif v == "learned_table":
            routed("learned_table", "learned", v, s)
        else:  # hist, cheap_proxy, variance_proxy, kurtosis_proxy
            routed(v, "proxy" if v == "cheap_proxy" else "adaptive", v)
    if args.hook_controls:
        stat = [c for c in cfgs if c.statistic and c.route]
        if stat:
            cfgs.append(Config("hook_constant", "hook", _label("hook_constant"), True, False, "constant", False,
                               None, lo, hi, 1, None))
        for c in stat:
            name = f"hook_{c.name}"
            cfgs.append(Config(name, "hook", _label(name), True, False, c.scheduler_mode, True, c.force_chunk,
                               c.chunk_min, c.chunk_max, c.entropy_stride, c.name))
    return cfgs


def stock_chunk(L: int) -> int:
    """Chunk the stock mamba_ssm dispatch uses for a sequence of length L."""
    for c in (128, 256, 512, 1024):
        if L <= c:
            return c
    return 2048


def run_key(L: int, rnd: int, name: str) -> str:
    return f"e10|L{L}|r{rnd}|{name}"


def length_order(seed: int, rnd: int, lengths: list[int]) -> list[int]:
    order = list(lengths)
    random.Random(f"e10|{seed}|r{rnd}|lengths").shuffle(order)
    return order


def config_order(seed: int, rnd: int, L: int, cfgs: list[Config]) -> list[Config]:
    order = list(cfgs)
    random.Random(f"e10|{seed}|r{rnd}|L{L}|configs").shuffle(order)
    return order


def rng_seed(random_seed: int, L: int, rnd: int) -> int:
    return zlib.crc32(f"{random_seed}|L{L}|r{rnd}".encode())


def complete_blocks(rows: list[dict], names: set[str]) -> set[tuple[int, int]]:
    """(L, round) blocks for which one attempt holds every configuration in ``names``."""
    have: dict[str, set[str]] = defaultdict(set)
    where: dict[str, tuple[int, int]] = {}
    for r in rows:
        if r.get("kind") == "run" and r.get("block_id"):
            have[r["block_id"]].add(r["config"])
            where[r["block_id"]] = (int(r["L"]), int(r["round"]))
    return {where[b] for b, got in have.items() if names <= got}


# --------------------------------------------------------------------------- runtime-chunk adapter
class _ChunkAdapterModule(types.ModuleType):
    """Stand-in for ``selective_scan_cuda``: the stock extension plus ``fwd_with_chunk_size``.

    Attributes not defined here fall through to the stock extension.
    """

    def __init__(self, stock: Any, chunk_mod: Any):
        super().__init__("selective_scan_cuda",
                         f"E10 adapter: stock selective_scan_cuda + {chunk_mod.__name__}.fwd_with_chunk_size")
        self._e10_stock = stock
        self._e10_chunk_module = chunk_mod
        self.fwd_with_chunk_size = chunk_mod.fwd_with_chunk_size
        self.__file__ = getattr(stock, "__file__", None)

    def __getattr__(self, name: str) -> Any:
        stock = self.__dict__.get("_e10_stock")
        if stock is None:
            raise AttributeError(name)
        return getattr(stock, name)


def install_chunk_adapter(chunk_module: str, log) -> dict[str, Any]:
    """Make ``selective_scan_cuda.fwd_with_chunk_size`` resolvable for the dispatch module."""
    info: dict[str, Any] = {"chunk_module": chunk_module, "available": False}
    try:
        import torch  # noqa: F401  (the extensions link against torch)
        import mamba_ssm.ops.selective_scan_interface  # noqa: F401  (binds the real stock extension first)
        import selective_scan_cuda as stock  # type: ignore
    except Exception as exc:
        info["reason"] = f"mamba_ssm / selective_scan_cuda unavailable: {exc!r}"
        log(f"WARNING: {info['reason']}")
        return info
    if isinstance(stock, _ChunkAdapterModule) or hasattr(stock, "fwd_with_chunk_size"):
        info.update(available=True, source="selective_scan_cuda.fwd_with_chunk_size (patched stock build)")
        log(f"runtime-chunk scan: {info['source']}")
        return info
    try:
        mod = importlib.import_module(chunk_module)
    except Exception as exc:
        info["reason"] = f"{chunk_module} not importable: {exc!r}"
        log(f"WARNING: {info['reason']}; the routed configurations are skipped")
        return info
    if not hasattr(mod, "fwd_with_chunk_size"):
        info["reason"] = f"{chunk_module} has no fwd_with_chunk_size"
        log(f"WARNING: {info['reason']}; the routed configurations are skipped")
        return info
    sys.modules["selective_scan_cuda"] = _ChunkAdapterModule(stock, mod)
    info.update(available=True, source=f"{chunk_module}.fwd_with_chunk_size via a selective_scan_cuda stand-in",
                chunk_module_file=getattr(mod, "__file__", None), stock_file=getattr(stock, "__file__", None))
    log(f"runtime-chunk scan: {info['source']} ({info['chunk_module_file']})")
    return info


def validate_routed_scan(dispatch: Any, chunks: list[int], log, L: int = 1000, dim: int = 64,
                         dstate: int = 16) -> dict[str, Any]:
    """Compare the dispatch module at every chunk with the stock scan (fp16 inputs, L not a multiple of any chunk)."""
    import torch
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as stock_fn

    g = torch.Generator(device="cuda")
    g.manual_seed(1234)

    def rnd(*shape, scale=1.0, half=True):
        t = scale * torch.randn(*shape, device="cuda", generator=g)
        return t.half() if half else t

    with torch.no_grad():
        u, delta, z = rnd(1, dim, L), rnd(1, dim, L, scale=0.5), rnd(1, dim, L)
        B, C = rnd(1, dstate, L), rnd(1, dstate, L)
        A = -torch.exp(rnd(dim, dstate, scale=0.5, half=False))
        D, bias = rnd(dim, half=False), rnd(dim, scale=0.1, half=False)
        y0, s0 = stock_fn(u, delta, A, B, C, D, z, bias, delta_softplus=True, return_last_state=True)
        out: dict[str, Any] = {"L": L, "dim": dim, "chunks": {}}
        ok = True
        for c in chunks:
            y, s = dispatch.selective_scan_fn(u, delta, A, B, C, D, z, bias, delta_softplus=True,
                                              return_last_state=True, chunk_size=c)
            dy = float((y.float() - y0.float()).abs().max())
            ds = float((s.float() - s0.float()).abs().max())
            rel_y = dy / max(float(y0.float().abs().max()), 1e-12)
            rel_s = ds / max(float(s0.float().abs().max()), 1e-12)
            out["chunks"][str(c)] = {"max_abs_dy": dy, "rel_dy": rel_y, "max_abs_dstate": ds, "rel_dstate": rel_s}
            ok &= rel_y < 1e-2 and rel_s < 1e-2
    out["passed"] = bool(ok)
    log("routed-scan check vs stock: " + ", ".join(f"c{c} rel|dy|={v['rel_dy']:.1e} rel|dh|={v['rel_dstate']:.1e}"
                                                    for c, v in out["chunks"].items())
        + f" -> {'PASS' if ok else 'FAIL'}")
    return out


# --------------------------------------------------------------------------- prompts
def build_prompt_ids(tokenizer: Any, text: str, L: int) -> tuple[list[int], int]:
    """First L token ids of ``text`` repeated k times (single spaces), k minimal."""
    reps = 1
    while True:
        ids = tokenizer(" ".join([text] * reps))["input_ids"]
        if len(ids) >= L:
            return [int(i) for i in ids[:L]], reps
        if reps > 100000:
            raise SystemExit("prompt text yields no tokens")
        reps += 1


def _sha1(values: list[int]) -> str:
    return hashlib.sha1(",".join(map(str, values)).encode()).hexdigest()


# --------------------------------------------------------------------------- timing
class ForwardTimer:
    """CUDA events around every top-level forward call of the model while ``active``."""

    def __init__(self, model: Any):
        self.active = False
        self.calls: list[tuple[Any, Any, int]] = []
        self._open: tuple[Any, int] | None = None
        self.handles = [model.register_forward_pre_hook(self._pre, with_kwargs=True),
                        model.register_forward_hook(self._post, with_kwargs=True)]

    def _pre(self, module, args, kwargs):
        if self.active:
            import torch

            ids = kwargs.get("input_ids") if kwargs else None
            if ids is None and args:
                ids = args[0]
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._open = (ev, int(ids.shape[-1]) if torch.is_tensor(ids) else -1)
        return None

    def _post(self, module, args, kwargs, output):
        if self.active and self._open is not None:
            import torch

            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self.calls.append((self._open[0], ev, self._open[1]))
            self._open = None
        return None


def apply_config(cfg: Config, args, policy: dict[str, Any], seed: int) -> None:
    from src.experiments.run_integrated_end_to_end import STATE

    STATE.enabled = cfg.enabled
    STATE.route_chunk = cfg.route
    STATE.scheduler_mode = cfg.scheduler_mode if cfg.enabled else "hist"
    STATE.force_chunk = cfg.force_chunk
    STATE.chunk_min = cfg.chunk_min
    STATE.chunk_max = cfg.chunk_max
    STATE.entropy_stride = max(cfg.entropy_stride, 1)
    STATE.num_bins = args.num_bins
    STATE.h_ref = None
    STATE.guard_fallback_chunk = args.guard_fallback_chunk
    STATE.guard_min_delta_buckets = max(args.guard_min_delta_buckets, 0)
    STATE.learned_policy = policy if cfg.scheduler_mode == "learned_table" else None
    STATE.random_seed = seed
    STATE.reset_random()
    STATE.records.clear()
    STATE.chunk_size_kwarg_supported = None


def run_once(model: Any, inputs: dict[str, Any], cfg: Config, L: int, rnd: int, args, ctx: dict[str, Any]) -> dict:
    """One greedy generation under ``cfg``; returns the timing and routing record."""
    import torch
    from src.experiments.run_integrated_end_to_end import STATE

    seed = rng_seed(args.random_seed, L, rnd)
    apply_config(cfg, args, ctx["policy"], seed)
    timer: ForwardTimer = ctx["timer"]
    g0, g1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    gc.collect()
    torch.cuda.synchronize()
    timer.calls.clear()
    timer._open = None
    timer.active = True
    gc.disable()
    try:
        t0 = time.perf_counter()
        g0.record()
        with torch.no_grad():
            out = model.generate(**inputs, **ctx["gen_kwargs"])
        g1.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        gc.enable()
        timer.active = False
        STATE.enabled = False
    calls = list(timer.calls)
    if not calls:
        raise RuntimeError("no forward call was observed during generate(); the timing hooks did not fire")
    p0, p1, prefill_len = calls[0]
    if prefill_len != L:
        raise RuntimeError(f"the first forward call consumed {prefill_len} tokens, not the {L}-token prompt "
                           "(chunked prefill?); the prefill timing would be wrong")
    prefill_ms = p0.elapsed_time(p1)
    pre_ms = g0.elapsed_time(p0)
    decode_ms = p1.elapsed_time(g1)
    total_ms = g0.elapsed_time(g1)
    decode_forward_ms = sum(s.elapsed_time(e) for s, e, _ in calls[1:])
    new_ids = [int(x) for x in out[0, L:].tolist()]

    recs = list(STATE.records)
    selected = Counter(int(r["chunk"]) for r in recs)
    raw = Counter(int(r["raw_chunk"]) for r in recs if r.get("raw_chunk") is not None)
    ent = [float(r["entropy_nats"]) for r in recs if r.get("entropy_nats") is not None]
    honored = STATE.chunk_size_kwarg_supported
    if cfg.route:
        if honored is not True or not recs or not all(r.get("routed") for r in recs):
            raise RuntimeError(f"{cfg.name}: the selected chunk was not routed into the runtime-chunk scan "
                               f"(chunk_size_honored={honored}, {len(recs)} scheduler records)")
        scan_counts = dict(selected)
        scan_source = "routed"
    else:
        scan_counts = {stock_chunk(L): ctx["n_layers"]} if ctx["fast_path"] else {}
        scan_source = "stock_rule" if ctx["fast_path"] else "hf_slow_path"
    return {
        "prefill_ms": prefill_ms, "decode_ms": decode_ms, "pre_ms": pre_ms, "total_ms": total_ms,
        "decode_forward_ms": decode_forward_ms, "wall_ms": wall_ms, "n_forward_calls": len(calls),
        "prefill_seq_len": prefill_len, "n_new_tokens": len(new_ids), "token_hash": _sha1(new_ids)[:16],
        "new_token_ids": new_ids, "n_sched_records": len(recs),
        "selected_chunk_counts": {str(k): v for k, v in sorted(selected.items())},
        "scan_chunk_counts": {str(k): v for k, v in sorted(scan_counts.items())}, "scan_chunk_source": scan_source,
        "raw_chunk_counts": {str(k): v for k, v in sorted(raw.items())} or None,
        "guarded_fallbacks": sum(1 for r in recs if r.get("guarded_fallback")),
        "policy_rules": dict(Counter(r["policy_rule"] for r in recs if r.get("policy_rule"))) or None,
        "entropy_nats": ({"min": min(ent), "mean": statistics.fmean(ent), "max": max(ent)} if ent else None),
        "chunk_size_honored": honored, "rng_seed": seed if cfg.scheduler_mode == "random" else None,
    }


# --------------------------------------------------------------------------- main loop
def _block_runs(model, inputs, cfgs, L, rnd, args, ctx, w, log, warmup: bool) -> bool | None:
    """Run one (round, length) block. Returns True when done, None when the guard deadline passed."""
    import torch

    attempt = 0
    while True:
        block_id = f"{ctx['session']}|L{L}|r{rnd}|a{attempt}"
        restart = False
        for pos, cfg in enumerate(cfgs):
            before = foreign_gpu_pids() if args.exclusive_guard else None
            if before:  # another process holds the GPU: wait, re-warm, and restart the block for clean pairing
                if wait_for_exclusive_gpu(args.guard_until, log) is None:
                    return None
                log(f"guard paused for {before}; re-warming block L={L} r={rnd}" + (" and restarting it" if pos else ""))
                for c in cfgs:
                    run_once(model, inputs, c, L, rnd, args, ctx)
                before = foreign_gpu_pids()
                if pos > 0 or before:
                    restart = True
                    break
            res = run_once(model, inputs, cfg, L, rnd, args, ctx)
            after = foreign_gpu_pids() if args.exclusive_guard else None
            if after:
                log(f"foreign GPU processes {after} during {cfg.name} L={L} r={rnd}; restarting the block")
                restart = True
                break
            row = {"kind": "warmup" if warmup else "run", "config": cfg.name, "config_kind": cfg.kind,
                   "label": cfg.label, "scheduler_mode": cfg.scheduler_mode, "route": cfg.route,
                   "enabled": cfg.enabled, "statistic": cfg.statistic, "base_config": cfg.base,
                   "force_chunk": cfg.force_chunk, "chunk_min": cfg.chunk_min, "chunk_max": cfg.chunk_max,
                   "entropy_stride": cfg.entropy_stride, "L": L, "round": rnd, "position": pos,
                   "block_id": block_id, "block_attempt": attempt, "session": ctx["session"],
                   "fast_path": ctx["fast_path"], "new_tokens_requested": args.new_tokens, **res,
                   "gpu_state": gpu_state(), "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
                   "utc": utc_now()}
            if warmup:
                row["wkey"] = run_key(L, rnd, cfg.name)
            else:
                row["key"] = run_key(L, rnd, cfg.name)
            w.write(row, track=not warmup)
            ctx["new_runs"] += 0 if warmup else 1
            log(f"{'warmup ' if warmup else ''}L={L} r={rnd} [{pos + 1}/{len(cfgs)}] {cfg.name:28s} prefill "
                f"{res['prefill_ms']:8.2f} ms  decode {res['decode_ms']:8.2f} ms  total {res['total_ms']:8.2f} ms  "
                f"chunks {res['scan_chunk_counts']}")
            if args.max_new_runs is not None and ctx["new_runs"] >= args.max_new_runs:
                return True
        if not restart:
            return True
        attempt += 1


def run(args) -> None:
    tag = args.tag
    log = Logger(f"e10_routed_{tag}", args.log_dir)
    gpu_baseline = all_gpu_pids()  # before this process creates its CUDA context
    check_disk(args.out_dir, need_gb=0.2, log=log)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (use --dry-run to check the plan on a machine without a GPU)")
    from src.experiments.run_integrated_end_to_end import STATE, _load, install_patch, restore

    os.environ.setdefault("COREY_SELECTIVE_SCAN_BACKEND", "auto")
    adapter = install_chunk_adapter(args.chunk_module, log)
    os.environ["COREY_SELECTIVE_SCAN_DISPATCH_MODULE"] = args.dispatch_module
    device = torch.device("cuda")
    model, tokenizer = _load(args.model, device)
    model.eval()
    from transformers.models.mamba import modeling_mamba as _mm

    missing = [n for n in FAST_PATH_NAMES if getattr(_mm, n, None) is None]
    fast_path = not missing
    own = register_own_gpu_pids(gpu_baseline, log=log) if args.exclusive_guard else set()

    cfgs = build_configs(args)
    dispatch_info = STATE.dispatch_info
    validation = None
    if not fast_path:
        if not args.allow_slow_path:
            raise SystemExit(f"mamba_ssm fast path unavailable (missing {missing}); install mamba_ssm and "
                             "causal_conv1d, or pass --allow-slow-path for a plumbing smoke test")
        log(f"WARNING: fast path unavailable (missing {missing}); only 'passive' runs, on the HF slow path")
        cfgs = [c for c in cfgs if c.name == "passive"]
    else:
        routed_ok = adapter["available"] and bool(dispatch_info and dispatch_info.get("runtime_cuda_available"))
        if routed_ok:
            dispatch = importlib.import_module(args.dispatch_module)
            chunks_valid = sorted(getattr(dispatch, "VALID_CHUNKS", STATIC_CHUNKS))
            from src.experiments.run_integrated_end_to_end import _valid_scheduler_chunks

            wanted = {c.force_chunk for c in cfgs if c.route and c.force_chunk}
            wanted |= {x for c in cfgs if c.route and not c.force_chunk
                       for x in _valid_scheduler_chunks(c.chunk_min, c.chunk_max)}
            wanted |= {args.guard_fallback_chunk}
            if not wanted <= set(chunks_valid):
                raise SystemExit(f"chunks {sorted(wanted - set(chunks_valid))} are not supported by the runtime-chunk "
                                 f"scan (valid: {chunks_valid})")
            validation = validate_routed_scan(dispatch, chunks_valid, log)
            if not validation["passed"]:
                raise SystemExit("the runtime-chunk scan does not match the stock scan; not timing (see log)")
        else:
            skipped = [c.name for c in cfgs if c.route]
            log(f"WARNING: runtime-chunk scan unavailable ({adapter.get('reason') or dispatch_info}); "
                f"skipping the routed configurations {skipped}")
            cfgs = [c for c in cfgs if not c.route]
    if not cfgs:
        raise SystemExit("no configuration left to run")

    policy = DEFAULT_LEARNED_POLICY
    if args.learned_policy_json:
        policy = json.loads(Path(args.learned_policy_json).read_text(encoding="utf-8"))
    text = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else DEFAULT_PROMPT
    prompts: dict[int, dict[str, Any]] = {}
    inputs: dict[int, dict[str, Any]] = {}
    for L in args.lengths:
        ids, reps = build_prompt_ids(tokenizer, text, L)
        t = torch.tensor([ids], dtype=torch.long, device=device)
        inputs[L] = {"input_ids": t, "attention_mask": torch.ones_like(t)}
        prompts[L] = {"L": L, "repeats_of_text": reps, "ids_sha1": _sha1(ids), "text_sha1":
                      hashlib.sha1(text.encode()).hexdigest(), "source": args.prompt_file or "DEFAULT_PROMPT"}
    from transformers import GenerationConfig

    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_kwargs: dict[str, Any] = {"max_new_tokens": args.new_tokens, "do_sample": False, "use_cache": True,
                                  "pad_token_id": pad}
    if hasattr(GenerationConfig(), "disable_compile"):
        gen_kwargs["disable_compile"] = True
    n_layers = int(getattr(model.config, "num_hidden_layers", 0) or getattr(model.config, "n_layer", 0))
    session = f"{utc_now()}|pid{os.getpid()}"
    meta = env_metadata()
    meta.update({"experiment": "E10", "session": session, "model": args.model,
                 "model_id": MODEL_REGISTRY.get(args.model, args.model), "n_layers": n_layers,
                 "fast_path": fast_path, "fast_path_missing": missing, "adapter": adapter,
                 "dispatch_module": args.dispatch_module, "dispatch_info": dispatch_info,
                 "routed_scan_validation": validation, "configs": [asdict(c) for c in cfgs], "prompts": prompts,
                 "gen_kwargs": gen_kwargs, "learned_policy": policy, "own_gpu_pids": sorted(own),
                 "stock_chunk_rule": {"<=128": 128, "<=256": 256, "<=512": 512, "<=1024": 1024, ">1024": 2048},
                 "args": {k: str(v) for k, v in vars(args).items()}})
    jsonl = args.out_dir / f"e10_routed_{tag}.jsonl"
    names = {c.name for c in cfgs}
    done_blocks = complete_blocks(read_jsonl(jsonl), names)
    todo = [(r, L) for r in range(args.rounds) for L in args.lengths if (L, r) not in done_blocks]
    log(f"E10 on {meta.get('gpu')}: {len(cfgs)} configurations x {len(args.lengths)} lengths x {args.rounds} rounds; "
        f"{len(done_blocks)} blocks complete, {len(todo)} to run; fast path {fast_path}; "
        f"routed {'yes' if any(c.route for c in cfgs) else 'no'}")
    if not todo:
        log("all blocks complete; nothing to time")
        analyze(args)
        return
    w = JsonlWriter(jsonl, meta=meta)
    timer = ForwardTimer(model)
    ctx = {"policy": policy, "timer": timer, "gen_kwargs": gen_kwargs, "n_layers": n_layers,
           "fast_path": fast_path, "session": session, "new_runs": 0}
    original = install_patch()
    stopped = False
    try:
        for rnd in list(range(-args.warmup_rounds, 0)) + list(range(args.rounds)):
            warm = rnd < 0
            for L in length_order(args.seed, rnd, args.lengths):
                if not warm and (L, rnd) in done_blocks:
                    continue
                order = config_order(args.seed, rnd, L, cfgs)
                if not warm:
                    have = [c.name for c in order if run_key(L, rnd, c.name) in w.done]
                    if have:
                        log(f"block L={L} r={rnd} is incomplete ({len(have)}/{len(order)} runs from an earlier "
                            "attempt); re-running it in full")
                ok = _block_runs(model, inputs[L], order, L, rnd, args, ctx, w, log, warm)
                if ok is None:
                    stopped = True
                    break
                if args.max_new_runs is not None and ctx["new_runs"] >= args.max_new_runs:
                    stopped = True
                    log(f"--max-new-runs {args.max_new_runs} reached")
                    break
            if stopped:
                break
    finally:
        STATE.enabled = False
        restore(original)
        for h in timer.handles:
            h.remove()
        w.close()
    analyze(args)
    if stopped and args.max_new_runs is None:
        raise SystemExit(3)


# --------------------------------------------------------------------------- analysis
def _tq(n: int) -> float:
    try:
        from scipy.stats import t as t_dist

        return float(t_dist.ppf(0.975, n - 1))
    except Exception:
        return t_crit(n - 1)


def ci(values: list[float]) -> dict[str, Any] | None:
    n = len(values)
    if n == 0:
        return None
    m = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else None
    half = _tq(n) * sd / math.sqrt(n) if n > 1 else None
    return {"n": n, "mean": m, "std": sd, "ci95_half": half, "ci95_lo": m - half if half is not None else None,
            "ci95_hi": m + half if half is not None else None, "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def paired(a: dict[int, float], b: dict[int, float]) -> dict[str, Any] | None:
    rounds = sorted(set(a) & set(b))
    if not rounds:
        return None
    d = [a[r] - b[r] for r in rounds]
    pct = [100.0 * (a[r] / b[r] - 1.0) for r in rounds if b[r]]
    return {"n": len(rounds), "diff_ms": ci(d), "diff_pct": ci(pct)}


def select_blocks(rows: list[dict]) -> tuple[dict[tuple[int, int], dict[str, dict]], int, int]:
    """Per (L, round): the attempt with the most configurations (latest on ties)."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r.get("block_id") or f"legacy|L{r['L']}|r{r['round']}", []).append(r)
    best: dict[tuple[int, int], tuple[int, str]] = {}
    for bid, rs in groups.items():
        key = (int(rs[0]["L"]), int(rs[0]["round"]))
        n = len({x["config"] for x in rs})
        if key not in best or n >= best[key][0]:
            best[key] = (n, bid)
    chosen = {key: {x["config"]: x for x in groups[bid]} for key, (_, bid) in best.items()}
    return chosen, len(groups), len(groups) - len(best)


def _merge_counts(dicts: list[dict | None]) -> dict[str, int]:
    tot: Counter = Counter()
    for d in dicts:
        for k, v in (d or {}).items():
            tot[k] += int(v)
    return {k: tot[k] for k in sorted(tot, key=lambda s: int(s))}


def analyze_length(L: int, blocks: dict[int, dict[str, dict]]) -> dict[str, Any]:
    by_cfg: dict[str, dict[int, dict]] = defaultdict(dict)
    for rnd, cfg_rows in blocks.items():
        for name, row in cfg_rows.items():
            by_cfg[name][rnd] = row
    info = {name: next(iter(rs.values())) for name, rs in by_cfg.items()}
    series = {m: {name: {r: float(row[m]) for r, row in rs.items() if row.get(m) is not None}
                  for name, rs in by_cfg.items()} for m in METRICS + EXTRA_METRICS}
    per_config = {}
    for name, rs in by_cfg.items():
        rows = list(rs.values())
        dists = {json.dumps(r.get("scan_chunk_counts"), sort_keys=True) for r in rows}
        ent = [r["entropy_nats"]["mean"] for r in rows if r.get("entropy_nats")]
        per_config[name] = {
            "kind": info[name].get("config_kind"), "label": info[name].get("label"), "route": info[name].get("route"),
            "statistic": info[name].get("statistic"), "scheduler_mode": info[name].get("scheduler_mode"),
            "base_config": info[name].get("base_config"),
            **{m: ci(list(series[m][name].values())) for m in METRICS + EXTRA_METRICS},
            "n_new_tokens": {"min": min(r["n_new_tokens"] for r in rows), "max": max(r["n_new_tokens"] for r in rows)},
            "scan_chunk_counts": _merge_counts([r.get("scan_chunk_counts") for r in rows]),
            "scan_chunk_source": info[name].get("scan_chunk_source"),
            "selected_chunk_counts": _merge_counts([r.get("selected_chunk_counts") for r in rows]),
            "raw_chunk_counts": _merge_counts([r.get("raw_chunk_counts") for r in rows]) or None,
            "distinct_chunk_distributions": len(dists), "guarded_fallbacks": sum(r.get("guarded_fallbacks", 0)
                                                                                for r in rows),
            "entropy_nats_mean": statistics.fmean(ent) if ent else None,
            "sm_mhz_mean": (statistics.fmean([r["gpu_state"]["sm_mhz"] for r in rows if r.get("gpu_state")])
                            if any(r.get("gpu_state") for r in rows) else None),
        }
    static = [n for n in by_cfg if info[n].get("config_kind") == "static"]
    out: dict[str, Any] = {"L": L, "rounds": len(blocks), "configs": sorted(by_cfg), "per_config": per_config}
    passive = "passive" if "passive" in by_cfg else None

    if static:
        best = {m: min(static, key=lambda n: statistics.fmean(series[m][n].values())) for m in METRICS}
        out["best_static"] = best
        pre = {n: statistics.fmean(series["prefill_ms"][n].values()) for n in static}
        lo, hi = min(pre, key=pre.get), max(pre, key=pre.get)
        out["static_prefill_spread"] = {"fastest": lo, "slowest": hi, "spread_ms": pre[hi] - pre[lo],
                                        "spread_pct": 100.0 * (pre[hi] / pre[lo] - 1.0)}
        out["paired_vs_best_static"] = {m: {n: paired(series[m][n], series[m][best[m]]) for n in by_cfg}
                                        for m in METRICS}
        round_min = {m: {r: min(series[m][n][r] for n in static if r in series[m][n])
                         for r in blocks if any(r in series[m][n] for n in static)} for m in METRICS}
        out["paired_vs_round_min_static"] = {m: {n: paired(series[m][n], round_min[m]) for n in by_cfg}
                                             for m in METRICS}
    if passive:
        out["paired_vs_passive"] = {m: {n: paired(series[m][n], series[m][passive]) for n in by_cfg if n != passive}
                                    for m in METRICS}
        pp = statistics.fmean(series["prefill_ms"][passive].values())
        # decision cost of every statistic-bearing routed variant
        dc = {}
        for name in by_cfg:
            if not (info[name].get("statistic") and info[name].get("route")):
                continue
            rec: dict[str, Any] = {"label": info[name].get("label")}
            hook = f"hook_{name}"
            if hook in by_cfg:
                for m in ("prefill_ms", "total_ms"):
                    rec[f"hook_minus_passive_{m}"] = paired(series[m][hook], series[m][passive])
                if "hook_constant" in by_cfg:
                    for m in ("prefill_ms", "total_ms"):
                        rec[f"hook_minus_hook_constant_{m}"] = paired(series[m][hook], series[m]["hook_constant"])
                p = rec["hook_minus_passive_prefill_ms"]
                if p and p["diff_ms"]:
                    rec["decision_cost_ms"] = p["diff_ms"]["mean"]
                    rec["decision_cost_pct_of_prefill"] = 100.0 * p["diff_ms"]["mean"] / pp if pp else None
            chunks = per_config[name]["scan_chunk_counts"]
            if per_config[name]["distinct_chunk_distributions"] == 1 and len(chunks) == 1:
                c = next(iter(chunks))
                ref = f"static_chunk_{c}"
                if ref in by_cfg:
                    rec["routed_chunk"] = int(c)
                    for m in ("prefill_ms", "total_ms"):
                        rec[f"routed_minus_static_same_chunk_{m}"] = paired(series[m][name], series[m][ref])
            dc[name] = rec
        out["decision_cost"] = dc
        out["passive_prefill_mean_ms"] = pp
        # output equality versus passive in the same block
        eq = {}
        p_rows = by_cfg[passive]
        out["passive_distinct_outputs"] = len({r["token_hash"] for r in p_rows.values()})
        for name, rs in by_cfg.items():
            if name == passive:
                continue
            common = sorted(set(rs) & set(p_rows))
            same = sum(1 for r in common if rs[r]["token_hash"] == p_rows[r]["token_hash"])
            mism = 0
            for r in common:
                a, b = rs[r].get("new_token_ids") or [], p_rows[r].get("new_token_ids") or []
                mism += sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
            eq[name] = {"rounds": len(common), "equal": same, "all_equal": same == len(common),
                        "token_mismatches": mism}
        out["output_equality_vs_passive"] = eq
    return out


def analyze(args) -> dict[str, Any]:
    tag = args.tag
    path = args.out_dir / f"e10_routed_{tag}.jsonl"
    all_rows = read_jsonl(path)
    rows = [r for r in all_rows if r.get("kind") == "run"]
    metas = read_jsonl(path, kind="meta")
    chosen, n_attempts, n_discarded = select_blocks(rows)
    res: dict[str, Any] = {
        "experiment": "E10-analysis", "tag": tag, "source": path.name, "n_run_rows": len(rows),
        "n_warmup_rows": sum(1 for r in all_rows if r.get("kind") == "warmup"),
        "sessions": sorted({r.get("session") for r in rows if r.get("session")}),
        "n_block_attempts": n_attempts, "n_block_attempts_discarded": n_discarded, "n_blocks_used": len(chosen),
        "metrics": list(METRICS), "extra_metrics": list(EXTRA_METRICS),
        "meta": ({k: metas[-1].get(k) for k in ("gpu", "torch", "pkg_transformers", "pkg_mamba_ssm", "model_id",
                                                "n_layers", "fast_path", "adapter", "routed_scan_validation",
                                                "prompts", "gen_kwargs", "git_commit")} if metas else None),
        "definitions": {
            "prefill_ms": "CUDA events around the forward call that consumes the prompt",
            "decode_ms": "end of the prefill forward to the end of generate (decode steps and sampling)",
            "total_ms": "CUDA events around generate",
            "paired": "difference within a (round, length) block, mean and t-distribution 95% CI over rounds",
            "best_static": "static chunk with the lowest mean of the metric at that length",
            "decision_cost_ms": "hook-only control minus passive, prefill, paired",
        },
        "lengths": {},
    }
    by_len: dict[int, dict[int, dict[str, dict]]] = defaultdict(dict)
    for (L, rnd), cfg_rows in chosen.items():
        by_len[L][rnd] = cfg_rows
    for L in sorted(by_len):
        res["lengths"][str(L)] = analyze_length(L, by_len[L])
    out = args.out_dir / f"e10_routed_{tag}.json"
    write_json(out, res)
    _print_summary(res)
    print(f"analysis written to {out}")
    return res


def _fmt(x: dict | None, key: str = "mean", digits: int = 2) -> str:
    if not x or x.get(key) is None:
        return "n/a"
    half = x.get("ci95_half")
    return f"{x[key]:.{digits}f}" + (f" +/- {half:.{digits}f}" if half is not None else "")


def _print_summary(res: dict[str, Any]) -> None:
    for Ls, d in res["lengths"].items():
        best = d.get("best_static", {})
        print(f"\nL={Ls}: {d['rounds']} rounds; best static prefill {best.get('prefill_ms')}, "
              f"total {best.get('total_ms')}")
        print(f"{'configuration':30s} {'prefill ms':>18s} {'total ms':>20s} {'dPrefill% vs best':>20s} "
              f"{'dTotal% vs best':>18s} {'same tokens':>12s}  scan chunks")
        eq = d.get("output_equality_vs_passive", {})
        for name in d["configs"]:
            pc = d["per_config"][name]
            pb = d.get("paired_vs_best_static", {})
            dp = (pb.get("prefill_ms", {}).get(name) or {}).get("diff_pct")
            dt = (pb.get("total_ms", {}).get(name) or {}).get("diff_pct")
            e = eq.get(name)
            same = f"{e['equal']}/{e['rounds']}" if e else "-"
            print(f"{name:30s} {_fmt(pc['prefill_ms']):>18s} {_fmt(pc['total_ms']):>20s} {_fmt(dp):>20s} "
                  f"{_fmt(dt):>18s} {same:>12s}  {pc['scan_chunk_counts']}")
        for name, rec in (d.get("decision_cost") or {}).items():
            if rec.get("decision_cost_ms") is not None:
                pct = rec.get("decision_cost_pct_of_prefill")
                print(f"  decision cost {name}: {rec['decision_cost_ms']:.2f} ms"
                      + (f" ({pct:.1f}% of passive prefill)" if pct is not None else ""))


# --------------------------------------------------------------------------- dry run
def dry_run(args) -> None:
    cfgs = build_configs(args)
    have_mamba = importlib.util.find_spec("mamba_ssm") is not None
    have_chunk = importlib.util.find_spec(args.chunk_module) is not None
    print(f"E10 plan (dry run; tag {args.tag}, out {args.out_dir / f'e10_routed_{args.tag}.jsonl'})")
    print(f"  model {args.model} ({MODEL_REGISTRY.get(args.model, args.model)}), new tokens {args.new_tokens}, "
          f"prompt {'file ' + str(args.prompt_file) if args.prompt_file else 'DEFAULT_PROMPT tiled'}")
    print(f"  lengths {args.lengths}; {args.warmup_rounds} warmup rounds + {args.rounds} timed rounds; "
          f"order seed {args.seed}")
    print(f"  mamba_ssm importable: {have_mamba}; {args.chunk_module} importable: {have_chunk} "
          f"(routed configurations are skipped without it)")
    print(f"  {len(cfgs)} configurations:")
    print(f"    {'name':28s} {'kind':10s} {'mode':24s} {'route':>5s} {'stat':>5s} {'range':>11s} {'stride':>6s} base")
    for c in cfgs:
        rng = f"{c.force_chunk}" if c.force_chunk else f"{c.chunk_min}-{c.chunk_max}"
        print(f"    {c.name:28s} {c.kind:10s} {c.scheduler_mode:24s} {str(c.route):>5s} {str(c.statistic):>5s} "
              f"{rng:>11s} {c.entropy_stride:>6d} {c.base or ''}")
    n_timed = len(cfgs) * len(args.lengths) * args.rounds
    n_warm = len(cfgs) * len(args.lengths) * args.warmup_rounds
    print(f"  runs: {n_timed} timed + {n_warm} warmup; at ~{args.est_s_per_run:.1f} s per run about "
          f"{(n_timed + n_warm) * args.est_s_per_run / 3600:.2f} h")
    jsonl = args.out_dir / f"e10_routed_{args.tag}.jsonl"
    if jsonl.exists():
        done = complete_blocks(read_jsonl(jsonl), {c.name for c in cfgs})
        print(f"  existing result file: {len(done)} of {len(args.lengths) * args.rounds} blocks complete")
    for rnd in ([-1] if args.warmup_rounds else []) + ([0] if args.rounds else []):
        for L in length_order(args.seed, rnd, args.lengths)[:1]:
            order = [c.name for c in config_order(args.seed, rnd, L, cfgs)]
            print(f"  round {rnd} length order {length_order(args.seed, rnd, args.lengths)}; "
                  f"first block (L={L}) config order: {order}")


# --------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="mamba-370m", choices=list(MODEL_REGISTRY))
    p.add_argument("--lengths", type=_parse_ints, default=list(LENGTHS), help="prompt lengths in tokens")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--warmup-rounds", type=int, default=3)
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--configs", default=DEFAULT_CONFIGS,
                   help=f"comma-separated subset of: {', '.join(VALID_VARIANTS)}")
    p.add_argument("--static-chunks", type=_parse_ints, default=list(STATIC_CHUNKS))
    p.add_argument("--adaptive-chunk-min", type=int, default=None, help="default: smallest static chunk")
    p.add_argument("--adaptive-chunk-max", type=int, default=None, help="default: largest static chunk")
    p.add_argument("--entropy-stride", type=int, default=8)
    p.add_argument("--num-bins", type=int, default=256)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--guard-fallback-chunk", type=int, default=512)
    p.add_argument("--guard-min-delta-buckets", type=int, default=2)
    p.add_argument("--learned-policy-json", type=Path, default=None,
                   help="rule table for learned_table (default: the embedded tpds-r1 policy)")
    p.add_argument("--no-hook-controls", dest="hook_controls", action="store_false",
                   help="do not add the hook-only (not routed) controls that measure the decision cost")
    p.add_argument("--prompt-file", type=Path, default=None, help="text to tile instead of DEFAULT_PROMPT")
    p.add_argument("--dispatch-module", default="src.corey_selective_scan_dispatch")
    p.add_argument("--chunk-module", default="selective_scan_chunk_cuda",
                   help="module that provides fwd_with_chunk_size when selective_scan_cuda lacks it")
    p.add_argument("--exclusive-guard", action="store_true")
    p.add_argument("--guard-until", type=float, default=None, help="epoch seconds after which a busy GPU stops the run")
    p.add_argument("--allow-slow-path", action="store_true",
                   help="plumbing smoke without mamba_ssm: run 'passive' on the HF slow path only")
    p.add_argument("--max-new-runs", type=int, default=None, help="stop after this many new timed runs (smoke)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run: 1 warmup + 2 timed rounds, L=256, 4 new tokens, smoke directory, slow path allowed")
    p.add_argument("--dry-run", action="store_true", help="print the schedule and exit (no torch, no model)")
    p.add_argument("--analyze", action="store_true", help="only (re)write the analysis from the JSONL")
    p.add_argument("--est-s-per-run", type=float, default=1.2, help="dry run: assumed seconds per run")
    p.add_argument("--out-dir", type=Path, default=None, help=f"default {RESULTS_DIR} ({SMOKE_DIR} with --smoke)")
    p.add_argument("--log-dir", type=Path, default=LOG_DIR)
    p.add_argument("--tag", default=None, help="result-file tag (default: host name; smoke-<host> with --smoke)")
    p.add_argument("--seed", type=int, default=0, help="seed of the length and configuration orders")
    args = p.parse_args(argv)
    if args.smoke:
        args.rounds, args.warmup_rounds, args.lengths, args.new_tokens = 2, 1, [256], 4
        args.allow_slow_path = True
    args.out_dir = Path(args.out_dir) if args.out_dir else (SMOKE_DIR if args.smoke else RESULTS_DIR)
    args.tag = args.tag or (f"smoke-{host_name()}" if args.smoke else host_name())
    args.lengths = sorted(dict.fromkeys(args.lengths))
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        dry_run(args)
    elif args.analyze:
        analyze(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
