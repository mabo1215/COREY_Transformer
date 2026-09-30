"""Serving-trace replay of the key-churn break-even (item M1b).

E5b bounds the calls per shape key below which an online per-call chunk
policy (decision overhead o per call, no tuning) can be cheaper than a
shape-keyed table (tuning cost P_k once per key, headroom h_k):
N_k < N_k* = P_k / (o - h_k). E5b evaluated N_k on the captured LongBench
workload, whose prompt lengths were truncated and rotated by design. This
script replays public LLM serving traces (per-request prompt lengths and
arrival times) through a prefill scheduler model and counts, per shape key,
the scan calls that the scheduler issues.

Traces (downloaded separately, see ``--data-dir``):
* Azure LLM inference trace 2023 (code, conversation; one hour each).
* Azure LLM inference trace 2024 (code, conversation; one week each):
  first hour, first 24 hours, and the whole week (the last only for the
  load-independent policies).
* BurstGPT v2.0 file 1 without failed requests (61 days): busiest calendar
  day and the whole trace.

Prefill policies (every policy issues one scan call per layer per step; a
request of the 48-layer checkpoint contributes 48 calls, the E5b convention):
* ``per_request``: one request per prefill call (batch 1, exact prompt length).
* ``chunked_budget``: continuous batching with chunked prefill and a token
  budget T per scheduler step (Sarathi-Serve / vLLM style). Requests arrive at
  their (rescaled) trace timestamps and are served first-come first-served; a
  step packs prefill chunks of up to T tokens in total, a long prompt spans
  several steps. ``colocated``: decoding requests share the step and consume
  one budget token each (their tokens go through the single-step state update,
  not through the scan); ``prefill_only``: a disaggregated prefill instance
  without decodes. Two call shapes per step: a padded batched kernel keyed by
  (number of sequences, longest chunk) and a packed variable-length kernel
  keyed by (total packed tokens, number of sequences).
* ``fixed_slice``: every prompt is cut into T-token slices processed one per
  call (batch 1), so all calls share the shape (1, T) except the tails.

The scheduler needs a step-time model: t_step = t_fix + t_pf * prefill tokens
+ t_dec * decoding sequences, with max_seqs concurrently running requests.
Arrival times are rescaled so that the offered prefill load is rho times the
prefill capacity 1 / t_pf (the burstiness of the trace is kept). Only the
``chunked_budget`` policy depends on these assumptions.

Key modes follow E5b: ``exact`` (L, batch), ``bucket`` (power-of-two length
bucket, batch), ``bucket_aligned`` (bucket, batch, L mod 16 == 0). For the
packed kernel, L is the total packed length and batch the number of packed
sequences; a fourth mode, ``bucket_aligned_tokens_only`` (bucket, L mod 16 ==
0), drops the sequence count, which only moves the state resets of a packed
scan.

Per-key break-even. Every exact call shape is mapped to the nearest measured
E5b key (bucket_aligned records of ``e5b_<host>.json``; distance = |d log2
bucket| + |d log2 batch| + 1 for an alignment mismatch, ties broken towards the
larger tuning cost); a packed call is costed as a batch-1 scan of its total
length. Beyond the measured range (L > 8192 or batch > 16) the tuning cost,
headroom and cost-model regret of the nearest key are scaled by L / 8192 and
batch / 16, i.e., by the growth of the summed chunk latency that the E5b
tuning_scale formula (P = seconds per ms of summed latency) multiplies. The
decision overheads are the E5b ones: one host synchronization, and the E4
histogram costs at the nearest measured tensor size (E5b ``_overhead``). A key
aggregates the call-weighted means of P, h and o over its exact shapes; the
online policy can only win on a key whose N_k (calls in the window) is below
N_k*. Cold-start keys: calls after the first X % of the calls (in time order)
whose key did not occur in that first X %; the table serves them with the
cost-model default, and the online policy beats that default on a key only if
o is below the model's regret there (E5b ``model_default``).

Outputs: ``serving_key_churn.json`` (all tables and provenance) in the results
directory and ``serving_traces_manifest.json`` (sizes, checksums, length
summaries) in the data directory.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import time
from array import array
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.tpds.common import REPO_ROOT, RESULTS_DIR, read_jsonl, utc_now, write_json
from src.experiments.tpds.e2_oracle_sweep import length_bucket
from src.experiments.tpds.e5b_key_churn import KEY_MODES, LAYERS_PER_REQUEST, _n_star, _overhead

DATA_DIR = REPO_ROOT / "src" / "data" / "serving_traces"

TRACE_FILES = {
    "azure2023_code": {
        "file": "AzureLLMInferenceTrace_code.csv",
        "url": "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_code.csv",
        "source": "Azure LLM inference trace 2023 (Patel et al., Splitwise, ISCA 2024), CC-BY",
        "format": "azure"},
    "azure2023_conv": {
        "file": "AzureLLMInferenceTrace_conv.csv",
        "url": "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv",
        "source": "Azure LLM inference trace 2023 (Patel et al., Splitwise, ISCA 2024), CC-BY",
        "format": "azure"},
    "azure2024_code": {
        "file": "AzureLLMInferenceTrace_code_1week.csv",
        "url": "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/"
               "AzureLLMInferenceTrace_code_1week.csv",
        "source": "Azure LLM inference trace 2024 (Stojkovic et al., DynamoLLM, HPCA 2025), CC-BY",
        "format": "azure"},
    "azure2024_conv": {
        "file": "AzureLLMInferenceTrace_conv_1week.csv",
        "url": "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/"
               "AzureLLMInferenceTrace_conv_1week.csv",
        "source": "Azure LLM inference trace 2024 (Stojkovic et al., DynamoLLM, HPCA 2025), CC-BY",
        "format": "azure"},
    "burstgpt": {
        "file": "BurstGPT_without_fails_1.csv",
        "url": "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_without_fails_1.csv",
        "source": "BurstGPT v2.0, file 1 without failed requests (Wang et al., KDD 2025), CC-BY-4.0",
        "format": "burstgpt"},
}

# (trace, window, policies); "cb" = chunked_budget, needs the scheduler simulation
WINDOWS = [
    ("azure2023_code", "1h", ("per_request", "fixed_slice", "cb")),
    ("azure2023_conv", "1h", ("per_request", "fixed_slice", "cb")),
    ("azure2024_code", "1h", ("per_request", "fixed_slice", "cb")),
    ("azure2024_code", "24h", ("per_request", "fixed_slice", "cb")),
    ("azure2024_code", "full", ("per_request", "fixed_slice")),
    ("azure2024_conv", "1h", ("per_request", "fixed_slice", "cb")),
    ("azure2024_conv", "24h", ("per_request", "fixed_slice", "cb")),
    ("azure2024_conv", "full", ("per_request", "fixed_slice")),
    ("burstgpt", "24h_busiest", ("per_request", "fixed_slice", "cb")),
    ("burstgpt", "full", ("per_request", "fixed_slice", "cb")),
]
BUDGETS = (512, 2048, 8192)
RHO_PRIMARY = 0.5
RHO_SENSITIVITY = (0.2, 0.85)
WARM_FRACS = (0.01, 0.10, 0.50)
# step-time model of a small (370M-parameter, 48-layer) checkpoint on one GPU; only ratios matter after
# the load normalization
T_FIX_S = 2.0e-3                     # per-step fixed cost (weights, launches)
T_PF_S = 10.0e-6                     # per prefill token
T_DEC_S = 20.0e-6                    # per decoding sequence
MAX_SEQS = 256                       # running requests (prefilling + decoding)
NSTAR_COMBOS = [f"{p}|{o}" for p in ("profile", "search") for o in ("host_sync", "sampled_histogram",
                                                                        "full_histogram")]
SHAPE_BITS = 21                      # batch < 2**21 in the integer key codes
# extra key for the packed variable-length kernel: the number of packed sequences only moves the state
# resets, so a packed-call table can be keyed by the packed length alone
PACKED_TOKENS_MODE = "bucket_aligned_tokens_only"


# --------------------------------------------------------------------------- data
def sha256_of(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_trace(name: str, data_dir: Path) -> dict:
    """Arrival time (s from trace start), prompt tokens, output tokens; time-sorted."""
    spec = TRACE_FILES[name]
    path = data_dir / spec["file"]
    if spec["format"] == "azure":
        df = pd.read_csv(path, dtype={"ContextTokens": np.int64, "GeneratedTokens": np.int64})
        ts = pd.to_datetime(df["TIMESTAMP"], format="ISO8601", utc=True)
        t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
        ctx, gen = df["ContextTokens"].to_numpy(), df["GeneratedTokens"].to_numpy()
        start = str(ts.iloc[0])
    else:
        df = pd.read_csv(path)
        t = df["Timestamp"].to_numpy(dtype=float)
        t = t - t[0]
        ctx, gen = df["Request tokens"].to_numpy(np.int64), df["Response tokens"].to_numpy(np.int64)
        start = "t=0 (relative timestamps)"
    order = np.argsort(t, kind="stable")
    t, ctx, gen = t[order], ctx[order], gen[order]
    keep = ctx >= 1
    return {"t": t[keep], "ctx": ctx[keep], "gen": np.maximum(gen[keep], 1), "start": start,
            "rows_raw": int(len(df)), "rows_dropped_zero_ctx": int((~keep).sum())}


def select_window(tr: dict, window: str) -> tuple[dict, dict]:
    t = tr["t"]
    if window == "full":
        lo, hi = 0.0, float(t[-1]) + 1e-9
        note = "whole trace"
    elif window == "1h":
        lo, hi, note = 0.0, 3600.0, "first hour"
    elif window == "24h":
        lo, hi, note = 0.0, 86400.0, "first 24 hours"
    elif window == "24h_busiest":
        counts = np.bincount(np.floor(t / 86400.0).astype(np.int64))
        d = int(np.argmax(counts[:-1])) if len(counts) > 1 else 0       # skip a partial last day
        lo, hi = 86400.0 * d, 86400.0 * (d + 1)
        note = f"busiest calendar day (day {d} from trace start)"
    else:
        raise ValueError(window)
    m = (t >= lo) & (t < hi)
    w = {"t": t[m] - lo, "ctx": tr["ctx"][m], "gen": tr["gen"][m]}
    info = {"window": window, "selection": note, "t_lo_s": lo, "t_hi_s": min(hi, float(t[-1])),
            "duration_s": float(min(hi, t[-1] + 1e-9) - lo), "n_requests": int(m.sum())}
    return w, info


def length_summary(ctx: np.ndarray, gen: np.ndarray, duration_s: float) -> dict:
    q = np.percentile(ctx, [1, 10, 25, 50, 75, 90, 99])
    lb = np.log2(np.maximum(ctx, 1)).astype(int)
    hist = np.bincount(lb)
    return {"prompt_tokens_quantiles": dict(zip(["p1", "p10", "p25", "p50", "p75", "p90", "p99"],
                                                 [float(x) for x in q])),
            "prompt_tokens_mean": float(ctx.mean()), "prompt_tokens_max": int(ctx.max()),
            "output_tokens_mean": float(gen.mean()),
            "distinct_prompt_lengths": int(len(np.unique(ctx))),
            "prompt_len_pow2_bucket_hist": {str(1 << i): int(c) for i, c in enumerate(hist) if c},
            "share_prompts_len_mod16_eq0": float((ctx % 16 == 0).mean()),
            "requests_per_s": float(len(ctx) / max(duration_s, 1e-9)),
            "prompt_tokens_per_s": float(ctx.sum() / max(duration_s, 1e-9)),
            "native_rho": float(ctx.sum() / max(duration_s, 1e-9) * T_PF_S)}


# --------------------------------------------------------------------------- scheduler
def simulate_chunked(arr: np.ndarray, ctx: np.ndarray, gen: np.ndarray, T: int, colocate: bool) -> dict:
    """Chunked-prefill continuous batching (FCFS). Returns per-step prefill shapes and queueing stats."""
    arr_l, rem, gen_l = arr.tolist(), ctx.tolist(), gen.tolist()
    n = len(arr_l)
    q: deque = deque()
    dec: list = []                    # heap of the step index at which a decoding request finishes
    ttft = np.empty(n)
    nseq_o, lmax_o, tot_o = array("i"), array("i"), array("i")
    tend_o = array("d")
    i, now, step, dec_only = 0, 0.0, 0, 0
    full_steps = 0
    while True:
        while i < n and arr_l[i] <= now:
            q.append(i)
            i += 1
        while dec and dec[0] <= step:
            heapq.heappop(dec)
        n_dec = len(dec)
        if not q:
            if not dec:
                if i >= n:
                    break
                now = arr_l[i]
                continue
            d = T_FIX_S + T_DEC_S * n_dec          # decode-only steps until the next event
            k = dec[0] - step
            if i < n:
                k = max(1, min(k, math.ceil((arr_l[i] - now) / d)))
            step += k
            now += k * d
            dec_only += k
            continue
        budget = T - n_dec if colocate else T
        slots = MAX_SEQS - n_dec if colocate else MAX_SEQS
        if budget <= 0 or slots <= 0:
            now += T_FIX_S + T_DEC_S * n_dec
            step += 1
            dec_only += 1
            continue
        avail = budget
        cnt = lmax = tot = 0
        done = []
        while q and budget > 0 and cnt < slots:
            j = q[0]
            r = rem[j]
            c = r if r < budget else budget
            rem[j] = r - c
            budget -= c
            tot += c
            cnt += 1
            if c > lmax:
                lmax = c
            if r == c:
                q.popleft()
                done.append(j)
                if colocate and gen_l[j] > 1:
                    heapq.heappush(dec, step + gen_l[j])
            else:
                break
        now += T_FIX_S + T_PF_S * tot + T_DEC_S * n_dec
        step += 1
        full_steps += tot == avail
        for j in done:
            ttft[j] = now - arr_l[j]
        nseq_o.append(cnt)
        lmax_o.append(lmax)
        tot_o.append(tot)
        tend_o.append(now)
    nseq = np.frombuffer(nseq_o, dtype=np.int32).astype(np.int64)
    tot = np.frombuffer(tot_o, dtype=np.int32).astype(np.int64)
    span = float(arr[-1]) if n else 0.0
    return {"nseq": nseq, "lmax": np.frombuffer(lmax_o, dtype=np.int32).astype(np.int64), "tot": tot,
            "stats": {"prefill_steps": int(len(nseq)), "decode_only_steps": int(dec_only),
                      "mean_seqs_per_prefill_step": float(nseq.mean()),
                      "share_multi_seq_steps": float((nseq > 1).mean()),
                      "mean_prefill_tokens_per_step": float(tot.mean()),
                      "share_steps_budget_full": float(full_steps / max(len(nseq), 1)),
                      "ttft_p50_ms": float(1000 * np.percentile(ttft, 50)),
                      "ttft_p99_ms": float(1000 * np.percentile(ttft, 99)),
                      "makespan_over_arrival_span": float(now / span) if span > 0 else None,
                      "overloaded": bool(span > 0 and (now / span > 1.05 or np.percentile(ttft, 99) > 0.1 * span))}}


# --------------------------------------------------------------------------- per-host cost mapping
class HostCosts:
    """Maps exact call shapes (L, batch) to E5b tuning cost, headroom, overheads and cost-model regret."""

    def __init__(self, host: str, results_dir: Path):
        self.host = host
        self.path = results_dir / f"e5b_{host}.json"
        e5b = json.loads(self.path.read_text())
        self.e5b = e5b
        self.e4_rows = read_jsonl(results_dir / f"e4_{host}.jsonl")
        e2 = pd.read_csv(results_dir / f"e2_mamba1_{host}.csv", usecols=["width", "L", "batch"])
        self.width = int(e2["width"].iloc[0])
        self.L_max, self.b_max = int(e2["L"].max()), int(e2["batch"].max())
        recs = e5b["keys"]
        self.rLb = np.log2([r["L_bucket"] for r in recs])
        self.rb = np.log2([r["batch"] for r in recs])
        self.ral = np.array([r["aligned"] for r in recs])
        self.rPp = np.array([r["P_s"]["profile"] for r in recs])
        self.rPs = np.array([r["P_s"]["search"] for r in recs])
        self.rh = np.array([r["headroom_ms"] for r in recs])
        self.rreg = np.array([r.get("model_regret_vs_oracle_ms", np.nan) for r in recs])
        self.sync_ms = float(e5b["host_sync_ms"])
        self._ov_cache: dict = {}

    def _ov(self, numel: int, stride: int) -> float:
        k = (numel, stride)
        if k not in self._ov_cache:
            self._ov_cache[k] = _overhead(self.e4_rows, numel, stride)
        return self._ov_cache[k]

    def costs(self, L: np.ndarray, b: np.ndarray) -> dict:
        Lb = np.left_shift(1, _bucket_log2(L))
        assert all(length_bucket(int(x)) == int(y) for x, y in zip(L[:2000], Lb[:2000]))
        al = (L % 16 == 0).astype(int)
        d = (np.abs(np.log2(Lb)[:, None] - self.rLb[None]) + np.abs(np.log2(b)[:, None] - self.rb[None])
             + (al[:, None] != self.ral[None]))
        d = d - 1e-9 * self.rPp[None] / self.rPp.max()           # ties -> the larger tuning cost
        nn = np.argmin(d, axis=1)
        scale = np.maximum(1.0, L / self.L_max) * np.maximum(1.0, b / self.b_max)
        numel = (b * self.width * L).astype(np.int64)
        return {"P_profile": self.rPp[nn] * scale, "P_search": self.rPs[nn] * scale,
                "h": self.rh[nn] * scale, "regret": self.rreg[nn] * scale,
                "o_host_sync": np.full(len(L), self.sync_ms),
                "o_sampled_histogram": np.array([self._ov(int(x), 8) for x in numel]),
                "o_full_histogram": np.array([self._ov(int(x), 1) for x in numel]),
                "extrapolated": scale > 1.0}


# --------------------------------------------------------------------------- key statistics
def _bucket_log2(L: np.ndarray) -> np.ndarray:
    """log2 of e2_oracle_sweep.length_bucket, vectorized."""
    return np.floor(np.log2(np.maximum(L, 1))).astype(np.int64)


def key_stats(Lk: np.ndarray, bk: np.ndarray, w: np.ndarray, packed: bool, hosts: list[HostCosts]) -> dict:
    """Per key mode and host: key counts, calls per key, break-even shares, cold-start shares.

    Lk, bk: key length and batch of every call entry (time order); w: steps represented by the entry.
    """
    code = Lk.astype(np.int64) << SHAPE_BITS | bk.astype(np.int64)
    ue, inv = np.unique(code, return_inverse=True)
    We = np.bincount(inv, weights=w)
    Le, be = ue >> SHAPE_BITS, ue & ((1 << SHAPE_BITS) - 1)
    host_costs = {h.host: h.costs(Le, np.ones_like(be) if packed else be) for h in hosts}
    total_steps = float(w.sum())
    cw = np.cumsum(w) - w                                 # steps before each entry
    out = {}
    modes = dict(KEY_MODES)
    if packed:
        modes[PACKED_TOKENS_MODE] = ["L_bucket", "aligned"]
    for mode, cols in modes.items():
        if mode == "exact":
            kcode = np.arange(len(ue))
        else:
            kc = _bucket_log2(Le)
            if "aligned" in cols:
                kc = kc * 2 + (Le % 16 == 0)
            kcode = kc << SHAPE_BITS | (be if "batch" in cols else 0)
        uk, kinv = np.unique(kcode, return_inverse=True)
        Wk = np.bincount(kinv, weights=We)
        calls = LAYERS_PER_REQUEST * Wk
        call_key = kinv[inv]
        rec = {"n_keys": int(len(uk)), "n_calls": float(calls.sum()), "n_steps": total_steps,
               "calls_per_key": {"median": float(np.median(calls)), "p10": float(np.percentile(calls, 10)),
                                 "p90": float(np.percentile(calls, 90)), "min": float(calls.min()),
                                 "max": float(calls.max()),
                                 "share_keys_single_step": float((Wk <= 1).mean()),
                                 "share_calls_on_single_step_keys": float(Wk[Wk <= 1].sum() / Wk.sum())},
               "cold_start": {}, "by_host": {}}
        for x in WARM_FRACS:
            warm = cw < x * total_steps
            seen = np.zeros(len(uk), bool)
            seen[call_key[warm]] = True
            rest = ~warm
            cold = rest & ~seen[call_key]
            rec["cold_start"][f"warm_{int(100 * x)}pct"] = {
                "share_later_calls_on_unseen_keys": float(w[cold].sum() / max(w[rest].sum(), 1e-12)),
                "n_keys_seen_in_warmup": int(seen.sum())}
        for h in hosts:
            c = host_costs[h.host]
            agg = {k: np.bincount(kinv, weights=We * v) / Wk for k, v in c.items() if k != "extrapolated"}
            hr = {"share_calls_extrapolated": float(We[c["extrapolated"]].sum() / We.sum()),
                  "share_calls_below_nstar": {}, "share_keys_below_nstar": {}, "nstar_median_calls": {},
                  "cache_tuning_total_s": {}, "online_min_excess_s": {}, "online_beats_model_default": {}}
            for combo in NSTAR_COMBOS:
                pk, ok = combo.split("|")
                ns = [_n_star(p, o, hh) for p, o, hh in zip(agg[f"P_{pk}"], agg[f"o_{ok}"], agg["h"])]
                below = np.array([s is None or n < s for n, s in zip(calls, ns)])
                finite = [s for s in ns if s is not None]
                hr["share_calls_below_nstar"][combo] = float(calls[below].sum() / calls.sum())
                hr["share_keys_below_nstar"][combo] = float(below.mean())
                hr["nstar_median_calls"][combo] = float(np.median(finite)) if finite else None
            for pk in ("profile", "search"):
                hr["cache_tuning_total_s"][pk] = float(agg[f"P_{pk}"].sum())
            for ok in ("host_sync", "sampled_histogram", "full_histogram"):
                hr["online_min_excess_s"][ok] = float((calls * (agg[f"o_{ok}"] - agg["h"])).sum() / 1000.0)
                beats = agg[f"o_{ok}"] < agg["regret"]
                hr["online_beats_model_default"][ok] = {
                    "share_calls": float(calls[beats].sum() / calls.sum()),
                    "share_cold_calls_warm10": float(
                        w[(cw >= 0.1 * total_steps) & beats[call_key]
                          & ~np.isin(call_key, np.unique(call_key[cw < 0.1 * total_steps]))].sum()
                        / max(w[cw >= 0.1 * total_steps].sum(), 1e-12))}
            rec["by_host"][h.host] = hr
        out[mode] = rec
    return out


# --------------------------------------------------------------------------- policies
def policy_entries(win: dict, policy: str, T: int | None = None) -> dict:
    ctx = win["ctx"]
    if policy == "per_request":
        return {"batch1": (ctx, np.ones_like(ctx), np.ones(len(ctx)))}
    if policy == "fixed_slice":
        nfull, tail = ctx // T, ctx % T
        L = np.stack([np.full(len(ctx), T), tail], 1).ravel()
        w = np.stack([nfull, (tail > 0).astype(np.int64)], 1).ravel().astype(float)
        m = w > 0
        return {"batch1": (L[m], np.ones(int(m.sum()), np.int64), w[m])}
    raise ValueError(policy)


def run_window(name: str, window: str, policies, tr: dict, hosts: list[HostCosts], log) -> tuple[dict, list]:
    win, info = select_window(tr, window)
    info.update(length_summary(win["ctx"], win["gen"], info["duration_s"]))
    rows = []
    for policy in policies:
        if policy == "per_request":
            t0 = time.time()
            Lk, bk, w = policy_entries(win, "per_request")["batch1"]
            rows.append({"policy": "per_request", "T": None, "kernel": "batch1", "sched": None, "rho": None,
                         "keys": key_stats(Lk, bk, w, False, hosts)})
            log(f"  {name}/{window} per_request: {time.time() - t0:.1f}s")
        elif policy == "fixed_slice":
            for T in BUDGETS:
                t0 = time.time()
                Lk, bk, w = policy_entries(win, "fixed_slice", T)["batch1"]
                rows.append({"policy": "fixed_slice", "T": T, "kernel": "batch1", "sched": None, "rho": None,
                             "keys": key_stats(Lk, bk, w, False, hosts)})
                log(f"  {name}/{window} fixed_slice T={T}: {time.time() - t0:.1f}s")
        elif policy == "cb":
            settings = ([("colocated", RHO_PRIMARY)] + [("colocated", r) for r in RHO_SENSITIVITY]
                        + [("prefill_only", RHO_PRIMARY)])
            tok = float(win["ctx"].sum())
            for sched, rho in settings:
                s = tok * T_PF_S / (info["duration_s"] * rho)     # time dilation for offered load rho
                arr = win["t"] * s
                for T in BUDGETS:
                    t0 = time.time()
                    sim = simulate_chunked(arr, win["ctx"], win["gen"], T, colocate=(sched == "colocated"))
                    w = np.ones(len(sim["nseq"]))
                    for kernel, (Lk, bk) in (("padded", (sim["lmax"], sim["nseq"])),
                                             ("packed", (sim["tot"], sim["nseq"]))):
                        rows.append({"policy": "chunked_budget", "T": T, "kernel": kernel, "sched": sched,
                                     "rho": rho, "time_dilation": s, "sim": sim["stats"],
                                     "keys": key_stats(Lk, bk, w, kernel == "packed", hosts)})
                    log(f"  {name}/{window} cb {sched} rho={rho} T={T}: {sim['stats']['prefill_steps']} steps, "
                        f"{sim['stats']['mean_seqs_per_prefill_step']:.2f} seqs/step, "
                        f"overloaded={sim['stats']['overloaded']}, {time.time() - t0:.1f}s")
    return info, rows


# --------------------------------------------------------------------------- summary
def summary_rows(results: list[dict], host: str) -> list[dict]:
    out = []
    for r in results:
        for mode in ("exact", "bucket_aligned"):
            k = r["keys"][mode]
            hk = k["by_host"][host]
            out.append({"trace": r["trace"], "window": r["window"], "policy": r["policy"], "T": r["T"],
                        "kernel": r["kernel"], "sched": r["sched"], "rho": r["rho"], "key_mode": mode,
                        "n_keys": k["n_keys"], "median_calls_per_key": k["calls_per_key"]["median"],
                        "p10_calls_per_key": k["calls_per_key"]["p10"],
                        "share_calls_below_nstar_profile_sync": hk["share_calls_below_nstar"]["profile|host_sync"],
                        "share_calls_below_nstar_search_sync": hk["share_calls_below_nstar"]["search|host_sync"],
                        "share_calls_below_nstar_profile_hist8":
                            hk["share_calls_below_nstar"]["profile|sampled_histogram"],
                        "cold_share_warm10": k["cold_start"]["warm_10pct"]["share_later_calls_on_unseen_keys"],
                        "online_sync_beats_model_share_calls":
                            hk["online_beats_model_default"]["host_sync"]["share_calls"]})
    return out


def _fmt(x, pct=False):
    if x is None:
        return "-"
    return f"{100 * x:5.1f}" if pct else (f"{x:.3g}" if isinstance(x, float) else str(x))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "serving_key_churn.json")
    p.add_argument("--hosts", default="pro6000,vgpu3090")
    p.add_argument("--windows", default="all", help="comma list of trace/window, e.g. azure2023_code/1h")
    p.add_argument("--no-sha", action="store_true", help="skip checksums (smoke runs)")
    args = p.parse_args(argv)
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    hosts = [HostCosts(h, args.in_dir) for h in args.hosts.split(",")]
    sel = WINDOWS if args.windows == "all" else [w for w in WINDOWS if f"{w[0]}/{w[1]}" in args.windows.split(",")]
    traces, results, cache = {}, [], {}
    for name, window, policies in sel:
        spec = TRACE_FILES[name]
        path = args.data_dir / spec["file"]
        if name not in cache:
            t0 = time.time()
            cache = {name: load_trace(name, args.data_dir)}       # keep one trace in memory
            if name not in traces:
                traces[name] = {"url": spec["url"], "source": spec["source"], "file": spec["file"],
                                "bytes": path.stat().st_size,
                                "sha256": None if args.no_sha else sha256_of(path),
                                "trace_start": cache[name]["start"], "rows": cache[name]["rows_raw"],
                                "rows_dropped_zero_prompt": cache[name]["rows_dropped_zero_ctx"],
                                "windows": {}}
            log(f"loaded {name}: {traces[name]['rows']} rows, {time.time() - t0:.1f}s")
        info, rows = run_window(name, window, policies, cache[name], hosts, log)
        traces[name]["windows"][window] = info
        for r in rows:
            results.append({"trace": name, "window": window, **r})
    host_meta = {h.host: {"e5b_json": str(h.path.relative_to(REPO_ROOT)), "host_sync_ms": h.sync_ms,
                          "width": h.width, "measured_L_max": h.L_max, "measured_batch_max": h.b_max,
                          "n_measured_keys": len(h.e5b["keys"]),
                          "e5b_summary_n_star": h.e5b["summary_n_star"],
                          "e5b_model_default": h.e5b.get("model_default")} for h in hosts}
    out = {"experiment": "serving_key_churn", "created_utc": utc_now(), "item": "M1b",
           "layers_per_request": LAYERS_PER_REQUEST,
           "config": {"budgets_T": list(BUDGETS), "rho_primary": RHO_PRIMARY,
                      "rho_sensitivity": list(RHO_SENSITIVITY), "warm_fracs": list(WARM_FRACS),
                      "step_time_model": {"t_fix_s": T_FIX_S, "t_prefill_token_s": T_PF_S,
                                          "t_decode_seq_s": T_DEC_S, "max_running_seqs": MAX_SEQS},
                      "key_modes": KEY_MODES, "nstar_combos": NSTAR_COMBOS,
                      "calls_convention": "calls = scheduler steps x 48 layers (E5b: a request contributes 48 calls)",
                      "cost_mapping": "nearest measured E5b bucket_aligned key (|dlog2 bucket|+|dlog2 batch|+"
                                      "1[alignment differs], ties to larger P); packed calls costed as batch-1 "
                                      "scans of the packed length; P, h, regret scaled by max(1,L/L_max)*"
                                      "max(1,b/b_max) beyond the measured range; histogram overheads from E4 at "
                                      "the nearest tensor size",
                      "time_scaling": "arrival times multiplied by (prompt tokens * t_prefill_token)/(duration * "
                                      "rho) so that the offered prefill load is rho of 1/t_prefill_token"},
           "hosts": host_meta, "traces": traces, "results": results,
           "summary": {h.host: summary_rows(results, h.host) for h in hosts}}
    write_json(args.out, out)
    manifest = {"created_utc": out["created_utc"],
                "files": {n: {k: v for k, v in t.items()} for n, t in traces.items()},
                "note": "raw files larger than 50 MB are not tracked in git; re-download from the URLs"}
    if args.windows == "all":
        write_json(args.data_dir / "serving_traces_manifest.json", manifest)
    # compact console summary (first host)
    h0 = hosts[0].host
    log(f"\nwrote {args.out}\nsummary ({h0}; below N* = share of calls on keys where an online policy can win; "
        f"sync = one host synchronization, hist8 = every-8th-element histogram)")
    log(f"{'trace/window':22s} {'policy':16s} {'T':>5s} {'kern':7s} {'mode':15s} {'keys':>7s} {'med c/k':>9s} "
        f"{'<N*prof,sync%':>13s} {'<N*srch,sync%':>13s} {'<N*prof,h8%':>11s} {'cold10%':>8s}")
    for s in out["summary"][h0]:
        if s["policy"] == "chunked_budget" and not (s["sched"] == "colocated" and s["rho"] == RHO_PRIMARY):
            continue
        log(f"{s['trace'] + '/' + s['window']:22s} {s['policy']:16s} {_fmt(s['T']):>5s} {s['kernel']:7s} "
            f"{s['key_mode']:15s} {s['n_keys']:7d} {s['median_calls_per_key']:9.0f} "
            f"{_fmt(s['share_calls_below_nstar_profile_sync'], True):>13s} "
            f"{_fmt(s['share_calls_below_nstar_search_sync'], True):>13s} "
            f"{_fmt(s['share_calls_below_nstar_profile_hist8'], True):>11s} "
            f"{_fmt(s['cold_share_warm10'], True):>8s}")


if __name__ == "__main__":
    main()
