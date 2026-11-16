"""Request-level prefill queueing under trace arrivals: does a small service-time gain of a chunk table grow? (item C2)

E32 reports that a held-out shape-keyed chunk table changes the service time of a whole prefill step by little.
Service time is not response time: in an M/G/1 queue the mean wait has elasticity 2 + rho / (1 - rho) with respect to
service time, so small service gains could grow under load. This script measures how far that happens (CPU only).

Scope. One model at a time on one GPU; a single FCFS server; request-level prefill (one prompt per step at batch 1,
the convention of the E32 workload weighting). Decode, prefix caching, continuous batching, chunked prefill and the
scheduler of a real engine are out of scope, so response time here means prefill queueing plus prefill service.

Service time. Piecewise-linear interpolation between the measured E32 lengths at batch 1 (CUDA-graph replay, RTX 3090),
per variant: the library default chunk, the held-out shape table (per-length-bucket chunk from another board's
single-call sweep), and a session-crossed best table (per measured shape, the best chunk of the other session). Prompts
below the lowest measured length are represented by that length's step (256 tokens today), prompts above 8192 tokens by
the 8192-token step; both shares are reported. ``--short-latencies`` plugs in measured latencies below 256 tokens:
a JSON file {label: {session: {batch: {length: {"c<chunk>": [round medians in ms, ...]}}}}} whose lengths simply become
further knots; the table chunk of a new length comes from the table's own bucket entries.

Arrivals. (a) The trace's own timestamps, dilated per window so that the default's utilization (sum of service times
over the dilated span) is rho; the burstiness of the trace is kept. (b) A Poisson control: exponential gaps at the same
rho with prompts drawn i.i.d. from the window's length mix. All variants of one cell share the arrival sequence.

Intervals. Round-to-round uncertainty: per (length, chunk) the round medians are resampled with replacement and their
median re-taken (the same draw serves every variant using that chunk at that length, so a variant equal to the default
at a length differs by exactly zero there), and the arrivals are re-dilated to pin the default at rho. The same draws
are shared across windows, so the bootstrap over windows can combine both sources. Placebos make timing noise visible:
``aa_rounds`` compares two independent resamples of the default, ``aa_capture`` two single-round curves of the default
(what an engine that captures each graph once could see), ``aa_session`` the default of the other session.

Validation (asserted): the vectorized recursion equals an event-driven FCFS loop, and under Poisson arrivals the
simulated mean wait matches the Pollaczek-Khinchine value within sampling error for every window mix, variant and rho.

Elasticity. ``uniform_*`` scales every service time by 0.99 on fixed arrivals and compares with 2 + rho / (1 - rho) (and
with the exact Pollaczek-Khinchine arc value for that 1% step); ``table_wait_mean`` is the table's own arc elasticity.

Static batching sensitivity. Only batch 1 and 4 were measured, and padding two or three requests to the batch-4 graph
costs more than serving them singly, so a batch-4 step runs when at least four requests of one length bucket are
waiting (padded length at most 2048, i.e., at most 8192 tokens) and the oldest request otherwise runs alone. No
unmeasured shape is used. Windows of at most one day only, since this path is event-driven Python.

Low-load sensitivity. Dilating a bursty trace to a whole-window utilization of 0.3 already drives its bursts above
capacity, so trace cells are also run at rho in {0.05, 0.1, 0.2} (trace arrivals only, not part of the requested grid);
``operating_points`` reports, per window, the highest grid rho at which the default keeps a p99 prefill response below
1, 5 and 30 s, and the table's effect there.

Gains are 100 * (default - variant) / default: positive means the variant is faster than the default.

Output: ``jpdc_load_sim.json`` in the results directory.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import sys
import tempfile
import time
import zlib
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # direct execution (python -I path/to/script.py)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.experiments.tpds.common import RESULTS_DIR, dedupe_by_key, read_jsonl, utc_now, write_json
from src.experiments.tpds.packed_steps import ensure_trace
from src.experiments.tpds.serving_key_churn import DATA_DIR, TRACE_FILES, WINDOWS, load_trace, select_window, sha256_of

LABELS = ("ssd370m", "ssd2p7b", "gdn1p4b")
SESSIONS = ("4c_e32s1", "4c_e32s2")
OTHER_SESSION = {SESSIONS[0]: SESSIONS[1], SESSIONS[1]: SESSIONS[0]}
BATCHES = (1, 4)
RHOS = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
RHOS_LOW = (0.05, 0.1, 0.2)         # trace arrivals only: operating points below the requested grid
RHOS_ALL = RHOS_LOW + RHOS
P99_TARGETS_MS = (1000.0, 5000.0, 30000.0)
METRICS = ("wait_mean", "wait_p99", "resp_mean", "resp_p99")
THRESHOLDS_PCT = (1.0, 2.0, 5.0)
UNIFORM_FACTOR = 0.99
N_SLICES = 100                      # slices of a window's span for the peak-to-mean load factor
STATIC_MAX_TOKENS = 8192
STATIC_WINDOWS = ("azure2023_code|1h", "azure2023_conv|1h", "azure2024_code|1h", "azure2024_conv|1h",
                  "burstgpt|24h_busiest")
JITTER_WINDOWS = ("burstgpt|24h_busiest", "burstgpt|full")     # whole-second timestamps
JITTER_SUFFIX = "+jitter"
FAMILIES = {"all": lambda tw: True,
            "azure": lambda tw: tw.startswith("azure"),
            "burstgpt": lambda tw: tw.startswith("burstgpt"),
            "span_1h": lambda tw: tw.endswith("|1h"),
            "span_day": lambda tw: tw.endswith("|24h") or tw.endswith("|24h_busiest"),
            "span_full": lambda tw: tw.endswith("|full")}
SCOPE = ("Single FCFS server, one model on one GPU, request-level prefill only. Decode, prefix caching, continuous "
         "batching, chunked prefill and the scheduler of a real engine are out of scope.")
LIMITS = [
    "Prefill only on one GPU with a single FCFS queue; a real engine interleaves decode, shares steps and reorders.",
    "Five rounds per (length, chunk); the percentile bootstrap of a median of five values is coarse and tends to "
    "undercover. Each round recaptures the graph, and the shortest steps are bimodal between captures.",
    "Prompts under 256 tokens are represented by the 256-token step until shorter steps are measured; prompts over "
    "8192 tokens by the 8192-token step.",
    "Service time is deterministic given the length; replay-to-replay variation inside a capture is not simulated.",
    "Lengths between measured points are interpolated; token counts come from other models' tokenizers.",
    "Dilation keeps the shape of a trace, including idle spans, so rho is a whole-window mean and a window can spend "
    "long periods above capacity at a moderate rho; requests are never dropped or timed out.",
    "BurstGPT timestamps have one-second resolution, so ties become simultaneous arrivals (see the jitter sensitivity).",
    "The ten windows are nested (1 h and 24 h inside a week, busiest day inside 61 days) and come from five traces, so "
    "the bootstrap over windows is not over independent units; a cluster bootstrap over traces is reported as well.",
    "One board model (stock RTX 3090) and three models; another GPU or engine can change the step latencies.",
]


# --------------------------------------------------------------------------- small helpers
def _g(x, digits: int = 5):
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    return float(f"{x:.{digits}g}")


def _crc(s: str) -> int:
    return zlib.crc32(s.encode("utf-8"))


def gain_pct(base, alt) -> np.ndarray:
    """100 * (base - alt) / base; NaN where the base is zero (e.g., a p99 wait of zero at low load)."""
    base, alt = np.asarray(base, dtype=float), np.asarray(alt, dtype=float)
    out = np.full(np.broadcast(base, alt).shape, np.nan)
    ok = base > 1e-6
    out[ok] = (100.0 * (base - alt) / np.where(ok, base, 1.0))[ok]
    return out


def ci95(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if not len(x):
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def arc_elasticity(base: float, alt: float, factor: float):
    if not (base > 1e-6 and alt > 1e-6) or factor <= 0 or abs(math.log(factor)) < 5e-4:
        return None
    return math.log(alt / base) / math.log(factor)


def pk_point_elasticity(rho: float) -> float:
    return 2.0 + rho / (1.0 - rho)


def pk_arc_elasticity(rho: float, factor: float) -> float:
    """Exact arc elasticity of the M/G/1 mean wait when every service time is multiplied by ``factor``."""
    return math.log(factor ** 2 * (1.0 - rho) / (1.0 - rho * factor)) / math.log(factor)


def pk_wait(lam: float, es: float, es2: float) -> float:
    return lam * es2 / (2.0 * (1.0 - lam * es))


# --------------------------------------------------------------------------- step latencies
def table_chunk(buckets: dict, L: int) -> int:
    """Chunk of a length under a per-length-bucket table (nearest bucket if the exact one is absent), as in E31."""
    b = 1 << max(0, int(math.floor(math.log2(max(L, 1)))))
    bk = {int(k): int(v) for k, v in buckets.items()}
    if b in bk:
        return bk[b]
    return bk[min(bk, key=lambda x: (abs(math.log2(x) - math.log2(b)), x))]


def load_step_data(label: str, in_dir: Path, short_path: str | None = None) -> dict:
    """session -> batch -> round medians of the CUDA-graph replays, one (length, chunk) per array cell."""
    joint = json.loads((in_dir / f"e32_model_graph_{SESSIONS[0]}_{label}_{SESSIONS[1]}_{label}.json").read_text())
    plug = {}
    if short_path:
        plug = json.loads(Path(short_path).read_text()).get(label, {})
    out = {}
    for ses in SESSIONS:
        rows = dedupe_by_key(read_jsonl(in_dir / f"e32_model_graph_{ses}_{label}.jsonl"))
        rows = [r for r in rows if r.get("kind") == "timing" and r["mode"] == "graph" and r["variant"] != "noscan"]
        default = rows[0]["default_chunk"]
        out[ses] = {}
        for B in BATCHES:
            cell: dict = defaultdict(lambda: defaultdict(list))
            for r in sorted((r for r in rows if r["batch"] == B), key=lambda r: r["round"]):
                cell[int(r["L"])][int(r["chunk"])].append(float(r["t"]["median_ms"]))
            chunks = sorted({c for v in cell.values() for c in v})
            for L, per in plug.get(ses, {}).get(str(B), {}).items():
                if any(f"c{c}" not in per for c in chunks):
                    raise ValueError(f"--short-latencies: {label} {ses} B{B} L{L} must give every chunk of {chunks}")
                for c in chunks:
                    cell[int(L)][c] = [float(x) for x in per[f"c{c}"]]
            Ls = sorted(cell)
            max_r = max(len(cell[L][c]) for L in Ls for c in chunks)
            rounds = np.full((len(Ls), len(chunks), max_r), np.nan)
            for i, L in enumerate(Ls):
                for j, c in enumerate(chunks):
                    rounds[i, j, :len(cell[L][c])] = cell[L][c]
            out[ses][B] = {"L": np.array(Ls, dtype=float), "chunks": chunks, "rounds": rounds,
                           "counts": (~np.isnan(rounds)).sum(axis=2), "default": int(default),
                           "buckets": joint["tables"][f"{label}|B{B}"]["buckets"], "table_source":
                           joint["tables"][f"{label}|B{B}"]["source"]}
    return out


class StepArms:
    """Knot latencies of the variants of one (model, session, batch), at the point estimate or a resample."""

    def __init__(self, data: dict, label: str, session: str, batch: int, seed: int):
        d, o = data[session][batch], data[OTHER_SESSION[session]][batch]
        if not np.array_equal(d["L"], o["L"]) or d["chunks"] != o["chunks"]:
            raise ValueError(f"{label} B{batch}: the two sessions must have the same shapes")
        self.label, self.session, self.batch, self.seed = label, session, batch, seed
        self.L, self.chunks, self.rounds, self.counts = d["L"], d["chunks"], d["rounds"], d["counts"]
        self.buckets = d["buckets"]
        self.rows = np.arange(len(self.L))
        self.med = np.nanmedian(self.rounds, axis=2)
        self.avg = np.nanmean(self.rounds, axis=2)
        self.other_med = np.nanmedian(o["rounds"], axis=2)
        self.i_default = self.chunks.index(d["default"])
        self.i_table = np.array([self.chunks.index(table_chunk(self.buckets, int(L))) for L in self.L])
        self.i_best = np.argmin(self.other_med, axis=1)
        self._cache: dict = {}

    def knots(self, arm: str, table: np.ndarray | None = None) -> np.ndarray:
        t = self.med if table is None else table
        if arm == "default":
            return t[self.rows, self.i_default]
        if arm == "table":
            return t[self.rows, self.i_table]
        if arm == "best":
            return t[self.rows, self.i_best]
        raise ValueError(arm)

    def _rng(self, stream: str, b: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, _crc(stream), _crc(self.label), _crc(self.session), self.batch, b])

    def resample(self, b: int, stream: str = "rounds") -> np.ndarray:
        """Median of the round medians resampled with replacement, per (length, chunk)."""
        key = (stream, b)
        if key not in self._cache:
            rng = self._rng(stream, b)
            out = np.empty(self.counts.shape)
            for n in np.unique(self.counts):
                mask = self.counts == n
                vals = self.rounds[mask][:, :n]
                idx = rng.integers(0, n, size=vals.shape)
                out[mask] = np.median(np.take_along_axis(vals, idx, axis=1), axis=1)
            self._cache[key] = out
        return self._cache[key]

    def single_capture(self, b: int, stream: str) -> np.ndarray:
        """One round per (length, chunk): an engine that captures each graph once."""
        rng = self._rng(stream, b)
        idx = np.floor(rng.random(self.counts.shape) * self.counts).astype(np.int64)
        return np.take_along_axis(self.rounds, idx[:, :, None], axis=2)[:, :, 0]

    def table_per_chunk(self, uniq: np.ndarray) -> np.ndarray:
        """Service per length with the table's chunk chosen by the length's own bucket and each chunk's curve
        interpolated, instead of interpolating between the table's knots (sensitivity at the point estimate)."""
        clipped = np.clip(uniq, self.L[0], self.L[-1]).astype(np.int64)
        out = np.empty(len(uniq))
        cu = np.array([table_chunk(self.buckets, int(x)) for x in clipped])
        for c in np.unique(cu):
            m = cu == c
            out[m] = np.interp(uniq[m], self.L, self.med[:, self.chunks.index(int(c))])
        return out


# --------------------------------------------------------------------------- simulators
def lindley(service: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    """FCFS waiting times W[i+1] = max(0, W[i] + S[i] - gap[i]), W[0] = 0, as a running-minimum of the cumulative sum."""
    x = service[:-1] - gaps
    np.cumsum(x, out=x)
    low = np.minimum.accumulate(x)
    np.minimum(low, 0.0, out=low)
    w = np.empty(len(service))
    w[0] = 0.0
    np.subtract(x, low, out=w[1:])
    return w


def metrics_of(wait: np.ndarray, service: np.ndarray) -> np.ndarray:
    r = wait + service
    return np.array([wait.mean(), np.percentile(wait, 99, method="higher"), r.mean(),
                     np.percentile(r, 99, method="higher")])


def simulate_static(arrival: np.ndarray, lengths: np.ndarray, bucket: np.ndarray, s1: np.ndarray, t4: list) -> dict:
    """Event-driven FCFS server: at least four waiting requests of the oldest request's bucket run as one padded batch-4
    step (``t4`` indexed by padded length), otherwise the oldest request runs alone. ``bucket`` < 0 never batches."""
    n = len(arrival)
    arr, ll, bk, s1l = arrival.tolist(), lengths.tolist(), bucket.tolist(), s1.tolist()
    wait, serv = [0.0] * n, [0.0] * n
    served = bytearray(n)
    fifo: deque = deque()
    per: dict = defaultdict(deque)
    i, now, busy, batched, left, last_end = 0, 0.0, 0.0, 0, n, 0.0
    while left:
        while i < n and arr[i] <= now:
            fifo.append(i)
            if bk[i] >= 0:
                per[bk[i]].append(i)
            i += 1
        while fifo and served[fifo[0]]:
            fifo.popleft()
        if not fifo:
            now = arr[i]
            continue
        h = fifo[0]
        q = per[bk[h]] if bk[h] >= 0 else None
        if q is not None and len(q) >= 4:
            members = [q.popleft() for _ in range(4)]
            d = t4[max(ll[m] for m in members)]
            batched += 4
        else:
            members = [h]
            if q is not None:
                q.popleft()
            d = s1l[h]
        for m in members:
            wait[m] = now - arr[m]
            serv[m] = d
            served[m] = 1
        left -= len(members)
        now += d
        busy += d
        last_end = now
    return {"wait": np.array(wait), "service": np.array(serv), "busy": busy, "share_batched": batched / n,
            "end": last_end}


# --------------------------------------------------------------------------- per-process caches
_STEP: dict = {}
_WIN: dict = {}


def step_arms(label: str, session: str, batch: int, cfg: dict) -> StepArms:
    key = (label, session, batch)
    if key not in _STEP:
        if label not in _STEP:
            _STEP[label] = load_step_data(label, Path(cfg["in_dir"]), cfg["short_latencies"])
        _STEP[key] = StepArms(_STEP[label], label, session, batch, cfg["seed"])
    return _STEP[key]


def _win_name(tw: str) -> str:
    return tw.replace("|", "__").replace("+", "_")


def window_arrays(tw: str, cfg: dict) -> dict:
    if tw not in _WIN:
        base = Path(cfg["work_dir"]) / _win_name(tw)
        d = np.load(f"{base}_meta.npz")
        _WIN[tw] = {"uniq": d["uniq"], "counts": d["counts"], "span": float(d["span"]),
                    "inv": np.load(f"{base}_inv.npy", mmap_mode="r"),
                    "gaps": np.load(f"{base}_gaps.npy", mmap_mode="r"),
                    "slice": np.load(f"{base}_slice.npy", mmap_mode="r")}
    return _WIN[tw]


# --------------------------------------------------------------------------- tasks
def trace_task(spec: dict) -> dict:
    """Trace arrivals of one (model, session, window, rho): point arms and round resamples."""
    cfg, label, session, tw, rho = spec["cfg"], spec["label"], spec["session"], spec["tw"], spec["rho"]
    t0 = time.time()
    arms = step_arms(label, session, 1, cfg)
    win = window_arrays(tw, cfg)
    uniq, inv, gaps, span = win["uniq"], np.asarray(win["inv"]), np.asarray(win["gaps"]), win["span"]

    def service(vals: np.ndarray) -> np.ndarray:
        return np.interp(uniq, arms.L, vals)[inv]

    s_def = service(arms.knots("default"))
    dil = float(s_def.sum()) / (rho * span)
    g = gaps * dil
    w_def = lindley(s_def, g)
    point = {"default": metrics_of(w_def, s_def)}
    makespan_ratio = float((span * dil + w_def[-1] + s_def[-1]) / (span * dil))
    share_waiting = float((w_def > 1e-6).mean())
    work = np.bincount(np.asarray(win["slice"]), weights=s_def, minlength=N_SLICES)
    peak_factor = float(work.max() / work.mean())
    del w_def
    mean_service = {"default": float(s_def.mean())}
    for arm in ("table", "best"):
        s = service(arms.knots(arm))
        point[arm] = metrics_of(lindley(s, g), s)
        mean_service[arm] = float(s.mean())
    s = s_def * UNIFORM_FACTOR
    point["uniform"] = metrics_of(lindley(s, g), s)
    s = arms.table_per_chunk(uniq)[inv]
    point["table_per_chunk"] = metrics_of(lindley(s, g), s)
    s = service(arms.other_med[arms.rows, arms.i_default])
    point["aa_session"] = metrics_of(lindley(s, g), s)
    # knots as the mean instead of the median of the rounds, with the default pinned at rho again
    s_m = service(arms.knots("default", arms.avg))
    g_m = gaps * (float(s_m.sum()) / (rho * span))
    point["default_mean"] = metrics_of(lindley(s_m, g_m), s_m)
    for arm in ("table", "best"):
        s = service(arms.knots(arm, arms.avg))
        point[f"{arm}_mean"] = metrics_of(lindley(s, g_m), s)
    del g_m, s_m, s
    # round resamples
    B = spec["reps"]
    reps = {a: np.empty((B, len(METRICS))) for a in ("default", "table", "best", "aa_rounds")}
    for b in range(B):
        med = arms.resample(b)
        s_b = service(arms.knots("default", med))
        g = gaps * (float(s_b.sum()) / (rho * span))
        reps["default"][b] = metrics_of(lindley(s_b, g), s_b)
        for arm in ("table", "best"):
            s = service(arms.knots(arm, med))
            reps[arm][b] = metrics_of(lindley(s, g), s)
        s = service(arms.knots("default", arms.resample(b, "rounds_placebo")))
        reps["aa_rounds"][b] = metrics_of(lindley(s, g), s)
    C = spec["capture_reps"]
    cap = {a: np.empty((C, len(METRICS))) for a in ("a", "b")}
    for b in range(C):
        s_a = service(arms.knots("default", arms.single_capture(b, "capture_a")))
        g = gaps * (float(s_a.sum()) / (rho * span))
        cap["a"][b] = metrics_of(lindley(s_a, g), s_a)
        s = service(arms.knots("default", arms.single_capture(b, "capture_b")))
        cap["b"][b] = metrics_of(lindley(s, g), s)
    return {"kind": "trace", "label": label, "session": session, "tw": tw, "rho": rho, "dilation": dil, "point": point,
            "reps": reps, "capture": cap, "mean_service": mean_service, "makespan_over_span": makespan_ratio,
            "share_waiting": share_waiting, "peak_slice_factor": peak_factor, "seconds": time.time() - t0}


def poisson_task(spec: dict) -> dict:
    """Poisson control with the window's length mix: Pollaczek-Khinchine comparison and round resamples."""
    cfg, label, session, tw, rho = spec["cfg"], spec["label"], spec["session"], spec["tw"], spec["rho"]
    t0 = time.time()
    arms = step_arms(label, session, 1, cfg)
    win = window_arrays(tw, cfg)
    uniq, counts = win["uniq"], win["counts"]
    p = counts / counts.sum()
    cdf = np.cumsum(p)
    cdf[-1] = 1.0

    def curves(table=None) -> dict:
        return {a: np.interp(uniq, arms.L, arms.knots(a, table)) for a in ("default", "table", "best")}

    cv = curves()
    cv["uniform"] = cv["default"] * UNIFORM_FACTOR
    es = {a: float((p * c).sum()) for a, c in cv.items()}
    es2 = {a: float((p * c * c).sum()) for a, c in cv.items()}
    lam = rho / es["default"]
    pk = {a: pk_wait(lam, es[a], es2[a]) for a in cv}
    n, R = cfg["poisson_n"], cfg["poisson_reps"]
    skip = n // 50
    base = [cfg["seed"], _crc("poisson"), _crc(tw), _crc(label), _crc(session), int(round(rho * 100))]
    sims = {a: np.empty((R, len(METRICS))) for a in cv}
    for r in range(R):
        rng = np.random.default_rng(base + [r])
        idx = np.searchsorted(cdf, rng.random(n))
        g = rng.exponential(1.0 / lam, n - 1)
        for a, c in cv.items():
            s = c[idx]
            sims[a][r] = metrics_of(lindley(s, g)[skip:], s[skip:])
    Bp, np_ = cfg["poisson_round_reps"], cfg["poisson_round_n"]
    skip = np_ // 50
    reps = {a: np.empty((Bp, len(METRICS))) for a in ("default", "table", "best")}
    for b in range(Bp):
        cb = curves(arms.resample(b))
        lam_b = rho / float((p * cb["default"]).sum())
        rng = np.random.default_rng(base + [10_000 + b])
        idx = np.searchsorted(cdf, rng.random(np_))
        g = rng.exponential(1.0 / lam_b, np_ - 1)
        for a in reps:
            s = cb[a][idx]
            reps[a][b] = metrics_of(lindley(s, g)[skip:], s[skip:])
    return {"kind": "poisson", "label": label, "session": session, "tw": tw, "rho": rho, "lam_per_ms": lam, "pk": pk,
            "es": es, "es2": es2, "sims": sims, "reps": reps, "seconds": time.time() - t0}


def static_task(spec: dict) -> dict:
    """Static batching sensitivity on one (model, session, window, rho), default pinned at a realized utilization rho."""
    cfg, label, session, tw, rho = spec["cfg"], spec["label"], spec["session"], spec["tw"], spec["rho"]
    t0 = time.time()
    a1, a4 = step_arms(label, session, 1, cfg), step_arms(label, session, 4, cfg)
    win = window_arrays(tw, cfg)
    uniq, inv, gaps, span = win["uniq"], np.asarray(win["inv"]), np.asarray(win["gaps"]), win["span"]
    lo4, hi4 = int(a4.L[0]), int(a4.L[-1])
    if hi4 * 4 > STATIC_MAX_TOKENS:
        raise ValueError("batch-4 knots above the token budget")
    t_rel = np.concatenate(([0.0], np.cumsum(gaps)))
    leff_u = np.maximum(uniq.astype(np.int64), lo4)
    bucket_u = np.where(leff_u <= hi4, np.floor(np.log2(leff_u)).astype(np.int64), -1)
    lengths, bucket = leff_u[inv], bucket_u[inv]
    grid = np.arange(hi4 + 1, dtype=float)

    def t4_of(vals: np.ndarray) -> list:
        return np.interp(grid, a4.L, vals).tolist()

    def run(arm: str, dil: float, med1=None, med4=None) -> dict:
        s1 = np.interp(uniq, a1.L, a1.knots(arm, med1))[inv]
        return simulate_static(t_rel * dil, lengths, bucket, s1, t4_of(a4.knots(arm, med4)))

    s1_def = np.interp(uniq, a1.L, a1.knots("default"))[inv]
    es_def = float(s1_def.mean())
    dil = float(s1_def.sum()) / (rho * span)
    for _ in range(8):      # fixed point of busy(dil) / (dil * span) = rho
        res = run("default", dil)
        new = res["busy"] / (rho * span)
        converged = abs(new / dil - 1.0) < 1e-3
        dil = new
        if converged:
            break
    point, info = {}, {}
    for arm in ("default", "table", "best"):
        res = run(arm, dil)
        point[arm] = metrics_of(res["wait"], res["service"])
        info[arm] = {"share_batched": res["share_batched"], "busy_fraction": res["busy"] / (span * dil),
                     "makespan_over_span": res["end"] / (span * dil)}
    B = spec["reps"]
    reps = {a: np.empty((B, len(METRICS))) for a in ("default", "table", "best")}
    for b in range(B):
        m1, m4 = a1.resample(b), a4.resample(b)
        d_b = dil * float(np.interp(uniq, a1.L, a1.knots("default", m1))[inv].mean()) / es_def
        for arm in reps:
            res = run(arm, d_b, m1, m4)
            reps[arm][b] = metrics_of(res["wait"], res["service"])
    return {"kind": "static", "label": label, "session": session, "tw": tw, "rho": rho, "dilation": dil, "point": point,
            "info": info, "reps": reps, "seconds": time.time() - t0}


def run_task(spec: dict) -> dict:
    return {"trace": trace_task, "poisson": poisson_task, "static": static_task}[spec["kind"]](spec)


# --------------------------------------------------------------------------- checks
def check_e32_reproduction(step: dict, in_dir: Path) -> dict:
    """The arrays read here reproduce the stored analysis: medians, table chunks, best chunks and the 14-shape gains."""
    out = {}
    for label in LABELS:
        joint = json.loads((in_dir / f"e32_model_graph_{SESSIONS[0]}_{label}_{SESSIONS[1]}_{label}.json").read_text())
        out[label] = {}
        for ses in SESSIONS:
            tag = f"{ses}_{label}"
            cells = [c for c in joint["cells"] if c["session"] == tag and c["mode"] == "graph"]
            total_d = total_t = 0.0
            for c in cells:
                B, L = c["batch"], c["L"]
                d = step[label][ses][B]
                if int(L) not in d["L"].astype(int):
                    raise AssertionError(f"{tag}: shape L={L} B={B} missing")
                i = int(np.where(d["L"] == L)[0][0])
                med = np.nanmedian(d["rounds"][i], axis=1)
                for j, ch in enumerate(d["chunks"]):
                    assert abs(med[j] - c["median_ms"][f"c{ch}"]) <= 1e-4 * med[j] + 5e-4, (tag, L, B, ch)
                assert table_chunk(d["buckets"], int(L)) == c["table_chunk"], (tag, L, B)
                assert f"c{d['chunks'][int(np.argmin(med))]}" == c["best"], (tag, L, B)
                total_d += med[d["chunks"].index(d["default"])]
                total_t += med[d["chunks"].index(c["table_chunk"])]
            g = 100.0 * (total_d - total_t) / total_d
            ref = joint["summary"][label]["graph"][tag]["gain_table_latency_weighted_pct"]
            assert abs(g - ref) < 5e-3, (tag, g, ref)
            out[label][ses] = {"n_shapes": len(cells), "gain_table_latency_weighted_pct": _g(g),
                               "stored_analysis_pct": ref}
    return out


def check_event_loop(seed: int) -> dict:
    """The vectorized recursion equals an event-driven FCFS loop on synthetic random inputs, including ties."""
    rng = np.random.default_rng([seed, _crc("event_loop")])
    worst = 0.0
    for _ in range(5):
        n = 20_000
        s = rng.lognormal(2.0, 1.0, n)
        gaps = rng.exponential(float(s.mean()) / rng.uniform(0.3, 1.1), n - 1)
        gaps[rng.random(n - 1) < 0.2] = 0.0       # simultaneous arrivals
        t = np.concatenate(([0.0], np.cumsum(gaps)))
        res = simulate_static(t, np.full(n, 256), np.full(n, -1), s, [0.0])
        worst = max(worst, float(np.abs(res["wait"] - lindley(s, gaps)).max()))
    assert worst < 1e-7, f"event loop differs from the vectorized recursion by {worst}"
    return {"max_abs_diff_ms": worst, "replications": 5, "n_per_replication": 20_000}


def check_pollaczek_khinchine(poisson: list[dict], family_alpha: float = 0.001) -> dict:
    """Simulated Poisson mean wait against the analytic value; replications give the sampling error."""
    from scipy import stats

    rows = []
    for r in poisson:
        for arm in ("default", "table", "best", "uniform"):
            x = r["sims"][arm][:, 0]
            se = float(x.std(ddof=1) / math.sqrt(len(x)))
            rows.append({"label": r["label"], "session": r["session"], "tw": r["tw"], "rho": r["rho"], "arm": arm,
                         "sim_ms": float(x.mean()), "pk_ms": r["pk"][arm], "se_ms": se,
                         "t": (float(x.mean()) - r["pk"][arm]) / se})
    t_vals = np.array([x["t"] for x in rows])
    reps = len(poisson[0]["sims"]["default"])
    crit = float(stats.t.ppf(1.0 - family_alpha / (2 * len(rows)), reps - 1))
    worst = rows[int(np.argmax(np.abs(t_vals)))]
    rel = np.array([100.0 * (x["sim_ms"] / x["pk_ms"] - 1.0) for x in rows])
    rho_arr = np.array([x["rho"] for x in rows])
    per_rho = {f"{rho}": {"max_abs_rel_err_pct": _g(np.abs(rel[rho_arr == rho]).max()),
                          "mean_rel_err_pct": _g(rel[rho_arr == rho].mean()),
                          "rms_t": _g(np.sqrt((t_vals[rho_arr == rho] ** 2).mean()))} for rho in sorted(set(rho_arr))}
    out = {"n_tests": len(rows), "replications": reps, "family_alpha": family_alpha, "critical_t": _g(crit),
           "max_abs_t": _g(float(np.abs(t_vals).max())), "worst": {k: (_g(v) if isinstance(v, float) else v)
                                                                  for k, v in worst.items()},
           "mean_t": _g(float(t_vals.mean())), "rms_t": _g(float(np.sqrt((t_vals ** 2).mean()))),
           "share_abs_t_above_2": _g(float((np.abs(t_vals) > 2).mean())), "per_rho": per_rho, "passed": True}
    assert float(np.abs(t_vals).max()) < crit, f"Pollaczek-Khinchine check failed: {worst} (critical t {crit})"
    assert abs(float(t_vals.mean())) < 0.75, f"Pollaczek-Khinchine check: systematic bias, mean t {t_vals.mean()}"
    assert float(np.sqrt((t_vals ** 2).mean())) < 1.5, "Pollaczek-Khinchine check: sampling errors are understated"
    return out


# --------------------------------------------------------------------------- data preparation
def locate_trace(name: str, dirs: list[Path]) -> Path | None:
    for d in dirs:
        if (d / TRACE_FILES[name]["file"]).exists():
            return d
    return None


def prepare_windows(selected: list[tuple[str, str]], dirs: list[Path], in_dir: Path, work_dir: Path, seed: int,
                    log) -> tuple[dict, dict, list[str]]:
    """Per window: gaps, length index, length mix and slice index as files that the worker processes map read-only."""
    recorded = json.loads((in_dir / "serving_key_churn.json").read_text())["traces"]
    traces, windows, ready = {}, {}, []
    by_trace = defaultdict(list)
    for name, window in selected:
        by_trace[name].append(window)
    for name, wins in by_trace.items():
        spec = TRACE_FILES[name]
        d = locate_trace(name, dirs)
        if d is None:
            try:
                log(f"{name}: not found locally, downloading {spec['url']}")
                ensure_trace(name, dirs[0])
                d = dirs[0]
            except Exception as exc:  # a missing trace is reported, the others still run
                traces[name] = {"status": "missing", "url": spec["url"], "error": repr(exc)[:300]}
                log(f"{name}: unavailable ({exc!r})")
                continue
        path = d / spec["file"]
        t0 = time.time()
        sha = sha256_of(path)
        tr = load_trace(name, d)
        traces[name] = {"status": "ok", "file": spec["file"], "url": spec["url"], "source": spec["source"],
                        "bytes": path.stat().st_size, "sha256": sha, "rows": tr["rows_raw"],
                        "sha256_matches_serving_replay": sha == recorded.get(name, {}).get("sha256")}
        log(f"loaded {name}: {tr['rows_raw']} rows, {time.time() - t0:.1f}s")
        for window in wins:
            win, info = select_window(tr, window)
            tw = f"{name}|{window}"
            windows[tw] = _save_window(tw, win["t"], win["ctx"], work_dir, info)
            ready.append(tw)
            if tw in JITTER_WINDOWS:
                rng = np.random.default_rng([seed, _crc("jitter"), _crc(tw)])
                t_j = win["t"] + rng.random(len(win["t"]))
                order = np.argsort(t_j, kind="stable")
                tw_j = tw + JITTER_SUFFIX
                windows[tw_j] = _save_window(tw_j, t_j[order], win["ctx"][order], work_dir, info)
                ready.append(tw_j)
        del tr
    return traces, windows, ready


def _save_window(tw: str, t: np.ndarray, ctx: np.ndarray, work_dir: Path, info: dict) -> dict:
    base = work_dir / _win_name(tw)
    gaps = np.diff(t)
    span = float(t[-1] - t[0])
    uniq, inv, counts = np.unique(ctx, return_inverse=True, return_counts=True)
    slc = np.minimum(((t - t[0]) / span * N_SLICES).astype(np.int64), N_SLICES - 1).astype(np.uint8)
    np.save(f"{base}_gaps.npy", gaps)
    np.save(f"{base}_inv.npy", inv.astype(np.int32))
    np.save(f"{base}_slice.npy", slc)
    np.savez(f"{base}_meta.npz", uniq=uniq, counts=counts, span=span)
    rate = np.bincount(slc, minlength=N_SLICES)
    return {"selection": info["selection"], "n_requests": int(len(t)), "span_s": _g(span), "requests_per_s": _g(len(t) / span),
            "interarrival_ca2": _g(float(gaps.var() / gaps.mean() ** 2)), "share_zero_gaps": _g(float((gaps == 0).mean())),
            "peak_slice_factor_requests": _g(float(rate.max() / rate.mean())),
            "prompt_tokens_mean": _g(float(ctx.mean())), "prompt_tokens_max": int(ctx.max()),
            "share_below_256": _g(float((ctx < 256).mean())), "share_above_8192": _g(float((ctx > 8192).mean())),
            "n_above_8192": int((ctx > 8192).sum())}


# --------------------------------------------------------------------------- aggregation
def gain_entry(base_pt, arm_pt, base_reps, arm_reps) -> dict:
    out = {}
    for i, m in enumerate(METRICS):
        lo, hi = ci95(gain_pct(base_reps[:, i], arm_reps[:, i]))
        out[m] = {"gain_pct": _g(float(gain_pct(base_pt[i], arm_pt[i]))), "ci95_rounds": [_g(lo), _g(hi)],
                  "resolved_from_zero": bool(lo > 0 or hi < 0)}
    return out


def point_gains(base_pt, arm_pt) -> dict:
    return {m: _g(float(gain_pct(base_pt[i], arm_pt[i]))) for i, m in enumerate(METRICS)}


def placebo_entry(base_reps, arm_reps) -> dict:
    out = {}
    for i, m in enumerate(METRICS):
        g = gain_pct(base_reps[:, i], arm_reps[:, i])
        lo, hi = ci95(g)
        out[m] = {"ci95": [_g(lo), _g(hi)], "median_abs_pct": _g(float(np.nanmedian(np.abs(g)))) if (~np.isnan(g)).any()
                  else None}
    return out


def metric_dict(values) -> dict:
    return {m: _g(float(values[i])) for i, m in enumerate(METRICS)}


def bootstrap_family(point: np.ndarray, reps: np.ndarray, clusters: list[str] | None, seed_seq: list[int],
                     draws: int = 4000) -> dict:
    """Family mean over windows and its intervals: rounds only, windows only, and both.

    ``point`` [windows]; ``reps`` [windows, shared round draws].
    """
    rng = np.random.default_rng(seed_seq)
    n = len(point)
    picks = rng.integers(0, n, size=(draws, n))
    bs = rng.integers(0, reps.shape[1], size=draws)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rounds_only = np.nanmean(reps, axis=0)
            windows_only = np.nanmean(point[picks], axis=1)
            both = np.nanmean(reps[picks, bs[:, None]], axis=1)
            out = {"mean_gain_pct": _g(float(np.nanmean(point))) if (~np.isnan(point)).any() else None,
                   "min_gain_pct": _g(float(np.nanmin(point))) if (~np.isnan(point)).any() else None,
                   "max_gain_pct": _g(float(np.nanmax(point))) if (~np.isnan(point)).any() else None,
                   "n_windows": n, "ci95_rounds": [_g(x) for x in ci95(rounds_only)],
                   "ci95_windows": [_g(x) for x in ci95(windows_only)],
                   "ci95_windows_rounds": [_g(x) for x in ci95(both)]}
            if clusters is not None:
                names = sorted(set(clusters))
                by = {c: [i for i, x in enumerate(clusters) if x == c] for c in names}
                m = np.stack([np.nanmean(reps[by[c]], axis=0) for c in names])
                pk = rng.integers(0, len(names), size=(draws, len(names)))
                out["ci95_traces_rounds"] = [_g(x) for x in ci95(np.nanmean(m[pk, bs[:, None]], axis=1))]
    return out


def first_crossings(rows: list[dict], thr: float, ci_key: str) -> dict:
    first_point = next((r for r in rows if r["gain"] is not None and abs(r["gain"]) >= thr), None)
    first_clear = next((r for r in rows if r[ci_key][0] is not None and (r[ci_key][0] >= thr or r[ci_key][1] <= -thr)),
                       None)
    return {"rhos_exceeding_point": [r["rho"] for r in rows if r["gain"] is not None and abs(r["gain"]) >= thr],
            "first_rho_point": first_point["rho"] if first_point else None,
            "gain_pct_at_first_rho_point": first_point["gain"] if first_point else None,
            "ci95_at_first_rho_point": first_point[ci_key] if first_point else None,
            "first_rho_ci_beyond_threshold": first_clear["rho"] if first_clear else None,
            "gain_pct_at_first_rho_ci_beyond": first_clear["gain"] if first_clear else None,
            "ci95_at_first_rho_ci_beyond": first_clear[ci_key] if first_clear else None}


def _sign(x) -> int:
    return 0 if x is None or x == 0 else (1 if x > 0 else -1)


def aggregate(raw: list[dict], args, step: dict, traces: dict, windows: dict, primary: list[str], jitter: list[str],
              log) -> dict:
    tr = {(r["label"], r["session"], r["tw"], r["rho"]): r for r in raw if r["kind"] == "trace"}
    po = {(r["label"], r["session"], r["tw"], r["rho"]): r for r in raw if r["kind"] == "poisson"}
    st = {(r["label"], r["session"], r["tw"], r["rho"]): r for r in raw if r["kind"] == "static"}
    checks = {"e32_reproduction": check_e32_reproduction(step, args.in_dir), "event_loop": check_event_loop(args.seed),
              "pollaczek_khinchine": check_pollaczek_khinchine(list(po.values()))}
    log("checks passed:", checks["pollaczek_khinchine"]["max_abs_t"], "<", checks["pollaczek_khinchine"]["critical_t"])
    cells: dict = {}
    for (lab, ses, tw, rho), r in tr.items():
        p, reps = r["point"], r["reps"]
        fac = r["mean_service"]["table"] / r["mean_service"]["default"]
        rec = {"dilation_factor": _g(r["dilation"] / 1000.0), "default": metric_dict(p["default"]),
               "makespan_over_span": _g(r["makespan_over_span"]), "share_requests_waiting": _g(r["share_waiting"]),
               "peak_slice_factor": _g(r["peak_slice_factor"]), "peak_slice_rho": _g(rho * r["peak_slice_factor"]),
               "overloaded": bool(r["makespan_over_span"] > 1.05),
               "rho_table": _g(rho * fac), "n_round_replicates": int(len(reps["default"])),
               "table": gain_entry(p["default"], p["table"], reps["default"], reps["table"]),
               "best": gain_entry(p["default"], p["best"], reps["default"], reps["best"]),
               "placebo_rounds": placebo_entry(reps["default"], reps["aa_rounds"]),
               "placebo_session_gain_pct": point_gains(p["default"], p["aa_session"]),
               "table_per_chunk_gain_pct": point_gains(p["default"], p["table_per_chunk"]),
               "knot_mean_of_rounds": {"table_gain_pct": point_gains(p["default_mean"], p["table_mean"]),
                                       "best_gain_pct": point_gains(p["default_mean"], p["best_mean"])}}
        if len(r["capture"]["a"]):
            rec["placebo_capture"] = placebo_entry(r["capture"]["a"], r["capture"]["b"])
        rec["elasticity"] = {"pk_point": _g(pk_point_elasticity(rho)), "pk_arc_1pct": _g(pk_arc_elasticity(rho, UNIFORM_FACTOR))}
        for i, m in enumerate(METRICS):
            rec["elasticity"][f"uniform_{m}"] = _g(arc_elasticity(p["default"][i], p["uniform"][i], UNIFORM_FACTOR))
        rec["elasticity"]["table_wait_mean"] = _g(arc_elasticity(p["default"][0], p["table"][0], fac)) \
            if abs(fac - 1.0) >= 5e-4 else None
        cells.setdefault(lab, {}).setdefault(ses, {}).setdefault(tw, {}).setdefault(f"{rho}", {})["trace"] = rec
    for (lab, ses, tw, rho), r in po.items():
        sims = {a: r["sims"][a].mean(axis=0) for a in r["sims"]}
        pk_g = {a: _g(float(gain_pct(r["pk"]["default"], r["pk"][a]))) for a in ("table", "best")}
        es_d = r["es"]["default"]
        pk_resp = {a: r["pk"][a] + r["es"][a] for a in r["pk"]}
        rec = {"lam_per_ms": _g(r["lam_per_ms"]), "default": metric_dict(sims["default"]),
               "pk_wait_mean_ms": {a: _g(r["pk"][a]) for a in r["pk"]},
               "sim_over_pk_wait_mean": {a: _g(sims[a][0] / r["pk"][a]) for a in r["pk"]},
               "rho_table": _g(rho * r["es"]["table"] / es_d),
               "pk_gain_pct": {"wait_mean": pk_g, "resp_mean": {a: _g(float(gain_pct(pk_resp["default"], pk_resp[a])))
                                                                for a in ("table", "best")}},
               "table": gain_entry(sims["default"], sims["table"], r["reps"]["default"], r["reps"]["table"]),
               "best": gain_entry(sims["default"], sims["best"], r["reps"]["default"], r["reps"]["best"]),
               "n_round_replicates": int(len(r["reps"]["default"])),
               "elasticity": {"pk_point": _g(pk_point_elasticity(rho)), "pk_arc_1pct": _g(pk_arc_elasticity(rho, UNIFORM_FACTOR))}}
        for i, m in enumerate(METRICS):
            rec["elasticity"][f"uniform_{m}"] = _g(arc_elasticity(sims["default"][i], sims["uniform"][i], UNIFORM_FACTOR))
        cells.setdefault(lab, {}).setdefault(ses, {}).setdefault(tw, {}).setdefault(f"{rho}", {})["poisson"] = rec

    # service-time gains over each window's length mix, with resampled rounds, independent of the arrivals
    service: dict = {}
    for lab in LABELS:
        for ses in SESSIONS:
            arms = StepArms(step[lab], lab, ses, 1, args.seed)
            for tw in primary:
                base = args.work_dir / _win_name(tw)
                d = np.load(f"{base}_meta.npz")
                uniq, p = d["uniq"], d["counts"] / d["counts"].sum()

                def cv(a, t=None):
                    return np.interp(uniq, arms.L, arms.knots(a, t))

                sd, stb, sb = cv("default"), cv("table"), cv("best")
                es = float((p * sd).sum())
                lowest = uniq <= arms.L[0]
                gain_tot = float((p * (sd - stb)).sum())
                B = args.reps
                gt, gb = np.empty(B), np.empty(B)
                for b in range(B):
                    m = arms.resample(b)
                    x = cv("default", m)
                    gt[b] = 100.0 * float((p * (x - cv("table", m))).sum()) / float((p * x).sum())
                    gb[b] = 100.0 * float((p * (x - cv("best", m))).sum()) / float((p * x).sum())
                s_avg = cv("default", arms.avg)
                service.setdefault(lab, {}).setdefault(ses, {})[tw] = {
                    "mean_service_ms": {"default": _g(es), "table": _g((p * stb).sum()), "best": _g((p * sb).sum())},
                    "service_cv2_default": _g((p * sd * sd).sum() / es ** 2 - 1.0),
                    "table_gain_pct": _g(100.0 * gain_tot / es), "table_gain_ci95_rounds": [_g(x) for x in ci95(gt)],
                    "best_gain_pct": _g(100.0 * (p * (sd - sb)).sum() / es), "best_gain_ci95_rounds": [_g(x) for x in ci95(gb)],
                    "table_gain_pct_mean_of_rounds": _g(100.0 * (p * (s_avg - cv("table", arms.avg))).sum() / (p * s_avg).sum()),
                    "table_gain_pct_per_chunk_interpolation": _g(100.0 * (p * (sd - arms.table_per_chunk(uniq))).sum() / es),
                    "share_requests_at_lowest_knot": _g(p[lowest].sum()), "share_work_at_lowest_knot": _g((p * sd)[lowest].sum() / es),
                    "share_table_gain_from_lowest_knot": _g((p * (sd - stb))[lowest].sum() / gain_tot) if abs(gain_tot) > 1e-9 else None}

    # summaries per family of windows
    summary: dict = {}
    per_cell: dict = {}
    for lab in LABELS:
        for ses in SESSIONS:
            for rho in RHOS_ALL:
                for arr, source in (("trace", tr), ("poisson", po)):
                    avail = [tw for tw in primary if (lab, ses, tw, rho) in source]
                    if not avail:
                        continue
                    for arm in ("table", "best"):
                        for mi, m in enumerate(METRICS):
                            pts, reps_l = [], []
                            for tw in avail:
                                r = source[(lab, ses, tw, rho)]
                                if arr == "trace":
                                    pts.append(float(gain_pct(r["point"]["default"][mi], r["point"][arm][mi])))
                                else:
                                    pts.append(float(gain_pct(r["sims"]["default"].mean(axis=0)[mi],
                                                              r["sims"][arm].mean(axis=0)[mi])))
                                reps_l.append(gain_pct(r["reps"]["default"][:, mi], r["reps"][arm][:, mi]))
                            bmax = max(len(x) for x in reps_l)
                            reps_m = np.stack([x[np.arange(bmax) % len(x)] for x in reps_l])
                            pts_a = np.array(pts)
                            per_cell[(lab, ses, arr, arm, m, rho)] = (avail, pts_a, reps_m)
                            for fam, pred in FAMILIES.items():
                                keep = [i for i, tw in enumerate(avail) if pred(tw)]
                                if not keep:
                                    continue
                                seeds = [args.seed, _crc("family"), _crc(lab), _crc(ses), _crc(arr), _crc(arm), _crc(m),
                                         _crc(fam), int(round(rho * 100))]
                                clusters = [avail[i].split("|")[0] for i in keep] if fam == "all" else None
                                summary.setdefault(lab, {}).setdefault(arr, {}).setdefault(fam, {}).setdefault(
                                    f"{rho}", {}).setdefault(ses, {}).setdefault(arm, {})[m] = bootstrap_family(
                                        pts_a[keep], reps_m[keep], clusters, seeds)
    # placebo summary over windows (window mean of the absolute placebo and its size relative to the table effect)
    placebo: dict = {}
    for lab in LABELS:
        for ses in SESSIONS:
            for rho in RHOS_ALL:
                avail = [tw for tw in primary if (lab, ses, tw, rho) in tr]
                for fam, pred in FAMILIES.items():
                    keep = [tw for tw in avail if pred(tw)]
                    if not keep:
                        continue
                    rec = {}
                    for i, m in enumerate(METRICS):
                        rr = [gain_pct(tr[(lab, ses, tw, rho)]["reps"]["default"][:, i],
                                       tr[(lab, ses, tw, rho)]["reps"]["aa_rounds"][:, i]) for tw in keep]
                        caps = [gain_pct(tr[(lab, ses, tw, rho)]["capture"]["a"][:, i],
                                         tr[(lab, ses, tw, rho)]["capture"]["b"][:, i]) for tw in keep
                                if len(tr[(lab, ses, tw, rho)]["capture"]["a"])]
                        sess = [float(gain_pct(tr[(lab, ses, tw, rho)]["point"]["default"][i],
                                               tr[(lab, ses, tw, rho)]["point"]["aa_session"][i])) for tw in keep]
                        rec[m] = {"rounds_median_abs_pct": _g(float(np.nanmedian(np.abs(np.concatenate(rr))))),
                                  "rounds_ci95": [_g(x) for x in ci95(np.concatenate(rr))],
                                  "capture_median_abs_pct": _g(float(np.nanmedian(np.abs(np.concatenate(caps))))) if caps else None,
                                  "capture_ci95": [_g(x) for x in ci95(np.concatenate(caps))] if caps else None,
                                  "session_mean_gain_pct": _g(float(np.nanmean(sess))),
                                  "session_mean_abs_pct": _g(float(np.nanmean(np.abs(sess))))}
                    placebo.setdefault(lab, {}).setdefault(fam, {}).setdefault(f"{rho}", {})[ses] = rec

    # thresholds: family means (with both intervals) and per window, and sign agreement of the sessions
    thresholds: dict = {}
    thresholds_window: dict = {}
    sign_agreement: dict = {}
    for lab in LABELS:
        for arr in ("trace", "poisson"):
            for fam in FAMILIES:
                if fam not in summary[lab].get(arr, {}):
                    continue
                for m in ("resp_mean", "resp_p99", "wait_mean", "wait_p99"):
                    for arm in ("table", "best"):
                        for thr in THRESHOLDS_PCT:
                            rec = {}
                            for ses in SESSIONS:
                                rows = [{"rho": rho, "gain": summary[lab][arr][fam][f"{rho}"][ses][arm][m]["mean_gain_pct"],
                                         "ci": summary[lab][arr][fam][f"{rho}"][ses][arm][m]["ci95_windows_rounds"]}
                                        for rho in RHOS_ALL if f"{rho}" in summary[lab][arr][fam]]
                                rec[ses] = first_crossings(rows, thr, "ci")
                            a, b = rec[SESSIONS[0]], rec[SESSIONS[1]]
                            agree = None
                            if a["first_rho_point"] is not None and b["first_rho_point"] is not None:
                                agree = _sign(a["gain_pct_at_first_rho_point"]) == _sign(b["gain_pct_at_first_rho_point"])
                            rec["sign_agree_at_first_rho"] = agree
                            thresholds.setdefault(lab, {}).setdefault(arr, {}).setdefault(fam, {}).setdefault(m, {}).setdefault(
                                arm, {})[f"{thr:g}pct"] = rec
        for arr, source in (("trace", tr), ("poisson", po)):
            for m in ("resp_mean", "resp_p99"):
                mi = METRICS.index(m)
                for arm in ("table", "best"):
                    for rho in RHOS:
                        tws = [tw for tw in primary if (lab, SESSIONS[0], tw, rho) in source and (lab, SESSIONS[1], tw, rho) in source]
                        pts, cis = {}, {}
                        for ses in SESSIONS:
                            pts[ses], cis[ses] = [], []
                            for tw in tws:
                                ce = cells[lab][ses][tw][f"{rho}"][arr][arm][m]
                                pts[ses].append(ce["gain_pct"])
                                cis[ses].append(ce["ci95_rounds"])
                        same = [_sign(x) == _sign(y) for x, y in zip(pts[SESSIONS[0]], pts[SESSIONS[1]])]
                        res = [[(c[0] is not None and (c[0] > 0 or c[1] < 0)) for c in cis[s]] for s in SESSIONS]
                        both = [x and y for x, y in zip(*res)]
                        fam_means = {s: summary[lab][arr]["all"][f"{rho}"][s][arm][m]["mean_gain_pct"] for s in SESSIONS}
                        sign_agreement.setdefault(lab, {}).setdefault(arr, {}).setdefault(m, {}).setdefault(arm, {})[f"{rho}"] = {
                            "n_windows": len(tws), "n_windows_sign_agree": int(sum(same)),
                            "n_windows_both_resolved": int(sum(both)),
                            "n_windows_both_resolved_sign_agree": int(sum(s and o for s, o in zip(same, both))),
                            "window_mean_gain_pct": fam_means,
                            "window_mean_sign_agree": _sign(fam_means[SESSIONS[0]]) == _sign(fam_means[SESSIONS[1]])}
                    for tw in primary:
                        for ses in SESSIONS:
                            rows = []
                            for rho in RHOS_ALL:
                                ce = cells.get(lab, {}).get(ses, {}).get(tw, {}).get(f"{rho}", {}).get(arr)
                                if ce:
                                    rows.append({"rho": rho, "gain": ce[arm][m]["gain_pct"], "ci": ce[arm][m]["ci95_rounds"]})
                            if rows:
                                thresholds_window.setdefault(lab, {}).setdefault(arr, {}).setdefault(tw, {}).setdefault(
                                    m, {}).setdefault(arm, {})[ses] = {f"{thr:g}pct": first_crossings(rows, thr, "ci")
                                                                       for thr in THRESHOLDS_PCT}

    # elasticity against the M/G/1 prediction, per window and rho over models and sessions
    elasticity: dict = {"definition": "arc elasticity of the metric when every service time is scaled by 0.99 on fixed "
                                      "arrivals; pk_point = 2 + rho/(1-rho), pk_arc_1pct = the exact P-K value of that step",
                        "by_window": {}, "diverging_over_25pct": []}
    for tw in primary + jitter:
        for rho in RHOS_ALL:
            rec = {"pk_point": _g(pk_point_elasticity(rho)), "pk_arc_1pct": _g(pk_arc_elasticity(rho, UNIFORM_FACTOR))}
            for arr in ("trace", "poisson"):
                for m in METRICS:
                    vals = [cells[lab][ses][tw][f"{rho}"][arr]["elasticity"][f"uniform_{m}"] for lab in LABELS for ses in SESSIONS
                            if arr in cells.get(lab, {}).get(ses, {}).get(tw, {}).get(f"{rho}", {})]
                    vals = [v for v in vals if v is not None]
                    if vals:
                        rec[f"{arr}_{m}"] = {"mean": _g(float(np.mean(vals))), "min": _g(min(vals)), "max": _g(max(vals))}
            elasticity["by_window"].setdefault(tw, {})[f"{rho}"] = rec
            if "trace_wait_mean" in rec:
                ratio = rec["trace_wait_mean"]["mean"] / rec["pk_arc_1pct"]
                rec["trace_wait_mean_over_pk_arc"] = _g(ratio)
                if abs(ratio - 1.0) > 0.25 and not tw.endswith(JITTER_SUFFIX):
                    elasticity["diverging_over_25pct"].append({"tw": tw, "rho": rho, "ratio": _g(ratio),
                                                               "observed": rec["trace_wait_mean"]["mean"],
                                                               "pk_arc_1pct": rec["pk_arc_1pct"]})
    elasticity["table_vs_uniform_note"] = ("cells.<label>.<session>.<tw>.<rho>.trace.elasticity.table_wait_mean is the "
                                           "table's own arc elasticity; it is null when the table changes the mean "
                                           "service time by less than 0.05%")

    operating: dict = {}
    for lab in LABELS:
        for ses in SESSIONS:
            for tw in primary:
                for target in P99_TARGETS_MS:
                    ok = [rho for rho in RHOS_ALL if cells[lab][ses][tw][f"{rho}"]["trace"]["default"]["resp_p99"] <= target]
                    rec = {"highest_rho": max(ok) if ok else None}
                    if ok:
                        c = cells[lab][ses][tw][f"{max(ok)}"]["trace"]
                        rec.update({"default_resp_mean_ms": c["default"]["resp_mean"], "default_resp_p99_ms": c["default"]["resp_p99"],
                                    "table_resp_mean": c["table"]["resp_mean"], "table_resp_p99": c["table"]["resp_p99"],
                                    "placebo_session_resp_mean_gain_pct": c["placebo_session_gain_pct"]["resp_mean"]})
                    operating.setdefault(lab, {}).setdefault(ses, {}).setdefault(tw, {})[f"p99_le_{target / 1000:g}s"] = rec

    jitter_out: dict = {}
    for tw in jitter:
        orig = tw[: -len(JITTER_SUFFIX)]
        for lab in LABELS:
            for ses in SESSIONS:
                for rho in RHOS_ALL:
                    if (lab, ses, tw, rho) not in tr or (lab, ses, orig, rho) not in tr:
                        continue
                    c, o = cells[lab][ses][tw][f"{rho}"]["trace"], cells[lab][ses][orig][f"{rho}"]["trace"]
                    jitter_out.setdefault(orig, {}).setdefault(lab, {}).setdefault(ses, {})[f"{rho}"] = {
                        "default_wait_mean_ms": {"ties": o["default"]["wait_mean"], "jitter": c["default"]["wait_mean"]},
                        "table_resp_mean_gain_pct": {"ties": o["table"]["resp_mean"]["gain_pct"],
                                                     "jitter": c["table"]["resp_mean"]["gain_pct"],
                                                     "jitter_ci95_rounds": c["table"]["resp_mean"]["ci95_rounds"]},
                        "table_resp_p99_gain_pct": {"ties": o["table"]["resp_p99"]["gain_pct"],
                                                    "jitter": c["table"]["resp_p99"]["gain_pct"],
                                                    "jitter_ci95_rounds": c["table"]["resp_p99"]["ci95_rounds"]}}

    static: dict = {"policy": "A padded batch-4 step runs when at least four requests of the oldest request's length "
                              "bucket are waiting (padded length at most 2048 tokens, so at most 8192 tokens per step); "
                              "otherwise the oldest request runs alone at batch 1. Batches of two or three have no "
                              "measured shape and padding them to batch 4 costs more than serving them singly, so they "
                              "are not used. Arrivals pinned so that the default's realized busy fraction is rho.",
                    "windows": list(STATIC_WINDOWS), "cells": {}, "summary": {}}
    for (lab, ses, tw, rho), r in st.items():
        static["cells"].setdefault(lab, {}).setdefault(ses, {}).setdefault(tw, {})[f"{rho}"] = {
            "default": metric_dict(r["point"]["default"]),
            "share_requests_batched_default": _g(r["info"]["default"]["share_batched"]),
            "busy_fraction_default": _g(r["info"]["default"]["busy_fraction"]),
            "makespan_over_span_default": _g(r["info"]["default"]["makespan_over_span"]),
            "busy_fraction_table": _g(r["info"]["table"]["busy_fraction"]),
            "table": gain_entry(r["point"]["default"], r["point"]["table"], r["reps"]["default"], r["reps"]["table"]),
            "best": gain_entry(r["point"]["default"], r["point"]["best"], r["reps"]["default"], r["reps"]["best"])}
    for lab in LABELS:
        for ses in SESSIONS:
            for rho in RHOS:
                keys = [(lab, ses, tw, rho) for tw in STATIC_WINDOWS if (lab, ses, tw, rho) in st]
                if not keys:
                    continue
                rec = {}
                for arm in ("table", "best"):
                    for mi, m in enumerate(METRICS):
                        pts = np.array([float(gain_pct(st[k]["point"]["default"][mi], st[k]["point"][arm][mi])) for k in keys])
                        reps_m = np.stack([gain_pct(st[k]["reps"]["default"][:, mi], st[k]["reps"][arm][:, mi]) for k in keys])
                        rec.setdefault(arm, {})[m] = bootstrap_family(
                            pts, reps_m, None, [args.seed, _crc("static"), _crc(lab), _crc(ses), _crc(arm), _crc(m),
                                                int(round(rho * 100))])
                static["summary"].setdefault(lab, {}).setdefault(ses, {})[f"{rho}"] = rec

    curves: dict = {}
    for lab in LABELS:
        for ses in SESSIONS:
            for B in BATCHES:
                arms = StepArms(step[lab], lab, ses, B, args.seed)
                curves.setdefault(lab, {}).setdefault(ses, {})[f"B{B}"] = {
                    "lengths": [int(x) for x in arms.L], "default_chunk": arms.chunks[arms.i_default],
                    "table_chunk": [arms.chunks[i] for i in arms.i_table],
                    "best_crossed_chunk": [arms.chunks[i] for i in arms.i_best],
                    "default_ms": [_g(x) for x in arms.knots("default")], "table_ms": [_g(x) for x in arms.knots("table")],
                    "best_crossed_ms": [_g(x) for x in arms.knots("best")],
                    "default_round_cv": [_g(float(np.nanstd(arms.rounds[i, arms.i_default]) / np.nanmean(arms.rounds[i, arms.i_default])))
                                         for i in arms.rows],
                    "n_rounds": int(arms.counts.max()), "table_source": step[lab][ses][B]["table_source"]}
    return {"experiment": "jpdc_load_sim", "item": "C2", "created_utc": utc_now(), "scope": SCOPE, "limits": LIMITS,
            "sign_convention": "gain_pct = 100 * (default - variant) / default; positive means the variant is faster",
            "config": {"rhos": list(RHOS), "rhos_low_sensitivity_trace_only": list(RHOS_LOW), "metrics": list(METRICS), "seed": args.seed, "round_replicates": args.reps,
                       "round_replicates_large_windows": args.reps_large, "large_window_requests": args.large_window,
                       "capture_replicates": args.capture_reps, "capture_replicates_large_windows": args.capture_reps_large,
                       "poisson_n": args.poisson_n, "poisson_replications": args.poisson_reps,
                       "poisson_round_replicates": args.poisson_round_reps, "poisson_round_n": args.poisson_round_n,
                       "static_round_replicates": args.static_reps, "uniform_factor": UNIFORM_FACTOR,
                       "n_slices_peak_factor": N_SLICES, "thresholds_pct": list(THRESHOLDS_PCT),
                       "short_latencies": bool(args.short_latencies),
                       "prompt_below_lowest_knot": "represented by the lowest measured step",
                       "prompt_above_highest_knot": "represented by the highest measured step"},
            "inputs": {"traces": traces, "missing_traces": [k for k, v in traces.items() if v["status"] != "ok"]},
            "windows": windows, "checks": checks, "step_curves": curves, "service": service, "summary": summary,
            "thresholds": thresholds, "thresholds_per_window": thresholds_window, "sign_agreement": sign_agreement,
            "placebo": placebo, "elasticity": elasticity, "operating_points": operating,
            "sensitivity_timestamp_jitter": jitter_out,
            "sensitivity_static_batching": static, "cells": cells}


def print_summary(res: dict, log) -> None:
    log("\nwindow-mean gain of the table over the default, trace arrivals (session 1 / session 2, 95% interval over "
        "windows and rounds)")
    for lab in LABELS:
        for fam in ("azure", "burstgpt"):
            if fam not in res["summary"].get(lab, {}).get("trace", {}):
                continue
            log(f"  {lab} {fam}")
            for rho in RHOS_ALL:
                row = res["summary"][lab]["trace"][fam].get(f"{rho}")
                if not row:
                    continue
                txt = []
                for m in ("wait_mean", "resp_mean", "resp_p99"):
                    a, b = (row[s]["table"][m] for s in SESSIONS)
                    txt.append(f"{m} {a['mean_gain_pct']:+.2f} {a['ci95_windows_rounds']} / {b['mean_gain_pct']:+.2f} "
                               f"{b['ci95_windows_rounds']}")
                log(f"    rho={rho}: " + "; ".join(txt))


# --------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dirs", default=str(DATA_DIR), help="comma list of directories searched for the trace files; a "
                                                              "missing trace is downloaded into the first one")
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "jpdc_load_sim.json")
    p.add_argument("--windows", default="all", help="comma list of trace|window, e.g. azure2023_code|1h")
    p.add_argument("--short-latencies", default=None, help="JSON with measured steps below 256 tokens (see the docstring)")
    p.add_argument("--work-dir", type=Path, default=None, help="kept if given (mapped windows and raw task results)")
    p.add_argument("--reuse-raw", action="store_true", help="re-aggregate the raw task results of --work-dir")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--reps", type=int, default=200, help="round resamples per trace cell")
    p.add_argument("--reps-large", type=int, default=40, help="round resamples per trace cell of a large window")
    p.add_argument("--large-window", type=int, default=5_000_000, help="requests from which a window counts as large")
    p.add_argument("--capture-reps", type=int, default=40)
    p.add_argument("--capture-reps-large", type=int, default=10)
    p.add_argument("--poisson-n", type=int, default=2_000_000, help="requests per Pollaczek-Khinchine replication")
    p.add_argument("--poisson-reps", type=int, default=16)
    p.add_argument("--poisson-round-reps", type=int, default=100)
    p.add_argument("--poisson-round-n", type=int, default=200_000)
    p.add_argument("--static-reps", type=int, default=30)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    keep = args.work_dir is not None
    args.work_dir = Path(args.work_dir or tempfile.mkdtemp(prefix="jpdc_load_sim_"))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.in_dir = Path(args.in_dir)
    raw_path = args.work_dir / "raw_results.pkl"
    meta_path = args.work_dir / "raw_meta.pkl"
    step = {lab: load_step_data(lab, args.in_dir, args.short_latencies) for lab in LABELS}
    if args.reuse_raw:
        with open(raw_path, "rb") as fh:
            raw = pickle.load(fh)
        with open(meta_path, "rb") as fh:
            traces, windows, primary, jitter = pickle.load(fh)
    else:
        selected = [(n, w) for n, w, _ in WINDOWS]
        if args.windows != "all":
            names = args.windows.split(",")
            selected = [(n, w) for n, w in selected if f"{n}|{w}" in names]
        dirs = [Path(d) for d in args.data_dirs.split(",")]
        absent = [str(d) for d in dirs[1:] if not d.is_dir()]
        if absent:      # a mistyped directory must not trigger a multi-gigabyte download
            raise SystemExit(f"--data-dirs: not a directory: {absent}")
        traces, windows, ready = prepare_windows(selected, dirs, args.in_dir, args.work_dir, args.seed, log)
        primary = [tw for tw in ready if not tw.endswith(JITTER_SUFFIX)]
        jitter = [tw for tw in ready if tw.endswith(JITTER_SUFFIX)]
        cfg = {"in_dir": str(args.in_dir), "work_dir": str(args.work_dir), "seed": args.seed,
               "short_latencies": args.short_latencies, "poisson_n": args.poisson_n, "poisson_reps": args.poisson_reps,
               "poisson_round_reps": args.poisson_round_reps, "poisson_round_n": args.poisson_round_n}
        specs = []
        for lab in LABELS:
            for ses in SESSIONS:
                for rho in RHOS_ALL:
                    for tw in primary:
                        large = windows[tw]["n_requests"] >= args.large_window
                        base = {"cfg": cfg, "label": lab, "session": ses, "tw": tw, "rho": rho}
                        specs.append({**base, "kind": "trace", "reps": args.reps_large if large else args.reps,
                                      "capture_reps": args.capture_reps_large if large else args.capture_reps})
                        if rho in RHOS:
                            specs.append({**base, "kind": "poisson"})
                            if tw in STATIC_WINDOWS:
                                specs.append({**base, "kind": "static", "reps": args.static_reps})
                    for tw in jitter:
                        specs.append({"cfg": cfg, "label": lab, "session": ses, "tw": tw, "rho": rho, "kind": "trace",
                                      "reps": args.reps_large, "capture_reps": 0})
        specs.sort(key=lambda s: windows[s["tw"]]["n_requests"])
        log(f"{len(specs)} tasks on {args.workers} workers; windows: {primary} + {jitter}")
        raw = []
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_task, s) for s in specs]
            for i, f in enumerate(as_completed(futs)):
                raw.append(f.result())
                if (i + 1) % 50 == 0 or i + 1 == len(futs):
                    log(f"  {i + 1}/{len(futs)} tasks, {time.time() - t0:.0f}s")
        raw.sort(key=lambda r: (r["kind"], r["label"], r["session"], r["tw"], r["rho"]))
        with open(raw_path, "wb") as fh:
            pickle.dump(raw, fh)
        with open(meta_path, "wb") as fh:
            pickle.dump((traces, windows, primary, jitter), fh)
    res = aggregate(raw, args, step, traces, windows, primary, jitter, log)
    write_json(args.out, res)
    print_summary(res, log)
    log(f"wrote {args.out}")
    if not keep:
        shutil.rmtree(args.work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
