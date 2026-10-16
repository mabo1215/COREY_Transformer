"""Shared helpers of the taco-r2 offline statistics (items M3a, M5a, M5b, M5d, M5h, M5i, M6a).

CPU only; reads stored result files and times nothing. The helpers wrap the conventions of the existing analyses
so that every new number is defined exactly as the manuscript's:

* per-call regret          T(c) / T(c*) - 1 with c* the per-call oracle of the view (``taco_report_numbers.record``);
* latency-weighted share   100 * sum(excess) / sum(oracle latency) (``lw_share_of_oracle_pct``, the column "LW" of
                           the CUDA-graph table);
* prompt-cluster bootstrap the connected components of the call-prompt graph (``e3b_information.prompt_clusters``),
                           resampled with replacement; the replicate matrix is drawn exactly as
                           ``e3b_information.cluster_bootstrap`` draws it, so with the same generator
                           (``taco_report_numbers.make_rng``) the stored percentile intervals are reproduced;
* BCa intervals            bias-corrected and accelerated, acceleration from the leave-one-cluster-out jackknife.

Statistics are written as functions of per-cluster sums (ratios of sums, differences of ratios), so the bootstrap
replicates, the jackknife and the point estimate use one code path. Result files are named
``taco_r2_stats_<item>.json``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import norm

from src.experiments.tpds import e22_graph_analysis as E22
from src.experiments.tpds import e3b_information as E3B
from src.experiments.tpds import taco_e26_analysis as E26
from src.experiments.tpds import taco_report_numbers as TRN
from src.experiments.tpds.common import RESULTS_DIR, utc_now, write_json

SEED = TRN.SEED
N_BOOT = 2000
SYNC_US = {"RTX 3090": 21.4, "RTX PRO 6000": 20.2, "H800": 20.0}   # one host synchronization (Table S37)

# CUDA-graph tables of Table 7 (three rounds, round medians): name -> (GPU, kernel label)
GRAPH_TABLES = {
    "ssd370m_graph": ("RTX 3090", "SSD 370M"),
    "ssd370m_graph_pro6000": ("RTX PRO 6000", "SSD 370M"),
    "ssd2p7b_graph": ("RTX 3090", "SSD 2.7B"),
    "ssd2p7b_graph_pro6000": ("RTX PRO 6000", "SSD 2.7B"),
    "gdn1p4b_graph": ("RTX 3090", "GDN 1.4B"),
    "gdn1p4b_graph_pro6000": ("RTX PRO 6000", "GDN 1.4B"),
    "m1_370m_graph": ("RTX 3090", "Triton Mamba-1 370M"),
    "m1_370m_graph_pro6000": ("RTX PRO 6000", "Triton Mamba-1 370M"),
}


def make_rng(*parts) -> np.random.Generator:
    return TRN.make_rng(*parts)


def write_result(name: str, obj: dict, in_dir: Path = RESULTS_DIR) -> Path:
    path = Path(in_dir) / f"taco_r2_stats_{name}.json"
    write_json(path, obj)
    return path


def load_json(path: Path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------- graph tables
_REGISTERED = {"done": False}


def register_graph_tables(in_dir: Path = RESULTS_DIR) -> list[str]:
    """Switch the RTX 3090 graph tables to their three-round sweeps and register the RTX PRO 6000 replays."""
    if not _REGISTERED["done"]:
        E22.use_three_rounds(Path(in_dir))
        E26.register(Path(in_dir), "pro6000", "pro6000")
        _REGISTERED["done"] = True
    return [n for n in GRAPH_TABLES if n in E22.TABLES]


def graph_table(name: str, in_dir: Path = RESULTS_DIR) -> "E22.Table":
    register_graph_tables(in_dir)
    t = E22.Table(name, Path(in_dir))
    t.set_floor()
    return t


# --------------------------------------------------------------------------- cluster bootstrap
class ClusterBoot:
    """Prompt-cluster bootstrap over a subset of calls with statistics written on per-cluster sums.

    ``M`` (B x K) is drawn as ``e3b_information.cluster_bootstrap`` draws it (one ``multinomial`` call on a fresh
    generator), so ``ClusterBoot(clusters, B, make_rng(label, sub), sel)`` reuses the replicates of
    ``taco_report_numbers.boot(..., make_rng(label, sub), ...)``.
    """

    def __init__(self, clusters: np.ndarray, B: int, rng: np.random.Generator, sel: np.ndarray | None = None):
        clusters = np.asarray(clusters)
        self.idx = np.arange(len(clusters)) if sel is None else np.flatnonzero(sel)
        _, cl = np.unique(clusters[self.idx], return_inverse=True)
        self.cl = cl.ravel()
        self.K = int(self.cl.max()) + 1
        self.B = int(B)
        self.M = rng.multinomial(self.K, np.full(self.K, 1.0 / self.K), size=self.B).astype(float)
        self.cnt = np.bincount(self.cl, minlength=self.K).astype(float)

    def sums(self, x: np.ndarray) -> np.ndarray:
        """Per-cluster sums of a per-call array (indexed over all calls)."""
        s = np.zeros(self.K)
        np.add.at(s, self.cl, np.asarray(x, float)[self.idx])
        return s

    def evaluate(self, arrays: dict[str, np.ndarray], g: Callable[[dict, np.ndarray], np.ndarray]) -> dict:
        """Point estimate, replicates and leave-one-cluster-out values of g(totals, n).

        ``arrays``: per-call arrays (over all calls; the subset is applied here). ``g`` receives a dict of totals
        (arrays of equal shape) and the matching call counts and returns the statistic elementwise.
        """
        S = {k: self.sums(v) for k, v in arrays.items()}
        point = float(g({k: np.array([s.sum()]) for k, s in S.items()}, np.array([self.cnt.sum()]))[0])
        reps = g({k: self.M @ s for k, s in S.items()}, self.M @ self.cnt)
        jack = g({k: s.sum() - s for k, s in S.items()}, self.cnt.sum() - self.cnt)
        return {"point": point, "reps": np.asarray(reps, float), "jack": np.asarray(jack, float)}


def pct_ci(reps: np.ndarray, level: float = 0.95) -> list[float]:
    reps = np.asarray(reps, float)
    reps = reps[np.isfinite(reps)]
    a = (1 - level) / 2
    return [float(np.percentile(reps, 100 * a)), float(np.percentile(reps, 100 * (1 - a)))]


def bca_ci(point: float, reps: np.ndarray, jack: np.ndarray, level: float = 0.95) -> list[float]:
    """Bias-corrected and accelerated interval (Efron 1987); acceleration from the cluster jackknife."""
    reps = np.asarray(reps, float)
    reps = reps[np.isfinite(reps)]
    jack = np.asarray(jack, float)
    jack = jack[np.isfinite(jack)]
    if len(reps) == 0:
        return [None, None]
    prop = (np.sum(reps < point) + 0.5 * np.sum(reps == point)) / len(reps)
    prop = min(max(prop, 0.5 / len(reps)), 1 - 0.5 / len(reps))
    z0 = norm.ppf(prop)
    d = jack.mean() - jack
    den = 6.0 * (np.sum(d ** 2) ** 1.5)
    acc = float(np.sum(d ** 3) / den) if den > 0 else 0.0
    out = []
    for q in ((1 - level) / 2, 1 - (1 - level) / 2):
        zq = norm.ppf(q)
        adj = norm.cdf(z0 + (z0 + zq) / (1 - acc * (z0 + zq)))
        out.append(float(np.percentile(reps, 100 * adj)))
    return out


def summarize(ev: dict, level: float = 0.95, digits: int | None = None) -> dict:
    """Point, percentile and BCa intervals, bootstrap SE and the share of replicates <= 0."""
    reps = ev["reps"][np.isfinite(ev["reps"])]
    rec = {"point": ev["point"], "ci95_percentile": pct_ci(reps, level), "ci95_bca": bca_ci(ev["point"], reps, ev["jack"], level),
           "boot_se": float(reps.std(ddof=1)) if len(reps) > 1 else None,
           "share_replicates_le_0": float(np.mean(reps <= 0)) if len(reps) else None}
    if digits is not None:
        rec = _round(rec, digits)
    return rec


def _round(obj, digits):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, list):
        return [_round(x, digits) for x in obj]
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    return obj


# --------------------------------------------------------------------------- policy statistics on sums
def lw_g(exc: str = "exc", orc: str = "orc") -> Callable:
    """Latency-weighted share of the oracle time (percent)."""
    return lambda T, n: 100.0 * T[exc] / T[orc]


def mean_g(name: str, scale: float = 1.0) -> Callable:
    return lambda T, n: scale * T[name] / n


def policy_arrays(lat: np.ndarray, orc: np.ndarray, pick: np.ndarray, prefix: str) -> dict:
    """Per-call excess latency (ms) and relative regret of a per-call chunk index."""
    ch = lat[np.arange(len(orc)), np.asarray(pick)]
    return {f"{prefix}_exc": ch - orc, f"{prefix}_reg": ch / orc - 1.0}


def cluster_sizes(clusters: np.ndarray, df=None) -> dict:
    """Number and size distribution of the prompt clusters (calls and prompts per cluster)."""
    clusters = np.asarray(clusters)
    _, cnt = np.unique(clusters, return_counts=True)
    rec = {"n_clusters": int(len(cnt)), "calls_per_cluster": _dist(cnt)}
    if df is not None and "prompt_ids" in df:
        prompts: dict[int, set] = {}
        for c, p in zip(clusters, df["prompt_ids"].astype(str)):
            prompts.setdefault(int(c), set()).update(p.split(";"))
        pc = np.array([len(prompts[k]) for k in sorted(prompts)])
        rec["prompts_per_cluster"] = _dist(pc)
        rec["n_prompts"] = int(pc.sum())
        rec["share_calls_in_largest_cluster"] = float(cnt.max() / cnt.sum())
    return rec


def _dist(v) -> dict:
    v = np.asarray(v, float)
    return {"min": float(v.min()), "p10": float(np.percentile(v, 10)), "median": float(np.median(v)),
            "mean": float(v.mean()), "p90": float(np.percentile(v, 90)), "max": float(v.max()),
            "values_sorted_desc": sorted((int(x) for x in v), reverse=True)}


def round_matrix(df, chunks: list[int]) -> np.ndarray:
    """(rounds, calls, chunks) round means parsed from ``lat_c<C>_rounds`` (as ``r4_rounds_analysis``)."""
    cols = []
    for c in chunks:
        col = f"lat_c{c}_rounds"
        cols.append(np.array([[float(x) if x not in ("", "nan") else np.nan for x in str(s).split(";")]
                              for s in df[col]]))
    return np.stack(cols, 2).transpose(1, 0, 2)


def holm(p: dict[str, float]) -> dict[str, float]:
    return E3B._holm(p)


def bh(p: dict[str, float]) -> dict[str, float]:
    """Benjamini-Hochberg adjusted p-values (step-up)."""
    items = sorted(p.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = [0.0] * m
    run = 1.0
    for i in range(m - 1, -1, -1):
        run = min(run, items[i][1] * m / (i + 1))
        adj[i] = min(1.0, run)
    return {k: adj[i] for i, (k, _) in enumerate(items)}


def meta(script: str, args=None) -> dict:
    return {"script": script, "created_utc": utc_now(), "seed": SEED,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()} if args is not None else None}


def fmt_ci(rec: dict, nd: int = 2, kind: str = "ci95_percentile") -> str:
    lo, hi = rec[kind]
    return f"{rec['point']:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def isfinite_or_none(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x
