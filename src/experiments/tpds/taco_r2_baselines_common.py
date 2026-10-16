"""Shared helpers of the taco-r2 offline re-analyses (review items M2b, M2c, M2d, M4a-M4d, M7b).

CPU only; reads stored result files and times nothing. The helpers wrap the machinery of the earlier
analyses so that every definition matches the manuscript's tables:

  per-call regret      lat(chosen chunk) / lat(per-call oracle) - 1 (``taco_report_numbers.record``)
  latency-weighted     lw_share_of_oracle_pct = 100 sum(excess) / sum(oracle latency): the share of the scan
                       time at the per-call oracle that a policy adds (the convention of Tables 4 and 7 and of
                       the review); lw_share_of_policy_pct = 100 sum(excess) / sum(policy latency) is also given
  intervals            95% prompt-cluster bootstrap (``taco_report_numbers.boot`` / ``e3b_information``)
  held out             chosen on the training prompt folds of the grouped five-fold split (column ``fold``),
                       scored on the held-out fold, as every held-out row of the manuscript

``CallTable`` holds one per-call table (latencies +inf where a chunk could not run, oracle, tie sets, prompt
folds and clusters, shape keys). ``untuned_picks`` returns the per-call chunk index of every untuned or
table policy of review item M2b.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.experiments.tpds import e3b_information as E3B  # noqa: E402
from src.experiments.tpds import taco_report_numbers as TRN  # noqa: E402
from src.experiments.tpds.analysis import chunk_columns, latency_matrix, load_e2, round_median_view  # noqa: E402
from src.experiments.tpds.common import RESULTS_DIR, utc_now, write_json  # noqa: E402

OUT_PREFIX = "taco_r2_baselines_"
SEED = 20261016

# Triton Mamba-1 per-call tables: name -> (csv, round-median view, GPU, convention, calibration tag of the cost
# model or None, scan layers, role)
TRITON_SETS = {
    "pro6000_device": ("e2_mamba1_pro6000_r4spin.csv", True, "RTX PRO 6000",
                       "device time, medians of three shuffled rounds", "pro6000_r4spin", 48, "primary"),
    "vgpu3090_device": ("e2_mamba1_vgpu3090_r4spin.csv", True, "RTX 3090",
                        "device time, medians of three shuffled rounds", "vgpu3090_r3spin", 48, "primary"),
    "h800_device": ("e2_mamba1_h800_r3spin.csv", False, "H800 PCIe", "device time, one block per chunk",
                    "h800_r3spin", 48, "primary"),
    "pro6000_device_pooled": ("e2_mamba1_pro6000_r4spin.csv", False, "RTX PRO 6000",
                              "device time, means pooled over three rounds (view of the cost-model analyses)",
                              "pro6000_r4spin", 48, "calibration-analysis view"),
    "vgpu3090_device_one_block": ("e2_mamba1_vgpu3090_r3spin.csv", False, "RTX 3090",
                                  "device time, one block per chunk (session of the cost-model analyses)",
                                  "vgpu3090_r3spin", 48, "calibration-analysis view"),
    "pro6000_launch": ("e2_mamba1_pro6000.csv", False, "RTX PRO 6000", "launch-inclusive, one block per chunk",
                       "pro6000", 48, "secondary"),
    "vgpu3090_launch": ("e2_mamba1_vgpu3090.csv", False, "RTX 3090", "launch-inclusive, one block per chunk",
                        "vgpu3090", 48, "secondary"),
    "h800_launch": ("e2_mamba1_h800.csv", False, "H800 PCIe", "launch-inclusive, one block per chunk", "h800", 48,
                    "secondary"),
    "vgpu3090_1p4b_device": ("e2_mamba1_vgpu3090_1p4b_r3.csv", False, "RTX 3090",
                             "device time, one block per chunk, Mamba-1 1.4B widths", None, 48, "larger checkpoint"),
    "vgpu3090_2p8b_device": ("e2_mamba1_vgpu3090_2p8b_r3.csv", False, "RTX 3090",
                             "device time, one block per chunk, Mamba-1 2.8B widths", None, 64, "larger checkpoint"),
    "vgpu3090_graph": ("e2_mamba1_vgpu3090_r4graph3.csv", True, "RTX 3090",
                       "CUDA-graph replay, warm L2, medians of three rounds", None, 48, "graph replay"),
    "pro6000_graph": ("e2_mamba1_pro6000_taco_graph3.csv", True, "RTX PRO 6000",
                      "CUDA-graph replay, warm L2, medians of three rounds", None, 48, "graph replay"),
}
PRIMARY = ("pro6000_device", "vgpu3090_device", "h800_device")
GPU_OF = {"pro6000_device": "pro6000", "vgpu3090_device": "vgpu3090", "h800_device": "h800"}

# production kernels: name -> (csv, round-median view, GPU, convention, family, library default chunk, layers)
PRODUCTION_SETS = {
    "ssd370m_graph_3090": ("e2_mamba2_vgpu3090_r4graph3.csv", True, "RTX 3090", "CUDA-graph replay, three rounds",
                           "Mamba-2 SSD 370M", 256, 48),
    "ssd370m_graph_pro6000": ("e2_mamba2_pro6000_taco_graph3.csv", True, "RTX PRO 6000",
                              "CUDA-graph replay, three rounds", "Mamba-2 SSD 370M", 256, 48),
    "ssd2p7b_graph_3090": ("e2_mamba2_vgpu3090_2p7b_r4graph3.csv", True, "RTX 3090", "CUDA-graph replay, three rounds",
                           "Mamba-2 SSD 2.7B", 256, 64),
    "ssd2p7b_graph_pro6000": ("e2_mamba2_pro6000_2p7b_taco_graph3.csv", True, "RTX PRO 6000",
                              "CUDA-graph replay, three rounds", "Mamba-2 SSD 2.7B", 256, 64),
    "gdn1p4b_graph_3090": ("e2_gdn_vgpu3090_1p4b_r4graph3.csv", True, "RTX 3090", "CUDA-graph replay, three rounds",
                           "Gated DeltaNet 1.4B", 64, 12),
    "gdn1p4b_graph_pro6000": ("e2_gdn_pro6000_1p4b_taco_graph3.csv", True, "RTX PRO 6000",
                              "CUDA-graph replay, three rounds", "Gated DeltaNet 1.4B", 64, 12),
    "ssd370m_launch_3090": ("e2_mamba2_vgpu3090.csv", False, "RTX 3090", "launch-inclusive, one block",
                            "Mamba-2 SSD 370M", 256, 48),
    "ssd370m_launch_pro6000": ("e2_mamba2_pro6000.csv", False, "RTX PRO 6000", "launch-inclusive, one block",
                               "Mamba-2 SSD 370M", 256, 48),
    "ssd2p7b_device_3090": ("e2_mamba2_vgpu3090_2p7b_r3.csv", False, "RTX 3090", "device time, one block",
                            "Mamba-2 SSD 2.7B", 256, 64),
    "gdn1p4b_device_3090": ("e2_gdn_vgpu3090_1p4b_r3.csv", False, "RTX 3090", "device time, one block",
                            "Gated DeltaNet 1.4B", 64, 12),
}

POLICY_INFO = {
    "default_chunk_256": "fixed chunk 256 (the Mamba-2 SSD default, borrowed; the manuscript's untuned reference of "
                         "the Triton scan)",
    "library_default": "the library default chunk of the kernel (SSD 256, Gated DeltaNet 64)",
    "free_length_rule": "C = clip(nextpow2(L), C_min, C_max) over the kernel's chunk set (no timing; mirrors the "
                        "stock dispatch of the reference CUDA scan, clip(nextpow2(L), 128, 2048))",
    "capped_default": "C = max(C_min, min(C_default, nextpow2(L))): the library default capped at the next power of "
                      "two of the length (no timing)",
    "stock_length_rule": "the reference scan's stock dispatch chunk, timed in the runtime-chunk build",
    "best_static_global_insample_regret": "one chunk for all calls, lowest mean per-call regret on the scored calls",
    "best_static_global_insample_latency": "one chunk for all calls, lowest total latency on the scored calls",
    "best_static_global_heldout_regret": "one chunk, lowest mean per-call regret on the training prompt folds",
    "best_static_global_heldout_latency": "one chunk, lowest total latency on the training prompt folds",
    "best_static_per_batch_insample_regret": "one chunk per batch size, lowest mean per-call regret (in sample)",
    "best_static_per_batch_insample_latency": "one chunk per batch size, lowest total latency (in sample)",
    "best_static_per_batch_heldout_regret": "one chunk per batch size, lowest mean per-call regret on the training "
                                            "folds (fallback: the global chunk of the training folds)",
    "best_static_per_batch_heldout_latency": "one chunk per batch size, lowest total latency on the training folds",
    "length_bucket_table_heldout_latency": "table keyed on (power-of-two length bucket, batch), entry = lowest mean "
                                           "latency on the training folds, fallback the chunk with the lowest mean "
                                           "latency (the estimator of the supplementary bucket-headroom table)",
    "length_bucket_aligned_table_heldout_latency": "as above, keyed on (length bucket, batch, L mod 16 = 0)",
    "length_bucket_table_heldout_regret": "table keyed on (length bucket, batch), entry = lowest mean per-call "
                                          "regret on the training folds",
    "exact_shape_table_heldout_regret": "exact-shape table (L, batch, width, alignment) with a (bucket, batch, "
                                        "alignment) and a global fallback, entries by mean per-call regret on the "
                                        "training folds (the manuscript's held-out exact-shape table)",
    "cost_model_frozen": "Eq. (3) with the stored eight-parameter fit of the GPU (form frozen on two GPUs' calls)",
}


# =========================================================================== data
def nextpow2(x) -> np.ndarray:
    x = np.maximum(np.asarray(x, float), 1.0)
    return (2 ** np.ceil(np.log2(x))).astype(np.int64)


def load_view(csv: str, rmed: bool, in_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    df = load_e2([in_dir / csv])
    if "regimes" in df and (df["regimes"] == "e4_key").any():
        df = df[df["regimes"] != "e4_key"]
    if rmed:
        df = round_median_view(df)
    return df.reset_index(drop=True)


class CallTable:
    """Per-call table with +inf latencies where a chunk could not run and the derived keys of the selectors."""

    def __init__(self, df: pd.DataFrame, name: str, layers: int, source: str, chunks: list[int] | None = None):
        self.df, self.name, self.layers, self.source = df.reset_index(drop=True), name, layers, source
        self.chunks = list(chunks) if chunks is not None else chunk_columns(self.df)
        lat = latency_matrix(self.df, self.chunks)
        self.avail = np.isfinite(lat)
        self.lat = np.where(self.avail, lat, np.inf)
        self.orc = self.lat.min(1)
        self.N, self.C = self.lat.shape
        self.norm = self.lat / self.orc[:, None]
        self.norm_fit = np.minimum(self.norm, 1e6)           # finite copy for selection (as e3b_information)
        self.lat_fit = np.where(self.avail, self.lat, 1e6)
        tied = [set(int(c) for c in str(x).split(";") if c) for x in self.df["tied_chunks"].fillna("")]
        self.tied = np.array([[c in s for c in self.chunks] for s in tied], dtype=bool)
        self.folds = self.df["fold"].to_numpy()
        self.batch = self.df["batch"].to_numpy()
        self.L = self.df["L"].to_numpy()
        self.aligned = (self.L % 16 == 0).astype(int)
        width = self.df["width"].to_numpy() if "width" in self.df else np.zeros(self.N, int)
        kd = pd.DataFrame({"L": self.L, "b": self.batch, "w": width, "a": self.aligned})
        self.key = kd.groupby(["L", "b", "w", "a"], sort=True).ngroup().to_numpy()
        lb = self.df["L_bucket"].to_numpy()
        self.bucket = pd.DataFrame({"lb": lb, "b": self.batch}).groupby(["lb", "b"]).ngroup().to_numpy()
        self.bucket_aligned = pd.DataFrame({"lb": lb, "b": self.batch, "a": self.aligned}) \
            .groupby(["lb", "b", "a"]).ngroup().to_numpy()
        self.batch_code = pd.factorize(self.batch, sort=True)[0]
        self.clusters = E3B.prompt_clusters(self.df)[0]

    def keys(self) -> list[str]:
        return [str(k) for k in self.df["key"]]


def triton_table(name: str, in_dir: Path = RESULTS_DIR) -> CallTable:
    csv, rmed, _gpu, _conv, _cal, layers, _role = TRITON_SETS[name]
    return CallTable(load_view(csv, rmed, in_dir), name, layers, csv)


def production_table(name: str, in_dir: Path = RESULTS_DIR) -> CallTable:
    csv, rmed, _gpu, _conv, _fam, _default, layers = PRODUCTION_SETS[name]
    return CallTable(load_view(csv, rmed, in_dir), name, layers, csv)


# =========================================================================== selection on prompt folds
def _fold_iter(folds):
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if tr.any() and te.any():
            yield tr, te


def table_pick(T: CallTable, levels: list[np.ndarray], crit: np.ndarray, held_out: bool = True,
               fallback_crit: np.ndarray | None = None) -> np.ndarray:
    """Hierarchical lookup table (finest level first; coarser levels and then one global chunk as fallbacks).

    Every entry is the chunk with the lowest sum of ``crit`` over its training calls (``crit`` = normalized
    latency: mean per-call regret; = latency: total latency). Held out: chosen on the training prompt folds and
    scored on the held-out fold (``e3b_information.cv_table`` for ``crit = norm_fit``); in sample: chosen on all
    calls. ``fallback_crit`` sets the criterion of the global fallback (default ``crit``)."""
    fb = crit if fallback_crit is None else fallback_crit
    splits = list(_fold_iter(T.folds)) if held_out else [(np.ones(T.N, bool), np.ones(T.N, bool))]
    ch = np.empty(T.N, dtype=int)
    for tr, te in splits:
        out = np.full(te.sum(), int(np.argmin(fb[tr].sum(0))))
        for codes in reversed(levels):
            sums = np.zeros((codes.max() + 1, T.C))
            np.add.at(sums, codes[tr], crit[tr])
            have = np.zeros(codes.max() + 1, dtype=bool)
            have[codes[tr]] = True
            c = codes[te]
            m = have[c]
            out[m] = sums.argmin(1)[c[m]]
        ch[te] = out
    return ch


def rule_pick(T: CallTable, target: np.ndarray) -> tuple[np.ndarray, int]:
    """Index of the smallest available chunk >= target per call (largest available chunk if none); returns the
    picks and the number of calls whose target chunk could not run and was replaced."""
    ch = np.array(T.chunks)
    out = np.empty(T.N, dtype=int)
    n_repl = 0
    for i, c in enumerate(target):
        cand = np.flatnonzero(ch >= c)
        j = int(cand[0]) if len(cand) else T.C - 1
        if not T.avail[i, j]:
            ok = np.flatnonzero(T.avail[i])
            j = int(ok[np.argmin(np.abs(np.log2(ch[ok]) - np.log2(ch[j])))])
            n_repl += 1
        out[i] = j
    return out, n_repl


def free_rule_pick(T: CallTable) -> tuple[np.ndarray, int]:
    lo, hi = min(T.chunks), max(T.chunks)
    return rule_pick(T, np.clip(nextpow2(T.L), lo, hi))


def capped_default_pick(T: CallTable, default: int) -> tuple[np.ndarray, int]:
    return rule_pick(T, np.maximum(min(T.chunks), np.minimum(int(default), nextpow2(T.L))))


def untuned_picks(T: CallTable, default: int | None = None) -> tuple[dict, dict]:
    """Per-call chunk index of every untuned, static and table policy (review item M2b)."""
    p, info = {}, {}
    if default is None and 256 in T.chunks:
        p["default_chunk_256"] = np.full(T.N, T.chunks.index(256))
    elif default is not None:
        p["library_default"] = np.full(T.N, T.chunks.index(int(default)))
        p["capped_default"], info["capped_default_n_replaced"] = capped_default_pick(T, default)
    p["free_length_rule"], info["free_length_rule_n_replaced"] = free_rule_pick(T)
    one = [np.zeros(T.N, dtype=int)]
    for crit_name, crit in (("regret", T.norm_fit), ("latency", T.lat_fit)):
        for ho in (False, True):
            tag = "heldout" if ho else "insample"
            p[f"best_static_global_{tag}_{crit_name}"] = table_pick(T, [], crit, ho)
            p[f"best_static_per_batch_{tag}_{crit_name}"] = table_pick(T, [T.batch_code], crit, ho)
    p["length_bucket_table_heldout_latency"] = table_pick(T, [T.bucket], T.lat_fit, True)
    p["length_bucket_aligned_table_heldout_latency"] = table_pick(T, [T.bucket_aligned], T.lat_fit, True)
    p["length_bucket_table_heldout_regret"] = table_pick(T, [T.bucket], T.norm_fit, True)
    p["exact_shape_table_heldout_regret"] = table_pick(T, [T.key, T.bucket_aligned], T.norm_fit, True)
    del one
    return p, info


# =========================================================================== evaluation
def evaluate(T: CallTable, picks: dict, label: str, n_boot: int, pairs=()) -> tuple[dict, dict, int]:
    """Records (mean / p95 / max per-call regret, within-2% share, latency-weighted shares, us per call, ms per
    forward pass, chosen chunks) with prompt-cluster intervals, overall and per batch, and paired differences."""
    for n, pk in picks.items():
        bad = ~np.isfinite(T.lat[np.arange(T.N), pk])
        if bad.any():
            raise RuntimeError(f"{label}: policy {n} picks an unavailable chunk on {int(bad.sum())} calls")
    return TRN.evaluate_picks(T.lat, T.orc, T.tied, picks, T.chunks, T.batch, T.clusters, T.layers, label, n_boot,
                              pairs)


def compact(r: dict, by_batch: bool = True) -> dict:
    keys = ("n", "mean_regret_pct", "p95_regret_pct", "max_regret_pct", "within_2pct_share", "ci_tie_share",
            "lw_share_of_oracle_pct", "lw_share_of_policy_pct", "us_per_call", "ms_per_forward_pass",
            "oracle_scan_ms_per_forward_pass")
    out = {k: r[k] for k in keys if k in r}
    if "ci95" in r:
        out["mean_regret_pct_ci95"] = r["ci95"]["mean_regret_pct"]
        out["lw_share_of_oracle_pct_ci95"] = r["ci95"]["lw_share_of_oracle_pct"]
        out["us_per_call_ci95"] = r["ci95"]["us_per_call"]
    if "chosen" in r:
        out["chosen"] = r["chosen"]
    if by_batch and "by_batch" in r:
        out["by_batch"] = {b: {k: v[k] for k in ("n", "mean_regret_pct", "p95_regret_pct", "lw_share_of_oracle_pct",
                                                  "us_per_call", "chosen") if k in v}
                           for b, v in r["by_batch"].items()}
        for b, v in r["by_batch"].items():
            if "ci95" in v:
                out["by_batch"][b]["mean_regret_pct_ci95"] = v["ci95"]["mean_regret_pct"]
                out["by_batch"][b]["lw_share_of_oracle_pct_ci95"] = v["ci95"]["lw_share_of_oracle_pct"]
    return out


def stats_of(T: CallTable, pick: np.ndarray, sel: np.ndarray | None = None) -> dict:
    """Point statistics only (no intervals)."""
    return TRN.record(T.lat, T.orc, pick, T.tied, T.layers, sel=sel, chunks=T.chunks)


def map_choice(T: CallTable, keys: list[str], chunks_by_key: list[int]) -> np.ndarray:
    """Per-call chunk index on ``T`` of a choice stored per call key (same 2360 captured calls)."""
    m = dict(zip(keys, chunks_by_key))
    return np.array([T.chunks.index(int(m[k])) for k in T.keys()])


def frozen_cost_model_pick(T: CallTable, cal_tag: str, in_dir: Path = RESULTS_DIR) -> np.ndarray:
    pc = json.loads((in_dir / f"cost_model_{cal_tag}.json").read_text())["per_call_choice"]["mamba1"]
    return map_choice(T, pc["keys"], pc["chunk"])


class Checks:
    """Reproduction checks of quoted or stored values (``value`` within ``tol`` of ``reference``)."""

    def __init__(self):
        self.rows = []

    def add(self, what: str, value, reference, source: str, tol: float = 0.005) -> bool:
        ok = value is not None and reference is not None and abs(float(value) - float(reference)) <= tol
        self.rows.append({"check": what, "computed": None if value is None else float(value), "reference": reference,
                          "tolerance": tol, "source": source, "match": bool(ok)})
        return ok

    def summary(self) -> dict:
        return {"n": len(self.rows), "n_match": sum(r["match"] for r in self.rows),
                "mismatches": [r for r in self.rows if not r["match"]], "rows": self.rows}


def save(name: str, obj: dict, in_dir: Path = RESULTS_DIR) -> Path:
    path = in_dir / f"{OUT_PREFIX}{name}.json"
    obj.setdefault("created_utc", utc_now())
    write_json(path, obj)
    return path


def r(x, nd=2):
    return None if x is None else round(float(x), nd)
