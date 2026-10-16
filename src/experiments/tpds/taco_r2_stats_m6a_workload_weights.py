"""Review item M6a (taco-r2): headline regrets of the captured calls post-stratified to serving-trace workloads.

CPU only and deterministic; reads stored result files and the public serving traces (nothing is timed).

Why. The 2360 captured prefill calls of each checkpoint are a constructed mix (batch 1 / 4 / 16 fixed at
1536 / 488 / 336 calls, about 80% power-of-two lengths). Every per-call mean and latency-weighted share depends on
that mix. This script re-weights the per-call results so that the (length bucket, batch) cells carry the call
shares of real serving traces, and reports the headline numbers beside the unweighted ones.

Trace-side distributions (one scan call per layer per prefill call or scheduler step; all layers share the shape,
so call shares equal call shares per layer):
  W1  request-level prefill: one batch-1 call per request at its exact prompt length, for the ten trace windows of
      the serving replay (Azure 2023 code / conversation, first hour; Azure 2024 code / conversation, first hour,
      first day, whole week; BurstGPT busiest day and whole trace). The length-bucket histogram is recomputed from
      the trace files (with the alignment bit L mod 16 == 0) and checked against the stored histograms of the
      replay.
  W0  reference for W1: the captured batch-1 calls with their own length mix (separates the restriction to
      batch 1 from the change of the length mix).
  W2  continuous batching with chunked prefill (colocated decodes, offered prefill load 0.5, FCFS, at most 256
      running requests, token budget T in {2048, 8192}), padded kernel: one call per step keyed by
      (longest prefill chunk of the step, number of sequences). Simulated with the replay's own scheduler and
      time dilation; the simulated step statistics are checked against the stored replay. Windows: Azure 2023
      code and conversation (first hour), BurstGPT busiest day, Azure 2024 code and conversation (first hour).

Captured cells. Length bucket 2^floor(log2 L) (the convention of the captured tables and of the replay) and
batch in {1, 4, 16}. A trace cell (bucket b, batch n) is mapped to the captured batch nearest in log2 n and then to
the captured bucket nearest in log2 b among the buckets present at that batch; exact ties in log2 distance (batch
2 between 1 and 4, batch 8 between 4 and 16, bucket 32 between 16 and 64) split the mass equally between the two
neighbours. Trace mass outside the captured cells is reported as remapped. Alignment-aware sensitivity: strata
(bucket, batch, L mod 16 == 0); a trace cell whose captured cell has no call of its alignment falls back to the
whole captured cell (the fallback mass is reported).

Estimator (post-stratification). Stratum weights W_s are fixed by the trace; within each stratum the captured
calls are averaged with equal weight. Mean per-call regret = sum_s W_s mean_s(reg) / sum_s W_s; latency-weighted
share of the oracle time = 100 sum_s W_s mean_s(excess) / sum_s W_s mean_s(oracle). Intervals: 95% percentile
intervals over 2000 prompt-cluster bootstrap replicates (``e3b_information.prompt_clusters``; the replicate matrix
drawn as ``e3b_information.cluster_bootstrap``), with the stratum means re-estimated in every replicate and the
weights renormalized over the strata that are non-empty in that replicate (how often a stratum is empty, and the
weight lost, is reported). The unweighted rows use the generator of ``taco_report_numbers`` (same labels), so they
reproduce its stored intervals.

Data sets and policies.
  (a) Triton Mamba-1 370M scan, device time, three GPUs (RTX PRO 6000 and RTX 3090: medians of three rounds; H800:
      one block): fixed chunk 256; best chunk per batch (in sample, by mean per-call regret); the free length rule
      clip(nextpow2(L), 32, 1024); the held-out shape tree; the held-out exact-shape table; the frozen cost model
      (stored per-call choice). The policies are the unweighted ones (no re-selection under the weights).
  (b) CUDA-graph replays of Table 7 (warm L2, three rounds, round medians; RTX 3090 and RTX PRO 6000): library
      default (SSD 256, Gated DeltaNet 64; Triton: best single chunk in sample), the held-out exact-shape table and
      the recoverable share (default minus table, paired within each replicate).

Output: ``taco_r2_stats_m6a.json`` in the results directory.

Usage:  PYTHONPATH=. python3 -m src.experiments.tpds.taco_r2_stats_m6a_workload_weights [--n-boot 2000]
        [--trace-cache FILE]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from src.experiments.tpds import deployment_units as DU  # noqa: E402
from src.experiments.tpds import e3b_information as E3B  # noqa: E402
from src.experiments.tpds import serving_key_churn as SKC  # noqa: E402
from src.experiments.tpds import taco_r2_stats_common as C  # noqa: E402
from src.experiments.tpds import taco_report_numbers as TRN  # noqa: E402
from src.experiments.tpds.common import RESULTS_DIR, utc_now  # noqa: E402

W1_WINDOWS = [(tr, w) for tr, w, _ in SKC.WINDOWS]
W2_WINDOWS = [("azure2023_code", "1h"), ("azure2023_conv", "1h"), ("burstgpt", "24h_busiest"),
              ("azure2024_code", "1h"), ("azure2024_conv", "1h")]
W2_BUDGETS = (2048, 8192)
W2_RHO = SKC.RHO_PRIMARY
M1_DATA = ("pro6000_device", "vgpu3090_device", "h800_device")
M1_POLICIES = ("fixed_chunk_256", "best_static_per_batch_in_sample", "free_length_rule", "shape_tree_held_out",
               "exact_table_held_out", "cost_model")
GRAPH_POLICIES = ("default", "exact_table_held_out")

# unweighted values printed in the manuscript (Tables 5 and 7) or quoted by the review, at printed precision
PRINTED_M1 = {  # name: {policy: (mean %, LW % or None)}
    "pro6000_device": {"fixed_chunk_256": (48.6, None), "best_static_per_batch_in_sample": (20.0, None),
                       "free_length_rule": (5.86, 5.96), "shape_tree_held_out": (0.46, None),
                       "exact_table_held_out": (0.14, None), "cost_model": (0.40, None)},
    "vgpu3090_device": {"fixed_chunk_256": (96.4, None), "best_static_per_batch_in_sample": (17.4, None),
                        "free_length_rule": (7.76, 16.31), "shape_tree_held_out": (0.38, None),
                        "exact_table_held_out": (0.35, None), "cost_model": (3.52, None)},
    "h800_device": {"fixed_chunk_256": (50.9, None), "best_static_per_batch_in_sample": (24.7, None),
                    "free_length_rule": (10.66, 13.94), "shape_tree_held_out": (1.63, None),
                    "exact_table_held_out": (0.20, None), "cost_model": (3.01, None)},
}
PRINTED_GRAPH = {  # name: (default mean %, default LW %, recoverable LW % of the review)
    "ssd370m_graph": (19.1, 2.8, 2.79), "ssd370m_graph_pro6000": (10.1, 1.7, 1.72),
    "ssd2p7b_graph": (22.3, 2.2, 2.17), "ssd2p7b_graph_pro6000": (19.2, 5.9, 5.39),
    "gdn1p4b_graph": (16.2, 1.0, 0.45), "gdn1p4b_graph_pro6000": (23.9, 3.2, 2.94),
    "m1_370m_graph": (31.8, 17.9, None), "m1_370m_graph_pro6000": (46.0, 32.3, None),
}


def _decimals(v: float) -> int:
    s = f"{v}"
    return len(s.split(".")[1]) if "." in s else 0


class Checks:
    def __init__(self):
        self.rows = []

    def printed(self, what: str, value: float, printed: float, location: str) -> None:
        nd = _decimals(printed)
        ok = round(value, nd) == round(printed, nd) or abs(value - printed) <= 0.5 * 10 ** (-nd) + 1e-9
        self.rows.append({"check": what, "computed": value, "printed": printed, "location": location, "match": bool(ok)})

    def exact(self, what: str, value, reference, source: str, tol: float = 1e-9) -> None:
        if isinstance(value, (int, float)) and isinstance(reference, (int, float)):
            ok = abs(float(value) - float(reference)) <= tol * max(1.0, abs(float(reference)))
        else:
            ok = value == reference
        self.rows.append({"check": what, "computed": value, "reference": reference, "source": source, "match": bool(ok)})

    def summary(self) -> dict:
        return {"n": len(self.rows), "n_match": sum(r["match"] for r in self.rows),
                "mismatches": [r for r in self.rows if not r["match"]]}


# =========================================================================== trace-side distributions
def _bucket(L: np.ndarray) -> np.ndarray:
    L = np.maximum(np.asarray(L, np.int64), 1)
    b = np.floor(np.log2(L)).astype(np.int64)
    out = np.left_shift(np.int64(1), b)
    # guard float rounding at exact powers of two
    out = np.where(out * 2 <= L, out * 2, out)
    out = np.where(out > L, out // 2, out)
    return out


def _dist(L: np.ndarray, B: np.ndarray) -> dict:
    """Counts per (bucket, batch, aligned) as a list of [bucket, batch, aligned, count]."""
    b = _bucket(L)
    a = (np.asarray(L) % 16 == 0).astype(np.int64)
    keys = np.stack([b, np.asarray(B, np.int64), a], 1)
    u, c = np.unique(keys, axis=0, return_counts=True)
    return {"cells": [[int(x), int(y), int(z), int(n)] for (x, y, z), n in zip(u, c)], "n_calls": int(len(L))}


def trace_side_summary(d: dict) -> dict:
    cells = np.array(d["cells"], dtype=np.int64)
    n = cells[:, 3].astype(float)
    tot = n.sum()
    return {"n_calls": int(tot), "share_batch1": float(n[cells[:, 1] == 1].sum() / tot),
            "share_batch_le16": float(n[cells[:, 1] <= 16].sum() / tot),
            "share_aligned": float(n[cells[:, 2] == 1].sum() / tot),
            "share_bucket_ge_8192": float(n[cells[:, 0] >= 8192].sum() / tot),
            "share_bucket_lt_16": float(n[cells[:, 0] < 16].sum() / tot),
            "mean_batch": float((cells[:, 1] * n).sum() / tot)}


def build_trace_distributions(args, checks: Checks, log) -> dict:
    """W1 and W2 distributions per window (loads every trace once)."""
    if args.trace_cache and Path(args.trace_cache).exists():
        log(f"trace distributions from cache {args.trace_cache}")
        return json.loads(Path(args.trace_cache).read_text())
    stored = C.load_json(args.in_dir / "serving_key_churn.json") or {}
    stored_res = {(r["trace"], r["window"], r["T"]): r for r in stored.get("results", [])
                  if r.get("policy") == "chunked_budget" and r.get("sched") == "colocated" and r.get("rho") == W2_RHO
                  and r.get("kernel") == "padded"}
    out = {"W1": {}, "W2": {}, "provenance": {}}
    traces = list(dict.fromkeys(tr for tr, _ in W1_WINDOWS))
    for tr in traces:
        t0 = time.time()
        data = SKC.load_trace(tr, args.data_dir)
        log(f"  {tr}: {len(data['t'])} requests loaded in {time.time() - t0:.0f} s")
        out["provenance"][tr] = {"file": SKC.TRACE_FILES[tr]["file"], "source": SKC.TRACE_FILES[tr]["source"],
                                 "requests": int(len(data["t"]))}
        for tr2, window in W1_WINDOWS:
            if tr2 != tr:
                continue
            win, info = SKC.select_window(data, window)
            key = f"{tr}|{window}"
            out["W1"][key] = {**_dist(win["ctx"], np.ones(len(win["ctx"]), np.int64)), "n_requests": info["n_requests"]}
            # check against the stored per-window bucket histogram of the replay
            ref = stored.get("traces", {}).get(tr, {}).get("windows", {}).get(window, {}).get("prompt_len_pow2_bucket_hist")
            if ref is not None:
                mine = defaultdict(int)
                for b, _, _, n in out["W1"][key]["cells"]:
                    mine[str(b)] += n
                checks.exact(f"W1.{key}.bucket_histogram_equals_stored", dict(sorted(mine.items(), key=lambda kv: int(kv[0]))),
                             dict(sorted(ref.items(), key=lambda kv: int(kv[0]))),
                             f"serving_key_churn.json traces.{tr}.windows.{window}.prompt_len_pow2_bucket_hist")
            if (tr, window) in W2_WINDOWS:
                tok = float(win["ctx"].sum())
                s = tok * SKC.T_PF_S / (info["duration_s"] * W2_RHO)
                arr = win["t"] * s
                for T in W2_BUDGETS:
                    t1 = time.time()
                    sim = SKC.simulate_chunked(arr, win["ctx"], win["gen"], T, colocate=True)
                    out["W2"][f"{key}|T{T}"] = {**_dist(sim["lmax"], sim["nseq"]),
                                                "sim": {k: sim["stats"][k] for k in ("prefill_steps",
                                                                                     "mean_seqs_per_prefill_step",
                                                                                     "mean_prefill_tokens_per_step")}}
                    ref = stored_res.get((tr, window, T))
                    if ref is not None:
                        for k in ("prefill_steps", "mean_seqs_per_prefill_step"):
                            checks.exact(f"W2.{key}|T{T}.sim.{k}", sim["stats"][k], ref["sim"][k],
                                         f"serving_key_churn.json results ({tr}, {window}, chunked_budget, T={T}, "
                                         "colocated, rho 0.5, padded) sim", tol=1e-9)
                    log(f"    {key} T={T}: {sim['stats']['prefill_steps']} steps in {time.time() - t1:.0f} s")
            del win
        del data
        gc.collect()
    if args.trace_cache:
        Path(args.trace_cache).write_text(json.dumps(out))
    return out


# =========================================================================== strata and weights
class Strata:
    """Maps a trace distribution onto the captured calls of one table."""

    def __init__(self, bucket: np.ndarray, batch: np.ndarray, aligned: np.ndarray):
        self.bucket, self.batch, self.aligned = (np.asarray(x, np.int64) for x in (bucket, batch, aligned))
        self.cells = defaultdict(int)
        for b, n in zip(self.bucket, self.batch):
            self.cells[(int(b), int(n))] += 1
        self.batches = sorted({n for _, n in self.cells})
        self.buckets_at = {n: sorted({b for b, m in self.cells if m == n}) for n in self.batches}
        self.n_aligned = defaultdict(int)
        for b, n, a in zip(self.bucket, self.batch, self.aligned):
            self.n_aligned[(int(b), int(n), int(a))] += 1

    @staticmethod
    def _nearest(x: int, options: list[int]) -> list[int]:
        d = np.array([abs(np.log2(x) - np.log2(o)) for o in options])
        return [o for o, v in zip(options, d) if v <= d.min() + 1e-9]

    def weights(self, dist: dict, align_aware: bool) -> tuple[dict, dict]:
        """Stratum masses {stratum: mass}; stratum = (bucket, batch, aligned or -1 for the whole cell)."""
        W = defaultdict(float)
        tot = remap_batch = remap_bucket = remap_any = fallback = 0.0
        for b, n, a, cnt in dist["cells"]:
            m = float(cnt)
            tot += m
            bt = self._nearest(n, self.batches)
            moved_b = n not in self.batches
            for B in bt:
                mb = m / len(bt)
                bk = self._nearest(b, self.buckets_at[B])
                moved_k = b not in self.buckets_at[B]
                for K in bk:
                    mk = mb / len(bk)
                    remap_batch += mk * moved_b
                    remap_bucket += mk * moved_k
                    remap_any += mk * (moved_b or moved_k)
                    if align_aware and self.n_aligned.get((K, B, a), 0) > 0:
                        W[(K, B, a)] += mk
                    else:
                        if align_aware:
                            fallback += mk
                        W[(K, B, -1)] += mk
        tot = max(tot, 1e-300)
        info = {"trace_calls": int(tot), "share_mass_remapped": remap_any / tot,
                "share_mass_batch_remapped": remap_batch / tot, "share_mass_bucket_remapped": remap_bucket / tot}
        if align_aware:
            info["share_mass_alignment_fallback"] = fallback / tot
        return {k: v / tot for k, v in W.items()}, info

    def membership(self, strata: list) -> np.ndarray:
        """(n_strata, n_calls) boolean membership."""
        G = np.zeros((len(strata), len(self.bucket)), dtype=bool)
        for s, (K, B, a) in enumerate(strata):
            m = (self.bucket == K) & (self.batch == B)
            if a >= 0:
                m &= self.aligned == a
            G[s] = m
        return G


def post_stratified(arrays: dict, G: np.ndarray, W: np.ndarray, clusters: np.ndarray, M: np.ndarray,
                    stats: dict) -> tuple[dict, dict]:
    """Point estimates and bootstrap replicates of post-stratified statistics.

    ``arrays``: per-call arrays; ``stats``: name -> (kind, args) with kind 'mean' (array) or 'ratio'
    (numerator array, denominator array) or 'diff' of two (stat names)."""
    K = M.shape[1]
    nS = G.shape[0]
    # stratum x cluster sums
    cnt = np.zeros((nS, K))
    for s in range(nS):
        np.add.at(cnt[s], clusters[G[s]], 1.0)
    sums = {}
    for name, x in arrays.items():
        S = np.zeros((nS, K))
        for s in range(nS):
            np.add.at(S[s], clusters[G[s]], x[G[s]])
        sums[name] = S
    Ones = np.ones((1, K))

    def strat_means(Mx):
        n = Mx @ cnt.T                       # (r, nS)
        ok = n > 0
        Wr = np.where(ok, W[None, :], 0.0)
        Wr = Wr / Wr.sum(1, keepdims=True)
        means = {name: np.where(ok, (Mx @ S.T) / np.where(ok, n, 1.0), 0.0) for name, S in sums.items()}
        lost = 1.0 - np.where(ok, W[None, :], 0.0).sum(1) / W.sum()
        return means, Wr, lost

    out_pt, out_rep = {}, {}
    for tag, Mx in (("point", Ones), ("reps", M)):
        means, Wr, lost = strat_means(Mx)
        res = {}
        for name, (kind, a) in stats.items():
            if kind == "mean":
                res[name] = 100.0 * (Wr * means[a]).sum(1)
            elif kind == "ratio":
                res[name] = 100.0 * (Wr * means[a[0]]).sum(1) / (Wr * means[a[1]]).sum(1)
        for name, (kind, a) in stats.items():
            if kind == "diff":
                res[name] = res[a[0]] - res[a[1]]
        if tag == "point":
            out_pt = {k: float(v[0]) for k, v in res.items()}
            out_pt["_weight_lost"] = float(lost[0])
        else:
            out_rep = res
            out_rep["_lost"] = lost
    return out_pt, out_rep


# =========================================================================== data sets
class DataSet:
    def __init__(self, name: str, label: str, gpu: str, df, lat: np.ndarray, picks: dict, kind: str, unw_label: str):
        self.name, self.label, self.gpu, self.kind, self.unw_label = name, label, gpu, kind, unw_label
        self.df = df.reset_index(drop=True)
        self.lat = lat
        self.orc = lat.min(1)
        self.picks = picks
        self.clusters = E3B.prompt_clusters(self.df)[0]
        L = self.df["L"].to_numpy()
        self.strata = Strata(self.df["L_bucket"].to_numpy(), self.df["batch"].to_numpy(), (L % 16 == 0).astype(int))
        self.arrays = {"orc": self.orc}
        for p, idx in picks.items():
            ch = lat[np.arange(len(self.orc)), np.asarray(idx)]
            self.arrays[f"{p}|exc"] = ch - self.orc
            self.arrays[f"{p}|reg"] = ch / self.orc - 1.0
        self.stats = {}
        for p in picks:
            self.stats[f"{p}|mean_regret_pct"] = ("mean", f"{p}|reg")
            self.stats[f"{p}|lw_share_of_oracle_pct"] = ("ratio", (f"{p}|exc", "orc"))
        if kind == "graph":
            self.stats["recoverable|mean_regret_pp"] = ("diff", ("default|mean_regret_pct", "exact_table_held_out|mean_regret_pct"))
            self.stats["recoverable|lw_share_of_oracle_pp"] = ("diff", ("default|lw_share_of_oracle_pct",
                                                                       "exact_table_held_out|lw_share_of_oracle_pct"))

    def evaluate(self, W: dict | None, rng: np.random.Generator, n_boot: int) -> dict:
        cl = self.clusters
        _, cl = np.unique(cl, return_inverse=True)
        cl = cl.ravel()
        K = int(cl.max()) + 1
        M = rng.multinomial(K, np.full(K, 1.0 / K), size=n_boot).astype(float)
        if W is None:
            strata = [(0, 0, -2)]
            G = np.ones((1, len(cl)), dtype=bool)
            Wv = np.ones(1)
        else:
            strata = sorted(W)
            G = self.strata.membership(strata)
            Wv = np.array([W[s] for s in strata])
        pt, reps = post_stratified(self.arrays, G, Wv, cl, M, self.stats)
        rec = {}
        for name in self.stats:
            rec[name] = [pt[name], *C.pct_ci(reps[name])]
        rec["_empty_stratum"] = {"share_replicates_with_empty_stratum": float(np.mean(reps["_lost"] > 0)),
                                 "max_weight_lost": float(reps["_lost"].max()),
                                 "mean_weight_lost": float(reps["_lost"].mean()),
                                 "share_replicates_losing_gt_10pct_weight": float(np.mean(reps["_lost"] > 0.10))}
        # effective sample size of the per-call weights (Kish) and concentration on prompt clusters
        n_s = G.sum(1).astype(float)
        w = G.T.astype(float) @ (Wv / np.maximum(n_s, 1.0))
        cw = np.bincount(cl, weights=w, minlength=K)
        rec["_diag"] = {"kish_n_eff_calls": float(w.sum() ** 2 / (w ** 2).sum()), "n_calls_weighted": int((w > 0).sum()),
                        "n_clusters_weighted": int((cw > 0).sum()),
                        "kish_n_eff_clusters": float(cw.sum() ** 2 / (cw ** 2).sum()),
                        "max_cluster_weight_share": float(cw.max() / cw.sum()),
                        "share_weight_on_strata_below_10_calls": float(Wv[n_s < 10].sum() / Wv.sum())}
        return rec


def m1_dataset(name: str, in_dir: Path) -> DataSet:
    spec = TRN.M1_SETS[name]
    t = TRN.load_table(in_dir, spec["csv"], spec["rmed"], name, "mamba1", spec["cal"])
    bp = TRN.base_picks(t)
    L = t.df["L"].to_numpy()
    target = np.clip(2 ** np.ceil(np.log2(np.maximum(L, 1))).astype(int), 32, 1024)
    picks = {"fixed_chunk_256": bp["default_chunk_256"],
             "best_static_per_batch_in_sample": bp["best_static_per_batch_in_sample"],
             "free_length_rule": np.array([t.chunks.index(int(c)) for c in target]),
             "shape_tree_held_out": bp["shape"],
             "exact_table_held_out": bp["exact_shape_table_held_out"]}
    cm, info = E3B.cost_model_choice(t, in_dir)
    if cm is None:
        raise SystemExit(f"{name}: no stored cost-model choice ({info})")
    picks["cost_model"] = cm
    return DataSet(name, f"Triton Mamba-1 370M, {spec['gpu']}, {spec['convention']}", spec["gpu"], t.df, t.lat, picks,
                   "m1", name)


def graph_dataset(name: str, in_dir: Path) -> DataSet:
    t = C.graph_table(name, in_dir)
    gpu, kernel = C.GRAPH_TABLES[name]
    picks = {"default": np.full(len(t.df), t.default_idx()), "exact_table_held_out": DU.cv_exact_pick(t.df, t.lat)}
    ds = DataSet(name, f"{kernel}, {gpu}, CUDA-graph replay (three rounds, round medians)", gpu, t.df, t.lat, picks,
                 "graph", name)
    ds.default_chunk = int(t.chunks[t.default_idx()])
    return ds


# =========================================================================== assembly
def _fmt(v) -> str:
    return f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"


def compact_families(per: dict, families: dict) -> dict:
    """Per family: per data set and statistic the unweighted value and the min-max over windows (points and CIs)."""
    out = {}
    for fam, wins in families.items():
        f = {}
        for ds, recs in per.items():
            unw = recs["unweighted"]
            rows = {}
            for stat in unw:
                if stat.startswith("_"):
                    continue
                vals = [recs[w][stat] for w in wins if w in recs]
                if not vals:
                    continue
                pts = [v[0] for v in vals]
                rows[stat] = {"unweighted": [round(x, 3) for x in unw[stat]],
                              "weighted_point_min_max": [round(min(pts), 3), round(max(pts), 3)],
                              "weighted_ci_lo_min_hi_max": [round(min(v[1] for v in vals), 3),
                                                            round(max(v[2] for v in vals), 3)]}
            f[ds] = rows
        out[fam] = f
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--data-dir", type=Path, default=SKC.DATA_DIR)
    p.add_argument("--n-boot", type=int, default=C.N_BOOT)
    p.add_argument("--trace-cache", type=Path, default=None,
                   help="optional JSON cache of the trace-side distributions (written if absent, read if present)")
    args = p.parse_args(argv)
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    t0 = time.time()
    checks = Checks()

    log("trace-side distributions")
    tdist = build_trace_distributions(args, checks, log)

    log("captured-call data sets")
    data = {n: m1_dataset(n, args.in_dir) for n in M1_DATA}
    for n in C.GRAPH_TABLES:
        data[n] = graph_dataset(n, args.in_dir)

    weightings = {}            # label -> (family, window key, dist, align_aware)
    for key, d in tdist["W1"].items():
        weightings[f"W1|{key}"] = ("W1_request_level_batch1", key, d, False)
        weightings[f"W1a|{key}"] = ("W1_request_level_batch1_alignment_aware", key, d, True)
    for key, d in tdist["W2"].items():
        T = key.split("|")[-1]
        weightings[f"W2|{key}"] = (f"W2_chunked_prefill_padded_{T}", key, d, False)
        weightings[f"W2a|{key}"] = (f"W2_chunked_prefill_padded_{T}_alignment_aware", key, d, True)

    per, mapping = {}, {}
    for n, ds in data.items():
        recs = {"unweighted": ds.evaluate(None, C.make_rng(ds.unw_label, "all"), args.n_boot)}
        # reference for W1: the captured batch-1 calls with their own (captured) length mix
        n1 = sum(c for (b, m), c in ds.strata.cells.items() if m == 1)
        W0 = {(b, 1, -1): c / n1 for (b, m), c in ds.strata.cells.items() if m == 1}
        recs["W0|captured_batch1"] = ds.evaluate(W0, C.make_rng("m6a", n, "W0"), args.n_boot)
        mapping[n] = {}
        for wl, (fam, key, d, aa) in weightings.items():
            W, info = ds.strata.weights(d, aa)
            recs[wl] = ds.evaluate(W, C.make_rng("m6a", n, wl), args.n_boot)
            mapping[n][wl] = {k: round(v, 4) for k, v in info.items()}
        per[n] = recs
        log(f"  {n}: {len(weightings)} weightings ({time.time() - t0:.0f} s)")

    # reproduction of the unweighted values
    for n, pol in PRINTED_M1.items():
        u = per[n]["unweighted"]
        for p_, (mean, lw) in pol.items():
            checks.printed(f"unweighted.{n}.{p_}.mean_regret_pct", u[f"{p_}|mean_regret_pct"][0], mean,
                           "Table 5 / review M2b")
            if lw is not None:
                checks.printed(f"unweighted.{n}.{p_}.lw_share_of_oracle_pct", u[f"{p_}|lw_share_of_oracle_pct"][0], lw,
                               "review M2b (free length rule, latency-weighted)")
    trn = C.load_json(args.in_dir / "taco_report_numbers.json")
    e26 = C.load_json(args.in_dir / "taco_e26_analysis_pro6000.json")
    for n, (mean, lw, rec) in PRINTED_GRAPH.items():
        u = per[n]["unweighted"]
        checks.printed(f"unweighted.{n}.default.mean_regret_pct", u["default|mean_regret_pct"][0], mean, "Table 7")
        checks.printed(f"unweighted.{n}.default.lw_share_of_oracle_pct", u["default|lw_share_of_oracle_pct"][0], lw, "Table 7")
        if rec is not None:
            checks.printed(f"unweighted.{n}.recoverable.lw_pp", u["recoverable|lw_share_of_oracle_pp"][0], rec,
                           "review M3a (R1 recomputation)")
        src = (trn or {}).get("M4b_latency_weighted", {}).get("a_cuda_graph", {}).get(n) or (e26 or {}).get("tables", {}).get(n)
        if src:
            pol = src["subsets"]["all"]["policies"]
            for mine, theirs in (("default", "default"), ("exact_table_held_out", "exact_table_held_out")):
                checks.exact(f"unweighted.{n}.{mine}.lw_share_of_oracle_pct", u[f"{mine}|lw_share_of_oracle_pct"][0],
                             pol[theirs]["lw_share_of_oracle_pct"], "taco_report_numbers.json / taco_e26_analysis_pro6000.json")
            if n in (trn or {}).get("M4b_latency_weighted", {}).get("a_cuda_graph", {}):
                ci = pol["default"]["ci95"]["lw_share_of_oracle_pct"]
                checks.exact(f"unweighted.{n}.default.lw_ci95_lo", u["default|lw_share_of_oracle_pct"][1], ci[0],
                             "taco_report_numbers.json (same bootstrap replicates)", tol=1e-9)
                checks.exact(f"unweighted.{n}.default.lw_ci95_hi", u["default|lw_share_of_oracle_pct"][2], ci[1],
                             "taco_report_numbers.json (same bootstrap replicates)", tol=1e-9)

    families = defaultdict(list)
    families["W0_captured_batch1_reference"].append("W0|captured_batch1")
    for wl, (fam, key, d, aa) in weightings.items():
        families[fam].append(wl)
    compact = compact_families(per, families)

    # headline: per window, the default / table / recoverable of the graph tables and the Mamba-1 policies
    headline = {}
    for wl, (fam, key, d, aa) in weightings.items():
        if aa:
            continue
        row = {}
        for n, ds in data.items():
            r = per[n][wl]
            if ds.kind == "graph":
                row[n] = {"default_mean": _fmt(r["default|mean_regret_pct"]),
                          "default_lw": _fmt(r["default|lw_share_of_oracle_pct"]),
                          "table_lw": _fmt(r["exact_table_held_out|lw_share_of_oracle_pct"]),
                          "recoverable_lw_pp": _fmt(r["recoverable|lw_share_of_oracle_pp"])}
            else:
                row[n] = {p_: f"{_fmt(r[f'{p_}|mean_regret_pct'])} / LW {_fmt(r[f'{p_}|lw_share_of_oracle_pct'])}"
                          for p_ in M1_POLICIES}
        headline[wl] = row
    headline["W0|captured_batch1"] = {
        n: ({"default_mean": _fmt(per[n]["W0|captured_batch1"]["default|mean_regret_pct"]),
             "default_lw": _fmt(per[n]["W0|captured_batch1"]["default|lw_share_of_oracle_pct"]),
             "table_lw": _fmt(per[n]["W0|captured_batch1"]["exact_table_held_out|lw_share_of_oracle_pct"]),
             "recoverable_lw_pp": _fmt(per[n]["W0|captured_batch1"]["recoverable|lw_share_of_oracle_pp"])}
            if data[n].kind == "graph" else
            {p_: f"{_fmt(per[n]['W0|captured_batch1'][f'{p_}|mean_regret_pct'])} / LW "
                 f"{_fmt(per[n]['W0|captured_batch1'][f'{p_}|lw_share_of_oracle_pct'])}" for p_ in M1_POLICIES})
        for n in data}
    headline["unweighted"] = {
        n: ({"default_mean": _fmt(per[n]["unweighted"]["default|mean_regret_pct"]),
             "default_lw": _fmt(per[n]["unweighted"]["default|lw_share_of_oracle_pct"]),
             "table_lw": _fmt(per[n]["unweighted"]["exact_table_held_out|lw_share_of_oracle_pct"]),
             "recoverable_lw_pp": _fmt(per[n]["unweighted"]["recoverable|lw_share_of_oracle_pp"])}
            if data[n].kind == "graph" else
            {p_: f"{_fmt(per[n]['unweighted'][f'{p_}|mean_regret_pct'])} / LW "
                 f"{_fmt(per[n]['unweighted'][f'{p_}|lw_share_of_oracle_pct'])}" for p_ in M1_POLICIES})
        for n in data}

    res = {
        "item": "M6a (taco-r2): workload-weighted (post-stratified) headline numbers",
        "created_utc": utc_now(), "definitions": __doc__,
        "settings": {"n_boot": args.n_boot, "seed": C.SEED, "w2_rho": W2_RHO, "w2_budgets": list(W2_BUDGETS),
                     "w2_scheduler": "colocated decodes, FCFS, max 256 running requests, step model 2 ms + 10 us per "
                                     "prefill token + 20 us per decoding sequence (the replay's)",
                     "batch_tie_rule": "nearest captured batch in log2; exact ties (2, 8) split equally",
                     "bucket_tie_rule": "nearest captured bucket in log2 at that batch; exact ties split equally",
                     "policies_not_reselected": "every policy is the unweighted one (no re-selection under the weights)"},
        "trace_side": {"W1": {k: {**trace_side_summary(v), "n_requests": v.get("n_requests")} for k, v in tdist["W1"].items()},
                       "W2": {k: {**trace_side_summary(v), "sim": v.get("sim")} for k, v in tdist["W2"].items()},
                       "provenance": tdist.get("provenance")},
        "captured_side": {n: {"label": ds.label, "n_calls": int(len(ds.orc)), "n_prompt_clusters":
                              int(np.unique(ds.clusters).size), "share_aligned": float(ds.strata.aligned.mean()),
                              "cells_bucket_batch": {f"{b}|{m}": c for (b, m), c in sorted(ds.strata.cells.items())},
                              **({"default_chunk": ds.default_chunk} if hasattr(ds, "default_chunk") else {})}
                          for n, ds in data.items()},
        "mapping": mapping,
        "headline_per_window": headline,
        "compact_by_family": compact,
        "per_data_set": {n: {w: {k: ([round(x, 4) for x in v] if isinstance(v, list) else v) for k, v in r.items()}
                             for w, r in recs.items()} for n, recs in per.items()},
        "checks": checks.rows, "checks_summary": checks.summary(), "runtime_s": round(time.time() - t0, 1)}
    path = C.write_result("m6a", res, args.in_dir)
    s = checks.summary()
    log(f"wrote {path}; checks {s['n_match']}/{s['n']} match; {time.time() - t0:.0f} s")
    for m in s["mismatches"]:
        log("  MISMATCH", m)
    print_summary(res)


def print_summary(res: dict) -> None:
    comp = res["compact_by_family"]
    for fam, rows in comp.items():
        print(f"\n== {fam}")
        for ds, stats in rows.items():
            keys = [k for k in stats if k.endswith("lw_share_of_oracle_pct") or k.endswith("_pp")
                    or k.endswith("mean_regret_pct")]
            for k in keys:
                v = stats[k]
                print(f"  {ds:24s} {k:52s} unw {v['unweighted'][0]:8.3f} [{v['unweighted'][1]:.2f}, {v['unweighted'][2]:.2f}]"
                      f"  w {v['weighted_point_min_max'][0]:8.3f}..{v['weighted_point_min_max'][1]:8.3f}"
                      f"  CI {v['weighted_ci_lo_min_hi_max'][0]:.2f}..{v['weighted_ci_lo_min_hi_max'][1]:.2f}")


if __name__ == "__main__":
    main()
