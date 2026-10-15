"""Report numbers for the ACM TACO revision, cycle taco-r1 (review items M4a, M4b, M4d, M4e, M3d, M7a, M10b,
N2, N3, N8).

CPU only and deterministic. Reads stored result files and times nothing. The only refits are those of the
calibration-only chunk selectors of ``cost_model_baselines_r4`` (their per-call choices are not stored): they
are fitted exactly as there (``Ctx`` + ``fit_selectors``) and checked against the stored summary. Output:
``taco_report_numbers.json`` in the results directory and a readable summary on stdout.

Conventions
-----------
regret            per-call relative regret T(c) / T(c*) - 1, c* the per-call oracle of the view. For the
                  three-round sweeps the view is ``analysis.round_median_view`` (latencies and oracle on the
                  medians of the round means, CI ties by between-round CI overlap). Reported: mean, p95 and max
                  (percent), the shares of calls within 2% and 5% of the oracle latency, the CI-tie share.
latency-weighted  lw_share_of_oracle_pct = 100 sum(excess latency) / sum(oracle latency) (the definition of
                  the review); lw_share_of_policy_pct = 100 sum(excess) / sum(latency of the policy) (the
                  definition of the manuscript). Both weight long calls more than the per-call mean.
absolute          us_per_call = mean excess latency per scan call; ms_per_forward_pass = us_per_call x scan
                  layers / 1000 (48 for Mamba-1 370M and 1.4B and for Mamba-2 370M, 64 for Mamba-1 2.8B and
                  Mamba-2 2.7B, 12 Gated DeltaNet layers in the 1.4B hybrid, as ``deployment_units``);
                  oracle_scan_ms_per_forward_pass = mean oracle latency x layers, the scan time of one forward
                  pass at the per-call oracle (the denominator of the share of the forward-pass scan time).
intervals         95% percentile intervals over ``--n-boot`` prompt-cluster bootstrap replicates
                  (``e3b_information.prompt_clusters`` and ``cluster_bootstrap``); per batch the clusters are
                  resampled among the calls of that batch (as ``tables_iv_v_ci``). Ratios (latency-weighted
                  shares, means over a subset of calls) are ratios of replicate means with common weights; paired
                  differences use the same replicates. Tensor-parallel shapes and per-shape means are resampled
                  by shape. Seeds: SeedSequence([SEED, crc32(label)]) per data set and subset.

Items
-----
M4b (1)   latency-weighted share and absolute regret next to the per-call headlines: (a) CUDA-graph replay
          of the production kernels on the RTX 3090 (three rounds, round-median oracle; library default,
          shape tree, held-out exact-shape tables, work-term model; all calls and above / below the graph
          launch floor); (b) the best single Triton Mamba-1 chunk per batch in device time on the three GPUs;
          (c) the cost model per GPU.
M4a (2)   selector tables of the 2360 captured 370M Mamba-1 calls per GPU in device time (primary) and
          launch-inclusive (secondary, flagged ``launch_inclusive``), and of the Mamba-2 370M calls.
M4d (3)   calibration-shape coverage: distinct shapes, calls on a calibration key, cost-model regret on and
          off the calibration keys, per-shape (unweighted) means with shape-level bootstrap intervals.
M4e (4)   cluster-bootstrap intervals of every selector row (all calls and per batch) and paired intervals of
          the cost model minus every other selector (inside the M4a records; compact table under ``M4e``).
M3d (5)   best static chunk per GPU and per (GPU, batch) as the untuned baseline, in deployment units and as
          data of the calibration-budget figure, next to the fixed chunk 256.
M7a (6)   tensor-parallel shard widths: pooled against pooled, per-shard maxima, retuning at the shard width
          (cross-round held-out regret) and its profiling cost.
M10b (7)  value invariance: inputs per kernel family, F-test degrees of freedom, pairwise equivalence bounds
          against the pooled-mean bounds.
N2/N3/N8 (8) normalized conditional mutual information, launch-floor coverage of SSD 2.7B and Gated DeltaNet,
          maximum next to the 95th-percentile regret of the cost model.

Every value that a stored analysis also reports is compared with it, and a set of values printed in the
manuscript is reproduced at its printed precision (``checks``). Quantities the files cannot give are listed
under ``unavailable`` with the reason.

Usage:  PYTHONPATH=. python3 -m src.experiments.tpds.taco_report_numbers [--jobs 6] [--n-boot 2000]
        [--cache-dir DIR] [--no-learned]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # deterministic BLAS; parallelism is across calibration tags

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import t as t_dist  # noqa: E402

from src.experiments.tpds import cost_model as CM  # noqa: E402
from src.experiments.tpds import cost_model_baselines as CMB  # noqa: E402
from src.experiments.tpds import cost_model_baselines_r4 as CMB4  # noqa: E402
from src.experiments.tpds import deployment_units as DU  # noqa: E402
from src.experiments.tpds import e1_analysis as E1A  # noqa: E402
from src.experiments.tpds import e12_analysis as E12  # noqa: E402
from src.experiments.tpds import e22_graph_analysis as E22  # noqa: E402
from src.experiments.tpds import e3_predictors as E3  # noqa: E402
from src.experiments.tpds import e3b_information as E3B  # noqa: E402
from src.experiments.tpds.analysis import load_e2, round_median_view  # noqa: E402
from src.experiments.tpds.common import MAMBA1_CHUNKS, RESULTS_DIR, utc_now, write_json  # noqa: E402
from src.experiments.tpds.e12_tp_widths import MODEL_WIDTHS, REF_CHUNKS  # noqa: E402

SEED = 20261015
MARGINS = (0.02, 0.05)
LAYERS_370M = 48

# calibration-only selectors of cost_model_baselines_r4 (fitted on the E4 / E1 calibration timings of one GPU)
CAL_SELECTORS = ("cost_model", "work_only", "work_term_model", "nearest_calibrated_key", "loglinear_interp_logL_logB",
                 "length_tree_calibration", "shape_tree_calibration", "gbr_basic", "gbr_structural", "mlp_basic",
                 "mlp_structural", "tfla_runtime")
LEARNED = ("gbr_basic", "gbr_structural", "mlp_basic", "mlp_structural")

SELECTOR_INFO = {
    "default_chunk_256": ("none", "fixed chunk 256 (the untuned choice of the deployment table and the calibration-budget figure)"),
    "best_static_global_in_sample": ("in sample", "one chunk for all calls, lowest mean per-call regret over all calls"),
    "best_static_global_by_total_latency": ("in sample", "one chunk for all calls, lowest total latency over all calls"),
    "best_static_global_held_out": ("held out", "one chunk chosen on the training prompt folds (E3 global_static)"),
    "best_static_per_batch_in_sample": ("in sample", "one chunk per batch size, lowest mean per-call regret (Table II)"),
    "best_static_per_batch_by_total_latency": ("in sample", "one chunk per batch size, lowest total latency"),
    "best_static_per_batch_held_out": ("held out", "one chunk per batch size chosen on the training folds (batch-only table)"),
    "entropy_rule": ("none", "entropy rule, C in [32, 512]"),
    "L_only": ("held out", "leaf-cost tree on the length (depth 3), grouped 5-fold CV"),
    "shape": ("held out", "leaf-cost tree on (L, batch, width, layer), depth 4"),
    "shape+entropy": ("held out", "leaf-cost tree on shape and entropy, depth 4"),
    "all": ("held out", "leaf-cost tree on shape and all activation statistics, depth 4"),
    "entropy_only": ("held out", "leaf-cost tree on entropy, depth 4"),
    "moments": ("held out", "leaf-cost tree on the activation moments, depth 4"),
    "exact_shape_table_held_out": ("held out", "exact-shape table, regret-optimal entries chosen on the training folds "
                                               "(bucket and global fallback; r4_rounds_analysis / e3b_information)"),
    "exact_shape_table_held_out_mean_latency": ("held out", "exact-shape table, entries by mean latency on the training "
                                                            "folds (deployment_units / e22 estimator)"),
    "exact_shape_floor_in_sample": ("in sample", "exact-shape table chosen on all calls (regret-optimal entries)"),
    "cost_model": ("none", "Eq. (5) with the stored parameters of the GPU's calibration (frozen form)"),
    "work_only": ("none", "three-parameter restriction of Eq. (5)"),
    "work_term_model": ("none", "work-term model (F^p + W^p)^(1/p) fitted on the calibration timings"),
    "nearest_calibrated_key": ("none", "best default-launch chunk of the nearest calibration key"),
    "loglinear_interp_logL_logB": ("none", "bilinear interpolation of log latency in (log2 L, log2 B) over the 4 x 3 calibration grid"),
    "length_tree_calibration": ("none", "regret-labelled tree on log2 L trained on the 12 calibration shapes"),
    "shape_tree_calibration": ("none", "regret-labelled tree on (log2 L, log2 B) trained on the 12 calibration shapes"),
    "gbr_basic": ("none", "gradient-boosted regressor on shape, launch and compile facts (calibration timings)"),
    "gbr_structural": ("none", "gradient-boosted regressor with the derived quantities of Eq. (5)"),
    "mlp_basic": ("none", "MLP (2 x 64) on shape, launch and compile facts (calibration timings)"),
    "mlp_structural": ("none", "MLP with the derived quantities of Eq. (5)"),
    "tfla_runtime": ("none", "TFLA runtime expression fitted on the calibration timings"),
    "cost_model_top2": ("profiles 2 chunks per shape", "cost-model ranking, 2 best-ranked chunks profiled per exact shape"),
    "cost_model_top3": ("profiles 3 chunks per shape", "cost-model ranking, 3 best-ranked chunks profiled per exact shape"),
    "all_chunks_profiled": ("in sample", "all chunks profiled per exact shape (mean-latency argmin; top-k with k = all)"),
}

# Mamba-1 370M data sets (the same 2360 captured calls on every GPU and in every convention)
M1_SETS = {
    "pro6000_device": dict(csv="e2_mamba1_pro6000_r4spin.csv", rmed=True, cal="pro6000_r4spin", gpu="RTX PRO 6000",
                           convention="device time, medians of three shuffled rounds", launch_inclusive=False, primary=True),
    "vgpu3090_device": dict(csv="e2_mamba1_vgpu3090_r4spin.csv", rmed=True, cal="vgpu3090_r3spin", gpu="RTX 3090",
                            convention="device time, medians of three shuffled rounds (calibration: device-time E4/E1 "
                                       "of the same GPU)", launch_inclusive=False, primary=True),
    "h800_device": dict(csv="e2_mamba1_h800_r3spin.csv", rmed=False, cal="h800_r3spin", gpu="H800 PCIe",
                        convention="device time, one block per chunk (no three-round sweep exists on this GPU)",
                        launch_inclusive=False, primary=True),
    "vgpu3090_device_one_block": dict(csv="e2_mamba1_vgpu3090_r3spin.csv", rmed=False, cal="vgpu3090_r3spin",
                                      gpu="RTX 3090", convention="device time, one block per chunk (E15 session)",
                                      launch_inclusive=False, primary=False),
    "pro6000_launch": dict(csv="e2_mamba1_pro6000.csv", rmed=False, cal="pro6000", gpu="RTX PRO 6000",
                           convention="launch-inclusive, session 1, one block per chunk", launch_inclusive=True, primary=False),
    "vgpu3090_launch": dict(csv="e2_mamba1_vgpu3090.csv", rmed=False, cal="vgpu3090", gpu="RTX 3090",
                            convention="launch-inclusive, one block per chunk", launch_inclusive=True, primary=False),
    "h800_launch": dict(csv="e2_mamba1_h800.csv", rmed=False, cal="h800", gpu="H800 PCIe",
                        convention="launch-inclusive, one block per chunk", launch_inclusive=True, primary=False),
    "pro6000_launch_session2": dict(csv="e2_mamba1_pro6000_r4plain.csv", rmed=True, cal="pro6000", gpu="RTX PRO 6000",
                                    convention="launch-inclusive, session 2, medians of three shuffled rounds",
                                    launch_inclusive=True, primary=False),
    "vgpu3090_launch_session2": dict(csv="e2_mamba1_vgpu3090_r4plain.csv", rmed=True, cal="vgpu3090", gpu="RTX 3090",
                                     convention="launch-inclusive, session 2, medians of three shuffled rounds",
                                     launch_inclusive=True, primary=False),
}
PRIMARY_M1 = ("pro6000_device", "vgpu3090_device", "h800_device")
LAUNCH_M1 = ("pro6000_launch", "vgpu3090_launch", "h800_launch")

# Mamba-2 SSD 370M data sets; the work-term model uses the stored V4 parameters of the GPU (eager calibration)
M2_SETS = {
    "mamba2_pro6000_launch": dict(csv="e2_mamba2_pro6000.csv", rmed=False, gpu_tag="pro6000", gpu="RTX PRO 6000",
                                  convention="launch-inclusive, one block per chunk", launch_inclusive=True),
    "mamba2_vgpu3090_launch": dict(csv="e2_mamba2_vgpu3090.csv", rmed=False, gpu_tag="vgpu3090", gpu="RTX 3090",
                                   convention="launch-inclusive, one block per chunk", launch_inclusive=True),
    "mamba2_vgpu3090_device_one_block": dict(csv="e2_mamba2_vgpu3090_r3spin.csv", rmed=False, gpu_tag="vgpu3090",
                                             gpu="RTX 3090", convention="device time, one block per chunk (E15b; the "
                                             "manuscript does not use these timings: two regimes mixed)",
                                             launch_inclusive=False),
    "mamba2_vgpu3090_graph": dict(csv="e2_mamba2_vgpu3090_r4graph3.csv", rmed=True, gpu_tag="vgpu3090", gpu="RTX 3090",
                                  convention="CUDA-graph replay, warm L2, medians of three rounds (device-bound)",
                                  launch_inclusive=False),
}

# CUDA-graph tables of e22_graph_analysis (three rounds after use_three_rounds)
GRAPH_SETS = ("ssd370m_graph", "ssd2p7b_graph", "gdn1p4b_graph", "m1_370m_graph")

# larger Triton Mamba-1 checkpoints (device time, one block), rows of the deployment table
LARGE_SETS = {"mamba1_1p4b_device": ("e2_mamba1_vgpu3090_1p4b_r3.csv", "mamba1_triton_1p4b_vgpu3090", 48, "Mamba-1 1.4B"),
              "mamba1_2p8b_device": ("e2_mamba1_vgpu3090_2p8b_r3.csv", "mamba1_triton_2p8b_vgpu3090", 64, "Mamba-1 2.8B")}

E1_TAGS = ("pro6000", "vgpu3090", "h800", "h800_r3spin", "pro6000_r4spin", "vgpu3090_r3spin")

M4A_ROWS = ("best_static_global_held_out", "best_static_per_batch_in_sample", "default_chunk_256", "entropy_rule",
            "entropy_only", "shape", "shape+entropy", "exact_shape_table_held_out", "cost_model", "work_only",
            "nearest_calibrated_key", "loglinear_interp_logL_logB", "length_tree_calibration", "shape_tree_calibration",
            "gbr_basic", "gbr_structural", "mlp_basic", "mlp_structural", "tfla_runtime", "work_term_model",
            "cost_model_top2", "cost_model_top3", "all_chunks_profiled")
M3D_POLICIES = ("default_chunk_256", "best_static_global_in_sample", "best_static_global_by_total_latency",
                "best_static_global_held_out", "best_static_per_batch_in_sample", "best_static_per_batch_by_total_latency",
                "best_static_per_batch_held_out", "cost_model", "cost_model_top2", "cost_model_top3", "all_chunks_profiled",
                "exact_shape_table_held_out_mean_latency")


# =========================================================================== generic helpers
def make_rng(*parts) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([SEED, zlib.crc32("|".join(map(str, parts)).encode())]))


def _ci(x) -> list[float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return E3B._pct_ci(x, 0.95) if len(x) else [None, None]


def record(lat: np.ndarray, orc: np.ndarray, pick: np.ndarray, tied: np.ndarray | None = None, layers: int | None = None,
           sel: np.ndarray | None = None, chunks=None) -> dict:
    """Per-call, latency-weighted and absolute regret of a per-call chunk index ``pick``."""
    idx = np.arange(len(orc)) if sel is None else np.flatnonzero(sel)
    if len(idx) == 0:
        return {"n": 0}
    p = np.asarray(pick)[idx]
    ch, o = lat[idx, p], orc[idx]
    reg, exc = ch / o - 1.0, ch - o
    rec = {"n": int(len(idx)), "mean_regret_pct": float(100 * reg.mean()),
           "p95_regret_pct": float(100 * np.percentile(reg, 95)), "max_regret_pct": float(100 * reg.max())}
    for m in MARGINS:
        rec[f"within_{round(100 * m)}pct_share"] = float((reg <= m + 1e-12).mean())
    rec["oracle_hit_share"] = float((reg <= 1e-12).mean())
    if tied is not None:
        rec["ci_tie_share"] = float(tied[idx, p].mean())
    rec["lw_share_of_oracle_pct"] = float(100 * exc.sum() / o.sum())
    rec["lw_share_of_policy_pct"] = float(100 * exc.sum() / ch.sum())
    rec["us_per_call"] = float(1000 * exc.mean())
    rec["oracle_us_per_call"] = float(1000 * o.mean())
    if layers:
        rec["ms_per_forward_pass"] = float(exc.mean() * layers)
        rec["oracle_scan_ms_per_forward_pass"] = float(o.mean() * layers)
    if chunks is not None:
        c, n = np.unique(np.asarray(chunks)[p], return_counts=True)
        rec["chosen"] = {str(int(a)): int(b) for a, b in zip(c, n)}
    return rec


def boot(lat, orc, picks: dict, clusters, rng, n_boot: int, sel=None, pairs=()) -> tuple[dict, dict, int]:
    """Prompt-cluster bootstrap intervals of every pick and paired differences (``e3b_information.cluster_bootstrap``)."""
    idx = np.arange(len(orc)) if sel is None else np.flatnonzero(sel)
    _, cl = np.unique(np.asarray(clusters)[idx], return_inverse=True)
    cl = cl.ravel()
    o = orc[idx]
    vals, p95 = {"__orc": o}, {}
    for n, p in picks.items():
        ch = lat[idx, np.asarray(p)[idx]]
        r = ch / o - 1.0
        vals[f"{n}|reg"] = 100 * r
        vals[f"{n}|w2"] = (r <= 0.02 + 1e-12).astype(float)
        vals[f"{n}|exc"] = ch - o
        p95[f"{n}|p95"] = 100 * r
    reps = E3B.cluster_bootstrap(vals, cl, n_boot, rng, p95)
    out = {}
    for n in picks:
        out[n] = {"mean_regret_pct": _ci(reps[f"{n}|reg"]), "p95_regret_pct": _ci(reps[f"{n}|p95"]),
                  "within_2pct_share": _ci(reps[f"{n}|w2"]),
                  "lw_share_of_oracle_pct": _ci(100 * reps[f"{n}|exc"] / reps["__orc"]),
                  "us_per_call": _ci(1000 * reps[f"{n}|exc"])}
    paired = {}
    for a, b in pairs:
        d = reps[f"{a}|reg"] - reps[f"{b}|reg"]
        du = 1000 * (reps[f"{a}|exc"] - reps[f"{b}|exc"])
        ci = _ci(d)
        paired[f"{a} - {b}"] = {"diff_mean_regret_pp": float(vals[f"{a}|reg"].mean() - vals[f"{b}|reg"].mean()),
                                "ci95_pp": ci,
                                "diff_us_per_call": float(1000 * (vals[f"{a}|exc"].mean() - vals[f"{b}|exc"].mean())),
                                "ci95_us": _ci(du), "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}
    return out, paired, int(cl.max() + 1)


def evaluate_picks(lat, orc, tied, picks: dict, chunks, batch, clusters, layers, label: str, n_boot: int,
                   pairs=()) -> tuple[dict, dict, int]:
    """Records with intervals of every pick over all calls and per batch, and the paired differences."""
    recs = {n: record(lat, orc, p, tied, layers, chunks=chunks) for n, p in picks.items()}
    ci, paired, k = boot(lat, orc, picks, clusters, make_rng(label, "all"), n_boot, pairs=pairs)
    for n in picks:
        recs[n]["ci95"] = ci[n]
        recs[n]["by_batch"] = {}
    paired_b = {}
    for b in sorted(np.unique(batch)):
        sel = batch == b
        cib, pb, kb = boot(lat, orc, picks, clusters, make_rng(label, f"batch{int(b)}"), n_boot, sel=sel, pairs=pairs)
        for n, p in picks.items():
            r = record(lat, orc, p, tied, layers, sel=sel, chunks=chunks)
            r["ci95"] = cib[n]
            r["n_prompt_clusters"] = kb
            recs[n]["by_batch"][str(int(b))] = r
        paired_b[str(int(b))] = pb
    return recs, {"all_calls": paired, "by_batch": paired_b}, k


class Checks:
    """Reproduction checks: stored analyses (exact) and values printed in the manuscript (printed precision)."""

    def __init__(self):
        self.rows = []

    def stored(self, what: str, computed, reference, source: str, tol: float = 1e-6):
        ok = (computed is not None and reference is not None
              and abs(float(computed) - float(reference)) <= tol * max(1.0, abs(float(reference))))
        self.rows.append({"kind": "stored", "check": what, "computed": computed, "reference": reference,
                          "source": source, "match": bool(ok)})

    def printed(self, what: str, computed, printed, location: str):
        ok = E22._matches(computed, printed) if isinstance(printed, str) else computed == printed
        self.rows.append({"kind": "manuscript", "check": what, "computed": computed, "printed": printed,
                          "location": location, "match": bool(ok)})

    def summary(self) -> dict:
        out = {}
        for kind in ("stored", "manuscript"):
            rows = [r for r in self.rows if r["kind"] == kind]
            out[kind] = {"n": len(rows), "n_match": sum(r["match"] for r in rows),
                         "mismatches": [r for r in rows if not r["match"]]}
        return out


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


# =========================================================================== calibration-only selectors
def calibration_worker(job) -> dict:
    """Fit the calibration-only selectors of one calibration tag (cost_model_baselines_r4) and return their
    per-call choice on the 2360 captured calls (keyed by call), the cost-model scores and the regrets on the tag's
    own call table (for the reproduction of the stored summary)."""
    cal, in_dir, learned = job
    t0 = time.time()
    ctx = CMB4.Ctx(cal, Path(in_dir))
    sels, _, _ = CMB4.fit_selectors(ctx, learned)
    out = {"cal": cal, "chunks": [int(c) for c in ctx.chunks], "keys": [str(k) for k in ctx.df["key"]],
           "own_table": f"e2_mamba1_{cal}.csv", "choice": {}, "own_table_regret": {}}
    for name in CAL_SELECTORS:
        if name not in sels:
            continue
        S = sels[name].scores(ctx.calls)
        j = S.argmin(1)
        out["choice"][name] = [int(ctx.chunks[x]) for x in j]
        st = CMB4.score_calls(ctx.df, ctx.chunks, j)
        out["own_table_regret"][name] = {k: float(st[k]) for k in ("mean_regret_pct", "p95_regret_pct", "max_regret_pct")}
        if name == "cost_model":
            out["cost_model_scores"] = np.where(np.isfinite(S), S, 1e30).tolist()
    stored = dict(zip(ctx.stored["per_call_choice"]["mamba1"]["keys"], ctx.stored["per_call_choice"]["mamba1"]["chunk"]))
    out["cost_model_choice_equals_stored"] = bool(all(int(stored[k]) == c for k, c in zip(out["keys"], out["choice"]["cost_model"])))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def calibration_selectors(cals: list[str], args, log) -> dict:
    out, todo = {}, []
    for cal in cals:
        p = args.cache_dir / f"taco_calibration_choices_{cal}{'' if args.learned else '_nolearned'}.json" if args.cache_dir else None
        if p is not None and p.exists():
            out[cal] = json.loads(p.read_text())
            log(f"  calibration selectors {cal}: cached")
        else:
            todo.append(cal)
    jobs = [(c, str(args.in_dir), args.learned) for c in todo]
    if jobs:
        log(f"  fitting calibration-only selectors for {todo} ({min(args.jobs, len(jobs))} processes)")
        if args.jobs > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as ex:
                res = list(ex.map(calibration_worker, jobs))
        else:
            res = [calibration_worker(j) for j in jobs]
        for r in res:
            out[r["cal"]] = r
            log(f"  calibration selectors {r['cal']}: {r['seconds']} s")
            if args.cache_dir:
                args.cache_dir.mkdir(parents=True, exist_ok=True)
                p = args.cache_dir / f"taco_calibration_choices_{r['cal']}{'' if args.learned else '_nolearned'}.json"
                p.write_text(json.dumps(r))
    return out


# =========================================================================== per-call tables (E3b machinery)
def load_table(in_dir: Path, csv: str, rmed: bool, label: str, model: str, host: str) -> E3B.Table:
    df = load_e2([in_dir / csv])
    if "regimes" in df and (df["regimes"] == "e4_key").any():
        df = df[df["regimes"] != "e4_key"]
    if rmed:
        df = round_median_view(df)
    return E3B.Table(df.reset_index(drop=True), label, model, host, csv)


def base_picks(t: E3B.Table) -> dict:
    """Per-call chunk index of the static, rule, held-out and table predictors on one table."""
    ch = {}
    if 256 in t.chunks:
        ch["default_chunk_256"] = np.full(t.N, t.chunks.index(256))
    ch["best_static_global_in_sample"] = np.full(t.N, int(t.norm_fit.sum(0).argmin()))
    lat_fin = np.where(np.isfinite(t.lat), t.lat, 1e30)
    ch["best_static_global_by_total_latency"] = np.full(t.N, int(lat_fin.sum(0).argmin()))
    ch["best_static_global_held_out"] = E3B.cv_static(t)
    K = t.batch_code.max() + 1
    s1, s2 = np.zeros((K, t.C)), np.zeros((K, t.C))
    np.add.at(s1, t.batch_code, t.norm_fit)
    np.add.at(s2, t.batch_code, lat_fin)
    ch["best_static_per_batch_in_sample"] = s1.argmin(1)[t.batch_code]
    ch["best_static_per_batch_by_total_latency"] = s2.argmin(1)[t.batch_code]
    ch["best_static_per_batch_held_out"] = E3B.cv_table(t, [t.batch_code])[0]
    ch["entropy_rule"] = E3B.entropy_rule_choice(t)
    for name, (cols, depth) in E3.FEATURE_SETS.items():
        ch[name] = E3B.cv_tree(t, cols, depth)
    ch["exact_shape_table_held_out"] = E3B.cv_table(t, [t.key, t.bucket_key])[0]
    ch["exact_shape_table_held_out_mean_latency"] = DU.cv_exact_pick(t.df, lat_fin)
    s3 = np.zeros((t.key.max() + 1, t.C))
    np.add.at(s3, t.key, t.norm_fit)
    ch["exact_shape_floor_in_sample"] = s3.argmin(1)[t.key]
    return ch


R4_MAP = {"best_static_global_held_out": "global_static", "entropy_rule": "entropy_rule", "L_only": "L_only",
          "shape": "shape", "shape+entropy": "shape+entropy", "all": "all", "entropy_only": "entropy_only",
          "moments": "moments", "exact_shape_table_held_out": "exact_shape_table_held_out",
          "exact_shape_floor_in_sample": "exact_shape_floor_in_sample",
          "best_static_per_batch_in_sample": "best_static_per_batch_in_sample",
          "best_static_per_batch_held_out": "batch_only", "cost_model": "cost_model", "cost_model_top2": "cost_model_top2",
          "cost_model_top3": "cost_model_top3", "all_chunks_profiled": "all_chunks_profiled",
          "nearest_calibrated_key": "nearest_calibrated_key"}
E3B_MAP = {"best_static_global_held_out": "global_static", "entropy_rule": "entropy_rule", "L_only": "L_only",
           "shape": "shape", "shape+entropy": "shape+entropy", "all": "all", "entropy_only": "entropy_only",
           "moments": "moments", "exact_shape_table_held_out": "exact_key_table",
           "exact_shape_floor_in_sample": "exact_key_floor_in_sample",
           "best_static_per_batch_in_sample": "best_static_per_batch_in_sample",
           "best_static_per_batch_held_out": "batch_only", "cost_model": "cost_model"}


def mamba1_set(name: str, spec: dict, cal: dict, args, checks: Checks, cache: dict) -> tuple[dict, dict]:
    t = load_table(args.in_dir, spec["csv"], spec["rmed"], name, "mamba1", spec["cal"])
    picks = base_picks(t)
    pos = {k: i for i, k in enumerate(cal["keys"])}
    rows = np.array([pos[k] for k in t.df["key"]])
    for n, ch in cal["choice"].items():
        picks[n] = np.array([t.chunks.index(int(c)) for c in np.asarray(ch)[rows]])
    P = np.asarray(cal["cost_model_scores"], float)[rows]
    if not np.array_equal(P.argmin(1), picks["cost_model"]):
        raise RuntimeError(f"{name}: cost-model scores and choice disagree")
    for k in (2, 3):
        picks[f"cost_model_top{k}"] = CMB4.topk_picks(t.df, t.chunks, P, k)
    picks["all_chunks_profiled"] = CMB4.topk_picks(t.df, t.chunks, P, t.C)
    pairs = [("cost_model", n) for n in picks if n != "cost_model"]
    recs, paired, k = evaluate_picks(t.lat, t.orc, t.tied_mask, picks, t.chunks, t.batch, t.clusters, LAYERS_370M,
                                     name, args.n_boot, pairs)
    for n, r in recs.items():
        r["uses_captured_call_timings"], r["definition"] = SELECTOR_INFO.get(n, ("?", ""))
    # existing variant of cost_model_baselines whose per-call choice is not exposed: summary statistics only
    try:
        var = CMB.nearest_key_selectors(spec["cal"], args.in_dir, t.df, t.chunks)[1]["interpolated_log_length"]
        recs["interp_log_length_same_batch"] = {
            "n": int(var["n"]), "mean_regret_pct": var["mean_regret_pct"], "p95_regret_pct": var["p95_regret_pct"],
            "max_regret_pct": var["max_regret_pct"], "uses_captured_call_timings": "none",
            "definition": "normalized curves of the two bracketing calibrated lengths at the same batch, linear in log2 L",
            "note": "per-call choice not exposed by cost_model_baselines.nearest_key_selectors: no within-2% share, "
                    "latency-weighted share or interval"}
    except Exception as exc:  # pragma: no cover - reported, not fatal
        recs["interp_log_length_same_batch"] = {"error": repr(exc)}
    res = {"gpu": spec["gpu"], "convention": spec["convention"], "launch_inclusive": spec["launch_inclusive"],
           "primary": spec["primary"], "source": spec["csv"], "round_median_view": spec["rmed"],
           "calibration_tag": spec["cal"], "n_calls": int(t.N), "n_prompt_clusters": k, "chunks": t.chunks,
           "selectors": recs, "paired_cost_model_minus": paired}
    m1_checks(name, spec, recs, cal, args, checks, cache)
    return res, {"t": t, "picks": picks, "P": P}


def m1_checks(name, spec, recs, cal, args, checks: Checks, cache: dict) -> None:
    tag = spec["csv"][len("e2_mamba1_"):-len(".csv")]
    if spec["rmed"]:
        r4 = cache.setdefault("r4", _load_json(args.in_dir / "r4_rounds_analysis.json"))
        if r4 and tag in r4["tags"]:
            v = r4["tags"][tag]["views"]["round_median"]
            for mine, theirs in R4_MAP.items():
                if mine in recs and theirs in v["predictors"]:
                    for f in ("mean_regret_pct", "p95_regret_pct", "max_regret_pct", "within_2pct_share"):
                        checks.stored(f"{name}.{mine}.{f}", recs[mine][f], v["predictors"][theirs][f],
                                      f"r4_rounds_analysis.json tags.{tag}.views.round_median.predictors.{theirs}")
            for b, x in v["table_II"].items():
                r = recs["best_static_per_batch_in_sample"]["by_batch"][str(b)]
                checks.stored(f"{name}.table_II.b{b}.best_static_chunk", int(next(iter(r["chosen"]))), x["best_static_chunk"],
                              "r4_rounds_analysis.json table_II")
                checks.stored(f"{name}.table_II.b{b}.static_mean", r["mean_regret_pct"], x["static"]["mean_regret_pct"],
                              "r4_rounds_analysis.json table_II")
                checks.stored(f"{name}.table_II.b{b}.static_p95", r["p95_regret_pct"], x["static"]["p95_regret_pct"],
                              "r4_rounds_analysis.json table_II")
        return
    e3b = _load_json(args.in_dir / f"e3b_information_mamba1_{tag}.json")
    if e3b:
        for mine, theirs in E3B_MAP.items():
            if mine in recs and theirs in e3b["predictors"]:
                for f in ("mean_regret_pct", "p95_regret_pct", "within_2pct_share"):
                    checks.stored(f"{name}.{mine}.{f}", recs[mine][f], e3b["predictors"][theirs][f],
                                  f"e3b_information_mamba1_{tag}.json predictors.{theirs}")
    summ = cache.setdefault("cmb4", _load_json(args.in_dir / "cost_model_baselines_r4_summary.json"))
    if summ and tag in summ["captured_calls"] and tag == spec["cal"]:
        cc = summ["captured_calls"][tag]
        for n in CAL_SELECTORS:
            if n in recs and n in cc:
                for f in ("mean_regret_pct", "p95_regret_pct", "max_regret_pct"):
                    checks.stored(f"{name}.{n}.{f}", recs[n][f], cc[n][f],
                                  f"cost_model_baselines_r4_summary.json captured_calls.{tag}.{n}")
        tk = summ["topk_cost_model"][tag]
        for mine, theirs in (("cost_model_top2", "top2"), ("cost_model_top3", "top3"), ("all_chunks_profiled", "all_chunks_profiled")):
            for f in ("mean_regret_pct", "p95_regret_pct", "max_regret_pct"):
                checks.stored(f"{name}.{mine}.{f}", recs[mine][f], tk[theirs][f],
                              f"cost_model_baselines_r4_summary.json topk_cost_model.{tag}.{theirs}")


def calibration_own_table_checks(cal_recs: dict, args, checks: Checks, cache: dict) -> None:
    summ = cache.setdefault("cmb4", _load_json(args.in_dir / "cost_model_baselines_r4_summary.json"))
    for cal, r in cal_recs.items():
        checks.stored(f"calibration.{cal}.cost_model_choice_equals_stored", int(r["cost_model_choice_equals_stored"]), 1,
                      f"cost_model_{cal}.json per_call_choice")
        if not summ or cal not in summ["captured_calls"]:
            continue
        for n, v in r["own_table_regret"].items():
            if n in summ["captured_calls"][cal]:
                checks.stored(f"calibration.{cal}.{n}.mean_regret_pct (own table {r['own_table']})", v["mean_regret_pct"],
                              summ["captured_calls"][cal][n]["mean_regret_pct"],
                              f"cost_model_baselines_r4_summary.json captured_calls.{cal}.{n}")


def mamba2_set(name: str, spec: dict, args, checks: Checks, cache: dict) -> dict:
    t = load_table(args.in_dir, spec["csv"], spec["rmed"], name, "mamba2", spec["gpu_tag"])
    picks = base_picks(t)
    pm = json.loads((args.in_dir / f"cost_model_{spec['gpu_tag']}.json").read_text())["V4_pipelines"]["mamba2"]
    th = [pm["params"][k] for k in CM.PIPE_PARAMS]
    Lf, Bf = t.df["L"].to_numpy(float), t.df["batch"].to_numpy(float)
    P = np.stack([CM.pipe_predict(th, Lf, Bf, c) for c in t.chunks], 1)
    P = np.where(np.isfinite(t.lat), P, np.inf)
    picks["work_term_model"] = CM.pipe_select(th, P, t.chunks)
    pairs = [("work_term_model", n) for n in picks if n != "work_term_model"]
    recs, paired, k = evaluate_picks(t.lat, t.orc, t.tied_mask, picks, t.chunks, t.batch, t.clusters, LAYERS_370M,
                                     name, args.n_boot, pairs)
    for n, r in recs.items():
        r["uses_captured_call_timings"], r["definition"] = SELECTOR_INFO.get(
            n, ("none", "work-term model of the SSD pipeline with the stored eager-calibrated parameters of the GPU "
                        "(cost_model.pipe_select: launch-bound ties to the work-bound optimum)"))
    # reproduction
    if name == "mamba2_pro6000_launch":
        tv = _load_json(args.in_dir / "tables_iv_v_ci.json")
        if tv:
            mp = {"best_static_global_held_out": "global_static", "entropy_rule": "entropy_rule",
                  "best_static_per_batch_held_out": "batch_only", "L_only": "L_only", "shape": "shape",
                  "shape+entropy": "shape+entropy", "all": "all", "entropy_only": "entropy_only", "moments": "moments",
                  "exact_shape_table_held_out": "exact_key_table", "work_term_model": "cost_model"}
            for mine, theirs in mp.items():
                checks.stored(f"{name}.{mine}.mean", recs[mine]["mean_regret_pct"], tv["mamba2"]["table_V"][theirs]["mean_pct"],
                              f"tables_iv_v_ci.json mamba2.table_V.{theirs}")
                checks.stored(f"{name}.{mine}.p95", recs[mine]["p95_regret_pct"], tv["mamba2"]["table_V"][theirs]["p95_pct"],
                              f"tables_iv_v_ci.json mamba2.table_V.{theirs}")
            for b, x in tv["mamba2"]["table_IV"].items():
                r = recs["best_static_per_batch_in_sample"]["by_batch"][b]
                checks.stored(f"{name}.table_IV.b{b}.chunk", int(next(iter(r["chosen"]))), x["best_static_chunk"],
                              "tables_iv_v_ci.json mamba2.table_IV")
                checks.stored(f"{name}.table_IV.b{b}.mean", r["mean_regret_pct"], x["static"]["mean_pct"],
                              "tables_iv_v_ci.json mamba2.table_IV")
    elif name == "mamba2_vgpu3090_graph":
        g = cache.setdefault("e22", _load_json(args.in_dir / "e22_graph_analysis_vgpu3090_rounds3.json"))
        if g:
            pol = g["tables"]["ssd370m_graph"]["policies"]
            for mine, theirs in (("default_chunk_256", "default"), ("best_static_global_in_sample", "best_static_insample"),
                                 ("best_static_global_held_out", "e3_global_static"), ("shape", "e3_shape"),
                                 ("shape+entropy", "e3_shape+entropy"), ("entropy_only", "e3_entropy_only"),
                                 ("work_term_model", "work_term_launch_default"),
                                 ("exact_shape_table_held_out_mean_latency", "exact_table_heldout")):
                checks.stored(f"{name}.{mine}.mean", recs[mine]["mean_regret_pct"], pol[theirs]["all"]["mean_pct"],
                              f"e22_graph_analysis_vgpu3090_rounds3.json tables.ssd370m_graph.policies.{theirs}")
    else:
        tag = spec["csv"][len("e2_mamba2_"):-len(".csv")]
        e3b = _load_json(args.in_dir / f"e3b_information_mamba2_{tag}.json")
        if e3b:
            for mine, theirs in E3B_MAP.items():
                if mine in recs and theirs in e3b["predictors"]:
                    checks.stored(f"{name}.{mine}.mean", recs[mine]["mean_regret_pct"], e3b["predictors"][theirs]["mean_regret_pct"],
                                  f"e3b_information_mamba2_{tag}.json predictors.{theirs}")
    return {"gpu": spec["gpu"], "convention": spec["convention"], "launch_inclusive": spec["launch_inclusive"],
            "source": spec["csv"], "round_median_view": spec["rmed"], "n_calls": int(t.N), "n_prompt_clusters": k,
            "chunks": t.chunks, "selectors": recs, "paired_work_term_model_minus": paired,
            "work_term_params": dict(zip(CM.PIPE_PARAMS, map(float, th)))}


# =========================================================================== item 1a: CUDA-graph tables
def graph_set(nm: str, args, checks: Checks, cache: dict) -> dict:
    t = E22.Table(nm, args.in_dir)
    t.set_floor()
    tied = np.array([[c in s for c in t.chunks] for s in t.tied], dtype=bool)
    picks = {"default": np.full(len(t.df), t.default_idx()),
             "best_static_in_sample": np.full(len(t.df), t.best_static_idx(np.ones(len(t.df), bool)))}
    if t.default is None and 256 in t.chunks:
        picks["fixed_chunk_256"] = np.full(len(t.df), t.chunks.index(256))
    picks["shape_tree_held_out"] = E22.e3_predictions(t.df, t.chunks)["shape"]
    picks["exact_table_held_out"] = DU.cv_exact_pick(t.df, t.lat)
    if t.family in ("mamba2", "gdn"):
        wp, _ = E22.work_term_policies(t, args.in_dir)
        picks["work_term_model_eager_calibration"] = wp["work_term_launch_default"]
        if "work_term_refit_cv" in wp:
            picks["work_term_model_refit_on_graph_timings_cv"] = wp["work_term_refit_cv"]
    clusters = E3B.prompt_clusters(t.df)[0]
    out = {"table": nm, "source": t.csv, "kernel_family": t.family, "checkpoint": t.checkpoint, "timing": t.timing,
           "gpu": "RTX 3090", "scan_layers": t.layers,
           "default_chunk": int(t.chunks[t.default_idx()]),
           "default_note": "library default" if t.default is not None else "no library default: best static chunk in sample",
           "floor_us": 1000 * t.floor_ms, "floor_definition": "2 x median oracle latency of the batch-1 calls with L <= 256",
           "n_calls": int(len(t.df)), "n_above_floor": int(t.above.sum()), "share_above_floor": float(t.above.mean()),
           "subsets": {}}
    pairs = [("default", n) for n in picks if n != "default"]
    for sub, m in t.masks().items():
        if not m.any():
            continue
        recs = {n: record(t.lat, t.orc, p, tied, t.layers, sel=m, chunks=t.chunks) for n, p in picks.items()}
        ci, paired, k = boot(t.lat, t.orc, picks, clusters, make_rng(nm, sub), args.n_boot, sel=m, pairs=pairs)
        for n in picks:
            recs[n]["ci95"] = ci[n]
        if sub == "all":
            for n, p in picks.items():
                recs[n]["by_batch"] = {str(int(b)): record(t.lat, t.orc, p, tied, t.layers, sel=t.batch == b, chunks=t.chunks)
                                       for b in sorted(np.unique(t.batch))}
        out["subsets"][sub] = {"n": int(m.sum()), "n_prompt_clusters": k, "policies": recs,
                               "paired_default_minus": paired}
    g = cache.setdefault("e22", _load_json(args.in_dir / "e22_graph_analysis_vgpu3090_rounds3.json"))
    if g and nm in g["tables"]:
        pol = g["tables"][nm]["policies"]
        for mine, theirs in (("default", "default"), ("best_static_in_sample", "best_static_insample"),
                             ("shape_tree_held_out", "e3_shape"), ("exact_table_held_out", "exact_table_heldout"),
                             ("work_term_model_eager_calibration", "work_term_launch_default"),
                             ("work_term_model_refit_on_graph_timings_cv", "work_term_refit_cv")):
            if mine not in picks or theirs not in pol:
                continue
            for sub in ("all", "above_floor", "below_floor"):
                if sub in out["subsets"] and pol[theirs].get(sub, {}).get("n"):
                    r = out["subsets"][sub]["policies"][mine]
                    src = f"e22_graph_analysis_vgpu3090_rounds3.json tables.{nm}.policies.{theirs}.{sub}"
                    checks.stored(f"graph.{nm}.{sub}.{mine}.mean", r["mean_regret_pct"], pol[theirs][sub]["mean_pct"], src)
                    checks.stored(f"graph.{nm}.{sub}.{mine}.lw_policy", r["lw_share_of_policy_pct"],
                                  pol[theirs][sub]["latency_weighted_share_pct"], src)
                    checks.stored(f"graph.{nm}.{sub}.{mine}.ms_per_pass", r["ms_per_forward_pass"],
                                  pol[theirs][sub]["ms_per_forward_pass"], src)
        checks.stored(f"graph.{nm}.n_above_floor", out["n_above_floor"], g["tables"][nm]["floor"]["n_above"],
                      f"e22_graph_analysis_vgpu3090_rounds3.json tables.{nm}.floor")
    return out


def floor_coverage(args, checks: Checks, cache: dict) -> dict:
    """N3: share of the SSD 2.7B and Gated DeltaNet captured calls above the launch floor (device time, eager)."""
    out = {}
    g = cache.setdefault("e22", _load_json(args.in_dir / "e22_graph_analysis_vgpu3090_rounds3.json"))
    for nm in ("ssd2p7b_device", "gdn1p4b_device", "ssd2p7b_graph", "gdn1p4b_graph", "ssd370m_graph"):
        t = E22.Table(nm, args.in_dir)
        t.set_floor()
        out[nm] = {"source": t.csv, "timing": t.timing, "n_calls": int(len(t.df)), "floor_us": 1000 * t.floor_ms,
                   "n_above_floor": int(t.above.sum()), "share_above_floor": float(t.above.mean()),
                   "by_batch": {str(int(b)): {"n": int((t.batch == b).sum()), "n_above": int((t.above & (t.batch == b)).sum()),
                                              "share_above": float(t.above[t.batch == b].mean())}
                                for b in sorted(np.unique(t.batch))}}
        if g and nm in g["tables"]:
            checks.stored(f"N3.{nm}.n_above_floor", out[nm]["n_above_floor"], g["tables"][nm]["floor"]["n_above"],
                          f"e22_graph_analysis_vgpu3090_rounds3.json tables.{nm}.floor")
    return out


# =========================================================================== item 3 (M4d)
M4D_SELECTORS = ("cost_model", "work_only", "nearest_calibrated_key", "loglinear_interp_logL_logB", "mlp_basic",
                 "gbr_structural", "shape", "exact_shape_table_held_out", "best_static_per_batch_in_sample")


def m4d_set(name: str, spec: dict, inner: dict, args) -> dict:
    t, picks = inner["t"], inner["picks"]
    keys = CMB.calibration_keys(spec["cal"], args.in_dir, t.chunks)
    kset = {(int(k["L"]), int(k["B"]), int(k["d"])) for k in keys}
    shp = list(zip(t.df["L"].astype(int), t.df["batch"].astype(int), t.df["width"].astype(int)))
    on = np.array([s in kset for s in shp])
    sid = t.df.groupby(["L", "batch", "width"]).ngroup().to_numpy()
    S = int(sid.max() + 1)
    shape_on = np.zeros(S, bool)
    shape_on[sid[on]] = True
    out = {"calibration_tag": spec["cal"], "calibration_keys": sorted([list(k) for k in kset]),
           "n_calls": int(t.N), "n_distinct_shapes_L_batch_width": S,
           "n_distinct_shapes_L_batch": int(t.df.groupby(["L", "batch"]).ngroups),
           "n_shapes_on_calibration_key": int(shape_on.sum()), "n_calls_on_calibration_key": int(on.sum()),
           "share_calls_on_calibration_key": float(on.mean()), "selectors": {}}
    sel_names = [n for n in M4D_SELECTORS if n in picks]
    ar = np.arange(t.N)
    # cluster bootstrap of the on-key and off-key means (ratio of replicate means with common weights)
    vals = {"on_n": on.astype(float), "off_n": (~on).astype(float)}
    regs = {}
    for n in sel_names:
        r = 100 * (t.norm[ar, picks[n]] - 1)
        regs[n] = r
        vals[f"{n}|on"], vals[f"{n}|off"] = r * on, r * ~on
    reps = E3B.cluster_bootstrap(vals, t.clusters, args.n_boot, make_rng(name, "m4d_onkey"), {})
    rng_s = make_rng(name, "m4d_shapes")
    idx_all = rng_s.integers(0, S, size=(args.n_boot, S))
    on_ids, off_ids = np.flatnonzero(shape_on), np.flatnonzero(~shape_on)
    idx_on = on_ids[rng_s.integers(0, len(on_ids), size=(args.n_boot, len(on_ids)))] if len(on_ids) else None
    idx_off = off_ids[rng_s.integers(0, len(off_ids), size=(args.n_boot, len(off_ids)))] if len(off_ids) else None
    with np.errstate(invalid="ignore", divide="ignore"):
        for n in sel_names:
            r = regs[n]
            m_on = reps[f"{n}|on"] / reps["on_n"]
            m_off = reps[f"{n}|off"] / reps["off_n"]
            shape_mean = np.bincount(sid, weights=r, minlength=S) / np.bincount(sid, minlength=S)
            rec = {"on_key": record(t.lat, t.orc, picks[n], t.tied_mask, LAYERS_370M, sel=on),
                   "off_key": record(t.lat, t.orc, picks[n], t.tied_mask, LAYERS_370M, sel=~on),
                   "on_key_mean_ci95": _ci(m_on), "off_key_mean_ci95": _ci(m_off),
                   "off_minus_on_mean_pp": float(r[~on].mean() - r[on].mean()) if on.any() else None,
                   "off_minus_on_ci95_pp": _ci(m_off - m_on),
                   "per_shape_unweighted_mean_pct": float(shape_mean.mean()),
                   "per_shape_unweighted_mean_ci95_shape_bootstrap": _ci(shape_mean[idx_all].mean(1)),
                   "per_shape_unweighted_mean_on_key_pct": float(shape_mean[shape_on].mean()) if shape_on.any() else None,
                   "per_shape_unweighted_mean_on_key_ci95": _ci(shape_mean[idx_on].mean(1)) if idx_on is not None else None,
                   "per_shape_unweighted_mean_off_key_pct": float(shape_mean[~shape_on].mean()),
                   "per_shape_unweighted_mean_off_key_ci95": _ci(shape_mean[idx_off].mean(1)) if idx_off is not None else None,
                   "per_shape_max_of_shape_means_pct": float(shape_mean.max()),
                   "per_shape_median_of_shape_means_pct": float(np.median(shape_mean))}
            out["selectors"][n] = rec
    return out


# =========================================================================== item 5 (M3d)
def large_set(name: str, csv: str, du_name: str, layers: int, args, checks: Checks, cache: dict) -> dict:
    t = load_table(args.in_dir, csv, False, name, "mamba1", "vgpu3090")
    allp = base_picks(t)
    picks = {n: allp[n] for n in ("default_chunk_256", "best_static_global_in_sample", "best_static_global_by_total_latency",
                                  "best_static_global_held_out", "best_static_per_batch_in_sample",
                                  "best_static_per_batch_by_total_latency", "best_static_per_batch_held_out",
                                  "exact_shape_table_held_out", "exact_shape_table_held_out_mean_latency",
                                  "exact_shape_floor_in_sample")}
    spec = next(s for s in DU.kernel_specs() if s["name"] == du_name)
    lat_fin = np.where(np.isfinite(t.lat), t.lat, np.inf)
    picks["cost_model"], minfo = DU.model_choice(args.in_dir, spec, t.df, t.chunks, lat_fin)
    pairs = [("cost_model", n) for n in picks if n != "cost_model"]
    recs, paired, k = evaluate_picks(t.lat, t.orc, t.tied_mask, picks, t.chunks, t.batch, t.clusters, layers, name,
                                     args.n_boot, pairs)
    for n, r in recs.items():
        r["uses_captured_call_timings"], r["definition"] = SELECTOR_INFO.get(
            n, ("none", "Eq. (5) with the device-time RTX 3090 370M parameters, not refitted to this width"))
    du = cache.setdefault("du", _load_json(args.in_dir / "deployment_units.json"))
    if du:
        p = du["kernels"][du_name]["pooled"]["all"]
        for mine, theirs in (("default_chunk_256", "default"), ("best_static_global_in_sample", "best_single_chunk"),
                             ("cost_model", "cost_model"), ("exact_shape_table_held_out_mean_latency", "exact_table_heldout")):
            checks.stored(f"{name}.{mine}.ms_per_forward_pass", recs[mine]["ms_per_forward_pass"],
                          p[theirs]["ms_per_forward_pass"], f"deployment_units.json kernels.{du_name}.pooled.all.{theirs}")
    return {"checkpoint": LARGE_SETS[name][3], "gpu": "RTX 3090", "convention": "device time, one block per chunk",
            "source": csv, "scan_layers": layers, "n_calls": int(t.N), "n_prompt_clusters": k, "chunks": t.chunks,
            "cost_model_source": minfo, "selectors": recs, "paired_cost_model_minus": paired}


def _compact(r: dict, extra=()) -> dict:
    keys = ("mean_regret_pct", "p95_regret_pct", "max_regret_pct", "within_2pct_share", "lw_share_of_oracle_pct",
            "lw_share_of_policy_pct", "us_per_call", "ms_per_forward_pass", "oracle_scan_ms_per_forward_pass", *extra)
    out = {k: r[k] for k in keys if k in r}
    if "ci95" in r:
        out["mean_regret_pct_ci95"] = r["ci95"]["mean_regret_pct"]
        out["lw_share_of_oracle_pct_ci95"] = r["ci95"]["lw_share_of_oracle_pct"]
    if "chosen" in r:
        out["chosen"] = r["chosen"]
    return out


def m3d(m1: dict, large: dict) -> dict:
    out = {"definition": ("untuned baseline = best static chunk per GPU (one chunk for all calls) or per (GPU, batch); "
                          "in sample = chosen on the scored calls (lowest mean per-call regret, as Table II), "
                          "by_total_latency = lowest total latency (the choice that minimizes ms per forward pass), "
                          "held_out = chosen on the training prompt folds; next to the fixed chunk 256"),
           "per_data_set": {}, "deployment_table_rows": {}, "calibration_budget_figure": {}}
    sets = {**{n: r for n, r in m1.items()}, **large}
    for n, r in sets.items():
        sel = r["selectors"]
        rec = {"gpu": r.get("gpu"), "convention": r.get("convention"), "source": r["source"]}
        for p in M3D_POLICIES:
            if p not in sel:
                continue
            c = _compact(sel[p])
            if p.startswith("best_static_per_batch") or p == "default_chunk_256":
                c["by_batch"] = {b: _compact(v) for b, v in sel[p]["by_batch"].items()}
            rec[p] = c
        out["per_data_set"][n] = rec
    rows = {"Triton Mamba-1 370M, PRO 6000 (launch-inclusive)": "pro6000_launch",
            "Triton Mamba-1 370M, RTX 3090 (launch-inclusive)": "vgpu3090_launch",
            "Triton Mamba-1 370M, H800 (launch-inclusive)": "h800_launch",
            "Triton Mamba-1 1.4B, RTX 3090, device time": "mamba1_1p4b_device",
            "Triton Mamba-1 2.8B, RTX 3090, device time": "mamba1_2p8b_device",
            "Triton Mamba-1 370M, PRO 6000, device time (round medians)": "pro6000_device",
            "Triton Mamba-1 370M, RTX 3090, device time (round medians)": "vgpu3090_device",
            "Triton Mamba-1 370M, H800, device time": "h800_device"}
    for label, n in rows.items():
        sel = sets[n]["selectors"]
        row = {"data_set": n, "scan_layers": sets[n].get("scan_layers", LAYERS_370M)}
        for p in ("default_chunk_256", "best_static_global_in_sample", "best_static_global_by_total_latency",
                  "best_static_per_batch_in_sample", "best_static_per_batch_by_total_latency", "best_static_global_held_out",
                  "best_static_per_batch_held_out", "cost_model", "exact_shape_table_held_out_mean_latency"):
            if p in sel:
                row[p] = {"ms_per_forward_pass": sel[p]["ms_per_forward_pass"],
                          "ms_per_forward_pass_ci95": [None if x is None else x * row["scan_layers"] / 1000
                                                       for x in sel[p]["ci95"]["us_per_call"]],
                          "lw_share_of_oracle_pct": sel[p]["lw_share_of_oracle_pct"],
                          "mean_regret_pct": sel[p]["mean_regret_pct"], "chosen": sel[p]["chosen"]}
        row["oracle_scan_ms_per_forward_pass"] = sel["default_chunk_256"]["oracle_scan_ms_per_forward_pass"]
        out["deployment_table_rows"][label] = row
    for n in ("pro6000_launch", "vgpu3090_launch", "h800_launch", "pro6000_device", "vgpu3090_device", "h800_device"):
        sel = m1[n]["selectors"]
        out["calibration_budget_figure"][n] = {
            "convention": m1[n]["convention"],
            **{p: {"mean_regret_pct": sel[p]["mean_regret_pct"], "p95_regret_pct": sel[p]["p95_regret_pct"],
                   "max_regret_pct": sel[p]["max_regret_pct"], "lw_share_of_oracle_pct": sel[p]["lw_share_of_oracle_pct"],
                   "mean_regret_pct_ci95": sel[p]["ci95"]["mean_regret_pct"]}
               for p in ("default_chunk_256", "best_static_global_in_sample", "best_static_per_batch_in_sample",
                         "best_static_global_held_out", "best_static_per_batch_held_out", "cost_model", "cost_model_top2",
                         "cost_model_top3", "all_chunks_profiled")}}
    return out


# =========================================================================== item 6 (M7a)
def _tp_arrays(tab: pd.DataFrame, chunks: list[int], choice: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat = np.stack(tab["lat"].to_numpy())
    keys = list(zip(tab["d"], tab["L"], tab["batch"]))
    pick = np.array([chunks.index(choice[k]) if k in choice else -1 for k in keys])
    ok = pick >= 0
    orc = lat.min(1)
    ch = np.where(ok, lat[np.arange(len(tab)), np.maximum(pick, 0)], np.nan)
    return ch / orc - 1.0, ch - orc, orc, ok


def tp_family(host: str, fam: str, raw: pd.DataFrame, meta: dict, stored: dict, args, checks: Checks, unavailable: list) -> dict:
    chunks = list(MAMBA1_CHUNKS) if fam == "triton" else list(REF_CHUNKS)
    tab = E12.shape_table(raw[raw["family"] == fam], chunks).reset_index(drop=True)
    widths = set(tab["d"])
    sel = {}
    try:
        sel["cost_model"] = (E12.triton_model_choice if fam == "triton" else E12.ref_model_choice)(tab, chunks, host, args.in_dir)
    except (FileNotFoundError, KeyError) as exc:
        unavailable.append({"item": "M7a", "quantity": f"cost model on {fam} shard widths, {host}",
                            "reason": f"no stored cost-model parameters for this kernel and GPU ({exc!r})"})
    if fam == "triton":
        nk = E12.nearest_key_choice(tab, chunks, host, args.in_dir)
        if nk:
            sel["nearest_key"] = nk
    sel["calibration_width_table"] = E12.table_choice(tab, lambda d: 2048)
    regs = {c: np.mean([r.lat[chunks.index(c)] / r.lat.min() - 1 for r in tab.itertuples()]) for c in chunks}
    best_static = min(regs, key=regs.get)
    sel["best_static"] = {(r.d, r.L, r.batch): best_static for r in tab.itertuples()}
    unsh = {}
    for model, full in MODEL_WIDTHS.items():          # pooled unsharded table exactly as e12_analysis (later models win)
        for tp in E12.TP_DEGREES[1:]:
            d = full // tp
            if full % tp == 0 and d in widths and full in widths:
                unsh.update(E12.table_choice(tab[tab["d"].isin([d, full])], lambda _d, full=full, d=d: full if _d == d else None))
    sel["unsharded_table"] = unsh
    shards = {d for d, lst in E12.shard_map(widths).items() if any(tp > 1 for _, tp in lst)}
    pooled = tab[tab["d"].isin(shards)].reset_index(drop=True)
    # per-round latencies: retuning at the shard width (cross-round held out) and its profiling cost
    piv = raw[raw["family"] == fam].pivot_table(index=["d", "L", "batch", "round"], columns="chunk", values="mean_ms")[chunks]
    rounds = {k: g.to_numpy() for k, g in piv.groupby(level=[0, 1, 2])}
    warm, rep = int(meta["args"]["warmup"]), int(meta["args"]["repeats"])

    def cross_round(R: np.ndarray) -> float:
        regs_r = []
        for r in range(R.shape[0]):
            j = int(np.delete(R, r, 0).mean(0).argmin())
            regs_r.append(R[r, j] / R[r].min() - 1)
        return float(np.mean(regs_r))

    keys_p = list(zip(pooled["d"], pooled["L"], pooled["batch"]))
    retune = np.array([cross_round(rounds[k]) for k in keys_p])
    prof_s = np.array([(warm + rep) * rounds[k].mean(0).sum() / 1000.0 for k in keys_p])
    # pooled statistics, latency-weighted shares and shape-bootstrap intervals
    n = len(pooled)
    rng = make_rng("tp", host, fam)
    bidx = rng.integers(0, n, size=(args.n_boot, n))
    arrs = {name: _tp_arrays(pooled, chunks, ch) for name, ch in sel.items()}
    res = {"n_shapes": int(len(tab)), "n_shard_shapes_pooled": int(n), "shard_widths": sorted(int(w) for w in shards),
           "best_static_chunk": int(best_static), "pooled_shards": {}, "paired_shape_bootstrap": {}, "per_shard": {},
           "per_shard_maxima": {}}
    for name, (reg, exc, orc, ok) in arrs.items():
        st = E12.regret(pooled, chunks, sel[name])
        st["lw_share_of_oracle_pct"] = float(100 * np.nansum(exc[ok]) / orc[ok].sum())
        st["us_per_call"] = float(1000 * np.nanmean(exc[ok]))
        if ok.all():
            st["mean_regret_pct_ci95_shape_bootstrap"] = _ci(100 * reg[bidx].mean(1))
            st["lw_share_of_oracle_pct_ci95_shape_bootstrap"] = _ci(100 * exc[bidx].sum(1) / orc[bidx].sum(1))
        res["pooled_shards"][name] = st
    res["pooled_shards"]["retuned_at_shard_width_cross_round"] = {
        "n": int(n), "mean_regret_pct": float(100 * retune.mean()), "p95_regret_pct": float(100 * np.percentile(retune, 95)),
        "max_regret_pct": float(100 * retune.max()), "mean_regret_pct_ci95_shape_bootstrap": _ci(100 * retune[bidx].mean(1)),
        "definition": "table retuned at the shard width: per shape the chunk with the lowest mean over two rounds, scored "
                      "against the held-out round's own oracle, averaged over the three held-out rounds (an upper bound: "
                      "it includes the noise of the held-out oracle; against the three-round mean it is 0 by definition)"}
    if "cost_model" in arrs:
        a = arrs["cost_model"]
        for other, b in arrs.items():
            if other == "cost_model" or not (a[3].all() and b[3].all()):
                continue
            d = 100 * (a[0] - b[0])
            res["paired_shape_bootstrap"][f"cost_model - {other}"] = {"diff_mean_regret_pp": float(d.mean()),
                                                                      "ci95_pp": _ci(d[bidx].mean(1))}
        d = 100 * (a[0] - retune)
        res["paired_shape_bootstrap"]["cost_model - retuned_at_shard_width_cross_round"] = {
            "diff_mean_regret_pp": float(d.mean()), "ci95_pp": _ci(d[bidx].mean(1))}
    res["retune_cost_per_key"] = {
        "definition": f"GPU kernel time of one profiling pass of a key at the E12 protocol: ({warm} warmup + {rep} "
                      "repeats) x latency, summed over all chunks; a lower bound (compilation, host overhead and "
                      "synchronization excluded)",
        "pooled_shards_s": {"median": float(np.median(prof_s)), "min": float(prof_s.min()), "max": float(prof_s.max()),
                            "sum_all_pooled_keys": float(prof_s.sum())}}
    # per (checkpoint, t): means per selector (unsharded per checkpoint as e12_analysis), maxima over shards
    per = {}
    for model, full in MODEL_WIDTHS.items():
        for tp in E12.TP_DEGREES[1:]:
            d = full // tp
            if full % tp or d not in widths:
                continue
            sub = tab[tab["d"] == d].reset_index(drop=True)
            row = {"d": int(d)}
            choices = dict(sel)
            if full in widths:
                choices["unsharded_table"] = E12.table_choice(tab[tab["d"].isin([d, full])],
                                                              lambda _d, full=full, d=d: full if _d == d else None)
            for name, ch in choices.items():
                st = E12.regret(sub, chunks, ch)
                reg, exc, orc, ok = _tp_arrays(sub, chunks, ch)
                if st.get("n"):
                    st["lw_share_of_oracle_pct"] = float(100 * np.nansum(exc[ok]) / orc[ok].sum())
                row[name] = st
            ks = list(zip(sub["d"], sub["L"], sub["batch"]))
            rr = np.array([cross_round(rounds[k]) for k in ks])
            row["retuned_at_shard_width_cross_round"] = {"n": int(len(rr)), "mean_regret_pct": float(100 * rr.mean()),
                                                         "max_regret_pct": float(100 * rr.max())}
            ps = np.array([(warm + rep) * rounds[k].mean(0).sum() / 1000.0 for k in ks])
            row["retune_cost_s"] = {"median_per_key": float(np.median(ps)), "sum_over_keys": float(ps.sum()), "n_keys": len(ps)}
            per[f"{model}_t{tp}"] = row
            if fam in stored and f"t{tp}" in stored[fam]["by_model"].get(model, {}):
                sm = stored[fam]["by_model"][model][f"t{tp}"]
                for name, theirs in (("cost_model", "cost_model"), ("unsharded_table", "unsharded_table"),
                                     ("calibration_width_table", "width2048_table")):
                    if name in row and theirs in sm and row[name].get("n"):
                        checks.stored(f"M7a.{host}.{fam}.{model}_t{tp}.{name}.mean", row[name]["mean_regret_pct"],
                                      sm[theirs]["mean_regret_pct"], f"e12_analysis_{host}.json families.{fam}.by_model")
    res["per_shard"] = per
    for name in list(sel) + ["retuned_at_shard_width_cross_round"]:
        vals = {k: v[name]["mean_regret_pct"] for k, v in per.items() if name in v and v[name].get("n")}
        if vals:
            kmax, kmin = max(vals, key=vals.get), min(vals, key=vals.get)
            res["per_shard_maxima"][name] = {"max_shard_mean_regret_pct": vals[kmax], "at": kmax,
                                             "min_shard_mean_regret_pct": vals[kmin], "min_at": kmin,
                                             "max_single_shape_regret_pct": float(max(v[name]["max_regret_pct"]
                                                                                       for v in per.values() if name in v and v[name].get("n")))}
    if fam in stored:
        sp = stored[fam]["pooled_shards"]
        for name, theirs in (("cost_model", "cost_model"), ("nearest_key", "nearest_key"),
                             ("calibration_width_table", "width2048_table"), ("best_static", "best_static"),
                             ("unsharded_table", "unsharded_table")):
            if name in res["pooled_shards"] and theirs in sp:
                for f in ("mean_regret_pct", "p95_regret_pct", "max_regret_pct"):
                    checks.stored(f"M7a.{host}.{fam}.pooled.{name}.{f}", res["pooled_shards"][name][f], sp[theirs][f],
                                  f"e12_analysis_{host}.json families.{fam}.pooled_shards.{theirs}")
    return res


def m7a(args, checks: Checks, unavailable: list, cache: dict) -> dict:
    out = {"definition": ("E12: per-device shard shapes timed on one GPU with synthetic inputs, three rounds; regret "
                          "against the round-mean oracle of each shape (e12_analysis); pooled over the 126 shard shapes "
                          "(t > 1); unsharded_table = measured best chunk of the same checkpoint at its full width; "
                          "calibration_width_table = measured best chunk at d = 2048 (the 370M width, where the cost "
                          "model is calibrated); intervals: shapes resampled with replacement"),
           "hosts": {}}
    du = cache.setdefault("du", _load_json(args.in_dir / "deployment_units.json"))
    if du:
        out["reference_per_key_profiling_cost_370M_keys_s"] = {
            h: {"P_profile_s_median": v["P_profile_s_median"], "profiling_12_keys_s": v["profiling_12_keys_s"]}
            for h, v in du["m6b_captured_keys"].items()}
        out["reference_note"] = ("key-churn model (deployment_units m6b_captured_keys): per-key profiling cost of the 33 "
                                 "captured 370M keys scaled from measured tuning wall clocks (RTX PRO 6000, RTX 3090)")
    for host in ("vgpu3090", "h800"):
        path = args.in_dir / f"e12_tp_widths_{host}.jsonl"
        if not path.exists():
            continue
        raw = E12.load(path)
        meta = json.loads((args.in_dir / f"e12_tp_widths_{host}.json").read_text())
        stored = (_load_json(args.in_dir / f"e12_analysis_{host}.json") or {}).get("families", {})
        out["hosts"][host] = {fam: tp_family(host, fam, raw, meta, stored, args, checks, unavailable)
                              for fam in sorted(raw["family"].unique())}
    return out


# =========================================================================== item 7 (M10b)
def m10b(args, checks: Checks) -> dict:
    out = {"definition": ("E1 value-invariance tables (inputs x rounds per family, length, mode and chunk). F test of "
                          "the input effect with rounds as blocks: df = (I - 1, (I - 1)(J - 1)). Pooled-mean bound "
                          "(e1_analysis): max_i (|m_i - m| + t_{0.95,df} se_i) / m with se_i = sqrt(MSE (I - 1) / (I J)). "
                          "Pairwise bound: max over input pairs (|m_i - m_j| + t se_d) / m with se_d = sqrt(2 MSE / J); "
                          "'iut' uses t_{0.95,df} (intersection-union test of all pairs, no multiplicity correction "
                          "needed), 'bonferroni' t_{1 - 0.05 / n_pairs, df} (simultaneous, conservative). Summaries at the "
                          "input-agnostic best chunk of each (family, length, mode)"),
           "tags": {}}
    for tag in E1_TAGS:
        p = args.in_dir / f"e1_{tag}.json"
        if not p.exists():
            continue
        rows = [r for r in json.loads(p.read_text())["rows"] if r.get("kind") == "timing"]
        dec = E1A.decision_analysis(rows)
        eff = E1A.effect_analysis(rows)
        eq = E1A.equivalence_analysis(rows, dec)
        c_star = {(d["family"], d["L"], d["mode"]): d["input_agnostic_best_chunk"] for d in dec}
        inputs = {}
        for r in rows:
            inputs.setdefault(r["family"], {}).setdefault(r["mode"], set()).add(r["input"])
        tab: dict = {}
        for r in rows:
            tab.setdefault((r["family"], r["L"], r["mode"], r["chunk"]), {}).setdefault(r["input"], {})[r["round"]] = r["mean_ms"]
        stored_eq = {(c["family"], c["L"], c["mode"], c["chunk"]): c["equivalence_bound_pct"] for c in eq["configs"]}
        configs = []
        for key, by_input in sorted(tab.items()):
            names = sorted(by_input)
            rnds = sorted(set.intersection(*[set(v) for v in by_input.values()]))
            if len(names) < 2 or len(rnds) < 2:
                continue
            X = np.array([[by_input[i][r] for r in rnds] for i in names])
            I, J = X.shape
            gm = X.mean()
            m = X.mean(1)
            resid = X - m[:, None] - X.mean(0)[None, :] + gm
            dof = (I - 1) * (J - 1)
            mse = (resid ** 2).sum() / dof
            se_i = np.sqrt(mse * (I - 1) / (I * J))
            se_d = np.sqrt(2 * mse / J)
            tq = float(t_dist.ppf(0.95, dof))
            npairs = I * (I - 1) // 2
            tb = float(t_dist.ppf(1 - 0.05 / npairs, dof))
            diff = np.abs(m[:, None] - m[None, :])
            iu = np.triu_indices(I, 1)
            worst = int(np.argmax(diff[iu]))
            pooled = float(100 * (np.abs(m - gm) + tq * se_i).max() / gm)
            rec = {"family": key[0], "L": key[1], "mode": key[2], "chunk": key[3], "n_inputs": I, "n_rounds": J,
                   "F_df": [I - 1, dof], "at_input_agnostic_best": key[3] == c_star.get(key[:3]),
                   "pooled_bound_pct": pooled,
                   "max_pair_abs_diff_pct": float(100 * diff[iu].max() / gm),
                   "max_abs_effect_vs_pooled_mean_pct": float(100 * np.abs(m - gm).max() / gm),
                   "pairwise_bound_iut_pct": float(100 * (diff[iu].max() + tq * se_d) / gm),
                   "pairwise_bound_bonferroni_pct": float(100 * (diff[iu].max() + tb * se_d) / gm),
                   "worst_pair": [names[iu[0][worst]], names[iu[1][worst]]]}
            configs.append(rec)
            if key in stored_eq:
                checks.stored(f"M10b.{tag}.{key}.pooled_bound", pooled, stored_eq[key], "e1_analysis.equivalence_analysis")

        def summ(sel: list[dict]) -> dict:
            if not sel:
                return {"n": 0}
            f = lambda k: np.array([c[k] for c in sel])  # noqa: E731
            ratio = f("pairwise_bound_iut_pct") / f("pooled_bound_pct")
            return {"n": len(sel),
                    "pooled_bound_pct": {"median": float(np.median(f("pooled_bound_pct"))), "max": float(f("pooled_bound_pct").max())},
                    "pairwise_bound_iut_pct": {"median": float(np.median(f("pairwise_bound_iut_pct"))),
                                               "max": float(f("pairwise_bound_iut_pct").max())},
                    "pairwise_bound_bonferroni_pct": {"median": float(np.median(f("pairwise_bound_bonferroni_pct"))),
                                                      "max": float(f("pairwise_bound_bonferroni_pct").max())},
                    "max_pair_abs_diff_pct": {"median": float(np.median(f("max_pair_abs_diff_pct"))),
                                              "max": float(f("max_pair_abs_diff_pct").max())},
                    "ratio_pairwise_iut_over_pooled": {"median": float(np.median(ratio)), "max": float(ratio.max())},
                    "n_pooled_within_2pct": int((f("pooled_bound_pct") < 2).sum()),
                    "n_pooled_within_5pct": int((f("pooled_bound_pct") < 5).sum()),
                    "n_pairwise_iut_within_2pct": int((f("pairwise_bound_iut_pct") < 2).sum()),
                    "n_pairwise_iut_within_5pct": int((f("pairwise_bound_iut_pct") < 5).sum())}

        fams = sorted({c["family"] for c in configs})
        res = {"source": p.name,
               "inputs_per_family_and_mode": {f: {m: len(v) for m, v in d.items()} for f, d in inputs.items()},
               "input_names": {f: {m: sorted(v) for m, v in d.items()} for f, d in inputs.items()},
               "F_test_df_per_family": {f: sorted({tuple(c["F_df"]) for c in configs if c["family"] == f}) for f in fams},
               "rounds_per_family": {f: sorted({c["n_rounds"] for c in configs if c["family"] == f}) for f in fams},
               "n_significant_input_effects_bh": int(sum(1 for r in eff if r.get("p_input_bh") is not None and r["p_input_bh"] < 0.05)),
               "n_fixed_chunk_tests": len(eff),
               "summary_at_best_chunk": {f: summ([c for c in configs if c["family"] == f and c["at_input_agnostic_best"]])
                                         for f in fams},
               "summary_all_chunks": {f: summ([c for c in configs if c["family"] == f]) for f in fams},
               "configs_at_best_chunk": [c for c in configs if c["at_input_agnostic_best"]]}
        res["summary_at_best_chunk"]["all"] = summ([c for c in configs if c["at_input_agnostic_best"]])
        res["summary_all_chunks"]["all"] = summ(configs)
        sa = _load_json(args.in_dir / f"e1_analysis_{tag}.json")
        if sa:
            st = sa["equivalence"]["summary"]["all"]["at_best_chunk"]
            checks.stored(f"M10b.{tag}.pooled_median_at_best", res["summary_at_best_chunk"]["all"]["pooled_bound_pct"]["median"],
                          st["median_bound_pct"], f"e1_analysis_{tag}.json equivalence.summary.all.at_best_chunk")
        out["tags"][tag] = res
    return out


# =========================================================================== item 8a (N2)
def n2(args) -> dict:
    out = {"definition": ("plug-in conditional mutual information of e3b_information (bits; entropy in 8 global quantile "
                          "bins, oracle chunk discrete) normalized by H(oracle | condition): normalized_cmi = I / H, "
                          "normalized_excess = (I - permutation-null mean) / H; context: the held-out best single "
                          "chunk (chunk effect) and the held-out exact-shape table (residual) regrets, and the paired "
                          "difference shape + entropy minus shape"),
           "data_sets": {}}
    for p in sorted(args.in_dir.glob("e3b_information_*.json")):
        if p.name.endswith("summary.json"):
            continue
        d = json.loads(p.read_text())
        rec = {"n_calls": d["n_calls"], "chunks": d["chunks"],
               "global_static_mean_regret_pct": d["predictors"].get("global_static", {}).get("mean_regret_pct"),
               "exact_key_table_mean_regret_pct": d["predictors"].get("exact_key_table", {}).get("mean_regret_pct"),
               "conditions": {}}
        for cond, xs in d["conditional_mi"]["by_condition"].items():
            rec["conditions"][cond] = {}
            for x in ("entropy_g8", "entropy_q4_within", "moments_kmeans8"):
                if x not in xs:
                    continue
                v = xs[x]
                H = v["H_oracle_given_cond_bits"]
                rec["conditions"][cond][x] = {
                    "cmi_bits": v["cmi_bits"], "null_mean_bits": v["null_mean_bits"], "excess_bits": v["excess_bits"],
                    "p_value": v["p_value"], "H_oracle_given_cond_bits": H,
                    "normalized_cmi": v["cmi_bits"] / H if H > 0 else None,
                    "normalized_excess": v["excess_bits"] / H if H > 0 else None}
        for k in ("shape+entropy - shape", "all - shape", "exact_key+entropy_bits - exact_key_table"):
            if k in d["differences"]:
                e = d["differences"][k]["prompt_cluster"]
                rec[f"diff {k}"] = {"diff_pp": e["diff_pp"], "ci95_pp": e["ci95_pp"]}
        out["data_sets"][p.stem[len("e3b_information_"):]] = rec
    ek = [(n, r["conditions"]["exact_key"]["entropy_g8"]) for n, r in out["data_sets"].items()
          if "exact_key" in r["conditions"] and r["conditions"]["exact_key"].get("entropy_g8", {}).get("normalized_cmi") is not None]
    if ek:
        out["max_normalized_cmi_given_exact_key"] = max((v["normalized_cmi"], n) for n, v in ek)
        out["max_normalized_excess_given_exact_key"] = max((v["normalized_excess"], n) for n, v in ek)
    gains = [(-r[f"diff {k}"]["diff_pp"], n, k) for n, r in out["data_sets"].items()
             for k in ("shape+entropy - shape", "all - shape") if f"diff {k}" in r]
    if gains:
        out["max_gain_of_activation_statistics_pp"] = max(gains)
    return out


# =========================================================================== manuscript checks
def manuscript_checks(res: dict, checks: Checks) -> None:
    m1 = res["M4a_selector_tables"]["mamba1"]
    pr = m1["pro6000_device"]["selectors"]
    loc = "paper/main.tex tab:oracle_regret (Mamba-1, device time, round medians)"
    for b, (c, mean, p95) in {"1": (256, "18.8", "50.5"), "4": (64, "20.5", "75.2"), "16": (64, "24.6", "93.3")}.items():
        r = pr["best_static_per_batch_in_sample"]["by_batch"][b]
        checks.printed(f"best static chunk, batch {b}", int(next(iter(r["chosen"]))), c, loc)
        checks.printed(f"best static mean regret, batch {b}", r["mean_regret_pct"], mean, loc)
        checks.printed(f"best static p95 regret, batch {b}", r["p95_regret_pct"], p95, loc)
    loc = "paper/main.tex tab:predictors (Mamba-1 column)"
    for n, (mean, p95, tied) in {"best_static_global_held_out": ("48.6", "308.6", "20"), "shape": ("0.5", "3.7", "93"),
                                 "exact_shape_table_held_out": ("0.1", "0.0", "98"), "cost_model": ("0.4", "3.8", "92"),
                                 "entropy_only": ("45.0", "247.5", "24")}.items():
        checks.printed(f"{n} mean", pr[n]["mean_regret_pct"], mean, loc)
        checks.printed(f"{n} p95", pr[n]["p95_regret_pct"], p95, loc)
        checks.printed(f"{n} tied %", 100 * pr[n]["ci_tie_share"], tied, loc)
    m2 = res["M4a_selector_tables"]["mamba2"]["mamba2_pro6000_launch"]["selectors"]
    loc = "paper/main.tex tab:predictors (Mamba-2 column)"
    for n, (mean, p95) in {"best_static_global_held_out": ("1.7", "5.8"), "shape": ("2.2", "5.6"),
                           "exact_shape_table_held_out": ("1.3", "5.2"), "work_term_model": ("1.6", "5.6")}.items():
        checks.printed(f"Mamba-2 {n} mean", m2[n]["mean_regret_pct"], mean, loc)
        checks.printed(f"Mamba-2 {n} p95", m2[n]["p95_regret_pct"], p95, loc)
    loc = "paper/main.tex tab:baselines (launch-inclusive)"
    printed = {"cost_model": (("0.88", "4.0"), ("3.25", "15.4"), ("3.54", "14.1")),
               "work_only": (("4.48", "50.1"), ("2.49", "13.1"), ("4.13", "17.4")),
               "nearest_calibrated_key": (("5.24", "16.0"), ("4.54", "20.5"), ("14.38", "78.3")),
               "loglinear_interp_logL_logB": (("3.88", "12.8"), ("4.16", "19.4"), ("14.34", "78.3")),
               "length_tree_calibration": (("2.67", "10.7"), ("9.87", "53.0"), ("19.92", "78.3")),
               "gbr_structural": (("1.26", "6.6"), ("6.18", "29.3"), ("6.26", "49.2")),
               "mlp_structural": (("3.95", "20.8"), ("4.15", "16.3"), ("4.79", "47.4")),
               "tfla_runtime": (("12.70", "88.1"), ("7.62", "31.8"), ("10.58", "67.4")),
               "cost_model_top2": (("0.52", "1.8"), ("2.30", "13.4"), ("0.73", "1.9")),
               "all_chunks_profiled": (("0.36", "1.2"), ("1.63", "12.1"), ("0.72", "1.9"))}
    for n, vals in printed.items():
        for s, (mean, p95) in zip(LAUNCH_M1, vals):
            if n in m1[s]["selectors"]:
                checks.printed(f"{s} {n} mean", m1[s]["selectors"][n]["mean_regret_pct"], mean, loc)
                checks.printed(f"{s} {n} p95", m1[s]["selectors"][n]["p95_regret_pct"], p95, loc)
    loc = "paper/main.tex sec:res_model (device time)"
    if "mlp_basic" in m1["h800_device"]["selectors"]:
        checks.printed("MLP (basic) loses on the H800, device time", m1["h800_device"]["selectors"]["mlp_basic"]["mean_regret_pct"], "8.0", loc)
    loc = "paper/main.tex sec:res_replication (RTX 3090 device time, round medians)"
    v = m1["vgpu3090_device"]["selectors"]
    checks.printed("single chunk (held out)", v["best_static_global_held_out"]["mean_regret_pct"], "33", loc)
    checks.printed("shape tree", v["shape"]["mean_regret_pct"], "0.4", loc)
    checks.printed("cost model", v["cost_model"]["mean_regret_pct"], "3.5", loc)
    checks.printed("H800 device-time cost model", m1["h800_device"]["selectors"]["cost_model"]["mean_regret_pct"], "3.0",
                   "paper/main.tex sec:res_datacentre")
    g = res["M4b_latency_weighted"]["a_cuda_graph"]
    loc = "paper/main.tex sec:res_reference (graph paragraph)"
    for nm, pct in (("ssd370m_graph", "19"), ("ssd2p7b_graph", "22"), ("gdn1p4b_graph", "16")):
        checks.printed(f"{nm} default per-call mean regret", g[nm]["subsets"]["all"]["policies"]["default"]["mean_regret_pct"], pct, loc)
    checks.printed("graph floor, SSD 370M (us)", g["ssd370m_graph"]["floor_us"], "45", loc)
    checks.printed("graph floor, Gated DeltaNet (us)", g["gdn1p4b_graph"]["floor_us"], "83", loc)
    checks.printed("share above graph floor, low end (%)", 100 * min(g[n]["share_above_floor"] for n in GRAPH_SETS[:3]), "64", loc)
    checks.printed("share above graph floor, high end (%)", 100 * max(g[n]["share_above_floor"] for n in GRAPH_SETS[:3]), "77", loc)
    loc = "paper/main.tex sec:res_breakeven (deployment units, CUDA graphs)"
    checks.printed("SSD 370M graph default ms per forward pass", g["ssd370m_graph"]["subsets"]["all"]["policies"]["default"]["ms_per_forward_pass"], "0.56", loc)
    checks.printed("SSD 2.7B graph default ms per forward pass", g["ssd2p7b_graph"]["subsets"]["all"]["policies"]["default"]["ms_per_forward_pass"], "1.44", loc)
    fc = res["N2_N3_N8"]["N3_launch_floor_coverage"]
    loc = "paper/main.tex sec:res_reference (production kernels at larger widths)"
    checks.printed("SSD 2.7B calls above the floor", fc["ssd2p7b_device"]["n_above_floor"], 397, loc)
    checks.printed("Gated DeltaNet calls above the floor", fc["gdn1p4b_device"]["n_above_floor"], 155, loc)
    checks.printed("Gated DeltaNet share above the floor (%)", 100 * fc["gdn1p4b_device"]["share_above_floor"], "9", loc)
    e1 = res["M10b_value_invariance"]["tags"]
    loc = "paper/main.tex sec:res_invariance (equivalence bounds at the selected chunk)"
    checks.printed("TOST median bound RTX PRO 6000", e1["pro6000"]["summary_at_best_chunk"]["all"]["pooled_bound_pct"]["median"], "3.6", loc)
    checks.printed("TOST median bound RTX 3090", e1["vgpu3090"]["summary_at_best_chunk"]["all"]["pooled_bound_pct"]["median"], "3.2", loc)
    tp = res["M7a_tensor_parallel"]["hosts"]
    loc = "paper/main.tex sec:res_reference (tensor-parallel widths)"
    t3, th = tp["vgpu3090"]["triton"]["pooled_shards"], tp["h800"]["triton"]["pooled_shards"]
    checks.printed("RTX 3090 cost model pooled mean", t3["cost_model"]["mean_regret_pct"], "6.5", loc)
    checks.printed("RTX 3090 cost model pooled p95", t3["cost_model"]["p95_regret_pct"], "31", loc)
    checks.printed("RTX 3090 calibration-width table pooled mean", t3["calibration_width_table"]["mean_regret_pct"], "16.5", loc)
    checks.printed("H800 cost model pooled mean", th["cost_model"]["mean_regret_pct"], "2.9", loc)
    checks.printed("H800 cost model pooled p95", th["cost_model"]["p95_regret_pct"], "13", loc)
    checks.printed("H800 calibration-width table pooled mean", th["calibration_width_table"]["mean_regret_pct"], "2.2", loc)
    ps = tp["vgpu3090"]["triton"]["per_shard"]
    for key, pct in (("1.4b_t2", "16"), ("1.4b_t4", "38"), ("1.4b_t8", "63"), ("2.8b_t2", "12"), ("2.8b_t4", "35"), ("2.8b_t8", "58")):
        checks.printed(f"RTX 3090 unsharded table {key}", ps[key]["unsharded_table"]["mean_regret_pct"], pct, loc)
    checks.printed("RTX 3090 cost model, lowest shard mean", tp["vgpu3090"]["triton"]["per_shard_maxima"]["cost_model"]["min_shard_mean_regret_pct"], "0.7", loc)
    checks.printed("RTX 3090 cost model, highest shard mean", tp["vgpu3090"]["triton"]["per_shard_maxima"]["cost_model"]["max_shard_mean_regret_pct"], "15", loc)
    checks.printed("H800 unsharded table, highest shard mean", tp["h800"]["triton"]["per_shard_maxima"]["unsharded_table"]["max_shard_mean_regret_pct"], "65", loc)
    dt = res["M3d_best_static_baseline"]["deployment_table_rows"]
    loc = "paper/main.tex tab:deployment"
    for label, (d, mdl) in {"Triton Mamba-1 370M, PRO 6000 (launch-inclusive)": ("4.30", "0.34"),
                            "Triton Mamba-1 370M, RTX 3090 (launch-inclusive)": ("5.71", "0.71"),
                            "Triton Mamba-1 370M, H800 (launch-inclusive)": ("4.13", "1.31"),
                            "Triton Mamba-1 1.4B, RTX 3090, device time": ("14.5", "2.66"),
                            "Triton Mamba-1 2.8B, RTX 3090, device time": ("25.2", "1.53")}.items():
        checks.printed(f"{label} default ms per pass", dt[label]["default_chunk_256"]["ms_per_forward_pass"], d, loc)
        checks.printed(f"{label} model ms per pass", dt[label]["cost_model"]["ms_per_forward_pass"], mdl, loc)


# =========================================================================== summary print
def _f(v, nd=2) -> str:
    if v is None:
        return "-"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_f(x, nd) for x in v) + "]"
    return f"{v:.{nd}f}"


def print_summary(res: dict) -> None:
    P = print
    P("\n=== M4b (1a) CUDA-graph replay, RTX 3090, three rounds (round-median oracle) ===")
    P(f"{'table':16s} {'subset':12s} {'policy':42s} {'n':>5s} {'mean%':>7s} {'p95%':>7s} {'max%':>7s} {'w2%':>5s} "
      f"{'lw/orc%':>8s} {'lw/pol%':>8s} {'us/call':>8s} {'ms/fp':>7s} {'orc ms/fp':>9s}")
    for nm, g in res["M4b_latency_weighted"]["a_cuda_graph"].items():
        for sub, s in g["subsets"].items():
            for p, r in s["policies"].items():
                P(f"{nm:16s} {sub:12s} {p:42s} {r['n']:5d} {_f(r['mean_regret_pct']):>7s} {_f(r['p95_regret_pct'], 1):>7s} "
                  f"{_f(r['max_regret_pct'], 1):>7s} {_f(100 * r['within_2pct_share'], 0):>5s} {_f(r['lw_share_of_oracle_pct']):>8s} "
                  f"{_f(r['lw_share_of_policy_pct']):>8s} {_f(r['us_per_call']):>8s} {_f(r['ms_per_forward_pass'], 3):>7s} "
                  f"{_f(r['oracle_scan_ms_per_forward_pass'], 2):>9s}")
        P(f"  {nm}: floor {g['floor_us']:.1f} us, {g['n_above_floor']}/{g['n_calls']} above "
          f"({100 * g['share_above_floor']:.1f}%); default mean CI {_f(g['subsets']['all']['policies']['default']['ci95']['mean_regret_pct'])}, "
          f"lw/orc CI {_f(g['subsets']['all']['policies']['default']['ci95']['lw_share_of_oracle_pct'])}")
    P("\n=== M4b (1b) best single Triton Mamba-1 chunk per batch (in sample), 370M ===")
    for n, d in res["M4b_latency_weighted"]["b_best_single_chunk_per_batch"].items():
        for b, r in d["by_batch"].items():
            P(f"{n:28s} b{b:>2s} chunk {r['chunk']:5d}  mean {_f(r['mean_regret_pct']):>6s}% {_f(r['mean_regret_pct_ci95'])}  "
              f"p95 {_f(r['p95_regret_pct'], 1):>6s}%  lw/orc {_f(r['lw_share_of_oracle_pct']):>6s}%  lw/pol "
              f"{_f(r['lw_share_of_policy_pct']):>6s}%  {_f(r['us_per_call'], 1):>7s} us/call  {_f(r['ms_per_forward_pass'], 3)} ms/fp "
              f"(oracle scan {_f(r['oracle_scan_ms_per_forward_pass'], 2)} ms/fp); by total latency: chunk "
              f"{r['by_total_latency']['chunk']} lw/orc {_f(r['by_total_latency']['lw_share_of_oracle_pct'])}%")
    P("\n=== M4b (1c) / N8 cost model per GPU ===")
    for n, r in res["M4b_latency_weighted"]["c_cost_model"].items():
        P(f"{n:28s} {r['convention'][:40]:40s} mean {_f(r['mean_regret_pct'])} {_f(r['mean_regret_pct_ci95'])} p95 "
          f"{_f(r['p95_regret_pct'], 1)} max {_f(r['max_regret_pct'], 1)} w2 {_f(100 * r['within_2pct_share'], 1)}% lw/orc "
          f"{_f(r['lw_share_of_oracle_pct'])} {_f(r['lw_share_of_oracle_pct_ci95'])} us/call {_f(r['us_per_call'], 1)} ms/fp "
          f"{_f(r['ms_per_forward_pass'], 3)}")
    P("\n=== M4a primary selector table (device time; mean [95% CI] / p95 / max / within-2% / lw-of-oracle) ===")
    t = res["M4a_selector_tables"]["primary_device_time_table"]
    P(f"{'selector':36s} " + " | ".join(f"{g:^52s}" for g in t["columns"]))
    for row, cells in t["rows"].items():
        P(f"{row:36s} " + " | ".join(f"{c:52s}" for c in cells))
    P("\n=== M4a secondary (launch-inclusive, session 1 / session 2 round medians), mean / p95 / max / w2 ===")
    t = res["M4a_selector_tables"]["secondary_launch_inclusive_table"]
    P(f"{'selector':36s} " + " | ".join(f"{g:^30s}" for g in t["columns"]))
    for row, cells in t["rows"].items():
        P(f"{row:36s} " + " | ".join(f"{c:30s}" for c in cells))
    P("\n=== M4a Mamba-2 370M (mean / p95 / max / w2 / lw-of-oracle) ===")
    for n, d in res["M4a_selector_tables"]["mamba2"].items():
        P(f"  {n} ({d['convention']})")
        for s in ("best_static_global_held_out", "best_static_per_batch_in_sample", "default_chunk_256", "shape",
                  "shape+entropy", "exact_shape_table_held_out", "work_term_model"):
            r = d["selectors"][s]
            P(f"    {s:36s} {_f(r['mean_regret_pct'])} {_f(r['ci95']['mean_regret_pct'])} / {_f(r['p95_regret_pct'], 1)} / "
              f"{_f(r['max_regret_pct'], 1)} / {_f(100 * r['within_2pct_share'], 0)}% / {_f(r['lw_share_of_oracle_pct'])}")
    P("\n=== M4e paired cost model minus baseline (pp, 95% CI), all calls ===")
    for n, d in res["M4e_intervals"]["paired_cost_model_minus"].items():
        P(f"  {n}: " + "; ".join(f"{k} {_f(v['diff_mean_regret_pp'])} {_f(v['ci95_pp'])}" for k, v in d.items()))
    P("\n=== M4d calibration-shape coverage (cost model) ===")
    for n, d in res["M4d_calibration_coverage"].items():
        r = d["selectors"]["cost_model"]
        P(f"  {n:28s} shapes {d['n_distinct_shapes_L_batch_width']} ({d['n_shapes_on_calibration_key']} on key), calls on key "
          f"{d['n_calls_on_calibration_key']}/{d['n_calls']} ({100 * d['share_calls_on_calibration_key']:.1f}%); on-key "
          f"{_f(r['on_key']['mean_regret_pct'])} {_f(r['on_key_mean_ci95'])}, off-key {_f(r['off_key']['mean_regret_pct'])} "
          f"{_f(r['off_key_mean_ci95'])}; per-shape mean {_f(r['per_shape_unweighted_mean_pct'])} "
          f"{_f(r['per_shape_unweighted_mean_ci95_shape_bootstrap'])} (on {_f(r['per_shape_unweighted_mean_on_key_pct'])}, "
          f"off {_f(r['per_shape_unweighted_mean_off_key_pct'])})")
    P("\n=== M3d deployment-table rows, ms per forward pass ===")
    for label, r in res["M3d_best_static_baseline"]["deployment_table_rows"].items():
        P(f"  {label:58s} " + "  ".join(f"{k.replace('best_static_', 'bs_')}={_f(v['ms_per_forward_pass'], 2)}"
                                         for k, v in r.items() if isinstance(v, dict) and "ms_per_forward_pass" in v)
          + f"  oracle scan {_f(r['oracle_scan_ms_per_forward_pass'], 2)}")
    P("\n=== M3d calibration-budget figure data (mean / p95) ===")
    for n, r in res["M3d_best_static_baseline"]["calibration_budget_figure"].items():
        P(f"  {n:16s} " + "  ".join(f"{k.replace('best_static_', 'bs_')}={_f(v['mean_regret_pct'])}/{_f(v['p95_regret_pct'], 1)}"
                                     for k, v in r.items() if isinstance(v, dict)))
    P("\n=== M7a tensor-parallel shard widths (pooled over shard shapes) ===")
    for host, fams in res["M7a_tensor_parallel"]["hosts"].items():
        for fam, r in fams.items():
            P(f"  {host} {fam}: {r['n_shard_shapes_pooled']} shard shapes")
            for name, s in r["pooled_shards"].items():
                P(f"    {name:40s} mean {_f(s.get('mean_regret_pct'))} {_f(s.get('mean_regret_pct_ci95_shape_bootstrap'))} p95 "
                  f"{_f(s.get('p95_regret_pct'), 1)} max {_f(s.get('max_regret_pct'), 1)} lw/orc {_f(s.get('lw_share_of_oracle_pct'))}")
            for name, s in r["per_shard_maxima"].items():
                P(f"    per-shard {name:30s} max {_f(s['max_shard_mean_regret_pct'])} at {s['at']}, min "
                  f"{_f(s['min_shard_mean_regret_pct'])} at {s['min_at']}")
            for k, v in r["paired_shape_bootstrap"].items():
                P(f"    paired {k:50s} {_f(v['diff_mean_regret_pp'])} {_f(v['ci95_pp'])}")
            rc = r["retune_cost_per_key"]["pooled_shards_s"]
            P(f"    retune profiling cost per key (kernel time): median {rc['median'] * 1000:.2f} ms, range "
              f"[{rc['min'] * 1000:.2f}, {rc['max'] * 1000:.2f}] ms")
    P("\n=== M10b value invariance ===")
    for tag, r in res["M10b_value_invariance"]["tags"].items():
        s = r["summary_at_best_chunk"]["all"]
        P(f"  {tag:16s} inputs {r['inputs_per_family_and_mode']} F df {r['F_test_df_per_family']}; at best chunk (n={s['n']}): "
          f"pooled median/max {_f(s['pooled_bound_pct']['median'])}/{_f(s['pooled_bound_pct']['max'])}, pairwise IUT "
          f"{_f(s['pairwise_bound_iut_pct']['median'])}/{_f(s['pairwise_bound_iut_pct']['max'])}, Bonferroni "
          f"{_f(s['pairwise_bound_bonferroni_pct']['median'])}/{_f(s['pairwise_bound_bonferroni_pct']['max'])}, ratio median "
          f"{_f(s['ratio_pairwise_iut_over_pooled']['median'])}")
    n = res["N2_N3_N8"]
    P("\n=== N2 normalized CMI I(entropy; oracle | exact key) / H(oracle | exact key) ===")
    for k, r in n["N2_normalized_cmi"]["data_sets"].items():
        e = r["conditions"].get("exact_key", {}).get("entropy_g8")
        if e:
            P(f"  {k:36s} I {e['cmi_bits']:.4f} null {e['null_mean_bits']:.4f} H {e['H_oracle_given_cond_bits']:.3f} -> "
              f"norm {_f(e['normalized_cmi'], 3)}, excess norm {_f(e['normalized_excess'], 3)} (p {e['p_value']:.3f}); "
              f"global static {_f(r['global_static_mean_regret_pct'])}%")
    P("\n=== N3 launch-floor coverage ===")
    for k, r in n["N3_launch_floor_coverage"].items():
        P(f"  {k:16s} floor {r['floor_us']:.1f} us: {r['n_above_floor']}/{r['n_calls']} above ({100 * r['share_above_floor']:.1f}%)")
    P("\n=== N8 cost model p95 and max ===")
    for k, r in n["N8_cost_model_p95_and_max"].items():
        P(f"  {k:28s} p95 {_f(r['p95_regret_pct'], 1)}  max {_f(r['max_regret_pct'], 1)}  ({r['convention']})")
    c = res["checks_summary"]
    P(f"\nchecks: stored {c['stored']['n_match']}/{c['stored']['n']}, manuscript {c['manuscript']['n_match']}/{c['manuscript']['n']}")
    for kind in ("stored", "manuscript"):
        for m in c[kind]["mismatches"][:40]:
            P(f"  MISMATCH [{kind}] {m['check']}: computed {m['computed']} vs {m.get('reference', m.get('printed'))} "
              f"({m.get('source', m.get('location'))})")
    P(f"\nunavailable: {len(res['unavailable'])} entries")
    for u in res["unavailable"]:
        P(f"  - {u['item']}: {u['quantity']}: {u['reason']}")


# =========================================================================== assembly
def _cell(r: dict) -> str:
    return (f"{_f(r['mean_regret_pct'])} {_f(r['ci95']['mean_regret_pct'])} / {_f(r['p95_regret_pct'], 1)} / "
            f"{_f(r['max_regret_pct'], 1)} / {_f(100 * r['within_2pct_share'], 0)}% / {_f(r['lw_share_of_oracle_pct'])}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--out", type=Path, default=None, help="output JSON (default: <in-dir>/taco_report_numbers.json)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--jobs", type=int, default=6, help="processes for the calibration-only selector fits")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="optional directory for the per-call choices of the calibration-only selectors (re-runs)")
    p.add_argument("--no-learned", dest="learned", action="store_false",
                   help="skip the gradient-boosted and MLP selectors (smoke test)")
    args = p.parse_args(argv)
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    t0 = time.time()
    checks, unavailable, cache = Checks(), [], {}
    res = {"experiment": "taco_report_numbers (cycle taco-r1: M4a, M4b, M4d, M4e, M3d, M7a, M10b, N2, N3, N8)",
           "created_utc": utc_now(), "definitions": __doc__,
           "settings": {"seed": SEED, "n_boot": args.n_boot, "within_margins": list(MARGINS), "learned_selectors": args.learned,
                        "scan_layers": {"Mamba-1 370M": 48, "Mamba-1 1.4B": 48, "Mamba-1 2.8B": 64, "Mamba-2 370M": 48,
                                        "Mamba-2 2.7B": 64, "Gated DeltaNet 1.4B hybrid": 12}}}

    log("calibration-only selectors (cost_model_baselines_r4.fit_selectors)")
    cals = list(dict.fromkeys(s["cal"] for s in M1_SETS.values()))
    cal = calibration_selectors(cals, args, log)
    calibration_own_table_checks(cal, args, checks, cache)

    log("Mamba-1 370M selector tables")
    m1, inner = {}, {}
    for name, spec in M1_SETS.items():
        m1[name], inner[name] = mamba1_set(name, spec, cal[spec["cal"]], args, checks, cache)
        log(f"  {name}: done ({time.time() - t0:.0f} s)")
    log("Mamba-2 370M selector tables")
    m2 = {name: mamba2_set(name, spec, args, checks, cache) for name, spec in M2_SETS.items()}

    # primary and secondary compact tables (M4a)
    prim = {"columns": [f"{m1[s]['gpu']} ({s})" for s in PRIMARY_M1], "cell": "mean [95% CI] / p95 / max / within 2% / "
            "latency-weighted share of oracle time (%)", "rows": {}}
    for row in M4A_ROWS:
        if all(row in m1[s]["selectors"] for s in PRIMARY_M1):
            prim["rows"][row] = [_cell(m1[s]["selectors"][row]) for s in PRIMARY_M1]
    sec_sets = ("pro6000_launch", "vgpu3090_launch", "h800_launch", "pro6000_launch_session2", "vgpu3090_launch_session2")
    sec = {"columns": [f"{m1[s]['gpu']} ({s})" for s in sec_sets], "cell": "mean / p95 / max / within 2%", "rows": {}}
    for row in M4A_ROWS:
        if all(row in m1[s]["selectors"] for s in sec_sets):
            sec["rows"][row] = [f"{_f(m1[s]['selectors'][row]['mean_regret_pct'])} / {_f(m1[s]['selectors'][row]['p95_regret_pct'], 1)} / "
                                f"{_f(m1[s]['selectors'][row]['max_regret_pct'], 1)} / "
                                f"{_f(100 * m1[s]['selectors'][row]['within_2pct_share'], 0)}%" for s in sec_sets]
    spread = {}
    for row in M4A_ROWS:
        a, b = (m1[s]["selectors"].get(row) for s in ("pro6000_launch", "pro6000_launch_session2"))
        c, d = (m1[s]["selectors"].get(row) for s in ("vgpu3090_launch", "vgpu3090_launch_session2"))
        if a and b and c and d:
            spread[row] = {"pro6000_session2_minus_session1_pp": b["mean_regret_pct"] - a["mean_regret_pct"],
                           "vgpu3090_session2_minus_session1_pp": d["mean_regret_pct"] - c["mean_regret_pct"]}
    sec["between_session_spread_mean_regret_pp"] = spread
    res["M4a_selector_tables"] = {
        "note": ("primary: device time (round medians where three rounds exist: RTX PRO 6000 and RTX 3090; H800 one "
                 "block); secondary: launch-inclusive, two sessions on the RTX PRO 6000 and RTX 3090. Every record has "
                 "per-call, latency-weighted and absolute regret, prompt-cluster intervals (ci95), per-batch records "
                 "(by_batch) and the paired cost model minus each selector (paired_cost_model_minus)"),
        "selector_definitions": {k: {"uses_captured_call_timings": v[0], "definition": v[1]} for k, v in SELECTOR_INFO.items()},
        "primary_device_time_table": prim, "secondary_launch_inclusive_table": sec, "mamba1": m1, "mamba2": m2}

    log("CUDA-graph tables (item 1a)")
    switched = E22.use_three_rounds(args.in_dir)
    graph = {nm: graph_set(nm, args, checks, cache) for nm in GRAPH_SETS}
    b1 = {}
    for s in ("pro6000_device", "vgpu3090_device", "vgpu3090_device_one_block", "h800_device"):
        sel = m1[s]["selectors"]
        rec = {"gpu": m1[s]["gpu"], "convention": m1[s]["convention"], "source": m1[s]["source"], "by_batch": {}}
        for b, r in sel["best_static_per_batch_in_sample"]["by_batch"].items():
            tl = sel["best_static_per_batch_by_total_latency"]["by_batch"][b]
            rec["by_batch"][b] = {"chunk": int(next(iter(r["chosen"]))), **_compact(r, ("ci_tie_share",)),
                                  "p95_regret_pct_ci95": r["ci95"]["p95_regret_pct"],
                                  "by_total_latency": {"chunk": int(next(iter(tl["chosen"]))), **_compact(tl)}}
        b1[s] = rec
    c1 = {}
    for s in M1_SETS:
        r = m1[s]["selectors"]["cost_model"]
        c1[s] = {"gpu": m1[s]["gpu"], "convention": m1[s]["convention"], **_compact(r, ("ci_tie_share",)),
                 "by_batch": {b: _compact(v) for b, v in r["by_batch"].items()}}
    res["M4b_latency_weighted"] = {
        "note": ("lw_share_of_oracle_pct = sum excess / sum oracle latency (the review's definition); "
                 "lw_share_of_policy_pct = sum excess / sum policy latency (the manuscript's definition); "
                 "ms_per_forward_pass = us_per_call x scan layers; oracle_scan_ms_per_forward_pass = oracle scan time of "
                 "one forward pass"),
        "three_round_graph_tables": switched, "a_cuda_graph": graph, "b_best_single_chunk_per_batch": b1, "c_cost_model": c1}

    log("M4d calibration coverage")
    res["M4d_calibration_coverage"] = {s: m4d_set(s, M1_SETS[s], inner[s], args) for s in M1_SETS}

    # M4e compact intervals
    m4e = {"rows": {}, "paired_cost_model_minus": {}}
    for s in PRIMARY_M1 + LAUNCH_M1:
        m4e["rows"][s] = {row: {"mean_regret_pct": m1[s]["selectors"][row]["mean_regret_pct"], **m1[s]["selectors"][row]["ci95"],
                                "by_batch_mean_ci95": {b: [v["mean_regret_pct"], v["ci95"]["mean_regret_pct"]]
                                                       for b, v in m1[s]["selectors"][row]["by_batch"].items()}}
                          for row in M4A_ROWS if row in m1[s]["selectors"]}
        m4e["paired_cost_model_minus"][s] = {k: v for k, v in m1[s]["paired_cost_model_minus"]["all_calls"].items()
                                             if k.split(" - ")[1] in M4A_ROWS}
    res["M4e_intervals"] = m4e

    log("M3d best static baseline")
    large = {n: large_set(n, csv, du, layers, args, checks, cache) for n, (csv, du, layers, _) in LARGE_SETS.items()}
    res["M3d_best_static_baseline"] = m3d(m1, large)
    res["M3d_best_static_baseline"]["larger_checkpoints"] = large

    log("M7a tensor-parallel widths")
    res["M7a_tensor_parallel"] = m7a(args, checks, unavailable, cache)
    log("M10b value invariance")
    res["M10b_value_invariance"] = m10b(args, checks)
    log("N2 / N3 / N8")
    res["N2_N3_N8"] = {"N2_normalized_cmi": n2(args), "N3_launch_floor_coverage": floor_coverage(args, checks, cache),
                       "N8_cost_model_p95_and_max": {s: {"convention": m1[s]["convention"],
                                                         "p95_regret_pct": m1[s]["selectors"]["cost_model"]["p95_regret_pct"],
                                                         "p95_regret_pct_ci95": m1[s]["selectors"]["cost_model"]["ci95"]["p95_regret_pct"],
                                                         "max_regret_pct": m1[s]["selectors"]["cost_model"]["max_regret_pct"],
                                                         "p99_note": "see cost_model_baselines_r4_summary for p99"}
                                                     for s in M1_SETS}}
    unavailable += [
        {"item": "M4b / M6a", "quantity": "CUDA-graph regrets on the RTX PRO 6000 and H800",
         "reason": "CUDA-graph timings of the captured calls exist only for the RTX 3090 (E22, E25)"},
        {"item": "M4a", "quantity": "H800 device time on round medians",
         "reason": "the H800 device-time sweep (E13b) has one block per chunk; no three-round sweep exists"},
        {"item": "M4a", "quantity": "Mamba-2 370M device time on the RTX PRO 6000 and H800",
         "reason": "Mamba-2 captured calls were timed launch-inclusive on the RTX PRO 6000 only; device time exists only on "
                   "the RTX 3090 (E15b, mixed regimes, and CUDA-graph replay)"},
        {"item": "M4a", "quantity": "calibration-only selectors other than the work-term model for Mamba-2 SSD",
         "reason": "no launch-space calibration sweep or compile facts exist for the SSD pipeline"},
        {"item": "M4a / M4e", "quantity": "within-2% share and intervals of interp_log_length_same_batch",
         "reason": "cost_model_baselines.nearest_key_selectors does not expose the per-call choice; mean / p95 / max are given"},
        {"item": "M7a", "quantity": "measured wall-clock profiling cost per key at the shard widths",
         "reason": "E12 records no tuning wall clock; the kernel-time lower bound from the E12 timings and the key-churn "
                   "per-key cost of the 370M keys are given instead"},
        {"item": "M4d", "quantity": "launch-inclusive second session on the H800",
         "reason": "no second launch-inclusive session was run on the H800"},
        {"item": "N2", "quantity": "count of data sets whose chunk effect exceeds the noise floor",
         "reason": "no per-data-set noise floor is defined in the result files; the held-out best-single-chunk and "
                   "exact-shape-table regrets are given per data set so a threshold can be applied"},
    ]
    res["unavailable"] = unavailable
    manuscript_checks(res, checks)
    res["checks"] = checks.rows
    res["checks_summary"] = checks.summary()
    res["runtime_s"] = round(time.time() - t0, 1)
    out = args.out or (args.in_dir / "taco_report_numbers.json")
    write_json(out, E22._clean(res))
    print_summary(E22._clean(res))
    log(f"\nwrote {out} ({out.stat().st_size / 1e6:.2f} MB, {res['runtime_s']} s)")


if __name__ == "__main__":
    main()
