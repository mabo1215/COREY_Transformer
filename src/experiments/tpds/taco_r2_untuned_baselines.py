"""taco-r2 items M2b and M2c: fair untuned baselines and one definition of "default" and "best static" per kernel.
CPU only; reads stored result files and times nothing.

M2b  Untuned and cheap baselines on every per-call data set of the Triton Mamba-1 scan of this study (device time,
     three GPUs; launch-inclusive; the 1.4B and 2.8B widths; CUDA-graph replay on two GPUs), of the reference CUDA
     scan (launch-inclusive and device time) and of the production kernels (Mamba-2 SSD 370M / 2.7B and Gated
     DeltaNet 1.4B under CUDA-graph replay on two GPUs, launch-inclusive and device time on the RTX 3090):
       free length rule   C = clip(nextpow2(L), C_min, C_max) over the kernel's chunk set, which mirrors the
                          reference scan's stock dispatch (clip(nextpow2(L), 128, 2048)); no timing at all
       capped default     max(C_min, min(C_default, nextpow2(L))) for kernels with a library default
       static chunks      one chunk for all calls / one per batch size, chosen by mean per-call regret or by total
                          latency, in sample or on the training prompt folds (held out)
       length-bucket table  keyed on (power-of-two length bucket, batch) [and the alignment bit], held out
       exact-shape table  the manuscript's held-out exact-shape table (profiled reference)
       cost model         the frozen eight-parameter fit and the a-priori forms of the M4a protocols (device time;
                          per-call choices from ``taco_r2_baselines_m4.json``)
     Each with per-call mean / p95 / max, within-2% share, latency-weighted share of the oracle scan time (LW),
     microseconds per call, ms per forward pass, per batch, and 95% prompt-cluster intervals; paired differences
     against the free length rule (per call and LW). Also why the calibration-only length tree loses to the rule.
M2c  An inventory of the definitions of "default" and "best static" used by the manuscript's tables and one
     proposed definition per kernel, with the affected numbers recomputed under it.

Outputs: ``taco_r2_baselines_m2b.json`` and ``taco_r2_baselines_m2c.json``.

Usage:  PYTHONPATH=. python3 -m src.experiments.tpds.taco_r2_untuned_baselines [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.experiments.tpds import cost_model_baselines as cmb
from src.experiments.tpds import cost_model_baselines_r4 as r4
from src.experiments.tpds import deployment_units as DU
from src.experiments.tpds import e3b_information as E3B
from src.experiments.tpds import taco_r2_baselines_common as C
from src.experiments.tpds.common import RESULTS_DIR

M1_DEFAULT_ROWS = ("default_chunk_256", "free_length_rule", "best_static_global_insample_regret",
                   "best_static_global_heldout_regret", "best_static_global_insample_latency",
                   "best_static_global_heldout_latency", "best_static_per_batch_insample_regret",
                   "best_static_per_batch_heldout_regret", "best_static_per_batch_insample_latency",
                   "best_static_per_batch_heldout_latency", "length_bucket_table_heldout_latency",
                   "length_bucket_aligned_table_heldout_latency", "length_bucket_table_heldout_regret",
                   "exact_shape_table_heldout_regret")
LARGE_SPEC = {"vgpu3090_1p4b_device": "mamba1_triton_1p4b_vgpu3090", "vgpu3090_2p8b_device": "mamba1_triton_2p8b_vgpu3090"}
REFSCAN = {"refscan_launch_3090": ("e8_refscan_vgpu3090.csv", False, "launch-inclusive, one block", 48),
           "refscan_device_3090": ("e8_refscan_vgpu3090_r4t32.csv", False, "device time, one block", 48)}
QUOTED = {"free_rule_per_call": {"pro6000_device": 5.86, "vgpu3090_device": 7.76, "h800_device": 10.66},
          "free_rule_lw": {"pro6000_device": 5.96, "vgpu3090_device": 16.31, "h800_device": 13.94},
          "per_batch_heldout": {"pro6000_device": 20.0, "vgpu3090_device": 17.4, "h800_device": 24.7},
          "chunk256_per_call": {"pro6000_device": 48.6, "vgpu3090_device": 96.4, "h800_device": 50.9},
          "global_latency_lw_pro6000": 15.5, "length_tree": {"pro6000_device": 16.6, "vgpu3090_device": 20.4, "h800_device": 38.4},
          "bucket_us": {"pro6000_device_pooled": 2.2, "vgpu3090_launch": 8.3, "h800_device": 19.8}}


# =========================================================================== helpers
def paired_lw(T: C.CallTable, a: np.ndarray, b: np.ndarray, label: str, n_boot: int) -> dict:
    """Difference of the latency-weighted shares (points of oracle scan time) of two picks with a prompt-cluster
    interval (ratio of replicate sums with common weights)."""
    ar = np.arange(T.N)
    ea, eb = T.lat[ar, a] - T.orc, T.lat[ar, b] - T.orc
    reps = E3B.cluster_bootstrap({"o": T.orc, "a": ea, "b": eb}, T.clusters, n_boot, C.TRN.make_rng(label, "lw_pair"), {})
    d = 100 * (reps["a"] - reps["b"]) / reps["o"]
    return {"diff_lw_pp": float(100 * (ea.sum() - eb.sum()) / T.orc.sum()), "ci95_pp": C.TRN._ci(d)}


def evaluate_set(T: C.CallTable, picks: dict, label: str, n_boot: int, ref: str = "free_length_rule") -> dict:
    pairs = [(n, ref) for n in picks if n != ref]
    recs, paired, k = C.evaluate(T, picks, label, n_boot, pairs)
    out = {"n_calls": T.N, "n_prompt_clusters": k, "chunks": T.chunks, "layers": T.layers, "source": T.source,
           "policies": {n: C.compact(recs[n]) for n in picks},
           "paired_vs_" + ref: {}}
    for n in picks:
        if n == ref:
            continue
        pc = paired["all_calls"][f"{n} - {ref}"]
        out["paired_vs_" + ref][n] = {"diff_mean_regret_pp": pc["diff_mean_regret_pp"], "ci95_pp": pc["ci95_pp"],
                                      "diff_us_per_call": pc["diff_us_per_call"], "ci95_us": pc["ci95_us"],
                                      **{f"lw_{k2}": v for k2, v in paired_lw(T, picks[n], picks[ref], f"{label}|{n}", n_boot).items()}}
    return out


def floor_mask(T: C.CallTable) -> tuple[float, np.ndarray]:
    small = (T.batch == 1) & (T.L <= 256)
    f = 2.0 * float(np.median(T.orc[small]))
    return f, T.orc > f


# =========================================================================== Triton Mamba-1
def triton_sets(args, m4: dict | None, checks: C.Checks) -> dict:
    out = {}
    for name, (csv, rmed, gpu, conv, cal, layers, role) in C.TRITON_SETS.items():
        T = C.triton_table(name, args.in_dir)
        picks, info = C.untuned_picks(T)
        picks = {n: picks[n] for n in M1_DEFAULT_ROWS}
        if cal is not None:
            picks["cost_model_frozen"] = C.frozen_cost_model_pick(T, cal, args.in_dir)
        if name in LARGE_SPEC:
            spec = next(s for s in DU.kernel_specs() if s["name"] == LARGE_SPEC[name])
            picks["cost_model_frozen"], _ = DU.model_choice(args.in_dir, spec, T.df, T.chunks, T.lat)
        if m4 is not None and name in m4["per_call_choices_device_time"]:
            pc = m4["per_call_choices_device_time"][name]
            for k in ("a_priori_paper_protocol", "a_priori_prespecified", "work_only"):
                picks[k] = C.map_choice(T, pc["keys"], pc[k])
        res = evaluate_set(T, picks, name, args.n_boot)
        res.update({"gpu": gpu, "convention": conv, "role": role, "rule_replacements": info})
        out[name] = res
        print(f"  {name:28s} free rule {res['policies']['free_length_rule']['mean_regret_pct']:.2f}% "
              f"(LW {res['policies']['free_length_rule']['lw_share_of_oracle_pct']:.2f}%)", flush=True)
    for nm in ("pro6000_device", "vgpu3090_device", "h800_device"):
        p = out[nm]["policies"]
        checks.add(f"free rule per call {nm}", p["free_length_rule"]["mean_regret_pct"], QUOTED["free_rule_per_call"][nm], "review M2b")
        checks.add(f"free rule LW {nm}", p["free_length_rule"]["lw_share_of_oracle_pct"], QUOTED["free_rule_lw"][nm], "review M2b")
        checks.add(f"best per batch held out {nm}", p["best_static_per_batch_heldout_regret"]["mean_regret_pct"],
                   QUOTED["per_batch_heldout"][nm], "Table 5", 0.05)
        checks.add(f"chunk 256 per call {nm}", p["default_chunk_256"]["mean_regret_pct"], QUOTED["chunk256_per_call"][nm], "Table 5", 0.05)
    checks.add("global static by total latency LW, RTX PRO 6000",
               out["pro6000_device"]["policies"]["best_static_global_insample_latency"]["lw_share_of_oracle_pct"],
               QUOTED["global_latency_lw_pro6000"], "review M2b", 0.05)
    for nm, q in QUOTED["bucket_us"].items():
        checks.add(f"bucket headroom us/call {nm} (Table S23)", out[nm]["policies"]["length_bucket_table_heldout_latency"]["us_per_call"],
                   q, "Table S23", 0.05)
    return out


def length_tree_explanation(args, triton: dict, m4: dict | None) -> dict:
    """The calibration-only length tree (12 keys, depth 3 on log2 L, leaves labelled with the lowest summed
    normalized latency over batches 1, 4 and 16) against the free rule, per batch."""
    out = {}
    for nm in ("pro6000_device", "vgpu3090_device", "h800_device"):
        cal = C.TRITON_SETS[nm][4]
        T = C.triton_table(nm, args.in_dir)
        keys = cmb.calibration_keys(cal, args.in_dir, T.chunks)
        tree = r4.CalibTree(3, ("log2_L",)).fit_keys(keys)
        Ls = np.array([2 ** i for i in range(4, 14)] + sorted(set(T.L.tolist())))
        sc = tree.scores(SimpleNamespace(shapes=pd.DataFrame({"L": Ls, "B": np.ones(len(Ls), int)})))
        tree_chunk = {int(L): int(T.chunks[j]) for L, j in zip(Ls[:10], sc[:10].argmin(1))}
        pk_tree = tree.scores(SimpleNamespace(shapes=pd.DataFrame({"L": T.L, "B": T.batch}))).argmin(1)
        pk_rule, _ = C.free_rule_pick(T)
        key_best = {f"L{k['L']}|B{k['B']}": int(T.chunks[int(np.argmin(k['mean_ms']))]) for k in keys}
        norm_sum = {}
        for L in sorted({k["L"] for k in keys}):
            ks = [k for k in keys if k["L"] == L]
            s = sum(k["mean_ms"] / k["mean_ms"].min() for k in ks)
            norm_sum[str(L)] = {"label_chunk": int(T.chunks[int(np.argmin(s))]),
                                "per_batch_best_chunk": {str(k["B"]): int(T.chunks[int(np.argmin(k["mean_ms"]))]) for k in ks},
                                "summed_normalized_latency": {str(c): round(float(v), 3) for c, v in zip(T.chunks, s)}}
        rec = {"tree_chunk_by_length_at_batch_1": tree_chunk, "calibration_key_best_chunk": key_best,
               "label_per_calibrated_length": norm_sum,
               "tree": C.compact(C.stats_of(T, pk_tree)), "free_rule": C.compact(C.stats_of(T, pk_rule))}
        for b in (1, 4, 16):
            m = T.batch == b
            rec[f"batch{b}"] = {"tree_mean_pct": C.stats_of(T, pk_tree, m)["mean_regret_pct"],
                                "rule_mean_pct": C.stats_of(T, pk_rule, m)["mean_regret_pct"],
                                "tree_lw_pct": C.stats_of(T, pk_tree, m)["lw_share_of_oracle_pct"],
                                "rule_lw_pct": C.stats_of(T, pk_rule, m)["lw_share_of_oracle_pct"]}
        # calls whose length is below the shortest calibration key (128)
        short = T.L < 128
        rec["calls_below_shortest_key"] = {"n": int(short.sum()), "share": float(short.mean()),
                                           "tree_mean_pct": C.stats_of(T, pk_tree, short)["mean_regret_pct"],
                                           "rule_mean_pct": C.stats_of(T, pk_rule, short)["mean_regret_pct"],
                                           "tree_chunks": C.stats_of(T, pk_tree, short)["chosen"],
                                           "oracle_chunks": {str(c): int(n) for c, n in zip(*np.unique(np.array(T.chunks)[T.lat[short].argmin(1)], return_counts=True))}}
        out[nm] = rec
    return out


# =========================================================================== production kernels and reference scan
def production_sets(args) -> dict:
    out = {}
    for name, (csv, rmed, gpu, conv, fam, default, layers) in C.PRODUCTION_SETS.items():
        if not (args.in_dir / csv).exists():
            continue
        T = C.production_table(name, args.in_dir)
        picks, info = C.untuned_picks(T, default=default)
        res = evaluate_set(T, picks, name, args.n_boot)
        f, above = floor_mask(T)
        res["floor_us"] = 1000 * f
        res["share_above_floor"] = float(above.mean())
        res["above_floor"] = {n: {k: v for k, v in C.stats_of(T, p, above).items() if k != "chosen"} for n, p in picks.items()}
        res["below_floor"] = {n: {k: v for k, v in C.stats_of(T, p, ~above).items() if k != "chosen"} for n, p in picks.items()}
        res.update({"gpu": gpu, "convention": conv, "kernel": fam, "library_default": default, "rule_replacements": info,
                    "n_calls_with_unavailable_chunk": int((~T.avail).any(1).sum())})
        out[name] = res
        p = res["policies"]
        print(f"  {name:26s} default {p['library_default']['mean_regret_pct']:.2f}% (LW {p['library_default']['lw_share_of_oracle_pct']:.2f}) "
              f"free rule {p['free_length_rule']['mean_regret_pct']:.2f}% (LW {p['free_length_rule']['lw_share_of_oracle_pct']:.2f}) "
              f"capped {p['capped_default']['mean_regret_pct']:.2f}%", flush=True)
    return out


def refscan_sets(args, checks: C.Checks) -> dict:
    out = {}
    for name, (csv, rmed, conv, layers) in REFSCAN.items():
        T = C.CallTable(C.load_view(csv, rmed, args.in_dir), name, layers, csv)
        picks, info = C.untuned_picks(T)
        picks = {n: picks[n] for n in M1_DEFAULT_ROWS if n in picks}
        picks["stock_length_rule"] = np.array([T.chunks.index(int(c)) for c in T.df["stock_chunk"]])
        res = evaluate_set(T, picks, name, args.n_boot)
        res["free_rule_equals_stock_rule_share"] = float((picks["free_length_rule"] == picks["stock_length_rule"]).mean())
        if "lat_stock_mean" in T.df:
            st = T.df["lat_stock_mean"].to_numpy(float)
            ok = np.isfinite(st)
            reg = st[ok] / T.orc[ok] - 1
            res["stock_binary_timed_separately"] = {"n": int(ok.sum()), "mean_regret_pct": float(100 * reg.mean()),
                                                    "lw_share_of_oracle_pct": float(100 * (st[ok] - T.orc[ok]).sum() / T.orc[ok].sum()),
                                                    "note": "the stock binary timed in its own block against the oracle of the "
                                                            "runtime-chunk build (Table 6's 'stock length dispatch')"}
        res.update({"convention": conv, "gpu": "RTX 3090", "kernel": "reference CUDA scan (runtime chunk, stock thread pairing)"})
        out[name] = res
    checks.add("reference scan stock rule within the runtime-chunk build (launch-inclusive)",
               out["refscan_launch_3090"]["policies"]["stock_length_rule"]["mean_regret_pct"], 5.35, "review N17", 0.01)
    checks.add("reference scan stock binary (launch-inclusive, Table 6)",
               out["refscan_launch_3090"]["stock_binary_timed_separately"]["mean_regret_pct"], 6.6, "Table 6", 0.05)
    checks.add("reference scan stock rule (device time, Table S21)",
               out["refscan_device_3090"]["policies"]["stock_length_rule"]["mean_regret_pct"], 3.09, "Table S21", 0.01)
    return out


# =========================================================================== M2c
def proposal_rows(tri: dict, prod: dict, ref: dict) -> dict:
    """The numbers of the main-text tables recomputed under the proposed definitions."""
    def pick(d, n, *keys):
        x = d["policies"][n]
        return {k: x[k] for k in keys if k in x}
    base = ("mean_regret_pct", "mean_regret_pct_ci95", "p95_regret_pct", "lw_share_of_oracle_pct", "lw_share_of_oracle_pct_ci95",
            "us_per_call", "ms_per_forward_pass", "oracle_scan_ms_per_forward_pass", "chosen")
    out = {}
    # Sec. 6 opening and Table 5 (device time, three GPUs)
    for nm in ("pro6000_device", "vgpu3090_device", "h800_device"):
        d = tri[nm]
        out[f"table5_{nm}"] = {
            "default_free_length_rule": pick(d, "free_length_rule", *base),
            "secondary_chunk_256": pick(d, "default_chunk_256", *base),
            "best_static_global_heldout_latency": pick(d, "best_static_global_heldout_latency", *base),
            "best_static_per_batch_heldout_latency": pick(d, "best_static_per_batch_heldout_latency", *base),
            "current_best_single_chunk_heldout_regret": pick(d, "best_static_global_heldout_regret", *base),
            "current_best_chunk_per_batch_heldout_regret": pick(d, "best_static_per_batch_heldout_regret", *base)}
    # Table 4 (RTX PRO 6000, device time, per batch)
    d = tri["pro6000_device"]["policies"]
    out["table4_pro6000_device_by_batch"] = {
        b: {"proposed_per_batch_heldout_latency": {k: d["best_static_per_batch_heldout_latency"]["by_batch"][b][k]
                                                   for k in ("chosen", "mean_regret_pct", "lw_share_of_oracle_pct")},
            "current_per_batch_insample_regret": {k: d["best_static_per_batch_insample_regret"]["by_batch"][b][k]
                                                  for k in ("chosen", "mean_regret_pct", "lw_share_of_oracle_pct")},
            "current_lw_column_insample_latency": {k: d["best_static_per_batch_insample_latency"]["by_batch"][b][k]
                                                   for k in ("chosen", "mean_regret_pct", "lw_share_of_oracle_pct")},
            "free_length_rule": {k: d["free_length_rule"]["by_batch"][b][k] for k in ("mean_regret_pct", "lw_share_of_oracle_pct")}}
        for b in ("1", "4", "16")}
    # Table 7 (graph replay): Triton default -> free rule; production kernels keep the library default
    for nm in ("vgpu3090_graph", "pro6000_graph"):
        dd = tri[nm]
        out[f"table7_triton_{nm}"] = {"default_free_length_rule": pick(dd, "free_length_rule", *base),
                                      "current_best_single_chunk_insample_regret": pick(dd, "best_static_global_insample_regret", *base),
                                      "best_static_global_heldout_latency": pick(dd, "best_static_global_heldout_latency", *base),
                                      "exact_shape_table_heldout_regret": pick(dd, "exact_shape_table_heldout_regret", *base)}
    for nm in [k for k in prod if "graph" in k]:
        dd = prod[nm]
        out[f"table7_{nm}"] = {"library_default": pick(dd, "library_default", *base),
                               "capped_default": pick(dd, "capped_default", *base),
                               "free_length_rule": pick(dd, "free_length_rule", *base),
                               "best_static_global_heldout_latency": pick(dd, "best_static_global_heldout_latency", *base),
                               "best_static_per_batch_heldout_latency": pick(dd, "best_static_per_batch_heldout_latency", *base),
                               "exact_shape_table_heldout_regret": pick(dd, "exact_shape_table_heldout_regret", *base)}
    # Table 8 (ms per forward pass)
    for nm in ("pro6000_device", "vgpu3090_device", "h800_device", "pro6000_launch", "vgpu3090_launch", "h800_launch",
               "vgpu3090_1p4b_device", "vgpu3090_2p8b_device"):
        dd = tri[nm]["policies"]
        out[f"table8_{nm}"] = {k: {"ms_per_forward_pass": dd[k]["ms_per_forward_pass"], "lw_share_of_oracle_pct": dd[k]["lw_share_of_oracle_pct"],
                                   "mean_regret_pct": dd[k]["mean_regret_pct"], "chosen": dd[k].get("chosen")}
                               for k in ("free_length_rule", "default_chunk_256", "best_static_per_batch_insample_latency",
                                         "best_static_per_batch_heldout_latency", "cost_model_frozen",
                                         "exact_shape_table_heldout_regret") if k in dd}
        out[f"table8_{nm}"]["oracle_scan_ms_per_forward_pass"] = dd["free_length_rule"]["oracle_scan_ms_per_forward_pass"]
    for nm in ref:
        dd = ref[nm]["policies"]
        out[f"table6_8_{nm}"] = {k: {f: dd[k][f] for f in ("mean_regret_pct", "lw_share_of_oracle_pct", "ms_per_forward_pass")}
                                 for k in ("stock_length_rule", "best_static_global_heldout_latency", "best_static_per_batch_heldout_latency",
                                           "best_static_global_insample_regret", "exact_shape_table_heldout_regret")}
        if "stock_binary_timed_separately" in ref[nm]:
            out[f"table6_8_{nm}"]["stock_binary_timed_separately"] = ref[nm]["stock_binary_timed_separately"]
    return out


INVENTORY = [
    {"where": "Sec. 6 opening (main.tex:653-658)", "kernel": "Triton Mamba-1 (ours)",
     "default": "chunk 256, 'the Mamba-2 default' (borrowed)",
     "best_static": "best chunk per batch size; per-call criterion, quoted with the latency-weighted share in parentheses",
     "values_printed": "49% per call (34% of scan time); 20% (27%)"},
    {"where": "Table 4 (tab:oracle_regret)", "kernel": "Triton Mamba-1 and Mamba-2 370M, RTX PRO 6000",
     "default": "none", "best_static": "column 'Best static chunk / Static regret': lowest mean per-call regret per batch, "
                                       "in sample; column 'LW': a different chunk, the one with the least total latency (512)",
     "values_printed": "B=1/4/16: 256/64/64, 18.8/20.5/24.6%; LW 512: 8.8/18.3/17.6"},
    {"where": "Table 5 (tab:predictors)", "kernel": "Triton Mamba-1",
     "default": "'Default chunk 256, entropy rule' (one row; coincide)",
     "best_static": "'Best single chunk, held out' (per-call criterion, prompt folds); 'Best chunk per batch, held out' "
                    "(per-call criterion; computed in sample in the stored analysis, identical values)",
     "values_printed": "48.6/96.4/50.9; 48.6/33.2/50.9; 20.0/17.4/24.7"},
    {"where": "Table 6 (tab:reference)", "kernel": "reference CUDA scan; Triton 1.4B/2.8B",
     "default": "'Stock length dispatch' = the stock binary timed in a separate block against the runtime-chunk oracle",
     "best_static": "'Best single chunk' (per-call criterion, in sample)", "values_printed": "6.6; 30.2/15.4/15.5"},
    {"where": "Table 7 (tab:graph) and Fig. 2", "kernel": "SSD, Gated DeltaNet, Triton Mamba-1",
     "default": "SSD 256, Gated DeltaNet 64 (library defaults); Triton: the best single chunk of the regime (per-call "
                "criterion, in sample)", "best_static": "--",
     "values_printed": "Triton default 31.8 / 46.0% per call, LW 17.9 / 32.3%"},
    {"where": "Table 8 (tab:deployment)", "kernel": "all",
     "default": "Triton 256; reference scan: stock dispatch; SSD 256; GDN 64",
     "best_static": "best chunk per batch chosen once by total latency (in sample); not given for production kernels",
     "values_printed": "Triton default 4.42/6.40/4.42/14.5/25.2 ms; best static 2.03/0.79/1.76/2.91/4.04 ms"},
    {"where": "Supplementary Table S23 (tab:supp_static_bucket)", "kernel": "Triton Mamba-1",
     "default": "none", "best_static": "best single chunk per batch (per-call criterion, in sample); bucket table by mean "
                                       "latency (held out) on pooled round means although labelled round medians",
     "values_printed": "PRO dev. 256/64/64 (18.8/20.5/24.6); bucket 2.2 us"},
    {"where": "Supplementary Fig. S (fig:calibration_budget)", "kernel": "Triton Mamba-1",
     "default": "chunk 256", "best_static": "best chunk per batch size chosen on held-out prompts (per-call criterion)",
     "values_printed": "--"},
    {"where": "Supplementary Table S (tab:supp_autotune)", "kernel": "Triton Mamba-1, Mamba-2 at the 12 keys",
     "default": "default launch table", "best_static": "best static chunk per key at the default launch; 'single global "
                                                       "static chunk' by the mean ratio to the tuned kernel (512/64 Mamba-1)",
     "values_printed": "--"},
    {"where": "Supplementary Tables S (tab:supp_graph, tab:supp_practitioner)", "kernel": "SSD, GDN, Triton",
     "default": "library defaults; Triton: best single chunk of the regime", "best_static": "best single chunk (in sample)",
     "values_printed": "--"},
    {"where": "Supplementary Table S (tab:supp_packed)", "kernel": "SSD, GDN packed steps",
     "default": "library defaults", "best_static": "best single chunk (in sample)", "values_printed": "--"},
    {"where": "Supplementary Tables S (tab:supp_deployment_buckets, tab:supp_serving)", "kernel": "Triton Mamba-1",
     "default": "chunk 256", "best_static": "--", "values_printed": "--"},
    {"where": "Supplementary Table S (tab:supp_largewidth)", "kernel": "SSD, GDN synthetic shapes",
     "default": "library defaults", "best_static": "single chunk with the lowest mean regret over the shapes (in sample)",
     "values_printed": "--"},
    {"where": "Supplementary TFLA table", "kernel": "TFLA mLSTM", "default": "library default 128",
     "best_static": "best single chunk (in sample); batch-keyed table (in sample)", "values_printed": "1.4%; 0.9%"},
]

PROPOSAL = {
    "default": {
        "Mamba-2 SSD": "library default chunk 256 (unchanged); the length-capped default max(C_min, min(256, nextpow2(L))) "
                       "as a zero-cost variant",
        "Gated DeltaNet": "library default chunk 64 (unchanged; the largest chunk the library path supports)",
        "TFLA mLSTM": "library default chunk 128 (unchanged)",
        "reference CUDA scan": "its stock length dispatch clip(nextpow2(L), 128, 2048), timed in the same (runtime-chunk) "
                               "binary and session as the other chunks",
        "Triton Mamba-1 scan (ours)": "no library default exists; use the free length rule clip(nextpow2(L), 32, 1024), "
                                      "the stock dispatch of the reference scan transplanted to this kernel's chunk range, "
                                      "as 'default'; keep chunk 256 as a secondary row labelled 'borrowed SSD default'"},
    "best_static": "one chunk per data set (and, as a separate row, one chunk per batch size), chosen as the chunk with "
                   "the lowest total latency (the deployment objective; equivalently the lowest latency-weighted share) "
                   "on the training prompt folds and scored on the held-out fold; report its per-call and latency-weighted "
                   "regret. Every table that uses another criterion (per-call regret) or an in-sample choice labels it.",
    "rationale": ["total latency is what a deployed static chunk minimizes, and the headline convention of the paper is "
                  "the latency-weighted share", "held out removes the in-sample/held-out label problem of Table 5",
                  "the same rule applies unchanged to every kernel and every timing convention",
                  "a kernel without a library default gets a zero-cost, untuned rule that an engineer would write, not a "
                  "borrowed constant"],
}


# =========================================================================== main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    t0 = time.time()
    checks = C.Checks()
    m4p = args.in_dir / f"{C.OUT_PREFIX}m4.json"
    m4 = json.loads(m4p.read_text()) if m4p.exists() else None
    print("Triton Mamba-1 data sets", flush=True)
    tri = triton_sets(args, m4, checks)
    print("reference scan", flush=True)
    ref = refscan_sets(args, checks)
    print("production kernels", flush=True)
    prod = production_sets(args)
    tree = length_tree_explanation(args, tri, m4)
    for nm, q in QUOTED["length_tree"].items():
        checks.add(f"calibration length tree {nm} (Table S19)", tree[nm]["tree"]["mean_regret_pct"], q, "Table S19", 0.06)
    # headline: what the calibrated model buys over the free rule (same data set)
    buys = {}
    for nm in ("pro6000_device", "vgpu3090_device", "h800_device", "pro6000_device_pooled", "vgpu3090_device_one_block"):
        pv = tri[nm]["paired_vs_free_length_rule"]
        buys[nm] = {k: {"per_call_points_below_rule": -pv[k]["diff_mean_regret_pp"], "per_call_ci95": [-x for x in pv[k]["ci95_pp"][::-1]],
                        "lw_points_below_rule": -pv[k]["lw_diff_lw_pp"], "lw_ci95": [-x for x in pv[k]["lw_ci95_pp"][::-1]]}
                    for k in ("cost_model_frozen", "a_priori_paper_protocol", "a_priori_prespecified", "work_only",
                              "best_static_per_batch_heldout_latency", "best_static_per_batch_heldout_regret",
                              "length_bucket_table_heldout_latency", "exact_shape_table_heldout_regret", "default_chunk_256")
                    if k in pv}
    m2b = {"experiment": "taco-r2 M2b fair untuned baselines", "definitions": __doc__,
           "policy_definitions": C.POLICY_INFO,
           "conventions": {"lw_share_of_oracle_pct": "100 x sum(excess latency) / sum(oracle latency) (Tables 4 and 7)",
                           "intervals": f"95% prompt-cluster bootstrap, {args.n_boot} replicates"},
           "triton_mamba1": tri, "reference_scan": ref, "production_kernels": prod,
           "calibration_length_tree_vs_free_rule": tree, "model_gain_over_free_rule": buys,
           "review_quote_note": "the review's 'calibration buys 1.7 / 1.5 / 7.7 points' subtracts the a-priori model of "
                                "Table 5 (RTX 3090 from the single-block session) from the free rule on the round-median "
                                "data set; model_gain_over_free_rule evaluates both on the same data set"}
    m2b["checks"] = checks.summary()
    path = C.save("m2b", m2b, args.in_dir)
    print(f"wrote {path}; checks {m2b['checks']['n_match']}/{m2b['checks']['n']}")
    for m in m2b["checks"]["mismatches"]:
        print("  MISMATCH", m["check"], m["computed"], m["reference"])
    m2c = {"experiment": "taco-r2 M2c definitions of default and best static", "inventory": INVENTORY, "proposal": PROPOSAL,
           "recomputed": proposal_rows(tri, prod, ref)}
    path = C.save("m2c", m2c, args.in_dir)
    print(f"wrote {path}; runtime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
