"""taco-r2 minor items N17, N18 and N23 (offline). CPU only; reads stored result files and times nothing.

N17  The held-out exact-shape table of the Triton Mamba-1 scan (Table 5; device time, three GPUs): the population it
     is averaged over, the fallback rule for calls whose exact shape has no entry in the training folds, the regret
     under other fallback rules (plain length bucket, no alignment bit; global chunk only) and entry criteria (mean
     per-call regret, as Table 5; mean latency, as the deployment table), and the absolute headroom in microseconds
     over covered calls and over all calls.
N18  The reference CUDA scan's stock length dispatch (RTX 3090, 2360 captured calls), launch-inclusive and in device
     time: (i) the runtime-chunk build at the stock (chunk, threads) pairing, (ii) the rebuilt module's own stock
     dispatch timed as a separate call, (iii) the installed stock binary timed in its own block (Table 6's 6.6%),
     the re-timing ratios of the same configuration across binaries and within the rebuilt binary, and the shape
     tree (held out) on the same basis, with prompt-cluster intervals and latency-weighted shares.
N23  The zero-parameter compile-facts rule of the top-k table, defined from the code that scores it
     (``taco_model_baselines.compile_facts_matrix``), re-implemented independently and checked.

Output: ``taco_r2_baselines_minor.json``.

Usage:  PYTHONPATH=. python3 -m src.experiments.tpds.taco_r2_minor_checks [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from src.experiments.tpds import cost_model as cm
from src.experiments.tpds import cost_model_baselines_r4 as r4
from src.experiments.tpds import e3_predictors as E3
from src.experiments.tpds import e3b_information as E3B
from src.experiments.tpds import taco_model_baselines as TMB
from src.experiments.tpds import taco_r2_baselines_common as C
from src.experiments.tpds import taco_report_numbers as TRN
from src.experiments.tpds.analysis import cv_exact_headroom
from src.experiments.tpds.common import RESULTS_DIR

TRITON = ("pro6000_device", "vgpu3090_device", "h800_device")
REFSCAN = {"launch_inclusive": "e8_refscan_vgpu3090.csv", "device_time": "e8_refscan_vgpu3090_r4t32.csv"}


def e3b_table(name: str, csv: str, rmed: bool, in_dir: Path) -> E3B.Table:
    return TRN.load_table(in_dir, csv, rmed, name, "mamba1", "x")


# =========================================================================== N17
def n17(in_dir: Path, n_boot: int, checks: C.Checks) -> dict:
    out = {}
    for nm in TRITON:
        csv, rmed = C.TRITON_SETS[nm][0], C.TRITON_SETS[nm][1]
        t = e3b_table(nm, csv, rmed, in_dir)
        T = C.triton_table(nm, in_dir)
        assert T.keys() == [str(k) for k in t.df["key"]]
        lat_fit = np.where(np.isfinite(t.lat), t.lat, 1e6)
        variants = {
            "table5_exact_then_bucket_aligned_then_global_regret": [t.key, t.bucket_key],
            "exact_then_plain_bucket_then_global_regret": [t.key, T.bucket],
            "exact_then_global_regret": [t.key],
        }
        picks, levels = {}, {}
        for vn, lv in variants.items():
            picks[vn], info = E3B.cv_table(t, lv)
            levels[vn] = info["calls_per_level"]
        picks["exact_then_bucket_aligned_then_global_latency"] = C.table_pick(T, [T.key, T.bucket_aligned], T.lat_fit, True)
        picks["exact_then_plain_bucket_then_global_latency"] = C.table_pick(T, [T.key, T.bucket], T.lat_fit, True)
        picks["exact_then_global_latency"] = C.table_pick(T, [T.key], T.lat_fit, True)
        # covered = the call's exact key (L, batch, width, alignment) has training calls in the other folds
        covered = np.zeros(t.N, bool)
        for f in np.unique(t.folds):
            tr, te = t.folds != f, t.folds == f
            have = set(t.key[tr].tolist())
            covered[te] = np.isin(t.key[te], list(have))
        recs, _, _ = C.evaluate(T, picks, f"{nm}|n17", n_boot)
        rec = {"population": "all 2360 captured calls; per prompt fold the table is built on the other four folds",
               "n_calls": int(t.N), "n_covered": int(covered.sum()), "n_not_covered": int((~covered).sum()),
               "calls_per_level_table5": {"exact": levels["table5_exact_then_bucket_aligned_then_global_regret"][0],
                                          "bucket_aligned": levels["table5_exact_then_bucket_aligned_then_global_regret"][1],
                                          "global": levels["table5_exact_then_bucket_aligned_then_global_regret"][2]},
               "variants": {}}
        for vn, pk in picks.items():
            r_all = C.stats_of(T, pk)
            r_cov = C.stats_of(T, pk, covered)
            r_unc = C.stats_of(T, pk, ~covered)
            rec["variants"][vn] = {"all_calls": {k: r_all[k] for k in ("mean_regret_pct", "lw_share_of_oracle_pct", "us_per_call")},
                                   "all_calls_mean_ci95": recs[vn]["ci95"]["mean_regret_pct"],
                                   "covered": {k: r_cov[k] for k in ("n", "mean_regret_pct", "us_per_call")},
                                   "not_covered": {k: r_unc[k] for k in ("n", "mean_regret_pct", "us_per_call")}}
        # the estimator of the headroom analyses (exact (L, batch) by mean latency, fallback bucket + alignment, global)
        h = cv_exact_headroom(t.df, t.chunks)
        rec["cv_exact_headroom_estimator"] = {"all_calls_us": 1000 * h["mean_ms"], "all_calls_pct": h["mean_pct"],
                                              "covered_us": 1000 * h["covered"]["mean_ms"], "covered_n": h["covered"]["n_calls"],
                                              "fallback_us": 1000 * h["fallback"]["mean_ms"], "fallback_n": h["fallback"]["n_calls"],
                                              "fallback_pct": h["fallback"]["mean_pct"], "covered_pct": h["covered"]["mean_pct"],
                                              "fallbacks": h["fallbacks"]}
        out[nm] = rec
    p = out["pro6000_device"]
    checks.add("N17 Table 5 exact-shape table, RTX PRO 6000", p["variants"]["table5_exact_then_bucket_aligned_then_global_regret"]
               ["all_calls"]["mean_regret_pct"], 0.14, "Table 5", 0.005)
    checks.add("N17 not-covered calls, RTX PRO 6000", p["n_not_covered"], 204, "review N17", 0)
    checks.add("N17 headroom over covered calls (us), RTX PRO 6000", p["cv_exact_headroom_estimator"]["covered_us"], 0.14,
               "main text", 0.005)
    checks.add("N17 headroom over all calls (us), RTX PRO 6000", p["cv_exact_headroom_estimator"]["all_calls_us"], 0.37,
               "review N17", 0.005)
    return out


# =========================================================================== N18
def n18(in_dir: Path, n_boot: int, checks: C.Checks) -> dict:
    out = {}
    for conv, csv in REFSCAN.items():
        nm = f"refscan_{conv}"
        t = e3b_table(nm, csv, False, in_dir)
        T = C.CallTable(C.load_view(csv, False, in_dir), nm, 48, csv)
        assert T.keys() == [str(k) for k in t.df["key"]]
        stock = np.array([T.chunks.index(int(c)) for c in T.df["stock_chunk"]])
        picks = {"stock_pairing_runtime_build": stock}
        for fs in ("shape", "L_only", "shape+entropy"):
            cols, depth = E3.FEATURE_SETS[fs]
            picks[f"tree_{fs}_heldout"] = E3B.cv_tree(t, cols, depth)
        picks["exact_shape_table_heldout"] = E3B.cv_table(t, [t.key, t.bucket_key])[0]
        recs, paired, k = C.evaluate(T, picks, nm, n_boot, [("tree_shape_heldout", "stock_pairing_runtime_build")])
        ar = np.arange(T.N)
        at_stock = T.lat[ar, stock]
        rec = {"timing": conv, "source": csv, "n_calls": T.N, "n_prompt_clusters": k, "chunks": T.chunks,
               "policies": {n: C.compact(recs[n], by_batch=True) for n in picks},
               "paired_shape_tree_minus_stock": paired["all_calls"]["tree_shape_heldout - stock_pairing_runtime_build"]}
        for col, label in (("lat_stock_mean", "installed_stock_binary_separate_block"),
                           ("lat_stock_rebuilt_mean", "rebuilt_module_stock_dispatch_separate_call")):
            if col not in T.df:
                continue
            v = T.df[col].to_numpy(float)
            ok = np.isfinite(v)
            reg = v[ok] / T.orc[ok] - 1
            ratio = v[ok] / at_stock[ok]
            rec[label] = {"n": int(ok.sum()), "mean_regret_pct_vs_runtime_build_oracle": float(100 * reg.mean()),
                          "lw_share_of_oracle_pct": float(100 * (v[ok] - T.orc[ok]).sum() / T.orc[ok].sum()),
                          "ratio_to_runtime_build_at_stock_chunk": {
                              "p10": float(np.percentile(ratio, 10)), "median": float(np.median(ratio)),
                              "p90": float(np.percentile(ratio, 90)), "min": float(ratio.min()), "max": float(ratio.max())}}
        s, tr = rec["policies"]["stock_pairing_runtime_build"]["mean_regret_pct"], rec["policies"]["tree_shape_heldout"]["mean_regret_pct"]
        rec["shape_tree_relative_reduction"] = {
            "vs_runtime_build_stock_pct": 100 * (1 - tr / s),
            "vs_installed_binary_pct": 100 * (1 - tr / rec["installed_stock_binary_separate_block"]["mean_regret_pct_vs_runtime_build_oracle"])}
        out[conv] = rec
    li, dt = out["launch_inclusive"], out["device_time"]
    checks.add("N18 stock pairing, runtime build, launch-inclusive", li["policies"]["stock_pairing_runtime_build"]["mean_regret_pct"],
               5.35, "review N18", 0.005)
    checks.add("N18 installed stock binary, launch-inclusive (Table 6)",
               li["installed_stock_binary_separate_block"]["mean_regret_pct_vs_runtime_build_oracle"], 6.6, "Table 6", 0.05)
    checks.add("N18 stock pairing, device time (Table S21)", dt["policies"]["stock_pairing_runtime_build"]["mean_regret_pct"],
               3.1, "Table S21", 0.05)
    checks.add("N18 shape tree, launch-inclusive (Table 6)", li["policies"]["tree_shape_heldout"]["mean_regret_pct"], 3.0,
               "Table 6", 0.05)
    checks.add("N18 shape tree, device time (Table S21)", dt["policies"]["tree_shape_heldout"]["mean_regret_pct"], 0.60,
               "Table S21", 0.005)
    checks.add("N18 cross-binary ratio p10, launch-inclusive",
               li["installed_stock_binary_separate_block"]["ratio_to_runtime_build_at_stock_chunk"]["p10"], 0.86, "review N18", 0.005)
    checks.add("N18 cross-binary ratio p90, launch-inclusive",
               li["installed_stock_binary_separate_block"]["ratio_to_runtime_build_at_stock_chunk"]["p90"], 1.13, "review N18", 0.005)
    return out


# =========================================================================== N23
def n23(in_dir: Path, checks: C.Checks) -> dict:
    out = {"definition": (
        "Candidates: the six chunks of the default launch table, each at its default (BLOCK_D, num_warps) and one "
        "stage. For each candidate the kernel is compiled (not run) for the target architecture and for the call's "
        "alignment specialization (L mod 16 = 0 or not; the aligned facts are used when an unaligned variant is "
        "missing), and the compiler's resource report gives the registers per thread and the spill store and load "
        "bytes. From the registers, the warps and the SM limits it computes the resident programs per SM "
        "k = min(floor(regs_per_SM / (ceil(regs / 8) x 8 x 32 x num_warps)), floor(max_warps_per_SM / num_warps), "
        "max_blocks_per_SM) (at least 1), the number of programs P = ceil(d / BLOCK_D) x B and the waves "
        "w = ceil(P / (n_SM x k)). The score is S(C) = 1e9 x [spill bytes > 0] + w x ceil(L / C) - 0.001 x log2 C; the "
        "rule serves the chunk with the lowest score: any non-spilling chunk before any spilling one, then the fewest "
        "sequential wave-tiles (waves times time tiles per program), ties to the larger chunk. No timing and no fitted "
        "parameter enter the rule; the hardware constants are the SM count and the CUDA occupancy limits."),
        "per_gpu": {}}
    for gpu, tag in TMB.TAGS["device_time"].items():
        ctx = r4.Ctx(tag, in_dir)
        S = TMB.compile_facts_matrix(ctx)
        ch = S.argmin(1)
        # independent re-implementation from the raw compile facts
        stats, hw = ctx.stats, ctx.hw
        rows = []
        for L, B, d in ctx.calls.shapes[["L", "B", "d"]].itertuples(index=False):
            sc = []
            for c in ctx.chunks:
                bd, nw = cm.DEFAULT_LAUNCH[c]
                r = cm._stats_row(stats, int(L) % 16 == 0, c, bd, nw, 1)
                regs = int(r["regs"])
                kk = max(1, min(hw["regs_per_sm"] // (int(math.ceil(regs / 8.0) * 8) * 32 * nw),
                                hw["max_warps_per_sm"] // nw, hw["max_blocks_per_sm"]))
                w = math.ceil(math.ceil(d / bd) * B / (hw["n_sm"] * kk))
                spill = (r.get("spill_store_bytes", 0) + r.get("spill_load_bytes", 0)) > 0
                sc.append(1e9 * spill + w * math.ceil(L / c) - 1e-3 * math.log2(c))
            rows.append(sc)
        mine = np.array(rows)
        same = float((mine.argmin(1) == ch).mean())
        reg = TMB.call_regret(ctx.lat, ch)
        spilling = {("aligned" if al else "unaligned"): {
            str(c): bool((cm._stats_row(stats, al, c, *cm.DEFAULT_LAUNCH[c], 1).get("spill_store_bytes", 0)
                          + cm._stats_row(stats, al, c, *cm.DEFAULT_LAUNCH[c], 1).get("spill_load_bytes", 0)) > 0)
            for c in ctx.chunks} for al in (True, False)}
        u, n = np.unique(np.array(ctx.chunks)[ch], return_counts=True)
        out["per_gpu"][gpu] = {"tag": tag, "independent_reimplementation_agrees_share": same,
                               "spilling_default_entries": spilling,
                               "chosen_chunks": {str(int(a)): int(b) for a, b in zip(u, n)},
                               "own_table_mean_regret_pct": float(100 * reg.mean())}
        checks.add(f"N23 re-implementation agrees ({gpu})", same, 1.0, "taco_model_baselines.compile_facts_matrix", 0)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    checks = C.Checks()
    res = {"experiment": "taco-r2 N17, N18, N23", "definitions": __doc__,
           "N17": n17(args.in_dir, args.n_boot, checks), "N18": n18(args.in_dir, args.n_boot, checks),
           "N23": n23(args.in_dir, checks)}
    res["checks"] = checks.summary()
    path = C.save("minor", res, args.in_dir)
    print(f"wrote {path}; checks {res['checks']['n_match']}/{res['checks']['n']}")
    for m in res["checks"]["mismatches"]:
        print("  MISMATCH", m["check"], m["computed"], m["reference"])


if __name__ == "__main__":
    main()
