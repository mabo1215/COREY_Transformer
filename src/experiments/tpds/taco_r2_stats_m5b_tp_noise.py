"""Review item M5b (cycle taco-r2): is the RTX 3090 tensor-parallel sweep noise-dominated? CPU only.

Data: the E12 sweep (``e12_tp_widths_<host>.jsonl``): synthetic inputs at the widths of three Mamba-1 checkpoints and
their tensor-parallel shards (512 to 5120 channels), seven lengths (128 to 8192), batch 1, 4 and 16, every chunk timed
in three rounds of one 50-repeat block (``family`` triton: the Triton scan, chunks 32 to 1024; ``ref``: the reference
CUDA scan, chunks 128 to 2048), on the RTX 3090 (``vgpu3090``) and the H800 (``h800``).

Selectors (exactly as ``taco_report_numbers.tp_family`` / ``e12_analysis``): the frozen cost model calibrated at
d = 2048 on the same GPU (none for the reference scan on the H800), the nearest calibration key (Triton), a table tuned
at the calibration width d = 2048, the best single chunk, the unsharded table (measured best chunk of the same
checkpoint at its full width), and the table retuned at the exact shard shape.

1. Reproduction of the stored M7a numbers (``taco_report_numbers.json``): pooled mean regret per selector against the
   three-round-mean oracle, the retuned table scored cross-round, and the paired shape-bootstrap intervals (same
   generator ``taco_report_numbers.make_rng('tp', host, fam)``); the round-agreement share of ``e12_analysis``
   (share of shapes whose per-round argmin is identical in all three rounds).
2. Round-to-round variation of the identical kernel: between-round relative SD per (shape, chunk), |relative
   difference| between two rounds, share of round pairs whose difference exceeds the within-block standard error
   (|z| > 1.96), per-round argmin agreement (all three rounds; pairwise).
3. Chunk effect against noise: per shape a two-way (chunk x round) analysis of log latency without replication
   (F test of the chunk effect against the chunk-by-round residual; Benjamini-Hochberg over shapes); the gap between
   the best and the second-best chunk against the round-to-round SD of that gap; per shard shape the unsharded-table
   regret against the shape's own noise floor (the retuned table scored cross-round).
4. One scoring for every selector. ``cross_round``: for each held-out round r the regret is R_r[c] / min_c R_r - 1;
   fixed selectors use their fixed chunk, the retuned table the chunk with the lowest mean over the other two rounds;
   averaged over the three held-out rounds. ``mean_oracle_loro``: every selector scored against the three-round mean
   oracle (as stored), the retuned table choosing on two rounds (leave one round out; optimistic, the scoring
   rounds include its choosing rounds). Paired differences (cost model minus each other selector) with 2000
   shape-bootstrap replicates (``taco_r2_stats_common.make_rng('m5b', host, fam)``).

Output: ``taco_r2_stats_m5b.json``.

Usage:  PYTHONPATH=. python3 -m src.experiments.tpds.taco_r2_stats_m5b_tp_noise [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from src.experiments.tpds import e12_analysis as E12
from src.experiments.tpds import taco_r2_stats_common as C
from src.experiments.tpds import taco_report_numbers as TRN
from src.experiments.tpds.common import MAMBA1_CHUNKS, RESULTS_DIR, read_jsonl
from src.experiments.tpds.e12_tp_widths import MODEL_WIDTHS, REF_CHUNKS

HOSTS = ("vgpu3090", "h800")
GPU = {"vgpu3090": "RTX 3090", "h800": "H800 PCIe"}


def _ci(x) -> list[float]:
    return C.pct_ci(np.asarray(x, float))


def selectors(tab: pd.DataFrame, chunks: list[int], host: str, fam: str, in_dir: Path) -> tuple[dict, dict]:
    """Fixed per-shape choices of the E12 selectors (as ``taco_report_numbers.tp_family``)."""
    sel, notes = {}, {}
    try:
        sel["cost_model"] = (E12.triton_model_choice if fam == "triton" else E12.ref_model_choice)(tab, chunks, host, in_dir)
    except (FileNotFoundError, KeyError) as exc:
        notes["cost_model"] = f"unavailable: no stored cost-model parameters for this kernel and GPU ({exc!r})"
    if fam == "triton":
        nk = E12.nearest_key_choice(tab, chunks, host, in_dir)
        if nk:
            sel["nearest_key"] = nk
    sel["calibration_width_table"] = E12.table_choice(tab, lambda d: 2048)
    regs = {c: np.mean([r.lat[chunks.index(c)] / r.lat.min() - 1 for r in tab.itertuples()]) for c in chunks}
    best_static = min(regs, key=regs.get)
    sel["best_static"] = {(r.d, r.L, r.batch): best_static for r in tab.itertuples()}
    widths = set(tab["d"])
    unsh = {}
    for model, full in MODEL_WIDTHS.items():
        for tp in E12.TP_DEGREES[1:]:
            d = full // tp
            if full % tp == 0 and d in widths and full in widths:
                unsh.update(E12.table_choice(tab[tab["d"].isin([d, full])],
                                             lambda _d, full=full, d=d: full if _d == d else None))
    sel["unsharded_table"] = unsh
    notes["best_static_chunk"] = int(best_static)
    return sel, notes


def round_cube(raw: pd.DataFrame, fam: str, chunks: list[int], keys: list) -> tuple[dict, dict]:
    """Per shape: (rounds x chunks) block means and within-block standard errors."""
    g = raw[raw["family"] == fam]
    piv = g.pivot_table(index=["d", "L", "batch", "round"], columns="chunk", values="mean_ms")[chunks]
    pse = g.pivot_table(index=["d", "L", "batch", "round"], columns="chunk", values="se_ms")[chunks]
    R = {k: v.to_numpy() for k, v in piv.groupby(level=[0, 1, 2])}
    S = {k: v.to_numpy() for k, v in pse.groupby(level=[0, 1, 2])}
    return {k: R[k] for k in keys}, {k: S[k] for k in keys}


def load_raw(path: Path) -> pd.DataFrame:
    """E12 rows with the within-block standard error (``e12_analysis.load`` plus std and n)."""
    recs = []
    for r in read_jsonl(path, kind="timing"):
        for c, cell in r["cells"].items():
            recs.append({"family": r["family"], "d": r["d"], "L": r["L"], "batch": r["batch"], "round": r["round"],
                         "chunk": int(c), "mean_ms": cell["mean_ms"],
                         "se_ms": cell["std_ms"] / np.sqrt(cell["n"])})
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- noise statistics
def variation(R: dict, S: dict, keys: list, chunks: list[int]) -> dict:
    rsd, rel, z, agree_all, agree_pair = [], [], [], [], []
    for k in keys:
        Rk, Sk = R[k], S[k]
        N = Rk.shape[0]
        rsd.append(Rk.std(0, ddof=1) / Rk.mean(0))
        for a in range(N):
            for b in range(a + 1, N):
                rel.append(np.abs(Rk[a] - Rk[b]) / (0.5 * (Rk[a] + Rk[b])))
                z.append((Rk[a] - Rk[b]) / np.sqrt(Sk[a] ** 2 + Sk[b] ** 2))
        am = Rk.argmin(1)
        agree_all.append(bool((am == am[0]).all()))
        agree_pair.extend(am[a] == am[b] for a in range(N) for b in range(a + 1, N))
    rsd, rel, z = np.concatenate(rsd), np.concatenate(rel), np.concatenate(z)
    return {"n_shapes": len(keys), "n_shape_chunk_cells": int(len(rsd)),
            "between_round_rel_sd_pct": {"median": float(100 * np.median(rsd)), "p90": float(100 * np.percentile(rsd, 90))},
            "abs_rel_diff_two_rounds_pct": {"median": float(100 * np.median(rel)), "p90": float(100 * np.percentile(rel, 90))},
            "share_round_pairs_abs_z_gt_1p96": float(np.mean(np.abs(z) > 1.96)),
            "argmin_agree_all_rounds_share": float(np.mean(agree_all)),
            "argmin_agree_pairwise_share": float(np.mean(agree_pair))}


def anova(R: dict, keys: list) -> dict:
    """Two-way chunk x round analysis of log latency per shape (no replication: residual = interaction)."""
    ps, ratio, gap_z = {}, [], []
    for k in keys:
        Y = np.log(R[k])                       # rounds x chunks
        N, Cn = Y.shape
        gm = Y.mean()
        ss_c = N * ((Y.mean(0) - gm) ** 2).sum()
        ss_r = Cn * ((Y.mean(1) - gm) ** 2).sum()
        ss_e = ((Y - gm) ** 2).sum() - ss_c - ss_r
        df_c, df_e = Cn - 1, (Cn - 1) * (N - 1)
        ms_c, ms_e = ss_c / df_c, max(ss_e, 1e-300) / df_e
        F = ms_c / ms_e
        ps[str(k)] = float(f_dist.sf(F, df_c, df_e))
        ratio.append(ms_c / ms_e)
        mu = R[k].mean(0)
        o = np.argsort(mu)
        d = np.log(R[k][:, o[1]]) - np.log(R[k][:, o[0]])        # second best minus best, per round
        sd = d.std(ddof=1)
        gap_z.append(d.mean() / (sd / np.sqrt(N)) if sd > 0 else np.inf)
    adj = C.bh(ps)
    gap_z = np.array(gap_z)
    return {"n_shapes": len(keys), "share_chunk_effect_significant_bh05": float(np.mean([adj[k] < 0.05 for k in adj])),
            "median_ms_chunk_over_ms_residual": float(np.median(ratio)),
            "top2_gap": {"definition": "log latency of the second-best minus the best chunk (three-round means), per round; "
                                       "t = mean / (between-round SD / sqrt(3))",
                         "share_t_gt_4p30": float(np.mean(gap_z > 4.303)),
                         "median_t": float(np.median(gap_z[np.isfinite(gap_z)])) if np.isfinite(gap_z).any() else None,
                         "note": "4.30 is the two-sided 5% critical value of t with 2 degrees of freedom: share of shapes "
                                 "whose best chunk is resolved from the runner-up by the three rounds"}}


# --------------------------------------------------------------------------- scoring
def cross_round(Rk: np.ndarray, j: int | None) -> float:
    """Mean over held-out rounds of the regret of chunk j (None: retuned on the other rounds)."""
    regs = []
    for r in range(Rk.shape[0]):
        jj = int(np.delete(Rk, r, 0).mean(0).argmin()) if j is None else j
        regs.append(Rk[r, jj] / Rk[r].min() - 1)
    return float(np.mean(regs))


def cross_round_min(Rk: np.ndarray, j: int | None) -> float:
    """As ``cross_round`` for a table retuned on the elementwise minimum of the other rounds (robust to upward spikes)."""
    regs = []
    for r in range(Rk.shape[0]):
        jj = int(np.delete(Rk, r, 0).min(0).argmin())
        regs.append(Rk[r, jj] / Rk[r].min() - 1)
    return float(np.mean(regs))


def spikes(R: dict, keys: list) -> dict:
    """Shape of the round-to-round noise: one-sided spikes or level shifts of whole rounds."""
    hi_dev, lo_dev, round_shift = [], [], []
    for k in keys:
        Rk = R[k]
        med = np.median(Rk, 0)
        dev = Rk / med - 1                                   # rounds x chunks, relative to the per-chunk median
        hi_dev.append(dev.max(0))
        lo_dev.append(dev.min(0))
        lr = np.log(Rk)
        round_shift.append(np.abs(lr.mean(1) - lr.mean()).max())
    hi, lo, rs = np.concatenate(hi_dev), np.concatenate(lo_dev), np.array(round_shift)
    return {"definition": "per (shape, chunk): largest and smallest deviation of a round mean from the median of the "
                          "three rounds; per shape: largest |deviation| of a round's mean log latency over chunks "
                          "(a whole-round level shift)",
            "share_cells_one_round_above_median_by_10pct": float(np.mean(hi > 0.10)),
            "share_cells_one_round_below_median_by_10pct": float(np.mean(lo < -0.10)),
            "share_cells_one_round_above_median_by_25pct": float(np.mean(hi > 0.25)),
            "share_cells_one_round_below_median_by_25pct": float(np.mean(lo < -0.25)),
            "round_level_shift_pct": {"median": float(100 * np.median(np.expm1(rs))),
                                      "p90": float(100 * np.percentile(np.expm1(rs), 90))}}


def mean_oracle(Rk: np.ndarray, j: int | None) -> float:
    mu = Rk.mean(0)
    if j is not None:
        return float(mu[j] / mu.min() - 1)
    return float(np.mean([mu[int(np.delete(Rk, r, 0).mean(0).argmin())] / mu.min() - 1 for r in range(Rk.shape[0])]))


def scored(R: dict, keys: list, chunks: list[int], sel: dict, how) -> dict:
    out = {}
    for name, ch in list(sel.items()) + [("retuned_at_shard_width", None)]:
        if ch is not None and not all(k in ch for k in keys):
            continue
        out[name] = np.array([how(R[k], None if ch is None else chunks.index(ch[k])) for k in keys])
    return out


def _spearman(a, b) -> dict:
    from scipy.stats import spearmanr
    r = spearmanr(a, b)
    return {"rho": float(r.statistic), "p": float(r.pvalue)}


def pooled(regs: dict, bidx: np.ndarray) -> dict:
    out = {}
    for n, v in regs.items():
        out[n] = {"mean_regret_pct": float(100 * v.mean()), "ci95_shape_bootstrap": _ci(100 * v[bidx].mean(1)),
                  "p95_regret_pct": float(100 * np.percentile(v, 95)), "max_regret_pct": float(100 * v.max())}
    return out


def paired(regs: dict, bidx: np.ndarray, base: str = "cost_model") -> dict:
    if base not in regs:
        return {}
    out = {}
    for n, v in regs.items():
        if n == base:
            continue
        d = 100 * (regs[base] - v)
        ci = _ci(d[bidx].mean(1))
        out[f"{base} - {n}"] = {"diff_pp": float(d.mean()), "ci95_pp": ci, "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}
    return out


# --------------------------------------------------------------------------- one (host, family)
def family(host: str, fam: str, raw: pd.DataFrame, raw_e12: pd.DataFrame, stored: dict, n_boot: int, in_dir: Path,
           checks: list) -> dict:
    chunks = list(MAMBA1_CHUNKS) if fam == "triton" else list(REF_CHUNKS)
    tab = E12.shape_table(raw_e12[raw_e12["family"] == fam], chunks).reset_index(drop=True)
    sel, notes = selectors(tab, chunks, host, fam, in_dir)
    widths = set(tab["d"])
    shards = {d for d, lst in E12.shard_map(widths).items() if any(tp > 1 for _, tp in lst)}
    pooled_tab = tab[tab["d"].isin(shards)].reset_index(drop=True)
    keys_all = list(zip(tab["d"], tab["L"], tab["batch"]))
    keys_p = list(zip(pooled_tab["d"], pooled_tab["L"], pooled_tab["batch"]))
    R, S = round_cube(raw, fam, chunks, keys_all)
    n = len(keys_p)
    # ---- 1. reproduction (same generator and replicate order as taco_report_numbers.tp_family)
    rng = TRN.make_rng("tp", host, fam)
    bidx_trn = rng.integers(0, n, size=(n_boot, n))
    rep = {"round_agreement_all_shapes_e12_definition": float(tab["rounds_agree"].mean()),
           "round_agreement_definition": "share of shapes whose per-round argmin is the same chunk in all three rounds "
                                         "(e12_analysis.shape_table rounds_agree), over all shapes of the family",
           "pooled_shards_mean_regret_vs_3round_mean_oracle_pct": {},
           "retuned_cross_round_pct": None, "paired": {}}
    mo = {nm: np.array([R[k].mean(0)[chunks.index(ch[k])] / R[k].mean(0).min() - 1 for k in keys_p])
          for nm, ch in sel.items() if all(k in ch for k in keys_p)}
    ret = np.array([cross_round(R[k], None) for k in keys_p])
    for nm, v in mo.items():
        rep["pooled_shards_mean_regret_vs_3round_mean_oracle_pct"][nm] = float(100 * v.mean())
    rep["retuned_cross_round_pct"] = float(100 * ret.mean())
    rep["retuned_cross_round_ci95"] = _ci(100 * ret[bidx_trn].mean(1))
    if "cost_model" in mo:
        d = 100 * (mo["cost_model"] - ret)
        rep["paired"]["cost_model - retuned_at_shard_width_cross_round (stored mixed scoring)"] = {
            "diff_pp": float(d.mean()), "ci95_pp": _ci(d[bidx_trn].mean(1))}
    # checks against the stored analysis
    sp = stored.get("pooled_shards", {})
    for nm, v in rep["pooled_shards_mean_regret_vs_3round_mean_oracle_pct"].items():
        if nm in sp:
            checks.append({"check": f"{host}.{fam}.{nm}.mean", "computed": v, "stored": sp[nm]["mean_regret_pct"],
                           "match": abs(v - sp[nm]["mean_regret_pct"]) < 1e-9})
    if "retuned_at_shard_width_cross_round" in sp:
        s = sp["retuned_at_shard_width_cross_round"]
        checks.append({"check": f"{host}.{fam}.retuned.mean", "computed": rep["retuned_cross_round_pct"],
                       "stored": s["mean_regret_pct"], "match": abs(rep["retuned_cross_round_pct"] - s["mean_regret_pct"]) < 1e-9})
        checks.append({"check": f"{host}.{fam}.retuned.ci", "computed": rep["retuned_cross_round_ci95"],
                       "stored": s["mean_regret_pct_ci95_shape_bootstrap"],
                       "match": bool(np.allclose(rep["retuned_cross_round_ci95"], s["mean_regret_pct_ci95_shape_bootstrap"]))})
    ps = stored.get("paired_shape_bootstrap", {}).get("cost_model - retuned_at_shard_width_cross_round")
    if ps and rep["paired"]:
        mine = next(iter(rep["paired"].values()))
        checks.append({"check": f"{host}.{fam}.paired_cm_minus_retuned", "computed": [mine["diff_pp"]] + mine["ci95_pp"],
                       "stored": [ps["diff_mean_regret_pp"]] + ps["ci95_pp"],
                       "match": bool(np.allclose([mine["diff_pp"]] + mine["ci95_pp"], [ps["diff_mean_regret_pp"]] + ps["ci95_pp"]))})
    # ---- 2. round-to-round variation
    var = {"shard_shapes": variation(R, S, keys_p, chunks), "all_shapes": variation(R, S, keys_all, chunks)}
    # ---- 3. chunk effect against noise
    av = {"shard_shapes": anova(R, keys_p), "all_shapes": anova(R, keys_all)}
    rng4 = C.make_rng("m5b", host, fam)
    bidx = rng4.integers(0, n, size=(n_boot, n))
    cr = scored(R, keys_p, chunks, sel, cross_round)
    floor = cr["retuned_at_shard_width"]
    shard_vs_noise = {}
    for nm in ("unsharded_table", "calibration_width_table", "best_static", "cost_model"):
        if nm not in cr:
            continue
        v = cr[nm]
        exc = v - floor
        shard_vs_noise[nm] = {"n_shapes": n, "n_regret_above_own_noise_floor": int((v > floor + 1e-12).sum()),
                              "n_regret_above_twice_noise_floor": int((v > 2 * floor + 1e-12).sum()),
                              "n_regret_above_floor_plus_5pp": int((v > floor + 0.05).sum()),
                              "mean_excess_over_floor_pp": float(100 * exc.mean()),
                              "mean_excess_over_floor_ci95_pp": _ci(100 * exc[bidx].mean(1))}
    rmin = np.array([cross_round_min(R[k], None) for k in keys_p])
    d_rmin = 100 * (cr["cost_model"] - rmin) if "cost_model" in cr else None
    noise_shape = {"shard_shapes": spikes(R, keys_p), "all_shapes": spikes(R, keys_all),
                   "retuned_on_min_of_other_rounds_cross_round_pct": float(100 * rmin.mean()),
                   "retuned_on_min_of_other_rounds_ci95": _ci(100 * rmin[bidx].mean(1)),
                   "cost_model_minus_retuned_on_min": None if d_rmin is None else
                   {"diff_pp": float(d_rmin.mean()), "ci95_pp": _ci(d_rmin[bidx].mean(1))}}
    if "cost_model" in cr:
        dd = cr["cost_model"] - floor
        noise_shape["cost_model_vs_retuned_per_shape"] = {
            "n_model_better": int((dd < -1e-12).sum()), "n_equal": int((np.abs(dd) <= 1e-12).sum()),
            "n_model_worse": int((dd > 1e-12).sum()),
            "spearman_model_advantage_vs_shape_noise": _spearman(-dd, np.array([np.median(R[k].std(0, ddof=1) / R[k].mean(0))
                                                                                for k in keys_p]))}
    noise = {"anova": av, "noise_shape": noise_shape, "noise_floor_definition": "per shard shape: table retuned at the shape, chosen on two rounds "
                                                    "and scored on the third against that round's oracle (cross_round)",
             "noise_floor_pooled_pct": float(100 * floor.mean()), "noise_floor_pooled_ci95": _ci(100 * floor[bidx].mean(1)),
             "noise_floor_median_shape_pct": float(100 * np.median(floor)),
             "noise_floor_shape_distribution": {"p75_pct": float(100 * np.percentile(floor, 75)),
                                                "p90_pct": float(100 * np.percentile(floor, 90)),
                                                "max_pct": float(100 * floor.max()),
                                                "share_shapes_above_2pct": float(np.mean(floor > 0.02)),
                                                "share_shapes_above_10pct": float(np.mean(floor > 0.10))},
             "selector_regret_vs_own_noise_floor": shard_vs_noise}
    # ---- 4. one scoring for every selector
    mo4 = scored(R, keys_p, chunks, sel, mean_oracle)
    sc = {"cross_round": {"definition": "every selector scored on each held-out round against that round's oracle; "
                                        "retuned table chosen on the other two rounds; mean over the three rounds",
                          "pooled": pooled(cr, bidx), "paired_cost_model_minus": paired(cr, bidx)},
          "mean_oracle_loro": {"definition": "every selector scored against the three-round-mean oracle; retuned table "
                                             "chosen on two rounds (leave one round out), mean over the three choices "
                                             "(optimistic for the retuned table)",
                               "pooled": pooled(mo4, bidx), "paired_cost_model_minus": paired(mo4, bidx)}}
    # unsharded table and retuned paired on cross-round scoring (the shard effect net of noise)
    if "unsharded_table" in cr:
        d = 100 * (cr["unsharded_table"] - floor)
        sc["cross_round"]["paired_unsharded_minus_retuned"] = {"diff_pp": float(d.mean()), "ci95_pp": _ci(d[bidx].mean(1))}
        d = 100 * (cr["calibration_width_table"] - floor)
        sc["cross_round"]["paired_calibration_width_minus_retuned"] = {"diff_pp": float(d.mean()), "ci95_pp": _ci(d[bidx].mean(1))}
    return {"gpu": GPU[host], "family": fam, "chunks": chunks, "n_shapes": int(len(tab)), "n_shard_shapes_pooled": int(n),
            "notes": notes, "reproduction": rep, "round_variation": var, "chunk_effect_vs_noise": noise, "scoring": sc}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    stored_trn = (C.load_json(args.in_dir / "taco_report_numbers.json") or {}).get("M7a_tensor_parallel", {}).get("hosts", {})
    out = {"item": "M5b", "meta": C.meta("taco_r2_stats_m5b_tp_noise", args), "definitions": __doc__, "hosts": {}, "checks": []}
    for host in HOSTS:
        path = args.in_dir / f"e12_tp_widths_{host}.jsonl"
        raw = load_raw(path)
        raw_e12 = E12.load(path)
        out["hosts"][host] = {}
        for fam in sorted(raw["family"].unique()):
            out["hosts"][host][fam] = family(host, fam, raw, raw_e12, stored_trn.get(host, {}).get(fam, {}), args.n_boot,
                                             args.in_dir, out["checks"])
    out["checks_summary"] = {"n": len(out["checks"]), "n_match": sum(c["match"] for c in out["checks"])}
    path = C.write_result("m5b", out, args.in_dir)
    print_summary(out)
    print(f"wrote {path}")


def print_summary(out: dict) -> None:
    for host, fams in out["hosts"].items():
        for fam, r in fams.items():
            rep = r["reproduction"]
            print(f"== {host} {fam}: {r['n_shard_shapes_pooled']} shard shapes; round agreement (all shapes) "
                  f"{100 * rep['round_agreement_all_shapes_e12_definition']:.1f}%")
            print("   stored scoring:", {k: round(v, 2) for k, v in rep["pooled_shards_mean_regret_vs_3round_mean_oracle_pct"].items()},
                  "retuned", round(rep["retuned_cross_round_pct"], 2), rep["paired"])
            v = r["round_variation"]["shard_shapes"]
            print(f"   variation: rel SD med {v['between_round_rel_sd_pct']['median']:.2f}% p90 {v['between_round_rel_sd_pct']['p90']:.2f}%;"
                  f" |diff| med {v['abs_rel_diff_two_rounds_pct']['median']:.2f}% p90 {v['abs_rel_diff_two_rounds_pct']['p90']:.2f}%;"
                  f" |z|>1.96 {100 * v['share_round_pairs_abs_z_gt_1p96']:.0f}%; argmin all/pair "
                  f"{100 * v['argmin_agree_all_rounds_share']:.0f}/{100 * v['argmin_agree_pairwise_share']:.0f}%")
            a = r["chunk_effect_vs_noise"]["anova"]["shard_shapes"]
            print(f"   anova: sig {100 * a['share_chunk_effect_significant_bh05']:.0f}%, MSc/MSe med {a['median_ms_chunk_over_ms_residual']:.0f},"
                  f" top-2 resolved {100 * a['top2_gap']['share_t_gt_4p30']:.0f}%; noise floor {r['chunk_effect_vs_noise']['noise_floor_pooled_pct']:.2f}%")
            for how in ("cross_round", "mean_oracle_loro"):
                s = r["scoring"][how]
                print(f"   {how}:", {k: round(v['mean_regret_pct'], 2) for k, v in s["pooled"].items()})
                for k, v in s["paired_cost_model_minus"].items():
                    print(f"      {k}: {v['diff_pp']:+.2f} [{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}]")
    print("checks:", out["checks_summary"])


if __name__ == "__main__":
    main()
