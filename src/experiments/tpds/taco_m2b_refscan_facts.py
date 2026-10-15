"""Reference-scan cost model with and without per-instantiation compile facts (taco-r1, review item M2b).

The reference-scan instance of the cost model (``cost_model.ref_predict``) is fitted with the compile facts
of the timed binary (``refscan_facts.py`` -> ``refscan_compile_facts_sm86.json``): per instantiation
(threads, items per thread, even length) the registers per thread and the dynamic shared memory per block
enter the occupancy term

    blocks per SM k = min(max blocks, max warps / warps, registers per SM / (ceil(regs / 8) * 8 * threads),
                          shared memory per SM / (dynamic smem + 1 KB reserved per block)).

The constant-register variant (168 registers for every instantiation, no shared-memory limit) is the form
published before the facts were read. This script compares the two on the same calls:

  published   twelve tuning keys of the reference-scan sweep, evaluated on its 2360 captured calls
              (stock pairing; ``cost_model.reference_scan_model``);
  CV          the fixed-thread session (``e23_fixed_threads_analysis``): grouped five-fold cross-validation
              over the prompt folds on the captured calls, for the stock pairing and for the 32-thread family;

and adds two diagnostics:

  paired bootstrap  difference facts - constant in mean regret, resampling prompt groups (the call's prompt
                    set; 2000 draws), for each comparison;
  bias per chunk    median predicted / measured latency per chunk and length bucket, the predicted blocks per
                    SM of each instantiation, and how often each chunk is chosen against how often it is the
                    oracle, which shows where the two variants differ.

The constant variant must reproduce the stored numbers of the published model and of the fixed-thread
analysis (checked). CPU only. Output: ``taco_m2b_refscan_facts.json`` in the results directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.experiments.tpds import cost_model as CM
from src.experiments.tpds import e23_fixed_threads_analysis as E23
from src.experiments.tpds.analysis import latency_matrix
from src.experiments.tpds.common import RESULTS_DIR, utc_now, write_json

CHUNKS = (128, 256, 512, 1024, 2048)
LO, HI = [0, 0, 0, 0, 0, 0, 1, 1], [1, 0.1, 0.1, 0.01, 0.01, 0.1, 64, 256]
STARTS = ([0.01, 1e-3, 1e-4, 1e-5, 1e-5, 1e-4, 8, 16], [0.02, 1e-4, 1e-5, 1e-6, 1e-6, 1e-5, 4, 8],
          [0.005, 1e-3, 1e-3, 1e-4, 1e-4, 1e-3, 16, 32])
N_BOOT = 2000


def long_form(df: pd.DataFrame):
    n = len(df)
    return (np.tile(df["L"].to_numpy(float), len(CHUNKS)), np.tile(df["batch"].to_numpy(float), len(CHUNKS)),
            np.tile(df["width"].to_numpy(float), len(CHUNKS)), np.repeat(np.array(CHUNKS, float), n))


def fit(train: pd.DataFrame, hw, threads, facts) -> tuple[np.ndarray, dict]:
    L, B, d, C = long_form(train)
    y = np.log(np.concatenate([train[f"lat_c{c}_mean"].to_numpy() for c in CHUNKS]))
    best = None
    for x0 in STARTS:
        sol = least_squares(lambda th: np.log(CM.ref_predict(th, L, B, d, C, hw, threads, facts)) - y, x0,
                            bounds=(LO, HI), loss="soft_l1", f_scale=0.1, x_scale="jac", max_nfev=20000)
        if best is None or sol.cost < best.cost:
            best = sol
    return best.x, CM.fit_quality(CM.ref_predict(best.x, L, B, d, C, hw, threads, facts), np.exp(y))


def matrix(th, df, hw, threads, facts) -> np.ndarray:
    L, B, d, C = long_form(df)
    return CM.ref_predict(th, L, B, d, C, hw, threads, facts).reshape(len(CHUNKS), len(df)).T


def summary(df: pd.DataFrame, P: np.ndarray) -> dict:
    st = CM.e2_eval(df, list(CHUNKS), np.nanargmin(P, 1), P)
    out = {k: st[k] for k in ("mean_regret_pct", "p95_regret_pct", "max_regret_pct", "tie_aware_accuracy",
                              "accuracy") if k in st}
    out["r2_log"] = st.get("curve_fit", {}).get("r2_log")
    return out


def per_call_regret(df: pd.DataFrame, P: np.ndarray) -> np.ndarray:
    lat = latency_matrix(df, list(CHUNKS))
    return lat[np.arange(len(df)), np.nanargmin(P, 1)] / lat.min(1) - 1


def paired_bootstrap(df: pd.DataFrame, r_facts: np.ndarray, r_const: np.ndarray, seed: int = 0) -> dict:
    """Difference facts - constant in mean regret (points), resampling prompt groups with replacement."""
    groups = df["prompt_ids"].astype(str).to_numpy()
    uniq, inv = np.unique(groups, return_inverse=True)
    diff = r_facts - r_const
    s, n = np.bincount(inv, diff, len(uniq)), np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(float)
        draws[b] = 100 * (w @ s) / (w @ n)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"diff_pp": float(100 * diff.mean()), "ci95_pp": [float(lo), float(hi)],
            "share_draws_above_0": float((draws > 0).mean()), "n_groups": int(len(uniq)), "n_boot": N_BOOT}


def bias_per_chunk(df: pd.DataFrame, P: np.ndarray, hw, threads, facts) -> dict:
    lat = latency_matrix(df, list(CHUNKS))
    ratio = P / lat
    bucket = np.where(df["L"].to_numpy() >= 4096, "L>=4096", np.where(df["L"].to_numpy() >= 1024, "1024<=L<4096",
                                                                         "L<1024"))
    out = {}
    for j, c in enumerate(CHUNKS):
        rec = {"median_pred_over_meas": float(np.median(ratio[:, j]))}
        for b in ("L<1024", "1024<=L<4096", "L>=4096"):
            m = bucket == b
            if m.any():
                rec[f"median_pred_over_meas_{b}"] = float(np.median(ratio[m, j]))
        thr = np.full(2, float(threads[c]))
        Cc = np.full(2, float(c))
        Lc = np.array([c * 7 + 1, c * 8], float)  # one unaligned, one aligned length
        if facts is None:
            regs = np.full(2, float(CM.REF_REGS))
            k = np.minimum.reduce([np.full(2, hw["max_blocks_per_sm"]), np.floor(hw["max_warps_per_sm"] / (thr / 32)),
                                   np.floor(hw["regs_per_sm"] / (np.ceil(regs / 8) * 8 * thr))])
        else:
            regs, smem = facts.lookup(thr, Cc, Lc)
            k = np.minimum.reduce([np.full(2, hw["max_blocks_per_sm"]), np.floor(hw["max_warps_per_sm"] / (thr / 32)),
                                   np.floor(hw["regs_per_sm"] / (np.ceil(regs / 8) * 8 * thr)),
                                   np.floor(facts.smem_per_sm / (smem + CM.SMEM_RESERVED_PER_BLOCK))])
        rec.update(threads=int(threads[c]), registers_unaligned_aligned=[int(x) for x in regs],
                   blocks_per_sm_unaligned_aligned=[int(x) for x in k])
        out[int(c)] = rec
    chosen = np.array(CHUNKS)[np.nanargmin(P, 1)]
    oracle = np.array(CHUNKS)[lat.argmin(1)]
    for c in CHUNKS:
        out[int(c)].update(chosen=int((chosen == c).sum()), oracle=int((oracle == c).sum()))
    return out


def cv_predictions(v: pd.DataFrame, hw, threads, facts) -> np.ndarray:
    folds = v["fold"].to_numpy()
    P = np.full((len(v), len(CHUNKS)), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        th, _ = fit(v[tr], hw, threads, facts)
        P[te] = matrix(th, v[te], hw, threads, facts)
    return P


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--e23-tag", default="vgpu3090_r4t32")
    args = p.parse_args(argv)
    hw = CM.HARDWARE["vgpu3090"]
    facts = CM.ref_facts(args.in_dir)
    if facts is None:
        raise SystemExit(f"{CM.REF_FACTS_FILE} missing in {args.in_dir}")
    res = {"experiment": "taco-r1 M2b reference scan with compile facts", "created_utc": utc_now(),
           "facts_source": "registers and dynamic shared memory per instantiation of the timed binary (sm_86)",
           "smem_per_sm_bytes": facts.smem_per_sm, "smem_reserved_per_block_bytes": CM.SMEM_RESERVED_PER_BLOCK,
           "constant_registers": CM.REF_REGS,
           "facts_stock_pairing": facts.rows(CM.REF_THREADS), "facts_t32": facts.rows(E23.T32)}

    # published fit: twelve tuning keys -> 2360 calls (stock pairing)
    df = pd.read_csv(args.in_dir / "e8_refscan_vgpu3090.csv")
    keys, calls = df[df["regimes"] == "e4_key"], df[df["regimes"] != "e4_key"].reset_index(drop=True)
    pub, regs = {}, {}
    for name, fx in (("facts", facts), ("constant", None)):
        th, q = fit(keys, hw, CM.REF_THREADS, fx)
        P = matrix(th, calls, hw, CM.REF_THREADS, fx)
        pub[name] = {"params": dict(zip(CM.REF_PARAMS, map(float, th))), "key_fit": q, "calls": summary(calls, P),
                     "bias_per_chunk": bias_per_chunk(calls, P, hw, CM.REF_THREADS, fx)}
        regs[name] = per_call_regret(calls, P)
    stored = json.loads((args.in_dir / "cost_model_refscan_vgpu3090.json").read_text())
    st_const = stored.get("constant_registers", stored)["e8_calls"]["mean_regret_pct"]
    assert abs(pub["constant"]["calls"]["mean_regret_pct"] - st_const) < 1e-9, "constant variant != stored fit"
    pub["paired_bootstrap_facts_minus_constant"] = paired_bootstrap(calls, regs["facts"], regs["constant"])
    res["published_fit"] = pub

    # CV on the fixed-thread session, both families
    e23 = pd.read_csv(args.in_dir / f"e8_refscan_{args.e23_tag}.csv")
    ecalls = e23[e23["regimes"] != "e4_key"].reset_index(drop=True)
    stored23 = json.loads((args.in_dir / f"e23_fixed_threads_{args.e23_tag}.json").read_text())["views"]
    res["cv_fit"] = {}
    for view, threads in (("stock", CM.REF_THREADS), ("t32", E23.T32)):
        v = E23.view(ecalls, view)
        out, rr = {}, {}
        for name, fx in (("facts", facts), ("constant", None)):
            P = cv_predictions(v, hw, threads, fx)
            out[name] = {**summary(v, P), "bias_per_chunk": bias_per_chunk(v, P, hw, threads, fx)}
            rr[name] = per_call_regret(v, P)
        assert abs(out["constant"]["mean_regret_pct"] - stored23[view]["cost_model_cv"]["mean_regret_pct"]) < 1e-9, \
            f"constant variant != stored fixed-thread analysis ({view})"
        out["paired_bootstrap_facts_minus_constant"] = paired_bootstrap(v, rr["facts"], rr["constant"])
        res["cv_fit"][view] = out
    write_json(args.in_dir / "taco_m2b_refscan_facts.json", res)

    def line(tag, r):
        f, c, b = r["facts"], r["constant"], r["paired_bootstrap_facts_minus_constant"]
        mf, mc = (f["calls"] if "calls" in f else f), (c["calls"] if "calls" in c else c)
        print(f"{tag:22s} constant {mc['mean_regret_pct']:6.2f}% (tied {mc['tie_aware_accuracy']:.0%}) | facts "
              f"{mf['mean_regret_pct']:6.2f}% (tied {mf['tie_aware_accuracy']:.0%}) | diff {b['diff_pp']:+.2f} pp "
              f"[{b['ci95_pp'][0]:+.2f}, {b['ci95_pp'][1]:+.2f}]")
    line("published (12 keys)", pub)
    for view in ("stock", "t32"):
        line(f"CV {view}", res["cv_fit"][view])
    for name in ("constant", "facts"):
        bp = pub[name]["bias_per_chunk"]
        print(f"published {name:8s}: " + " | ".join(
            f"C{c}: pred/meas {bp[c]['median_pred_over_meas']:.2f} (L>=4096 {bp[c].get('median_pred_over_meas_L>=4096', float('nan')):.2f}),"
            f" k {bp[c]['blocks_per_sm_unaligned_aligned']}, chosen {bp[c]['chosen']} / oracle {bp[c]['oracle']}"
            for c in CHUNKS))


if __name__ == "__main__":
    main()
