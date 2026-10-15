"""Reference-scan cost model with per-instantiation compile facts (taco-r1, review item M2b).

The published reference-scan instance of the cost model (``cost_model.ref_predict``) assumes 168 registers
per thread for every instantiation and no shared-memory limit. The compile facts read from the timed binary
(``refscan_facts.py`` -> ``refscan_compile_facts_sm86.json``) give, per instantiation (threads, items per
thread, even length), the registers per thread and the dynamic shared memory per block. This script refits
the unchanged model form with those facts substituted into the occupancy term:

    blocks per SM k = min(max blocks, max warps / warps, registers per SM / (ceil(regs / 8) * 8 * threads),
                          shared memory per SM / (dynamic smem + 1 KB reserved per block))

with the instantiation chosen per call as the kernel's dispatch does it (fp16 inputs, fp32 weights, variable
B and C, with z, even length iff L is a multiple of the chunk). Two fits, as in the paper:

  published   twelve tuning keys of the reference-scan sweep, evaluated on its 2360 captured calls
              (stock pairing; ``cost_model.reference_scan_model``);
  CV          the fixed-thread session (``e23_fixed_threads_analysis``): grouped five-fold cross-validation
              over the prompt folds on the captured calls, for the stock pairing and for the 32-thread family.

Each fit is run without facts (the published constant, reproducing the stored numbers) and with facts.
A check asserts that the predictor with constant facts equals ``cost_model.ref_predict``. CPU only.

Output: ``taco_m2b_refscan_facts.json`` in the results directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.experiments.tpds import e23_fixed_threads_analysis as E23
from src.experiments.tpds.analysis import latency_matrix
from src.experiments.tpds.common import RESULTS_DIR, utc_now, write_json
from src.experiments.tpds.cost_model import (D_STATE, HARDWARE, REF_PARAMS, REF_REGS, REF_THREADS, e2_eval,
                                             fit_quality, ref_predict)

CHUNKS = (128, 256, 512, 1024, 2048)
SMEM_PER_SM = {86: 100 * 1024}       # maximum shared memory per SM (sm_86)
SMEM_RESERVED_PER_BLOCK = 1024       # reserved by the CUDA runtime per resident block (sm_80 and later)
LO, HI = [0, 0, 0, 0, 0, 0, 1, 1], [1, 0.1, 0.1, 0.01, 0.01, 0.1, 64, 256]
STARTS = ([0.01, 1e-3, 1e-4, 1e-5, 1e-5, 1e-4, 8, 16], [0.02, 1e-4, 1e-5, 1e-6, 1e-6, 1e-5, 4, 8],
          [0.005, 1e-3, 1e-3, 1e-4, 1e-4, 1e-3, 16, 32])


class Facts:
    """Per-instantiation registers and dynamic shared memory, looked up per (threads, chunk, even length)."""

    def __init__(self, path: Path | None, arch: int = 86):
        self.table, self.smem_per_sm = {}, SMEM_PER_SM.get(arch)
        if path is None:
            return
        for f in json.loads(path.read_text())["instantiations"]:
            if (f["input_t"].endswith("Half") and f["weight_t"] == "float" and f["variable_b"] and f["variable_c"]
                    and f["has_z"] and f["rows"] == 1):
                self.table[(f["threads"], f["chunk"], bool(f["even_len"]))] = f

    def lookup(self, thr: np.ndarray, C: np.ndarray, L: np.ndarray):
        if not self.table:  # constant facts of the published model
            return np.full(len(C), float(REF_REGS)), None
        even = (L % C) == 0
        regs = np.array([self.table[(int(t), int(c), bool(e))]["registers"] for t, c, e in zip(thr, C, even)], float)
        smem = np.array([self.table[(int(t), int(c), bool(e))]["dynamic_smem"] for t, c, e in zip(thr, C, even)],
                        float)
        return regs, smem

    def rows(self, threads: dict) -> list[dict]:
        out = []
        for c in CHUNKS:
            for e in (False, True):
                f = self.table.get((threads[c], c, e))
                if f:
                    out.append({"chunk": c, "threads": threads[c], "items": c // threads[c], "even_len": e,
                                "registers": f["registers"], "stack": f["stack"], "dynamic_smem": f["dynamic_smem"]})
        return out


def predict(th, L, B, d, C, hw, threads, facts: Facts):
    """``cost_model.ref_predict`` with the registers and shared memory of each call's instantiation."""
    t0, tf, tm, te, ts, tx, W0, W1 = th
    thr = np.vectorize(threads.get)(C).astype(float)
    items, warps = C / thr, thr / 32
    regs, smem = facts.lookup(thr, C, L)
    alloc = np.ceil(regs / 8.0) * 8 * thr
    limits = [np.full(len(C), hw["max_blocks_per_sm"]), np.floor(hw["max_warps_per_sm"] / warps),
              np.floor(hw["regs_per_sm"] / alloc)]
    if smem is not None and facts.smem_per_sm:
        limits.append(np.floor(facts.smem_per_sm / (smem + SMEM_RESERVED_PER_BLOCK)))
    k = np.maximum(1, np.minimum.reduce(limits))
    P = B * d
    waves, m = np.ceil(P / (hw["n_sm"] * k)), np.minimum(k, np.ceil(P / hw["n_sm"]))
    W = m * warps
    A = tf + D_STATE * tm
    I = D_STATE * (te * items + ts * np.log2(np.minimum(thr, 32)) + tx * (warps > 1) * np.log2(warps + 1))
    return t0 + waves * np.ceil(L / C) * (A * np.minimum(1.0, W1 / W) + I * np.maximum(1.0, W / W0))


def long_form(df: pd.DataFrame):
    n = len(df)
    return (np.tile(df["L"].to_numpy(float), len(CHUNKS)), np.tile(df["batch"].to_numpy(float), len(CHUNKS)),
            np.tile(df["width"].to_numpy(float), len(CHUNKS)), np.repeat(np.array(CHUNKS, float), n))


def fit(train: pd.DataFrame, hw, threads, facts) -> tuple[np.ndarray, dict]:
    L, B, d, C = long_form(train)
    y = np.log(np.concatenate([train[f"lat_c{c}_mean"].to_numpy() for c in CHUNKS]))
    best = None
    for x0 in STARTS:
        sol = least_squares(lambda th: np.log(predict(th, L, B, d, C, hw, threads, facts)) - y, x0, bounds=(LO, HI),
                            loss="soft_l1", f_scale=0.1, x_scale="jac", max_nfev=20000)
        if best is None or sol.cost < best.cost:
            best = sol
    return best.x, fit_quality(predict(best.x, L, B, d, C, hw, threads, facts), np.exp(y))


def matrix(th, df, hw, threads, facts) -> np.ndarray:
    L, B, d, C = long_form(df)
    return predict(th, L, B, d, C, hw, threads, facts).reshape(len(CHUNKS), len(df)).T


def summary(st: dict) -> dict:
    keep = ("mean_regret_pct", "p95_regret_pct", "max_regret_pct", "tie_aware_accuracy", "accuracy")
    out = {k: st[k] for k in keep if k in st}
    if "curve_fit" in st:
        out["r2_log"] = st["curve_fit"].get("r2_log")
    return out


def published_fit(in_dir: Path, hw, facts) -> dict:
    df = pd.read_csv(in_dir / "e8_refscan_vgpu3090.csv")
    keys, calls = df[df["regimes"] == "e4_key"], df[df["regimes"] != "e4_key"].reset_index(drop=True)
    th, q = fit(keys, hw, REF_THREADS, facts)
    P = matrix(th, calls, hw, REF_THREADS, facts)
    return {"params": dict(zip(REF_PARAMS, map(float, th))), "key_fit": q,
            "calls": summary(e2_eval(calls, list(CHUNKS), P.argmin(1), P)), "n_calls": int(len(calls))}


def cv_fit(v: pd.DataFrame, hw, threads, facts) -> dict:
    folds = v["fold"].to_numpy()
    P = np.full((len(v), len(CHUNKS)), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        th, _ = fit(v[tr], hw, threads, facts)
        P[te] = matrix(th, v[te], hw, threads, facts)
    st = e2_eval(v, list(CHUNKS), np.nanargmin(P, 1), P)
    lat = latency_matrix(v, list(CHUNKS))
    # where the fit puts the long-chunk 32-thread instantiations relative to the measured order
    pred_best = np.array(CHUNKS)[np.nanargmin(P, 1)]
    return {**summary(st), "predicted_choice": {int(c): int((pred_best == c).sum()) for c in CHUNKS},
            "measured_oracle": {int(c): int((np.array(CHUNKS)[lat.argmin(1)] == c).sum()) for c in CHUNKS}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--facts", type=Path, default=None, help="default: refscan_compile_facts_sm86.json in --in-dir")
    p.add_argument("--e23-tag", default="vgpu3090_r4t32")
    args = p.parse_args(argv)
    hw = HARDWARE["vgpu3090"]
    facts = Facts(args.facts or args.in_dir / "refscan_compile_facts_sm86.json")
    const = Facts(None)

    # the predictor with constant facts is the published one
    rng = np.random.default_rng(0)
    th = np.array(STARTS[0], float)
    L = rng.integers(16, 8192, 200).astype(float)
    B, d, C = rng.choice([1, 4, 16], 200).astype(float), np.full(200, 1024.0), rng.choice(CHUNKS, 200).astype(float)
    assert np.allclose(predict(th, L, B, d, C, hw, REF_THREADS, const), ref_predict(th, L, B, d, C, hw)), "form check"

    res = {"experiment": "taco-r1 M2b reference scan with compile facts", "created_utc": utc_now(),
           "facts_source": "registers and dynamic shared memory per instantiation of the timed binary (sm_86)",
           "smem_per_sm_bytes": SMEM_PER_SM[86], "smem_reserved_per_block_bytes": SMEM_RESERVED_PER_BLOCK,
           "constant_registers_of_published_model": REF_REGS,
           "facts_stock_pairing": facts.rows(REF_THREADS), "facts_t32": facts.rows(E23.T32),
           "published_fit": {"constant": published_fit(args.in_dir, hw, const),
                             "facts": published_fit(args.in_dir, hw, facts)}}
    e23 = pd.read_csv(args.in_dir / f"e8_refscan_{args.e23_tag}.csv")
    calls = e23[e23["regimes"] != "e4_key"].reset_index(drop=True)
    res["cv_fit"] = {}
    for name, threads in (("stock", REF_THREADS), ("t32", E23.T32)):
        v = E23.view(calls, name)
        res["cv_fit"][name] = {"constant": cv_fit(v, hw, threads, const), "facts": cv_fit(v, hw, threads, facts)}
    write_json(args.in_dir / "taco_m2b_refscan_facts.json", res)
    pf = res["published_fit"]
    print(f"published fit (12 keys -> {pf['facts']['n_calls']} calls): constant "
          f"{pf['constant']['calls']['mean_regret_pct']:.2f}% | facts {pf['facts']['calls']['mean_regret_pct']:.2f}%")
    for name, r in res["cv_fit"].items():
        print(f"CV {name:5s}: constant {r['constant']['mean_regret_pct']:.2f}% (tied {r['constant']['tie_aware_accuracy']:.0%},"
              f" R2 {r['constant'].get('r2_log', float('nan')):.3f}) | facts {r['facts']['mean_regret_pct']:.2f}% "
              f"(tied {r['facts']['tie_aware_accuracy']:.0%}, R2 {r['facts'].get('r2_log', float('nan')):.3f}); "
              f"choices {r['facts']['predicted_choice']} oracle {r['facts']['measured_oracle']}")


if __name__ == "__main__":
    main()
