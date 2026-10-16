"""taco-r2 item M5a: noise floors next to every regret, and the sensitivity of the ad hoc floors and thresholds.

CPU only, deterministic; reads stored result files and times nothing.

Why a noise floor
-----------------
Every regret of the manuscript is measured against the per-call oracle, the fastest of the timed chunks. With noisy
timings that minimum is biased low, so even a selector that always picks the truly fastest chunk shows a positive
regret. The noise floor of a data set estimates that regret from the data set's own repeated timings:

per-call floor     the chunk that is fastest on one set of timings is scored on an independent set of timings of
                   the same calls (regret against that set's own per-call oracle); this is the regret of a perfect
                   per-call selector caused by noise alone
per-shape floor    the exact-shape (L, batch) table chosen on one set of timings, scored on the other set: the regret
                   of a perfect shape-keyed table, i.e. noise plus the call-to-call variation within a shape that no
                   shape key can follow

Independent sets of timings, by availability:
cross-round        three-round sweeps: choose on the median (= mean) of two rounds, score on the third; each round
                   held out once, averaged per call (the "cross-round" estimate of the headroom table, S37)
cross-session      two sessions of the same calls and convention (RTX 3090 device time: single block vs three
                   rounds; launch-inclusive RTX PRO 6000 and RTX 3090: first session vs three-round session; 1.4B and
                   2.8B launch-inclusive: two sessions; CUDA graphs on the RTX 3090: single round vs three rounds);
                   choose on one, score on the other, both directions averaged
split-half         single-block tables: each 50-repeat mean m with standard deviation s is split into two half-means
                   m +- D, D ~ N(0, s^2 / n) (the exact conditional law used in S37), choose on one half, score on the
                   other, both ways; 100 draws. This captures within-block noise only and is therefore a LOWER bound
                   of the floor wherever block-level noise exists (it is the only estimate for the H800, the 1.4B and
                   2.8B device-time sweeps, the reference scan and the Mamba-2 / Gated DeltaNet single-block tables)

Every floor is reported as the mean per-call regret (%), the latency-weighted share of the oracle time (%) and
microseconds per call, next to the paper-view regrets of the data set's default / best static chunk and its held-out
exact-shape table (whose own regret bounds the oracle bias of the paper view from above). The cross-round floor is
scored against a single round, i.e. against a noisier oracle than the round medians used in the paper; the default's
regret scored the same way (single round) is given next to it so the two are on one scale.

Information tests above the floor (M5a, last sentence)
-------------------------------------------------------
For the 19 data sets of the information analysis the chunk effect (held-out best single chunk, mean per-call regret)
is compared with the per-shape floor of the same data set; a data set "resolves the chunk" when the effect exceeds
twice the floor and the floor-to-effect ratio is given for all.

Floor sensitivity
-----------------
launch floor       "twice the median batch-1 oracle latency at L <= 256" with the factor 1, 1.5, 2 (published) and 3:
                   for the device-time SSD 2.7B and Gated DeltaNet 1.4B calls (calls above the floor, how often chunk
                   256 / 64 is their oracle, its largest per-batch mean regret, held-out headroom above the floor,
                   Table 8 rows) and the eight CUDA-graph tables (floor, share above, default regret and LW above)
100 us threshold   factor 0.5, 1, 1.5, 2, 3: share of the captured 370M Mamba-1 calls whose launch-inclusive oracle
                   latency is below the threshold, and the median / 95th percentile of the relative difference between
                   launch-inclusive and device-time latency above it (per call at the oracle, and per (call, chunk))

Output: ``taco_r2_stats_m5a.json``.
Usage:  PYTHONPATH=. python3 -m src.experiments.tpds.taco_r2_stats_m5a_noise_floors [--draws 100]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.experiments.tpds import deployment_units as DU  # noqa: E402
from src.experiments.tpds import e22_graph_analysis as E22  # noqa: E402
from src.experiments.tpds import e3_predictors as E3  # noqa: E402
from src.experiments.tpds import e3b_information as E3B  # noqa: E402
from src.experiments.tpds import taco_r2_stats_common as C  # noqa: E402
from src.experiments.tpds import taco_report_numbers as TRN  # noqa: E402
from src.experiments.tpds.analysis import chunk_columns, cv_exact_headroom, latency_matrix, load_e2, round_median_view  # noqa: E402
from src.experiments.tpds.common import RESULTS_DIR  # noqa: E402

# data sets behind Tables 4-8: name -> (csv, round-median view, kind, GPU, convention, scan layers, tables)
DATASETS = {
    "m1_pro6000_device": ("e2_mamba1_pro6000_r4spin.csv", True, "mamba1", "RTX PRO 6000", "device time, 3 rounds", 48,
                          "Tables 4, 5, 8"),
    "m1_vgpu3090_device": ("e2_mamba1_vgpu3090_r4spin.csv", True, "mamba1", "RTX 3090", "device time, 3 rounds", 48,
                           "Tables 5, 8"),
    "m1_h800_device": ("e2_mamba1_h800_r3spin.csv", False, "mamba1", "H800", "device time, one block", 48, "Tables 5, 8"),
    "m1_vgpu3090_device_one_block": ("e2_mamba1_vgpu3090_r3spin.csv", False, "mamba1", "RTX 3090",
                                     "device time, one block (session of the a-priori row)", 48, "Table 5 (a-priori row)"),
    "m1_pro6000_launch": ("e2_mamba1_pro6000.csv", False, "mamba1", "RTX PRO 6000", "launch-inclusive, one block", 48,
                          "Supplement (launch-inclusive tables)"),
    "m1_vgpu3090_launch": ("e2_mamba1_vgpu3090.csv", False, "mamba1", "RTX 3090", "launch-inclusive, one block", 48,
                           "Supplement"),
    "m1_h800_launch": ("e2_mamba1_h800.csv", False, "mamba1", "H800", "launch-inclusive, one block", 48, "Supplement"),
    "m1_1p4b_device": ("e2_mamba1_vgpu3090_1p4b_r3.csv", False, "mamba1", "RTX 3090", "device time, one block", 48,
                       "Tables 6, 8"),
    "m1_2p8b_device": ("e2_mamba1_vgpu3090_2p8b_r3.csv", False, "mamba1", "RTX 3090", "device time, one block", 64,
                       "Tables 6, 8"),
    "m1_1p4b_launch": ("e2_mamba1_vgpu3090_1p4b_r3plain.csv", False, "mamba1", "RTX 3090",
                       "launch-inclusive, one block (session of the conventions table)", 48, "Supplement"),
    "m1_2p8b_launch": ("e2_mamba1_vgpu3090_2p8b_r3plain.csv", False, "mamba1", "RTX 3090",
                       "launch-inclusive, one block (session of the conventions table)", 64, "Supplement"),
    "ref_vgpu3090_launch": ("e8_refscan_vgpu3090.csv", False, "ref", "RTX 3090", "launch-inclusive, one block", 48,
                            "Tables 6, 8"),
    "m2_pro6000_launch": ("e2_mamba2_pro6000.csv", False, "mamba2", "RTX PRO 6000", "launch-inclusive, one block", 48,
                          "Tables 4, 8"),
    "m2_vgpu3090_launch": ("e2_mamba2_vgpu3090.csv", False, "mamba2", "RTX 3090", "launch-inclusive, one block", 48,
                           "Table 8"),
    "m2_2p7b_device": ("e2_mamba2_vgpu3090_2p7b_r3.csv", False, "mamba2", "RTX 3090", "device time, one block", 64,
                       "Table 8 (above the floor)"),
    "gdn_1p4b_device": ("e2_gdn_vgpu3090_1p4b_r3.csv", False, "gdn", "RTX 3090", "device time, one block", 12,
                        "Table 8 (above the floor)"),
}
# second sessions of the same calls and convention
SESSION_PAIRS = {
    "m1_vgpu3090_device": ("e2_mamba1_vgpu3090_r3spin.csv", False, "single-block device-time session"),
    "m1_vgpu3090_device_one_block": ("e2_mamba1_vgpu3090_r4spin.csv", True, "three-round device-time session"),
    "m1_pro6000_launch": ("e2_mamba1_pro6000_r4plain.csv", True, "second launch-inclusive session, three rounds"),
    "m1_vgpu3090_launch": ("e2_mamba1_vgpu3090_r4plain.csv", True, "second launch-inclusive session, three rounds"),
    "m1_1p4b_launch": ("e2_mamba1_vgpu3090_1p4b.csv", False,
                       "earlier launch-inclusive session (median within-block CV 5.9%, 90th percentile 94%)"),
    "m1_2p8b_launch": ("e2_mamba1_vgpu3090_2p8b.csv", False, "earlier launch-inclusive session"),
}
DEFAULT_CHUNK = {"mamba1": 256, "mamba2": 256, "gdn": 64, "ref": None}
FLOOR_FACTORS = (1.0, 1.5, 2.0, 3.0)
THRESH_FACTORS = (0.5, 1.0, 1.5, 2.0, 3.0)
EFFECT_OVER_FLOOR = 2.0
# e3b information data sets -> (csv, round-median view)
E3B_TAGS = {
    "gdn_vgpu3090_1p4b_r3": ("e2_gdn_vgpu3090_1p4b_r3.csv", False),
    "mamba1_h800": ("e2_mamba1_h800.csv", False), "mamba1_h800_r3spin": ("e2_mamba1_h800_r3spin.csv", False),
    "mamba1_pro6000": ("e2_mamba1_pro6000.csv", False),
    "mamba1_pro6000_r4spin_rmed": ("e2_mamba1_pro6000_r4spin.csv", True),
    "mamba1_vgpu3090": ("e2_mamba1_vgpu3090.csv", False),
    "mamba1_vgpu3090_1p4b": ("e2_mamba1_vgpu3090_1p4b.csv", False),
    "mamba1_vgpu3090_1p4b_r3": ("e2_mamba1_vgpu3090_1p4b_r3.csv", False),
    "mamba1_vgpu3090_1p4b_r3plain": ("e2_mamba1_vgpu3090_1p4b_r3plain.csv", False),
    "mamba1_vgpu3090_2p8b": ("e2_mamba1_vgpu3090_2p8b.csv", False),
    "mamba1_vgpu3090_2p8b_r3": ("e2_mamba1_vgpu3090_2p8b_r3.csv", False),
    "mamba1_vgpu3090_2p8b_r3plain": ("e2_mamba1_vgpu3090_2p8b_r3plain.csv", False),
    "mamba1_vgpu3090_r3spin": ("e2_mamba1_vgpu3090_r3spin.csv", False),
    "mamba1_vgpu3090_r4spin_rmed": ("e2_mamba1_vgpu3090_r4spin.csv", True),
    "mamba2_pro6000": ("e2_mamba2_pro6000.csv", False), "mamba2_vgpu3090": ("e2_mamba2_vgpu3090.csv", False),
    "mamba2_vgpu3090_2p7b_r3": ("e2_mamba2_vgpu3090_2p7b_r3.csv", False),
    "mamba2_vgpu3090_r3spin": ("e2_mamba2_vgpu3090_r3spin.csv", False),
    "refscan_vgpu3090": ("e8_refscan_vgpu3090.csv", False),
}


# --------------------------------------------------------------------------- loading
def load(in_dir: Path, csv: str, rmed: bool) -> pd.DataFrame:
    df = load_e2([in_dir / csv])
    if "regimes" in df and (df["regimes"] == "e4_key").any():
        df = df[df["regimes"] != "e4_key"]
    df = df.reset_index(drop=True)
    return round_median_view(df) if rmed else df


def matrices(df: pd.DataFrame) -> tuple[list[int], np.ndarray, np.ndarray]:
    chunks = chunk_columns(df)
    lat = latency_matrix(df, chunks)
    lat = np.where(np.isfinite(lat), lat, np.inf)
    std = df[[f"lat_c{c}_std" for c in chunks]].to_numpy(float)
    return chunks, lat, std


def shape_codes(df: pd.DataFrame) -> np.ndarray:
    return df.groupby(["L", "batch"], sort=True).ngroup().to_numpy()


# --------------------------------------------------------------------------- floors
def _score(sel_lat: np.ndarray, score_lat: np.ndarray, keys: np.ndarray) -> dict[str, np.ndarray]:
    """Per-call excess and regret of the per-call and per-shape choices made on ``sel_lat``, scored on ``score_lat``."""
    n = len(score_lat)
    orc = np.min(score_lat, 1)
    pc = np.argmin(np.where(np.isfinite(sel_lat), sel_lat, np.inf), 1)
    sums = np.zeros((keys.max() + 1, sel_lat.shape[1]))
    np.add.at(sums, keys, np.where(np.isfinite(sel_lat), sel_lat, 1e30))
    ps = sums.argmin(1)[keys]
    out = {}
    for name, p in (("per_call", pc), ("per_shape", ps)):
        ch = score_lat[np.arange(n), p]
        out[name] = (ch - orc, ch / orc - 1.0, orc)
    return out


def _acc(recs: list[dict[str, tuple]]) -> dict:
    """Average per call over several (selection, scoring) pairs, then summarize."""
    out = {}
    for name in ("per_call", "per_shape"):
        exc = np.mean([r[name][0] for r in recs], 0)
        reg = np.mean([r[name][1] for r in recs], 0)
        orc = np.mean([r[name][2] for r in recs], 0)
        out[name] = {"mean_regret_pct": float(100 * reg.mean()), "lw_share_of_oracle_pct": float(100 * exc.sum() / orc.sum()),
                     "us_per_call": float(1000 * exc.mean())}
    return out


def floor_split_half(lat: np.ndarray, std: np.ndarray, n_rep: np.ndarray, keys: np.ndarray, draws: int, rng,
                     sel: np.ndarray | None = None) -> dict:
    idx = np.arange(len(lat)) if sel is None else np.flatnonzero(sel)
    m, s = lat[idx], np.nan_to_num(std[idx], nan=0.0, posinf=0.0)
    se = s / np.sqrt(n_rep[idx])[:, None]
    k = pd.factorize(keys[idx])[0]
    per_draw = {"per_call": [], "per_shape": []}
    acc = []
    for _ in range(draws):
        D = rng.standard_normal(m.shape) * se
        A, B = m + D, m - D
        r = [_score(A, B, k), _score(B, A, k)]
        a = _acc(r)
        for name in per_draw:
            per_draw[name].append(a[name]["mean_regret_pct"])
        acc.extend(r)
    out = _acc(acc)
    for name in per_draw:
        out[name]["mean_regret_pct_draw_range95"] = C.pct_ci(np.array(per_draw[name]))
    out["method"] = f"split-half of each block (within-block noise only; lower bound), {draws} draws"
    return out


def floor_cross_round(R: np.ndarray, keys: np.ndarray, sel: np.ndarray | None = None, default_idx: int | None = None) -> dict:
    idx = np.arange(R.shape[1]) if sel is None else np.flatnonzero(sel)
    Rs = np.where(np.isfinite(R[:, idx]), R[:, idx], np.inf)
    k = pd.factorize(keys[idx])[0]
    recs, dflt = [], []
    for r in range(Rs.shape[0]):
        other = np.median(np.delete(Rs, r, 0), 0)
        recs.append(_score(other, Rs[r], k))
        if default_idx is not None:
            o = Rs[r].min(1)
            ch = Rs[r][:, default_idx]
            dflt.append((ch - o, ch / o - 1.0, o))
    out = _acc(recs)
    out["method"] = "cross-round: chosen on the median of the other rounds, scored on the held-out round (single-round oracle)"
    if dflt:
        exc, reg, orc = (np.mean([d[i] for d in dflt], 0) for i in range(3))
        out["default_scored_on_single_rounds"] = {"mean_regret_pct": float(100 * reg.mean()),
                                                  "lw_share_of_oracle_pct": float(100 * exc.sum() / orc.sum()),
                                                  "us_per_call": float(1000 * exc.mean())}
    return out


def floor_cross_session(dfA: pd.DataFrame, dfB: pd.DataFrame, sel_keys: set | None = None) -> dict:
    ca, la, _ = matrices(dfA)
    cb, lb, _ = matrices(dfB)
    common = [c for c in ca if c in cb]
    ka = dfA.set_index("key")
    kb = dfB.set_index("key")
    keys = [k for k in dfA["key"] if k in set(dfB["key"]) and (sel_keys is None or k in sel_keys)]
    A = np.where(np.isfinite(latency_matrix(ka.loc[keys], common)), latency_matrix(ka.loc[keys], common), np.inf)
    B = np.where(np.isfinite(latency_matrix(kb.loc[keys], common)), latency_matrix(kb.loc[keys], common), np.inf)
    shp = pd.DataFrame({"L": ka.loc[keys, "L"].to_numpy(), "batch": ka.loc[keys, "batch"].to_numpy()})
    k = shp.groupby(["L", "batch"], sort=True).ngroup().to_numpy()
    out = _acc([_score(A, B, k), _score(B, A, k)])
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = A / B
    ratio = ratio[np.isfinite(ratio)]
    out.update({"n_calls_matched": len(keys), "chunks": common,
                "method": "cross-session: chosen on one session, scored on the other, both directions averaged",
                "same_chunk_latency_ratio_p10_p50_p90": [float(np.percentile(ratio, q)) for q in (10, 50, 90)],
                "oracle_chunk_agreement": float((A.argmin(1) == B.argmin(1)).mean())})
    return out


# --------------------------------------------------------------------------- per data set
def policy_rows(df: pd.DataFrame, kind: str, chunks: list[int], lat: np.ndarray, layers: int, sel=None) -> dict:
    orc = lat.min(1)
    picks = {}
    d = DEFAULT_CHUNK[kind]
    if d is not None and d in chunks:
        picks["default"] = np.full(len(df), chunks.index(d))
    norm = np.minimum(lat / orc[:, None], 1e6)
    picks["best_single_chunk_in_sample"] = np.full(len(df), int(norm.sum(0).argmin()))
    b = pd.factorize(df["batch"], sort=True)[0]
    s = np.zeros((b.max() + 1, len(chunks)))
    np.add.at(s, b, norm)
    picks["best_static_per_batch_in_sample"] = s.argmin(1)[b]
    picks["exact_table_held_out"] = DU.cv_exact_pick(df, np.where(np.isfinite(lat), lat, 1e30))
    out = {}
    if kind == "ref" and "lat_stock_mean" in df:
        idx = np.arange(len(df)) if sel is None else np.flatnonzero(sel)
        st = df["lat_stock_mean"].to_numpy(float)[idx]
        o = orc[idx]
        out["stock_dispatch_separate_binary"] = {
            "n": int(len(idx)), "mean_regret_pct": float(100 * (st / o - 1).mean()),
            "lw_share_of_oracle_pct": float(100 * (st - o).sum() / o.sum()), "us_per_call": float(1000 * (st - o).mean()),
            "note": "stock binary timed separately; its oracle comes from the runtime-chunk build"}
        sc = df["stock_chunk"].to_numpy()
        picks["stock_chunk_runtime_build"] = np.array([chunks.index(int(c)) for c in sc])
    for n, p in picks.items():
        r = TRN.record(lat, orc, p, layers=layers, sel=sel, chunks=chunks)
        out[n] = {k: r[k] for k in ("n", "mean_regret_pct", "p95_regret_pct", "lw_share_of_oracle_pct", "us_per_call",
                                    "ms_per_forward_pass") if k in r}
        out[n]["chunk"] = r.get("chosen")
    return out


def analyze_dataset(name: str, args, rng) -> dict:
    csv, rmed, kind, gpu, conv, layers, tables = DATASETS[name]
    df = load(args.in_dir, csv, rmed)
    chunks, lat, std = matrices(df)
    keys = shape_codes(df)
    rec = {"source": csv, "view": "round medians" if rmed else "one block", "kind": kind, "gpu": gpu, "convention": conv,
           "tables": tables, "n_calls": int(len(df)), "chunks": chunks,
           "median_cv_pct": float(100 * np.nanmedian(std / np.where(np.isfinite(lat), lat, np.nan))),
           "policies_paper_view": policy_rows(df, kind, chunks, lat, layers), "floors": {}}
    dflt = DEFAULT_CHUNK[kind]
    didx = chunks.index(dflt) if dflt in chunks else int(np.minimum(lat / lat.min(1)[:, None], 1e6).sum(0).argmin())
    if rmed:
        R = C.round_matrix(df, chunks)
        rec["floors"]["cross_round"] = floor_cross_round(R, keys, default_idx=didx)
    else:
        n_rep = df["repeats"].to_numpy(float)
        rec["floors"]["split_half"] = floor_split_half(lat, std, n_rep, keys, args.draws, rng)
    if name in SESSION_PAIRS:
        csv2, rmed2, note = SESSION_PAIRS[name]
        dfB = load(args.in_dir, csv2, rmed2)
        rec["floors"]["cross_session"] = {**floor_cross_session(df, dfB), "other_session": csv2, "note": note}
    if kind in ("mamba2", "gdn") and conv.startswith("device"):
        small = (df["batch"].to_numpy() == 1) & (df["L"].to_numpy() <= 256)
        above = lat.min(1) > 2.0 * np.median(lat.min(1)[small])
        rec["above_floor_n"] = int(above.sum())
        rec["policies_paper_view_above_floor"] = policy_rows(df, kind, chunks, lat, layers, sel=above)
        n_rep = df["repeats"].to_numpy(float)
        rec["floors"]["split_half_above_floor"] = floor_split_half(lat, std, n_rep, keys, args.draws, rng, sel=above)
    return rec


def analyze_graph(name: str, args, rng) -> dict:
    t = C.graph_table(name, args.in_dir)
    keys = shape_codes(t.df)
    R = C.round_matrix(t.df, t.chunks)
    didx = t.default_idx()
    rec = {"source": t.csv, "gpu": C.GRAPH_TABLES[name][0], "kernel": C.GRAPH_TABLES[name][1], "tables": "Table 7",
           "view": "round medians of three rounds", "n_calls": int(len(t.df)), "floor_us": 1000 * t.floor_ms,
           "policies_paper_view": policy_rows(t.df, t.family if t.family in DEFAULT_CHUNK else "mamba1", t.chunks,
                                              t.lat, t.layers),
           "floors": {"cross_round": floor_cross_round(R, keys, default_idx=didx),
                      "cross_round_above_floor": floor_cross_round(R, keys, sel=t.above, default_idx=didx)}}
    if t.default is None:      # Triton: Table 7's default is the best single chunk
        p = np.full(len(t.df), didx)
        r = TRN.record(t.lat, t.orc, p, layers=t.layers)
        rec["policies_paper_view"]["default"] = {k: r[k] for k in ("n", "mean_regret_pct", "p95_regret_pct",
                                                                   "lw_share_of_oracle_pct", "us_per_call", "ms_per_forward_pass")}
        rec["policies_paper_view"]["default"]["note"] = "best single chunk in sample (Table 7)"
    one = name + "1"
    if one in E22.TABLES:          # single-round session of the same calls (RTX 3090)
        t1 = E22.Table(one, args.in_dir)
        rec["floors"]["cross_session"] = {**floor_cross_session(t.df, t1.df), "other_session": t1.csv,
                                          "note": "single-round graph session of the same calls"}
    return rec


# --------------------------------------------------------------------------- information data sets
def _effect(df: pd.DataFrame, tag: str) -> tuple[float, float]:
    """Held-out best single chunk and held-out shape tree (``e3b_information``), mean per-call regret (%)."""
    t = E3B.Table(df.reset_index(drop=True), tag, "", "", "")
    reg = lambda p: float(100 * (t.lat[np.arange(t.N), p] / t.orc - 1).mean())  # noqa: E731
    cols, depth = E3.FEATURE_SETS["shape"]
    return reg(E3B.cv_static(t)), reg(E3B.cv_tree(t, cols, depth))


def information_sets(args, rng) -> dict:
    summ = C.load_json(args.in_dir / "e3b_information_summary.json") or {"rows": {}}
    out = {}
    checks = []
    jobs = [(tag, csv, rmed, False) for tag, (csv, rmed) in E3B_TAGS.items()]
    jobs += [(tag + "|above_floor", csv, rmed, True) for tag, (csv, rmed) in E3B_TAGS.items()
             if tag in ("gdn_vgpu3090_1p4b_r3", "mamba2_vgpu3090_2p7b_r3", "mamba2_vgpu3090_r3spin")]
    for tag, csv, rmed, above_only in jobs:
        df = load(args.in_dir, csv, rmed)
        if above_only:
            o = latency_matrix(df, chunk_columns(df)).min(1)
            small = (df["batch"].to_numpy() == 1) & (df["L"].to_numpy() <= 256)
            df = df[o > 2.0 * np.median(o[small])].reset_index(drop=True)
        chunks, lat, std = matrices(df)
        keys = shape_codes(df)
        if rmed:
            fl = floor_cross_round(C.round_matrix(df, chunks), keys)
        else:
            fl = floor_split_half(lat, std, df["repeats"].to_numpy(float), keys, args.draws, rng)
        eff, shape = _effect(df, tag)
        base = tag.split("|")[0]
        stored = C.load_json(args.in_dir / f"e3b_information_{base}.json")
        pred = (stored or {}).get("predictors", {})
        if not above_only and pred.get("global_static"):
            checks.append({"tag": tag, "global_static_recomputed": eff, "stored": pred["global_static"]["mean_regret_pct"],
                           "shape_recomputed": shape, "shape_stored": pred.get("shape", {}).get("mean_regret_pct"),
                           "match": bool(abs(eff - pred["global_static"]["mean_regret_pct"]) < 1e-6
                                         and abs(shape - pred["shape"]["mean_regret_pct"]) < 1e-6)})
        row = summ["rows"].get(base, {}) if not above_only else {}
        floor_shape = fl["per_shape"]["mean_regret_pct"]
        floor_call = fl["per_call"]["mean_regret_pct"]
        out[tag] = {"source": csv, "n_calls": int(len(df)), "floor_method": fl["method"], "floor_per_call_pct": floor_call,
                    "floor_per_shape_pct": floor_shape, "chunk_effect_best_single_chunk_held_out_pct": eff,
                    "shape_tree_pct": shape, "paired_shape_entropy_minus_shape": row.get("shape+entropy - shape pp [95% CI]"),
                    "effect_over_floor_ratio": (eff / floor_shape) if (eff is not None and floor_shape > 0) else None,
                    "resolves_chunk_effect": bool(eff is not None and eff > EFFECT_OVER_FLOOR * floor_shape),
                    "shape_tree_minus_floor_pp": (shape - floor_shape) if shape is not None else None}
    main = {k: v for k, v in out.items() if "|" not in k}
    n_res = sum(v["resolves_chunk_effect"] for v in main.values())
    return {"criterion": f"chunk effect (held-out best single chunk) > {EFFECT_OVER_FLOOR} x per-shape noise floor",
            "checks_against_stored_e3b": checks,
            "n_datasets": len(main), "n_resolving": n_res,
            "resolving": sorted(k for k, v in main.items() if v["resolves_chunk_effect"]),
            "not_resolving": sorted(k for k, v in main.items() if not v["resolves_chunk_effect"]),
            "above_floor_subsets_resolving": sorted(k for k, v in out.items() if "|" in k and v["resolves_chunk_effect"]),
            "datasets": out}


# --------------------------------------------------------------------------- floor sensitivity
def launch_floor_sensitivity(args) -> dict:
    out = {"definition": "floor = factor x median oracle latency of the batch-1 calls with L <= 256 (published factor 2)"}
    for nm, (csv, fam, layers, dchunk) in {"ssd2p7b_device": ("e2_mamba2_vgpu3090_2p7b_r3.csv", "mamba2", 64, 256),
                                           "gdn1p4b_device": ("e2_gdn_vgpu3090_1p4b_r3.csv", "gdn", 12, 64)}.items():
        df = load(args.in_dir, csv, False)
        chunks, lat, _ = matrices(df)
        orc = lat.min(1)
        oidx = lat.argmin(1)
        L, b = df["L"].to_numpy(), df["batch"].to_numpy()
        base = float(np.median(orc[(b == 1) & (L <= 256)]))
        j = chunks.index(dchunk)
        reg = lat[:, j] / orc - 1
        tab = DU.cv_exact_pick(df, np.where(np.isfinite(lat), lat, 1e30))
        rows = {}
        for f in FLOOR_FACTORS:
            above = orc > f * base
            sub = df[above].reset_index(drop=True)
            h = cv_exact_headroom(sub, chunk_columns(sub), threshold_ms=C.SYNC_US["RTX 3090"] / 1000) if len(sub) >= 10 else {}
            cov = h.get("covered", {})
            r_def = TRN.record(lat, orc, np.full(len(df), j), layers=layers, sel=above)
            r_tab = TRN.record(lat, orc, tab, layers=layers, sel=above)
            rows[str(f)] = {"floor_us": 1000 * f * base, "n_above": int(above.sum()), "share_above": float(above.mean()),
                            f"chunk_{dchunk}_is_oracle": int((oidx[above] == j).sum()),
                            f"chunk_{dchunk}_max_per_batch_mean_regret_pct": float(max(
                                100 * reg[above & (b == bb)].mean() for bb in np.unique(b) if (above & (b == bb)).any())),
                            "heldout_headroom_covered_us": 1000 * cov["mean_ms"] if cov.get("n_calls") else None,
                            "heldout_headroom_covered_keys_above_sync": cov.get("n_keys_above_threshold"),
                            "table8_default_ms_per_forward_pass": r_def.get("ms_per_forward_pass"),
                            "table8_heldout_table_ms_per_forward_pass": r_tab.get("ms_per_forward_pass")}
        out[nm] = {"source": csv, "median_b1_L_le_256_us": 1000 * base, "by_factor": rows}
    graph = {}
    for nm in C.register_graph_tables(args.in_dir):
        t = C.graph_table(nm, args.in_dir)
        small = (t.batch == 1) & (t.L <= 256)
        base = float(np.median(t.orc[small]))
        j = t.default_idx()
        p = np.full(len(t.df), j)
        rows = {}
        for f in FLOOR_FACTORS:
            above = t.orc > f * base
            r = TRN.record(t.lat, t.orc, p, layers=t.layers, sel=above)
            rows[str(f)] = {"floor_us": 1000 * f * base, "share_above": float(above.mean()),
                            "default_mean_regret_above_pct": r.get("mean_regret_pct"),
                            "default_lw_above_pct": r.get("lw_share_of_oracle_pct")}
        graph[nm] = {"gpu": C.GRAPH_TABLES[nm][0], "kernel": C.GRAPH_TABLES[nm][1], "by_factor": rows}
    out["graph_tables"] = graph
    return out


def threshold_sensitivity(args) -> dict:
    """Launch-inclusive vs device time on the captured 370M Mamba-1 calls around the 100 us threshold."""
    pairs = {"RTX 3090 (device: single block)": ("e2_mamba1_vgpu3090.csv", "e2_mamba1_vgpu3090_r3spin.csv", False),
             "RTX 3090 (device: three-round medians)": ("e2_mamba1_vgpu3090.csv", "e2_mamba1_vgpu3090_r4spin.csv", True),
             "RTX PRO 6000 (device: three-round medians)": ("e2_mamba1_pro6000.csv", "e2_mamba1_pro6000_r4spin.csv", True),
             "H800 (device: single block)": ("e2_mamba1_h800.csv", "e2_mamba1_h800_r3spin.csv", False)}
    out = {"definition": ("threshold = factor x 100 us on the launch-inclusive oracle latency; relative difference = "
                          "launch-inclusive / device time - 1, per call at each convention's own oracle and per "
                          "(call, chunk) cell; oracle change = the two conventions' per-call oracles differ")}
    for label, (li_csv, dev_csv, rmed) in pairs.items():
        li = load(args.in_dir, li_csv, False).set_index("key")
        dv = load(args.in_dir, dev_csv, rmed).set_index("key")
        keys = [k for k in li.index if k in dv.index]
        chunks = [c for c in chunk_columns(li.reset_index()) if c in chunk_columns(dv.reset_index())]
        A = latency_matrix(li.loc[keys], chunks)
        B = latency_matrix(dv.loc[keys], chunks)
        oa, ob = A.min(1), B.min(1)
        rows = {}
        for f in THRESH_FACTORS:
            thr = 0.1 * f
            below = oa < thr
            rel_call = oa[~below] / ob[~below] - 1
            cell = (A / B - 1)[~below]
            rows[f"{int(round(100 * f))}us"] = {
                "share_calls_li_oracle_below": float(below.mean()),
                "above_median_abs_rel_diff_at_oracle_pct": float(100 * np.median(np.abs(rel_call))) if len(rel_call) else None,
                "above_p95_abs_rel_diff_at_oracle_pct": float(100 * np.percentile(np.abs(rel_call), 95)) if len(rel_call) else None,
                "above_median_abs_rel_diff_cells_pct": float(100 * np.nanmedian(np.abs(cell))) if cell.size else None,
                "above_p95_abs_rel_diff_cells_pct": float(100 * np.nanpercentile(np.abs(cell), 95)) if cell.size else None,
                "below_max_ratio_at_oracle": float((oa[below] / ob[below]).max()) if below.any() else None,
                "oracle_change_share_below": float((A[below].argmin(1) != B[below].argmin(1)).mean()) if below.any() else None,
                "oracle_change_share_above": float((A[~below].argmin(1) != B[~below].argmin(1)).mean()) if (~below).any() else None}
        out[label] = {"n_calls": len(keys), "by_threshold": rows,
                      "oracle_change_share_all": float((A.argmin(1) != B.argmin(1)).mean())}
    return out


# --------------------------------------------------------------------------- compact table map
def table_map(ds: dict, gr: dict) -> dict:
    def fl(rec):
        f = rec["floors"]
        best = f.get("cross_session") or f.get("cross_round") or f.get("split_half")
        meth = "cross_session" if "cross_session" in f else ("cross_round" if "cross_round" in f else "split_half")
        return {"method": meth, "per_call_pct": round(best["per_call"]["mean_regret_pct"], 3),
                "per_shape_pct": round(best["per_shape"]["mean_regret_pct"], 3),
                "per_shape_lw_pct": round(best["per_shape"]["lw_share_of_oracle_pct"], 3),
                "per_shape_us": round(best["per_shape"]["us_per_call"], 2),
                "split_half_per_call_pct (lower bound)": round(f["split_half"]["per_call"]["mean_regret_pct"], 3)
                if "split_half" in f else None,
                "cross_round_per_call_pct": round(f["cross_round"]["per_call"]["mean_regret_pct"], 3) if "cross_round" in f else None}

    def pol(rec, n):
        p = rec["policies_paper_view"].get(n)
        return None if p is None else {"mean_regret_pct": round(p["mean_regret_pct"], 3),
                                       "lw_share_of_oracle_pct": round(p["lw_share_of_oracle_pct"], 3),
                                       "us_per_call": round(p["us_per_call"], 2)}
    out = {}
    for n, r in ds.items():
        out[n] = {"tables": r["tables"], "gpu": r["gpu"], "convention": r["convention"], "floor": fl(r),
                  "default": pol(r, "default"), "best_static_per_batch": pol(r, "best_static_per_batch_in_sample"),
                  "exact_table_held_out": pol(r, "exact_table_held_out")}
        if "split_half_above_floor" in r["floors"]:
            fa = r["floors"]["split_half_above_floor"]
            out[n]["floor_above_launch_floor_split_half"] = {"per_call_pct": round(fa["per_call"]["mean_regret_pct"], 3),
                                                             "per_shape_pct": round(fa["per_shape"]["mean_regret_pct"], 3)}
    for n, r in gr.items():
        out[n] = {"tables": "Table 7", "gpu": r["gpu"], "kernel": r["kernel"], "floor": fl(r),
                  "floor_above_launch_floor_cross_round": {
                      "per_call_pct": round(r["floors"]["cross_round_above_floor"]["per_call"]["mean_regret_pct"], 3),
                      "per_shape_pct": round(r["floors"]["cross_round_above_floor"]["per_shape"]["mean_regret_pct"], 3)},
                  "default_single_round_scoring": r["floors"]["cross_round"].get("default_scored_on_single_rounds"),
                  "default": pol(r, "default"), "exact_table_held_out": pol(r, "exact_table_held_out")}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--draws", type=int, default=100)
    args = p.parse_args(argv)
    t0 = time.time()
    rng = C.make_rng("m5a")
    ds = {}
    for n in DATASETS:
        ds[n] = analyze_dataset(n, args, C.make_rng("m5a", n))
        print(f"{n}: done ({time.time() - t0:.0f} s)", flush=True)
    gr = {}
    for n in C.register_graph_tables(args.in_dir):
        gr[n] = analyze_graph(n, args, C.make_rng("m5a", n))
        print(f"{n}: done ({time.time() - t0:.0f} s)", flush=True)
    info = information_sets(args, rng)
    print(f"information data sets: done ({time.time() - t0:.0f} s)", flush=True)
    out = {"item": "M5a", "meta": C.meta(__name__, args), "definitions": __doc__,
           "table_map": table_map(ds, gr), "datasets": ds, "graph_tables": gr, "information_datasets": info,
           "launch_floor_sensitivity": launch_floor_sensitivity(args), "threshold_100us_sensitivity": threshold_sensitivity(args),
           "runtime_s": None}
    out["runtime_s"] = round(time.time() - t0, 1)
    path = C.write_result("m5a", out, args.in_dir)
    for n, r in out["table_map"].items():
        f = r["floor"]
        d = r.get("default") or {}
        print(f"{n:30s} floor[{f['method']}] call {f['per_call_pct']:.3f}% shape {f['per_shape_pct']:.3f}% "
              f"({f['per_shape_us']:.2f} us)  default {d.get('mean_regret_pct')}  table "
              f"{(r.get('exact_table_held_out') or {}).get('mean_regret_pct')}")
    print(f"information data sets resolving the chunk effect: {info['n_resolving']} of {info['n_datasets']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
