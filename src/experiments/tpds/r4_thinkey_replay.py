"""Thin serving key of the RTX 3090 replay, re-estimated from 75 calls (review item M10b; tpds-r4 E21).

CPU only, deterministic; reads stored result files and the serving traces, times nothing.

The serving replay (``serving_key_churn``) maps every call shape to the nearest measured (length bucket,
batch, L mod 16 == 0) key of the captured calls and charges that key's tuning cost P_k, oracle headroom
h_k and cost-model regret, as estimated by ``e5b_key_churn.analyze`` from the captured calls of the key
(``e5b_<gpu>.json``). An online policy can beat a tuning cache on a key only if N_k < N*_k = P_k / (o - h_k),
unbounded when h_k >= o. On the RTX 3090 the padded budget-step rows (Azure code, 1 h, budget 8192: 90.8% of
the calls below the host-synchronization break-even, 92.8% at prefill load 0.85) put most of their calls on one
key, (bucket 2048, batch 4, L mod 16 != 0), estimated from 11 captured calls with h_k = 26.2 us above one
synchronization (21.4 us), hence N* unbounded.

E21 timed 64 synthesized calls of this key (target lengths drawn without replacement from [2048, 4096) with
L mod 16 != 0, four captured prompts of one fold per call, layers rotating; three rounds, chunk order shuffled
per call and round; launch-inclusive and device time). This script

 1. Key estimates. Identifies the key's 11 captured calls in the multi-round re-timing of the captured calls
    (E19, ``e2_mamba1_<gpu>_<conv>.csv``) and pools them with the 64 synthesized calls
    (``e2_mamba1_<gpu>_<conv>_u2048b4.csv``). For the samples cap11, syn64 and all75, in both conventions and in
    the views round_median (median of the round means), r0 / r1 / r2 (one block per chunk) and, for cap11,
    the published single-block table, it recomputes every key-level quantity of ``e5b_key_churn.analyze``:
    table chunk (in sample), h_k, P_k for profiling and for the launch search (tuning scale of
    ``e5b_<gpu>.json``), the decision overheads at the key's median tensor size, the cost-model regret
    against the oracle and against the table (frozen parameters of the same convention) and N* for every
    tuning cost / overhead pair. Added: h_k with the table chosen on held-out prompt folds, the cross-round
    h_k (table and per-call oracle chosen on the median of two rounds, scored on the third; every round once),
    and the key split into length thirds (oracle distribution, table chunk, headroom of the bucket table).
    Check: the same code applied to the published table reproduces all keys of ``e5b_<gpu>.json``.

 2. Replay. Re-runs every row of the replay (``serving_key_churn.WINDOWS``: per-request, fixed-slice and
    chunked-budget policies, every budget, scheduler and load) with the module's own trace loader, window
    selection, scheduler simulation and nearest-key mapping (``HostCosts``), for three cost sets of the GPU:
      published        ``e5b_<gpu>.json`` as stored (validated against ``serving_key_churn.json``: identical
                       shares below N* for every row and tuning-cost / overhead pair);
      thin_key_75      the thin key replaced by its 75-call round-median launch-inclusive estimate (h, P,
                       model regret, 75 calls); every other key as stored;
      all_keys_round_median   every key re-estimated on the round medians of the E19 launch-inclusive re-timing,
                       the thin key from the 75 calls (sensitivity: the round-median oracle everywhere).
    Per row and cost set: shares of calls below N* (bucket key; profiling or search against host
    synchronization, every-8th-element or full histogram), for packed rows also under the packed-length key,
    the share of calls whose nearest measured key has fewer than 30 calls (M10b; ``deployment_units``
    definition) and the share on the thin key, plus the numbers the manuscript quotes.
    Batch-1 calls (per-request, fixed-slice and packed rows) never map to a batch-4 key (checked on every
    length up to 2^17), so only padded budget rows can change under ``thin_key_75``.

Output: ``--out`` (default ``r4_thinkey_<gpu>.json`` in the results directory).

Usage:  python3 -m src.experiments.tpds.r4_thinkey_replay [--workers 6] [--windows all]
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.tpds import cost_model as cm
from src.experiments.tpds import serving_key_churn as skc
from src.experiments.tpds.analysis import chunk_columns, latency_matrix, load_e2, round_median_view
from src.experiments.tpds.common import RESULTS_DIR, read_jsonl, utc_now, write_json
from src.experiments.tpds.deployment_units import THIN_KEY_CALLS, nearest_measured
from src.experiments.tpds.e5b_key_churn import KEY_MODES, LAYERS_PER_REQUEST, _n_star, _overhead
from src.experiments.tpds.r4_rounds_analysis import round_matrix

THIN_KEY = (2048, 4, 0)                  # (L_bucket, batch, aligned)
COMBOS = skc.NSTAR_COMBOS
PRIMARY_CONV = "r4plain"                 # the replay's costs are launch-inclusive (e5b from the plain E2 table)


# --------------------------------------------------------------------------- key-level quantities (E5b)
class KeyContext:
    """Tuning scale, overheads and frozen cost model of one GPU and convention."""

    def __init__(self, gpu: str, cal: str, in_dir: Path):
        self.e5b = json.loads((in_dir / f"e5b_{gpu}.json").read_text())
        self.scale = self.e5b["tuning_scale"]
        self.sync_ms = float(self.e5b["host_sync_ms"])
        self.e4_rows = read_jsonl(in_dir / f"e4_{gpu}.jsonl")
        self.cal = cal
        st = json.loads((in_dir / f"cost_model_{cal}.json").read_text())
        self.theta = np.array([st["params"][k] for k in cm.PARAMS])
        self.hw = cm.hardware(cal, in_dir)
        self.stats = cm.load_kernel_stats(in_dir, self.hw["arch"])

    def model_choice(self, df: pd.DataFrame, chunks: list[int]) -> np.ndarray:
        return cm.select_e2(self.theta, df, chunks, self.stats, self.hw)[0]


def key_record(df: pd.DataFrame, lat: np.ndarray, chunks: list[int], model: np.ndarray, ctx: KeyContext) -> dict:
    """The record of ``e5b_key_churn.analyze`` for one key (calls = rows of df / lat)."""
    orc = lat.min(1)
    ar = np.arange(len(lat))
    table = int(np.argmin(lat.mean(0)))
    h = float((lat[:, table] - orc).mean())
    sum_lat = float(lat.mean(0).sum())
    b = int(df["batch"].iloc[0])
    numel = int(b * int(df["width"].iloc[0]) * int(df["L"].median()))
    P = {"profile": ctx.scale["profile_s_per_ms"] * sum_lat, "search": ctx.scale["search_s_per_ms"] * sum_lat}
    o = {"host_sync": ctx.sync_ms, "sampled_histogram": _overhead(ctx.e4_rows, numel, 8),
         "full_histogram": _overhead(ctx.e4_rows, numel, 1)}
    rec = {"n_calls": int(len(lat)), "table_chunk": int(chunks[table]), "table_ms": float(lat[:, table].mean()),
           "oracle_ms": float(orc.mean()), "headroom_ms": h, "headroom_us": 1000 * h,
           "headroom_pct": 100 * h / float(lat[:, table].mean()), "P_s": P, "overhead_ms": o,
           "headroom_above_host_sync": bool(h >= ctx.sync_ms),
           "oracle_distribution": {int(k): int(v) for k, v in pd.Series(np.array(chunks)[lat.argmin(1)]).value_counts().items()},
           "model_regret_vs_oracle_ms": float((lat[ar, model] - orc).mean()),
           "model_regret_vs_table_ms": float((lat[ar, model] - lat[:, table]).mean()),
           "n_star_calls": {}}
    for pk, pv in P.items():
        for ok, ov in o.items():
            rec["n_star_calls"][f"{pk}|{ok}"] = _n_star(pv, ov, h) if ov is not None else None
    return rec


def heldout_headroom(df: pd.DataFrame, lat: np.ndarray) -> float:
    """h_k with the key's table chunk chosen on the other prompt folds (calls of folds without others: skipped)."""
    orc = lat.min(1)
    folds = df["fold"].to_numpy()
    h = []
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if tr.any():
            j = int(np.argmin(lat[tr].mean(0)))
            h += list(lat[te, j] - orc[te])
    return float(np.mean(h))


def cross_round_headroom(R: np.ndarray) -> float:
    """h_k with table and per-call oracle chosen on the median of the other rounds, scored on the held-out round."""
    out = []
    ar = np.arange(R.shape[1])
    for r in range(R.shape[0]):
        sel = np.median(np.delete(R, r, 0), 0)
        j, o = int(np.argmin(sel.mean(0))), sel.argmin(1)
        out.append(float((R[r][:, j] - R[r][ar, o]).mean()))
    return float(np.mean(out))


def thin_key_rows(df: pd.DataFrame) -> pd.Series:
    lb, b, al = THIN_KEY
    return (df["L_bucket"] == lb) & (df["batch"] == b) & ((df["L"] % 16 == 0).astype(int) == al)


def key_estimates(gpu: str, conv: str, in_dir: Path, ctx: KeyContext) -> dict:
    cap = load_e2([in_dir / f"e2_mamba1_{gpu}_{conv}.csv"])
    cap = cap[thin_key_rows(cap)].reset_index(drop=True)
    syn = load_e2([in_dir / f"e2_mamba1_{gpu}_{conv}_u2048b4.csv"]).reset_index(drop=True)
    assert thin_key_rows(syn).all(), "synthesized calls outside the thin key"
    both = pd.concat([cap, syn], ignore_index=True)
    chunks = chunk_columns(both)
    out = {"n_captured": int(len(cap)), "n_synthesized": int(len(syn)),
           "captured_lengths": sorted(int(x) for x in cap["L"]),
           "synthesized_lengths": {"min": int(syn["L"].min()), "max": int(syn["L"].max()),
                                   "median": float(syn["L"].median()), "n_distinct": int(syn["L"].nunique())},
           "synthesized_unique_prompts_per_call": {int(k): int(v) for k, v in syn["n_unique_prompts"].value_counts().items()},
           "samples": {}}
    for name, d in (("cap11", cap), ("syn64", syn), ("all75", both)):
        R = round_matrix(d, chunks)
        model = ctx.model_choice(d, chunks)
        rec = {}
        views = {"round_median": latency_matrix(round_median_view(d), chunks), **{f"r{r}": R[r] for r in range(R.shape[0])}}
        for v, lat in views.items():
            rec[v] = key_record(d, lat, chunks, model, ctx)
            rec[v]["headroom_held_out_folds_us"] = 1000 * heldout_headroom(d, lat)
        rec["round_median"]["headroom_cross_round_us"] = 1000 * cross_round_headroom(R)
        out["samples"][name] = rec
    # the key split into length thirds of the bucket (all 75 calls, round medians)
    lat = latency_matrix(round_median_view(both), chunks)
    j_key = int(np.argmin(lat.mean(0)))
    edges = np.linspace(2048, 4096, 4)
    thirds = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = ((both["L"] >= lo) & (both["L"] < hi)).to_numpy()
        if not s.any():
            continue
        orc = lat[s].min(1)
        j = int(np.argmin(lat[s].mean(0)))
        thirds.append({"L_range": [int(np.ceil(lo)), int(np.ceil(hi)) - 1], "n_calls": int(s.sum()),
                       "oracle_distribution": {int(k): int(v) for k, v in
                                               pd.Series(np.array(chunks)[lat[s].argmin(1)]).value_counts().items()},
                       "table_chunk_of_third": int(chunks[j]),
                       "headroom_of_third_table_us": float(1000 * (lat[s, j] - orc).mean()),
                       "headroom_of_bucket_table_us": float(1000 * (lat[s, j_key] - orc).mean()),
                       "mean_oracle_latency_us": float(1000 * orc.mean())})
    out["length_thirds_all75_round_median"] = {"bucket_table_chunk": int(chunks[j_key]), "thirds": thirds}
    # relative gap between the two leading chunks per call
    i256, i512 = chunks.index(256), chunks.index(512)
    gap = lat[:, i512] / lat[:, i256] - 1
    out["lat512_over_lat256_minus_1_all75_pct"] = {"median": float(100 * np.median(gap)), "p10": float(100 * np.percentile(gap, 10)),
                                                   "p90": float(100 * np.percentile(gap, 90)),
                                                   "corr_with_L": float(np.corrcoef(gap, both["L"].to_numpy(float))[0, 1])}
    return out


def published_key_records(gpu: str, in_dir: Path, ctx: KeyContext, csv: str) -> tuple[list[dict], dict]:
    """All keys of a table with the E5b definitions; with the stored per-call model choice when the keys match."""
    df = load_e2([in_dir / csv])
    df["aligned"] = (df["L"] % 16 == 0).astype(int)
    chunks = chunk_columns(df)
    lat = latency_matrix(df, chunks)
    stored = json.loads((in_dir / f"cost_model_{gpu}.json").read_text())["per_call_choice"]["mamba1"]
    m = dict(zip(stored["keys"], stored["chunk"]))
    model = np.array([chunks.index(int(m[k])) for k in df["key"]])
    recs = []
    for r in ctx.e5b["keys"]:                       # the replay's key order
        g = df[(df["L_bucket"] == r["L_bucket"]) & (df["batch"] == r["batch"]) & (df["aligned"] == r["aligned"])]
        i = g.index.to_numpy()
        recs.append({"L_bucket": r["L_bucket"], "batch": r["batch"], "aligned": r["aligned"],
                     **key_record(g, lat[i], chunks, model[i], ctx)})
    return recs, {"chunks": chunks}


# --------------------------------------------------------------------------- cost sets of the replay
def variant_costs(base: skc.HostCosts, recs: list[dict] | None, n_calls: np.ndarray, name: str) -> skc.HostCosts:
    """A copy of ``base`` whose per-key P, h and model regret come from ``recs`` (None: unchanged)."""
    hc = object.__new__(skc.HostCosts)
    hc.__dict__.update(base.__dict__)
    hc.host = name
    hc._ov_cache = base._ov_cache                   # overheads depend on the call shape only
    if recs is not None:
        hc.rPp = np.array([r["P_s"]["profile"] for r in recs])
        hc.rPs = np.array([r["P_s"]["search"] for r in recs])
        hc.rh = np.array([r["headroom_ms"] for r in recs])
        hc.rreg = np.array([r["model_regret_vs_oracle_ms"] for r in recs])
    hc.n_calls_measured = np.asarray(n_calls, float)
    hc.thin_index = int(next(i for i, (lb, b, al) in enumerate(zip(base.rLb, base.rb, base.ral))
                             if (int(round(2 ** lb)), int(round(2 ** b)), int(al)) == THIN_KEY))
    return hc


def row_shares(Lk: np.ndarray, bk: np.ndarray, w: np.ndarray, packed: bool, variants: list[skc.HostCosts]) -> dict:
    """Bucket-key shares below N* (``serving_key_churn.key_stats``) and M10b thin-key shares per cost set."""
    code = Lk.astype(np.int64) << skc.SHAPE_BITS | bk.astype(np.int64)
    ue, inv = np.unique(code, return_inverse=True)
    We = np.bincount(inv, weights=w)
    Le, be = ue >> skc.SHAPE_BITS, ue & ((1 << skc.SHAPE_BITS) - 1)
    bcost = np.ones_like(be) if packed else be
    calls_e = LAYERS_PER_REQUEST * We
    modes = {"bucket_aligned": KEY_MODES["bucket_aligned"]}
    if packed:
        modes[skc.PACKED_TOKENS_MODE] = ["L_bucket", "aligned"]
    kinv = {}
    for mode, cols in modes.items():
        kc = skc._bucket_log2(Le) * 2 + (Le % 16 == 0)
        kcode = kc << skc.SHAPE_BITS | (be if "batch" in cols else 0)
        kinv[mode] = np.unique(kcode, return_inverse=True)[1]
    out = {"n_calls": float(calls_e.sum()), "n_keys": {m: int(k.max() + 1) for m, k in kinv.items()}, "by_cost_set": {}}
    for hc in variants:
        c = hc.costs(Le, bcost)
        nn, _ = nearest_measured(hc, Le, bcost)
        rec = {}
        for mode, ki in kinv.items():
            Wk = np.bincount(ki, weights=We)
            calls = LAYERS_PER_REQUEST * Wk
            agg = {k: np.bincount(ki, weights=We * v) / Wk for k, v in c.items() if k != "extrapolated"}
            sh = {}
            for combo in COMBOS:
                pk, ok = combo.split("|")
                ns = [_n_star(p, o, h) for p, o, h in zip(agg[f"P_{pk}"], agg[f"o_{ok}"], agg["h"])]
                below = np.array([s is None or n < s for n, s in zip(calls, ns)])
                sh[combo] = float(calls[below].sum() / calls.sum())
            rec[mode] = sh
        thin = hc.n_calls_measured[nn] < THIN_KEY_CALLS
        rec["m10b"] = {"share_calls_on_thin_measured_keys": float(calls_e[thin].sum() / calls_e.sum()),
                       "share_calls_on_key_2048_b4_unaligned": float(calls_e[nn == hc.thin_index].sum() / calls_e.sum()),
                       "thin_measured_keys_hit": sorted({int(k) for k in np.unique(nn[thin])})}
        out["by_cost_set"][hc.host] = rec
    return out


# --------------------------------------------------------------------------- replay driver (fork workers)
_G: dict = {}


def _tasks(name: str, window: str, policies) -> list[tuple]:
    t = []
    for pol in policies:
        if pol == "per_request":
            t.append((name, window, "per_request", None, None, None))
        elif pol == "fixed_slice":
            t += [(name, window, "fixed_slice", T, None, None) for T in skc.BUDGETS]
        else:
            settings = ([("colocated", skc.RHO_PRIMARY)] + [("colocated", r) for r in skc.RHO_SENSITIVITY]
                        + [("prefill_only", skc.RHO_PRIMARY)])
            t += [(name, window, "chunked_budget", T, s, r) for s, r in settings for T in skc.BUDGETS]
    return t


def _run(task) -> list[dict]:
    name, window, policy, T, sched, rho = task
    tr, variants = _G["trace"], _G["variants"]
    win, info = skc.select_window(tr, window)
    base = {"trace": name, "window": window, "policy": policy, "T": T, "sched": sched, "rho": rho}
    t0 = time.time()
    if policy in ("per_request", "fixed_slice"):
        Lk, bk, w = skc.policy_entries(win, policy, T)["batch1"]
        rows = [{**base, "kernel": "batch1", **row_shares(Lk, bk, w, False, variants)}]
    else:
        tok = float(win["ctx"].sum())
        s = tok * skc.T_PF_S / (info["duration_s"] * rho)
        sim = skc.simulate_chunked(win["t"] * s, win["ctx"], win["gen"], T, colocate=(sched == "colocated"))
        w = np.ones(len(sim["nseq"]))
        rows = [{**base, "kernel": k, **row_shares(Lk, bk, w, k == "packed", variants)}
                for k, (Lk, bk) in (("padded", (sim["lmax"], sim["nseq"])), ("packed", (sim["tot"], sim["nseq"])))]
    for r in rows:
        r["seconds"] = round(time.time() - t0, 1)
    return rows


def replay(variants: list[skc.HostCosts], data_dir: Path, workers: int, windows: str, log) -> list[dict]:
    sel = skc.WINDOWS if windows == "all" else [w for w in skc.WINDOWS if f"{w[0]}/{w[1]}" in windows.split(",")]
    _G["variants"] = variants
    rows = []
    for name in dict.fromkeys(w[0] for w in sel):
        t0 = time.time()
        _G["trace"] = skc.load_trace(name, data_dir)
        log(f"loaded {name}: {len(_G['trace']['ctx'])} requests, {time.time() - t0:.1f}s")
        tasks = [t for nm, win, pols in sel if nm == name for t in _tasks(nm, win, pols)]
        tasks.sort(key=lambda t: (t[2] != "chunked_budget", t[1] not in ("full", "24h"), -(t[3] or 0)))
        with get_context("fork").Pool(max(1, min(workers, len(tasks)))) as pool:
            for rr in pool.imap_unordered(_run, tasks):
                rows += rr
                for r in rr:
                    log(f"  {r['trace']}/{r['window']} {r['policy']} T={r['T']} {r['kernel']} {r['sched']} "
                        f"rho={r['rho']}: {r['seconds']}s")
        _G.pop("trace", None)
    rows.sort(key=lambda r: (r["trace"], r["window"], r["policy"], r["T"] or 0, r["kernel"], r["sched"] or "", r["rho"] or 0))
    return rows


def _key(r) -> tuple:
    return (r["trace"], r["window"], r["policy"], r["T"], r["kernel"], r["sched"], r["rho"])


def validate(rows: list[dict], gpu: str, in_dir: Path) -> dict:
    ref = {_key(r): r for r in json.loads((in_dir / "serving_key_churn.json").read_text())["results"]}
    du_p = in_dir / "deployment_units.json"
    du = {_key(r): r for r in json.loads(du_p.read_text())["serving"]["rows"]} if du_p.exists() else {}
    max_diff, max_thin, n, n_du, missing = 0.0, 0.0, 0, 0, []
    for r in rows:
        rr = ref.get(_key(r))
        if rr is None:
            missing.append(list(_key(r)))
            continue
        a = r["by_cost_set"]["published"]
        for mode in a:
            if mode == "m10b":
                continue
            b = rr["keys"][mode]["by_host"][gpu]["share_calls_below_nstar"]
            assert r["n_keys"][mode] == rr["keys"][mode]["n_keys"], (_key(r), mode)
            for combo in COMBOS:
                max_diff = max(max_diff, abs(a[mode][combo] - b[combo]))
            n += 1
        d = du.get(_key(r))
        if d is not None:
            m = d["by_host"][gpu]["m10b"]
            max_thin = max(max_thin, abs(m["share_calls_on_thin_measured_keys"] - a["m10b"]["share_calls_on_thin_measured_keys"]),
                           abs(m["share_calls_on_key_2048_b4_unaligned"] - a["m10b"]["share_calls_on_key_2048_b4_unaligned"]))
            n_du += 1
    return {"rows_mode_pairs_checked_vs_serving_key_churn": n, "max_abs_diff_share_below_nstar": max_diff,
            "rows_checked_vs_deployment_units_m10b": n_du, "max_abs_diff_m10b_shares": max_thin,
            "rows_missing_in_reference": missing}


def quoted_numbers(rows: list[dict], cost_set: str) -> dict:
    """The replay numbers the manuscript prints for the GPU, under one cost set."""
    def get(trace, window, policy, T, kernel, sched, rho, mode="bucket_aligned", combo="profile|host_sync"):
        for r in rows:
            if _key(r) == (trace, window, policy, T, kernel, sched, rho):
                return r["by_cost_set"][cost_set][mode][combo]
        return None

    def argmax(filt, mode="bucket_aligned", combo="profile|host_sync"):
        cand = [(r["by_cost_set"][cost_set][mode][combo], _key(r)) for r in rows if filt(r) and mode in r["by_cost_set"][cost_set]]
        if not cand:
            return None
        v, k = max(cand, key=lambda x: x[0])
        return {"share": v, "row": list(k)}

    budget = lambda r: r["policy"] == "chunked_budget" and r["kernel"] == "padded"  # noqa: E731
    out = {
        "max_padded_budget_row_below_nstar_profile_sync": argmax(budget),
        "max_padded_budget_row_colocated_rho0.5": argmax(lambda r: budget(r) and r["sched"] == "colocated" and r["rho"] == 0.5),
        "azure2023_code_1h_T8192_padded_colocated_rho0.5": get("azure2023_code", "1h", "chunked_budget", 8192, "padded", "colocated", 0.5),
        "azure2023_code_1h_T8192_padded_colocated_rho0.85": get("azure2023_code", "1h", "chunked_budget", 8192, "padded", "colocated", 0.85),
        "azure2024_code_24h_T8192_padded_colocated_rho0.5": get("azure2024_code", "24h", "chunked_budget", 8192, "padded", "colocated", 0.5),
        "burstgpt_24h_busiest_T8192_padded_colocated_rho0.5": get("burstgpt", "24h_busiest", "chunked_budget", 8192, "padded", "colocated", 0.5),
        "burstgpt_24h_busiest_T8192_padded_prefill_only_rho0.5": get("burstgpt", "24h_busiest", "chunked_budget", 8192, "padded", "prefill_only", 0.5),
        "max_any_row_below_nstar_sampled_histogram": argmax(lambda r: True, combo="profile|sampled_histogram"),
        "max_request_level_or_fixed_slice_below_nstar_profile_sync": argmax(lambda r: r["policy"] in ("per_request", "fixed_slice")),
        "max_packed_length_key_colocated": argmax(lambda r: r["kernel"] == "packed" and r["sched"] == "colocated",
                                                  mode=skc.PACKED_TOKENS_MODE),
        "max_packed_length_key_prefill_only": argmax(lambda r: r["kernel"] == "packed" and r["sched"] == "prefill_only",
                                                     mode=skc.PACKED_TOKENS_MODE),
    }
    thin = max(rows, key=lambda r: r["by_cost_set"][cost_set]["m10b"]["share_calls_on_thin_measured_keys"])
    out["max_share_calls_on_thin_measured_keys"] = {
        "share": thin["by_cost_set"][cost_set]["m10b"]["share_calls_on_thin_measured_keys"], "row": list(_key(thin))}
    return out


# --------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--data-dir", type=Path, default=skc.DATA_DIR)
    p.add_argument("--gpu", default="vgpu3090")
    p.add_argument("--out", type=Path, default=None, help="output JSON (default: <in-dir>/r4_thinkey_<gpu>.json)")
    p.add_argument("--workers", type=int, default=6, help="processes for the replay (fork)")
    p.add_argument("--windows", default="all", help="comma list of trace/window for the replay, or 'all'")
    p.add_argument("--skip-replay", action="store_true", help="key estimates only")
    args = p.parse_args(argv)
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    t0 = time.time()
    gpu, in_dir = args.gpu, args.in_dir
    ctx = {"r4plain": KeyContext(gpu, gpu, in_dir), "r4spin": KeyContext(gpu, f"{gpu}_r3spin", in_dir)}
    out = {"experiment": "r4_thinkey_replay (review tpds-r4 M10b; E21)", "created_utc": utc_now(), "gpu": gpu,
           "definitions": __doc__, "thin_key": {"L_bucket": THIN_KEY[0], "batch": THIN_KEY[1], "aligned": THIN_KEY[2]},
           "host_sync_us": 1000 * ctx["r4plain"].sync_ms, "thin_key_calls_threshold": THIN_KEY_CALLS}
    # 0. the E5b records of the published table, recomputed (check of key_record)
    recs_pub, _ = published_key_records(gpu, in_dir, ctx["r4plain"], f"e2_mamba1_{gpu}.csv")
    diffs = []
    for a, b in zip(recs_pub, ctx["r4plain"].e5b["keys"]):
        diffs.append(max(abs(a["headroom_ms"] - b["headroom_ms"]), abs(a["P_s"]["profile"] - b["P_s"]["profile"]),
                         abs(a["P_s"]["search"] - b["P_s"]["search"]),
                         abs(a["model_regret_vs_oracle_ms"] - b["model_regret_vs_oracle_ms"]),
                         abs(a["n_calls"] - b["n_calls"])))
    out["check_e5b_records_reproduced_max_abs_diff"] = float(max(diffs))
    pub_thin = next(r for r in ctx["r4plain"].e5b["keys"] if (r["L_bucket"], r["batch"], r["aligned"]) == THIN_KEY)
    out["published_thin_key_record"] = {k: pub_thin[k] for k in ("n_calls", "table_chunk", "headroom_ms", "P_s",
                                                                "model_regret_vs_oracle_ms", "n_star_calls")}
    # 1. key estimates, both conventions
    out["key_estimates"] = {}
    for conv in ("r4plain", "r4spin"):
        out["key_estimates"][conv] = {"convention": "launch-inclusive" if conv == "r4plain" else "device time",
                                      "cost_model": ctx[conv].cal, **key_estimates(gpu, conv, in_dir, ctx[conv])}
    # cap11 on the published single-block tables of both conventions
    for conv, csv in (("r4plain", f"e2_mamba1_{gpu}.csv"), ("r4spin", f"e2_mamba1_{gpu}_r3spin.csv")):
        d = load_e2([in_dir / csv])
        d = d[thin_key_rows(d)].reset_index(drop=True)
        ch = chunk_columns(d)
        out["key_estimates"][conv]["samples"]["cap11"]["published_single_block"] = key_record(
            d, latency_matrix(d, ch), ch, ctx[conv].model_choice(d, ch), ctx[conv])
    new = out["key_estimates"][PRIMARY_CONV]["samples"]["all75"]["round_median"]
    # 2. cost sets
    base = skc.HostCosts(gpu, in_dir)
    n_meas = np.array([r["n_calls"] for r in ctx["r4plain"].e5b["keys"]], float)
    k_thin = next(i for i, r in enumerate(ctx["r4plain"].e5b["keys"]) if (r["L_bucket"], r["batch"], r["aligned"]) == THIN_KEY)
    recs_thin = [dict(r) for r in ctx["r4plain"].e5b["keys"]]
    recs_thin[k_thin] = {**recs_thin[k_thin], "P_s": new["P_s"], "headroom_ms": new["headroom_ms"],
                         "model_regret_vs_oracle_ms": new["model_regret_vs_oracle_ms"], "n_calls": new["n_calls"]}
    n_thin = n_meas.copy()
    n_thin[k_thin] = new["n_calls"]
    # every key on the round medians of the E19 re-timing of the 2360 captured calls (same keys as e5b)
    df_rm = round_median_view(load_e2([in_dir / f"e2_mamba1_{gpu}_{PRIMARY_CONV}.csv"]))
    df_rm["aligned"] = (df_rm["L"] % 16 == 0).astype(int)
    ch = chunk_columns(df_rm)
    lat_rm = latency_matrix(df_rm, ch)
    model_rm = ctx["r4plain"].model_choice(df_rm, ch)
    recs_rm = []
    for r in ctx["r4plain"].e5b["keys"]:
        g = df_rm[(df_rm["L_bucket"] == r["L_bucket"]) & (df_rm["batch"] == r["batch"]) & (df_rm["aligned"] == r["aligned"])]
        i = g.index.to_numpy()
        recs_rm.append({"L_bucket": r["L_bucket"], "batch": r["batch"], "aligned": r["aligned"],
                        **key_record(g, lat_rm[i], ch, model_rm[i], ctx["r4plain"])})
    recs_rm[k_thin] = {**recs_rm[k_thin], **{k: new[k] for k in ("P_s", "headroom_ms", "model_regret_vs_oracle_ms", "n_calls")}}
    out["all_keys_round_median_records"] = [
        {k: r[k] for k in ("L_bucket", "batch", "aligned", "n_calls", "table_chunk", "headroom_ms", "P_s",
                           "model_regret_vs_oracle_ms", "headroom_above_host_sync") if k in r}
        | {"published_headroom_ms": b["headroom_ms"], "published_table_chunk": b["table_chunk"]}
        for r, b in zip(recs_rm, ctx["r4plain"].e5b["keys"])]
    out["keys_headroom_above_host_sync"] = {
        "published": int(sum(r["headroom_ms"] >= ctx["r4plain"].sync_ms for r in ctx["r4plain"].e5b["keys"])),
        "all_keys_round_median": int(sum(r["headroom_ms"] >= ctx["r4plain"].sync_ms for r in recs_rm))}
    variants = [variant_costs(base, None, n_meas, "published"),
                variant_costs(base, recs_thin, n_thin, "thin_key_75"),
                variant_costs(base, recs_rm, n_thin, "all_keys_round_median")]
    # batch-1 calls never map to the thin key (every length up to 2^17)
    L = np.arange(1, 1 << 17)
    nn, _ = nearest_measured(base, L, np.ones_like(L))
    out["check_batch1_never_maps_to_thin_key"] = bool(not np.any(nn == k_thin))
    nn4, _ = nearest_measured(base, L, np.full_like(L, 4))
    out["lengths_mapping_to_thin_key_at_batch4"] = {"min": int(L[nn4 == k_thin].min()), "max": int(L[nn4 == k_thin].max()),
                                                    "aligned_share": float((L[nn4 == k_thin] % 16 == 0).mean())}
    # 3. replay
    if not args.skip_replay:
        log("replay ...")
        rows = replay(variants, args.data_dir, args.workers, args.windows, log)
        out["replay"] = {"validation": validate(rows, gpu, in_dir),
                         "quoted_numbers": {v.host: quoted_numbers(rows, v.host) for v in variants},
                         "rows": rows}
        for r in rows:
            r.pop("seconds", None)
    out["runtime_s"] = round(time.time() - t0, 1)
    path = args.out or (in_dir / f"r4_thinkey_{gpu}.json")
    write_json(path, out)
    # console summary
    for conv, ke in out["key_estimates"].items():
        for s, rec in ke["samples"].items():
            rm = rec.get("round_median")
            log(f"{conv} {s}: h_k round median {rm['headroom_us']:.2f} us (table {rm['table_chunk']}), r0/r1/r2 "
                + "/".join(f"{rec[f'r{r}']['headroom_us']:.2f}" for r in range(3))
                + f", cross-round {rm['headroom_cross_round_us']:.2f}, held-out folds {rm['headroom_held_out_folds_us']:.2f}, "
                f"P {rm['P_s']['profile']:.3f} s, model regret {1000 * rm['model_regret_vs_oracle_ms']:.2f} us")
    if "replay" in out:
        log(f"validation: {out['replay']['validation']}")
        for v, q in out["replay"]["quoted_numbers"].items():
            log(f"{v}: " + json.dumps({k: (x if not isinstance(x, dict) else x['share']) for k, x in q.items()}))
    log(f"wrote {path} ({out['runtime_s']} s)")


if __name__ == "__main__":
    main()
