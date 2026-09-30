"""E12 analysis: chunk selection at tensor-parallel widths (review item M1c). CPU only.

Reads ``e12_tp_widths_<tag>.jsonl`` and, for every (family, width, length, batch) shape,
averages the per-chunk mean latency over the timing rounds and takes the argmin as the oracle.
A chunk is tied with the oracle when the 95% confidence intervals of their round means overlap
(t-distribution over rounds).

Selectors, none of which is fitted on E12 timings:
  cost_model       frozen cost model with the parameters calibrated at d = 2048 on the same GPU
                   (Triton: Eq. (5), ``cost_model_<host>.json``; reference scan:
                   ``cost_model_refscan_<host>.json``)
  unsharded_table  the measured best chunk of the same model at its full width, same (L, batch):
                   what a cache tuned on a single-GPU deployment would serve after sharding
  width2048_table  the measured best chunk at d = 2048, same (L, batch): a table tuned at the
                   calibration width
  nearest_key      best default-launch chunk of the nearest E4 calibration key (Triton only)
  best_static      the single chunk with the lowest mean regret over all E12 shapes of the family

Also reported: how often the oracle at a shard differs from the oracle at the full width, and
the stability of the per-round argmin (share of shapes whose rounds agree).

Output: ``e12_analysis_<tag>.json``.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.tpds import cost_model as CM
from src.experiments.tpds.common import MAMBA1_CHUNKS, RESULTS_DIR, read_jsonl, t_crit, write_json
from src.experiments.tpds.e12_tp_widths import MODEL_WIDTHS, REF_CHUNKS

TP_DEGREES = (1, 2, 4, 8)


def load(path: Path) -> pd.DataFrame:
    rows = [r for r in read_jsonl(path, kind="timing")]
    recs = []
    for r in rows:
        for c, cell in r["cells"].items():
            recs.append({"family": r["family"], "d": r["d"], "L": r["L"], "batch": r["batch"], "round": r["round"],
                         "chunk": int(c), "mean_ms": cell["mean_ms"]})
    return pd.DataFrame(recs)


def shape_table(df: pd.DataFrame, chunks: list[int]) -> pd.DataFrame:
    """One row per shape: round-mean latency per chunk, oracle, tie set, round agreement."""
    out = []
    for (d, L, b), g in df.groupby(["d", "L", "batch"]):
        piv = g.pivot_table(index="round", columns="chunk", values="mean_ms")
        piv = piv[[c for c in chunks if c in piv.columns]]
        if piv.shape[1] != len(chunks):
            continue
        n = len(piv)
        mu = piv.mean(axis=0).to_numpy()
        if n > 1:
            half = t_crit(n - 1) * piv.std(axis=0, ddof=1).to_numpy() / math.sqrt(n)
        else:
            half = np.zeros(len(chunks))
        o = int(mu.argmin())
        tied = [chunks[i] for i in range(len(chunks)) if mu[i] - half[i] <= mu[o] + half[o]]
        per_round = piv.to_numpy().argmin(1)
        out.append({"d": int(d), "L": int(L), "batch": int(b), "rounds": n, "lat": mu, "half": half,
                    "oracle": chunks[o], "tied": tied, "rounds_agree": bool((per_round == per_round[0]).all()),
                    "spread": float(mu.max() / mu.min())})
    return pd.DataFrame(out)


def regret(tab: pd.DataFrame, chunks: list[int], choice: dict) -> dict:
    """choice: {(d, L, b): chunk}. Mean / p95 regret and tied share over the shapes it covers."""
    reg, tie = [], []
    for r in tab.itertuples():
        c = choice.get((r.d, r.L, r.batch))
        if c is None:
            continue
        lat = r.lat[chunks.index(c)]
        reg.append(lat / r.lat.min() - 1)
        tie.append(c in r.tied)
    if not reg:
        return {"n": 0}
    reg = np.array(reg)
    return {"n": int(len(reg)), "mean_regret_pct": float(100 * reg.mean()),
            "p95_regret_pct": float(100 * np.percentile(reg, 95)), "max_regret_pct": float(100 * reg.max()),
            "tied_share": float(np.mean(tie))}


def triton_model_choice(tab: pd.DataFrame, chunks: list[int], host: str, in_dir: Path) -> dict:
    hw = CM.HARDWARE[host]
    stats = CM.load_kernel_stats(in_dir, hw["arch"])
    import json
    p = json.loads((in_dir / f"cost_model_{host}.json").read_text())["params"]
    theta = np.array([p[k] for k in CM.PARAMS], float)
    rows = []
    for r in tab.itertuples():
        for c in chunks:
            bd, nw = CM.DEFAULT_LAUNCH[c]
            rows.append({"key": (r.d, r.L, r.batch), "L": r.L, "B": r.batch, "d": r.d, "C": c, "BD": bd, "nw": nw,
                         "ns": 1})
    cand = pd.DataFrame(rows)
    f = CM.features(cand, stats, hw)
    pred = pd.Series(np.inf, index=cand.index)
    pred[f.index] = CM.predict(theta, f)
    cand["pred"] = pred
    return {k: int(g.loc[g["pred"].idxmin(), "C"]) for k, g in cand.groupby("key")}


def ref_model_choice(tab: pd.DataFrame, chunks: list[int], host: str, in_dir: Path) -> dict:
    import json
    hw = CM.HARDWARE[host]
    p = json.loads((in_dir / f"cost_model_refscan_{host}.json").read_text())["params"]
    th = np.array([p[k] for k in ("t0", "tau_f", "tau_m", "tau_e", "tau_s", "tau_x", "W0", "W1")], float)
    C = np.array(chunks)
    out = {}
    for r in tab.itertuples():
        pred = CM.ref_predict(th, np.full(len(C), r.L), np.full(len(C), r.batch), np.full(len(C), r.d), C, hw)
        out[(r.d, r.L, r.batch)] = int(C[int(np.argmin(pred))])
    return out


def nearest_key_choice(tab: pd.DataFrame, chunks: list[int], host: str, in_dir: Path) -> dict | None:
    try:
        from src.experiments.tpds.cost_model_baselines import calibration_keys, nearest_key
    except ImportError:
        return None
    keys = calibration_keys(host, in_dir, chunks)
    if not keys:
        return None
    out = {}
    for r in tab.itertuples():
        k = keys[nearest_key(r.L, r.batch, r.d, keys)]
        out[(r.d, r.L, r.batch)] = chunks[int(np.argmin(k["mean_ms"]))]
    return out


def table_choice(tab: pd.DataFrame, width_of) -> dict:
    """Serve each shape with the measured oracle at width width_of(d), same (L, batch)."""
    ora = {(r.d, r.L, r.batch): r.oracle for r in tab.itertuples()}
    out = {}
    for r in tab.itertuples():
        src = width_of(r.d)
        if src is not None and (src, r.L, r.batch) in ora:
            out[(r.d, r.L, r.batch)] = ora[(src, r.L, r.batch)]
    return out


def shard_map(widths: set[int]) -> dict:
    """(model, t) for every width that is a shard d_full / t of a checkpoint."""
    m = {}
    for model, full in MODEL_WIDTHS.items():
        for t in TP_DEGREES:
            if full % t == 0 and full // t in widths:
                m.setdefault(full // t, []).append((model, t))
    return m


def analyze_family(df: pd.DataFrame, fam: str, host: str, in_dir: Path) -> dict:
    chunks = list(MAMBA1_CHUNKS) if fam == "triton" else list(REF_CHUNKS)
    tab = shape_table(df[df["family"] == fam], chunks)
    if tab.empty:
        return {}
    widths = set(tab["d"])
    sm = shard_map(widths)
    selectors = {}
    try:
        selectors["cost_model"] = (triton_model_choice if fam == "triton" else ref_model_choice)(tab, chunks, host, in_dir)
    except (FileNotFoundError, KeyError) as exc:
        selectors["cost_model"] = None
        print(f"cost model unavailable for {fam}: {exc!r}")
    if fam == "triton":
        selectors["nearest_key"] = nearest_key_choice(tab, chunks, host, in_dir)
    selectors["width2048_table"] = table_choice(tab, lambda d: 2048)
    # static: the chunk with the lowest mean regret over all shapes
    regs = {c: np.mean([r.lat[chunks.index(c)] / r.lat.min() - 1 for r in tab.itertuples()]) for c in chunks}
    best_static = min(regs, key=regs.get)
    selectors["best_static"] = {(r.d, r.L, r.batch): best_static for r in tab.itertuples()}
    res = {"chunks": chunks, "n_shapes": int(len(tab)), "best_static_chunk": int(best_static),
           "round_agreement": float(tab["rounds_agree"].mean()),
           "spread_median": float(tab["spread"].median()), "by_model": {}}
    for model, full in MODEL_WIDTHS.items():
        per_t = {}
        for t in TP_DEGREES:
            d = full // t
            if full % t or d not in widths:
                continue
            sub = tab[tab["d"] == d]
            sel = dict(selectors)
            sel["unsharded_table"] = table_choice(sub, lambda _d, full=full: full) if t > 1 else None
            entry = {"d": int(d), "oracle_distribution": {int(k): int(v) for k, v in
                                                        sub["oracle"].value_counts().items()}}
            if t > 1 and full in widths:
                fo = {(r.L, r.batch): r.oracle for r in tab[tab["d"] == full].itertuples()}
                diff = [r.oracle != fo.get((r.L, r.batch)) and fo.get((r.L, r.batch)) not in r.tied
                        for r in sub.itertuples() if (r.L, r.batch) in fo]
                entry["share_oracle_moves_beyond_ties"] = float(np.mean(diff)) if diff else None
            for name, ch in sel.items():
                if ch is None:
                    continue
                entry[name] = regret(sub, chunks, ch)
            per_t[f"t{t}"] = entry
        res["by_model"][model] = per_t
    # pooled over all shards (t > 1)
    shards = {d for d, lst in sm.items() if any(t > 1 for _, t in lst)}
    pooled = tab[tab["d"].isin(shards)]
    res["pooled_shards"] = {name: regret(pooled, chunks, ch) for name, ch in selectors.items() if ch is not None}
    unsh = {}
    for model, full in MODEL_WIDTHS.items():
        for t in TP_DEGREES[1:]:
            d = full // t
            if full % t == 0 and d in widths and full in widths:
                unsh.update(table_choice(tab[tab["d"].isin([d, full])], lambda _d, full=full, d=d: full if _d == d else None))
    res["pooled_shards"]["unsharded_table"] = regret(pooled, chunks, unsh)
    res["per_shape"] = [{"d": int(r.d), "L": int(r.L), "batch": int(r.batch), "oracle": int(r.oracle),
                         "tied": [int(x) for x in r.tied], "rounds_agree": bool(r.rounds_agree),
                         "lat_ms": [float(x) for x in r.lat],
                         **{f"choice_{n}": (int(ch[(r.d, r.L, r.batch)]) if ch and (r.d, r.L, r.batch) in ch else None)
                            for n, ch in selectors.items()}} for r in tab.itertuples()]
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="vgpu3090", help="host of the E12 run and of the cost-model parameters")
    p.add_argument("--tag", default=None, help="E12 file tag (default: host)")
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--params-dir", type=Path, default=RESULTS_DIR, help="where cost_model_<host>.json etc. live")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv)
    tag = args.tag or args.host
    df = load(args.in_dir / f"e12_tp_widths_{tag}.jsonl")
    res = {"experiment": "E12-analysis", "host": args.host, "tag": tag, "families": {}}
    for fam in sorted(df["family"].unique()):
        res["families"][fam] = analyze_family(df, fam, args.host, args.params_dir)
    write_json(args.out_dir / f"e12_analysis_{tag}.json", res)
    for fam, r in res["families"].items():
        if not r:
            continue
        print(f"== {fam}: {r['n_shapes']} shapes, rounds agree {100 * r['round_agreement']:.0f}%, "
              f"median spread {r['spread_median']:.2f}x, best static {r['best_static_chunk']}")
        for name, st in r["pooled_shards"].items():
            if st.get("n"):
                print(f"   shards pooled  {name:16s} mean {st['mean_regret_pct']:6.2f}%  p95 {st['p95_regret_pct']:6.1f}%"
                      f"  tied {100 * st['tied_share']:5.1f}%  (n={st['n']})")


if __name__ == "__main__":
    main()
