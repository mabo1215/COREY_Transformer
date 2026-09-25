"""E1 analysis - is the chunk decision (and the latency) independent of input values?

Reads ``e1_<tag>.json`` and writes ``e1_analysis_<tag>.json`` with, per kernel family,
sequence length and perturbation mode:

decision  - the input-agnostic best chunk (median over all inputs and rounds) and how often it is
            the best chunk of an individual (input, round) cell: exactly (argmin agreement) and
            within noise (its 95% CI overlaps the CI of the cell's best chunk), plus the latency
            regret of the input-agnostic chunk against each input's own best chunk
            (median over rounds, so a single-round latency spike does not dominate).
effects   - at every fixed chunk, a two-way decomposition of the per-cell mean latency into input,
            round and residual sums of squares, an F test of the input effect with rounds as
            blocks, Benjamini-Hochberg adjusted p-values, and the input range in percent.
null      - (only when the run used ``--extra-seeds``) the between-distribution range compared with
            the range between redraws of one distribution, which is the null spread for inputs
            that differ in value but not in distribution.
clock     - (only when the run used ``--record-gpu-state``) the correlation of latency with the SM
            clock and the input F test after regressing out the clock.
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.experiments.tpds.common import RESULTS_DIR, write_json

try:
    from scipy.stats import f as f_dist
except ImportError:  # pragma: no cover - scipy ships with scikit-learn
    f_dist = None


def _bh(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    running = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        running = min(running, pvals[i] * n / rank)
        adj[i] = running
    return adj


def _f_pvalue(F: float, d1: int, d2: int) -> float | None:
    if f_dist is None or not np.isfinite(F):
        return 0.0 if F == float("inf") else None
    return float(1.0 - f_dist.cdf(F, d1, d2))


def _base_dist(name: str) -> str:
    return name.partition("#s")[0]


def _sm_mhz(row: dict) -> float | None:
    vals = [s.get("sm_mhz") for s in (row.get("gpu_state_before"), row.get("gpu_state_after")) if s]
    vals = [float(v) for v in vals if isinstance(v, (int, float))]
    return statistics.fmean(vals) if vals else None


def decision_analysis(rows: list[dict]) -> list[dict]:
    cells: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        cells[(r["family"], r["L"], r["mode"], r["input"], r["round"])][r["chunk"]] = r
    groups: dict[tuple, list] = defaultdict(list)
    for (fam, L, mode, inp, rnd), per_chunk in cells.items():
        groups[(fam, L, mode)].append((inp, rnd, per_chunk))
    out = []
    for (fam, L, mode), members in sorted(groups.items()):
        chunks = sorted(set.intersection(*[set(m[2]) for m in members]))
        agg = {c: statistics.median(m[2][c]["mean_ms"] for m in members) for c in chunks}
        c_star = min(agg, key=agg.get)
        exact = tied = 0
        for _, _, pc in members:
            best = min(chunks, key=lambda c: pc[c]["mean_ms"])
            exact += best == c_star
            tied += pc[c_star]["ci95_lo_ms"] <= pc[best]["ci95_hi_ms"]
        # regret of the input-agnostic chunk against each input's own best chunk (median over rounds)
        per_input: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
        for inp, _, pc in members:
            for c in chunks:
                per_input[inp][c].append(pc[c]["mean_ms"])
        regrets, own_best = [], {}
        for inp, by_c in per_input.items():
            means = {c: statistics.median(v) for c, v in by_c.items()}  # robust to single-round spikes
            b = min(means, key=means.get)
            own_best[inp] = b
            regrets.append(means[c_star] / means[b] - 1.0)
        out.append({
            "family": fam, "L": L, "mode": mode, "chunks": chunks, "n_cells": len(members),
            "input_agnostic_best_chunk": c_star,
            "between_chunk_max_over_min": max(agg.values()) / min(agg.values()),
            "argmin_agreement": exact / len(members), "tie_aware_agreement": tied / len(members),
            "own_best_chunk_per_input": own_best,
            "inputs_with_other_best_chunk": sorted(i for i, b in own_best.items() if b != c_star),
            "regret_of_input_agnostic_chunk_pct": {"mean": 100 * statistics.fmean(regrets),
                                                   "max": 100 * max(regrets)},
        })
    return out


def _two_way(X: np.ndarray) -> tuple[float, float, float, float | None]:
    """Inputs x rounds table -> (ss_input share, ss_round share, F_input, p)."""
    I, J = X.shape
    gm = X.mean()
    im, rm = X.mean(axis=1), X.mean(axis=0)
    ssi = J * ((im - gm) ** 2).sum()
    ssr = I * ((rm - gm) ** 2).sum()
    sse = ((X - im[:, None] - rm[None, :] + gm) ** 2).sum()
    tot = ssi + ssr + sse
    d1, d2 = I - 1, (I - 1) * (J - 1)
    F = (ssi / d1) / (sse / d2) if sse > 0 else float("inf")
    return ssi / tot if tot else 0.0, ssr / tot if tot else 0.0, F, _f_pvalue(F, d1, d2)


def effect_analysis(rows: list[dict]) -> list[dict]:
    tab: dict[tuple, dict] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        tab[(r["family"], r["L"], r["mode"], r["chunk"])][r["input"]][r["round"]] = r
    out = []
    for key, by_input in sorted(tab.items()):
        inputs = sorted(by_input)
        rounds = sorted(set.intersection(*[set(v) for v in by_input.values()]))
        if len(inputs) < 2 or len(rounds) < 2:
            continue
        X = np.array([[by_input[i][r]["mean_ms"] for r in rounds] for i in inputs])
        share_i, share_r, F, p = _two_way(X)
        means = X.mean(axis=1)
        rec = {"family": key[0], "L": key[1], "mode": key[2], "chunk": key[3], "n_inputs": len(inputs),
               "n_rounds": len(rounds), "ss_share_input": share_i, "ss_share_round": share_r, "F_input": F,
               "p_input": p, "input_range_pct": 100 * (means.max() - means.min()) / means.mean(),
               "fastest_input": inputs[int(means.argmin())], "slowest_input": inputs[int(means.argmax())]}
        # null control: spread between redraws of one distribution vs spread between distributions
        groups: dict[str, list[int]] = defaultdict(list)
        for k, i in enumerate(inputs):
            groups[_base_dist(i)].append(k)
        redraw = [g for g in groups.values() if len(g) > 1]
        if redraw:
            base_idx = [k for k, i in enumerate(inputs) if "#s" not in i]
            base_means = means[base_idx]
            rec["between_distribution_range_pct"] = 100 * (base_means.max() - base_means.min()) / base_means.mean()
            rec["between_redraw_range_pct_max"] = max(
                100 * (means[g].max() - means[g].min()) / means[g].mean() for g in redraw)
            rec["between_redraw_range_pct_median"] = float(np.median(
                [100 * (means[g].max() - means[g].min()) / means[g].mean() for g in redraw]))
        # clock: correlation with SM clock, and the input F test after regressing out the clock
        clocks = [[_sm_mhz(by_input[i][r]) for r in rounds] for i in inputs]
        if all(c is not None for row in clocks for c in row):
            C = np.array(clocks, dtype=float)
            y, x = X.ravel(), C.ravel()
            if x.std() > 0:
                rec["corr_latency_sm_clock"] = float(np.corrcoef(x, y)[0, 1])
                slope, icpt = np.polyfit(x, y, 1)
                share_i2, _, F2, p2 = _two_way((y - (slope * x + icpt) + y.mean()).reshape(X.shape))
                rec["F_input_clock_adjusted"], rec["p_input_clock_adjusted"] = F2, p2
            rec["sm_mhz_range"] = [float(x.min()), float(x.max())]
        out.append(rec)
    ps = [r["p_input"] for r in out if r["p_input"] is not None]
    adj = iter(_bh(ps)) if ps else iter(())
    for r in out:
        if r["p_input"] is not None:
            r["p_input_bh"] = next(adj)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--tag", required=True, help="E1 file tag, e.g. pro6000")
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args(argv)
    import json

    data = json.loads((args.in_dir / f"e1_{args.tag}.json").read_text())
    rows = [r for r in data["rows"] if r.get("kind") == "timing"]
    dec = decision_analysis(rows)
    eff = effect_analysis(rows)
    n_sig = sum(1 for r in eff if r.get("p_input_bh") is not None and r["p_input_bh"] < args.alpha)
    tot_cells = sum(d["n_cells"] for d in dec)
    res = {
        "experiment": "E1-analysis", "tag": args.tag, "source": f"e1_{args.tag}.json", "alpha": args.alpha,
        "meta": data.get("meta"),
        "overall": {
            "n_cells": tot_cells,
            "argmin_agreement": sum(d["argmin_agreement"] * d["n_cells"] for d in dec) / tot_cells,
            "tie_aware_agreement": sum(d["tie_aware_agreement"] * d["n_cells"] for d in dec) / tot_cells,
            "n_fixed_chunk_tests": len(eff), "n_significant_input_effects_bh": n_sig,
        },
        "decision": dec, "effects": eff,
    }
    out = args.in_dir / f"e1_analysis_{args.tag}.json"
    write_json(out, res)
    print(f"wrote {out}: tie-aware agreement {res['overall']['tie_aware_agreement']:.3f}, "
          f"{n_sig}/{len(eff)} significant input effects (BH, alpha={args.alpha})")


if __name__ == "__main__":
    main()
