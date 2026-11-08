"""Equivalence test of the whole-step null result, compositional bound, launch-floor model (review C5, M4; CPU only).

Reads stored result files and times nothing. Output: ``jpdc_step_bound.json`` in the results directory; every number is
regenerated from the raw JSONL / CSV tables with fixed seeds, and the stored E32 point estimates are asserted first.

Task 1 (C5): E32 whole-step graphs, graph mode, per (model, session): latency-weighted gain of the held-out table and of
the cross-session best chunk (tuned in one session, scored in the other) over the library default.
  * Estimator: T(shape, variant) = median over the five rounds of the round's replay median; gain = 100 * sum(T(default)
    - T(alt)) / sum(T(default)) over the 14 shapes (the stored definition). Pooled: T averaged over the two sessions.
    Cross-session pooled: mean of the two directions.
  * Hierarchical bootstrap: rounds resampled within each (session, shape, variant); sessions resampled where pooled (two
    sessions: four outcomes, so its between-session spread is a lower limit; the 1-df t procedure below is the
    conservative counterpart). The 14 shapes are a fixed design: the headline intervals are conditional on them and on
    their latency weights. A second version also resamples shapes. A third resamples rounds jointly per shape (paired).
  * TOST at +-0.5, +-1 and +-2% of step latency: equivalence at the 5% level iff the 90% percentile interval lies inside
    the margin; p_tost = max(share of replicates <= -m, share >= +m). ``equivalence_bound_pct`` is the smallest symmetric
    margin that the 90% interval fits in.
  * Smallest detectable effect at 80% power from the bootstrap between-round variance of the session gain and the
    between-session variance (sample variance of the two session gains minus the round part, truncated at zero).
  * Shapes where the table equals the default contribute exactly zero, by construction; that share is reported, as is
    the gain restricted to the other shapes, and each static chunk's gain as a positive control of assay sensitivity.

Task 2 (M4): under additivity T_step(C) = R + T_scan(C), with R = T_step(noscan) and s = 1 - R / T_step(default),
  step gain(C) = s * (1 - T_scan(C) / T_scan(default)).
With T_scan(C) := T_step(C) - T_step(noscan) this is an identity, so no test of the step data alone is possible. What is
tested is the substitution of isolated single-call replays for T_scan: is 1 - r(C), r(C) = sum over captured calls at the
same (L, B) of T_call(C) / sum of T_call(default), the scan-only gain inside the step? Inputs differ: another board (48 GB
RTX 3090; the stock-board subset of the E30 sessions has only one or two calls per shape), other tensors, a call replayed
alone. The single-call mix is the same for all shapes: every captured call at the exact (L, B), equal weight. s comes from
the other E32 session (no shared noise with the measured gain); the same-session value is a sensitivity.
  (a) additivity residual measured gain - s * (1 - r(C)) per (shape, chunk, session); absolute check k(C) = (T_step(C) -
      T_step(noscan)) / (scan layers * mean T_call(C)).
  (b) prediction of the gain of the table chunk and of every chunk; measured gain of the chunk the single calls prefer.
  (c) bound B = max(0, max_C s * (1 - r(C))) versus the measured best gain; violations strict and beyond the session
      noise, Spearman, mean absolute error, the same against a zero predictor; latency-weighted aggregates on the shapes
      with exact matches in all three models.

Task 3: same captured calls timed launch-inclusive (LI) and as graph replay on one board. Fits on (call, chunk) points
with leave-one-prompt-fold-out cross-validation (no prompt and no prompt cluster crosses folds; checked):
  linear      T_LI = a + b * T_graph
  bottleneck  T_LI = max(a, b * T_graph) + c   (flat level a + c; crossover T_graph = a / b; nests the linear model)
Relative least squares is the primary loss, absolute least squares a sensitivity. Compared with the heuristic floor
(twice the median oracle latency of the batch-1 calls with L <= 256) and scored on the prediction of ties (within 2% of
the call's oracle).

Usage:  python -X utf8 -m src.experiments.tpds.jpdc_step_bound [--n-boot N]
"""
from __future__ import annotations

import argparse
import json
import math
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import optimize, stats

from src.experiments.tpds.analysis import chunk_columns, load_e2, round_median_view
from src.experiments.tpds.common import RESULTS_DIR, dedupe_by_key, read_jsonl, write_json
from src.experiments.tpds.e3b_information import prompt_clusters

SEED = 20261015             # as taco_report_numbers
N_BOOT = 10000
MARGINS = (0.5, 1.0, 2.0)   # percent of step latency
TIE = 1.02                  # glossary: a tie is within 2% of the oracle
MEDIAN_OF_5_VAR = 0.2868    # variance of the median of five normal values, in units of the variance of one value
SESSIONS = ("4c_e32s1", "4c_e32s2")
# label: single-call sweep (three rounds, 48 GB RTX 3090; defines the held-out table), key prefix, default, printed gains
MODELS = {"ssd370m": ("e2_mamba2_vgpu3090_r4graph3.csv", "mamba2|", 256, (0.29, 0.18)),
          "ssd2p7b": ("e2_mamba2_vgpu3090_2p7b_r4graph3.csv", "mamba2|", 256, (-0.03, -0.04)),
          "gdn1p4b": ("e2_gdn_vgpu3090_1p4b_r4graph3.csv", "gdn|", 64, (-0.04, -0.02))}
E30_TAGS = ("4c_e30", "4c_e30s2")
E30_STATES = ("pinned", "retuned")
# launch-inclusive and graph timings of the same calls on one board
PAIRS = {"ssd370m_rtx3090": ("e2_mamba2_vgpu3090.csv", "e2_mamba2_vgpu3090_r4graph3.csv"),
         "m1_370m_rtx3090": ("e2_mamba1_vgpu3090_r4plain.csv", "e2_mamba1_vgpu3090_r4graph3.csv"),
         "ssd370m_rtx3090_same_driver": ("e2_mamba2_vgpu3090.csv", "e2_mamba2_vgpu3090_r4graph.csv"),
         "m1_370m_rtx3090_same_driver": ("e2_mamba1_vgpu3090_r4plain.csv", "e2_mamba1_vgpu3090_r4graph.csv"),
         "ssd370m_pro6000": ("e2_mamba2_pro6000.csv", "e2_mamba2_pro6000_taco_graph3.csv"),
         "m1_370m_pro6000": ("e2_mamba1_pro6000_r4plain.csv", "e2_mamba1_pro6000_taco_graph3.csv")}
# device-time tables, the only other timings of these checkpoints on the board: they cannot be paired
NOT_PAIRED = {"ssd2p7b": "e2_mamba2_vgpu3090_2p7b_r3", "gdn1p4b": "e2_gdn_vgpu3090_1p4b_r3"}


# CRC-32 of the seed keys that contain a pair name, stored rather than computed: the pair names carry
# GPU labels, and a key's random stream must not change when a GPU is labelled differently (for
# example in a renamed copy of the results). Equal to zlib.crc32 of each key as written here.
PAIR_SEED_CRC = {
    "t3_clusters|ssd370m_rtx3090": 0x66fdbaad,
    "t3_clusters|m1_370m_rtx3090": 0x8f8b5c47,
    "t3_clusters|ssd370m_rtx3090_same_driver": 0xbd77b850,
    "t3_clusters|m1_370m_rtx3090_same_driver": 0xbd97f852,
    "t3_clusters|ssd370m_pro6000": 0x13e74870,
    "t3_clusters|m1_370m_pro6000": 0xfa91ae9a,
    "t3_ceiling|ssd370m_rtx3090": 0x96fc2537,
    "t3_ceiling|m1_370m_rtx3090": 0x7f8ac3dd,
    "t3_ceiling|ssd370m_rtx3090_same_driver": 0x08447f1b,
    "t3_ceiling|m1_370m_rtx3090_same_driver": 0x08a43f19,
    "t3_ceiling|ssd370m_pro6000": 0xe3e6d7ea,
    "t3_ceiling|m1_370m_pro6000": 0x0a903100,
}


def make_rng(*parts) -> np.random.Generator:
    key = "|".join(map(str, parts))
    crc = PAIR_SEED_CRC.get(key)
    if crc is None:
        crc = zlib.crc32(key.encode())
    return np.random.default_rng(np.random.SeedSequence([SEED, crc]))


def _sig(x: float, digits: int = 6) -> float | None:
    x = float(x)
    return float(f"{x:.{digits}g}") if math.isfinite(x) else None


def _clean(obj):
    """JSON-safe copy; floats to six significant digits, non-finite floats to None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _sig(obj)
    return obj


def shape_name(L: int, B: int) -> str:
    return f"{L}x{B}"


# --------------------------------------------------------------------------- E32 data
class StepData:
    """Round medians of one model: ``y[session, shape, variant, round]``; ``T`` is the median over rounds."""

    def __init__(self, label: str, in_dir: Path):
        self.label = label
        self.default = MODELS[label][2]
        self.layers: list[int] = []
        rows_by_session = []
        for ses in SESSIONS:
            path = in_dir / f"e32_model_graph_{ses}_{label}.jsonl"
            meta = read_jsonl(path, kind="meta")[-1]
            self.layers.append(int(meta["model_info"]["n_scan_layers"]))
            rows = [r for r in dedupe_by_key(read_jsonl(path)) if r.get("kind") == "timing" and r["mode"] == "graph"]
            if not all(r.get("chunk_ok") for r in rows):
                raise SystemExit(f"{path.name}: a capture did not run at the intended chunk")
            rows_by_session.append(rows)
        self.shapes = sorted({(r["L"], r["batch"]) for r in rows_by_session[0]}, key=lambda s: (s[1], s[0]))
        self.chunks = sorted({int(r["variant"][1:]) for r in rows_by_session[0] if r["variant"] != "noscan"})
        self.variants = [f"c{c}" for c in self.chunks] + ["noscan"]
        n_rounds = max(r["round"] for r in rows_by_session[0]) + 1
        self.y = np.full((2, len(self.shapes), len(self.variants), n_rounds), np.nan)
        spos = {s: i for i, s in enumerate(self.shapes)}
        vpos = {v: i for i, v in enumerate(self.variants)}
        for j, rows in enumerate(rows_by_session):
            for r in rows:
                self.y[j, spos[(r["L"], r["batch"])], vpos[r["variant"]], r["round"]] = r["t"]["median_ms"]
        if np.isnan(self.y).any():
            raise SystemExit(f"{label}: missing cells in E32")
        self.T = np.median(self.y, axis=3)
        self.d = vpos[f"c{self.default}"]
        self.ns = vpos["noscan"]
        self.chunk_idx = np.array([vpos[f"c{c}"] for c in self.chunks])
        self.vpos = vpos
        self.ai = np.arange(len(self.shapes))


def draw_rounds(y: np.ndarray, n: int, rng: np.random.Generator, paired: bool = False) -> np.ndarray:
    """Medians of resampled rounds: ``y`` (shapes, variants, rounds) to (n, shapes, variants)."""
    S, V, R = y.shape
    out = np.empty((n, S, V))
    for s in range(S):
        idx = rng.integers(0, R, size=(n, R))
        for v in range(V):
            if not paired and v:
                idx = rng.integers(0, R, size=(n, R))
            out[:, s, v] = np.median(y[s, v][idx], axis=1)
    return out


def lw_gain(td: np.ndarray, tc: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Latency-weighted gain (percent) over the last axis."""
    if w is None:
        return 100.0 * (td - tc).sum(-1) / td.sum(-1)
    return 100.0 * (w * (td - tc)).sum(-1) / (w * td).sum(-1)


def summ(reps: np.ndarray, point: float) -> dict:
    lo95, hi95, lo90, hi90 = np.percentile(reps, [2.5, 97.5, 5, 95])
    return {"point": point, "boot_se": float(reps.std(ddof=1)), "ci95": [lo95, hi95], "ci90": [lo90, hi90]}


def tost_boot(reps: np.ndarray) -> dict:
    lo, hi = np.percentile(reps, [5, 95])
    per = {str(m): {"p_tost": max(float(np.mean(reps <= -m)), float(np.mean(reps >= m))),
                    "equivalent": bool(lo > -m and hi < m)} for m in MARGINS}
    return {"margins": per, "equivalence_bound_pct": max(abs(lo), abs(hi)),
            "smallest_margin_pct": next((m for m in MARGINS if per[str(m)]["equivalent"]), None)}


def tost_session_t(g1: float, g2: float, se_floor: float) -> dict:
    """TOST with the two sessions as the sample (t, 1 df); the SE never below the round-only SE of the pooled gain."""
    m = 0.5 * (g1 + g2)
    se = max(0.5 * abs(g1 - g2), se_floor)
    tc = float(stats.t.ppf(0.95, 1))
    lo, hi = m - tc * se, m + tc * se
    per = {str(d): {"p_tost": max(float(stats.t.sf((m + d) / se, 1)), float(stats.t.sf((d - m) / se, 1))),
                    "equivalent": bool(lo > -d and hi < d)} for d in MARGINS}
    return {"mean_of_session_gains": m, "se": se, "se_from_session_gap": 0.5 * abs(g1 - g2), "se_floor_from_rounds": se_floor,
            "t_crit_0p95_df1": tc, "ci90": [lo, hi], "equivalence_bound_pct": max(abs(lo), abs(hi)), "margins": per,
            "smallest_margin_pct": next((d for d in MARGINS if per[str(d)]["equivalent"]), None)}


def power_constants() -> dict:
    """Multipliers of the SE: effect vs zero (two-sided 5%) and TOST margin (5%), 80% power at a true zero for TOST."""
    z = stats.norm.ppf
    tc2 = float(stats.t.ppf(0.975, 1))
    tc1 = float(stats.t.ppf(0.95, 1))
    # two sessions, df = 1: t = (Z + ncp) / |Z'|, simulated because scipy's noncentral t is not finite at df = 1 here
    rng = make_rng("power_constants")
    zz, zp = rng.standard_normal(2_000_000), np.abs(rng.standard_normal(2_000_000))
    ncp = optimize.brentq(lambda c: float(np.mean(np.abs(zz + c) > tc2 * zp)) - 0.8, tc2, 500.0)
    tost_t1 = float(np.percentile(np.abs(zz) + tc1 * zp, 80))
    return {"mde_normal": float(z(0.975) + z(0.8)), "tost_margin_normal": float(z(0.95) + z(0.9)),
            "mde_t_df1": float(ncp), "tost_margin_t_df1": tost_t1}


def mde_block(g1: float, g2: float, v_round: list[float], k: dict) -> dict:
    v_e = float(np.mean(v_round))
    s2 = 0.5 * (g1 - g2) ** 2
    v_u = max(0.0, s2 - v_e)
    se_one = math.sqrt(v_e)
    se_two_rounds = math.sqrt(v_e / 2)
    se_two_total = math.sqrt((v_e + v_u) / 2)
    return {"variance_between_rounds_of_session_gain_pp2": v_round, "sample_variance_of_session_gains_pp2": s2,
            "variance_between_sessions_pp2": v_u, "se_one_session": se_one, "se_two_sessions_rounds_only": se_two_rounds,
            "se_two_sessions_with_session_variance": se_two_total,
            "mde_80pct_one_session": k["mde_normal"] * se_one,
            "mde_80pct_two_sessions_rounds_only": k["mde_normal"] * se_two_rounds,
            "mde_80pct_two_sessions_with_session_variance": k["mde_normal"] * se_two_total,
            "mde_80pct_two_sessions_t_df1": k["mde_t_df1"] * se_two_total,
            "tost_margin_80pct_two_sessions_with_session_variance": k["tost_margin_normal"] * se_two_total,
            "tost_margin_80pct_two_sessions_t_df1": k["tost_margin_t_df1"] * se_two_total}


# --------------------------------------------------------------------------- held-out table (as e31_divergence.load_table)
def length_bucket(L: int) -> int:
    return 1 << max(0, int(math.floor(math.log2(max(int(L), 1)))))


def load_sweep(path: Path):
    df = load_e2([path])
    if "oracle_chunk_med" in df.columns:
        df = round_median_view(df)
    return df.reset_index(drop=True)


def held_out_table(df, prefix: str, batch: int) -> dict[int, int]:
    sub = df[(df["batch"] == batch) & (df["layer"] >= 0) & df["key"].astype(str).str.startswith(prefix)]
    chunks = chunk_columns(sub)
    out = {}
    for b, g in sub.groupby("L_bucket"):
        full = [c for c in chunks if g[f"lat_c{c}_mean"].notna().all()] or chunks
        means = {c: float(g[f"lat_c{c}_mean"].mean()) for c in full}
        out[int(b)] = min(means, key=lambda c: (means[c], c))
    return out


def table_chunk(table: dict[int, int], L: int) -> int:
    b = length_bucket(L)
    return table[b] if b in table else table[min(table, key=lambda x: (abs(math.log2(x) - math.log2(b)), x))]


# --------------------------------------------------------------------------- reproduction of the stored estimates
def reproduce(st: StepData, tidx: np.ndarray, stored: dict, printed: tuple[float, float]) -> dict:
    """Stored gains, on the five significant digits the stored summaries were computed from; raises on a mismatch."""
    label = st.label
    tags = [f"{s}_{label}" for s in SESSIONS]
    summ_s = stored["summary"][label]["graph"]
    out = {}
    rd = np.vectorize(lambda x: float(f"{x:.5g}"))
    gains = []
    for j, tag in enumerate(tags):
        T5 = rd(st.T[j])
        g5 = float(lw_gain(T5[st.ai, st.d], T5[st.ai, tidx]))
        g = float(lw_gain(st.T[j, st.ai, st.d], st.T[j, st.ai, tidx]))
        val = summ_s[tag]["gain_table_latency_weighted_pct"]
        ok = abs(g5 - val) < 1.5e-3 and round(g, 2) == printed[j]
        if not ok:
            raise SystemExit(f"cannot reproduce the stored estimate: {label} {tag}: {g} / {g5}, stored {val}, printed {printed[j]}")
        share = 100.0 * (st.T[j, :, st.d] - st.T[j, :, st.ns]).sum() / st.T[j, :, st.d].sum()
        if abs(share - summ_s[tag]["scan_share_latency_weighted_pct"]) > 5e-3:
            raise SystemExit(f"cannot reproduce the stored scan share: {label} {tag}: {share}")
        out[f"s{j + 1}"] = {"table_gain_lw_pct": g, "stored": val, "printed": printed[j], "scan_share_lw_pct": share,
                            "stored_scan_share_lw_pct": summ_s[tag]["scan_share_latency_weighted_pct"]}
        gains.append(g)
    # cross-session transferred gain, both directions
    xfer = []
    for a, b in ((0, 1), (1, 0)):
        best = st.chunk_idx[np.argmin(st.T[a][:, st.chunk_idx], axis=1)]
        xfer.append(float(lw_gain(st.T[b, st.ai, st.d], st.T[b, st.ai, best])))
    val = summ_s["cross_session"]["in_session_best_table_transferred_gain_latency_weighted_pct"]
    if max(abs(x - v) for x, v in zip(xfer, val)) > 2e-3:
        raise SystemExit(f"cannot reproduce the stored cross-session gain: {label}: {xfer}, stored {val}")
    out["cross_session_s1_to_s2_s2_to_s1"] = xfer
    out["stored_cross_session"] = val
    return out


# --------------------------------------------------------------------------- task 1
def task1_model(st: StepData, tidx: np.ndarray, D: list, k: dict, n: int) -> dict:
    label, ai, d, S = st.label, st.ai, st.d, len(st.shapes)
    W = make_rng("t1_shapes", label).multinomial(S, np.full(S, 1.0 / S), size=n).astype(float)
    pick = make_rng("t1_sessions", label).integers(0, 2, size=(n, 2))
    Pd = [[draw_rounds(st.y[j], n, make_rng("t1_paired", label, j, q), paired=True) for q in (0, 1)] for j in (0, 1)]
    Tbar = st.T.mean(axis=0)
    trivial = tidx == d

    def pool(A, fixed=False):
        if fixed:
            return 0.5 * (A[0][0] + A[1][0])
        a0 = np.where((pick[:, 0] == 0)[:, None, None], A[0][0], A[1][0])
        a1 = np.where((pick[:, 1] == 0)[:, None, None], A[0][1], A[1][1])
        return 0.5 * (a0 + a1)

    pooled = pool(D)
    pooled_fixed = pool(D, fixed=True)
    pooled_paired = pool(Pd)

    def gain_block(ti, mask=None) -> dict:
        """Table-like policy: chunk variant index per shape."""
        m = np.ones(S, bool) if mask is None else mask
        sel = np.flatnonzero(m)

        W_sel = W[:, sel] if mask is None else make_rng("t1_shapes_subset", label).multinomial(
            len(sel), np.full(len(sel), 1.0 / len(sel)), size=n).astype(float)

        def g(A):
            return lw_gain(A[:, sel, d], A[:, sel, ti[sel]])

        def gw(A):
            return lw_gain(A[:, sel, d], A[:, sel, ti[sel]], W_sel)

        pts = [float(lw_gain(st.T[j, sel, d], st.T[j, sel, ti[sel]])) for j in (0, 1)]
        pt = float(lw_gain(Tbar[sel, d], Tbar[sel, ti[sel]]))
        rep = {}
        for j in (0, 1):
            reps = g(D[j][0])
            rep[f"s{j + 1}"] = {"rounds": summ(reps, pts[j]), "tost_rounds": tost_boot(reps),
                                "rounds_and_shapes": summ(gw(D[j][0]), pts[j]), "tost_rounds_and_shapes": tost_boot(gw(D[j][0])),
                                "rounds_paired": summ(g(Pd[j][0]), pts[j])}
        v_round = [rep[f"s{j + 1}"]["rounds"]["boot_se"] ** 2 for j in (0, 1)]
        se_floor = math.sqrt(float(np.mean(v_round)) / 2)
        rep["pooled"] = {"rounds_sessions": summ(g(pooled), pt), "tost_rounds_sessions": tost_boot(g(pooled)),
                         "rounds_only_sessions_fixed": summ(g(pooled_fixed), pt), "tost_rounds_only_sessions_fixed": tost_boot(g(pooled_fixed)),
                         "rounds_sessions_shapes": summ(gw(pooled), pt), "tost_rounds_sessions_shapes": tost_boot(gw(pooled)),
                         "rounds_sessions_paired": summ(g(pooled_paired), pt), "tost_session_t_df1": tost_session_t(pts[0], pts[1], se_floor)}
        rep["session_gap_pp"] = abs(pts[0] - pts[1])
        rep["variance"] = mde_block(pts[0], pts[1], v_round, k)
        return rep

    res = {"design": {}, "table": gain_block(tidx)}
    weights = st.T[:, :, d] / st.T[:, :, d].sum(axis=1, keepdims=True)
    res["design"] = {
        "shapes": [shape_name(*s) for s in st.shapes], "fixed_design_note": "the 14 shapes are a fixed design, not a sample",
        "table_chunk": {shape_name(*s): st.chunks[list(st.chunk_idx).index(tidx[i])] for i, s in enumerate(st.shapes)},
        "shapes_where_table_equals_default": [shape_name(*s) for i, s in enumerate(st.shapes) if trivial[i]],
        "latency_share_where_table_equals_default_pct": [100.0 * float(weights[j, trivial].sum()) for j in (0, 1)],
        "latency_weight_pct_session1": {shape_name(*s): 100.0 * float(weights[0, i]) for i, s in enumerate(st.shapes)},
        "largest_shape_weight_pct": [100.0 * float(weights[j].max()) for j in (0, 1)]}
    if (~trivial).any():
        res["table_on_shapes_where_it_differs"] = gain_block(tidx, ~trivial)

    # cross-session best chunk, tuned in one session and scored in the other, re-tuned in every replicate
    def xfer_reps(Ta, Tb, w=None):
        best = st.chunk_idx[np.argmin(Ta[:, :, st.chunk_idx], axis=2)]
        tc = Tb[np.arange(n)[:, None], ai[None, :], best]
        return lw_gain(Tb[:, :, d], tc, w)

    best_pt = [st.chunk_idx[np.argmin(st.T[j][:, st.chunk_idx], axis=1)] for j in (0, 1)]
    x_pt = {"s1_to_s2": float(lw_gain(st.T[1, ai, d], st.T[1, ai, best_pt[0]])),
            "s2_to_s1": float(lw_gain(st.T[0, ai, d], st.T[0, ai, best_pt[1]]))}
    x_rep = {"s1_to_s2": [xfer_reps(D[0][q], D[1][q]) for q in (0, 1)], "s2_to_s1": [xfer_reps(D[1][q], D[0][q]) for q in (0, 1)]}
    x_rep_w = {"s1_to_s2": xfer_reps(D[0][0], D[1][0], W), "s2_to_s1": xfer_reps(D[1][0], D[0][0], W)}
    cross = {}
    for name in x_pt:
        cross[name] = {"rounds": summ(x_rep[name][0], x_pt[name]), "tost_rounds": tost_boot(x_rep[name][0]),
                       "rounds_and_shapes": summ(x_rep_w[name], x_pt[name]), "tost_rounds_and_shapes": tost_boot(x_rep_w[name])}
    pt = 0.5 * (x_pt["s1_to_s2"] + x_pt["s2_to_s1"])
    fixed = 0.5 * (x_rep["s1_to_s2"][0] + x_rep["s2_to_s1"][0])
    d0 = np.where(pick[:, 0] == 0, x_rep["s1_to_s2"][0], x_rep["s2_to_s1"][0])
    d1 = np.where(pick[:, 1] == 0, x_rep["s1_to_s2"][1], x_rep["s2_to_s1"][1])
    resampled = 0.5 * (d0 + d1)
    fixed_w = 0.5 * (x_rep_w["s1_to_s2"] + x_rep_w["s2_to_s1"])
    v_round = [float(np.var(x_rep[nm][0], ddof=1)) for nm in x_pt]
    cross["pooled"] = {"rounds_sessions": summ(resampled, pt), "tost_rounds_sessions": tost_boot(resampled),
                       "rounds_only_directions_fixed": summ(fixed, pt), "tost_rounds_only_directions_fixed": tost_boot(fixed),
                       "rounds_and_shapes_directions_fixed": summ(fixed_w, pt), "tost_rounds_and_shapes_directions_fixed": tost_boot(fixed_w),
                       "tost_session_t_df1": tost_session_t(x_pt["s1_to_s2"], x_pt["s2_to_s1"], math.sqrt(float(np.mean(v_round)) / 2))}
    cross["variance"] = mde_block(x_pt["s1_to_s2"], x_pt["s2_to_s1"], v_round, k)
    res["cross_session_best"] = cross

    # the in-session best, only to quantify how far its minimum is biased upward
    in_s = [float(lw_gain(st.T[j, ai, d], st.T[j, ai, best_pt[j]])) for j in (0, 1)]
    res["selection_bias_of_in_session_best_pp"] = {
        "note": "scored in one session: its own best chunks minus the chunks tuned in the other session; quantifies the "
                "bias of the in-session minimum and is not evidence of a gain",
        "scored_in_s1": in_s[0] - x_pt["s2_to_s1"], "scored_in_s2": in_s[1] - x_pt["s1_to_s2"]}

    # positive control: can this design see a chunk effect at all? each static chunk against the default
    ctrl = {}
    for c, v in zip(st.chunks, st.chunk_idx):
        if v == d:
            continue
        g1, g2 = float(lw_gain(st.T[0, ai, d], st.T[0, ai, v])), float(lw_gain(st.T[1, ai, d], st.T[1, ai, v]))
        v_r = [float(np.var(lw_gain(D[j][0][:, :, d], D[j][0][:, :, v]), ddof=1)) for j in (0, 1)]
        ctrl[f"c{c}"] = {"s1": g1, "s2": g2, "pooled": summ(lw_gain(pooled[:, :, d], pooled[:, :, v]), float(lw_gain(Tbar[:, d], Tbar[:, v]))),
                         "variance": mde_block(g1, g2, v_r, k)}
    res["positive_control_static_chunks"] = ctrl

    # delta-method cross-check of the bootstrap SE of the session gain, from the round standard deviations
    delta = []
    for j in (0, 1):
        var_t = MEDIAN_OF_5_VAR * st.y[j].var(axis=2, ddof=1)
        dsum, nsum = st.T[j, ai, d].sum(), st.T[j, ai, tidx].sum()
        v = 0.0
        for i in range(S):
            if tidx[i] == d:
                v += (1.0 / dsum - nsum / dsum ** 2) ** 2 * var_t[i, d]
            else:
                v += var_t[i, tidx[i]] / dsum ** 2 + (nsum / dsum ** 2) ** 2 * var_t[i, d]
        delta.append(100.0 * math.sqrt(v))
    res["table"]["delta_method_se_session"] = delta

    # per-shape
    per_shape = []
    for i, s in enumerate(st.shapes):
        ti = tidx[i]

        def sg(A):
            return 100.0 * (A[:, i, d] - A[:, i, ti]) / A[:, i, d]

        gs = [100.0 * float((st.T[j, i, d] - st.T[j, i, ti]) / st.T[j, i, d]) for j in (0, 1)]
        pt = 100.0 * float((Tbar[i, d] - Tbar[i, ti]) / Tbar[i, d])
        v_r = [float(np.var(sg(D[j][0]), ddof=1)) for j in (0, 1)]
        rec = {"shape": shape_name(*s), "L": s[0], "batch": s[1], "table_chunk": st.chunks[list(st.chunk_idx).index(ti)],
               "table_equals_default": bool(ti == d), "default_ms": [float(st.T[j, i, d]) for j in (0, 1)],
               "default_session_gap_pct": 100.0 * abs(float(st.T[0, i, d] / st.T[1, i, d]) - 1.0),
               "round_cv_default_pct": [100.0 * float(st.y[j, i, d].std(ddof=1) / st.y[j, i, d].mean()) for j in (0, 1)],
               "scan_share_pct": [100.0 * float((st.T[j, i, d] - st.T[j, i, st.ns]) / st.T[j, i, d]) for j in (0, 1)],
               "table_gain_pct": gs, "table_session_gap_pp": abs(gs[0] - gs[1]),
               "table_ci95": [summ(sg(D[j][0]), gs[j])["ci95"] for j in (0, 1)],
               "table_gain_pooled_pct": pt, "table_pooled_ci95": summ(sg(pooled), pt)["ci95"]}
        if ti != d:
            m = mde_block(gs[0], gs[1], v_r, k)
            rec["table_mde_80pct_two_sessions"] = m["mde_80pct_two_sessions_with_session_variance"]
            rec["table_mde_80pct_two_sessions_t_df1"] = m["mde_80pct_two_sessions_t_df1"]
        # cross-session best at this shape
        for name, (a, b) in (("s1_to_s2", (0, 1)), ("s2_to_s1", (1, 0))):
            c_star = int(best_pt[a][i])
            best = st.chunk_idx[np.argmin(D[a][0][:, i][:, st.chunk_idx], axis=1)]
            reps = 100.0 * (D[b][0][:, i, d] - D[b][0][np.arange(n), i, best]) / D[b][0][:, i, d]
            g = 100.0 * float((st.T[b, i, d] - st.T[b, i, c_star]) / st.T[b, i, d])
            rec[f"cross_best_{name}"] = {"chunk": st.chunks[list(st.chunk_idx).index(c_star)], "gain_pct": g, "ci95": summ(reps, g)["ci95"]}
        per_shape.append(rec)
    res["per_shape"] = per_shape
    res["largest_table_session_gap"] = max(per_shape, key=lambda r: r["table_session_gap_pp"])["shape"]
    res["default_session_gap_median_pct"] = float(np.median([r["default_session_gap_pct"] for r in per_shape]))
    res["table_gain_range_over_shapes_pct"] = [min(min(r["table_gain_pct"]) for r in per_shape), max(max(r["table_gain_pct"]) for r in per_shape)]
    return res


# --------------------------------------------------------------------------- single calls
class Calls:
    """Captured calls of one model at exact shapes: sums per chunk, ratio to the default, prompt-cluster bootstrap."""

    def __init__(self, df, default: int, label: str, n: int):
        self.df, self.default, self.label, self.n = df, default, label, n
        self.chunks = chunk_columns(df)
        self.lat = df[[f"lat_c{c}_mean" for c in self.chunks]].to_numpy(float)
        self.L, self.B = df["L"].to_numpy(), df["batch"].to_numpy()
        self.pos = {c: i for i, c in enumerate(self.chunks)}

    def at(self, L: int, B: int) -> dict | None:
        sel = np.flatnonzero((self.L == L) & (self.B == B))
        if not len(sel):
            return None
        lat = self.lat[sel]
        dj = self.pos[self.default]
        clusters, _ = prompt_clusters(self.df.iloc[sel].reset_index(drop=True))
        K = int(clusters.max()) + 1
        M = make_rng("t2_clusters", self.label, L, B).multinomial(K, np.full(K, 1.0 / K), size=self.n).astype(float)
        out = {"n_calls": int(len(sel)), "n_prompt_clusters": K, "mean_ms": {}, "scan_only_gain_pct": {}, "scan_only_gain_ci95": {}}
        reps_all = [np.zeros(self.n)]
        for c, j in self.pos.items():
            ok = np.isfinite(lat[:, j]) & np.isfinite(lat[:, dj])
            if not ok.any():
                continue
            out["mean_ms"][c] = float(np.mean(lat[ok, j]))
            num, den = np.zeros(K), np.zeros(K)
            np.add.at(num, clusters[ok], lat[ok, j])
            np.add.at(den, clusters[ok], lat[ok, dj])
            out["scan_only_gain_pct"][c] = 100.0 * (1.0 - num.sum() / den.sum())
            reps = 100.0 * (1.0 - (M @ num) / (M @ den))
            out["scan_only_gain_ci95"][c] = np.percentile(reps, [2.5, 97.5]).tolist()
            reps_all.append(reps)
        out["best_scan_only_gain_ci95"] = np.percentile(np.max(np.stack(reps_all), axis=0), [2.5, 97.5]).tolist()
        return out

    def nearest(self, L: int, B: int) -> tuple[int, dict] | None:
        lengths = np.unique(self.L[self.B == B])
        if not len(lengths):
            return None
        near = int(lengths[np.argmin(np.abs(np.log2(lengths) - math.log2(L)))])
        return near, self.at(near, B)


def load_e30(tag: str, label: str, state: str, in_dir: Path) -> dict[tuple[int, int], dict]:
    """Stock-board E30 subset: mean latency per chunk at each (L, B), one or two calls per shape."""
    rows = [r for r in read_jsonl(in_dir / f"e30_autotune_state_{tag}_{label}.jsonl") if r.get("kind") == "call"]
    by = defaultdict(list)
    for r in rows:
        med = r["med_ms"]
        by[(int(r["L"]), int(r["batch"]))].append({int(c): float(v) for c, v in med[state].items() if v is not None})
    out = {}
    for sh, calls in by.items():
        chunks = sorted(set.intersection(*(set(x) for x in calls)))
        out[sh] = {"n_calls": len(calls), "mean_ms": {c: float(np.mean([x[c] for x in calls])) for c in chunks}}
    return out


def ols(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0:
        return {"n": int(len(x))}
    b, a = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    return {"n": int(len(x)), "slope": float(b), "intercept": float(a), "r2": float(r ** 2)}


def spearman(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return {"rho": None, "p": None, "n": int(len(x))}
    rho, p = stats.spearmanr(x, y)
    return {"rho": float(rho), "p": float(p), "n": int(len(x))}


def err_block(pred, meas, noise=None) -> dict:
    pred, meas = np.asarray(pred, float), np.asarray(meas, float)
    res = meas - pred
    nz = (pred != 0) & (meas != 0)
    rec = {"n": int(len(res)), "mae_pp": float(np.abs(res).mean()), "max_abs_error_pp": float(np.abs(res).max()),
           "mean_error_measured_minus_predicted_pp": float(res.mean()), "mae_of_zero_predictor_pp": float(np.abs(meas).mean()),
           "sign_agreement_share_nonzero": float((np.sign(pred[nz]) == np.sign(meas[nz])).mean()) if nz.any() else None,
           "n_nonzero": int(nz.sum()), "spearman": spearman(pred, meas), "ols_measured_on_predicted": ols(pred, meas)}
    if noise is not None:
        noise = np.asarray(noise, float)
        rec["share_error_beyond_noise"] = float((np.abs(res) > noise).mean())
        rec["n_error_beyond_noise"] = int((np.abs(res) > noise).sum())
    return rec


# --------------------------------------------------------------------------- task 2
def task2_model(st: StepData, tidx: np.ndarray, calls: Calls, D: list, in_dir: Path, common: list[tuple[int, int]]) -> dict:
    label, d, ns = st.label, st.d, st.ns
    layers = st.layers[0]
    chunk_rows, shape_rows = [], []
    no_match, near_used = [], {}
    infos = {}
    for s in st.shapes:
        info = calls.at(*s)
        infos[s] = info
        if info is None:
            no_match.append(shape_name(*s))
            near = calls.nearest(*s)
            if near is not None:
                near_used[s] = near
    for i, s in enumerate(st.shapes):
        info = infos[s] or (near_used[s][1] if s in near_used else None)
        if info is None:
            continue
        exact = infos[s] is not None
        for j in (0, 1):
            o = 1 - j
            T, To = st.T[j, i], st.T[o, i]
            s_own = (T[d] - T[ns]) / T[d]
            s_oth = (To[d] - To[ns]) / To[d]
            A = D[j][0][:, i]
            G, Go, Gpred, half = {}, {}, {}, {}
            for c, v in zip(st.chunks, st.chunk_idx):
                G[c] = 100.0 * (T[d] - T[v]) / T[d]
                Go[c] = 100.0 * (To[d] - To[v]) / To[d]
                reps = 100.0 * (A[:, d] - A[:, v]) / A[:, d]
                half[c] = 0.5 * float(np.subtract(*np.percentile(reps, [97.5, 2.5])))
                if c in info["scan_only_gain_pct"]:
                    Gpred[c] = s_oth * info["scan_only_gain_pct"][c]
                    if c != st.default:
                        scan_step = T[v] - T[ns]
                        chunk_rows.append({
                            "shape": shape_name(*s), "exact_match": exact, "session": j + 1, "chunk": c,
                            "is_table_chunk": bool(v == tidx[i]), "measured_gain_pct": G[c], "predicted_gain_pct": Gpred[c],
                            "predicted_gain_same_session_share_pct": s_own * info["scan_only_gain_pct"][c],
                            "error_pp": G[c] - Gpred[c], "round_half_width_pp": half[c], "session_gap_pp": abs(G[c] - Go[c]),
                            "scan_only_gain_from_step_pct": G[c] / s_own, "scan_only_gain_from_calls_pct": info["scan_only_gain_pct"][c],
                            "k_step_scan_over_call_sum": scan_step / (layers * info["mean_ms"][c])})
            c_best = min(G, key=lambda c: (-G[c], c))
            c_other = min(Go, key=lambda c: (-Go[c], c))
            reps_best = np.max(100.0 * (A[:, [d]] - A[:, st.chunk_idx]) / A[:, [d]], axis=1)
            half_best = 0.5 * float(np.subtract(*np.percentile(reps_best, [97.5, 2.5])))
            gap_best = abs(G[c_best] - Go[c_best])
            bound = max(0.0, max(Gpred.values()))
            c_pred = min(Gpred, key=lambda c: (-Gpred[c], c))
            tc = st.chunks[list(st.chunk_idx).index(tidx[i])]
            ci = info["best_scan_only_gain_ci95"]
            shape_rows.append({
                "shape": shape_name(*s), "exact_match": exact, "matched_length": s[0] if exact else near_used[s][0], "session": j + 1,
                "n_calls": info["n_calls"], "n_prompt_clusters": info["n_prompt_clusters"], "default_ms": float(T[d]),
                "scan_share_pct": 100.0 * s_own, "scan_share_other_session_pct": 100.0 * s_oth,
                "scan_ms_step": float(T[d] - T[ns]), "scan_ms_call_sum": layers * info["mean_ms"][st.default],
                "k_default": float(T[d] - T[ns]) / (layers * info["mean_ms"][st.default]),
                "bound_pct": bound, "bound_ci95_call_mix": [s_oth * ci[0], s_oth * ci[1]], "bound_over_share": bound / (100.0 * s_oth),
                "single_call_best_chunk": c_pred, "measured_best_gain_pct": G[c_best], "measured_best_chunk": c_best,
                "measured_gain_of_single_call_best_chunk_pct": G.get(c_pred), "chunk_tuned_in_other_session": c_other,
                "measured_cross_session_best_gain_pct": G[c_other], "table_chunk": tc, "table_equals_default": bool(tc == st.default),
                "predicted_table_gain_pct": Gpred.get(tc), "measured_table_gain_pct": G[tc], "table_session_gap_pp": abs(G[tc] - Go[tc]),
                "table_round_half_width_pp": half[tc], "round_half_width_best_pp": half_best, "session_gap_best_pp": gap_best,
                "slack_bound_minus_best_pp": bound - G[c_best], "violation_strict": bool(G[c_best] > bound),
                "violation_beyond_session_gap": bool(G[c_best] - bound > gap_best),
                "violation_beyond_round_half_width": bool(G[c_best] - bound > half_best),
                "violation_beyond_session_noise": bool(G[c_best] - bound > max(gap_best, half_best)),
                "cross_session_violation_beyond_session_noise": bool(G[c_other] - bound > max(gap_best, half_best))})

    def rows_for(sel_rows, key, session=None, exact_only=True):
        return [r[key] for r in sel_rows if (session is None or r["session"] == session) and (r["exact_match"] or not exact_only)]

    out = {"shapes_without_exact_match": no_match,
           "nearest_length_used_for_secondary_rows": {shape_name(*s): shape_name(near_used[s][0], s[1]) for s in near_used},
           "scan_layers": layers, "per_shape": shape_rows, "per_shape_chunk": chunk_rows}

    # (a) additivity
    add = {}
    for ses in (None, 1, 2):
        rows = [r for r in chunk_rows if r["exact_match"] and (ses is None or r["session"] == ses)]
        noise = [max(r["round_half_width_pp"], r["session_gap_pp"]) for r in rows]
        block = err_block([r["predicted_gain_pct"] for r in rows], [r["measured_gain_pct"] for r in rows], noise)
        block["mae_using_same_session_share_pp"] = float(np.mean([abs(r["measured_gain_pct"] - r["predicted_gain_same_session_share_pct"]) for r in rows]))
        block["scan_only_gain_mae_pp_of_scan"] = float(np.mean([abs(r["scan_only_gain_from_step_pct"] - r["scan_only_gain_from_calls_pct"]) for r in rows]))
        k = np.array([r["k_step_scan_over_call_sum"] for r in rows])
        block["k_step_scan_over_call_sum"] = {"median": float(np.median(k)), "p5": float(np.percentile(k, 5)), "p95": float(np.percentile(k, 95))}
        add["all_sessions" if ses is None else f"s{ses}"] = block
    by_chunk = {}
    for c in st.chunks:
        rows = [r for r in chunk_rows if r["exact_match"] and r["chunk"] == c]
        if rows:
            by_chunk[f"c{c}"] = err_block([r["predicted_gain_pct"] for r in rows], [r["measured_gain_pct"] for r in rows],
                                          [max(r["round_half_width_pp"], r["session_gap_pp"]) for r in rows])
            by_chunk[f"c{c}"]["k_median"] = float(np.median([r["k_step_scan_over_call_sum"] for r in rows]))
    add["by_chunk"] = by_chunk
    add["k_default_by_shape_session"] = {f"{r['shape']}|s{r['session']}": r["k_default"] for r in shape_rows if r["exact_match"]}
    k_d = [r["k_default"] for r in shape_rows if r["exact_match"]]
    add["k_default_range"] = [min(k_d), max(k_d)]
    within = []
    for s in {r["shape"] for r in chunk_rows if r["exact_match"]}:
        for ses in (1, 2):
            k = [r["k_step_scan_over_call_sum"] for r in chunk_rows if r["shape"] == s and r["session"] == ses]
            if len(k) >= 2:
                within.append(np.std(k, ddof=1) / np.mean(k))
    add["k_within_shape_cv_across_chunks_median"] = float(np.median(within))
    out["additivity"] = add

    # (b) prediction
    pred = {}
    for ses in (None, 1, 2):
        rows = [r for r in shape_rows if r["exact_match"] and (ses is None or r["session"] == ses)]
        tab = [r for r in rows if r["predicted_table_gain_pct"] is not None]
        tab_diff = [r for r in tab if not r["table_equals_default"]]
        blk = {"table_chunk_all_shapes": err_block([r["predicted_table_gain_pct"] for r in tab], [r["measured_table_gain_pct"] for r in tab]),
               "table_chunk_shapes_where_it_differs": err_block([r["predicted_table_gain_pct"] for r in tab_diff], [r["measured_table_gain_pct"] for r in tab_diff],
                                                                [max(r["table_session_gap_pp"], r["table_round_half_width_pp"]) for r in tab_diff]) if tab_diff else None,
               "single_call_best_chunk": err_block([r["bound_pct"] for r in rows], [r["measured_gain_of_single_call_best_chunk_pct"] for r in rows]),
               "share_shapes_single_call_best_equals_measured_best": float(np.mean([r["single_call_best_chunk"] == r["measured_best_chunk"] for r in rows]))}
        pred["all_sessions" if ses is None else f"s{ses}"] = blk
    out["prediction"] = pred

    # (c) bound
    bnd = {}
    for ses in (None, 1, 2):
        rows = [r for r in shape_rows if r["exact_match"] and (ses is None or r["session"] == ses)]
        rec = err_block([r["bound_pct"] for r in rows], [r["measured_best_gain_pct"] for r in rows])
        rec["cross_session_best"] = err_block([r["bound_pct"] for r in rows], [r["measured_cross_session_best_gain_pct"] for r in rows])
        pos = [r for r in rows if r["bound_pct"] > 0]
        rec["where_bound_positive"] = {"spearman_in_session_best": spearman([r["bound_pct"] for r in pos], [r["measured_best_gain_pct"] for r in pos]),
                                       "spearman_cross_session_best": spearman([r["bound_pct"] for r in pos], [r["measured_cross_session_best_gain_pct"] for r in pos])}
        rec.update({"n_violations_strict": sum(r["violation_strict"] for r in rows),
                    "n_violations_beyond_session_gap": sum(r["violation_beyond_session_gap"] for r in rows),
                    "n_violations_beyond_round_half_width": sum(r["violation_beyond_round_half_width"] for r in rows),
                    "n_violations_beyond_session_noise": sum(r["violation_beyond_session_noise"] for r in rows),
                    "n_cross_session_violations_beyond_session_noise": sum(r["cross_session_violation_beyond_session_noise"] for r in rows),
                    "shapes_violating_strict": [f"{r['shape']}|s{r['session']}" for r in rows if r["violation_strict"]],
                    "shapes_violating_beyond_session_noise": [f"{r['shape']}|s{r['session']}" for r in rows if r["violation_beyond_session_noise"]],
                    "mean_bound_pct": float(np.mean([r["bound_pct"] for r in rows])), "mean_measured_best_pct": float(np.mean([r["measured_best_gain_pct"] for r in rows])),
                    "mean_measured_cross_session_best_pct": float(np.mean([r["measured_cross_session_best_gain_pct"] for r in rows])),
                    "mean_bound_over_scan_share": float(np.mean([r["bound_over_share"] for r in rows])),
                    "mean_scan_share_pct": float(np.mean([r["scan_share_pct"] for r in rows])),
                    "share_bound_zero": float(np.mean([r["bound_pct"] == 0.0 for r in rows])),
                    "max_excess_over_bound_pp": max(0.0, max(-r["slack_bound_minus_best_pp"] for r in rows))})
        bnd["all_sessions" if ses is None else f"s{ses}"] = rec
    out["bound_check"] = bnd

    # aggregates on the common mix (exact matches in all models) and on all 14 shapes with nearest lengths
    agg = {}
    names = {shape_name(*s) for s in common}
    for mix, keep in (("common_exact_shapes", lambda r: r["shape"] in names), ("all_14_shapes_nearest_length_for_unmatched", lambda r: True)):
        rec = {}
        for ses in (1, 2):
            rows = [r for r in shape_rows if r["session"] == ses and keep(r)]
            w = np.array([r["default_ms"] for r in rows])

            def wavg(key):
                vals = np.array([0.0 if r[key] is None else r[key] for r in rows])
                return float((w * vals).sum() / w.sum())

            rec[f"s{ses}"] = {"n_shapes": len(rows), "scan_share_pct": float(np.sum([r["scan_ms_step"] for r in rows]) / w.sum() * 100.0),
                              "bound_pct": wavg("bound_pct"), "measured_best_pct": wavg("measured_best_gain_pct"),
                              "measured_cross_session_best_pct": wavg("measured_cross_session_best_gain_pct"),
                              "predicted_table_pct": wavg("predicted_table_gain_pct"), "measured_table_pct": wavg("measured_table_gain_pct"),
                              "measured_gain_of_single_call_best_chunk_pct": wavg("measured_gain_of_single_call_best_chunk_pct")}
        agg[mix] = rec
    out["aggregate_latency_weighted"] = agg

    # stock-board subset (one or two calls per shape): the same check against the E30 sessions
    stock = {}
    for tag in E30_TAGS:
        for state in E30_STATES:
            if not (in_dir / f"e30_autotune_state_{tag}_{label}.jsonl").exists():
                continue
            e30 = load_e30(tag, label, state, in_dir)
            pts, shapes, bound_rows = [], set(), []
            by_key = {(r["shape"], r["session"]): r for r in shape_rows}
            for i, s in enumerate(st.shapes):
                if s not in e30 or st.default not in e30[s]["mean_ms"]:
                    continue
                shapes.add(shape_name(*s))
                m = e30[s]["mean_ms"]
                for j in (0, 1):
                    T, To = st.T[j, i], st.T[1 - j, i]
                    s_oth = (To[d] - To[ns]) / To[d]
                    for c, v in zip(st.chunks, st.chunk_idx):
                        if c == st.default or c not in m:
                            continue
                        g48 = infos[s]["scan_only_gain_pct"].get(c) if infos[s] else None
                        pts.append({"shape": shape_name(*s), "session": j + 1, "chunk": c, "measured": 100.0 * (T[d] - T[v]) / T[d],
                                    "pred_stock": s_oth * 100.0 * (1.0 - m[c] / m[st.default]),
                                    "pred_48gb": None if g48 is None else s_oth * g48,
                                    "k": (T[v] - T[ns]) / (layers * m[c])})
                    gains = {c: s_oth * 100.0 * (1.0 - m[c] / m[st.default]) for c in m}
                    c_pred = min(gains, key=lambda c: (-gains[c], c))
                    r = by_key[(shape_name(*s), j + 1)]
                    b_s = max(0.0, gains[c_pred])
                    bound_rows.append({"shape": shape_name(*s), "session": j + 1, "bound_pct": b_s, "single_call_best_chunk": c_pred,
                                       "measured_best_gain_pct": r["measured_best_gain_pct"],
                                       "measured_gain_of_single_call_best_chunk_pct": 100.0 * (T[d] - T[st.vpos[f"c{c_pred}"]]) / T[d],
                                       "violation_strict": bool(r["measured_best_gain_pct"] > b_s),
                                       "violation_beyond_session_noise": bool(r["measured_best_gain_pct"] - b_s > max(r["session_gap_best_pp"], r["round_half_width_best_pp"]))})
            both = [p for p in pts if p["pred_48gb"] is not None]
            stock[f"{tag}|{state}"] = {"shapes": sorted(shapes), "calls_per_shape": sorted({e30[k]["n_calls"] for k in e30 if shape_name(*k) in shapes}),
                                       "stock_calls": err_block([p["pred_stock"] for p in pts], [p["measured"] for p in pts]),
                                       "same_points_48gb_calls": err_block([p["pred_48gb"] for p in both], [p["measured"] for p in both]),
                                       "k_median": float(np.median([p["k"] for p in pts])),
                                       "bound_check": {**err_block([r["bound_pct"] for r in bound_rows], [r["measured_best_gain_pct"] for r in bound_rows]),
                                                       "n_violations_strict": sum(r["violation_strict"] for r in bound_rows),
                                                       "n_violations_beyond_session_noise": sum(r["violation_beyond_session_noise"] for r in bound_rows)},
                                       "per_shape_bound": bound_rows}
    out["stock_board_subset"] = stock
    out["second_session_tile_effect"] = second_session_tile_effect(st, label, shape_rows, in_dir)
    return out


def second_session_tile_effect(st: StepData, label: str, shape_rows: list[dict], in_dir: Path) -> dict | None:
    """Same-chunk latency of the long calls between the two stock-board E30 sessions, and what it implies for the step."""
    paths = [in_dir / f"e30_autotune_state_{tag}_{label}.jsonl" for tag in E30_TAGS]
    if not all(p.exists() for p in paths):
        return None
    first, second = ([r for r in read_jsonl(p) if r.get("kind") == "call"] for p in paths)
    if [(r["L"], r["batch"], r["plan_index"]) for r in first] != [(r["L"], r["batch"], r["plan_index"]) for r in second]:
        raise SystemExit(f"E30 sessions of {label} do not time the same calls in the same order")

    def ratios(state: str, chunk: int, lmin: int = 4096, bmax: int = 10**6) -> list[tuple[int, int, float]]:
        out = []
        for x, y in zip(first, second):
            a, b = x["med_ms"][state].get(str(chunk)), y["med_ms"][state].get(str(chunk))
            if x["L"] >= lmin and x["batch"] <= bmax and a and b:
                out.append((int(x["L"]), int(x["batch"]), float(b) / float(a)))
        return out

    def block(rows: list[tuple[int, int, float]]) -> dict:
        v = [r[2] for r in rows]
        return {"n_calls": len(v), "median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v))}

    rec = {"note": "second over first session, same call and same chunk, calls of at least 4096 tokens; the sessions ran on two boards of "
                   "one host and tuned afresh, so board and tile choice are confounded; a chunk whose ratio stays near one argues against a "
                   "plain board speed difference. The step effect is an arithmetic product of the scan share and the ratio, not a measurement: "
                   "no whole-step run sampled the second tile choice.",
           "default_pinned_all_batches": block(ratios("pinned", st.default)),
           "default_retuned_all_batches": block(ratios("retuned", st.default)),
           "default_pinned_batch_at_most_4": block(ratios("pinned", st.default, bmax=4)),
           "default_pinned_calls": [{"L": a, "batch": b, "ratio": r} for a, b, r in ratios("pinned", st.default)],
           "by_chunk_pinned_all_batches": {f"c{c}": block(ratios("pinned", c)) for c in st.chunks if ratios("pinned", c)}}
    implied = {}
    for name in ("4096x1", "8192x1", "2048x4"):
        rows = [r for r in shape_rows if r["shape"] == name]
        if len(rows) != 2:
            continue
        L, B = (int(x) for x in name.split("x"))
        match = [r for x, r in ((c, c[2]) for c in ratios("pinned", st.default, lmin=L)) if x[0] == L and x[1] == B]
        ms = {r["session"]: r["default_ms"] for r in rows}
        implied[name] = {"scan_share_pct": float(np.mean([r["scan_share_pct"] for r in rows])),
                         "stock_ratio_same_shape_mean": float(np.mean(match)) if match else None,
                         "implied_step_change_at_same_shape_pct": float(np.mean([r["scan_share_pct"] for r in rows]) * (np.mean(match) - 1.0)) if match else None,
                         "e32_default_step_s2_over_s1": float(ms[2] / ms[1])}
    rec["implied_step_change"] = implied
    return rec


def review_product(st: StepData, df) -> dict:
    """The review's own product: latency-weighted scan share times the default's latency-weighted regret over all calls."""
    chunks = chunk_columns(df)
    lat = df[[f"lat_c{c}_mean" for c in chunks]].to_numpy(float)
    orc = np.nanmin(lat, axis=1)
    dflt = lat[:, chunks.index(st.default)]
    lw = float(np.nansum(dflt - orc) / np.nansum(orc))
    shares = [float((st.T[j, :, st.d] - st.T[j, :, st.ns]).sum() / st.T[j, :, st.d].sum()) for j in (0, 1)]
    return {"default_lw_regret_all_calls_pct": 100.0 * lw, "scan_share_lw_pct": [100.0 * x for x in shares],
            "product_pct": [100.0 * x * lw for x in shares]}


# --------------------------------------------------------------------------- task 3
def wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)
    return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]


def fit_linear(x, y, w) -> dict:
    c = wls(np.column_stack([np.ones_like(x), x]), y, w)
    return {"a": float(c[0]), "b": float(c[1])}


def fit_bottleneck(x, y, w) -> dict:
    """max(a, b x) + c: for a fixed crossover tau = a / b the model is linear in (c, b); tau by two nested grids."""
    def sse(tau):
        z = np.maximum(tau, x)
        c, b = wls(np.column_stack([np.ones_like(z), z]), y, w)
        return float((w * (y - c - b * z) ** 2).sum()), float(c), float(b)

    grid = np.geomspace(x.min() * 0.5, x.max(), 241)
    vals = [sse(t) for t in grid]
    ok = [i for i, v in enumerate(vals) if v[2] > 0]
    i = min(ok, key=lambda k: vals[k][0])
    fine = np.geomspace(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)], 121)
    fv = [(sse(t), t) for t in fine]
    (s, c, b), tau = min(((v, t) for v, t in fv if v[2] > 0), key=lambda r: r[0][0])
    return {"a": float(b * tau), "b": float(b), "c": float(c), "crossover_T_graph_ms": float(tau), "flat_level_ms": float(b * tau + c)}


def pred_linear(p, x):
    return p["a"] + p["b"] * x


def pred_bottleneck(p, x):
    return np.maximum(p["a"], p["b"] * x) + p["c"]


def errs(pred, y) -> dict:
    ape = np.abs(pred - y) / y
    return {"n": int(len(y)), "mape_pct": 100.0 * float(ape.mean()), "median_ape_pct": 100.0 * float(np.median(ape)),
            "p95_ape_pct": 100.0 * float(np.percentile(ape, 95)), "rmse_ms": float(np.sqrt(np.mean((pred - y) ** 2))),
            "mean_signed_error_pct": 100.0 * float(((pred - y) / y).mean()),
            "r2_log": float(1.0 - np.sum((np.log(y) - np.log(np.maximum(pred, 1e-9))) ** 2) / np.sum((np.log(y) - np.log(y).mean()) ** 2))}


def classify(actual: np.ndarray, pred: np.ndarray) -> dict:
    a, p = np.asarray(actual, bool), np.asarray(pred, bool)
    tp, fp, fn, tn = [int(np.sum(x)) for x in (a & p, ~a & p, a & ~p, ~a & ~p)]
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else None
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2 if n else None
    return {"n": n, "base_rate": (tp + fn) / n if n else None, "predicted_rate": (tp + fp) / n if n else None, "accuracy": acc,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp)) if (tp + fn and tn + fp) else None,
            "kappa": (acc - pe) / (1 - pe) if (n and pe < 1) else None}


def tie_matrix(H: np.ndarray, tol: float = TIE) -> np.ndarray:
    return np.isfinite(H) & (H <= tol * np.nanmin(H, axis=1, keepdims=True))


def tie_scores(actual_lat: np.ndarray, pred_lat: np.ndarray | None = None, pred_ties: np.ndarray | None = None,
               tol: float = TIE) -> dict:
    """Pairwise: does chunk c tie with the call's oracle (the measured oracle excluded). Per call: do all chunks tie."""
    fin = np.isfinite(actual_lat)
    at = tie_matrix(actual_lat, tol)
    pt = tie_matrix(pred_lat, tol) if pred_ties is None else pred_ties
    orc = np.zeros_like(fin)
    orc[np.arange(len(actual_lat)), np.nanargmin(actual_lat, axis=1)] = True
    mask = fin & ~orc
    return {"pairs_chunk_ties_with_oracle": classify(at[mask], pt[mask]),
            "calls_all_chunks_tie": classify((at | ~fin).all(axis=1), (pt | ~fin).all(axis=1))}


def noise_ceiling(df, chunks: list[int], rng: np.random.Generator) -> dict:
    """Agreement of the measured tie labels with themselves: between rounds, or from the timing blocks of one round."""
    if f"lat_c{chunks[0]}_rounds" in df.columns:
        rounds = np.stack([np.array([[float(x) if x not in ("", "nan") else np.nan for x in str(s).split(";")] for s in df[f"lat_c{c}_rounds"]])
                           for c in chunks], axis=2)
        n_rounds = rounds.shape[1]
        kap = []
        for p in range(n_rounds):
            for q in range(p + 1, n_rounds):
                sc = tie_scores(rounds[:, p, :], rounds[:, q, :])
                kap.append((sc["pairs_chunk_ties_with_oracle"]["kappa"], sc["calls_all_chunks_tie"]["kappa"]))
        return {"source": "between rounds of the launch-inclusive table", "pairs_kappa": float(np.mean([k[0] for k in kap])),
                "calls_kappa": float(np.mean([k[1] for k in kap if k[1] is not None])) if any(k[1] is not None for k in kap) else None}
    mean = df[[f"lat_c{c}_mean" for c in chunks]].to_numpy(float)
    se = (df[[f"lat_c{c}_ci_hi" for c in chunks]].to_numpy(float) - df[[f"lat_c{c}_ci_lo" for c in chunks]].to_numpy(float)) / (2 * 1.96)
    a = mean + np.nan_to_num(se) * rng.standard_normal(mean.shape)
    b = mean + np.nan_to_num(se) * rng.standard_normal(mean.shape)
    sc = tie_scores(a, b)
    return {"source": "simulated from the within-block standard errors of the single round (optimistic: no between-round variance)",
            "pairs_kappa": sc["pairs_chunk_ties_with_oracle"]["kappa"], "calls_kappa": sc["calls_all_chunks_tie"]["kappa"]}


def task3_pair(name: str, li_csv: str, graph_csv: str, in_dir: Path, n: int) -> dict:
    a = load_e2([in_dir / li_csv])
    b = load_e2([in_dir / graph_csv])
    a_view = round_median_view(a) if "oracle_chunk_med" in a.columns else a
    b_view = round_median_view(b) if "oracle_chunk_med" in b.columns else b
    keys = sorted(set(a_view["key"]) & set(b_view["key"]))
    A = a_view.set_index("key").loc[keys].reset_index()
    G = b_view.set_index("key").loc[keys].reset_index()
    if not ((A["L"] == G["L"]).all() and (A["batch"] == G["batch"]).all() and (A["fold"] == G["fold"]).all()):
        raise SystemExit(f"{name}: the tables do not describe the same calls")
    meta = {}
    for tab, csv in (("launch_inclusive", li_csv), ("graph", graph_csv)):
        m = json.loads((in_dir / csv.replace(".csv", ".json")).read_text())["meta"]
        argv = m.get("argv", [])
        spin = float(argv[argv.index("--pre-spin-ms") + 1]) if "--pre-spin-ms" in argv else 0.0
        if tab == "launch_inclusive" and ("--cuda-graph" in argv or spin != 0.0):
            raise SystemExit(f"{csv} is not launch-inclusive")
        if tab == "graph" and "--cuda-graph" not in argv:
            raise SystemExit(f"{csv} is not a graph replay")
        meta[tab] = {"file": csv, "host": m.get("host"), "hostname": m.get("hostname"), "gpu": m.get("gpu"), "gpu_mem_gb": m.get("gpu_mem_gb"),
                     "utc": m.get("utc"), "driver": str(m.get("nvidia_smi", "")).split("\n")[0].split(", ")[1] if m.get("nvidia_smi") else None,
                     "mamba_ssm": m.get("pkg_mamba_ssm"), "pre_spin_ms": spin}
    if meta["launch_inclusive"]["hostname"] != meta["graph"]["hostname"]:
        raise SystemExit(f"{name}: not the same host")
    chunks = [c for c in chunk_columns(A) if c in chunk_columns(G)]
    li = A[[f"lat_c{c}_mean" for c in chunks]].to_numpy(float)
    gr = G[[f"lat_c{c}_mean" for c in chunks]].to_numpy(float)
    bad = ~(np.isfinite(li) & np.isfinite(gr))
    li, gr = np.where(bad, np.nan, li), np.where(bad, np.nan, gr)
    L, B, folds = A["L"].to_numpy(), A["batch"].to_numpy(), A["fold"].to_numpy()
    clusters, n_prompts = prompt_clusters(A)
    if any(len(set(folds[clusters == k])) > 1 for k in range(clusters.max() + 1)):
        raise SystemExit(f"{name}: a prompt cluster crosses folds")
    orc_li, orc_gr = np.nanmin(li, axis=1), np.nanmin(gr, axis=1)
    short = (B == 1) & (L <= 256)
    floor_li, floor_gr = 2.0 * float(np.median(orc_li[short])), 2.0 * float(np.median(orc_gr[short]))
    mask = np.isfinite(li)
    idx_call, idx_chunk = np.where(mask)
    x, y = gr[mask], li[mask]
    fold_pts = folds[idx_call]

    out = {"tables": meta, "n_calls": int(len(A)), "n_points": int(len(y)), "chunks": chunks, "n_prompts": n_prompts,
           "n_prompt_clusters": int(clusters.max()) + 1, "folds": sorted(int(f) for f in np.unique(folds)),
           "heuristic_floor": {"launch_inclusive_floor_ms": floor_li, "median_short_oracle_li_ms": 0.5 * floor_li, "graph_floor_ms": floor_gr,
                               "median_short_oracle_graph_ms": 0.5 * floor_gr, "n_short_calls": int(short.sum()),
                               "share_calls_above_li_floor": float((orc_li > floor_li).mean()), "share_calls_above_graph_floor": float((orc_gr > floor_gr).mean())}}
    # fits: relative least squares (primary) and absolute least squares
    fits = {}
    held = {}
    for loss, wt in (("relative", 1.0 / y ** 2), ("absolute", np.ones_like(y))):
        full = {"linear": fit_linear(x, y, wt), "bottleneck": fit_bottleneck(x, y, wt)}
        H = {"linear": np.full_like(y, np.nan), "bottleneck": np.full_like(y, np.nan)}
        fold_par = {"linear": [], "bottleneck": []}
        for f in np.unique(folds):
            tr, te = fold_pts != f, fold_pts == f
            pl = fit_linear(x[tr], y[tr], wt[tr])
            pb = fit_bottleneck(x[tr], y[tr], wt[tr])
            H["linear"][te] = pred_linear(pl, x[te])
            H["bottleneck"][te] = pred_bottleneck(pb, x[te])
            fold_par["linear"].append(pl)
            fold_par["bottleneck"].append(pb)
        held[loss] = H
        fits[loss] = {"all_data": full,
                      "held_out": {m: errs(H[m], y) for m in H},
                      "in_sample": {"linear": errs(pred_linear(full["linear"], x), y), "bottleneck": errs(pred_bottleneck(full["bottleneck"], x), y)},
                      "fold_range": {"linear_a_ms": [min(p["a"] for p in fold_par["linear"]), max(p["a"] for p in fold_par["linear"])],
                                     "linear_b": [min(p["b"] for p in fold_par["linear"]), max(p["b"] for p in fold_par["linear"])],
                                     "bottleneck_a_ms": [min(p["a"] for p in fold_par["bottleneck"]), max(p["a"] for p in fold_par["bottleneck"])],
                                     "bottleneck_b": [min(p["b"] for p in fold_par["bottleneck"]), max(p["b"] for p in fold_par["bottleneck"])],
                                     "bottleneck_c_ms": [min(p["c"] for p in fold_par["bottleneck"]), max(p["c"] for p in fold_par["bottleneck"])],
                                     "bottleneck_flat_level_ms": [min(p["flat_level_ms"] for p in fold_par["bottleneck"]), max(p["flat_level_ms"] for p in fold_par["bottleneck"])],
                                     "bottleneck_crossover_T_graph_ms": [min(p["crossover_T_graph_ms"] for p in fold_par["bottleneck"]), max(p["crossover_T_graph_ms"] for p in fold_par["bottleneck"])]}}
    # held-out comparison, prompt-cluster bootstrap of the difference in mean absolute percentage error (relative loss)
    K = int(clusters.max()) + 1
    M = make_rng("t3_clusters", name).multinomial(K, np.full(K, 1.0 / K), size=n).astype(float)
    cl_pts = clusters[idx_call]
    ape = {m: np.abs(held["relative"][m] - y) / y for m in held["relative"]}
    sums = {m: np.bincount(cl_pts, weights=ape[m], minlength=K) for m in ape}
    cnt = np.bincount(cl_pts, minlength=K).astype(float)
    diff = 100.0 * (M @ sums["linear"] - M @ sums["bottleneck"]) / (M @ cnt)
    pt_diff = 100.0 * float((sums["linear"].sum() - sums["bottleneck"].sum()) / cnt.sum())
    fits["relative"]["mape_linear_minus_bottleneck_pp"] = {"point": pt_diff, "ci95_prompt_clusters": np.percentile(diff, [2.5, 97.5]).tolist()}
    # baselines (the heuristic floor as a model parameter, refitted on the training folds)
    base = {k: np.full_like(y, np.nan) for k in ("graph_only", "max_median_short_oracle_graph", "max_heuristic_floor_graph")}
    base["graph_only"] = x.copy()
    for f in np.unique(folds):
        te = fold_pts == f
        tr_call = (folds != f) & short
        med = float(np.median(orc_li[tr_call]))
        base["max_median_short_oracle_graph"][te] = np.maximum(med, x[te])
        base["max_heuristic_floor_graph"][te] = np.maximum(2.0 * med, x[te])
    out["baselines_held_out"] = {k: errs(v, y) for k, v in base.items()}
    # error by regime
    above = (orc_li > floor_li)[idx_call]
    fits["relative"]["held_out_by_regime"] = {
        reg: {m: errs(held["relative"][m][sel], y[sel]) for m in held["relative"]}
        for reg, sel in (("below_li_floor", ~above), ("above_li_floor", above)) if sel.any()}
    out["fits"] = fits
    # implied floor against the heuristic
    pb, pl = fits["relative"]["all_data"]["bottleneck"], fits["relative"]["all_data"]["linear"]
    pb_abs, pl_abs = fits["absolute"]["all_data"]["bottleneck"], fits["absolute"]["all_data"]["linear"]
    out["implied_floor"] = {
        "linear_a_relative_loss_ms": pl["a"], "linear_a_absolute_loss_ms": pl_abs["a"],
        "bottleneck_flat_level_relative_loss_ms": pb["flat_level_ms"], "bottleneck_flat_level_absolute_loss_ms": pb_abs["flat_level_ms"],
        "bottleneck_a_relative_loss_ms": pb["a"], "bottleneck_c_relative_loss_ms": pb["c"],
        "bottleneck_crossover_T_graph_relative_loss_ms": pb["crossover_T_graph_ms"],
        "flat_level_over_median_short_oracle_li": pb["flat_level_ms"] / (0.5 * floor_li), "flat_level_over_heuristic_li_floor": pb["flat_level_ms"] / floor_li,
        "linear_a_over_heuristic_li_floor": pl["a"] / floor_li,
        "crossover_T_graph_over_heuristic_graph_floor": pb["crossover_T_graph_ms"] / floor_gr,
        "share_calls_graph_oracle_above_crossover": float((orc_gr > pb["crossover_T_graph_ms"]).mean()),
        "share_calls_flat_regime_agree_with_heuristic_graph_floor": float(((orc_gr > pb["crossover_T_graph_ms"]) == (orc_gr > floor_gr)).mean())}
    # ties: predicted latencies on the held-out fold of each call
    H_call = {m: np.full_like(li, np.nan) for m in held["relative"]}
    for m in H_call:
        H_call[m][mask] = held["relative"][m]
    tie = {"linear_model": tie_scores(li, H_call["linear"]), "bottleneck_model": tie_scores(li, H_call["bottleneck"]),
           "graph_timing_only": tie_scores(li, gr)}
    fin = np.isfinite(li)
    below_li = np.zeros_like(fin)
    below_gr = np.zeros_like(fin)
    for f in np.unique(folds):
        te = folds == f
        tr = (folds != f) & short
        below_li[te] = (orc_li[te] <= 2.0 * np.median(orc_li[tr]))[:, None]
        below_gr[te] = (orc_gr[te] <= 2.0 * np.median(orc_gr[tr]))[:, None]
    orc_mask = np.zeros_like(fin)
    orc_mask[np.arange(len(li)), np.nanargmin(li, axis=1)] = True
    tie["heuristic_li_floor_uses_the_target_timing"] = tie_scores(li, pred_ties=(below_li & fin) | (tie_matrix(li) & orc_mask))
    tie["heuristic_graph_floor"] = tie_scores(li, pred_ties=(below_gr & fin) | (tie_matrix(gr) & (gr == np.nanmin(gr, axis=1, keepdims=True))))
    tie["noise_ceiling"] = noise_ceiling(a if "oracle_chunk_med" in a.columns else a_view.set_index("key").loc[keys].reset_index(), chunks, make_rng("t3_ceiling", name))
    at = tie_matrix(li)
    all_tie = (at | ~fin).all(axis=1)
    tie["within_5pct_sensitivity"] = {"linear_model": tie_scores(li, H_call["linear"], tol=1.05), "bottleneck_model": tie_scores(li, H_call["bottleneck"], tol=1.05),
                                      "graph_timing_only": tie_scores(li, gr, tol=1.05)}

    def pair_share(lat, sel):
        t, fn = tie_matrix(lat), np.isfinite(lat)
        o = np.zeros_like(fn)
        o[np.arange(len(lat)), np.nanargmin(lat, axis=1)] = True
        m_ = fn & ~o & sel[:, None]
        return float(t[m_].mean()) if m_.any() else None

    b_lat = np.where(fin, li, np.nan)
    spread = np.nanmax(b_lat, axis=1) / np.nanmin(b_lat, axis=1)
    tie["heuristic_floor_claim"] = {
        "note": "the glossary says chunks tie below the floor; share of calls whose chunks all lie within 2%, by side of the floor",
        "below_li_floor": {"n": int((orc_li <= floor_li).sum()), "share_all_chunks_tie": float(all_tie[orc_li <= floor_li].mean()), "median_spread": float(np.median(spread[orc_li <= floor_li])),
                           "share_non_oracle_chunks_tie": pair_share(li, orc_li <= floor_li)},
        "above_li_floor": {"n": int((orc_li > floor_li).sum()), "share_all_chunks_tie": float(all_tie[orc_li > floor_li].mean()) if (orc_li > floor_li).any() else None,
                           "median_spread": float(np.median(spread[orc_li > floor_li])) if (orc_li > floor_li).any() else None,
                           "share_non_oracle_chunks_tie": pair_share(li, orc_li > floor_li)}}
    gfin = np.isfinite(gr)
    g_all = (tie_matrix(gr) | ~gfin).all(axis=1)
    g_spread = np.nanmax(gr, axis=1) / np.nanmin(gr, axis=1)
    below = orc_gr <= floor_gr
    tie["heuristic_floor_claim_graph_timing"] = {
        "below_graph_floor": {"n": int(below.sum()), "share_all_chunks_tie": float(g_all[below].mean()), "median_spread": float(np.median(g_spread[below])),
                              "share_non_oracle_chunks_tie": pair_share(gr, below)},
        "above_graph_floor": {"n": int((~below).sum()), "share_all_chunks_tie": float(g_all[~below].mean()), "median_spread": float(np.median(g_spread[~below])),
                              "share_non_oracle_chunks_tie": pair_share(gr, ~below)}}
    flat = np.nanmax(gr, axis=1) < pb["crossover_T_graph_ms"]
    tie["flat_regime"] = {"note": "calls whose graph latency lies below the fitted crossover at every chunk: the model predicts a tie",
                          "n_calls": int(flat.sum()), "share_all_chunks_tie": float(all_tie[flat].mean()) if flat.any() else None,
                          "median_spread": float(np.median(spread[flat])) if flat.any() else None,
                          "median_li_ms_by_chunk": {str(c): float(np.nanmedian(li[flat, j])) for j, c in enumerate(chunks)} if flat.any() else None,
                          "share_non_oracle_chunks_tie": pair_share(li, flat)}
    out["ties"] = tie
    return out


# --------------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args(argv)
    n, in_dir = args.n_boot, args.in_dir
    if n < 2000:
        raise SystemExit("at least 2000 bootstrap replicates are required")
    k = power_constants()
    res = {"script": "jpdc_step_bound", "seed": SEED, "n_boot": n, "margins_pct": list(MARGINS),
           "power_multipliers_of_se": k, "reproduction": {}}
    t1, t2, reviews = {}, {}, {}
    data = {lab: StepData(lab, in_dir) for lab in MODELS}
    sweeps = {lab: load_sweep(in_dir / MODELS[lab][0]) for lab in MODELS}
    calls = {lab: Calls(sweeps[lab], MODELS[lab][2], lab, n) for lab in MODELS}
    shapes = data["ssd370m"].shapes
    if any(data[lab].shapes != shapes for lab in MODELS) or len(shapes) != 14:
        raise SystemExit("the models do not share the 14 E32 shapes")
    common = [s for s in shapes if all(calls[lab].at(*s) is not None for lab in MODELS)]
    for lab, st in data.items():
        csv, prefix, default, printed = MODELS[lab]
        stored = json.loads((in_dir / f"e32_model_graph_{SESSIONS[0]}_{lab}_{SESSIONS[1]}_{lab}.json").read_text())
        tables = {b: held_out_table(sweeps[lab], prefix, b) for b in {s[1] for s in st.shapes}}
        tchunks = [table_chunk(tables[s[1]], s[0]) for s in st.shapes]
        for c in (c for c in stored["cells"] if c["mode"] == "graph"):
            if tchunks[st.shapes.index((c["L"], c["batch"]))] != c["table_chunk"]:
                raise SystemExit(f"cannot reproduce the stored table: {lab} {c['L']}x{c['batch']}")
        tidx = np.array([st.vpos[f"c{c}"] for c in tchunks])
        res["reproduction"][lab] = reproduce(st, tidx, stored, printed)
        print(lab, "reproduced", {k_: v["table_gain_lw_pct"] for k_, v in res["reproduction"][lab].items() if isinstance(v, dict)})
        D = [[draw_rounds(st.y[j], n, make_rng("t1_rounds", lab, j, q)) for q in (0, 1)] for j in (0, 1)]
        t1[lab] = task1_model(st, tidx, D, k, n)
        t2[lab] = task2_model(st, tidx, calls[lab], D, in_dir, common)
        reviews[lab] = review_product(st, sweeps[lab])
        del D
    res["reproduction"]["all_asserted"] = True

    def smallest(block, *path):
        for p in path:
            block = block[p]
        return block["smallest_margin_pct"]

    res["task1_equivalence"] = {
        "method": "see the module docstring; shapes are a fixed design in every block without shapes in its name",
        "summary": {lab: {
            "table_pooled": t1[lab]["table"]["pooled"]["rounds_sessions"],
            "table_smallest_margin_rounds_sessions": smallest(t1[lab]["table"], "pooled", "tost_rounds_sessions"),
            "table_smallest_margin_rounds_sessions_shapes": smallest(t1[lab]["table"], "pooled", "tost_rounds_sessions_shapes"),
            "table_smallest_margin_session_t_df1": smallest(t1[lab]["table"], "pooled", "tost_session_t_df1"),
            "cross_session_pooled": t1[lab]["cross_session_best"]["pooled"]["rounds_sessions"],
            "cross_session_smallest_margin_rounds_sessions": smallest(t1[lab]["cross_session_best"], "pooled", "tost_rounds_sessions"),
            "cross_session_smallest_margin_shapes": smallest(t1[lab]["cross_session_best"], "pooled", "tost_rounds_and_shapes_directions_fixed"),
            "cross_session_smallest_margin_session_t_df1": smallest(t1[lab]["cross_session_best"], "pooled", "tost_session_t_df1")} for lab in MODELS},
        "models": t1}
    rows_all = [dict(r, model=lab) for lab in MODELS for r in t2[lab]["per_shape"] if r["exact_match"]]
    totals = err_block([r["bound_pct"] for r in rows_all], [r["measured_best_gain_pct"] for r in rows_all])
    totals["cross_session_best"] = err_block([r["bound_pct"] for r in rows_all], [r["measured_cross_session_best_gain_pct"] for r in rows_all])
    pos_all = [r for r in rows_all if r["bound_pct"] > 0]
    totals["where_bound_positive"] = {"spearman_in_session_best": spearman([r["bound_pct"] for r in pos_all], [r["measured_best_gain_pct"] for r in pos_all]),
                                      "spearman_cross_session_best": spearman([r["bound_pct"] for r in pos_all], [r["measured_cross_session_best_gain_pct"] for r in pos_all])}
    totals.update({"n_violations_strict": sum(r["violation_strict"] for r in rows_all),
                   "n_violations_beyond_session_gap": sum(r["violation_beyond_session_gap"] for r in rows_all),
                   "n_violations_beyond_round_half_width": sum(r["violation_beyond_round_half_width"] for r in rows_all),
                   "n_violations_beyond_session_noise": sum(r["violation_beyond_session_noise"] for r in rows_all),
                   "n_cross_session_violations_beyond_session_noise": sum(r["cross_session_violation_beyond_session_noise"] for r in rows_all)})
    res["task2_compositional_bound"] = {
        "totals_over_models_exact_shapes": totals,
        "what_is_tested": ("with T_scan(C) := T_step(C) - T_step(noscan) the formula is an identity; the rest of the step cannot be observed at a "
                           "chunk, so additivity is only testable jointly with the substitution of isolated single-call replays for the in-step scan"),
        "boards": "E32: stock 24 GB RTX 3090, sessions two hours apart on one board; single calls: 48 GB RTX 3090, other tensors, driver 595.71.05 on both; E30: stock board, 1-2 calls per shape",
        "call_mix": "every captured call at the exact (L, B), equal weight, ratio of sums over calls; layers sampled every 6th (370M), 8th (2.7B), 4th (GDN)",
        "common_exact_shapes": [shape_name(*s) for s in common],
        "shapes_without_exact_match": [shape_name(*s) for s in shapes if s not in common],
        "review_product_all_calls": reviews, "models": t2}

    t3 = {"not_paired": {}}
    for lab, stem in NOT_PAIRED.items():
        m = json.loads((in_dir / f"{stem}.json").read_text())["meta"]
        argv = m.get("argv", [])
        spin = float(argv[argv.index("--pre-spin-ms") + 1]) if "--pre-spin-ms" in argv else 0.0
        if spin == 0.0:
            raise SystemExit(f"{stem} is launch-inclusive: pair it")
        t3["not_paired"][lab] = {"only_non_graph_table": stem, "pre_spin_ms": spin,
                                 "reason": "device time with a GPU-side pre-spin: no launch-inclusive timing of this checkpoint exists on the board"}
    t3["pairs"] = {name: task3_pair(name, li, gr, in_dir, n) for name, (li, gr) in PAIRS.items()}
    e22 = json.loads((in_dir / "e22_graph_analysis_vgpu3090.json").read_text())["tables"]
    checks = {"ssd370m_rtx3090|launch_inclusive_floor_ms": ("ssd370m_eager", t3["pairs"]["ssd370m_rtx3090"]["heuristic_floor"]["launch_inclusive_floor_ms"]),
              "ssd370m_rtx3090_same_driver|graph_floor_ms": ("ssd370m_graph", t3["pairs"]["ssd370m_rtx3090_same_driver"]["heuristic_floor"]["graph_floor_ms"]),
              "ssd370m_pro6000|launch_inclusive_floor_ms": ("ssd370m_pro6000_eager", t3["pairs"]["ssd370m_pro6000"]["heuristic_floor"]["launch_inclusive_floor_ms"])}
    t3["stored_floor_check"] = {}
    for name, (tab, val) in checks.items():
        stored_ms = e22[tab]["floor"]["floor_us"] / 1000.0
        if abs(val - stored_ms) > 1e-6:
            raise SystemExit(f"cannot reproduce the stored heuristic floor: {name}: {val} vs {stored_ms}")
        t3["stored_floor_check"][name] = {"stored_table": tab, "stored_ms": stored_ms, "computed_ms": val}
    res["task3_launch_floor_model"] = t3

    path = in_dir / "jpdc_step_bound.json"
    write_json(path, _clean(res))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
