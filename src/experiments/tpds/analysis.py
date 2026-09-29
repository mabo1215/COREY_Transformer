"""Shared CPU-side analysis helpers for E2 tables (used by E3, E4, E5 and the figures)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def load_e2(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        p = Path(p)
        if p.exists() and p.stat().st_size > 0:
            try:
                df = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                continue
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["key", "host"], keep="last")
    return df.reset_index(drop=True)


def chunk_columns(df: pd.DataFrame) -> list[int]:
    out = []
    for c in df.columns:
        m = re.fullmatch(r"lat_c(\d+)_mean", c)
        if m and df[c].notna().any():
            out.append(int(m.group(1)))
    return sorted(out)


def latency_matrix(df: pd.DataFrame, chunks: list[int]) -> np.ndarray:
    return df[[f"lat_c{c}_mean" for c in chunks]].to_numpy(dtype=float)


def regret_of_choice(df: pd.DataFrame, chosen: np.ndarray, chunks: list[int]) -> np.ndarray:
    """Relative regret (latency of the chosen chunk / per-call oracle latency - 1)."""
    lat = latency_matrix(df, chunks)
    idx = np.array([chunks.index(int(c)) for c in chosen])
    picked = lat[np.arange(len(df)), idx]
    return picked / lat.min(axis=1) - 1.0


def regret_stats(reg: np.ndarray) -> dict:
    if len(reg) == 0:
        return {"n": 0}
    return {"n": int(len(reg)), "mean_regret_pct": float(100 * reg.mean()),
            "p95_regret_pct": float(100 * np.percentile(reg, 95)), "max_regret_pct": float(100 * reg.max()),
            "oracle_hit_rate": float((reg <= 1e-12).mean())}


def static_regret_table(df: pd.DataFrame, chunks=None) -> dict:
    chunks = list(chunks or chunk_columns(df))
    chunks = [c for c in chunks if f"lat_c{c}_mean" in df.columns]
    out = {}
    for c in chunks:
        out[f"static_{c}"] = regret_stats(regret_of_choice(df, np.full(len(df), c), chunks))
    if "entropy_rule_chunk" in df.columns:
        rule = df["entropy_rule_chunk"].clip(min(chunks), max(chunks)).to_numpy()
        out["entropy_rule"] = regret_stats(regret_of_choice(df, rule, chunks))
    best = min((k for k in out if k.startswith("static_")), key=lambda k: out[k]["mean_regret_pct"])
    out["best_static"] = best
    return out


def conditional_mutual_information(x: np.ndarray, y: np.ndarray, cond: np.ndarray, bins: int = 8) -> dict:
    """I(X; Y | Cond) in bits with X discretized into quantile bins; Y and Cond are discrete.

    Also returns a permutation baseline (X shuffled within each Cond stratum), since
    plug-in MI estimates are biased upward for small samples.
    """
    def _disc(v):
        qs = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1]))
        return np.searchsorted(qs, v, side="right")

    def _mi(a, b):
        n = len(a)
        if n == 0:
            return 0.0
        joint = pd.crosstab(a, b).to_numpy(dtype=float) / n
        pa = joint.sum(1, keepdims=True)
        pb = joint.sum(0, keepdims=True)
        nz = joint > 0
        return float((joint[nz] * np.log2(joint[nz] / (pa @ pb)[nz])).sum())

    xd = _disc(np.asarray(x, dtype=float))
    y = np.asarray(y)
    cond = np.asarray(cond)
    total, total_perm, h_y = 0.0, 0.0, 0.0
    rng = np.random.default_rng(0)
    n = len(y)
    for c in np.unique(cond):
        m = cond == c
        w = m.sum() / n
        total += w * _mi(xd[m], y[m])
        perms = []
        for _ in range(20):
            perms.append(_mi(rng.permutation(xd[m]), y[m]))
        total_perm += w * float(np.mean(perms))
        p = pd.Series(y[m]).value_counts(normalize=True).to_numpy()
        h_y += w * float(-(p * np.log2(p)).sum())
    return {"cmi_bits": total, "cmi_perm_baseline_bits": total_perm, "cmi_excess_bits": total - total_perm,
            "H_oracle_given_cond_bits": h_y, "bins": bins}


def cv_exact_headroom(df: pd.DataFrame, chunks: list[int], threshold_ms: float | None = None) -> dict:
    """Out-of-sample oracle headroom of a chunk table keyed on the exact shape (L, batch).

    For every prompt fold, the table is chosen on the other folds; a shape without training
    calls falls back to the (length bucket, batch, L mod 16 == 0) table and then to the best
    single chunk. Returns the per-call mean and, per exact shape, the mean headroom.
    """
    lat = latency_matrix(df, chunks)
    orc = lat.min(1)
    aligned = (df["L"].to_numpy() % 16 == 0).astype(int)
    h = np.full(len(df), np.nan)
    source = {"exact": 0, "bucket_aligned": 0, "global": 0}
    folds = df["fold"].to_numpy()
    for f in np.unique(folds):
        te, tr = folds == f, folds != f
        dtr = df[tr]
        exact = {k: int(np.argmin(lat[g.index].mean(0))) for k, g in dtr.groupby(["L", "batch"])}
        bucket = {k: int(np.argmin(lat[g.index].mean(0)))
                  for k, g in dtr.assign(aligned=aligned[tr]).groupby(["L_bucket", "batch", "aligned"])}
        glob = int(np.argmin((lat[tr] / orc[tr, None]).sum(0)))
        for i in np.where(te)[0]:
            L, b, lb = int(df["L"].iat[i]), int(df["batch"].iat[i]), int(df["L_bucket"].iat[i])
            if (L, b) in exact:
                j = exact[(L, b)]; source["exact"] += 1
            elif (lb, b, int(aligned[i])) in bucket:
                j = bucket[(lb, b, int(aligned[i]))]; source["bucket_aligned"] += 1
            else:
                j = glob; source["global"] += 1
            h[i] = lat[i, j] - orc[i]
    per_key = pd.Series(h).groupby([df["L"].to_numpy(), df["batch"].to_numpy()]).mean()
    out = {"mean_ms": float(np.nanmean(h)), "mean_pct": float(100 * np.nanmean(h / orc)),
           "max_key_ms": float(per_key.max()), "n_keys": int(len(per_key)), "fallbacks": source,
           "n_keys_single_prompt_group": int((df.groupby(["L", "batch"])["group_id"].nunique() == 1).sum())
           if "group_id" in df else None}
    if threshold_ms is not None:
        above = per_key[per_key > threshold_ms]
        out["threshold_ms"] = threshold_ms
        out["n_keys_above_threshold"] = int(len(above))
        out["calls_on_keys_above_threshold"] = int(sum(
            ((df["L"] == L) & (df["batch"] == b)).sum() for L, b in above.index))
    return out

