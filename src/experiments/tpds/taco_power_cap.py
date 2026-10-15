"""Power-cap audit (taco-r1, review item S9P5): software power-cap share of the logged GPU state per experiment,
and stability of the chunk rankings when the capped samples are excluded. CPU only, reads stored result files.

Several experiments read the GPU state through NVML (or one ``nvidia-smi`` query) before and after every timing
block: SM and memory clock, board power, temperature and the bitmask of active clock-event ("throttle") reasons.
The bit values below are those of the ``nvmlClocksEventReason*`` / ``nvmlClocksThrottleReason*`` macros in NVIDIA's
``nvml.h`` (NVML API version 13) and of the constants of NVIDIA's ``nvidia-ml-py`` binding (13.610.43); both were
checked before writing this script. ``nvidia-smi --query-gpu=clocks_event_reasons.active`` reports the same mask
in hexadecimal. Bits 0x200 (board limit) and 0x400 (reliability) are newer than the drivers used and never set.

  0x001 GPU idle            0x010 sync boost                 0x080 hardware power brake slowdown
  0x002 application clocks  0x020 software thermal slowdown  0x100 display clock setting
  0x004 software power cap  0x040 hardware thermal slowdown
  0x008 hardware slowdown

Part 1, per result file that records the state (every ``.jsonl`` whose text contains ``throttle_reasons``; the
summary ``.json`` files that repeat the rows of a ``.jsonl`` are skipped): number of state samples (one sample is
one before- or after-block reading), share with the software power cap bit, with any thermal bit (0x20 or 0x40),
with a hardware bit (0x08, 0x40, 0x80), the reason codes seen, SM clock median and range (also separately for capped
and uncapped samples), memory clocks seen, power median / max and temperature median / max. Covers the RTX 3090
and, for comparison, the RTX PRO 6000 and H800 files, plus a per-GPU summary.

Part 2, ranking stability on the RTX 3090 data sets where the cap is common. The state granularity decides what
can be excluded:

  per block   E24 (TFLA sweep), E12 (tensor-parallel widths), E17c (production kernels): one row is one shape in
              one round with every chunk timed back to back between the two readings, so a capped sample removes
              the whole (shape, round) block, i.e. every chunk of that round together.
  per cell    E9 (placement control): one row is one (input, chunk, round) cell with its own two readings.
  per run     E10 (routed end-to-end study): one reading per generate call.

Two exclusion rules: ``drop_any_capped`` removes a block or cell when either of its readings shows the cap
(strict), ``drop_capped_throughout`` only when both do (lenient). For each rule the key quantities of the paper
are recomputed with the experiment's own conventions (functions of ``e24_tfla_chunk``, ``e12_analysis``,
``e17_analysis`` and the decision rule of ``e1_analysis``) on the retained data and compared with all data on the
same shapes: oracle agreement (exact, and the retained-data oracle inside the all-data tie set), the all-data regret
of the retained-data oracle, and the regret of the library default, the best static chunk and the other selectors
the paper reports for that data set.

Dropping rounds alone makes the oracle noisier (fewer rounds per shape), whatever the cap does. Every comparison
is therefore paired with a matched random control: per shape (per cell group for E9) the same number of rounds is
kept as under the cap rule, but drawn at random (``--n-perm`` draws, fixed seeds). The cap matters only if the cap
rule moves the result beyond that control (``share_random_le`` / ``share_random_ge``: share of random draws at or
below / at or above the cap-rule value). A round-pair diagnostic adds, within a shape, the argmin agreement and
latency difference of rounds whose cap status differs against rounds whose status agrees.

The per-call selector tables (the captured-call sweeps behind the oracle, predictor, cost-model and graph tables)
were timed without state logging, so their stability cannot be checked this way; they are listed explicitly.

Output: ``taco_power_cap.json`` in the results directory.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.tpds import e12_analysis as E12
from src.experiments.tpds import e17_analysis as E17
from src.experiments.tpds import e24_tfla_chunk as E24
from src.experiments.tpds.common import MAMBA1_CHUNKS, RESULTS_DIR, read_jsonl, t_crit, utc_now, write_json
from src.experiments.tpds.e12_tp_widths import MODEL_WIDTHS, REF_CHUNKS

NVML_BITS = {"gpu_idle": 0x1, "applications_clocks_setting": 0x2, "sw_power_cap": 0x4, "hw_slowdown": 0x8,
             "sync_boost": 0x10, "sw_thermal_slowdown": 0x20, "hw_thermal_slowdown": 0x40,
             "hw_power_brake_slowdown": 0x80, "display_clock_setting": 0x100, "board_limit": 0x200,
             "reliability": 0x400}
SW_POWER_CAP = NVML_BITS["sw_power_cap"]
THERMAL_MASK = NVML_BITS["sw_thermal_slowdown"] | NVML_BITS["hw_thermal_slowdown"]
HW_MASK = NVML_BITS["hw_slowdown"] | NVML_BITS["hw_thermal_slowdown"] | NVML_BITS["hw_power_brake_slowdown"]
STATE_KEYS = ("gpu_state", "gpu_state_before", "gpu_state_after")

EXPERIMENT_LABELS = {
    "e1_": "value-invariance redraw and fixed-placement controls (E1b)",
    "e9_placement": "placement control (E9)",
    "e10_routed": "routed end-to-end study (E10)",
    "e12_tp_widths": "tensor-parallel widths (E12)",
    "e16c_l2_placement": "warm versus flushed L2 (E16c)",
    "e17_workbound": "production kernels at larger widths, synthetic (E17c)",
    "e24_tfla": "TFLA mLSTM chunk sweep (E24)",
}
GRANULARITY = {"e9_placement": "per cell", "e16c_l2_placement": "per cell", "e1_": "per cell",
               "e10_routed": "per run (one reading per generate call)",
               "e12_tp_widths": "per block (shape x round, all chunks)",
               "e17_workbound": "per block (shape x round, all chunks)",
               "e24_tfla": "per block (shape x round, all chunks)"}
# files that are not the data set the paper uses (facts read from the files themselves)
SUPERSEDED = {
    "e24_tfla_vgpu3090_r4plain.jsonl": "one timing row only; the launch-inclusive TFLA sweep is the r4plainb file",
    "e24_tfla_vgpu3090_r4spin.jsonl": "one timing row only; the device-time TFLA sweep is the r4spinb file",
    "e17_workbound_vgpu3090.jsonl": "63 timing rows stored (the run summary lists 360); the stored E17 analysis uses "
                                    "the complete vgpu3090full file",
}
VARIANTS = {"drop_any_capped": lambda b, a: not (b or a), "drop_capped_throughout": lambda b, a: not (b and a)}
VARIANT_TEXT = {"drop_any_capped": "a block or cell is removed when either of its two readings shows the cap",
                "drop_capped_throughout": "a block or cell is removed only when both of its readings show the cap"}


# --------------------------------------------------------------------------- helpers
def _label(name: str, table: dict) -> str | None:
    for prefix, v in table.items():
        if name.startswith(prefix):
            return v
    return None


def _reason(s: dict | None) -> int | None:
    if not isinstance(s, dict):
        return None
    v = s.get("throttle_reasons")
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        try:
            return int(v, 16) if v.lower().startswith("0x") else int(v)
        except ValueError:
            return None
    return None


def _capped(s: dict | None) -> bool | None:
    r = _reason(s)
    return None if r is None else bool(r & SW_POWER_CAP)


def row_flags(r: dict) -> tuple[bool, bool]:
    """(cap before, cap after) of a block or cell row; E10 rows carry one reading, used for both.
    A missing or unparsable reading counts as not capped (none occurs in the stored files)."""
    if "gpu_state" in r and "gpu_state_before" not in r:
        c = bool(_capped(r.get("gpu_state")))
        return c, c
    return bool(_capped(r.get("gpu_state_before"))), bool(_capped(r.get("gpu_state_after")))


def _num(xs):
    xs = [float(x) for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    if not xs:
        return None
    return {"n": len(xs), "median": float(statistics.median(xs)), "min": float(min(xs)), "max": float(max(xs))}


def _share(k: int, n: int) -> float | None:
    return None if n == 0 else k / n


def _summ(regs: list[float], tied: list[bool] | None = None) -> dict | None:
    """Mean / p95 / max of percent regrets (p95 by the numpy default interpolation)."""
    if not regs:
        return None
    a = np.asarray(regs, float)
    out = {"n": int(a.size), "mean_pct": float(a.mean()), "p95_pct": float(np.percentile(a, 95)),
           "max_pct": float(a.max())}
    if tied is not None:
        out["tied_share"] = float(np.mean(tied))
    return out


def _get(d: dict | None, key: str = "mean_pct") -> float | None:
    return None if not d or d.get(key) is None else float(d[key])


def _delta(a: dict | None, b: dict | None, key: str = "mean_pct") -> float | None:
    x, y = _get(a, key), _get(b, key)
    return None if x is None or y is None else y - x


def _by(rows: list[dict], keyfn) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[keyfn(r)].append(r)
    return by


def kept_counts(rows: list[dict], keep, keyfn) -> dict:
    kept = [r for r in rows if keep(*row_flags(r))]
    by_all, by_kept = _by(rows, keyfn), _by(kept, keyfn)
    return {"rows_total": len(rows), "rows_kept": len(kept), "shapes_total": len(by_all), "shapes_kept": len(by_kept),
            "shapes_with_all_rounds_kept": sum(1 for k, v in by_kept.items() if len(v) == len(by_all[k]))}


def cap_breakdown(rows: list[dict], fields: dict) -> dict:
    """Share of rows (blocks or cells) with the cap in any / in both readings, per value of each field."""
    out = {}
    for name, fn in fields.items():
        by = _by(rows, fn)
        out[name] = {str(k): {"rows": len(rs), "cap_in_any_reading": float(np.mean([any(row_flags(r)) for r in rs])),
                              "cap_in_both_readings": float(np.mean([all(row_flags(r)) for r in rs]))}
                     for k, rs in sorted(by.items(), key=lambda kv: kv[0])}
    return out


def matched_control(rows: list[dict], keep, groupfn, evalfn, n_perm: int, label: str) -> dict:
    """Cap-rule statistics against a matched random control: per group the same number of rows kept, at random."""
    groups = _by(rows, groupfn)
    k_of = {g: sum(1 for r in rs if keep(*row_flags(r))) for g, rs in groups.items()}
    cap_stats = evalfn([r for r in rows if keep(*row_flags(r))])
    rng = np.random.default_rng(zlib.crc32(label.encode()))
    draws = defaultdict(list)
    for _ in range(n_perm):
        sub = []
        for g, rs in groups.items():
            k = k_of[g]
            if k:
                sub += [rs[i] for i in rng.choice(len(rs), size=k, replace=False)]
        for name, v in evalfn(sub).items():
            if v is not None and np.isfinite(v):
                draws[name].append(float(v))
    out = {}
    for name, v in cap_stats.items():
        d = np.asarray(draws.get(name, []), float)
        rec = {"cap_rule": v}
        if v is not None and d.size:
            rec.update({"random_mean": float(d.mean()), "random_p025": float(np.percentile(d, 2.5)),
                        "random_p975": float(np.percentile(d, 97.5)), "share_random_le": float(np.mean(d <= v + 1e-12)),
                        "share_random_ge": float(np.mean(d >= v - 1e-12)), "n_draws": int(d.size)})
        out[name] = rec
    return out


def round_pair_diagnostic(rows: list[dict], keyfn) -> dict:
    """Within a shape, pairs of rounds with equal / different cap status: argmin agreement and latency difference.

    Latency difference of a pair: geometric mean over the common chunks of the per-chunk ratio of round means, in
    percent; for mixed pairs also signed as the capped round over the uncapped one."""
    out = {}
    for flag_name, flag in (("cap_in_any_reading", lambda b, a: bool(b or a)),
                            ("cap_in_both_readings", lambda b, a: bool(b and a))):
        st = {"same_status": {"pairs": 0, "agree": 0, "abs": []},
              "mixed_status": {"pairs": 0, "agree": 0, "abs": [], "signed": []}}
        for rs in _by(rows, keyfn).values():
            for i in range(len(rs)):
                for j in range(i + 1, len(rs)):
                    a, b = rs[i], rs[j]
                    common = sorted(set(a["cells"]) & set(b["cells"]), key=int)
                    if len(common) < 2:
                        continue
                    fa, fb = flag(*row_flags(a)), flag(*row_flags(b))
                    am = min(common, key=lambda c: a["cells"][c]["mean_ms"])
                    bm = min(common, key=lambda c: b["cells"][c]["mean_ms"])
                    lr = float(np.mean([math.log(a["cells"][c]["mean_ms"] / b["cells"][c]["mean_ms"]) for c in common]))
                    s = st["mixed_status" if fa != fb else "same_status"]
                    s["pairs"] += 1
                    s["agree"] += am == bm
                    s["abs"].append(100 * (math.exp(abs(lr)) - 1))
                    if fa != fb:
                        s["signed"].append(100 * (math.exp(lr if fa else -lr) - 1))
        res = {}
        for typ, s in st.items():
            r = {"pairs": s["pairs"], "argmin_agreement": _share(s["agree"], s["pairs"]),
                 "abs_latency_diff_pct_median": float(np.median(s["abs"])) if s["abs"] else None}
            if typ == "mixed_status":
                v = s["signed"]
                r["capped_vs_uncapped_latency_diff_pct"] = (
                    {"median": float(np.median(v)), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
                     "share_capped_slower": float(np.mean(np.asarray(v) > 0))} if v else None)
            res[typ] = r
        out[flag_name] = res
    return out


# --------------------------------------------------------------------------- part 1: state statistics per file
def state_files(in_dir: Path) -> tuple[list[Path], list[dict]]:
    raw, skipped = [], []
    for p in sorted(in_dir.iterdir()):
        if not p.is_file() or p.suffix not in (".jsonl", ".json", ".csv") or p.name.startswith("taco_power_cap"):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "throttle_reasons" not in text:
            continue
        if p.suffix == ".jsonl":
            raw.append(p)
        else:
            twin = p.with_suffix(".jsonl")
            skipped.append({"file": p.name, "reason": (f"summary of the rows of {twin.name}" if twin.exists()
                                                       else "derived file without raw rows")})
    return raw, skipped


def file_stats(path: Path) -> dict:
    meta = next(iter(read_jsonl(path, kind="meta")), {})
    rows = read_jsonl(path)                       # every non-meta row
    samples, kinds = [], Counter()
    row_any = row_both = n_rows_state = 0
    for r in rows:
        ss = [r.get(k) for k in STATE_KEYS if isinstance(r.get(k), dict)]
        if not ss:
            continue
        n_rows_state += 1
        kinds[r.get("kind")] += 1
        samples += ss
        flags = [_capped(s) for s in ss]
        row_any += any(f for f in flags if f is not None)
        row_both += all(f is True for f in flags)
    codes = [_reason(s) for s in samples]
    ok = [c for c in codes if c is not None]
    n = len(ok)
    cap = [s for s, c in zip(samples, codes) if c is not None and c & SW_POWER_CAP]
    unc = [s for s, c in zip(samples, codes) if c is not None and not c & SW_POWER_CAP]
    args = meta.get("args") or {}
    spin = args.get("pre_spin_ms")
    timing = None if spin is None else ("device time (GPU-side pre-spin)" if float(spin) > 0 else "launch-inclusive")
    rec = {
        "file": path.name, "experiment": meta.get("experiment"), "label": _label(path.name, EXPERIMENT_LABELS),
        "gpu": meta.get("gpu"), "host_tag": meta.get("host"), "run_tag": args.get("tag"), "run_start_utc": meta.get("utc"),
        "driver": meta.get("nvidia_smi"), "timing": timing, "state_granularity": _label(path.name, GRANULARITY),
        "state_source": dict(Counter(s.get("source") for s in samples)),
        "n_rows_with_state": n_rows_state, "rows_with_state_by_kind": dict(kinds),
        "n_samples": len(samples), "n_samples_unparsed_reasons": len(samples) - n,
        "reason_code_counts": {hex(k): v for k, v in sorted(Counter(ok).items())},
        "n_samples_sw_power_cap": len(cap),
        "share_sw_power_cap": _share(len(cap), n),
        "share_any_thermal": _share(sum(1 for c in ok if c & THERMAL_MASK), n),
        "share_sw_thermal": _share(sum(1 for c in ok if c & NVML_BITS["sw_thermal_slowdown"]), n),
        "share_hw_thermal": _share(sum(1 for c in ok if c & NVML_BITS["hw_thermal_slowdown"]), n),
        "share_any_hw_slowdown": _share(sum(1 for c in ok if c & HW_MASK), n),
        "share_gpu_idle": _share(sum(1 for c in ok if c & NVML_BITS["gpu_idle"]), n),
        "share_rows_cap_in_any_reading": _share(row_any, n_rows_state),
        "share_rows_cap_in_all_readings": _share(row_both, n_rows_state),
        "sm_mhz": _num(s.get("sm_mhz") for s in samples),
        "sm_mhz_capped": _num(s.get("sm_mhz") for s in cap),
        "sm_mhz_uncapped": _num(s.get("sm_mhz") for s in unc),
        "mem_mhz_values": sorted({float(s["mem_mhz"]) for s in samples if isinstance(s.get("mem_mhz"), (int, float))}),
        "power_w": _num(s.get("power_w") for s in samples),
        "power_w_capped": _num(s.get("power_w") for s in cap),
        "power_w_uncapped": _num(s.get("power_w") for s in unc),
        "temp_c": _num(s.get("temp_c") for s in samples),
    }
    if path.name in SUPERSEDED:
        rec["superseded"] = SUPERSEDED[path.name]
    return rec


def gpu_summary(recs: list[dict]) -> dict:
    out = {}
    for gpu in sorted({r["gpu"] for r in recs if r["gpu"]}):
        rs = [r for r in recs if r["gpu"] == gpu and "superseded" not in r and r["n_samples"]]
        if not rs:
            continue
        caps = {r["file"]: r["share_sw_power_cap"] for r in rs}
        n_all = sum(r["n_samples"] - r["n_samples_unparsed_reasons"] for r in rs)
        n_cap = sum(r["n_samples_sw_power_cap"] for r in rs)
        out[gpu] = {
            "files": sorted(caps), "sw_power_cap_share_per_file": caps,
            "sw_power_cap_share_range": [min(caps.values()), max(caps.values())],
            "sw_power_cap_share_pooled": _share(n_cap, n_all), "n_samples": n_all,
            "any_thermal_share_max_per_file": max(r["share_any_thermal"] for r in rs),
            "any_hw_slowdown_share_max_per_file": max(r["share_any_hw_slowdown"] for r in rs),
            "reason_codes_seen": sorted({k for r in rs for k in r["reason_code_counts"]}, key=lambda h: int(h, 16)),
            "sm_mhz_range": [min(r["sm_mhz"]["min"] for r in rs), max(r["sm_mhz"]["max"] for r in rs)],
            "mem_mhz_values": sorted({v for r in rs for v in r["mem_mhz_values"]}),
            "power_w_max": max(r["power_w"]["max"] for r in rs), "temp_c_max": max(r["temp_c"]["max"] for r in rs),
            "excluded_superseded_files": sorted(r["file"] for r in recs if r["gpu"] == gpu and "superseded" in r),
        }
    return out


# --------------------------------------------------------------------------- E24 (TFLA)
def _e24_quantities(table: dict) -> dict:
    chunks = sorted({c for s in table.values() for c in s["median_ms"]})
    statics = {c: E24.regret_of(table, lambda s, c=c: c) for c in chunks}
    full = [c for c in chunks if statics[c] and statics[c]["n"] == len(table)]
    best = min(full, key=lambda c: statics[c]["mean_pct"]) if full else None
    b1 = [s for s in table.values() if s["batch"] == 1]
    bb = [s for s in table.values() if s["batch"] != 1]
    return {"n_shapes": len(table),
            "oracle_counts": {str(k): v for k, v in sorted(Counter(s["oracle"] for s in table.values()).items())},
            "batch1_oracle_counts": {str(k): v for k, v in sorted(Counter(s["oracle"] for s in b1).items())},
            "batched_oracle_counts": {str(k): v for k, v in sorted(Counter(s["oracle"] for s in bb).items())},
            "library_default_128": statics.get(E24.LIBRARY_DEFAULT_CHUNK),
            "best_static": {"chunk": best, **(statics.get(best) or {})} if best else None,
            "table_by_batch_in_sample": E24.regret_of(table, E24.best_table(table, lambda s: s["batch"]))}


def _e24_eval(full: dict, kept: list[dict]) -> tuple[dict, dict]:
    tab = E24.shape_table(kept)
    if not tab:
        return {}, {}
    sub = {k: full[k] for k in tab}
    qa, qv = _e24_quantities(sub), _e24_quantities(tab)
    scal = {"oracle_agreement_exact": float(np.mean([tab[k]["oracle"] == full[k]["oracle"] for k in tab])),
            "oracle_in_all_data_tie_set": float(np.mean([tab[k]["oracle"] in full[k]["tied"] for k in tab])),
            "all_data_regret_of_retained_oracle_mean_pct": float(np.mean(
                [100 * (full[k]["median_ms"][tab[k]["oracle"]] / full[k]["median_ms"][full[k]["oracle"]] - 1) for k in tab])),
            "delta_pp_library_default_128_mean": _delta(qa["library_default_128"], qv["library_default_128"]),
            "delta_pp_best_static_mean": _delta(qa["best_static"], qv["best_static"]),
            "delta_pp_table_by_batch_mean": _delta(qa["table_by_batch_in_sample"], qv["table_by_batch_in_sample"])}
    detail = {"retained_data_same_shapes": qv, "all_data_same_shapes": qa,
              "changed_oracles": [{"layout": k[0], "L": k[1], "batch": k[2], "all_data": full[k]["oracle"],
                                   "retained": tab[k]["oracle"], "retained_in_all_data_ties": tab[k]["oracle"] in full[k]["tied"]}
                                  for k in sorted(tab) if tab[k]["oracle"] != full[k]["oracle"]]}
    return scal, detail


def stability_e24(path: Path, n_perm: int) -> dict:
    rows = [r for r in read_jsonl(path) if r.get("kind") == "timing"]
    keyfn = lambda r: (r["layout"], r["L"], r["batch"])  # noqa: E731
    full = E24.shape_table(rows)
    res = {"file": path.name, "conventions": "round median of the per-round mean per chunk; oracle and tie set "
                                             "(95% CI over rounds overlaps) as in the TFLA analysis",
           "all_data": _e24_quantities(full), "variants": {}}
    for name, keep in VARIANTS.items():
        kept = [r for r in rows if keep(*row_flags(r))]
        v = {"rule": VARIANT_TEXT[name], **kept_counts(rows, keep, keyfn)}
        if kept:
            _, v["detail"] = _e24_eval(full, kept)
            v["statistics"] = matched_control(rows, keep, keyfn, lambda sub: _e24_eval(full, sub)[0], n_perm,
                                              f"{path.name}|{name}")
        res["variants"][name] = v
    res["round_pairs"] = round_pair_diagnostic(rows, keyfn)
    res["cap_share_of_blocks"] = cap_breakdown(rows, {"batch": lambda r: r["batch"], "layout": lambda r: r["layout"],
                                                      "L": lambda r: r["L"]})
    res["not_recomputed"] = ("leave-one-shape-out cost model: refitting on a subset of shapes is not comparable "
                             "with the 63-shape fit")
    return res


# --------------------------------------------------------------------------- E12 (tensor-parallel widths)
def _e12_table(rows: list[dict], chunks: list[int]) -> pd.DataFrame:
    """Same rows and columns as ``e12_analysis.shape_table`` (checked against it), without the pandas pivot."""
    by = _by(rows, lambda r: (r["d"], r["L"], r["batch"]))
    out = []
    for (d, L, b) in sorted(by):
        rs = sorted(by[(d, L, b)], key=lambda r: r["round"])
        if any(str(c) not in r["cells"] for r in rs for c in chunks):
            continue
        R = np.array([[r["cells"][str(c)]["mean_ms"] for c in chunks] for r in rs])
        n = R.shape[0]
        mu = R.mean(0)
        half = t_crit(n - 1) * R.std(0, ddof=1) / math.sqrt(n) if n > 1 else np.zeros(len(chunks))
        o = int(mu.argmin())
        tied = [chunks[i] for i in range(len(chunks)) if mu[i] - half[i] <= mu[o] + half[o]]
        pr = R.argmin(1)
        out.append({"d": int(d), "L": int(L), "batch": int(b), "rounds": n, "lat": mu, "half": half,
                    "oracle": chunks[o], "tied": tied, "rounds_agree": bool((pr == pr[0]).all()),
                    "spread": float(mu.max() / mu.min()), "R": R})
    return pd.DataFrame(out)


def _e12_selectors(tab: pd.DataFrame, chunks: list[int], widths: set, cost_model: dict | None) -> dict:
    sel = {"cost_model": cost_model, "calibration_width_table": E12.table_choice(tab, lambda d: 2048)}
    regs = {c: np.mean([r.lat[chunks.index(c)] / r.lat.min() - 1 for r in tab.itertuples()]) for c in chunks}
    best = min(regs, key=regs.get)
    sel["best_static"] = {(r.d, r.L, r.batch): best for r in tab.itertuples()}
    unsh = {}
    for model, full in MODEL_WIDTHS.items():          # pooled unsharded table exactly as e12_analysis
        for tp in E12.TP_DEGREES[1:]:
            d = full // tp
            if full % tp == 0 and d in widths and full in widths:
                unsh.update(E12.table_choice(tab[tab["d"].isin([d, full])], lambda _d, full=full, d=d: full if _d == d else None))
    sel["unsharded_table"] = unsh
    return {"choices": sel, "best_static_chunk": int(best)}


def _e12_per_shape(tab: pd.DataFrame, chunks: list[int], choice: dict | None, keys: set) -> dict:
    out = {}
    if not choice:
        return out
    for r in tab.itertuples():
        k = (r.d, r.L, r.batch)
        if k in keys and k in choice:
            out[k] = (100 * (r.lat[chunks.index(choice[k])] / r.lat.min() - 1), choice[k] in r.tied)
    return out


def _cross_round(R: np.ndarray) -> float:
    """Table retuned at the shard width, scored on a held-out round (as in the paper's E12 numbers)."""
    regs = []
    for r in range(R.shape[0]):
        j = int(np.delete(R, r, 0).mean(0).argmin())
        regs.append(R[r, j] / R[r].min() - 1)
    return float(np.mean(regs))


E12_SELECTORS = ("unsharded_table", "calibration_width_table", "cost_model", "best_static")


def _e12_eval(ctx: dict, kept: list[dict]) -> tuple[dict, dict]:
    chunks, tab_all, sel_all, shard_keys = ctx["chunks"], ctx["tab_all"], ctx["sel_all"], ctx["shard_keys"]
    tab_v = _e12_table(kept, chunks)
    if tab_v.empty:
        return {}, {}
    o_all = ctx["oracle_all"]
    keys_v = [(r.d, r.L, r.batch) for r in tab_v.itertuples()]
    lat_all = ctx["lat_all"]
    scal = {"oracle_agreement_exact": float(np.mean([r.oracle == o_all[k][0] for r, k in zip(tab_v.itertuples(), keys_v)])),
            "oracle_in_all_data_tie_set": float(np.mean([r.oracle in o_all[k][1] for r, k in zip(tab_v.itertuples(), keys_v)])),
            "all_data_regret_of_retained_oracle_mean_pct": float(np.mean(
                [100 * (lat_all[k][chunks.index(r.oracle)] / lat_all[k].min() - 1) for r, k in zip(tab_v.itertuples(), keys_v)]))}
    sel_v = _e12_selectors(tab_v, chunks, ctx["widths"], sel_all["choices"]["cost_model"])
    kv = set(keys_v) & shard_keys
    detail = {"best_static_chunk_retained": sel_v["best_static_chunk"], "pooled_shards": {},
              "round_argmin_agreement_retained": (float(tab_v.loc[tab_v["rounds"] > 1, "rounds_agree"].mean())
                                                  if (tab_v["rounds"] > 1).any() else None)}
    for n in E12_SELECTORS:
        pa = _e12_per_shape(tab_all, chunks, sel_all["choices"].get(n), kv)
        pv = _e12_per_shape(tab_v, chunks, sel_v["choices"].get(n), kv)
        common = sorted(set(pa) & set(pv))
        ra = _summ([pa[k][0] for k in common], [pa[k][1] for k in common])
        rv = _summ([pv[k][0] for k in common], [pv[k][1] for k in common])
        detail["pooled_shards"][n] = {"all_data_same_shapes": ra, "retained_data": rv}
        scal[f"delta_pp_{n}_mean"] = _delta(ra, rv)
    Rv = {(r.d, r.L, r.batch): r.R for r in tab_v.itertuples()}
    ck = sorted(k for k in kv if Rv[k].shape[0] >= 2)
    ra = _summ([100 * _cross_round(ctx["R_all"][k]) for k in ck])
    rv = _summ([100 * _cross_round(Rv[k]) for k in ck])
    detail["pooled_shards"]["retuned_at_shard_width_cross_round"] = {
        "shapes": "shard shapes with at least two retained rounds", "all_data_same_shapes": ra, "retained_data": rv}
    scal["delta_pp_retuned_at_shard_width_cross_round_mean"] = _delta(ra, rv)
    return scal, detail


def stability_e12(path: Path, host: str, in_dir: Path, n_perm: int) -> dict:
    rows = [r for r in read_jsonl(path) if r.get("kind") == "timing"]
    keyfn = lambda r: (r["d"], r["L"], r["batch"])  # noqa: E731
    res = {"file": path.name, "conventions": "round mean per chunk; oracle and tie set as in the E12 analysis; "
                                             "selectors pooled over the shard shapes (tensor-parallel degree > 1); "
                                             "frozen cost-model choices (they do not depend on these timings)",
           "families": {}}
    for fam in sorted({r["family"] for r in rows}):
        chunks = list(MAMBA1_CHUNKS) if fam == "triton" else list(REF_CHUNKS)
        fam_rows = [r for r in rows if r["family"] == fam]
        tab_all = _e12_table(fam_rows, chunks)
        # consistency with the E12 analysis' own shape table
        ref = E12.shape_table(E12.load(path).query("family == @fam"), chunks).reset_index(drop=True)
        same = (len(ref) == len(tab_all) and (ref["oracle"].to_numpy() == tab_all["oracle"].to_numpy()).all()
                and all(list(a) == list(b) for a, b in zip(ref["tied"], tab_all["tied"]))
                and all(np.allclose(a, b) for a, b in zip(ref["lat"], tab_all["lat"])))
        widths = set(tab_all["d"])
        shards = {d for d, lst in E12.shard_map(widths).items() if any(tp > 1 for _, tp in lst)}
        shard_keys = {(r.d, r.L, r.batch) for r in tab_all.itertuples() if r.d in shards}
        try:
            cm = (E12.triton_model_choice if fam == "triton" else E12.ref_model_choice)(tab_all, chunks, host, in_dir)
            cm_note = None
        except (FileNotFoundError, KeyError) as exc:
            cm, cm_note = None, f"cost model unavailable: {exc!r}"
        sel_all = _e12_selectors(tab_all, chunks, widths, cm)
        ctx = {"chunks": chunks, "tab_all": tab_all, "sel_all": sel_all, "shard_keys": shard_keys, "widths": widths,
               "oracle_all": {(r.d, r.L, r.batch): (r.oracle, r.tied) for r in tab_all.itertuples()},
               "lat_all": {(r.d, r.L, r.batch): r.lat for r in tab_all.itertuples()},
               "R_all": {(r.d, r.L, r.batch): r.R for r in tab_all.itertuples()}}
        pooled_all = {}
        for n in E12_SELECTORS:
            ps = _e12_per_shape(tab_all, chunks, sel_all["choices"].get(n), shard_keys)
            pooled_all[n] = _summ([v[0] for v in ps.values()], [v[1] for v in ps.values()])
        pooled_all["retuned_at_shard_width_cross_round"] = _summ([100 * _cross_round(ctx["R_all"][k]) for k in sorted(shard_keys)])
        fres = {"n_shapes": int(len(tab_all)), "n_shard_shapes": len(shard_keys),
                "shape_table_matches_e12_analysis": bool(same), "cost_model_note": cm_note,
                "all_data": {"best_static_chunk": sel_all["best_static_chunk"],
                             "round_argmin_agreement": float(tab_all["rounds_agree"].mean()),
                             "pooled_shards": pooled_all}, "variants": {}}
        for vname, keep in VARIANTS.items():
            kept = [r for r in fam_rows if keep(*row_flags(r))]
            v = {"rule": VARIANT_TEXT[vname], **kept_counts(fam_rows, keep, keyfn)}
            if kept:
                _, v["detail"] = _e12_eval(ctx, kept)
                v["statistics"] = matched_control(fam_rows, keep, keyfn, lambda sub: _e12_eval(ctx, sub)[0], n_perm,
                                                  f"{path.name}|{fam}|{vname}")
            fres["variants"][vname] = v
        fres["round_pairs"] = round_pair_diagnostic(fam_rows, keyfn)
        fres["cap_share_of_blocks"] = cap_breakdown(fam_rows, {"batch": lambda r: r["batch"], "d": lambda r: r["d"],
                                                               "L": lambda r: r["L"]})
        res["families"][fam] = fres
    return res


# --------------------------------------------------------------------------- E17c (production kernels)
def _e17_claims(syn: dict) -> dict:
    cells = syn["cells"]
    ssd_big = [c for c in cells if c["family"] == "ssd" and c["tokens"] > 8192]
    gdn = [c for c in cells if c["family"] == "gdn"]
    return {"n_cells": len(cells),
            "ssd_cells_above_8192_tokens": len(ssd_big),
            "ssd_cells_above_8192_tokens_best_256": sum(1 for c in ssd_big if c["best_chunk"] == 256),
            "gdn_cells": len(gdn), "gdn_cells_best_64": sum(1 for c in gdn if c["best_chunk"] == 64),
            "per_shape": {f"{s['family']}:{s['shape']}": {"n_cells": s["n_cells"], "best_static_chunk": s["best_static_chunk"],
                                                           "best_static_regret_pct": s["best_static_regret_pct"],
                                                           "best_chunk_counts": s["best_chunk_counts"]}
                          for s in syn["shapes"]}}


def _e17_eval(ctx: dict, kept: list[dict]) -> tuple[dict, dict]:
    if not kept:
        return {}, {}
    keyfn, best_all = ctx["keyfn"], ctx["best_all"]
    syn_v = E17.synthetic(kept, ctx["floor"], ctx["tie_pct"], ctx["floor_factor"])
    cv = {(c["family"], c["shape"], c["L"], c["batch"]): c for c in syn_v["cells"]}
    syn_sub = E17.synthetic([r for r in ctx["rows"] if keyfn(r) in cv], ctx["floor"], ctx["tie_pct"], ctx["floor_factor"])
    ca, cr = _e17_claims(syn_sub), _e17_claims(syn_v)
    scal = {"oracle_agreement_exact": float(np.mean([cv[k]["best_chunk"] == best_all[k]["best_chunk"] for k in cv])),
            "oracle_within_tie_margin_of_all_data": float(np.mean([cv[k]["best_chunk"] in best_all[k]["tied_margin"] for k in cv])),
            "oracle_in_all_data_ci_tie_set": float(np.mean([cv[k]["best_chunk"] in best_all[k]["tied_ci"] for k in cv])),
            "all_data_regret_of_retained_oracle_mean_pct": float(np.mean(
                [100 * (best_all[k]["mean_ms"][str(cv[k]["best_chunk"])] / best_all[k]["best_ms"] - 1) for k in cv])),
            "ssd_above_8192_share_best_256": _share(cr["ssd_cells_above_8192_tokens_best_256"], cr["ssd_cells_above_8192_tokens"]),
            "gdn_share_best_64": _share(cr["gdn_cells_best_64"], cr["gdn_cells"])}
    for s, va in ca["per_shape"].items():
        vr = cr["per_shape"].get(s)
        if vr:
            scal[f"delta_pp_best_static_mean_{s}"] = vr["best_static_regret_pct"]["mean"] - va["best_static_regret_pct"]["mean"]
    return scal, {"retained_data": cr, "all_data_same_cells": ca}


def stability_e17(path: Path, n_perm: int, tie_pct: float = 2.0, floor_factor: float = 2.0) -> dict:
    meta = next(iter(read_jsonl(path, kind="meta")), {})
    floor = (meta.get("empty_kernel") or {}).get("mean_ms")
    rows = [r for r in read_jsonl(path) if r.get("kind") == "timing"]
    keyfn = lambda r: (r["family"], r["shape"], r["L"], r["batch"])  # noqa: E731
    syn_all = E17.synthetic(rows, floor, tie_pct, floor_factor)
    ctx = {"rows": rows, "keyfn": keyfn, "floor": floor, "tie_pct": tie_pct, "floor_factor": floor_factor,
           "best_all": {(c["family"], c["shape"], c["L"], c["batch"]): c for c in syn_all["cells"]}}
    res = {"file": path.name, "conventions": f"round mean per chunk; tie margin {tie_pct}% as in the E17 analysis",
           "launch_floor_ms": floor, "all_data": _e17_claims(syn_all), "variants": {}}
    for name, keep in VARIANTS.items():
        kept = [r for r in rows if keep(*row_flags(r))]
        v = {"rule": VARIANT_TEXT[name], **kept_counts(rows, keep, keyfn)}
        if kept:
            _, v["detail"] = _e17_eval(ctx, kept)
            v["statistics"] = matched_control(rows, keep, keyfn, lambda sub: _e17_eval(ctx, sub)[0], n_perm,
                                              f"{path.name}|{name}")
        res["variants"][name] = v
    res["round_pairs"] = round_pair_diagnostic(rows, keyfn)
    res["cap_share_of_blocks"] = cap_breakdown(rows, {"family": lambda r: r["family"], "batch": lambda r: r["batch"],
                                                      "L": lambda r: r["L"]})
    return res


# --------------------------------------------------------------------------- E9 (placement control, per-cell state)
def _e9_decision(rows: list[dict]) -> dict:
    """Input-agnostic chunk per (design, L, mode) (median over every retained input-round cell) and each input's own
    best chunk (median over its retained rounds): the decision rule of the value-invariance analysis."""
    grp = defaultdict(lambda: defaultdict(list))
    per_in = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grp[(r["family"], r["L"], r["mode"])][r["chunk"]].append(r["mean_ms"])
        per_in[(r["family"], r["L"], r["mode"])][(r["input"], r["chunk"])].append(r["mean_ms"])
    out = {}
    for g, by_c in grp.items():
        med = {c: statistics.median(v) for c, v in by_c.items()}
        cstar = min(med, key=med.get)
        by_in = defaultdict(dict)
        for (inp, c), v in per_in[g].items():
            by_in[inp][c] = statistics.median(v)
        regs, own = [], {}
        for inp, mi in by_in.items():
            b = min(mi, key=mi.get)
            own[inp] = b
            if cstar in mi:
                regs.append(100 * (mi[cstar] / mi[b] - 1))
        out[g] = {"input_agnostic_chunk": cstar, "medians": med, "input_medians": dict(by_in), "own_best": own,
                  "regret_mean_pct": float(np.mean(regs)) if regs else None,
                  "regret_max_pct": float(np.max(regs)) if regs else None}
    return out


def _e9_eval(dec_all: dict, kept: list[dict]) -> tuple[dict, dict]:
    dec_v = _e9_decision(kept)
    common = [k for k in dec_all if k in dec_v]
    if not common:
        return {}, {}
    pairs = [(k, i) for k in common for i in dec_v[k]["own_best"] if i in dec_all[k]["own_best"]]
    ra = [dec_all[k]["regret_mean_pct"] for k in common if dec_v[k]["regret_mean_pct"] is not None]
    rv = [dec_v[k]["regret_mean_pct"] for k in common if dec_v[k]["regret_mean_pct"] is not None]
    # the cap depends on the chunk (high-power chunks at the long length are capped in every round), so a chunk can
    # vanish from the retained data; the ranking comparison is therefore made among the chunks that remain
    best_r = {k: min(dec_v[k]["medians"], key=dec_all[k]["medians"].get) for k in common}
    own_r = {(k, i): min(dec_v[k]["input_medians"][i], key=dec_all[k]["input_medians"][i].get) for k, i in pairs}
    med = lambda k, c: dec_all[k]["medians"][c]  # noqa: E731
    imed = lambda k, i, c: dec_all[k]["input_medians"][i][c]  # noqa: E731
    scal = {"share_decisions_all_data_chunk_retained": float(np.mean([dec_all[k]["input_agnostic_chunk"] in dec_v[k]["medians"]
                                                                       for k in common])),
            "input_agnostic_chunk_agreement_within_retained_chunks": float(np.mean(
                [dec_v[k]["input_agnostic_chunk"] == best_r[k] for k in common])),
            "all_data_regret_vs_best_retained_chunk_mean_pct": float(np.mean(
                [100 * (med(k, dec_v[k]["input_agnostic_chunk"]) / med(k, best_r[k]) - 1) for k in common])),
            "own_best_chunk_agreement_within_retained_chunks": float(np.mean(
                [dec_v[k]["own_best"][i] == own_r[(k, i)] for k, i in pairs])) if pairs else None,
            "input_agnostic_chunk_agreement_unrestricted": float(np.mean(
                [dec_v[k]["input_agnostic_chunk"] == dec_all[k]["input_agnostic_chunk"] for k in common])),
            "all_data_regret_of_retained_input_agnostic_chunk_unrestricted_mean_pct": float(np.mean(
                [100 * (med(k, dec_v[k]["input_agnostic_chunk"]) / med(k, dec_all[k]["input_agnostic_chunk"]) - 1)
                 for k in common])),
            "own_best_chunk_agreement_unrestricted": float(np.mean(
                [dec_v[k]["own_best"][i] == dec_all[k]["own_best"][i] for k, i in pairs])) if pairs else None,
            "all_data_regret_of_retained_own_best_unrestricted_mean_pct": float(np.mean(
                [100 * (imed(k, i, dec_v[k]["own_best"][i]) / imed(k, i, dec_all[k]["own_best"][i]) - 1)
                 for k, i in pairs])) if pairs else None,
            "delta_pp_regret_of_input_agnostic_chunk_mean": (float(np.mean(rv)) - float(np.mean(ra))) if rv else None}
    detail = {"decisions_compared": len(common), "input_decisions_compared": len(pairs),
              "changed_decisions": [{"design": k[0], "L": k[1], "mode": k[2], "all_data": dec_all[k]["input_agnostic_chunk"],
                                     "retained": dec_v[k]["input_agnostic_chunk"], "retained_chunks": sorted(dec_v[k]["medians"]),
                                     "all_data_best_among_retained_chunks": best_r[k],
                                     "all_data_regret_of_retained_pct": 100 * (dec_all[k]["medians"][dec_v[k]["input_agnostic_chunk"]]
                                                                              / dec_all[k]["medians"][dec_all[k]["input_agnostic_chunk"]] - 1)}
                                    for k in common if dec_v[k]["input_agnostic_chunk"] != dec_all[k]["input_agnostic_chunk"]],
              "regret_of_input_agnostic_chunk_mean_pct": {"all_data": float(np.mean(ra)) if ra else None,
                                                          "retained_data": float(np.mean(rv)) if rv else None}}
    return scal, detail


def _cell_latency_effect(rows: list[dict]) -> dict:
    """Per (design, L, mode, input, chunk) with capped and uncapped rounds: median capped / median uncapped - 1."""
    by = defaultdict(lambda: {True: [], False: []})
    for r in rows:
        b, a = row_flags(r)
        by[(r["family"], r["L"], r["mode"], r["input"], r["chunk"])][b or a].append(r["mean_ms"])
    d = [100 * (statistics.median(v[True]) / statistics.median(v[False]) - 1) for v in by.values() if v[True] and v[False]]
    if not d:
        return {"n_cells": 0}
    return {"definition": "per (design, length, mode, input, chunk) with both statuses: median latency of the rounds "
                          "with a capped reading over that of the other rounds, minus one",
            "n_cells": len(d), "median_pct": float(np.median(d)), "p05_pct": float(np.percentile(d, 5)),
            "p95_pct": float(np.percentile(d, 95)), "share_capped_slower": float(np.mean(np.asarray(d) > 0))}


def stability_e9(path: Path, in_dir: Path, n_perm: int) -> dict:
    rows = [r for r in read_jsonl(path) if r.get("kind") == "timing"]
    dec_all = _e9_decision(rows)
    groupfn = lambda r: (r["family"], r["L"], r["mode"], r["input"], r["chunk"])  # noqa: E731
    res = {"file": path.name, "conventions": "per-cell readings; decision rule of the value-invariance analysis "
                                             "(input-agnostic chunk = argmin of the median over input-round cells; "
                                             "own best chunk of an input = argmin of its median over rounds)",
           "n_cells": len(rows), "n_decisions": len(dec_all), "variants": {}}
    stored = in_dir / path.name.replace(".jsonl", ".json")
    if stored.exists():
        try:
            st = json.loads(stored.read_text())
            ref = {(d["family"], d["L"], d["mode"]): d for fam in st["designs"].values() for d in fam["decision"]}
            res["all_data_reproduces_stored_decisions"] = bool(set(ref) == set(dec_all) and all(
                ref[k]["input_agnostic_best_chunk"] == v["input_agnostic_chunk"]
                and abs(ref[k]["regret_of_input_agnostic_chunk_pct"]["mean"] - v["regret_mean_pct"]) < 1e-6
                and {i: int(c) for i, c in ref[k]["own_best_chunk_per_input"].items()} == v["own_best"]
                for k, v in dec_all.items()))
        except (KeyError, ValueError, TypeError) as exc:
            res["all_data_reproduces_stored_decisions"] = f"not checked: {exc!r}"
    res["all_data"] = {
        "input_agnostic_chunks": {f"{k[0]}|L{k[1]}|{k[2]}": v["input_agnostic_chunk"] for k, v in sorted(dec_all.items())},
        "regret_of_input_agnostic_chunk_mean_pct": float(np.mean([v["regret_mean_pct"] for v in dec_all.values()])),
        "regret_of_input_agnostic_chunk_max_pct": float(np.max([v["regret_max_pct"] for v in dec_all.values()]))}
    for name, keep in VARIANTS.items():
        kept = [r for r in rows if keep(*row_flags(r))]
        v = {"rule": VARIANT_TEXT[name], "cells_kept": len(kept), "cells_total": len(rows)}
        _, v["detail"] = _e9_eval(dec_all, kept)
        v["statistics"] = matched_control(rows, keep, groupfn, lambda sub: _e9_eval(dec_all, sub)[0], n_perm,
                                          f"{path.name}|{name}")
        res["variants"][name] = v
    res["latency_effect"] = _cell_latency_effect(rows)
    res["cap_share_of_cells_by_design_length_chunk"] = {
        k: v for k, v in cap_breakdown(rows, {"cell": lambda r: f"{r['family']}|L{r['L']}|c{r['chunk']}"})["cell"].items()}
    res["not_recomputed"] = ("input-effect F tests and TOST: they need complete inputs x rounds blocks, which the "
                             "exclusion breaks")
    return res


# --------------------------------------------------------------------------- E10 (routed, per-run state)
def stability_e10(path: Path) -> dict:
    from scipy.stats import kendalltau

    rows = [r for r in read_jsonl(path) if r.get("kind") == "run"]
    flags = [row_flags(r)[0] for r in rows]
    capped = [r for r, f in zip(rows, flags) if f]
    res = {"file": path.name, "conventions": "per length, mean over rounds of each configuration (as the routed "
                                             "analysis); one reading per generate call; warm-up runs not used",
           "n_runs": len(rows), "n_runs_capped": len(capped),
           "capped_runs": [{"config": r["config"], "L": r["L"], "round": r["round"],
                            "reason": hex(_reason(r["gpu_state"])), "sm_mhz": r["gpu_state"].get("sm_mhz"),
                            "power_w": r["gpu_state"].get("power_w"), "temp_c": r["gpu_state"].get("temp_c")}
                           for r in capped], "lengths": {}}
    for L in sorted({r["L"] for r in rows}):
        out = {}
        for metric in ("prefill_ms", "total_ms"):
            def means(rs):
                by = defaultdict(list)
                for r in rs:
                    by[r["config"]].append(r[metric])
                return {c: statistics.fmean(v) for c, v in by.items()}
            ma = means([r for r in rows if r["L"] == L])
            mv = means([r for r, f in zip(rows, flags) if r["L"] == L and not f])
            cfg = sorted(ma)
            st_a = {c: ma[c] for c in cfg if c.startswith("static_chunk_")}
            st_v = {c: mv[c] for c in cfg if c.startswith("static_chunk_")}
            out[metric] = {"n_configs": len(cfg), "kendall_tau_all_vs_uncapped": float(kendalltau([ma[c] for c in cfg],
                                                                                                  [mv[c] for c in cfg])[0]),
                           "same_order": sorted(cfg, key=ma.get) == sorted(cfg, key=mv.get),
                           "max_abs_change_pct": float(max(abs(mv[c] / ma[c] - 1) * 100 for c in cfg)),
                           "best_static_all": min(st_a, key=st_a.get), "best_static_uncapped": min(st_v, key=st_v.get)}
        res["lengths"][str(L)] = out
    return res


# --------------------------------------------------------------------------- data sets without state
def no_state_datasets(in_dir: Path, host_tag: str) -> list[str]:
    return [p.name for p in sorted(in_dir.iterdir())
            if p.is_file() and host_tag in p.name and p.suffix in (".jsonl", ".csv")
            and "throttle_reasons" not in p.read_text(encoding="utf-8", errors="ignore")]


# --------------------------------------------------------------------------- summary and main
def _stat(v: dict, name: str) -> dict | None:
    s = (v.get("statistics") or {}).get(name)
    if not s:
        return None
    return {k: s.get(k) for k in ("cap_rule", "random_mean", "random_p025", "random_p975", "share_random_le",
                                  "share_random_ge")}


def summary(res: dict) -> dict:
    out = {"sw_power_cap_share_rtx3090": {r["file"]: r["share_sw_power_cap"] for r in res["files"]
                                          if r["gpu"] and "3090" in r["gpu"] and "superseded" not in r},
           "by_gpu_range": {g: v["sw_power_cap_share_range"] for g, v in res["by_gpu"].items()}, "stability": {}}
    st = res["stability"]
    for label in ("E24_tfla_device_time", "E24_tfla_launch_inclusive", "E17c_production_kernels_rtx3090"):
        s = st.get(label, {})
        out["stability"][label] = {vn: {"rows_kept": v.get("rows_kept"), "rows_total": v.get("rows_total"),
                                        "shapes_kept": v.get("shapes_kept"),
                                        "oracle_agreement_exact": _stat(v, "oracle_agreement_exact"),
                                        "all_data_regret_of_retained_oracle_mean_pct":
                                            _stat(v, "all_data_regret_of_retained_oracle_mean_pct"),
                                        "delta_pp_library_default_128_mean": _stat(v, "delta_pp_library_default_128_mean")}
                                   for vn, v in s.get("variants", {}).items()}
    for fam, f in st.get("E12_tp_widths_rtx3090", {}).get("families", {}).items():
        out["stability"][f"E12_{fam}"] = {vn: {"rows_kept": v.get("rows_kept"), "rows_total": v.get("rows_total"),
                                               "oracle_agreement_exact": _stat(v, "oracle_agreement_exact"),
                                               **{f"delta_pp_{n}_mean": _stat(v, f"delta_pp_{n}_mean")
                                                  for n in (*E12_SELECTORS, "retuned_at_shard_width_cross_round")}}
                                          for vn, v in f["variants"].items()}
    e9 = st.get("E9_placement_rtx3090", {})
    out["stability"]["E9"] = {vn: {"cells_kept": v["cells_kept"], "cells_total": v["cells_total"],
                                   **{m: _stat(v, m) for m in ("share_decisions_all_data_chunk_retained",
                                                               "input_agnostic_chunk_agreement_within_retained_chunks",
                                                               "all_data_regret_vs_best_retained_chunk_mean_pct",
                                                               "own_best_chunk_agreement_within_retained_chunks")}}
                              for vn, v in e9.get("variants", {}).items()}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-perm", type=int, default=200, help="draws of the matched random control")
    args = p.parse_args(argv)
    in_dir, out_dir = args.in_dir, args.out_dir or args.in_dir
    files, skipped = state_files(in_dir)
    recs = [file_stats(f) for f in files]
    res = {"experiment": "taco-r1 power-cap audit (S9P5)", "created_utc": utc_now(), "n_perm": args.n_perm,
           "nvml_clock_event_reason_bits": {k: hex(v) for k, v in NVML_BITS.items()},
           "bit_verification": ("values of the nvmlClocksEventReason* / nvmlClocksThrottleReason* macros in NVIDIA's "
                                "nvml.h (NVML API version 13) and of the constants of nvidia-ml-py 13.610.43; "
                                "nvidia-smi clocks_event_reasons.active reports the same mask in hex"),
           "sample_definition": "one sample = one reading of the GPU state (before or after a timing block, cell or "
                                "run); shares are over all samples of a file",
           "any_thermal_mask": hex(THERMAL_MASK), "hw_slowdown_mask": hex(HW_MASK),
           "files": recs, "skipped_files": skipped, "by_gpu": gpu_summary(recs)}
    for r in recs:
        print(f"{r['file']:36s} {str(r['gpu'])[:28]:28s} n={r['n_samples']:6d} cap {100 * (r['share_sw_power_cap'] or 0):5.1f}%"
              f" thermal {100 * (r['share_any_thermal'] or 0):4.1f}% hw {100 * (r['share_any_hw_slowdown'] or 0):3.1f}%"
              f" SM {r['sm_mhz']['min']:.0f}-{r['sm_mhz']['max']:.0f} (cap {r['sm_mhz_capped']['median'] if r['sm_mhz_capped'] else '-'}"
              f" / uncap {r['sm_mhz_uncapped']['median'] if r['sm_mhz_uncapped'] else '-'})"
              + (" [superseded]" if "superseded" in r else ""))

    f = {q.name: q for q in files}
    stab = {"variants": VARIANT_TEXT}
    for name, label in (("e24_tfla_vgpu3090_r4spinb.jsonl", "E24_tfla_device_time"),
                        ("e24_tfla_vgpu3090_r4plainb.jsonl", "E24_tfla_launch_inclusive")):
        stab[label] = stability_e24(f[name], args.n_perm) if name in f else {"unavailable": f"{name} not found"}
    name = "e12_tp_widths_vgpu3090.jsonl"
    stab["E12_tp_widths_rtx3090"] = (stability_e12(f[name], "vgpu3090", in_dir, args.n_perm) if name in f
                                     else {"unavailable": f"{name} not found"})
    name = "e17_workbound_vgpu3090full.jsonl"
    stab["E17c_production_kernels_rtx3090"] = (stability_e17(f[name], args.n_perm) if name in f
                                               else {"unavailable": f"{name} not found"})
    name = "e9_placement_vgpu3090.jsonl"
    stab["E9_placement_rtx3090"] = (stability_e9(f[name], in_dir, args.n_perm) if name in f
                                    else {"unavailable": f"{name} not found"})
    name = "e10_routed_vgpu3090.jsonl"
    stab["E10_routed_rtx3090"] = stability_e10(f[name]) if name in f else {"unavailable": f"{name} not found"}
    res["stability"] = stab
    res["rtx3090_datasets_without_state"] = {
        "note": ("timed without GPU-state logging (no clock, power or throttle readings): ranking stability under "
                 "capped-sample exclusion cannot be evaluated for them. They include every per-call selector table "
                 "(captured-call sweeps, their repeated rounds and CUDA-graph replays, the reference-scan sweeps), "
                 "the value-invariance runs and the calibration and auto-tuning sweeps"),
        "files": no_state_datasets(in_dir, "vgpu3090")}
    res["summary"] = summary(res)
    path = out_dir / "taco_power_cap.json"
    write_json(path, res)
    print(json.dumps(res["summary"], indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
