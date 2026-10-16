"""Review item M5d (cycle taco-r2): input effects of the value-invariance sweep on the H800, for the main text. CPU only.

Data: the value-invariance sweep ``e1_<tag>.json`` (every chunk of each kernel timed on eight inputs: uniform,
Gaussian, Laplace, sparse 50 / 10 / 2 %, constant and captured activations; batch 1, L in {512, 2048, 8192}; two
perturbation modes, ``act`` and ``all``; three shuffled rounds). The H800 ran Mamba-1 (Triton scan) only, once
launch-inclusive (``h800``) and once in device time (``h800_r3spin``). Next to it, the RTX PRO 6000 (``pro6000``,
launch-inclusive: the value-invariance table of the main text; ``pro6000_r4spin``: device time) and the RTX 3090
(``vgpu3090``, launch-inclusive; ``vgpu3090_r3spin``, device time).

Every quantity is recomputed from the raw rows with the functions of ``e1_analysis`` (decision, effects with
Benjamini-Hochberg over the fixed-chunk tests of the tag, TOST equivalence) and checked against the stored
``e1_analysis_<tag>.json`` where it exists; the tie share under the 2% rule is ``taco_within2_ties.invariance_ties``
(share of (input, round) cells in which the input-agnostic chunk is within 2% of the cell's own best chunk; minimum
over the two modes, the column "Tied" of the value-invariance table).

Reported per tag and per kernel family (Mamba-1 for the comparison across GPUs): the number of fixed-chunk
configurations with a significant input effect (BH, alpha 0.05); the input range (slowest minus fastest input mean,
percent of the mean over inputs) maximum overall, per length and at the input-agnostic chunk; the per-input best chunk
per length and mode and whether it moves; the tie shares (2% rule; 95% CI overlap); the largest regret of the
input-agnostic chunk against an input's own best (median over rounds); the TOST summaries (median and maximum
equivalence bound, configurations equivalent at 2% and 5%). The H800 has no placement or L2 control.

Output: ``taco_r2_stats_m5d.json``.

Usage:  PYTHONPATH=. python3 -m src.experiments.tpds.taco_r2_stats_m5d_h800_inputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.experiments.tpds import e1_analysis as E1A
from src.experiments.tpds import taco_r2_stats_common as C
from src.experiments.tpds import taco_within2_ties as W2
from src.experiments.tpds.common import RESULTS_DIR

TAGS = {
    "h800": ("H800 PCIe", "launch-inclusive"),
    "h800_r3spin": ("H800 PCIe", "device time"),
    "pro6000": ("RTX PRO 6000", "launch-inclusive (value-invariance table of the main text)"),
    "pro6000_r4spin": ("RTX PRO 6000", "device time"),
    "vgpu3090": ("RTX 3090", "launch-inclusive"),
    "vgpu3090_r3spin": ("RTX 3090", "device time"),
}
ALPHA = 0.05


def analyze(rows: list[dict]) -> tuple[list, list, dict]:
    dec = E1A.decision_analysis(rows)
    eff = E1A.effect_analysis(rows)
    eq = E1A.equivalence_analysis(rows, dec, alpha=ALPHA)
    return dec, eff, eq


def family_block(fam: str, dec: list, eff: list, eq: dict, ties: dict) -> dict:
    d_f = [d for d in dec if d["family"] == fam]
    e_f = [e for e in eff if e["family"] == fam]
    c_star = {(d["L"], d["mode"]): d["input_agnostic_best_chunk"] for d in d_f}
    sig = [e for e in e_f if e.get("p_input_bh") is not None and e["p_input_bh"] < ALPHA]
    at_sel = [e for e in e_f if e["chunk"] == c_star.get((e["L"], e["mode"]))]
    rec = {"n_fixed_chunk_tests": len(e_f), "n_significant_input_effects_bh": len(sig),
           "input_range_pct": {
               "max_all_configs": _argmax(e_f), "max_significant_configs": _argmax(sig),
               "max_at_input_agnostic_chunk": _argmax(at_sel),
               "max_significant_at_input_agnostic_chunk": _argmax([e for e in at_sel if e in sig]),
               "median_all_configs": float(np.median([e["input_range_pct"] for e in e_f])),
               "median_at_input_agnostic_chunk": float(np.median([e["input_range_pct"] for e in at_sel]))},
           "by_length": {}}
    for L in sorted({d["L"] for d in d_f}):
        eL = [e for e in e_f if e["L"] == L]
        modes = {}
        for d in [d for d in d_f if d["L"] == L]:
            own = d["own_best_chunk_per_input"]
            modes[d["mode"]] = {"input_agnostic_chunk": d["input_agnostic_best_chunk"],
                                "own_best_chunk_per_input": own,
                                "distinct_own_best_chunks": sorted(set(own.values())),
                                "inputs_with_other_best_chunk": d["inputs_with_other_best_chunk"],
                                "argmin_agreement": d["argmin_agreement"],
                                "ci_overlap_tie_share": d["tie_aware_agreement"],
                                "max_regret_of_input_agnostic_chunk_pct": d["regret_of_input_agnostic_chunk_pct"]["max"],
                                "mean_regret_of_input_agnostic_chunk_pct": d["regret_of_input_agnostic_chunk_pct"]["mean"],
                                "chunk_spread": d["between_chunk_max_over_min"]}
        tk = ties["per_kernel_length"].get(f"{fam}|L{L}", {})
        rec["by_length"][str(L)] = {
            "n_tests": len(eL), "n_significant_bh": sum(1 for e in eL if e.get("p_input_bh") is not None and e["p_input_bh"] < ALPHA),
            "max_input_range_pct": _argmax(eL),
            "max_input_range_at_input_agnostic_chunk_pct": _argmax([e for e in eL if e["chunk"] == c_star.get((L, e["mode"]))]),
            "share_within_2pct_min_over_modes": tk.get("min_over_modes_share_within_2pct"),
            "share_within_2pct_by_mode": {m: v["share_within_2pct"] for m, v in tk.get("modes", {}).items()},
            "own_best_chunk_moves": any(len(m["distinct_own_best_chunks"]) > 1 for m in modes.values()),
            "max_regret_of_input_agnostic_chunk_pct": max(m["max_regret_of_input_agnostic_chunk_pct"] for m in modes.values()),
            "modes": modes}
    n_cells = sum(d["n_cells"] for d in d_f)
    rec["cells"] = {"n_cells": n_cells,
                    "ci_overlap_tie_share": sum(d["tie_aware_agreement"] * d["n_cells"] for d in d_f) / n_cells,
                    "argmin_agreement": sum(d["argmin_agreement"] * d["n_cells"] for d in d_f) / n_cells,
                    "within_2pct_tie_share_pooled": _pooled_within(ties, fam),
                    "within_2pct_min_over_modes_range": _range([v["min_over_modes_share_within_2pct"]
                                                                for k, v in ties["per_kernel_length"].items()
                                                                if k.startswith(fam + "|")]),
                    "max_regret_of_input_agnostic_chunk_pct": max(d["regret_of_input_agnostic_chunk_pct"]["max"] for d in d_f)}
    s = eq["summary"].get(fam, {})
    rec["tost"] = {"alpha": ALPHA, "all_chunks": s.get("all_chunks"), "at_input_agnostic_chunk": s.get("at_best_chunk")}
    return rec


def _argmax(es: list) -> dict | None:
    if not es:
        return None
    e = max(es, key=lambda e: e["input_range_pct"])
    return {"pct": float(e["input_range_pct"]), "L": e["L"], "mode": e["mode"], "chunk": e["chunk"],
            "fastest_input": e["fastest_input"], "slowest_input": e["slowest_input"], "p_bh": e.get("p_input_bh")}


def _range(v: list) -> list | None:
    return [float(min(v)), float(max(v))] if v else None


def _pooled_within(ties: dict, fam: str) -> float:
    tot = w = 0.0
    for k, o in ties["per_kernel_length"].items():
        if not k.startswith(fam + "|"):
            continue
        for m in o["modes"].values():
            tot += m["n_cells"]
            w += m["n_cells"] * m["share_within_2pct"]
    return w / tot if tot else None


def stored_checks(tag: str, dec, eff, eq, in_dir: Path, checks: list) -> None:
    st = C.load_json(in_dir / f"e1_analysis_{tag}.json")
    if not st:
        checks.append({"check": f"{tag}: no stored e1_analysis (recomputed only)", "match": None})
        return
    n_sig = sum(1 for e in eff if e.get("p_input_bh") is not None and e["p_input_bh"] < ALPHA)
    checks.append({"check": f"{tag}.n_significant_input_effects_bh", "computed": n_sig,
                   "stored": st["overall"]["n_significant_input_effects_bh"],
                   "match": n_sig == st["overall"]["n_significant_input_effects_bh"]})
    a = sorted((e["family"], e["L"], e["mode"], e["chunk"], round(e["input_range_pct"], 9)) for e in eff)
    b = sorted((e["family"], e["L"], e["mode"], e["chunk"], round(e["input_range_pct"], 9)) for e in st["effects"])
    checks.append({"check": f"{tag}.effects.input_range_pct (all configs)", "match": a == b})
    a = [(d["family"], d["L"], d["mode"], d["input_agnostic_best_chunk"], round(d["regret_of_input_agnostic_chunk_pct"]["max"], 9))
         for d in dec]
    b = [(d["family"], d["L"], d["mode"], d["input_agnostic_best_chunk"], round(d["regret_of_input_agnostic_chunk_pct"]["max"], 9))
         for d in st["decision"]]
    checks.append({"check": f"{tag}.decision (C*, max regret)", "match": a == b})
    checks.append({"check": f"{tag}.tost summary", "match": _close(eq["summary"], st["equivalence"]["summary"])})


def _close(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= 1e-9 * max(1.0, abs(b))
    return a == b


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv)
    out = {"item": "M5d", "meta": C.meta("taco_r2_stats_m5d_h800_inputs", args), "definitions": __doc__,
           "alpha": ALPHA, "tags": {}, "checks": [],
           "controls_note": ("No placement or L2 control was run on the H800 (the fixed-placement, multi-allocation and "
                             "flushed-L2 controls exist for the RTX PRO 6000 and the RTX 3090 only), so the H800 input "
                             "effects cannot be attributed to buffer placement or to the values.")}
    for tag, (gpu, conv) in TAGS.items():
        path = args.in_dir / f"e1_{tag}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        rows = [r for r in data["rows"] if r.get("kind") == "timing"]
        dec, eff, eq = analyze(rows)
        ties = W2.invariance_ties(data)
        stored_checks(tag, dec, eff, eq, args.in_dir, out["checks"])
        n_sig = sum(1 for e in eff if e.get("p_input_bh") is not None and e["p_input_bh"] < ALPHA)
        out["tags"][tag] = {"gpu": gpu, "convention": conv, "source": path.name,
                            "families": sorted({d["family"] for d in dec}),
                            "all_families": {"n_fixed_chunk_tests": len(eff), "n_significant_input_effects_bh": n_sig,
                                             "within_2pct_tie_share_pooled": ties["pooled_share_within_2pct"],
                                             "n_cells": ties["n_cells"]},
                            "by_family": {fam: family_block(fam, dec, eff, eq, ties) for fam in sorted({d["family"] for d in dec})}}
    out["checks_summary"] = {"n": len(out["checks"]), "n_match": sum(1 for c in out["checks"] if c["match"]),
                             "n_not_stored": sum(1 for c in out["checks"] if c["match"] is None)}
    out["headline_mamba1"] = headline(out)
    path = C.write_result("m5d", out, args.in_dir)
    print_summary(out)
    print(f"wrote {path}")


def headline(out: dict) -> dict:
    """Mamba-1 numbers per tag, side by side."""
    h = {}
    for tag, r in out["tags"].items():
        m = r["by_family"].get("mamba1")
        if not m:
            continue
        h[tag] = {"gpu": r["gpu"], "convention": r["convention"],
                  "significant_input_effects": f"{m['n_significant_input_effects_bh']} of {m['n_fixed_chunk_tests']}",
                  "max_input_range_pct": m["input_range_pct"]["max_all_configs"]["pct"],
                  "max_input_range_at": {k: m["input_range_pct"]["max_all_configs"][k] for k in ("L", "mode", "chunk")},
                  "max_input_range_at_input_agnostic_chunk_pct": m["input_range_pct"]["max_at_input_agnostic_chunk"]["pct"],
                  "max_significant_input_range_pct": (m["input_range_pct"]["max_significant_configs"] or {}).get("pct"),
                  "max_significant_input_range_at_input_agnostic_chunk_pct":
                      (m["input_range_pct"]["max_significant_at_input_agnostic_chunk"] or {}).get("pct"),
                  "within_2pct_tie_share_min_over_modes_by_L": {L: v["share_within_2pct_min_over_modes"] for L, v in m["by_length"].items()},
                  "ci_overlap_tie_share": m["cells"]["ci_overlap_tie_share"],
                  "max_regret_of_input_agnostic_chunk_pct": m["cells"]["max_regret_of_input_agnostic_chunk_pct"],
                  "own_best_chunks_by_L": {L: {mo: v["distinct_own_best_chunks"] for mo, v in d["modes"].items()}
                                           for L, d in m["by_length"].items()},
                  "input_agnostic_chunk_by_L": {L: {mo: v["input_agnostic_chunk"] for mo, v in d["modes"].items()}
                                                for L, d in m["by_length"].items()},
                  "tost_all_chunks": m["tost"]["all_chunks"], "tost_at_input_agnostic_chunk": m["tost"]["at_input_agnostic_chunk"]}
    return h


def print_summary(out: dict) -> None:
    for tag, h in out["headline_mamba1"].items():
        print(f"== {tag} ({h['gpu']}, {h['convention']}), Mamba-1")
        print(f"   significant {h['significant_input_effects']}; max range {h['max_input_range_pct']:.1f}% at {h['max_input_range_at']}; "
              f"at C* {h['max_input_range_at_input_agnostic_chunk_pct']:.2f}%; largest significant {h['max_significant_input_range_pct']}, "
              f"at C* {h['max_significant_input_range_at_input_agnostic_chunk_pct']}")
        print(f"   within-2% tie share (min over modes) by L {({k: round(100 * v, 1) for k, v in h['within_2pct_tie_share_min_over_modes_by_L'].items()})}; "
              f"CI-overlap {100 * h['ci_overlap_tie_share']:.1f}%; max regret {h['max_regret_of_input_agnostic_chunk_pct']:.2f}%")
        print(f"   C* by L {h['input_agnostic_chunk_by_L']}; own best by L {h['own_best_chunks_by_L']}")
        print(f"   TOST all {h['tost_all_chunks']}; at C* {h['tost_at_input_agnostic_chunk']}")
    for tag, r in out["tags"].items():
        print(f"   {tag} all families: {r['all_families']}")
    print("checks:", out["checks_summary"], [c["check"] for c in out["checks"] if c["match"] is False])


if __name__ == "__main__":
    main()
