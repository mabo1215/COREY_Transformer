"""LaTeX tables for the Supplementary Material, generated from the TPDS result files.

Prints table bodies (tabular rows only) for:
  validation  kernel correctness per family and chunk (E1 validation rows)
  latency     latency vs chunk at batch 1, Gaussian input, act mode: mean and 95% CI half-width (E1)
  effects     fixed-chunk configurations with a significant input effect (E1 analysis, BH-adjusted)
  autotune    per-key tuned / best static / global static / entropy rule latency and tuning cost (E4)
  overhead    per-call decision cost vs scan latency (E4)

Example:
    python -m src.experiments.tpds.make_tables --in-dir src/experiments/results/tpds --host pro6000 latency
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from src.experiments.tpds.common import RESULTS_DIR

FAM = {"mamba1": "Mamba-1", "mamba2": "Mamba-2 SSD", "mamba2_ssd": "Mamba-2 SSD", "gdn": "Gated DeltaNet"}
SHORT = {"mamba1": "Mamba-1", "mamba2": "Mamba-2", "mamba2_ssd": "Mamba-2", "gdn": "GDN"}  # narrow tables


def _load(p: Path):
    return json.loads(p.read_text())


def _sci(x: float) -> str:
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def _pval(p: float) -> str:
    if p < 1e-6:
        return "$<10^{-6}$"
    if p >= 1e-3:
        return f"{p:.3f}"
    m, e = f"{p:.0e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def t_validation(d, host):
    e1 = _load(d / f"e1_{host}.json")
    by = defaultdict(list)
    for v in e1["validation"]:
        by[(v["family"], v["dtype"])].append(v)
    rows = []
    for (fam, dtype), vs in by.items():
        chunks = ", ".join(str(v["chunk"]) for v in sorted(vs, key=lambda v: v["chunk"]))
        rows.append(f"{FAM[fam]} & {dtype.replace('float', 'fp').replace('bfp16', 'bf16')} & {chunks} & "
                    f"{_sci(max(v['max_abs_err'] for v in vs))} & {_sci(max(v['max_rel_err'] for v in vs))} & "
                    f"{'yes' if all(v['passed'] for v in vs) else 'no'} \\\\")
    return rows


def t_latency(d, host):
    e1 = _load(d / f"e1_{host}.json")
    cell = defaultdict(list)
    for r in e1["rows"]:
        if r.get("kind") == "timing" and r["input"] == "gaussian" and r["mode"] == "act":
            cell[(r["family"], r["L"], r["chunk"])].append(r)
    rows = []
    for fam in ("mamba1", "mamba2", "gdn"):
        for L in sorted({k[1] for k in cell if k[0] == fam}):
            chunks = sorted(k[2] for k in cell if k[0] == fam and k[1] == L)
            means = {c: statistics.fmean(r["mean_ms"] for r in cell[(fam, L, c)]) for c in chunks}
            best = min(means, key=means.get)
            parts = []
            for c in chunks:
                rs = cell[(fam, L, c)]
                half = statistics.fmean((r["ci95_hi_ms"] - r["ci95_lo_ms"]) / 2 for r in rs)
                txt = f"{c}: {means[c]:.3f}$\\pm${half:.3f}"
                parts.append(f"\\textbf{{{txt}}}" if c == best else txt)
            rows.append(f"{FAM[fam]} & {L} & " + "; ".join(parts) + " \\\\")
    return rows


def t_effects(d, host, alpha=0.05):
    a = _load(d / f"e1_analysis_{host}.json")
    rows = []
    for e in a["effects"]:
        if e.get("p_input_bh") is not None and e["p_input_bh"] < alpha:
            rows.append(f"{SHORT[e['family']]} & {e['L']} & {e['mode']} & {e['chunk']} & "
                        f"{e['input_range_pct']:.1f}\\% & {e['fastest_input']} & {e['slowest_input']} & "
                        f"{_pval(e['p_input_bh'])} \\\\")
    return rows


def t_autotune(d, host):
    s = _load(d / f"e4_{host}.json")["summary"]
    rows = []
    for fam in ("mamba1", "mamba2"):
        for k in s[fam]["per_key"]:
            rows.append(f"{FAM[fam]} & {k['L']} & {k['batch']} & {k['tuned_ms']:.3f} & "
                        f"{k['best_static_key'].split('_')[1]}: {k['best_static_ms']:.3f} & {k['global_static_ms']:.3f} & "
                        f"{k['entropy_rule_ms']:.3f} & {k['tuning_wall_s']:.1f} \\\\")
        rows.append("\\midrule")
    return rows[:-1]


def t_overhead(d, host):
    s = _load(d / f"e4_{host}.json")["summary"]
    by = defaultdict(dict)
    for o in s["overhead"]:
        by[(o["L"], o["batch"])][o["stride"]] = o
    rows = []
    for (L, b), v in sorted(by.items()):
        f, g = v[1], v[8]
        rows.append(f"{L} & {b} & {f['scan_ms']:.3f} & {f['decision_ms']:.3f} & {f['decision_ms'] / f['scan_ms']:.1f} & "
                    f"{g['decision_ms']:.3f} & {g['decision_ms'] / g['scan_ms']:.1f} \\\\")
    return rows


def e1c_valid_configs(d, host, control_suffix="_e1c", factor=1.5):
    """Mamba-1 (L, chunk) configurations of the E1c control that are GPU-bound in that run.

    The host launch path of the control run was slower than in E1, which put a latency floor
    under every short kernel. The floor is the smallest Mamba-1 median latency of the control;
    a configuration is kept if its E1 median latency is at least ``factor`` times that floor.
    """
    import statistics as st
    ctl = _load(d / f"e1_{host}{control_suffix}.json")["rows"]
    ref = _load(d / f"e1_{host}.json")["rows"]

    def medians(rows):
        m = defaultdict(list)
        for r in rows:
            if r.get("kind", "timing") == "timing" and r["family"] == "mamba1":
                m[(r["L"], r["chunk"])].append(r["mean_ms"])
        return {k: st.median(v) for k, v in m.items()}

    mc, mr = medians(ctl), medians(ref)
    floor = min(mc.values())
    keep = sorted(k for k in mc if k in mr and mr[k] >= factor * floor)
    return floor, keep, mc, mr


def t_redraw(d, host):
    """E1c control: between-distribution vs between-redraw range on the GPU-bound configurations."""
    floor, keep, mc, mr = e1c_valid_configs(d, host)
    a = _load(d / f"e1_analysis_{host}_e1c.json")
    rows = [f"% host floor {floor:.3f} ms; kept (L, chunk): {keep}"]
    for e in sorted(a["effects"], key=lambda e: (e["L"], e["chunk"], e["mode"])):
        if e["family"] != "mamba1" or (e["L"], e["chunk"]) not in keep:
            continue
        rows.append(f"{e['L']} & {e['chunk']} & {e['mode']} & {mr[(e['L'], e['chunk'])]:.3f} & "
                    f"{mc[(e['L'], e['chunk'])]:.3f} & {e['between_distribution_range_pct']:.1f}\\% & "
                    f"{e['between_redraw_range_pct_max']:.1f}\\% & {_pval(e['p_input_bh'])} \\\\")
    return rows


TABLES = {"redraw": t_redraw, "validation": t_validation, "latency": t_latency, "effects": t_effects, "autotune": t_autotune,
          "overhead": t_overhead}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--host", default="pro6000")
    p.add_argument("tables", nargs="*", default=list(TABLES))
    args = p.parse_args(argv)
    for t in args.tables:
        print(f"% ---- {t} ({args.host})")
        for row in TABLES[t](args.in_dir, args.host):
            print(row)


if __name__ == "__main__":
    main()
