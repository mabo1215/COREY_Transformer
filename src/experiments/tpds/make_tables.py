"""LaTeX tables for the Supplementary Material, generated from the TPDS result files.

Prints table bodies (tabular rows only) for:
  validation  kernel correctness per family and chunk (E1 validation rows)
  latency     latency vs chunk at batch 1, Gaussian input, act mode: mean and 95% CI half-width (E1)
  effects     fixed-chunk configurations with a significant input effect (E1 analysis, BH-adjusted)
  autotune    per-key tuned / best static / global static / entropy rule latency and tuning cost (E4)
  overhead    per-call decision cost vs scan latency (E4)
  tost        TOST equivalence of value invariance at the selected chunk and at all chunks (E1 analysis)
  model_params    fitted cost-model parameters and validation summary, both GPUs (T1)
  model_ablation  drop-one / add-one ablation of the cost model, both GPUs (T1)
  model_compile   compile facts of the default launch table, both architectures (T1)
  routed_e10  single-session interleaved routed study: paired prefill/total deltas to the best static chunk (E10)

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


def t_reference(d, host):
    """E7: Triton Mamba-1 scan at its best chunk vs the reference CUDA selective scan, per E4 key."""
    e7 = _load(d / f"e7_{host}.json")
    rows = []
    for r in sorted(e7["rows"], key=lambda r: (r["L"], r["batch"])):
        ref, best = r["ref"]["mean_ms"], r["best_static"]["mean_ms"]
        rows.append(f"{r['L']} & {r['batch']} & {ref:.3f} & {r['best_static_chunk']} & {best:.3f} & "
                    f"{best / ref:.2f} & {_sci(r['max_abs_out_diff'])} \\\\")
    return rows


def t_tost(d, host):
    """E1 TOST: configurations whose input effects are equivalent to zero within 2% / 5% (alpha = 0.05)."""
    a = _load(d / f"e1_analysis_{host}.json")
    summ = a["equivalence"]["summary"]
    gpu = {"pro6000": "RTX PRO~6000", "vgpu3090": "RTX~3090"}.get(host, host)
    rows = []
    for fam in ("mamba1", "mamba2", "gdn", "all"):
        if fam not in summ:
            continue
        b, c = summ[fam]["at_best_chunk"], summ[fam]["all_chunks"]
        name = "All kernels" if fam == "all" else SHORT[fam]
        rows.append(f"{gpu} & {name} & {b['equivalent_2pct']}/{b['n']} & {b['equivalent_5pct']}/{b['n']} & "
                    f"{b['median_bound_pct']:.1f} & {b['max_bound_pct']:.1f} & {c['equivalent_2pct']}/{c['n']} & "
                    f"{c['equivalent_5pct']}/{c['n']} & {c['median_bound_pct']:.1f} \\\\")
    return rows


MODEL_HOSTS = ("pro6000", "vgpu3090")
PARAM_TEX = {"t0": ("$t_0$", "ms", 1.0), "tau_m": ("$\\tau_m$", "$\\mu$s", 1e3), "tau_e": ("$\\tau_e$", "ns", 1e6),
             "tau_s": ("$\\tau_s$", "ns", 1e6), "tau_q": ("$\\tau_q$", "ns", 1e6), "tau_p": ("$\\tau_p$", "ns", 1e6),
             "W0": ("$W_0$", "warps", 1.0), "W1": ("$W_1$", "warps", 1.0)}


def t_model_params(d, host):
    ms = {h: _load(d / f"cost_model_{h}.json") for h in MODEL_HOSTS}
    rows = []
    for k, (tex, unit, sc) in PARAM_TEX.items():
        rows.append(f"{tex} & {unit} & " + " & ".join(f"{ms[h]['params'][k] * sc:.3g}" for h in MODEL_HOSTS) + " \\\\")
    rows.append("\\midrule")

    def line(name, fn):
        rows.append(f"{name} & & " + " & ".join(fn(ms[h]) for h in MODEL_HOSTS) + " \\\\")

    line("Fit points", lambda m: str(m["n_points"]))
    line("$R^2$ (log), fit / captured calls", lambda m: f"{m['V1']['all']['r2_log']:.3f} / {m['V2_e2_calls']['curve_fit']['r2_log']:.3f}")
    line("Median rel.\\ error, fit", lambda m: f"{100 * m['V1']['all']['median_rel_err']:.0f}\\%")
    line("Regret, no profiling (mean / p95)", lambda m: f"{m['V2_e2_calls']['mean_regret_pct']:.2f} / {m['V2_e2_calls']['p95_regret_pct']:.1f}")
    line("Tied with oracle", lambda m: f"{100 * m['V2_e2_calls']['tie_aware_accuracy']:.0f}\\%")
    for k in (2, 3):
        line(f"Regret, top-{k} profiled", lambda m, k=k: f"{m['V2_topk_profiling'][f'top{k}']['mean_regret_pct']:.2f}")
    line("Regret floor, exact-shape table", lambda m: f"{m['V2_topk_profiling']['all_chunks_profiled']['mean_regret_pct']:.2f}")
    line("Leave one key out: chunk regret (mean / max)", lambda m: f"{m['V2_leave_one_key_out']['chunk_mean_regret_pct']:.2f} / {m['V2_leave_one_key_out']['chunk_max_regret_pct']:.1f}")
    line("Leave one key out: config.\\ rank of 184 (median)", lambda m: f"{m['V2_leave_one_key_out']['config_median_rank']:.1f}")
    line("Calibration on 18 batch-1 medians", lambda m: f"{m['V2_tiny_calibration']['e2']['mean_regret_pct']:.2f}")
    line("Parameters of the other GPU", lambda m: f"{next(v for k, v in m.items() if k.startswith('V3'))['e2']['mean_regret_pct']:.2f}")
    return rows


ABL_TEX = {"drop_state_latency": "without state latency $\\tau_m$", "drop_scan_work": "without scan work $\\tau_e$",
           "drop_shuffle": "without shuffle levels $\\tau_s$", "drop_spill": "without spill terms $\\tau_q,\\tau_p$",
           "drop_spill_l1_miss": "without L1 miss of spills $\\tau_p$", "drop_issue_sharing": "without issue sharing ($W_0$)",
           "drop_latency_hiding": "without latency hiding ($W_1$)", "add_tile_fixed": "with per-tile fixed cost",
           "add_l2_miss": "with L2-miss term", "add_load_issue": "with load issue cost", "add_cross_warp": "with cross-warp exchange"}


def t_model_ablation(d, host):
    ms = {h: _load(d / f"cost_model_{h}.json") for h in MODEL_HOSTS}
    rows = ["Eq.~(\\ref{eq:cost_model}), 8 parameters & " + " & ".join(
        f"{ms[h]['V1']['all']['r2_log']:.3f} & {ms[h]['V2_e2_calls']['mean_regret_pct']:.2f}" for h in MODEL_HOSTS) + " \\\\",
            "\\midrule"]
    for k, name in ABL_TEX.items():
        if k == "add_tile_fixed":
            rows.append("\\midrule")
        rows.append(f"{name} & " + " & ".join(
            f"{ms[h]['ablation'][k]['fit']['r2_log']:.3f} & {ms[h]['ablation'][k]['e2_mean_regret_pct']:.2f}"
            for h in MODEL_HOSTS) + " \\\\")
    return rows


def t_model_compile(d, host):
    from src.experiments.tpds.kernels import DEFAULT_MAMBA1_LAUNCH

    st = {a: {(r["aligned"], r["block_t"], r["block_d"], r["num_warps"], r["num_stages"]): r
              for r in _load(d / f"kernel_stats_sm{a}.json")["configs"]} for a in (120, 86)}
    rows = []
    for c, (bd, nw) in sorted(DEFAULT_MAMBA1_LAUNCH.items()):
        cells = []
        for al in (True, False):
            for a in (120, 86):
                r = st[a][(al, c, bd, nw, 1)]
                sp = r["spill_store_bytes"]
                cells.append(f"{r['regs']}" + (f"/{sp}" if sp else ""))
        lay = st[86][(True, c, bd, nw, 1)]["layout"]
        rows.append(f"{c} & {bd} & {nw} & {lay['size_per_thread'][1]}$\\times${lay['threads_per_warp'][1]}$\\times$"
                    f"{lay['warps_per_cta'][1]} & " + " & ".join(cells) + " \\\\")
    return rows


E10_ROWS = [("static_chunk_128", "Static 128", "none"), ("static_chunk_256", "Static 256", "none"),
            ("static_chunk_512", "Static 512", "none"), ("static_chunk_1024", "Static 1024", "none"),
            ("static_chunk_2048", "Static 2048", "none"), ("passive", "Passive (stock dispatch)", "none"),
            None,
            ("no_entropy_mid", "Midpoint, no statistic", "none"), ("random_seed_0", "Random", "uniform draw"),
            None,
            ("hist", "Full histogram", "entropy"), ("sampled_hist_s8", "Sampled histogram", "entropy, stride 8"),
            ("token_hist_s8", "Token-level histogram", "entropy, stride 8"),
            ("cheap_proxy", "Moment proxy", "$|x|$ mean / RMS"), ("variance_proxy", "Variance proxy", "variance"),
            ("kurtosis_proxy", "Kurtosis proxy", "kurtosis"),
            ("guarded_sampled_hist_s8", "Guarded sampled histogram", "entropy + guard"),
            ("learned_table", "Learned table", "sequence length")]


def _pm(st, digits=1):
    return f"${st['mean']:+.{digits}f}\\pm{st['ci95_half']:.{digits}f}$"


def t_routed_e10(d, host):
    """E10: paired differences (percent, mean +- 95% CI over rounds) to the best static chunk of each length."""
    e = _load(d / f"e10_routed_{host}.json")["lengths"]
    Ls = sorted(e, key=int)
    rows = []
    for spec in E10_ROWS:
        if spec is None:
            rows.append("\\midrule")
            continue
        key, name, dec = spec
        cells = []
        for L in Ls:
            pv = e[L]["paired_vs_best_static"]["prefill_ms"].get(key)
            cells.append(_pm(pv["diff_pct"]) if pv else "--")
        for L in (Ls[1], Ls[-1]):
            pv = e[L]["paired_vs_best_static"]["total_ms"].get(key)
            cells.append(_pm(pv["diff_pct"]) if pv else "--")
        dc = e[Ls[-1]]["decision_cost"].get(key)
        cells.append(f"{dc['hook_minus_passive_prefill_ms']['diff_ms']['mean']:.1f}" if dc else "--")
        chunks = e[Ls[-1]]["per_config"][key].get("scan_chunk_counts") or e[Ls[-1]]["per_config"][key].get("chunk_counts") or {}
        tot = sum(chunks.values()) if chunks else 0
        if chunks and len(chunks) == 1:
            ch = f"{next(iter(chunks))} (all)"
        elif chunks and len(chunks) >= 5:
            ch = "uniform"
        elif chunks:
            ch = ", ".join(f"{k}: {100 * v / tot:.0f}\\%" for k, v in sorted(chunks.items(), key=lambda kv: int(kv[0])))
        else:
            ch = "--"
        rows.append(f"{name} & {dec} & " + " & ".join(cells) + f" & {ch} \\\\")
    return rows


def t_deployment_buckets(d, host):
    """M6a: Triton Mamba-1 370M per (length bucket, batch): oracle scan time and regret of the default
    chunk, the cost model and a held-out exact-shape table, in ms per forward pass (48 layers), on the
    three GPUs (launch-inclusive; ``deployment_units.json``)."""
    k = _load(d / "deployment_units.json")["kernels"]
    gpus = ("mamba1_triton_370m_pro6000", "mamba1_triton_370m_vgpu3090", "mamba1_triton_370m_h800")
    by = {g: {(r["L_bucket"], r["batch"]): r for r in k[g]["by_bucket_batch"]} for g in gpus}
    layers = {g: k[g]["layers"] for g in gpus}

    def f(x):
        return "$<$0.01" if x < 0.005 else f"{x:.2f}"

    rows = []
    for key in sorted(by[gpus[0]]):
        cells = []
        for g in gpus:
            r = by[g].get(key)
            if r is None:
                cells += ["--"] * 4
                continue
            cells += [f(r["oracle_us"] * layers[g] / 1e3), f(r["default"]["ms_per_forward_pass"]),
                      f(r["cost_model"]["ms_per_forward_pass"]), f(r["exact_table_heldout"]["ms_per_forward_pass"])]
        rows.append(f"{key[0]} & {key[1]} & " + " & ".join(cells) + " \\\\")
    return rows


def t_tp_widths(d, host):
    """M4a / re-check P5: regret at tensor-parallel widths per (kernel, model, t) on the RTX 3090 and the H800
    (``e12_analysis_<gpu>.json``): share of shapes whose oracle moves beyond the ties, then the unsharded
    table, the table at the calibration width d=2048, the cost model (Triton only) and the best static chunk."""
    gpus = ("vgpu3090", "h800")
    fam = {g: _load(d / f"e12_analysis_{g}.json")["families"] for g in gpus}

    def f(y, k):
        return f"{y[k]['mean_regret_pct']:.1f}" if k in y else "--"

    rows = []
    for kernel, name in (("triton", "Triton"), ("ref", "Reference")):
        first_kernel = True
        for model, label in (("370m", "370M"), ("1.4b", "1.4B"), ("2.8b", "2.8B")):
            first_model = True
            for t, y0 in fam[gpus[0]][kernel]["by_model"][model].items():
                cells = []
                for g in gpus:
                    y = fam[g][kernel]["by_model"][model][t]
                    mv = y.get("share_oracle_moves_beyond_ties")
                    cells += ["--" if mv is None else f"{100 * mv:.0f}", f(y, "unsharded_table"),
                              f(y, "width2048_table"), f(y, "cost_model"), f(y, "best_static")]
                rows.append(f"{name if first_kernel else ''} & {label if first_model else ''} & {t[1:]} & {y0['d']} & "
                            + " & ".join(cells) + " \\\\")
                first_kernel = first_model = False
        if kernel == "triton":
            rows.append("\\midrule")
    return rows


def _static_by_batch(path, rmed):
    """Best single chunk and its mean per-call relative regret (%) per batch size of one E2 Mamba-1 table."""
    import numpy as np
    from src.experiments.tpds.analysis import chunk_columns, latency_matrix, load_e2, round_median_view

    df = load_e2([path])
    if rmed:
        df = round_median_view(df)
    ch = chunk_columns(df)
    lat = np.asarray(latency_matrix(df, ch))
    orc = lat.min(1)
    out = {}
    for b in sorted(df["batch"].unique()):
        m = (df["batch"] == b).to_numpy()
        reg = [100 * float(((lat[m, j] - orc[m]) / orc[m]).mean()) for j in range(len(ch))]
        j = int(np.argmin(reg))
        out[int(b)] = (ch[j], reg[j], dict(zip(ch, reg)))
    return out


def t_static_bucket(d, host):
    """Re-check A3: per Mamba-1 data set, the best single chunk per batch size with its regret, and the oracle
    headroom over the held-out (length bucket, batch) table (``bucket_headroom.json``)."""
    bh = _load(d / "bucket_headroom.json")["data_sets"]
    sets = (("pro6000", "PRO~6000", "LI", False), ("pro6000_r4spin", "PRO~6000", "dev., 3 r.", True),
            ("vgpu3090", "RTX~3090", "LI", False), ("vgpu3090_r4spin", "RTX~3090", "dev., 3 r.", True),
            ("h800", "H800", "LI", False), ("h800_r3spin", "H800", "dev.", False),
            ("vgpu3090_1p4b_r3", "RTX~3090, 1.4B", "dev.", False), ("vgpu3090_2p8b_r3", "RTX~3090, 2.8B", "dev.", False))
    rows = []
    for tag, gpu, conv, rmed in sets:
        st = _static_by_batch(d / f"e2_mamba1_{tag}.csv", rmed)
        cells = [f"{st[b][0]} ({st[b][1]:.1f})" for b in (1, 4, 16)]
        h = bh[tag]
        rows.append(f"{gpu} & {conv} & " + " & ".join(cells) + f" & {h['headroom_us']:.1f} ({h['headroom_pct']:.2f}) \\\\")
    return rows


def t_chunk_transfer(d, host):
    """Re-check A3/P7: launch-inclusive batch-1 regret (%) of the best batch-1 chunk of one GPU on the captured
    calls of each GPU (370M Mamba-1, one block)."""
    gpus = (("pro6000", "PRO~6000"), ("vgpu3090", "RTX~3090"), ("h800", "H800"))
    st = {g: _static_by_batch(d / f"e2_mamba1_{g}.csv", False)[1] for g, _ in gpus}
    rows = []
    for g, name in gpus:
        c = st[g][0]
        rows.append(f"{name} ({c}) & " + " & ".join(f"{st[h][2][c]:.1f}" for h, _ in gpus) + " \\\\")
    return rows


def _rounds_cv(csv_path):
    """Median over (call, chunk) cells of the within-round CV of a multi-round sweep: RMS of the per-round SDs over
    the mean of the round means (the convention of Table S-conventions); None for a single-round sweep."""
    import numpy as np
    from collections import defaultdict

    p = csv_path.with_name(csv_path.stem + "_rounds.jsonl")
    if not p.exists():
        return None
    acc = defaultdict(list)
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") != "round":
                continue
            for c, st in r["stats"].items():
                if st.get("mean_ms") is not None and np.isfinite(st["mean_ms"]):
                    acc[(r["call"], c)].append((st["mean_ms"], st["std_ms"]))
    n = max(len(v) for v in acc.values())
    cv = [100 * np.sqrt(np.mean([sd ** 2 for _, sd in v])) / np.mean([m for m, _ in v]) for v in acc.values() if len(v) == n]
    return float(np.median(cv))


def t_graph(d, host, source="e22_graph_analysis_vgpu3090_rounds3.json"):
    """M3b / re-check P3: Table S-graph rows (CUDA-graph replays against eager and device-time timings of the same
    captured calls, RTX 3090) from ``e22_graph_analysis`` (default: the three-round analysis, ``--rounds3``)."""
    if not (d / source).exists():
        source = "e22_graph_analysis_vgpu3090.json"
    a = _load(d / source)
    T = a["tables"]
    three = set(a.get("three_round_tables", []))
    rows_spec = (("SSD 370M", (("ssd370m_eager", "eager"), ("ssd370m_graph", "graph"), ("ssd370m_graphcold", "graph, flushed"))),
                 ("SSD 2.7B", (("ssd2p7b_device", "device"), ("ssd2p7b_graph", "graph"), ("ssd2p7b_graphcold", "graph, flushed"))),
                 ("GDN 1.4B", (("gdn1p4b_device", "device"), ("gdn1p4b_graph", "graph"), ("gdn1p4b_graphcold", "graph, flushed"))),
                 ("Mamba-1 370M", (("m1_370m_eager", "eager"), ("m1_370m_device_r4", "device, 3 rounds"), ("m1_370m_graph", "graph"))))

    def one(x):
        return "--" if x is None else f"{x:.1f}"

    def fine(x):
        return "--" if x is None else (f"{x:.2f}" if x < 2 else f"{x:.1f}")

    rows = []
    for label, items in rows_spec:
        for k, (nm, timing) in enumerate(items):
            t = T[nm]
            P = t["policies"]

            def g(key):
                return P[key]["all"]["mean_pct"] if key in P else None
            graph = "graph" in nm
            model = g("work_term_launch_default")
            mcell = "--" if model is None else (f"{model:.1f} ({g('work_term_refit_cv'):.1f})" if graph
                                                 and g("work_term_refit_cv") is not None else f"{model:.1f}")
            h = t["headroom"]["all"]
            hc = h["covered_us"]
            if nm in three:
                timing += ", 3 rounds"
            rows.append(" & ".join([label if k == 0 else "", timing, str(t["n_calls"]), f"{t['floor']['floor_us']:.0f}",
                                    f"{100 * t['floor']['share_above']:.0f}\\%",
                                    f"{_rounds_cv(d / t['source']) or t['within_block_cv_pct']['all']['cell_median']:.1f}",
                                    "--" if t["library_default"] is None else one(g("default")),
                                    one(g("best_static_insample")), fine(g("e3_shape")), fine(g("e3_shape+entropy")),
                                    one(g("e3_entropy_only")), fine(g("exact_table_heldout")), mcell,
                                    f"{hc:.2f} ({h['covered_max_key_us']:.1f})" if hc < 10
                                    else f"{hc:.1f} ({h['covered_max_key_us']:.1f})"]) + " \\\\")
    return rows


TABLES = {"routed_e10": t_routed_e10, "model_params": t_model_params, "model_ablation": t_model_ablation, "model_compile": t_model_compile,
          "tost": t_tost, "reference": t_reference, "redraw": t_redraw, "validation": t_validation, "latency": t_latency, "effects": t_effects, "autotune": t_autotune,
          "overhead": t_overhead, "deployment_buckets": t_deployment_buckets,
          "tp_widths": t_tp_widths,
          "static_bucket": t_static_bucket, "chunk_transfer": t_chunk_transfer,
          "graph": t_graph}


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
