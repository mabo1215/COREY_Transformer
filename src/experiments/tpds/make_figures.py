"""Paper figures for the TPDS experiments (E1-E5). Works on partial results.

Figures (IEEE single-column width 3.5 in, fonts >= 8 pt, Okabe-Ito palette
plus marker/line-style redundancy for colorblind safety):
  tpds_latency_vs_chunk.pdf   latency vs chunk per family and L, 95% CI bands (E1, Gaussian input)
  tpds_value_invariance.pdf   latency across input distributions at each chunk (E1, median over rounds)
  tpds_entropy_by_regime.pdf  activation entropy per workload regime (E2, batch 1)
  tpds_entropy_vs_oracle.pdf  entropy vs per-call oracle chunk (E2)
  tpds_regret_cdf.pdf         held-out regret CDF per policy (E3)
  tpds_breakeven.pdf          cumulative cost vs number of calls (E5)
  tpds_cost_model.pdf         measured vs cost-model latency per chunk, both GPUs (T1; full width)
  tpds_key_churn.pdf          key-churn break-even N* vs decision overhead, both GPUs (T3)

Example:
    python -m src.experiments.tpds.make_figures --in-dir src/experiments/results/tpds --host pro6000 \
        --out-dir paper/figs
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402

from src.experiments.tpds.analysis import load_e2  # noqa: E402
from src.experiments.tpds.common import RESULTS_DIR, entropy_rule_chunk  # noqa: E402

COL_W = 3.5
OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
FAMILY_LABEL = {"mamba1": "Mamba-1 selective scan", "mamba2": "Mamba-2 SSD", "gdn": "Gated DeltaNet"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "grid.color": "#d0d0d0", "lines.linewidth": 1.2,
    "lines.markersize": 4, "axes.spines.top": False, "axes.spines.right": False,
})


PNG_PREVIEW = False


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    if PNG_PREVIEW:
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def _load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _plain_log_y(ax):
    """Log y axis with plain labels; narrow ranges (< 1.5 decades) get intermediate ticks so that
    every panel carries readable values. Call after plotting."""
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    span = hi / lo if lo > 0 else 10.0
    if span < 30:
        subs = tuple(range(1, 10)) if span < 3 else (1.0, 2.0, 3.0, 5.0) if span < 10 else (1.0, 2.0, 5.0)
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=subs, numticks=20))
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _pow2_axis(ax, chunks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(chunks)
    ax.set_xticklabels([str(c) for c in chunks])
    ax.minorticks_off()


# --------------------------------------------------------------------------- E1
def fig_latency_vs_chunk(e1, out_dir):
    rows = [r for r in e1["rows"] if r["input"] == "gaussian" and r["mode"] == "act"]
    fams = [f for f in ("mamba1", "mamba2", "gdn") if any(r["family"] == f for r in rows)]
    if not fams:
        return
    fig, axes = plt.subplots(len(fams), 1, figsize=(COL_W, 1.45 * len(fams) + 0.3), squeeze=False)
    for ax, fam in zip(axes[:, 0], fams):
        rs = [r for r in rows if r["family"] == fam]
        Ls = sorted({r["L"] for r in rs})
        chunks = sorted({r["chunk"] for r in rs})
        for i, L in enumerate(Ls):
            m, lo, hi = [], [], []
            for c in chunks:
                cr = [r for r in rs if r["L"] == L and r["chunk"] == c]
                m.append(np.mean([r["mean_ms"] for r in cr]))
                lo.append(np.mean([r["ci95_lo_ms"] for r in cr]))
                hi.append(np.mean([r["ci95_hi_ms"] for r in cr]))
            m, lo, hi = map(np.array, (m, lo, hi))
            ax.plot(chunks, m, color=OKABE[i], ls=STYLES[i], marker=MARKERS[i], label=f"L={L}")
            ax.fill_between(chunks, lo, hi, color=OKABE[i], alpha=0.2, lw=0)
        _pow2_axis(ax, chunks)
        _plain_log_y(ax)
        ax.set_ylabel("latency (ms)")
        ax.set_title(FAMILY_LABEL[fam], loc="left")
        ax.grid(True, which="major", axis="y")
    axes[-1, 0].set_xlabel("chunk size")
    handles, labels = {}, {}
    for ax in axes[:, 0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    order = sorted(handles, key=lambda l: int(l.split("=")[1]))
    fig.legend([handles[l] for l in order], order, loc="lower center", ncol=len(order), frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=0.8)
    fig.tight_layout(h_pad=0.4)
    _save(fig, out_dir, "tpds_latency_vs_chunk.pdf")


def fig_value_invariance(e1, out_dir):
    rows = [r for r in e1["rows"] if r["mode"] == "act"]
    fams = [f for f in ("mamba1", "mamba2", "gdn") if any(r["family"] == f for r in rows)]
    if not fams:
        return
    fig, axes = plt.subplots(len(fams), 1, figsize=(COL_W, 1.45 * len(fams) + 0.35), squeeze=False)
    inputs_all = ["uniform", "gaussian", "laplace", "sparse50", "sparse10", "sparse02", "constant", "real"]
    for ax, fam in zip(axes[:, 0], fams):
        rs = [r for r in rows if r["family"] == fam]
        L = max(r["L"] for r in rs)
        rs = [r for r in rs if r["L"] == L]
        chunks = sorted({r["chunk"] for r in rs})
        inputs = [i for i in inputs_all if any(r["input"] == i for r in rs)]
        width = 0.7
        for j, inp in enumerate(inputs):
            xs, ys = [], []
            for k, c in enumerate(chunks):
                cr = [r["mean_ms"] for r in rs if r["chunk"] == c and r["input"] == inp]
                if cr:
                    xs.append(k + (j - (len(inputs) - 1) / 2) * width / max(len(inputs), 1))
                    ys.append(np.median(cr))  # median over rounds: robust to single-round spikes
            color = OKABE[0] if inp == "real" else "#7a7a7a"
            ax.plot(xs, ys, ls="none", marker=MARKERS[j % len(MARKERS)], color=color, mfc="none" if inp != "real" else color,
                    ms=3.5, mew=0.7, label=inp)
        ax.set_xticks(range(len(chunks)))
        ax.set_xticklabels([str(c) for c in chunks])
        _plain_log_y(ax)
        ax.set_ylabel("latency (ms)")
        ax.set_title(f"{FAMILY_LABEL[fam]}, L={L}", loc="left")
        ax.grid(True, axis="y")
    axes[-1, 0].set_xlabel("chunk size")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3,
               columnspacing=0.8)
    fig.tight_layout(h_pad=0.4)
    _save(fig, out_dir, "tpds_value_invariance.pdf")


# --------------------------------------------------------------------------- E2
def fig_entropy_by_regime(dfs, out_dir):
    models = [m for m in ("mamba1", "mamba2") if m in dfs]
    if not models:
        return
    regimes = sorted(set().union(*[set(dfs[m][dfs[m]["batch"] == 1]["regimes"]) for m in models]))
    if not regimes:
        return
    fig, ax = plt.subplots(figsize=(COL_W, 0.22 * len(regimes) + 0.9))
    for i, m in enumerate(models):
        d = dfs[m][dfs[m]["batch"] == 1]
        data = [d[d["regimes"] == r]["entropy_bits"].to_numpy() for r in regimes]
        pos = np.arange(len(regimes)) + (i - (len(models) - 1) / 2) * 0.36
        keep = [k for k, v in enumerate(data) if len(v)]
        bp = ax.boxplot([data[k] for k in keep], positions=pos[keep], vert=False, widths=0.3, patch_artist=True,
                        showfliers=False, manage_ticks=False)
        for b in bp["boxes"]:
            b.set(facecolor=OKABE[i], alpha=0.35, edgecolor=OKABE[i], linewidth=0.8, hatch="" if i == 0 else "///")
        for part in ("whiskers", "caps", "medians"):
            for ln in bp[part]:
                ln.set(color=OKABE[i], linewidth=0.8)
        ax.plot([], [], color=OKABE[i], lw=4, alpha=0.5, label=FAMILY_LABEL[m].split()[0])
    ax.set_yticks(range(len(regimes)))
    ax.set_yticklabels([r.replace("longbench_", "LB/").replace("_", " ") for r in regimes])
    ax.set_xlabel("activation entropy, K=256 (bits)")
    ax.grid(True, axis="x")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    _save(fig, out_dir, "tpds_entropy_by_regime.pdf")


def rule_switch_points(bits_max=8.0, step=1e-3):
    """Entropies (bits, K=256) at which Eq. (2) with the kernel-level range [32, 512] changes chunk."""
    hs = np.arange(0.0, bits_max + step, step)
    cs = [entropy_rule_chunk(h / bits_max) for h in hs]
    return [(float(hs[i]), cs[i - 1], cs[i]) for i in range(1, len(hs)) if cs[i] != cs[i - 1]]


def fig_entropy_vs_oracle(dfs, out_dir):
    models = [m for m in ("mamba1", "mamba2") if m in dfs]
    if not models:
        return
    fig, axes = plt.subplots(len(models), 1, figsize=(COL_W, 1.5 * len(models) + 0.4), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, m in zip(axes[:, 0], models):
        d = dfs[m]
        for j, b in enumerate(sorted(d["batch"].unique())):
            s = d[d["batch"] == b]
            y = np.log2(s["oracle_chunk"].to_numpy()) + rng.uniform(-0.15, 0.15, len(s)) + (j - 1) * 0.0
            ax.scatter(s["entropy_bits"], y, s=7, marker=MARKERS[j], facecolors="none", edgecolors=OKABE[j],
                       linewidths=0.6, label=f"batch {b}")
        chunks = sorted(d["oracle_chunk"].unique())
        # Eq. (2) decision boundaries inside the plotted entropy range, labelled with the rule's chunk
        lo, hi = float(d["entropy_bits"].min()), float(d["entropy_bits"].max())
        pad = 0.05 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        switches = [(h, c0, c1) for h, c0, c1 in rule_switch_points() if lo - pad < h < hi + pad]
        top = math.log2(max(chunks)) + 0.55
        for h, c0, c1 in switches:  # label each switch point on its own line: chunk left | chunk right
            ax.axvline(h, color="#555555", ls="--", lw=0.7)
            ax.text(h, top, f"rule {c0} | {c1}", ha="center", va="bottom", fontsize=6, color="#555555",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
        if not switches:
            c = entropy_rule_chunk(0.5 * (lo + hi) / 8.0)
            ax.text(0.5 * (lo + hi), top, f"rule: {c} for every call", ha="center", va="bottom", fontsize=6,
                    color="#555555")
        ax.set_ylim(math.log2(min(chunks)) - 0.5, top + 0.6)
        ax.set_yticks(np.log2(chunks))
        ax.set_yticklabels([str(int(c)) for c in chunks])
        ax.set_ylabel("oracle chunk")
        ax.set_title(FAMILY_LABEL[m], loc="left")
        ax.grid(True, axis="y")
    axes[-1, 0].set_xlabel("activation entropy, K=256 (bits)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(h_pad=0.4)
    _save(fig, out_dir, "tpds_entropy_vs_oracle.pdf")


# --------------------------------------------------------------------------- E3
POLICY_LABEL = {"global_static": "best static", "entropy_rule": "entropy rule (Eq. 2)", "L_only": "L-only tree",
                "shape": "shape tree", "entropy_only": "entropy-only tree", "all": "all features"}


def fig_regret_cdf(e3s, out_dir):
    if not e3s:
        return
    fig, axes = plt.subplots(len(e3s), 1, figsize=(COL_W, 1.6 * len(e3s) + 0.4), squeeze=False)
    for ax, (m, e3) in zip(axes[:, 0], e3s.items()):
        pa = e3["predictors_all_batches"]
        for i, (k, lab) in enumerate(POLICY_LABEL.items()):
            if k not in pa or "per_call_regret" not in pa[k]:
                continue
            r = np.sort(np.array(pa[k]["per_call_regret"])) * 100
            y = np.arange(1, len(r) + 1) / len(r)
            ax.step(np.maximum(r, 1e-2), y, where="post", color=OKABE[i], ls=STYLES[i], label=lab)
        ax.set_xscale("log")
        ax.set_xlabel("regret vs per-call oracle (%)")
        ax.set_ylabel("CDF of calls")
        ax.set_title(FAMILY_LABEL[m] + " (held-out)", loc="left")
        ax.grid(True)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(h_pad=0.4)
    _save(fig, out_dir, "tpds_regret_cdf.pdf")


# --------------------------------------------------------------------------- E5
def fig_breakeven(e5, out_dir):
    views = []
    for fam in ("mamba1", "mamba2"):
        v = (e5.get("view_e4_keys") or {}).get(fam)
        if v and "cost_curves_s" in v:
            views.append((fam, v["cost_curves_s"]))
    if not views:
        return
    fig, axes = plt.subplots(len(views), 1, figsize=(COL_W, 1.6 * len(views) + 0.4), squeeze=False)
    labels = {"default_static": "default chunk 256", "profiled_static": "profiled static", "autotuned": "autotuned",
              "entropy_online_full": "entropy online (full)", "entropy_online_sampled": "entropy online (1/8)"}
    for ax, (fam, cv) in zip(axes[:, 0], views):
        n = np.array(cv["N_calls"], dtype=float)
        base = np.array(cv.get("profiled_static") or cv[next(k for k in cv if k != "N_calls")])
        for i, (k, lab) in enumerate(labels.items()):
            if k not in cv:
                continue
            ax.plot(n, np.array(cv[k]) / base, color=OKABE[i], ls=STYLES[i], label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(1.0, color="#999999", lw=0.6)
        ax.set_xlabel("number of scan calls N")
        ax.set_ylabel("cost / profiled static")
        ax.set_title(FAMILY_LABEL[fam], loc="left")
        ax.grid(True, which="major")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(h_pad=0.4)
    _save(fig, out_dir, "tpds_breakeven.pdf")


# --------------------------------------------------------------------------- T1 / T3 (both GPUs)
HOST_LABEL = {"pro6000": "RTX PRO 6000", "vgpu3090": "RTX 3090"}
FULL_W = 7.16


def fig_cost_model(curves, out_dir, lengths=(512, 2048, 8192, 2382)):
    """Rows: GPUs; columns: batch sizes. Markers: measured medians; lines: cost-model prediction."""
    import pandas as pd

    hosts = [h for h in curves if not curves[h].empty]
    if not hosts:
        return
    batches = sorted(set.intersection(*[set(curves[h]["batch"]) for h in hosts]))
    fig, axes = plt.subplots(len(hosts), len(batches), figsize=(FULL_W, 1.55 * len(hosts) + 0.35), squeeze=False)
    for i, h in enumerate(hosts):
        df = curves[h]
        for j, b in enumerate(batches):
            ax = axes[i, j]
            for k, L in enumerate(lengths):
                g = df[(df["batch"] == b) & (df["L"] == L)].sort_values("chunk")
                if g.empty:
                    continue
                lab = f"L={L}" + ("" if L % 16 == 0 else r" ($L$ mod 16 $\neq$ 0)")
                ax.plot(g["chunk"], g["predicted_ms"], color=OKABE[k], ls=STYLES[k], lw=1.0)
                ax.plot(g["chunk"], g["measured_ms"], color=OKABE[k], marker=MARKERS[k], ls="none", label=lab)
            _pow2_axis(ax, sorted(df["chunk"].unique()))
            _plain_log_y(ax)
            ax.grid(True, which="major", axis="y")
            ax.set_title(f"{HOST_LABEL.get(h, h)}, batch {b}", loc="left")
            if j == 0:
                ax.set_ylabel("latency (ms)")
            if i == len(hosts) - 1:
                ax.set_xlabel("chunk size")
    handles, labels = {}, {}
    for ax in axes.ravel():
        for hd, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, hd)
    extra = [plt.Line2D([], [], color="#555555", marker="o", ls="none"), plt.Line2D([], [], color="#555555", lw=1.0)]
    fig.legend(list(handles.values()) + extra, list(handles) + ["measured (median over calls)", "cost model"],
               loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.8,
               columnspacing=1.0)
    fig.tight_layout(h_pad=0.5, w_pad=0.6)
    _save(fig, out_dir, "tpds_cost_model.pdf")


def fig_key_churn(e5bs, out_dir):
    """N*_k = P_k / (o - h_k) versus the per-call decision overhead o, per GPU and tuning cost.

    Lines are the median over shape keys (the statistic quoted in the paper); the band is the
    interquartile range over keys. Keys with h_k >= o have unbounded N*_k and count as +inf.
    """
    if not e5bs:
        return
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    plabel = {"profile": "profile 6 chunks", "search": "full launch search"}
    for i, (h, d) in enumerate(e5bs.items()):
        o = np.array(d["curve"]["overhead_ms"])
        hk = np.array([k["headroom_ms"] for k in d["keys"]], dtype=float)
        for j, pk in enumerate(("profile", "search")):
            Pk = np.array([k["P_s"][pk] for k in d["keys"]], dtype=float)
            gap = o[:, None] - hk[None, :]
            nk = np.where(gap > 0, 1000.0 * Pk[None, :] / np.where(gap > 0, gap, 1.0), np.inf)
            q1, med, q3 = (np.percentile(nk, q, axis=1) for q in (25, 50, 75))
            med = np.where(np.isfinite(med), med, np.nan)
            ax.plot(o, med, color=OKABE[i], ls=STYLES[j], label=f"{HOST_LABEL.get(h, h)}, {plabel[pk]}")
            ax.fill_between(o, np.where(np.isfinite(q1), q1, np.nan), np.where(np.isfinite(q3), q3, np.nan),
                            color=OKABE[i], alpha=0.12, lw=0)
    d0 = next(iter(e5bs.values()))
    marks = {"host sync": d0["host_sync_ms"]}
    for name, key in (("1/8 histogram", "sampled_histogram"), ("full histogram", "full_histogram")):
        vals = [k["overhead_ms"][key] for k in d0["keys"] if k["overhead_ms"].get(key)]
        if vals:
            marks[name] = float(np.median(vals))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ylo, yhi = ax.get_ylim()
    for name, x in marks.items():
        ax.axvline(x, color="#999999", lw=0.6, ls=":")
        # label in the empty region: below the curves at small o, above them at large o
        low = name == "host sync"
        ax.text(x * 1.08, ylo * 2 if low else yhi * 0.6, name, rotation=90, va="bottom" if low else "top",
                ha="left", fontsize=7, color="#444444")
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("per-call decision overhead $o$ (ms)")
    ax.set_ylabel("break-even calls per key $N^\\star$")
    ax.grid(True, which="major")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    _save(fig, out_dir, "tpds_key_churn.pdf")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--host", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--png", action="store_true", help="also write PNG previews")
    p.add_argument("--cross-hosts", default="pro6000,vgpu3090", help="hosts of the two-GPU figures (T1, T3)")
    args = p.parse_args(argv)
    global PNG_PREVIEW
    PNG_PREVIEW = args.png
    jobs = []
    e1 = _load_json(args.in_dir / f"e1_{args.host}.json")
    if e1 and e1.get("rows"):
        jobs += [lambda: fig_latency_vs_chunk(e1, args.out_dir), lambda: fig_value_invariance(e1, args.out_dir)]
    dfs = {}
    for m in ("mamba1", "mamba2"):
        df = load_e2([args.in_dir / f"e2_{m}_{args.host}.csv"])
        if not df.empty:
            dfs[m] = df
    if dfs:
        jobs += [lambda: fig_entropy_by_regime(dfs, args.out_dir), lambda: fig_entropy_vs_oracle(dfs, args.out_dir)]
    e3s = {m: j for m in ("mamba1", "mamba2") if (j := _load_json(args.in_dir / f"e3_{m}_{args.host}.json"))}
    if e3s:
        jobs.append(lambda: fig_regret_cdf(e3s, args.out_dir))
    e5 = _load_json(args.in_dir / f"e5_{args.host}.json")
    if e5:
        jobs.append(lambda: fig_breakeven(e5, args.out_dir))
    import pandas as pd

    cross = [h for h in args.cross_hosts.split(",") if h]
    curves = {h: pd.read_csv(f) for h in cross if (f := args.in_dir / f"cost_model_curves_{h}.csv").exists()}
    if curves:
        jobs.append(lambda: fig_cost_model(curves, args.out_dir))
    e5bs = {h: j for h in cross if (j := _load_json(args.in_dir / f"e5b_{h}.json"))}
    if e5bs:
        jobs.append(lambda: fig_key_churn(e5bs, args.out_dir))
    if not jobs:
        print("no results found; nothing to plot")
    for job in jobs:
        try:
            job()
        except Exception as exc:  # partial results must not stop the other figures
            print(f"WARNING: figure failed: {exc!r}")


if __name__ == "__main__":
    main()
