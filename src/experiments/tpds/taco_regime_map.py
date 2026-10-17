"""Regime map (taco-r1, review item M9d): default-chunk regret against tokens per call, per execution regime.

For each production kernel and the Triton Mamba-1 scan, the per-call relative regret of the library default
(SSD 256, Gated DeltaNet 64; Triton Mamba-1: its best single chunk, as in the CUDA-graph table) is binned by
the tokens of the call (length x batch, powers of two) and averaged per bin, separately for every timing
regime that exists for that kernel:

  host-bound     launch-inclusive eager timing (RTX 3090)
  device time    GPU-side pre-spin, the host runs ahead (RTX 3090; Triton scan only: for the multi-kernel
                 pipelines device time mixes launch-bound and work-bound calls below the floor)
  CUDA graph     every (call, chunk) captured and replayed (RTX 3090, and RTX PRO 6000 from E26)

Also writes the practitioner summary (default regret below and above the launch floor of each regime).
Reads the stored E2-style tables through ``e22_graph_analysis`` (round-median view where three rounds exist)
and ``taco_e26_analysis.register`` for the RTX PRO 6000 replays. CPU only.

Output: ``taco_regime_map.json`` in the results directory and ``fig_regime_map.pdf`` in ``--fig-dir``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.experiments.tpds import e22_graph_analysis as E22
from src.experiments.tpds import taco_e26_analysis as E26
from src.experiments.tpds.common import RESULTS_DIR, utc_now, write_json

# panel -> list of (series label, table name); series labels keep one colour across panels
PANELS = {
    "Mamba-2 SSD 370M": [("launch-bound (RTX 3090)", "ssd370m_eager"), ("CUDA graph (RTX 3090)", "ssd370m_graph"),
                         ("CUDA graph (RTX PRO 6000)", "ssd370m_graph_pro6000")],
    "Mamba-2 SSD 2.7B": [("CUDA graph (RTX 3090)", "ssd2p7b_graph"), ("CUDA graph (RTX PRO 6000)", "ssd2p7b_graph_pro6000")],
    "Gated DeltaNet 1.4B": [("CUDA graph (RTX 3090)", "gdn1p4b_graph"), ("CUDA graph (RTX PRO 6000)", "gdn1p4b_graph_pro6000")],
    "Triton Mamba-1 370M": [("launch-bound (RTX 3090)", "m1_370m_eager"), ("device time (RTX 3090)", "m1_370m_device_r4"),
                            ("CUDA graph (RTX 3090)", "m1_370m_graph"), ("CUDA graph (RTX PRO 6000)", "m1_370m_graph_pro6000")],
}
# recorded in the JSON for the practitioner summary but not plotted (device time of the multi-kernel pipelines
# mixes launch-bound and work-bound calls below the floor; only the calls above it are interpretable)
EXTRA = {"Mamba-2 SSD 370M": [("device time (RTX 3090)", "ssd370m_device")],
         "Mamba-2 SSD 2.7B": [("device time (RTX 3090)", "ssd2p7b_device")],
         "Gated DeltaNet 1.4B": [("device time (RTX 3090)", "gdn1p4b_device")]}
TITLES = {"Mamba-2 SSD 370M": "SSD 370M", "Mamba-2 SSD 2.7B": "SSD 2.7B", "Gated DeltaNet 1.4B": "Gated DeltaNet 1.4B",
          "Triton Mamba-1 370M": "Triton scan (ours), 370M"}
SERIES = ["launch-bound (RTX 3090)", "device time (RTX 3090)", "CUDA graph (RTX 3090)", "CUDA graph (RTX PRO 6000)"]
MIN_BIN = 10


def table_record(name: str, in_dir: Path) -> dict:
    t = E22.Table(name, in_dir)
    t.set_floor()
    d = t.default_idx()
    reg = t.lat[:, d] / t.orc - 1.0
    tokens = t.L.astype(float) * t.batch.astype(float)
    b = np.floor(np.log2(np.maximum(tokens, 1))).astype(int)
    bins = []
    total_orc = float(t.orc.sum())
    for k in sorted(set(b.tolist())):
        m = b == k
        if m.sum() >= MIN_BIN:
            bins.append({"tokens_lo": int(2 ** k), "tokens_hi": int(2 ** (k + 1)), "n": int(m.sum()),
                         "mean_regret_pct": float(100 * reg[m].mean()),
                         "median_regret_pct": float(100 * np.median(reg[m])),
                         # this bin's excess over the oracle as a share of the whole workload's oracle scan time
                         "excess_share_of_total_pct": float(100 * (t.lat[m, d] - t.orc[m]).sum() / total_orc)})

    def part(mask):
        if not mask.any():
            return None
        exc = (t.lat[mask, d] - t.orc[mask]).sum()
        return {"n": int(mask.sum()), "mean_regret_pct": float(100 * reg[mask].mean()),
                "lw_share_of_oracle_pct": float(100 * exc / t.orc[mask].sum())}

    return {"table": name, "timing": t.timing, "gpu": t.gpu, "default_chunk": int(t.chunks[d]),
            "default_is_best_static": t.default is None, "floor_us": 1e3 * t.floor_ms, "n_calls": int(len(reg)),
            "bins": bins, "all": part(np.ones(len(reg), bool)), "below_floor": part(~t.above),
            "above_floor": part(t.above)}


def figure(res: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.experiments.tpds.make_figures import MARKERS, OKABE, STYLES, _save

    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "legend.fontsize": 8})
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.6), sharex=True, sharey="row")
    ticks = [2 ** k for k in range(4, 18, 2)]
    for j, (panel, series) in enumerate(PANELS.items()):
        for label, name in series:
            rec = res["panels"][panel].get(label)
            if not rec:
                continue
            i = SERIES.index(label)
            x = [np.sqrt(bn["tokens_lo"] * bn["tokens_hi"]) for bn in rec["bins"]]
            y = [bn["mean_regret_pct"] for bn in rec["bins"]]
            z = [bn.get("excess_share_of_total_pct", np.nan) for bn in rec["bins"]]
            kw = dict(color=OKABE[i], marker=MARKERS[i], ls=STYLES[i], lw=1.4, ms=3.5, label=label)
            axes[0, j].plot(x, y, **kw)
            axes[1, j].plot(x, z, **kw)
        axes[0, j].set_title(TITLES.get(panel, panel), loc="left")
        # calls per bin (the series of a panel time the same captured calls; take the first plotted one)
        first = next((res["panels"][panel].get(lb) for lb, _ in series if res["panels"][panel].get(lb)), None)
        if first:
            trans = matplotlib.transforms.blended_transform_factory(axes[0, j].transData, axes[0, j].transAxes)
            for q, bn in enumerate(first["bins"]):        # staggered on two rows so that neighbours do not overlap
                axes[0, j].text(np.sqrt(bn["tokens_lo"] * bn["tokens_hi"]), 0.98 - 0.075 * (q % 2), str(bn["n"]),
                                transform=trans, ha="center", va="top", fontsize=5, color="0.35")
        for ax in axes[:, j]:
            ax.set_xscale("log", base=2)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"$2^{{{k}}}$" for k in range(4, 18, 2)])
            ax.minorticks_off()
            ax.grid(True, which="major", axis="y", lw=0.4, alpha=0.5)
        axes[1, j].set_xlabel("tokens per call")
    lo, hi = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(lo, hi + 0.2 * (hi - lo))      # room above the curves for the call counts
    axes[0, 0].set_ylabel("default regret\nper call (%)")
    axes[1, 0].set_ylabel("share of scan\ntime lost (%)")
    h, lab = [], []
    for ax in axes[0]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    order = [lab.index(s) for s in SERIES if s in lab]
    fig.legend([h[i] for i in order], [lab[i] for i in order], loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(w_pad=0.6, h_pad=0.4)
    _save(fig, out, "fig_regime_map.pdf")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--fig-dir", type=Path, default=None, help="where to write fig_regime_map.pdf (default: none)")
    args = p.parse_args(argv)
    E22.use_three_rounds(args.in_dir)
    E26.register(args.in_dir, "pro6000", "pro6000")
    res = {"experiment": "taco-r1 regime map (M9d)", "created_utc": utc_now(), "min_bin_calls": MIN_BIN,
           "panels": {}}
    for panel, series in PANELS.items():
        res["panels"][panel] = {}
        for label, name in series:
            if name not in E22.TABLES:
                continue
            rec = table_record(name, args.in_dir)
            res["panels"][panel][label] = rec
            a, lo, hi = rec["all"], rec["below_floor"], rec["above_floor"]
            print(f"{panel:22s} {label:26s} default {rec['default_chunk']:4d}: all {a['mean_regret_pct']:5.1f}% "
                  f"(lw {a['lw_share_of_oracle_pct']:.1f}) | below floor "
                  f"{'-' if lo is None else f'{lo['mean_regret_pct']:.1f}% (n {lo['n']})'} | above floor "
                  f"{'-' if hi is None else f'{hi['mean_regret_pct']:.1f}% (n {hi['n']})'}")
    res["extra_not_plotted"] = {panel: {label: table_record(name, args.in_dir) for label, name in series}
                                for panel, series in EXTRA.items()}
    write_json(args.in_dir / "taco_regime_map.json", res)
    if args.fig_dir:
        figure(res, args.fig_dir)


if __name__ == "__main__":
    main()
