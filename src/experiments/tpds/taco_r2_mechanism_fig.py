"""Mechanism figure (taco-r2 M8d): where the chunk acts, from the per-kernel breakdown (E35; CPU only).

Panels (stock RTX 3090, device time of each kernel from the profiler, batch 1):
  (a) Triton scan: latency versus chunk at L = 512, 2048 and 8192, with the launch grid (programs) of the default
      launch table and the registers per thread of each chunk;
  (b) Mamba-2 SSD (370M shapes), L = 4096: stacked per-kernel time versus chunk;
  (c) Gated DeltaNet (1.4B shapes), L = 4096: stacked per-kernel time versus chunk.

Output: ``fig_mechanism.pdf`` in ``--fig-dir`` and the plotted numbers in ``taco_r2_mechanism.json``.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.experiments.tpds.common import RESULTS_DIR

SSD_GROUPS = [("_chunk_scan_fwd_kernel", "intra-chunk output"), ("_chunk_state_fwd_kernel", "chunk states"),
              ("_state_passing_fwd_kernel", "state passing"), ("_bmm_chunk_fwd_kernel", "C$^\\top$B products"),
              ("_chunk_cumsum_fwd_kernel", "decay cumsum")]
GDN_GROUPS = [("chunk_gated_delta_rule_fwd_k", "inter-chunk states"), ("chunk_fwd_kernel_o", "output"),
              ("recompute_w_u_fwd_kernel", "WY recompute"),
              (("solve_tril", "merge_16x16", "chunk_scaled_dot_kkt"), "WY inverse"), (None, "other")]


def group_kernels(kernels: list[dict], groups) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for k in kernels:
        name = k["kernel"]
        label = None
        for pat, lab in groups:
            if pat is None:
                continue
            pats = pat if isinstance(pat, tuple) else (pat,)
            if any(name.startswith(p) for p in pats):
                label = lab
                break
        out[label or "other"] += k["dur_us_per_call"]
    return dict(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="4c_e35")
    ap.add_argument("--fig-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    d = json.loads((RESULTS_DIR / f"e35_breakdown_{a.tag}.json").read_text())
    cells = d["cells"]
    res: dict = {"source": f"e35_breakdown_{a.tag}.json", "triton": {}, "ssd": {}, "gdn": {}}
    for c in cells:
        if c["family"] == "m1" and c["batch"] == 1 and c["launch"] == "default" and c["placement"] == "fresh":
            k = c["kernels"][0]
            res["triton"].setdefault(str(c["L"]), {})[str(c["chunk"])] = {
                "us": c["total_us_per_call"], "grid": k["grid"][0] * k["grid"][1] * k["grid"][2],
                "regs": k["registers_per_thread"], "occ_pct": k["est_achieved_occupancy_pct"]}
        if c["family"] in ("ssd", "gdn") and c["batch"] == 1 and c["L"] == 4096:
            groups = SSD_GROUPS if c["family"] == "ssd" else GDN_GROUPS
            res[c["family"]][str(c["chunk"])] = group_kernels(c["kernels"], groups)
    (RESULTS_DIR / "taco_r2_mechanism.json").write_text(json.dumps(res, indent=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.experiments.tpds.make_figures import MARKERS, OKABE, STYLES, _save

    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "legend.fontsize": 7})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    ax = axes[0]
    for i, L in enumerate(("512", "2048", "8192")):
        pts = res["triton"][L]
        cs = sorted(pts, key=int)
        y = [pts[c]["us"] / pts[min(cs, key=lambda x: pts[x]["us"])]["us"] for c in cs]
        ax.plot([int(c) for c in cs], y, color=OKABE[i], marker=MARKERS[i], ls=STYLES[i], lw=1.3, ms=3.5, label=f"L={L}")
    pts = res["triton"]["8192"]
    for c in sorted(pts, key=int):
        ax.annotate(f"{pts[c]['regs']}", (int(c), 0.86), ha="center", va="center", fontsize=6.5, color="0.35",
                    annotation_clip=False)
    ax.axvspan(181, 1500, color="0.92", zorder=0)
    ax.text(520, 2.22, "512 programs", fontsize=6.5, ha="center", color="0.3")
    ax.text(64, 2.22, "256 programs", fontsize=6.5, ha="center", color="0.3")
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64, 128, 256, 512, 1024])
    ax.set_xticklabels(["32", "64", "128", "256", "512", "1024"])
    ax.minorticks_off()
    ax.set_ylim(0.8, 2.3)
    ax.set_xlabel("chunk (registers per thread below)")
    ax.set_ylabel("latency / best")
    ax.set_title("(a) Triton scan, batch 1", loc="left")
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.6), handlelength=1.6)
    for ax, fam, groups, title in ((axes[1], "ssd", SSD_GROUPS, "(b) SSD 370M, $L$=4096"),
                                   (axes[2], "gdn", GDN_GROUPS, "(c) Gated DeltaNet 1.4B, $L$=4096")):
        data = res[fam]
        cs = sorted(data, key=int)
        bottom = [0.0] * len(cs)
        for j, (_, lab) in enumerate(groups):
            vals = [data[c].get(lab, 0.0) for c in cs]
            if not any(vals):
                continue
            ax.bar(range(len(cs)), vals, bottom=bottom, color=OKABE[j % len(OKABE)], width=0.7, label=lab)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xticks(range(len(cs)))
        ax.set_xticklabels(cs)
        ax.set_xlabel("chunk")
        ax.set_ylabel("kernel time ($\\mu$s)")
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 1.55 * max(bottom))
        ax.legend(frameon=False, fontsize=6.2, loc="upper right", handlelength=0.9, labelspacing=0.15,
                  borderaxespad=0.2)
        ax.grid(True, axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout(w_pad=0.8)
    _save(fig, a.fig_dir, "fig_mechanism.pdf")


if __name__ == "__main__":
    main()
