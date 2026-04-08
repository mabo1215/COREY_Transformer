"""
Generate placeholder figures for the ADAMA paper.

Run with:
    python paper/figures/generate_figures.py

Requires: matplotlib, numpy, torch
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

FIGURE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Figure 1: Overview diagram
# ---------------------------------------------------------------------------

def draw_adama_overview():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # -- Panel 1: Calibration pass (entropy bar chart) --
    ax = axes[0]
    layers = [f"L{i}" for i in range(1, 9)]
    entropies = [6.1, 6.4, 2.8, 5.9, 6.2, 2.6, 6.0, 5.8]
    colors = ["steelblue" if h >= 5.0 else "tomato" for h in entropies]
    ax.bar(layers, entropies, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(5.0, color="black", linestyle="--", linewidth=1.2, label="τ_H = 5.0 bits")
    ax.set_ylim(0, 8)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Entropy (bits)", fontsize=11)
    ax.set_title("(a) Calibration: Per-Layer Entropy", fontsize=11)
    ax.legend(fontsize=9)
    blue_patch = mpatches.Patch(color="steelblue", label="High-entropy (fuse)")
    red_patch = mpatches.Patch(color="tomato", label="Low-entropy (WHT boundary)")
    ax.legend(handles=[blue_patch, red_patch], fontsize=8, loc="lower right")

    # -- Panel 2: Fusion grouping --
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("(b) EGFS: Fusion Groups", fontsize=11)
    groups = [(0.5, 3, "G1\nL1, L2", "steelblue"),
              (4.0, 1, "G2\nL3 (WHT)", "tomato"),
              (5.5, 3, "G3\nL4, L5", "steelblue"),
              (7.5, 2, "G4\nL6 (WHT)", "tomato"),
              (8.5, 3.5, "G5\nL7, L8", "steelblue")]
    for (x, w, lbl, c) in groups:
        rect = plt.Rectangle((x, 1.5), w, 1.5, color=c, alpha=0.8, ec="black")
        ax.add_patch(rect)
        ax.text(x + w / 2, 2.25, lbl, ha="center", va="center",
                fontsize=8, fontweight="bold")
    ax.text(5, 0.5, "→ 5 kernels (vs. 8 without fusion)", ha="center", fontsize=9)

    # -- Panel 3: Fused Hadamard Layer detail --
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("(c) Fused Hadamard Layer (FHL)", fontsize=11)
    boxes = [(1.0, 3.5, "Linear\n(W_fused)"), (4.5, 3.5, "WHT\nabsorbed"),
             (7.5, 3.5, "Linear\n(W_out)")]
    for (x, y, lbl) in boxes:
        rect = plt.Rectangle((x, y - 0.5), 2.0, 1.0, color="mediumseagreen",
                              alpha=0.8, ec="black")
        ax.add_patch(rect)
        ax.text(x + 1.0, y, lbl, ha="center", va="center", fontsize=8)
    for x1, x2 in [(3.0, 4.5), (6.5, 7.5)]:
        ax.annotate("", xy=(x2, 3.5), xytext=(x1, 3.5),
                    arrowprops=dict(arrowstyle="->", color="black"))
    ax.text(5, 1.8, "W_fused = H_n · W / √n\n(pre-computed once)", ha="center",
            fontsize=9, style="italic")
    brace_note = "Zero runtime overhead"
    ax.text(5, 1.0, brace_note, ha="center", fontsize=9, color="darkgreen",
            fontweight="bold")

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, "adama_overview.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Entropy gain per layer
# ---------------------------------------------------------------------------

def draw_entropy_gain():
    fig, ax = plt.subplots(figsize=(7, 4))

    np.random.seed(42)
    n_layers = 24
    layer_ids = np.arange(1, n_layers + 1)
    # Simulate before-WHT entropy (y tensor is lowest)
    entropy_before = np.random.uniform(3.5, 6.5, n_layers)
    entropy_before[3::6] = np.random.uniform(1.8, 3.0, len(entropy_before[3::6]))  # outlier layers
    entropy_after = entropy_before + np.random.uniform(0.3, 2.1, n_layers)
    entropy_after = np.clip(entropy_after, 0, 8)

    ax.plot(layer_ids, entropy_before, "o--", color="tomato", label="Before WHT",
            linewidth=1.5, markersize=5)
    ax.plot(layer_ids, entropy_after, "s-", color="steelblue", label="After WHT (FHL)",
            linewidth=1.5, markersize=5)
    ax.axhline(5.0, color="black", linestyle=":", linewidth=1.2, label="τ_H threshold")
    ax.fill_between(layer_ids, entropy_before, entropy_after, alpha=0.15, color="steelblue",
                    label="ΔH gain")

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Activation Entropy (bits)", fontsize=12)
    ax.set_title("Per-Layer Entropy Before and After FHL (Mamba-2.8B)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0.5, n_layers + 0.5)
    ax.set_ylim(0, 8.5)
    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, "entropy_gain.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Latency comparison bar chart
# ---------------------------------------------------------------------------

def draw_latency_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    methods = ["No-Fuse", "Fuse-All", "EGFS", "ADAMA"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    for ax, (model, latencies, tps) in zip(axes, [
        ("Mamba-370M", [18.4, 10.2, 9.5, 7.8], [543, 980, 1052, 1281]),
        ("Mamba-2.8B", [142.1, 79.6, 73.8, 29.6], [70, 125, 135, 338]),
    ]):
        x = np.arange(len(methods))
        bars = ax.bar(x, latencies, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=10)
        ax.set_ylabel("Latency (ms)", fontsize=11)
        ax.set_title(f"Inference Latency – {model}", fontsize=11)
        for bar, lat in zip(bars, latencies):
            ax.text(bar.get_x() + bar.get_width() / 2, lat + 1,
                    f"{lat}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, "latency_comparison.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    draw_adama_overview()
    draw_entropy_gain()
    draw_latency_comparison()
    print("All figures generated.")
