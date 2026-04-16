"""
M6 Fix: Regenerate entropy_gain figure as a cleaner grouped bar chart.

Replaces the multi-line curve (hard to read "post always above pre") with:
  - Primary: grouped bar chart (pre/post bars side-by-side per sequence length)
    → post bars are always taller → gap is immediately readable

Reads data from src/outputs/hadamard_validation.csv (same source as the
original generate_figures.py).  Saves to paper/figs/entropy_gain.jpg
(and .pdf), overwriting the old figure.  The original is renamed to
entropy_gain_old.jpg before the new file is written.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ROOT / "src" / "outputs"
PAPER_FIGURES_DIR = ROOT / "paper" / "figs"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def generate_entropy_gain_bar_figure() -> Path:
    rows = _read_csv(OUTPUTS_DIR / "hadamard_validation.csv")
    seq_lens = sorted(set(int(r["sequence_length"]) for r in rows))

    pre_means = [mean(float(r["entropy_before"]) for r in rows if int(r["sequence_length"]) == sl)
                 for sl in seq_lens]
    post_means = [mean(float(r["entropy_after"]) for r in rows if int(r["sequence_length"]) == sl)
                  for sl in seq_lens]
    pre_stds = []
    post_stds = []
    for sl in seq_lens:
        vals_pre = [float(r["entropy_before"]) for r in rows if int(r["sequence_length"]) == sl]
        vals_post = [float(r["entropy_after"]) for r in rows if int(r["sequence_length"]) == sl]
        pre_stds.append(float(np.std(vals_pre)))
        post_stds.append(float(np.std(vals_post)))

    x = np.arange(len(seq_lens))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars_pre = ax.bar(
        x - width / 2,
        pre_means,
        width,
        yerr=pre_stds,
        capsize=3,
        color="#b44b3d",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
        label="Before Hadamard",
    )
    bars_post = ax.bar(
        x + width / 2,
        post_means,
        width,
        yerr=post_stds,
        capsize=3,
        color="#2f6c8f",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
        label="After Hadamard",
    )

    # Annotate each pair with the delta
    for i, (pre, post) in enumerate(zip(pre_means, post_means)):
        delta = post - pre
        ax.annotate(
            f"+{delta:.3f}",
            xy=(x[i], max(pre, post) + max(pre_stds[i], post_stds[i]) + 0.003),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
        )

    tick_labels = [
        f"$2^{{{sl.bit_length()-1}}}$\n({sl})" for sl in seq_lens
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Sequence length", fontsize=9)
    ax.set_ylabel("Normalized entropy", fontsize=9)
    ax.set_title(
        "Entropy before and after Hadamard reparameterization\n"
        "(post always exceeds pre; $\\Delta$ annotated above each pair)",
        fontsize=9,
    )
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()

    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Rename old file with _old suffix before overwriting
    for ext in (".jpg", ".pdf"):
        old = PAPER_FIGURES_DIR / f"entropy_gain{ext}"
        backup = PAPER_FIGURES_DIR / f"entropy_gain_old{ext}"
        if old.exists():
            shutil.move(str(old), str(backup))
            print(f"[M6] Renamed {old.name} -> {backup.name}")

    jpg_path = PAPER_FIGURES_DIR / "entropy_gain.jpg"
    pdf_path = PAPER_FIGURES_DIR / "entropy_gain.pdf"
    fig.savefig(str(jpg_path), bbox_inches="tight", dpi=300)
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)
    print(f"[M6] Saved new entropy_gain grouped bar chart -> {jpg_path}")
    return jpg_path


if __name__ == "__main__":
    path = generate_entropy_gain_bar_figure()
    print(f"Done: {path}")
