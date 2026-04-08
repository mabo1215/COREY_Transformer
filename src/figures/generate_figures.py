from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ROOT / "src" / "outputs"
PAPER_FIGURES_DIR = ROOT / "paper" / "figs"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def generate_entropy_gain_figure() -> Path:
    rows = _read_csv(OUTPUTS_DIR / "hadamard_validation.csv")
    sequence_lengths = [int(row["sequence_length"]) for row in rows]
    entropy_before = [float(row["entropy_before"]) for row in rows]
    entropy_after = [float(row["entropy_after"]) for row in rows]

    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    axis.plot(sequence_lengths, entropy_before, "o--", color="#b44b3d", label="Before Hadamard")
    axis.plot(sequence_lengths, entropy_after, "s-", color="#2f6c8f", label="After Hadamard")
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Normalized entropy")
    axis.set_title("Entropy increase after Hadamard reparameterization")
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.legend(frameon=False)
    figure.tight_layout()

    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PAPER_FIGURES_DIR / "entropy_gain.pdf"
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def generate_performance_summary_figure() -> Path:
    rows = _read_csv(OUTPUTS_DIR / "bucket_summary.csv")
    selected = [row for row in rows if row["precision"] == "fp16"]
    buckets = ["short", "medium", "long", "ultra_long"]
    methods = ["no_fusion", "static_fusion", "entropy_guided"]
    labels = ["No Fusion", "Static Fusion", "Entropy-Guided"]
    colors = ["#c44e52", "#dd8452", "#4c956c"]

    latency_lookup = {(row["bucket"], row["method"]): float(row["latency_ms"]) for row in selected}
    throughput_lookup = {(row["bucket"], row["method"]): float(row["throughput_tokens_per_s"]) for row in selected}

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x_positions = range(len(buckets))
    width = 0.23

    for offset, method, label, color in zip([-width, 0.0, width], methods, labels, colors):
        axes[0].bar(
            [position + offset for position in x_positions],
            [latency_lookup[(bucket, method)] for bucket in buckets],
            width=width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )
        axes[1].bar(
            [position + offset for position in x_positions],
            [throughput_lookup[(bucket, method)] for bucket in buckets],
            width=width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )

    for axis, ylabel, title in (
        (axes[0], "Latency (ms)", "Latency by sequence bucket"),
        (axes[1], "Throughput (tokens/s)", "Throughput by sequence bucket"),
    ):
        axis.set_xticks(list(x_positions), [bucket.replace("_", "-") for bucket in buckets])
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)

    axes[1].legend(frameon=False, loc="upper left")
    figure.tight_layout()

    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PAPER_FIGURES_DIR / "performance_summary.pdf"
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    generated = [generate_entropy_gain_figure(), generate_performance_summary_figure()]
    for path in generated:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()