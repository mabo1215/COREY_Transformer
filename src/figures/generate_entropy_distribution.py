"""Generate a real-checkpoint entropy distribution figure for the appendix.

Reads prompt-level `entropy_before` values from LongBench prediction logs and
produces a grouped strip/violin plot showing per-task dispersion together with
the coarse chunk-bucket boundaries used by COREY. The resulting figure is
saved as JPEG under `paper/figs/entropy_distribution.jpg` for LaTeX inclusion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COARSE_BOUNDARIES_NATS = (
    (32, 1.5),
    (64, 2.5),
    (128, 3.5),
    (256, 4.5),
    (512, 5.5),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate real-checkpoint entropy distribution figure.",
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        type=Path,
        required=True,
        help="Prediction JSONL files produced by the LongBench runner.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figs/entropy_distribution.jpg"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return [row for row in rows if row.get("entropy_before") is not None]


def _group_by_task(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    by_task: dict[str, list[float]] = {}
    for row in rows:
        task = str(row.get("task", "unknown"))
        by_task.setdefault(task, []).append(float(row["entropy_before"]))
    return {task: np.asarray(values, dtype=np.float64) for task, values in by_task.items()}


def _render(by_task: dict[str, np.ndarray], output: Path, dpi: int) -> None:
    tasks = sorted(by_task.keys())
    task_display = {
        "narrativeqa": "NarrativeQA",
        "qasper": "Qasper",
        "multifieldqa_en": "MultiFieldQA-EN",
        "gov_report": "GovReport",
    }
    labels = [task_display.get(task, task) for task in tasks]
    data = [by_task[task] for task in tasks]

    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=dpi)

    low_limit = 1.2
    high_limit = 5.8
    for _, boundary in COARSE_BOUNDARIES_NATS:
        ax.axhline(boundary, color="#c0c0c0", linewidth=0.7, linestyle="--", zorder=0)

    bucket_centers = {
        1.5: "chunk 32",
        2.5: "chunk 64",
        3.5: "chunk 128",
        4.5: "chunk 256",
        5.5: "chunk 512",
    }
    for boundary, label in bucket_centers.items():
        ax.text(
            len(tasks) + 0.55,
            boundary,
            label,
            fontsize=8,
            color="#606060",
            va="center",
            ha="left",
        )

    violin_parts = ax.violinplot(
        data,
        positions=np.arange(len(tasks)) + 1,
        widths=0.7,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    for body in violin_parts["bodies"]:
        body.set_facecolor("#6495ED")
        body.set_edgecolor("#1F4E99")
        body.set_alpha(0.35)
        body.set_linewidth(0.8)

    rng = np.random.default_rng(0)
    for index, values in enumerate(data, start=1):
        jitter = rng.normal(loc=0.0, scale=0.045, size=values.shape)
        ax.scatter(
            np.full_like(values, index, dtype=np.float64) + jitter,
            values,
            color="#1F4E99",
            s=14,
            alpha=0.75,
            linewidths=0.0,
            zorder=3,
        )
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        ax.errorbar(
            [index],
            [mean],
            yerr=[[std], [std]],
            fmt="D",
            color="#B22222",
            markersize=4.5,
            linewidth=1.1,
            capsize=3,
            zorder=4,
        )

    all_values = np.concatenate(data)
    theoretical_max = math.log(256)
    ax.axhline(
        theoretical_max,
        color="#B22222",
        linewidth=0.9,
        linestyle=":",
        zorder=0,
    )
    ax.text(
        0.05,
        theoretical_max - 0.08,
        r"theoretical max $\log K$ = 5.55",
        fontsize=8,
        color="#B22222",
        va="top",
        ha="left",
    )

    ax.set_xticks(np.arange(len(tasks)) + 1)
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_ylabel("Input entropy (nats)")
    ax.set_xlabel("LongBench task")
    ax.set_ylim(low_limit, high_limit)
    ax.set_xlim(0.4, len(tasks) + 1.4)
    ax.set_title(
        f"Prompt-level entropy across {all_values.size} LongBench prompts (Mamba-370M)",
        fontsize=10,
    )
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    rows = _read_rows(args.predictions)
    if not rows:
        raise ValueError("No rows with entropy_before were found in the provided prediction files.")
    by_task = _group_by_task(rows)
    _render(by_task, args.output, args.dpi)
    print(f"Wrote figure: {args.output}")


if __name__ == "__main__":
    main()
