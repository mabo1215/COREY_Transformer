from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize prompt-level entropy distributions from prediction logs."
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        type=Path,
        required=True,
        help="Prediction JSONL files produced by run_longbench_inference or related runners.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/outputs/entropy_variance_summary"),
    )
    parser.add_argument(
        "--coarse-buckets",
        nargs="+",
        type=int,
        default=[32, 64, 128, 256, 512],
    )
    return parser.parse_args()


def _read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must be non-empty")
    index = (len(sorted_values) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    fraction = index - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _nearest_bucket(value: int, buckets: list[int]) -> int:
    return min(buckets, key=lambda bucket: (abs(bucket - value), bucket))


def _summarize_scope(rows: list[dict[str, Any]], coarse_buckets: list[int], label: str) -> dict[str, Any]:
    values = sorted(float(row["entropy_before"]) for row in rows if row.get("entropy_before") is not None)
    tile_values = [int(row["suggested_tile_size"]) for row in rows if row.get("suggested_tile_size") is not None]
    if not values:
        raise ValueError(f"No entropy_before values found for scope {label}")

    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    bucket_counts = Counter(_nearest_bucket(tile_value, coarse_buckets) for tile_value in tile_values)
    continuous_counts = Counter(tile_values)

    summary = {
        "scope": label,
        "samples": len(values),
        "mean_entropy": round(mean_value, 6),
        "std_entropy": round(math.sqrt(variance), 6),
        "p5_entropy": round(_percentile(values, 0.05), 6),
        "p95_entropy": round(_percentile(values, 0.95), 6),
        "min_entropy": round(values[0], 6),
        "max_entropy": round(values[-1], 6),
        "continuous_tile_values": json.dumps(dict(sorted(continuous_counts.items())), sort_keys=True),
        "coarse_bucket_values": json.dumps(dict(sorted(bucket_counts.items())), sort_keys=True),
    }
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    rows = [row for row in _read_rows(args.predictions) if row.get("entropy_before") is not None]
    if not rows:
        raise ValueError("No rows with entropy_before were found in the provided prediction files.")

    overall_summary = _summarize_scope(rows, args.coarse_buckets, "all")
    task_rows: list[dict[str, Any]] = []
    for task in sorted({row["task"] for row in rows}):
        task_rows.append(_summarize_scope([row for row in rows if row["task"] == task], args.coarse_buckets, task))

    bucket_rows = []
    bucket_counts = Counter(
        _nearest_bucket(int(row["suggested_tile_size"]), args.coarse_buckets)
        for row in rows
        if row.get("suggested_tile_size") is not None
    )
    for bucket in sorted(args.coarse_buckets):
        count = bucket_counts.get(bucket, 0)
        bucket_rows.append(
            {
                "bucket": bucket,
                "count": count,
                "fraction": round(count / len(rows), 6),
            }
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "entropy_distribution_overall.csv", [overall_summary])
    _write_csv(output_dir / "entropy_distribution_by_task.csv", task_rows)
    _write_csv(output_dir / "entropy_distribution_buckets.csv", bucket_rows)

    payload = {
        "overall": overall_summary,
        "by_task": task_rows,
        "buckets": bucket_rows,
        "input_files": [str(path) for path in args.predictions],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()