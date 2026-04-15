from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _collect_policy_rows(root: Path, policy: str, expected_models: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    benchmark_root = root / "benchmark"
    if not benchmark_root.exists():
        for model in expected_models:
            rows.append(
                {
                    "policy": policy,
                    "model": model,
                    "status": "missing",
                    "task_count": 0,
                    "latency_mean_ms": None,
                    "tokens_per_second_mean": None,
                    "metric_mean": None,
                    "source": str(benchmark_root / model / "fp16" / "summary.csv"),
                }
            )
        return rows

    discovered = {p.name: p for p in benchmark_root.iterdir() if p.is_dir()}
    for model_name in expected_models:
        model_dir = discovered.get(model_name)
        if model_dir is None:
            rows.append(
                {
                    "policy": policy,
                    "model": model_name,
                    "status": "missing",
                    "task_count": 0,
                    "latency_mean_ms": None,
                    "tokens_per_second_mean": None,
                    "metric_mean": None,
                    "source": str(benchmark_root / model_name / "fp16" / "summary.csv"),
                }
            )
            continue

        summary_path = model_dir / "fp16" / "summary.csv"
        summary_rows = _read_csv(summary_path)
        task_rows = [r for r in summary_rows if r.get("task") not in {"wikitext103", "pg19"}]
        if not task_rows:
            rows.append(
                {
                    "policy": policy,
                    "model": model_dir.name,
                    "status": "missing",
                    "task_count": 0,
                    "latency_mean_ms": None,
                    "tokens_per_second_mean": None,
                    "metric_mean": None,
                    "source": str(summary_path),
                }
            )
            continue

        latencies = [_to_float(r.get("latency_mean_ms")) for r in task_rows]
        tps_values = [_to_float(r.get("tokens_per_second_mean")) for r in task_rows]
        metric_values = [_to_float(r.get("metric_mean")) for r in task_rows]

        latencies = [x for x in latencies if x is not None]
        tps_values = [x for x in tps_values if x is not None]
        metric_values = [x for x in metric_values if x is not None]

        rows.append(
            {
                "policy": policy,
                "model": model_dir.name,
                "status": "ok",
                "task_count": len(task_rows),
                "latency_mean_ms": round(mean(latencies), 4) if latencies else None,
                "tokens_per_second_mean": round(mean(tps_values), 4) if tps_values else None,
                "metric_mean": round(mean(metric_values), 6) if metric_values else None,
                "source": str(summary_path),
            }
        )

    return rows


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

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact policy comparison table for W1 (off/static/corey) from checkpoint-matrix outputs."
    )
    parser.add_argument("--off-root", type=Path, required=True)
    parser.add_argument("--static-root", type=Path, required=True)
    parser.add_argument("--corey-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["mamba-370m", "mamba-1.4b", "mamba-2.8b"])
    parser.add_argument("--output-csv", type=Path, default=Path("src/outputs/w1_policy_comparison.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("src/outputs/w1_policy_comparison.json"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    rows: list[dict[str, Any]] = []
    rows.extend(_collect_policy_rows(args.off_root, "off", list(args.models)))
    rows.extend(_collect_policy_rows(args.static_root, "static", list(args.models)))
    rows.extend(_collect_policy_rows(args.corey_root, "corey", list(args.models)))

    _write_csv(args.output_csv, rows)

    payload = {
        "off_root": str(args.off_root),
        "static_root": str(args.static_root),
        "corey_root": str(args.corey_root),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({"rows": len(rows), "output_csv": str(args.output_csv), "output_json": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
