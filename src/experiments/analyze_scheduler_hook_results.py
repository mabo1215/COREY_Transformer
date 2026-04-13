from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare hook-disabled and hook-enabled benchmark outputs and summarize real scheduler overhead."
    )
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--hook-root", type=Path, required=True)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["task"], row["sample_index"], row["metric"]


def _summarize_model(
    model_name: str,
    baseline_rows: list[dict[str, str]],
    hook_rows: list[dict[str, str]],
    baseline_metadata: dict[str, Any],
    hook_metadata: dict[str, Any],
) -> dict[str, Any]:
    baseline_map = {_row_key(row): row for row in baseline_rows}
    hook_map = {_row_key(row): row for row in hook_rows}
    shared_keys = sorted(set(baseline_map) & set(hook_map))
    if not shared_keys:
        raise ValueError(f"No overlapping benchmark rows found for {model_name}.")

    latency_baseline: list[float] = []
    latency_hook: list[float] = []
    metric_baseline: list[float] = []
    metric_hook: list[float] = []
    entropy_values: list[float] = []
    tile_values: list[float] = []
    prompt_tokens: list[float] = []
    generated_tokens: list[float] = []

    per_sample_rows: list[dict[str, Any]] = []
    for key in shared_keys:
        baseline_row = baseline_map[key]
        hook_row = hook_map[key]
        baseline_latency = _as_float(baseline_row.get("latency_mean_ms"))
        hook_latency = _as_float(hook_row.get("latency_mean_ms"))
        baseline_metric = _as_float(baseline_row.get("metric_mean"))
        hook_metric = _as_float(hook_row.get("metric_mean"))
        entropy = _as_float(hook_row.get("entropy_before"))
        tile_size = _as_float(hook_row.get("suggested_tile_size"))
        prompt_token_count = _as_float(hook_row.get("prompt_tokens"))
        generated_token_count = _as_float(hook_row.get("generated_tokens"))

        if baseline_latency is not None:
            latency_baseline.append(baseline_latency)
        if hook_latency is not None:
            latency_hook.append(hook_latency)
        if baseline_metric is not None:
            metric_baseline.append(baseline_metric)
        if hook_metric is not None:
            metric_hook.append(hook_metric)
        if entropy is not None:
            entropy_values.append(entropy)
        if tile_size is not None:
            tile_values.append(tile_size)
        if prompt_token_count is not None:
            prompt_tokens.append(prompt_token_count)
        if generated_token_count is not None:
            generated_tokens.append(generated_token_count)

        overhead_pct = None
        if baseline_latency not in {None, 0.0} and hook_latency is not None:
            overhead_pct = (hook_latency - baseline_latency) / baseline_latency * 100.0

        per_sample_rows.append(
            {
                "model": model_name,
                "task": key[0],
                "sample_index": key[1],
                "metric": key[2],
                "baseline_metric": baseline_metric,
                "hook_metric": hook_metric,
                "metric_delta": None if baseline_metric is None or hook_metric is None else hook_metric - baseline_metric,
                "baseline_latency_ms": baseline_latency,
                "hook_latency_ms": hook_latency,
                "hook_overhead_pct": overhead_pct,
                "entropy_before": entropy,
                "suggested_tile_size": tile_size,
                "prompt_tokens": prompt_token_count,
                "generated_tokens": generated_token_count,
            }
        )

    baseline_latency_mean = mean(latency_baseline)
    hook_latency_mean = mean(latency_hook)
    metric_delta_mean = mean(hook - base for base, hook in zip(metric_baseline, metric_hook, strict=False))
    hook_overhead_pct = (hook_latency_mean - baseline_latency_mean) / baseline_latency_mean * 100.0

    return {
        "model": model_name,
        "task_count": len(shared_keys),
        "baseline_latency_ms": round(baseline_latency_mean, 4),
        "hook_latency_ms": round(hook_latency_mean, 4),
        "hook_overhead_pct": round(hook_overhead_pct, 4),
        "baseline_metric": round(mean(metric_baseline), 6),
        "hook_metric": round(mean(metric_hook), 6),
        "metric_delta": round(metric_delta_mean, 6),
        "mean_entropy_before": round(mean(entropy_values), 6) if entropy_values else None,
        "mean_suggested_tile_size": round(mean(tile_values), 4) if tile_values else None,
        "mean_prompt_tokens": round(mean(prompt_tokens), 4) if prompt_tokens else None,
        "mean_generated_tokens": round(mean(generated_tokens), 4) if generated_tokens else None,
        "baseline_fast_path_available": baseline_metadata.get("fast_path_available"),
        "hook_fast_path_available": hook_metadata.get("fast_path_available"),
        "device": hook_metadata.get("system", {}).get("platform"),
        "per_sample_rows": per_sample_rows,
    }


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = sorted(
        path.name
        for path in args.hook_root.iterdir()
        if path.is_dir() and (args.baseline_root / path.name / args.precision / "summary.csv").exists()
    )
    aggregate_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []

    for model_name in model_dirs:
        baseline_dir = args.baseline_root / model_name / args.precision
        hook_dir = args.hook_root / model_name / args.precision
        summary = _summarize_model(
            model_name=model_name,
            baseline_rows=_read_csv(baseline_dir / "summary.csv"),
            hook_rows=_read_csv(hook_dir / "summary.csv"),
            baseline_metadata=_load_metadata(baseline_dir / "metadata.json"),
            hook_metadata=_load_metadata(hook_dir / "metadata.json"),
        )
        per_sample_rows.extend(summary.pop("per_sample_rows"))
        aggregate_rows.append(summary)

    _write_csv(output_dir / "hook_overhead_summary.csv", aggregate_rows)
    _write_csv(output_dir / "hook_overhead_per_sample.csv", per_sample_rows)
    (output_dir / "hook_overhead_summary.json").write_text(
        json.dumps({"models": aggregate_rows}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()