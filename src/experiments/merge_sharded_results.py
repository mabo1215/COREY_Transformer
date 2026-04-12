"""Merge sharded LongBench output directories into a single canonical summary.

Usage:
    python -m src.experiments.merge_sharded_results \
        --shard-dirs src/outputs/mgpu_shard_0 src/outputs/mgpu_shard_1 ... \
        --output-dir src/outputs/mgpu_merged \
        --model mamba-370m \
        --precision fp16
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded LongBench output directories.")
    parser.add_argument("--shard-dirs", nargs="+", required=True, type=Path, help="Shard output directories.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination for merged outputs.")
    parser.add_argument("--model", default="mamba-370m")
    parser.add_argument("--precision", default="fp16")
    return parser.parse_args()


def _load_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_predictions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge_shards(shard_dirs: list[Path], output_dir: Path, model: str, precision: str) -> None:
    sub = Path(model) / precision
    merged_preds: list[dict] = []
    # task_name -> list of per-shard metric rows (each shard has 1 row per task)
    task_rows: dict[str, list[dict]] = defaultdict(list)

    for shard_dir in shard_dirs:
        shard_sub = shard_dir / sub
        merged_preds.extend(_load_predictions(shard_sub / "predictions.jsonl"))
        for row in _load_summary(shard_sub / "summary.csv"):
            task_rows[row.get("task", row.get("dataset", ""))].append(row)

    # Write merged predictions
    out_sub = output_dir / sub
    out_sub.mkdir(parents=True, exist_ok=True)
    pred_path = out_sub / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for rec in merged_preds:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[merge] {len(merged_preds)} prediction records → {pred_path}")

    # Aggregate summary rows by task
    merged_summary: list[dict] = []
    fieldnames: list[str] = []
    for task, rows in task_rows.items():
        if not rows:
            continue
        base = dict(rows[0])
        # Average numeric fields across shards
        numeric_keys = ["score", "perplexity", "latency_ms", "tokens_per_second"]
        for key in numeric_keys:
            vals = []
            for r in rows:
                v = r.get(key)
                if v not in (None, "", "None"):
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass
            if vals:
                base[key] = round(mean(vals), 6)
        # Sum sample counts
        total_samples = 0
        for r in rows:
            try:
                total_samples += int(r.get("samples", 0) or 0)
            except ValueError:
                pass
        if total_samples:
            base["samples"] = total_samples
        for k in base:
            if k not in fieldnames:
                fieldnames.append(k)
        merged_summary.append(base)

    summary_path = out_sub / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_summary)
    print(f"[merge] {len(merged_summary)} summary rows → {summary_path}")

    # Write provenance metadata
    meta = {
        "merged_from": [str(d) for d in shard_dirs],
        "model": model,
        "precision": precision,
        "prediction_count": len(merged_preds),
        "tasks": list(task_rows.keys()),
    }
    (out_sub / "merge_metadata.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    args = _parse_args()
    merge_shards(args.shard_dirs, args.output_dir, args.model, args.precision)


if __name__ == "__main__":
    main()
