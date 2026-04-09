from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.experiments.run_longbench_inference import LM_DATASET_SPECS
from src.experiments.run_longbench_inference import LONG_BENCH_TASKS
from src.experiments.run_longbench_inference import run_longbench
from src.experiments.run_official_mamba_benchmark import run_benchmark


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable matrix of LongBench and official benchmark evaluations across Mamba checkpoints."
    )
    parser.add_argument("--modes", nargs="+", choices=["longbench", "benchmark"], default=["longbench", "benchmark"])
    parser.add_argument("--models", nargs="+", choices=["mamba-370m", "mamba-1.4b", "mamba-2.8b"], default=["mamba-370m", "mamba-1.4b", "mamba-2.8b"])
    parser.add_argument("--precisions", nargs="+", choices=["fp16", "fp32", "w8a8", "w4a8"], default=["fp16"])
    parser.add_argument("--tasks", nargs="+", choices=sorted(LONG_BENCH_TASKS.keys()), default=sorted(LONG_BENCH_TASKS.keys()))
    parser.add_argument("--lm-datasets", nargs="+", choices=sorted(LM_DATASET_SPECS.keys()), default=["wikitext103"])
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-source", default="auto", choices=["auto", "local", "hf"])
    parser.add_argument("--dataset-name", default="zai-org/LongBench")
    parser.add_argument("--dataset-config")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--benchmark-task", choices=sorted(LONG_BENCH_TASKS.keys()), default="narrativeqa")
    parser.add_argument("--benchmark-max-samples", type=int, default=1)
    parser.add_argument("--benchmark-max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    parser.add_argument("--backend", default="hf", choices=["hf", "ollama"])
    parser.add_argument("--hf-model-id")
    parser.add_argument("--ollama-model")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--hf-token")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--quant-backend", choices=["awq", "gptq"])
    parser.add_argument("--quant-bits", type=int)
    parser.add_argument("--quant-group-size", type=int, default=128)
    parser.add_argument("--use-exllama", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--eval-perplexity", action="store_true")
    parser.add_argument("--ppl-max-samples", type=int, default=0)
    parser.add_argument("--lm-max-samples", type=int, default=1)
    parser.add_argument("--disable-entropy-hook", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("src/outputs/checkpoint_matrix"))
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _status_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _longbench_completed(run_dir: Path, tasks: list[str], lm_datasets: list[str]) -> bool:
    rows = _read_csv_rows(run_dir / "summary.csv")
    if not rows:
        return False
    completed = {
        row.get("task", "")
        for row in rows
        if row.get("status") in {"ok", "blocked", "dry_run"}
    }
    expected = set(tasks) | set(lm_datasets)
    return expected.issubset(completed)


def _benchmark_completed(run_dir: Path, benchmark_task: str, lm_datasets: list[str]) -> bool:
    summary_rows = _read_csv_rows(run_dir / "summary.csv")
    if not any(row.get("task") == benchmark_task for row in summary_rows):
        return False
    if not lm_datasets:
        return True
    lm_rows = _read_csv_rows(run_dir / "lm_summary.csv")
    completed = {
        row.get("task", "")
        for row in lm_rows
        if row.get("status") in {"ok", "blocked", "dry_run"}
    }
    return set(lm_datasets).issubset(completed)


def _collect_longbench_rows(run_dir: Path, mode: str, model: str, precision: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv_rows(run_dir / "summary.csv"):
        enriched = dict(row)
        enriched.update(
            {
                "mode": mode,
                "model": model,
                "precision": precision,
                "source_dir": str(run_dir),
            }
        )
        rows.append(enriched)
    return rows


def _collect_benchmark_rows(run_dir: Path, mode: str, model: str, precision: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_name in ["summary.csv", "lm_summary.csv"]:
        for row in _read_csv_rows(run_dir / file_name):
            enriched = dict(row)
            enriched.update(
                {
                    "mode": mode,
                    "model": model,
                    "precision": precision,
                    "source_dir": str(run_dir),
                    "source_file": file_name,
                }
            )
            rows.append(enriched)
    return rows


def _build_longbench_namespace(args: argparse.Namespace, model: str, precision: str, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model=model,
        dataset_root=args.dataset_root,
        dataset_source=args.dataset_source,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        hf_model_id=args.hf_model_id,
        output_dir=output_dir,
        tasks=list(args.tasks),
        split=args.split,
        max_samples=args.max_samples,
        precision=precision,
        device=args.device,
        dtype=args.dtype,
        backend=args.backend,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        quant_backend=args.quant_backend,
        quant_bits=args.quant_bits,
        quant_group_size=args.quant_group_size,
        use_exllama=args.use_exllama,
        temperature=args.temperature,
        top_p=args.top_p,
        cache_dir=args.cache_dir,
        hf_token=args.hf_token,
        batch_size=args.batch_size,
        max_length=args.max_length,
        disable_entropy_hook=args.disable_entropy_hook,
        eval_perplexity=args.eval_perplexity,
        ppl_max_samples=args.ppl_max_samples,
        lm_datasets=list(args.lm_datasets),
        lm_max_samples=args.lm_max_samples,
        dry_run=args.dry_run,
    )


def _build_benchmark_namespace(args: argparse.Namespace, model: str, precision: str, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model=model,
        dataset_root=args.dataset_root,
        dataset_source=args.dataset_source,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        task=args.benchmark_task,
        max_samples=args.benchmark_max_samples,
        hf_model_id=args.hf_model_id,
        backend=args.backend,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        cache_dir=args.cache_dir,
        hf_token=args.hf_token,
        device=args.device,
        dtype=args.dtype,
        precision=precision,
        quant_backend=args.quant_backend,
        quant_bits=args.quant_bits,
        quant_group_size=args.quant_group_size,
        use_exllama=args.use_exllama,
        max_length=args.max_length,
        max_new_tokens=args.benchmark_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        warmup_runs=args.warmup_runs,
        benchmark_repeats=args.benchmark_repeats,
        batch_size=args.batch_size,
        lm_datasets=list(args.lm_datasets),
        lm_max_samples=args.lm_max_samples,
        disable_entropy_hook=args.disable_entropy_hook,
        output_dir=output_dir,
    )


def _append_failure_row(
    aggregate_rows: list[dict[str, Any]],
    mode: str,
    model: str,
    precision: str,
    run_dir: Path,
    error: str,
) -> None:
    aggregate_rows.append(
        {
            "mode": mode,
            "model": model,
            "precision": precision,
            "eval_type": "run",
            "task": "__run__",
            "status": "failed",
            "error": error,
            "source_dir": str(run_dir),
        }
    )


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []

    for mode in args.modes:
        for model in args.models:
            for precision in args.precisions:
                mode_root = output_root / mode
                run_dir = mode_root / model / precision
                manifest_row = {
                    "mode": mode,
                    "model": model,
                    "precision": precision,
                    "run_dir": str(run_dir),
                    "status": "pending",
                    "skipped": False,
                    "error": None,
                }
                try:
                    if mode == "longbench":
                        already_complete = _longbench_completed(run_dir, list(args.tasks), list(args.lm_datasets))
                        if args.skip_existing and already_complete:
                            manifest_row["status"] = "skipped"
                            manifest_row["skipped"] = True
                        else:
                            run_longbench(_build_longbench_namespace(args, model, precision, mode_root))
                            manifest_row["status"] = "ok"
                        aggregate_rows.extend(_collect_longbench_rows(run_dir, mode, model, precision))
                    else:
                        already_complete = _benchmark_completed(run_dir, args.benchmark_task, list(args.lm_datasets))
                        if args.skip_existing and already_complete:
                            manifest_row["status"] = "skipped"
                            manifest_row["skipped"] = True
                        else:
                            run_benchmark(_build_benchmark_namespace(args, model, precision, mode_root))
                            manifest_row["status"] = "ok"
                        aggregate_rows.extend(_collect_benchmark_rows(run_dir, mode, model, precision))
                except Exception as exc:
                    manifest_row["status"] = "failed"
                    manifest_row["error"] = str(exc)
                    _append_failure_row(aggregate_rows, mode, model, precision, run_dir, str(exc))
                    if args.fail_fast:
                        manifest.append(manifest_row)
                        raise
                manifest.append(manifest_row)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "modes": list(args.modes),
        "models": list(args.models),
        "precisions": list(args.precisions),
        "tasks": list(args.tasks),
        "lm_datasets": list(args.lm_datasets),
        "status_counts": _status_counts(manifest),
        "runs": manifest,
    }
    _write_csv_rows(output_root / "aggregate_summary.csv", aggregate_rows)
    _write_json(output_root / "run_manifest.json", summary)
    return {
        "output_dir": str(output_root),
        "aggregate_rows": len(aggregate_rows),
        "runs": len(manifest),
        "status_counts": summary["status_counts"],
    }


def main() -> None:
    args = _parse_args()
    result = run_matrix(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()