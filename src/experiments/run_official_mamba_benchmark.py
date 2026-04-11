from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import psutil

from src.algorithms.mamba_integration import GenerationRequest
from src.algorithms.mamba_integration import default_mamba_model_specs
from src.algorithms.mamba_integration import official_mamba_fast_path_status
from src.experiments.run_longbench_inference import (
    LONG_BENCH_TASKS,
    _build_backend,
    _load_lm_samples,
    _load_task_samples,
    _metric_value,
    _reference_text,
    _render_prompt,
)


def _parse_args() -> argparse.Namespace:
    model_choices = [spec.name for spec in default_mamba_model_specs()]
    parser = argparse.ArgumentParser(
        description="Run repeated official Hugging Face Mamba benchmark passes and export structured results."
    )
    parser.add_argument("--model", default="mamba-370m", choices=model_choices)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-source", default="auto", choices=["auto", "local", "hf"])
    parser.add_argument("--dataset-name", default="THUDM/LongBench")
    parser.add_argument("--dataset-config")
    parser.add_argument("--split", default="test")
    parser.add_argument("--task", default="narrativeqa", choices=sorted(LONG_BENCH_TASKS.keys()))
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--hf-model-id")
    parser.add_argument("--backend", default="hf", choices=["hf", "ollama"])
    parser.add_argument("--ollama-model")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--hf-token")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--precision", default="fp32", choices=["fp16", "fp32", "w8a8", "w4a8"])
    parser.add_argument("--quant-backend", choices=["awq", "gptq"])
    parser.add_argument("--quant-bits", type=int)
    parser.add_argument("--quant-group-size", type=int, default=128)
    parser.add_argument("--use-exllama", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lm-datasets", nargs="+", choices=["wikitext103", "pg19"], default=[])
    parser.add_argument("--lm-max-samples", type=int, default=1)
    parser.add_argument("--disable-entropy-hook", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("src/outputs/official_hf_benchmark"))
    return parser.parse_args()


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
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


def _process_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    backend = _build_backend(args)
    backend.load()
    is_mamba_model = args.model.startswith("mamba-")
    if is_mamba_model:
        fast_path_status: dict[str, Any] = official_mamba_fast_path_status()
        fast_path_available: bool | None = all(
            fast_path_status.get(key) is True
            for key in [
                "selective_scan_fn",
                "mamba_inner_fn",
                "selective_state_update",
                "causal_conv1d_fn",
                "causal_conv1d_update",
            ]
        )
    else:
        fast_path_status = {
            "selective_scan_fn": None,
            "mamba_inner_fn": None,
            "selective_state_update": None,
            "causal_conv1d_fn": None,
            "causal_conv1d_update": None,
            "error": "not_applicable_non_mamba_model",
        }
        fast_path_available = None

    task = LONG_BENCH_TASKS[args.task]
    samples = _load_task_samples(args, args.task, args.split, args.max_samples)
    repeated_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for sample_index, sample in enumerate(samples):
        prompt = _render_prompt(task, sample)
        request = GenerationRequest(
            prompt=prompt,
            max_new_tokens=args.max_new_tokens or task.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0.0,
        )
        reference = _reference_text(sample, task.answer_field)

        for _ in range(max(0, args.warmup_runs)):
            backend.generate(request)

        latencies: list[float] = []
        throughputs: list[float] = []
        rss_values: list[float] = []
        metric_values: list[float] = []
        latest_output = None
        for repeat_index in range(max(1, args.benchmark_repeats)):
            output = backend.generate(request)
            latest_output = output
            metric_value = _metric_value(task.metric, output.text, reference)
            latencies.append(output.telemetry.latency_ms)
            throughputs.append(output.telemetry.tokens_per_second)
            rss_values.append(_process_rss_mb())
            metric_values.append(metric_value)
            repeated_rows.append(
                {
                    "task": args.task,
                    "sample_index": sample_index,
                    "repeat_index": repeat_index,
                    "metric": task.metric,
                    "metric_value": round(metric_value, 6),
                    "latency_ms": round(output.telemetry.latency_ms, 4),
                    "tokens_per_second": round(output.telemetry.tokens_per_second, 4),
                    "prompt_tokens": output.telemetry.prompt_tokens,
                    "generated_tokens": output.telemetry.generated_tokens,
                    "rss_mb": round(rss_values[-1], 4),
                    "entropy_before": output.entropy_before,
                    "entropy_after": output.entropy_after,
                    "suggested_tile_size": output.suggested_tile_size,
                    "prediction": output.text,
                    "reference": reference,
                }
            )

        summary_rows.append(
            {
                "task": args.task,
                "sample_index": sample_index,
                "metric": task.metric,
                "metric_mean": round(mean(metric_values), 6) if metric_values else None,
                "latency_mean_ms": round(mean(latencies), 4) if latencies else None,
                "latency_std_ms": round(pstdev(latencies), 4) if len(latencies) > 1 else 0.0,
                "latency_min_ms": round(min(latencies), 4) if latencies else None,
                "tokens_per_second_mean": round(mean(throughputs), 4) if throughputs else None,
                "tokens_per_second_std": round(pstdev(throughputs), 4) if len(throughputs) > 1 else 0.0,
                "peak_rss_mb": round(max(rss_values), 4) if rss_values else None,
                "prompt_tokens": latest_output.telemetry.prompt_tokens if latest_output is not None else None,
                "generated_tokens": latest_output.telemetry.generated_tokens if latest_output is not None else None,
                "entropy_before": latest_output.entropy_before if latest_output is not None else None,
                "entropy_after": latest_output.entropy_after if latest_output is not None else None,
                "suggested_tile_size": latest_output.suggested_tile_size if latest_output is not None else None,
                "fast_path_available": fast_path_available,
                "deployment_grade": args.device.startswith("cuda") and (fast_path_available if is_mamba_model else True),
            }
        )

    lm_rows: list[dict[str, Any]] = []
    for dataset_key in args.lm_datasets:
        try:
            texts = _load_lm_samples(args, dataset_key)
            perplexities = [value for value in backend.score_perplexity_batch(texts) if value is not None]
            lm_rows.append(
                {
                    "task": dataset_key,
                    "metric": "perplexity",
                    "samples": len(texts),
                    "perplexity_mean": round(mean(perplexities), 6) if perplexities else None,
                    "status": "ok" if perplexities else "blocked",
                    "error": None if perplexities else "No perplexity values were produced.",
                }
            )
        except Exception as exc:
            lm_rows.append(
                {
                    "task": dataset_key,
                    "metric": "perplexity",
                    "samples": 0,
                    "perplexity_mean": None,
                    "status": "blocked",
                    "error": str(exc),
                }
            )

    output_dir = args.output_dir / args.model / args.precision
    _write_rows(output_dir / "repeats.csv", repeated_rows)
    _write_rows(output_dir / "summary.csv", summary_rows)
    _write_rows(output_dir / "lm_summary.csv", lm_rows)
    metadata = {
        "model": args.model,
        "hf_model_id": args.hf_model_id,
        "backend": args.backend,
        "ollama_model": args.ollama_model,
        "ollama_host": args.ollama_host,
        "device": args.device,
        "dtype": args.dtype,
        "precision": args.precision,
        "quant_backend": args.quant_backend,
        "quant_bits": args.quant_bits,
        "quant_group_size": args.quant_group_size,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "warmup_runs": args.warmup_runs,
        "benchmark_repeats": args.benchmark_repeats,
        "batch_size": args.batch_size,
        "disable_entropy_hook": args.disable_entropy_hook,
        "dataset_source": args.dataset_source,
        "dataset_name": args.dataset_name,
        "task": args.task,
        "split": args.split,
        "max_samples": args.max_samples,
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "logical_cpus": psutil.cpu_count(logical=True),
            "physical_cpus": psutil.cpu_count(logical=False),
            "memory_gb": round(psutil.virtual_memory().total / (1024.0 ** 3), 2),
        },
        "fast_path_status": fast_path_status,
        "fast_path_available": fast_path_available,
        "deployment_grade": args.device.startswith("cuda") and (fast_path_available if is_mamba_model else True),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "samples": len(samples),
        "repeats": len(repeated_rows),
        "lm_rows": len(lm_rows),
    }


def main() -> None:
    args = _parse_args()
    result = run_benchmark(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()