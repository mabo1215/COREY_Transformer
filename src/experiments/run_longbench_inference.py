from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from src.algorithms.mamba_integration import (
    EntropyGuidedSchedulerHook,
    GenerationRequest,
    HuggingFaceMambaBackend,
    ModelSpec,
    QuantizationConfig,
    RuntimeConfig,
    default_mamba_model_specs,
)


@dataclass(frozen=True)
class LongBenchTaskSpec:
    name: str
    prompt_template: str
    input_field: str
    answer_field: str
    metric: str
    max_new_tokens: int


@dataclass(frozen=True)
class PredictionRecord:
    task: str
    sample_id: str
    prediction: str
    reference: str
    metric_name: str
    metric_value: float
    latency_ms: float
    tokens_per_second: float
    prompt_tokens: int
    generated_tokens: int
    entropy_before: float | None
    entropy_after: float | None
    suggested_tile_size: int | None


LONG_BENCH_TASKS = {
    "narrativeqa": LongBenchTaskSpec(
        name="narrativeqa",
        prompt_template="Context:\n{context}\n\nQuestion: {input}\nAnswer:",
        input_field="input",
        answer_field="answers",
        metric="token_f1",
        max_new_tokens=128,
    ),
    "qasper": LongBenchTaskSpec(
        name="qasper",
        prompt_template="Document:\n{context}\n\nQuestion: {input}\nAnswer:",
        input_field="input",
        answer_field="answers",
        metric="token_f1",
        max_new_tokens=96,
    ),
    "multifieldqa_en": LongBenchTaskSpec(
        name="multifieldqa_en",
        prompt_template="Document:\n{context}\n\nQuestion: {input}\nShort answer:",
        input_field="input",
        answer_field="answers",
        metric="exact_match",
        max_new_tokens=64,
    ),
    "gov_report": LongBenchTaskSpec(
        name="gov_report",
        prompt_template="Report:\n{context}\n\nWrite a concise summary:\n",
        input_field="input",
        answer_field="answers",
        metric="rouge_l",
        max_new_tokens=256,
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LongBench inference with a Mamba backend skeleton.")
    parser.add_argument("--model", default="mamba-370m", choices=[spec.name for spec in default_mamba_model_specs()])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("src/outputs/longbench"))
    parser.add_argument("--tasks", nargs="+", default=sorted(LONG_BENCH_TASKS.keys()))
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--precision", default="fp16", choices=["fp16", "w8a8", "w4a8"])
    parser.add_argument("--backend", default="hf")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _normalize_text(text: str) -> str:
    normalized = text.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9 ]", "", normalized)
    return normalized


def _tokenize(text: str) -> list[str]:
    normalized = _normalize_text(text)
    return [token for token in normalized.split(" ") if token]


def _exact_match(prediction: str, reference: str) -> float:
    return 1.0 if _normalize_text(prediction) == _normalize_text(reference) else 0.0


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    remaining = list(ref_tokens)
    overlap = 0
    for token in pred_tokens:
        if token in remaining:
            overlap += 1
            remaining.remove(token)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _lcs_length(lhs: list[str], rhs: list[str]) -> int:
    rows = len(lhs) + 1
    cols = len(rhs) + 1
    dp = [[0] * cols for _ in range(rows)]
    for row in range(1, rows):
        for col in range(1, cols):
            if lhs[row - 1] == rhs[col - 1]:
                dp[row][col] = dp[row - 1][col - 1] + 1
            else:
                dp[row][col] = max(dp[row - 1][col], dp[row][col - 1])
    return dp[-1][-1]


def _rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _metric_value(metric_name: str, prediction: str, reference: str) -> float:
    if metric_name == "exact_match":
        return _exact_match(prediction, reference)
    if metric_name == "token_f1":
        return _token_f1(prediction, reference)
    if metric_name == "rouge_l":
        return _rouge_l(prediction, reference)
    raise ValueError(f"Unsupported metric: {metric_name}")


def _load_task_samples(dataset_root: Path, task_name: str, split: str, max_samples: int) -> list[dict[str, object]]:
    task_path = dataset_root / task_name / f"{split}.jsonl"
    if not task_path.exists():
        raise FileNotFoundError(f"Expected LongBench file at {task_path}")

    samples: list[dict[str, object]] = []
    with task_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_samples and index >= max_samples:
                break
            samples.append(json.loads(line))
    return samples


def _render_prompt(task: LongBenchTaskSpec, sample: dict[str, object]) -> str:
    context = str(sample.get("context", ""))
    user_input = str(sample.get(task.input_field, ""))
    return task.prompt_template.format(context=context, input=user_input)


def _reference_text(sample: dict[str, object], answer_field: str) -> str:
    answer = sample.get(answer_field, "")
    if isinstance(answer, list):
        return str(answer[0]) if answer else ""
    return str(answer)


def _build_backend(model_name: str, precision: str) -> HuggingFaceMambaBackend:
    model_lookup = {spec.name: spec for spec in default_mamba_model_specs()}
    model_spec: ModelSpec = model_lookup[model_name]
    runtime_config = RuntimeConfig(
        quantization=QuantizationConfig(mode=precision),
        dtype="float16",
        max_length=32768,
    )
    scheduler_hook = EntropyGuidedSchedulerHook(entropy_threshold=5.0)
    return HuggingFaceMambaBackend(model_spec=model_spec, runtime_config=runtime_config, scheduler_hook=scheduler_hook)


def _write_predictions(path: Path, rows: list[PredictionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_longbench(args: argparse.Namespace) -> dict[str, object]:
    backend = _build_backend(args.model, args.precision)
    prediction_rows: list[PredictionRecord] = []
    summary_rows: list[dict[str, object]] = []

    for task_name in args.tasks:
        task = LONG_BENCH_TASKS[task_name]
        samples = _load_task_samples(args.dataset_root, task_name, args.split, args.max_samples)
        if args.dry_run:
            summary_rows.append(
                {
                    "task": task_name,
                    "samples": len(samples),
                    "metric": task.metric,
                    "max_new_tokens": task.max_new_tokens,
                    "model": args.model,
                    "precision": args.precision,
                    "mode": "dry_run",
                }
            )
            continue

        task_scores: list[float] = []
        task_latency: list[float] = []
        task_tps: list[float] = []
        for index, sample in enumerate(samples):
            prompt = _render_prompt(task, sample)
            reference = _reference_text(sample, task.answer_field)
            request = GenerationRequest(
                prompt=prompt,
                max_new_tokens=task.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.temperature > 0.0,
            )
            output = backend.generate(request)
            metric_value = _metric_value(task.metric, output.text, reference)
            task_scores.append(metric_value)
            task_latency.append(output.telemetry.latency_ms)
            task_tps.append(output.telemetry.tokens_per_second)
            prediction_rows.append(
                PredictionRecord(
                    task=task_name,
                    sample_id=str(sample.get("id", index)),
                    prediction=output.text,
                    reference=reference,
                    metric_name=task.metric,
                    metric_value=metric_value,
                    latency_ms=output.telemetry.latency_ms,
                    tokens_per_second=output.telemetry.tokens_per_second,
                    prompt_tokens=output.telemetry.prompt_tokens,
                    generated_tokens=output.telemetry.generated_tokens,
                    entropy_before=output.entropy_before,
                    entropy_after=output.entropy_after,
                    suggested_tile_size=output.suggested_tile_size,
                )
            )

        summary_rows.append(
            {
                "task": task_name,
                "samples": len(samples),
                "metric": task.metric,
                "score": round(mean(task_scores), 6) if task_scores else 0.0,
                "latency_ms": round(mean(task_latency), 4) if task_latency else 0.0,
                "tokens_per_second": round(mean(task_tps), 4) if task_tps else 0.0,
                "model": args.model,
                "precision": args.precision,
                "mode": "inference",
            }
        )

    output_dir = args.output_dir / args.model / args.precision
    _write_predictions(output_dir / "predictions.jsonl", prediction_rows)
    _write_summary(output_dir / "summary.csv", summary_rows)
    metadata = {
        "model": args.model,
        "precision": args.precision,
        "tasks": args.tasks,
        "split": args.split,
        "max_samples": args.max_samples,
        "dry_run": args.dry_run,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"predictions": len(prediction_rows), "summaries": len(summary_rows), "output_dir": str(output_dir)}


def main() -> None:
    args = _parse_args()
    run_longbench(args)


if __name__ == "__main__":
    main()