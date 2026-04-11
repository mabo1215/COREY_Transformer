from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from src.algorithms.mamba_integration import (
    EntropyGuidedSchedulerHook,
    GenerationRequest,
    HuggingFaceMambaBackend,
    ModelSpec,
    OllamaBackend,
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
    perplexity: float | None


LM_DATASET_SPECS = {
    "wikitext103": {
        "dataset_name": "wikitext",
        "config": "wikitext-103-raw-v1",
        "text_field": "text",
        "split": "test",
    },
    "pg19": {
        "dataset_name": "deepmind/pg19",
        "fallback_dataset_names": ["mrsndmn/pg19"],
        "config": None,
        "text_field": "text",
        "split": "test",
    },
}

DEFAULT_LONG_BENCH_DATASET_NAMES = ("zai-org/LongBench", "THUDM/LongBench")


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
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-source", default="auto", choices=["auto", "local", "hf"])
    parser.add_argument("--dataset-name", default=DEFAULT_LONG_BENCH_DATASET_NAMES[0])
    parser.add_argument("--dataset-config")
    parser.add_argument("--hf-model-id")
    parser.add_argument("--output-dir", type=Path, default=Path("src/outputs/longbench"))
    parser.add_argument("--tasks", nargs="+", default=sorted(LONG_BENCH_TASKS.keys()))
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32", "w8a8", "w4a8"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--backend", default="hf")
    parser.add_argument("--ollama-model")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--quant-backend", choices=["awq", "gptq"])
    parser.add_argument("--quant-bits", type=int)
    parser.add_argument("--quant-group-size", type=int, default=128)
    parser.add_argument("--use-exllama", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--hf-token")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--disable-entropy-hook", action="store_true")
    parser.add_argument("--eval-perplexity", action="store_true")
    parser.add_argument("--ppl-max-samples", type=int, default=0)
    parser.add_argument("--lm-datasets", nargs="+", choices=sorted(LM_DATASET_SPECS.keys()), default=[])
    parser.add_argument("--lm-max-samples", type=int, default=0)
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


def _require_hf_datasets() -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face dataset loading requires the datasets package."
        ) from exc
    return load_dataset


def _candidate_longbench_dataset_names(requested_name: str) -> list[str]:
    candidates = [requested_name, *DEFAULT_LONG_BENCH_DATASET_NAMES]
    resolved: list[str] = []
    for candidate in candidates:
        if candidate not in resolved:
            resolved.append(candidate)
    return resolved


def _normalize_longbench_sample(task: LongBenchTaskSpec, sample: dict[str, Any], sample_id: int) -> dict[str, object]:
    normalized = dict(sample)
    context_value = _first_value(
        normalized,
        [
            "context",
            "article",
            "document",
            "documents",
            "passage",
            "passages",
            "text",
            "content",
            "input_context",
        ],
    )
    if context_value is not None:
        normalized["context"] = _stringify_value(context_value)

    input_value = _first_value(
        normalized,
        [
            task.input_field,
            "question",
            "query",
            "instruction",
            "prompt",
            "query_text",
        ],
    )
    if input_value is not None:
        normalized[task.input_field] = _stringify_value(input_value)

    answer_value = _first_value(
        normalized,
        [task.answer_field, "answers", "answer", "label", "labels", "output", "outputs", "target", "targets"],
    )
    if answer_value is not None:
        normalized[task.answer_field] = _coerce_answer_list(answer_value)

    normalized.setdefault("id", str(sample_id))
    normalized.setdefault("context", "")
    normalized.setdefault(task.input_field, "")
    normalized.setdefault(task.answer_field, [])
    return normalized


def _first_value(sample: dict[str, Any], candidate_keys: list[str]) -> Any | None:
    for key in candidate_keys:
        if key in sample and sample[key] not in (None, "", []):
            return sample[key]
    metadata = sample.get("metadata")
    if isinstance(metadata, dict):
        for key in candidate_keys:
            if key in metadata and metadata[key] not in (None, "", []):
                return metadata[key]
    return None


def _stringify_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_stringify_value(item) for item in value)
    if isinstance(value, dict):
        for candidate_key in ("text", "content", "value"):
            if candidate_key in value:
                return _stringify_value(value[candidate_key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _coerce_answer_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_stringify_value(item) for item in value if _stringify_value(item)]
    return [_stringify_value(value)] if _stringify_value(value) else []


def _load_local_task_samples(dataset_root: Path, task: LongBenchTaskSpec, split: str, max_samples: int) -> list[dict[str, object]]:
    task_path = dataset_root / task.name / f"{split}.jsonl"
    if not task_path.exists():
        raise FileNotFoundError(f"Expected LongBench file at {task_path}")

    samples: list[dict[str, object]] = []
    with task_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_samples and index >= max_samples:
                break
            loaded = json.loads(line)
            samples.append(_normalize_longbench_sample(task, loaded, index))
    return samples


def _load_hf_task_samples(args: argparse.Namespace, task: LongBenchTaskSpec, max_samples: int) -> list[dict[str, object]]:
    load_dataset = _require_hf_datasets()
    config_candidates: list[tuple[str | None, str]] = []
    if args.dataset_config:
        config_candidates.append((args.dataset_config, args.split))
    config_candidates.append((task.name, args.split))
    config_candidates.append((None, args.split))

    last_error: Exception | None = None
    dataset = None
    for dataset_name in _candidate_longbench_dataset_names(args.dataset_name):
        for config_name, split_name in config_candidates:
            try:
                dataset = load_dataset(
                    dataset_name,
                    name=config_name,
                    split=split_name,
                    cache_dir=str(args.cache_dir) if args.cache_dir else None,
                    token=args.hf_token,
                )
                break
            except Exception as exc:  # pragma: no cover - depends on external dataset layouts
                last_error = exc
        if dataset is not None:
            break
    if dataset is None:
        raise RuntimeError(f"Unable to load task {task.name} from any known LongBench dataset id.") from last_error

    samples: list[dict[str, object]] = []
    for index, sample in enumerate(dataset):
        loaded = dict(sample)
        dataset_name = str(loaded.get("dataset", loaded.get("subset", loaded.get("task", "")))).lower()
        if dataset_name and task.name not in dataset_name:
            continue
        samples.append(_normalize_longbench_sample(task, loaded, index))
        if max_samples and len(samples) >= max_samples:
            break
    return samples


def _load_lm_samples(args: argparse.Namespace, dataset_key: str) -> list[str]:
    load_dataset = _require_hf_datasets()
    spec = LM_DATASET_SPECS[dataset_key]
    dataset_names = [spec["dataset_name"], *spec.get("fallback_dataset_names", [])]
    dataset = None
    last_error: Exception | None = None
    for dataset_name in dataset_names:
        try:
            dataset = load_dataset(
                dataset_name,
                name=spec["config"],
                split=spec["split"],
                cache_dir=str(args.cache_dir) if args.cache_dir else None,
                token=args.hf_token,
            )
            break
        except Exception as exc:  # pragma: no cover - depends on external dataset availability
            last_error = exc
    if dataset is None:
        raise RuntimeError(f"Unable to load language-model dataset {dataset_key} from any configured source.") from last_error

    texts: list[str] = []
    for sample in dataset:
        text = _stringify_value(sample.get(spec["text_field"], "")).strip()
        if not text:
            continue
        texts.append(text)
        if args.lm_max_samples and len(texts) >= args.lm_max_samples:
            break
    return texts


def _load_task_samples(args: argparse.Namespace, task_name: str, split: str, max_samples: int) -> list[dict[str, object]]:
    task = LONG_BENCH_TASKS[task_name]
    source = args.dataset_source
    if source in {"auto", "local"} and args.dataset_root is not None:
        task_path = args.dataset_root / task_name / f"{split}.jsonl"
        if task_path.exists():
            return _load_local_task_samples(args.dataset_root, task, split, max_samples)
        if source == "local":
            raise FileNotFoundError(f"Expected LongBench file at {task_path}")
    if source in {"auto", "hf"}:
        try:
            return _load_hf_task_samples(args, task, max_samples)
        except RuntimeError as exc:
            task_path = args.dataset_root / task_name / f"{split}.jsonl" if args.dataset_root is not None else None
            dataset_script_disabled = "Dataset scripts are no longer supported" in str(exc.__cause__ or exc)
            if dataset_script_disabled and task_path is not None and task_path.exists():
                return _load_local_task_samples(args.dataset_root, task, split, max_samples)
            raise
    raise ValueError("No dataset source could be resolved for LongBench loading.")


def _render_prompt(task: LongBenchTaskSpec, sample: dict[str, object]) -> str:
    context = str(sample.get("context", ""))
    user_input = str(sample.get(task.input_field, ""))
    return task.prompt_template.format(context=context, input=user_input)


def _reference_text(sample: dict[str, object], answer_field: str) -> str:
    answer = sample.get(answer_field, "")
    if isinstance(answer, list):
        return str(answer[0]) if answer else ""
    return str(answer)


def _build_backend(args: argparse.Namespace):
    model_lookup = {spec.name: spec for spec in default_mamba_model_specs()}
    model_id = args.ollama_model or args.hf_model_id or model_lookup[args.model].model_id
    if args.backend != "ollama" and model_id == "mlx-community/mamba-1.4b-hf-f32":
        model_id = "state-spaces/mamba-1.4b-hf"
    model_spec = ModelSpec(
        name=args.model,
        model_id=model_id,
        revision=model_lookup[args.model].revision,
        trust_remote_code=model_lookup[args.model].trust_remote_code,
    )
    runtime_config = RuntimeConfig(
        device=args.device,
        batch_size=max(1, args.batch_size),
        quantization=QuantizationConfig(
            mode=args.precision,
            backend=args.quant_backend,
            bits=args.quant_bits,
            group_size=args.quant_group_size,
            use_exllama=args.use_exllama,
        ),
        dtype=args.dtype,
        max_length=args.max_length,
    )
    scheduler_hook = None if args.disable_entropy_hook else EntropyGuidedSchedulerHook(entropy_threshold=5.0)
    if args.backend == "ollama":
        return OllamaBackend(
            model_spec=model_spec,
            runtime_config=runtime_config,
            scheduler_hook=None,
            host=args.ollama_host,
        )
    return HuggingFaceMambaBackend(model_spec=model_spec, runtime_config=runtime_config, scheduler_hook=scheduler_hook)


def _batched(items: list[dict[str, object]], batch_size: int) -> list[list[dict[str, object]]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _perplexity_text(prompt: str, reference: str) -> str:
    return f"{prompt}\n{reference}" if reference else prompt


def _write_predictions(path: Path, rows: list[PredictionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_longbench(args: argparse.Namespace) -> dict[str, object]:
    backend = _build_backend(args)
    prediction_rows: list[PredictionRecord] = []
    summary_rows: list[dict[str, object]] = []
    output_dir = args.output_dir / args.model / args.precision
    metadata = {
        "model": args.model,
        "requested_model_id": args.ollama_model or args.hf_model_id or {spec.name: spec.model_id for spec in default_mamba_model_specs()}[args.model],
        "resolved_model_id": _build_backend(args).model_spec.model_id,
        "backend": args.backend,
        "ollama_host": args.ollama_host if args.backend == "ollama" else None,
        "device": args.device,
        "dtype": args.dtype,
        "precision": args.precision,
        "quant_backend": args.quant_backend,
        "quant_bits": args.quant_bits,
        "quant_group_size": args.quant_group_size,
        "use_exllama": args.use_exllama,
        "dataset_source": args.dataset_source,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "disable_entropy_hook": args.disable_entropy_hook,
        "eval_perplexity": args.eval_perplexity,
        "ppl_max_samples": args.ppl_max_samples,
        "lm_datasets": args.lm_datasets,
        "lm_max_samples": args.lm_max_samples,
        "tasks": args.tasks,
        "split": args.split,
        "max_samples": args.max_samples,
        "dry_run": args.dry_run,
        "completed_tasks": [],
    }
    _write_metadata(output_dir / "metadata.json", metadata)

    for task_name in args.tasks:
        task = LONG_BENCH_TASKS[task_name]
        samples = _load_task_samples(args, task_name, args.split, args.max_samples)
        if args.dry_run:
            summary_rows.append(
                {
                    "task": task_name,
                    "eval_type": "longbench",
                    "samples": len(samples),
                    "metric": task.metric,
                    "max_new_tokens": task.max_new_tokens,
                    "model": args.model,
                    "backend": args.backend,
                    "precision": args.precision,
                    "status": "dry_run",
                    "error": None,
                    "mode": "dry_run",
                }
            )
            continue

        task_scores: list[float] = []
        task_latency: list[float] = []
        task_tps: list[float] = []
        task_perplexities: list[float] = []
        perplexity_budget = args.ppl_max_samples if args.ppl_max_samples > 0 else len(samples)
        seen_for_perplexity = 0
        for batch in _batched(samples, max(1, args.batch_size)):
            requests = [
                GenerationRequest(
                    prompt=_render_prompt(task, sample),
                    max_new_tokens=task.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=args.temperature > 0.0,
                )
                for sample in batch
            ]
            outputs = backend.generate_batch(requests)

            perplexity_inputs: list[str] = []
            perplexity_indices: list[int] = []
            if args.eval_perplexity:
                for local_index, sample in enumerate(batch):
                    if seen_for_perplexity >= perplexity_budget:
                        break
                    perplexity_inputs.append(
                        _perplexity_text(requests[local_index].prompt, _reference_text(sample, task.answer_field))
                    )
                    perplexity_indices.append(local_index)
                    seen_for_perplexity += 1
            perplexity_values: dict[int, float | None] = {}
            if perplexity_inputs:
                for offset, perplexity in zip(perplexity_indices, backend.score_perplexity_batch(perplexity_inputs), strict=False):
                    perplexity_values[offset] = perplexity

            for index, (sample, request, output) in enumerate(zip(batch, requests, outputs, strict=False)):
                reference = _reference_text(sample, task.answer_field)
                metric_value = _metric_value(task.metric, output.text, reference)
                task_scores.append(metric_value)
                task_latency.append(output.telemetry.latency_ms)
                task_tps.append(output.telemetry.tokens_per_second)
                perplexity = perplexity_values.get(index)
                if perplexity is not None:
                    task_perplexities.append(perplexity)
                prediction_rows.append(
                    PredictionRecord(
                        task=task_name,
                        sample_id=str(sample.get("id", len(prediction_rows))),
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
                        perplexity=perplexity,
                    )
                )

        summary_rows.append(
            {
                "task": task_name,
                "eval_type": "longbench",
                "samples": len(samples),
                "metric": task.metric,
                "score": round(mean(task_scores), 6) if task_scores else 0.0,
                "perplexity": round(mean(task_perplexities), 6) if task_perplexities else None,
                "latency_ms": round(mean(task_latency), 4) if task_latency else 0.0,
                "tokens_per_second": round(mean(task_tps), 4) if task_tps else 0.0,
                "model": args.model,
                "backend": args.backend,
                "precision": args.precision,
                "status": "ok",
                "error": None,
                "mode": "inference",
            }
        )
        metadata["completed_tasks"].append(task_name)
        _write_predictions(output_dir / "predictions.jsonl", prediction_rows)
        _write_summary(output_dir / "summary.csv", summary_rows)
        _write_metadata(output_dir / "metadata.json", metadata)

    for dataset_key in args.lm_datasets:
        try:
            lm_texts = _load_lm_samples(args, dataset_key)
        except Exception as exc:
            summary_rows.append(
                {
                    "task": dataset_key,
                    "eval_type": "language_modeling",
                    "samples": 0,
                    "metric": "perplexity",
                    "score": None,
                    "perplexity": None,
                    "latency_ms": None,
                    "tokens_per_second": None,
                    "model": args.model,
                    "backend": args.backend,
                    "precision": args.precision,
                    "status": "blocked",
                    "error": str(exc),
                    "mode": "inference",
                }
            )
            continue
        if args.dry_run:
            summary_rows.append(
                {
                    "task": dataset_key,
                    "eval_type": "language_modeling",
                    "samples": len(lm_texts),
                    "metric": "perplexity",
                    "score": None,
                    "perplexity": None,
                    "latency_ms": None,
                    "tokens_per_second": None,
                    "model": args.model,
                    "backend": args.backend,
                    "precision": args.precision,
                    "status": "dry_run",
                    "error": None,
                    "mode": "dry_run",
                }
            )
            continue

        try:
            perplexities = [value for value in backend.score_perplexity_batch(lm_texts) if value is not None]
            status = "ok"
            error_message = None
        except Exception as exc:
            perplexities = []
            status = "blocked"
            error_message = str(exc)
        summary_rows.append(
            {
                "task": dataset_key,
                "eval_type": "language_modeling",
                "samples": len(lm_texts),
                "metric": "perplexity",
                "score": None,
                "perplexity": round(mean(perplexities), 6) if perplexities else None,
                "latency_ms": None,
                "tokens_per_second": None,
                "model": args.model,
                "backend": args.backend,
                "precision": args.precision,
                "status": status,
                "error": error_message,
                "mode": "inference",
            }
        )
        _write_summary(output_dir / "summary.csv", summary_rows)
        _write_metadata(output_dir / "metadata.json", metadata)

    _write_predictions(output_dir / "predictions.jsonl", prediction_rows)
    _write_summary(output_dir / "summary.csv", summary_rows)
    _write_metadata(output_dir / "metadata.json", metadata)
    return {"predictions": len(prediction_rows), "summaries": len(summary_rows), "output_dir": str(output_dir)}


def main() -> None:
    args = _parse_args()
    run_longbench(args)


if __name__ == "__main__":
    main()