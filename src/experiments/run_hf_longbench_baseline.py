from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from src.experiments.run_longbench_inference import (
    LONG_BENCH_TASKS,
    _load_task_samples,
    _metric_value,
    _reference_text,
    _render_prompt,
)


@dataclass
class BaselineRow:
    task: str
    samples: int
    metric: str
    score: float | None
    latency_ms: float | None
    latency_std_ms: float | None
    tokens_per_second: float | None
    prompt_tokens_mean: float | None
    generated_tokens_mean: float | None
    status: str
    error: str | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a real HuggingFace causal-LM LongBench baseline. This is used "
            "for H800 Transformer+FlashAttention-3 and Mamba-2 SSD baselines."
        )
    )
    parser.add_argument("--model-name", default="transformer-fa3")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("src/data/longbench_subset"))
    parser.add_argument("--dataset-source", choices=["auto", "local", "hf"], default="auto")
    parser.add_argument("--dataset-name", default="zai-org/LongBench")
    parser.add_argument("--dataset-config")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--hf-token")
    parser.add_argument("--split", default="test")
    parser.add_argument("--tasks", nargs="+", default=["narrativeqa", "qasper", "gov_report", "multifieldqa_en"])
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=0, help="0 uses each task default.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("src/outputs/hf_longbench_baseline"))
    return parser.parse_args()


def _dtype(torch: Any, name: str) -> Any:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _patch_transformers_generation_compat() -> None:
    """Bridge older hub-kernel imports against newer Transformers generation names."""
    try:
        import transformers.generation as generation
    except Exception:
        return

    decoder_output = getattr(generation, "GenerateDecoderOnlyOutput", None)
    encoder_decoder_output = getattr(generation, "GenerateEncoderDecoderOutput", None)
    if decoder_output is not None:
        for name in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput"):
            if not hasattr(generation, name):
                setattr(generation, name, decoder_output)
    if encoder_decoder_output is not None:
        for name in ("GreedySearchEncoderDecoderOutput", "SampleEncoderDecoderOutput"):
            if not hasattr(generation, name):
                setattr(generation, name, encoder_decoder_output)


def _load_model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _patch_transformers_generation_compat()

    kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": _dtype(torch, args.dtype),
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)
    model.to(args.device)
    model.eval()
    return torch, model, tokenizer


def _batched(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), max(1, size))]


def _write_summary(path: Path, rows: list[BaselineRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BaselineRow.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "model_name": args.model_name,
        "model_id": args.model_id,
        "attn_implementation": args.attn_implementation,
        "device": args.device,
        "dtype": args.dtype,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "status": "starting",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    try:
        torch, model, tokenizer = _load_model(args)
    except Exception as exc:
        metadata.update({"status": "blocked", "error": repr(exc)})
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {"status": "blocked", "error": repr(exc), "output_dir": str(output_dir)}

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        metadata["gpu_name"] = torch.cuda.get_device_name(0)
        metadata["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        if args.attn_implementation == "flash_attention_3" and torch.cuda.get_device_capability(0)[0] < 9:
            metadata["status"] = "blocked"
            metadata["error"] = "flash_attention_3 requires Hopper/sm_90-class GPU."
            (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return {"status": "blocked", "error": metadata["error"], "output_dir": str(output_dir)}

    prediction_path = output_dir / "predictions.jsonl"
    rows: list[BaselineRow] = []
    with prediction_path.open("w", encoding="utf-8") as pred_handle:
        for task_name in args.tasks:
            task = LONG_BENCH_TASKS[task_name]
            task_args = argparse.Namespace(**vars(args))
            task_args.sample_offset = args.sample_offset
            samples = _load_task_samples(task_args, task_name, args.split, args.max_samples)
            scores: list[float] = []
            latencies: list[float] = []
            tps_values: list[float] = []
            prompt_tokens: list[int] = []
            generated_tokens: list[int] = []
            try:
                for batch_index, batch in enumerate(_batched(samples, args.batch_size)):
                    prompts = [_render_prompt(task, sample) for sample in batch]
                    encoded = tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=args.max_length,
                    ).to(args.device)
                    encoded["input_ids"] = encoded["input_ids"].long()
                    max_new_tokens = args.max_new_tokens or task.max_new_tokens

                    if batch_index == 0 and args.warmup > 0:
                        with torch.no_grad():
                            for _ in range(args.warmup):
                                model.generate(
                                    **encoded,
                                    max_new_tokens=min(8, max_new_tokens),
                                    do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id,
                                )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    generation_kwargs: dict[str, Any] = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": args.temperature > 0.0,
                        "pad_token_id": tokenizer.pad_token_id,
                    }
                    if args.temperature > 0.0:
                        generation_kwargs["temperature"] = args.temperature
                    with torch.no_grad():
                        output_ids = model.generate(**encoded, **generation_kwargs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start

                    attention_mask = encoded.get("attention_mask")
                    per_sample_latency = (elapsed * 1000.0) / len(batch)
                    for index, sample in enumerate(batch):
                        prompt_len = int(attention_mask[index].sum().item()) if attention_mask is not None else int(encoded["input_ids"].shape[1])
                        generated = output_ids[index, prompt_len : prompt_len + max_new_tokens]
                        text = tokenizer.decode(generated, skip_special_tokens=True)
                        reference = _reference_text(sample, task.answer_field)
                        score = _metric_value(task.metric, text, reference)
                        gen_count = int(generated.shape[-1])
                        scores.append(score)
                        latencies.append(per_sample_latency)
                        tps_values.append(gen_count / elapsed if elapsed > 0 else 0.0)
                        prompt_tokens.append(prompt_len)
                        generated_tokens.append(gen_count)
                        pred_handle.write(
                            json.dumps(
                                {
                                    "task": task_name,
                                    "sample_id": str(sample.get("id", len(scores))),
                                    "prediction": text,
                                    "reference": reference,
                                    "metric": task.metric,
                                    "metric_value": score,
                                    "latency_ms": per_sample_latency,
                                    "prompt_tokens": prompt_len,
                                    "generated_tokens": gen_count,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                rows.append(
                    BaselineRow(
                        task=task_name,
                        samples=len(samples),
                        metric=task.metric,
                        score=round(mean(scores), 6) if scores else 0.0,
                        latency_ms=round(mean(latencies), 4) if latencies else 0.0,
                        latency_std_ms=round(pstdev(latencies), 4) if len(latencies) > 1 else 0.0,
                        tokens_per_second=round(mean(tps_values), 4) if tps_values else 0.0,
                        prompt_tokens_mean=round(mean(prompt_tokens), 2) if prompt_tokens else 0.0,
                        generated_tokens_mean=round(mean(generated_tokens), 2) if generated_tokens else 0.0,
                        status="ok",
                        error=None,
                    )
                )
            except Exception as exc:
                rows.append(
                    BaselineRow(
                        task=task_name,
                        samples=len(samples),
                        metric=task.metric,
                        score=None,
                        latency_ms=None,
                        latency_std_ms=None,
                        tokens_per_second=None,
                        prompt_tokens_mean=None,
                        generated_tokens_mean=None,
                        status="blocked",
                        error=repr(exc),
                    )
                )
            _write_summary(output_dir / "summary.csv", rows)

    metadata["status"] = "ok" if all(row.status == "ok" for row in rows) else "partial"
    metadata["rows"] = [row.__dict__ for row in rows]
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"status": metadata["status"], "output_dir": str(output_dir), "rows": len(rows)}


def main() -> None:
    print(json.dumps(run(_parse_args()), indent=2))


if __name__ == "__main__":
    main()
