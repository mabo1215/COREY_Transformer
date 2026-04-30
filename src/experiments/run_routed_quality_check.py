"""
Quality-preservation check for passive vs COREY routed Mamba inference.

The check runs the same model and prompts in two modes:
  1. passive stock fast path
  2. active+routed COREY scheduler path

It records greedy-generation token equality and optional prompt perplexity
under both modes. The goal is to produce an auditable quality-preservation row
for the manuscript without mixing it into the latency ablation harness.

Smoke example:
    python -m src.experiments.run_routed_quality_check \
        --model mamba-370m \
        --new-tokens 8 \
        --max-prompt-length 512 \
        --output-dir src/outputs/routed_quality_3090_smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.run_integrated_end_to_end import (
    DEFAULT_PROMPT,
    MODEL_REGISTRY,
    STATE,
    _load,
    _set_hf_token_from_envfile,
    install_patch,
    restore,
)
from src.experiments.run_longbench_inference import (
    LONG_BENCH_TASKS,
    _load_task_samples,
    _metric_value,
    _reference_text,
    _render_prompt,
)


DEFAULT_PROMPTS = [
    {
        "task": "manual",
        "sample_id": "manual-0",
        "prompt": DEFAULT_PROMPT,
        "reference": "",
        "metric_name": None,
    },
    {
        "task": "manual",
        "sample_id": "manual-1",
        "prompt": (
            "Write a concise technical summary of why chunk size can affect "
            "selective-scan kernel latency in long-context state space models."
        ),
        "reference": "",
        "metric_name": None,
    },
    {
        "task": "manual",
        "sample_id": "manual-2",
        "prompt": (
            "Consider a log file with repeated status messages, sparse error "
            "events, and tabular counters. Explain what runtime statistics might "
            "change across those regions."
        ),
        "reference": "",
        "metric_name": None,
    },
]


def _load_prompt_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.longbench_tasks:
        return _load_longbench_records(args)
    if args.prompts_jsonl is None:
        return DEFAULT_PROMPTS[: args.max_prompts]
    records: list[dict[str, Any]] = []
    with args.prompts_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                prompt = obj
                reference = ""
                metric_name = None
            elif isinstance(obj, dict):
                prompt = None
                for key in ("prompt", "text", "context", "input"):
                    value = obj.get(key)
                    if isinstance(value, str) and value.strip():
                        prompt = value
                        break
                reference = str(obj.get("reference", obj.get("answer", "")))
                metric_name = obj.get("metric_name")
            else:
                continue
            if prompt:
                records.append({
                    "task": "jsonl",
                    "sample_id": str(len(records)),
                    "prompt": prompt,
                    "reference": reference,
                    "metric_name": metric_name,
                })
            if len(records) >= args.max_prompts:
                break
    if not records:
        raise ValueError(f"No prompts found in {args.prompts_jsonl}")
    return records


def _load_longbench_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    loader_args = argparse.Namespace(
        dataset_root=args.dataset_root,
        dataset_source=args.dataset_source,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        cache_dir=args.cache_dir,
        hf_token=args.hf_token,
        sample_offset=args.sample_offset,
    )
    for task_name in args.longbench_tasks:
        task = LONG_BENCH_TASKS[task_name]
        samples = _load_task_samples(loader_args, task_name, args.split, args.samples_per_task)
        for sample_index, sample in enumerate(samples):
            reference = _reference_text(sample, task.answer_field)
            prompt = _render_prompt(task, sample)
            records.append({
                "task": task_name,
                "sample_id": str(sample.get("id", sample_index)),
                "prompt": prompt,
                "reference": reference,
                "metric_name": task.metric,
            })
    if not records:
        raise ValueError("No LongBench records were loaded.")
    return records[: args.max_prompts] if args.max_prompts else records


def _configure_state(args: argparse.Namespace) -> None:
    STATE.num_bins = args.num_bins
    STATE.h_ref = None
    STATE.chunk_min = args.chunk_min
    STATE.chunk_max = args.chunk_max
    STATE.force_chunk = args.force_chunk
    STATE.scheduler_mode = args.scheduler_mode
    STATE.entropy_stride = max(args.entropy_stride, 1)
    STATE.random_seed = args.random_seed
    STATE.reset_random()


def _generate(model: Any, tokenizer: Any, prompt: str, device: Any, args: argparse.Namespace) -> dict[str, Any]:
    import torch

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_prompt_length,
    ).to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.new_tokens,
            do_sample=False,
            use_cache=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    full_ids = [int(x) for x in output[0].detach().cpu().tolist()]
    prompt_len = int(inputs["input_ids"].shape[-1])
    new_ids = full_ids[prompt_len:]
    return {
        "generated_ids": full_ids,
        "new_token_ids": new_ids,
        "generated_text": tokenizer.decode(new_ids, skip_special_tokens=True),
    }


def _loss_and_ppl(model: Any, tokenizer: Any, text: str, device: Any, args: argparse.Namespace) -> dict[str, float]:
    import torch

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_prompt_length,
    ).to(device)
    labels = inputs["input_ids"].clone()
    with torch.no_grad():
        out = model(**inputs, labels=labels)
    loss = float(out.loss.detach().float().cpu().item())
    return {"loss": loss, "ppl": float(math.exp(min(loss, 20.0)))}


def _run_mode(
    mode: str,
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    device: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    STATE.enabled = mode == "routed"
    STATE.route_chunk = mode == "routed"
    STATE.records.clear()
    STATE.reset_random()

    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        prompt = str(record["prompt"])
        generated = _generate(model, tokenizer, prompt, device, args)
        reference = str(record.get("reference", ""))
        ppl_text = prompt if not reference else f"{prompt}\n{reference}"
        quality = _loss_and_ppl(model, tokenizer, ppl_text, device, args) if args.include_perplexity else {}
        rows.append({
            "prompt_index": idx,
            "task": record.get("task", "manual"),
            "sample_id": record.get("sample_id", str(idx)),
            "reference": reference,
            "metric_name": record.get("metric_name"),
            **generated,
            **quality,
        })
    chunk_dist: dict[int, int] = {}
    for rec in STATE.records:
        chunk = int(rec["chunk"])
        chunk_dist[chunk] = chunk_dist.get(chunk, 0) + 1
    for row in rows:
        row["chunk_dist_total"] = chunk_dist
    return rows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="mamba-370m", choices=list(MODEL_REGISTRY))
    p.add_argument("--prompts-jsonl", type=Path, default=None)
    p.add_argument("--longbench-tasks", nargs="+", choices=sorted(LONG_BENCH_TASKS), default=None)
    p.add_argument("--dataset-root", type=Path, default=Path("src/data/longbench_subset"))
    p.add_argument("--dataset-source", choices=("auto", "local", "hf"), default="local")
    p.add_argument("--dataset-name", default="zai-org/LongBench")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--samples-per-task", type=int, default=2)
    p.add_argument("--sample-offset", type=int, default=0)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--max-prompts", type=int, default=3)
    p.add_argument("--max-prompt-length", type=int, default=1024)
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--num-bins", type=int, default=256)
    p.add_argument("--chunk-min", type=int, default=128)
    p.add_argument("--chunk-max", type=int, default=512)
    p.add_argument(
        "--scheduler-mode",
        choices=(
            "hist",
            "sampled_hist",
            "token_hist",
            "cheap_proxy",
            "variance_proxy",
            "kurtosis_proxy",
            "constant",
            "no_entropy",
            "random",
        ),
        default="sampled_hist",
    )
    p.add_argument("--entropy-stride", type=int, default=8)
    p.add_argument("--force-chunk", type=int, default=None)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--include-perplexity", action="store_true")
    p.add_argument(
        "--selective-scan-dispatch-module",
        default=None,
        help="Dispatch module exposing selective_scan_fn(..., chunk_size=...).",
    )
    p.add_argument("--dry-run", action="store_true", help="Load prompts and write a manifest without requiring CUDA.")
    p.add_argument("--output-dir", type=Path, default=Path("src/outputs/routed_quality_check"))
    return p.parse_args()


def _metric(metric_name: str | None, prediction: str, reference: str) -> float | None:
    if not metric_name or not reference:
        return None
    return float(_metric_value(metric_name, prediction, reference))


def _summarize_by_task(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = sorted({str(row["task"]) for row in comparisons})
    summary: list[dict[str, Any]] = []
    for task in tasks:
        rows = [row for row in comparisons if row["task"] == task]
        passive_metrics = [row["passive_metric"] for row in rows if row.get("passive_metric") is not None]
        routed_metrics = [row["routed_metric"] for row in rows if row.get("routed_metric") is not None]
        ppl_ratios = [row["ppl_ratio"] for row in rows if row.get("ppl_ratio") is not None]
        summary.append({
            "task": task,
            "samples": len(rows),
            "exact_generation_matches": sum(1 for row in rows if row["generated_token_exact_match"]),
            "token_mismatches": sum(int(row["token_mismatches"]) for row in rows),
            "metric_name": rows[0].get("metric_name"),
            "passive_metric_mean": mean(passive_metrics) if passive_metrics else None,
            "routed_metric_mean": mean(routed_metrics) if routed_metrics else None,
            "metric_delta_mean": (
                mean(routed_metrics) - mean(passive_metrics)
                if passive_metrics and routed_metrics and len(passive_metrics) == len(routed_metrics)
                else None
            ),
            "ppl_ratio_mean": mean(ppl_ratios) if ppl_ratios else None,
        })
    return summary


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _set_hf_token_from_envfile()
    if args.selective_scan_dispatch_module:
        os.environ["COREY_SELECTIVE_SCAN_DISPATCH_MODULE"] = args.selective_scan_dispatch_module

    records = _load_prompt_records(args)
    manifest = [
        {
            "prompt_index": idx,
            "task": record.get("task"),
            "sample_id": record.get("sample_id"),
            "metric_name": record.get("metric_name"),
            "reference_chars": len(str(record.get("reference", ""))),
            "prompt_chars": len(str(record.get("prompt", ""))),
        }
        for idx, record in enumerate(records)
    ]
    (args.output_dir / "prompt_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.dry_run:
        print(f"[quality] Dry run loaded {len(records)} prompts; manifest saved to {args.output_dir / 'prompt_manifest.json'}")
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the routed Mamba quality check.")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[quality] Device: {gpu_name}")

    _configure_state(args)
    model, tokenizer = _load(args.model, device)
    original = install_patch()
    try:
        passive_rows = _run_mode("passive", model, tokenizer, records, device, args)
        routed_rows = _run_mode("routed", model, tokenizer, records, device, args)
    finally:
        STATE.enabled = False
        restore(original)

    comparisons: list[dict[str, Any]] = []
    for passive, routed in zip(passive_rows, routed_rows):
        passive_ids = passive["new_token_ids"]
        routed_ids = routed["new_token_ids"]
        min_len = min(len(passive_ids), len(routed_ids))
        token_mismatches = sum(
            1 for i in range(min_len) if passive_ids[i] != routed_ids[i]
        ) + abs(len(passive_ids) - len(routed_ids))
        row = {
            "prompt_index": passive["prompt_index"],
            "task": passive["task"],
            "sample_id": passive["sample_id"],
            "metric_name": passive.get("metric_name"),
            "generated_token_exact_match": passive["new_token_ids"] == routed["new_token_ids"],
            "token_mismatches": token_mismatches,
            "passive_tokens": len(passive["new_token_ids"]),
            "routed_tokens": len(routed["new_token_ids"]),
            "passive_metric": _metric(passive.get("metric_name"), passive["generated_text"], passive["reference"]),
            "routed_metric": _metric(routed.get("metric_name"), routed["generated_text"], routed["reference"]),
            "metric_delta": None,
            "routed_chunk_dist": routed.get("chunk_dist_total", {}),
        }
        if row["passive_metric"] is not None and row["routed_metric"] is not None:
            row["metric_delta"] = row["routed_metric"] - row["passive_metric"]
        if args.include_perplexity:
            row.update({
                "passive_loss": passive["loss"],
                "routed_loss": routed["loss"],
                "loss_delta": routed["loss"] - passive["loss"],
                "passive_ppl": passive["ppl"],
                "routed_ppl": routed["ppl"],
                "ppl_ratio": routed["ppl"] / passive["ppl"] if passive["ppl"] else None,
            })
        comparisons.append(row)

    exact_matches = sum(1 for row in comparisons if row["generated_token_exact_match"])
    output = {
        "gpu": gpu_name,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "model": args.model,
        "max_prompt_length": args.max_prompt_length,
        "new_tokens": args.new_tokens,
        "scheduler_mode": args.scheduler_mode,
        "entropy_stride": max(args.entropy_stride, 1),
        "force_chunk": args.force_chunk,
        "random_seed": args.random_seed,
        "dispatch_module": args.selective_scan_dispatch_module,
        "dataset_root": str(args.dataset_root) if args.longbench_tasks else None,
        "longbench_tasks": args.longbench_tasks,
        "samples_per_task": args.samples_per_task if args.longbench_tasks else None,
        "num_prompts": len(records),
        "exact_generation_matches": exact_matches,
        "all_generation_exact_match": exact_matches == len(comparisons),
        "task_summary": _summarize_by_task(comparisons),
        "comparisons": comparisons,
    }
    out_path = args.output_dir / "summary.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    lines = [
        "# Routed Quality Check",
        "",
        "| task | sample | exact generated tokens | mismatches | passive metric | routed metric | metric delta | passive ppl | routed ppl | ppl ratio | routed chunks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparisons:
        lines.append(
            "| {task} | {idx} | {exact} | {mismatch} | {pm} | {rm} | {md} | {pp} | {rp} | {ratio} | `{chunks}` |".format(
                task=row["task"],
                idx=row["sample_id"],
                exact=row["generated_token_exact_match"],
                mismatch=row["token_mismatches"],
                pm=f"{row['passive_metric']:.6f}" if row.get("passive_metric") is not None else "n/a",
                rm=f"{row['routed_metric']:.6f}" if row.get("routed_metric") is not None else "n/a",
                md=f"{row['metric_delta']:+.6f}" if row.get("metric_delta") is not None else "n/a",
                pp=f"{row['passive_ppl']:.4f}" if "passive_ppl" in row else "n/a",
                rp=f"{row['routed_ppl']:.4f}" if "routed_ppl" in row else "n/a",
                ratio=f"{row['ppl_ratio']:.6f}x" if row.get("ppl_ratio") is not None else "n/a",
                chunks=row["routed_chunk_dist"],
            )
        )
    lines.extend(["", "## Task Summary", "", "| task | samples | exact matches | mismatches | passive metric | routed metric | delta | mean ppl ratio |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in output["task_summary"]:
        lines.append(
            "| {task} | {samples} | {exact} | {mismatch} | {pm} | {rm} | {delta} | {ppl} |".format(
                task=row["task"],
                samples=row["samples"],
                exact=row["exact_generation_matches"],
                mismatch=row["token_mismatches"],
                pm=f"{row['passive_metric_mean']:.6f}" if row.get("passive_metric_mean") is not None else "n/a",
                rm=f"{row['routed_metric_mean']:.6f}" if row.get("routed_metric_mean") is not None else "n/a",
                delta=f"{row['metric_delta_mean']:+.6f}" if row.get("metric_delta_mean") is not None else "n/a",
                ppl=f"{row['ppl_ratio_mean']:.6f}x" if row.get("ppl_ratio_mean") is not None else "n/a",
            )
        )
    (args.output_dir / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[quality] Summary saved to {out_path}")
    print(f"[quality] Exact generation matches: {exact_matches}/{len(comparisons)}")


if __name__ == "__main__":
    main()
