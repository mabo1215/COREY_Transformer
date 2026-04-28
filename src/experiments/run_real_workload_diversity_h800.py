from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
from pathlib import Path
from typing import Any

from src.experiments.run_integrated_end_to_end import (
    MODEL_REGISTRY,
    STATE,
    _load,
    _time_generate,
    install_patch,
    restore,
)
from src.experiments.run_longbench_inference import LONG_BENCH_TASKS, _load_task_samples, _render_prompt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="H800 real-workload entropy diversity run with LongBench plus realistic structured regimes."
    )
    parser.add_argument("--model", default="mamba-370m", choices=list(MODEL_REGISTRY))
    parser.add_argument("--dataset-root", type=Path, default=Path("src/data/longbench_subset"))
    parser.add_argument("--dataset-source", choices=["auto", "local", "hf"], default="auto")
    parser.add_argument("--dataset-name", default="zai-org/LongBench")
    parser.add_argument("--dataset-config")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--hf-token")
    parser.add_argument("--tasks", nargs="+", default=["narrativeqa", "qasper", "gov_report", "multifieldqa_en"])
    parser.add_argument("--samples-per-task", type=int, default=20)
    parser.add_argument("--extra-regime-file", type=Path, action="append", default=[])
    parser.add_argument("--max-extra-per-regime", type=int, default=20)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--num-bins", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, default=Path("src/outputs/h800_real_workload_diversity"))
    return parser.parse_args()


def _load_longbench_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    for task_name in args.tasks:
        task = LONG_BENCH_TASKS[task_name]
        task_args = argparse.Namespace(**vars(args))
        task_args.sample_offset = 0
        samples = _load_task_samples(task_args, task_name, "test", args.samples_per_task)
        for sample in samples:
            prompts.append({"regime": f"longbench:{task_name}", "prompt": _render_prompt(task, sample)})
    return prompts


def _structured_prompts() -> list[dict[str, str]]:
    log_block = "\n".join(
        f"2026-04-28T12:{minute:02d}:00Z service=router shard={minute % 8} status=200 latency_ms={18 + minute % 13}"
        for minute in range(160)
    )
    table_block = "\n".join(
        f"row={idx}, account=A{idx % 17:03d}, region={['us','eu','apac'][idx % 3]}, value={idx * 7 % 991}, flag={idx % 2}"
        for idx in range(180)
    )
    code_block = "\n".join(
        [
            "def reconcile_events(events, watermarks):",
            "    pending = {}",
            "    for event in events:",
            "        bucket = event.customer_id, event.region",
            "        if bucket not in pending:",
            "            pending[bucket] = []",
            "        if event.timestamp <= watermarks.get(bucket, 0):",
            "            pending[bucket].append(event)",
            "    return {key: sorted(value, key=lambda item: item.timestamp) for key, value in pending.items()}",
        ]
        * 18
    )
    repeated_policy = ("Policy clause 7.3 requires reviewer approval before escalation. " * 220).strip()
    return [
        {"regime": "structured:logs", "prompt": f"Summarize anomalies in this log stream:\n{log_block}\nSummary:"},
        {"regime": "structured:tables", "prompt": f"Find unusual rows in this CSV-like table:\n{table_block}\nFindings:"},
        {"regime": "structured:code", "prompt": f"Explain the behavior and complexity of this code:\n{code_block}\nExplanation:"},
        {"regime": "structured:repetition", "prompt": f"Compress the repeated policy text:\n{repeated_policy}\nCompressed:"},
    ]


def _load_extra_regimes(args: argparse.Namespace) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    for path in args.extra_regime_file:
        regime = path.stem
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if count >= args.max_extra_per_regime:
                    break
                loaded = json.loads(line)
                text = loaded.get("prompt") or loaded.get("context") or loaded.get("text") or loaded.get("input")
                if not text:
                    continue
                prompts.append({"regime": f"extra:{regime}", "prompt": str(text)})
                count += 1
    return prompts


def _summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    regimes = sorted({row["regime"] for row in records})
    summary: list[dict[str, Any]] = []
    for regime in regimes:
        subset = [row for row in records if row["regime"] == regime]
        entropies = [rec["entropy_nats"] for row in subset for rec in row["scheduler_records"]]
        chunks: dict[int, int] = {}
        for row in subset:
            for rec in row["scheduler_records"]:
                chunk = int(rec["chunk"])
                chunks[chunk] = chunks.get(chunk, 0) + 1
        summary.append(
            {
                "regime": regime,
                "prompts": len(subset),
                "entropy_mean": round(statistics.mean(entropies), 6) if entropies else None,
                "entropy_std": round(statistics.pstdev(entropies), 6) if len(entropies) > 1 else 0.0,
                "entropy_min": round(min(entropies), 6) if entropies else None,
                "entropy_max": round(max(entropies), 6) if entropies else None,
                "chunk_counts": dict(sorted(chunks.items())),
                "non_degenerate": len(chunks) > 1,
            }
        )
    return summary


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/H800 is required for this run.")
    device = torch.device("cuda")
    model, tokenizer = _load(args.model, device)

    prompts = _load_longbench_prompts(args) + _structured_prompts() + _load_extra_regimes(args)
    random.shuffle(prompts)

    STATE.num_bins = args.num_bins
    STATE.h_ref = None
    original = install_patch()
    rows: list[dict[str, Any]] = []
    try:
        STATE.enabled = True
        STATE.route_chunk = False
        for index, item in enumerate(prompts):
            STATE.records.clear()
            timing = _time_generate(
                model,
                tokenizer,
                item["prompt"],
                args.new_tokens,
                device,
                args.warmup if index == 0 else 0,
                args.repeats,
            )
            rows.append(
                {
                    "index": index,
                    "regime": item["regime"],
                    "prompt_chars": len(item["prompt"]),
                    "prompt_len": timing["prompt_len"],
                    "lat_mean_ms": timing["lat_mean_ms"],
                    "scheduler_records": list(STATE.records),
                }
            )
            print(f"[diversity] {index + 1}/{len(prompts)} {item['regime']} chunks={len(STATE.records)}")
    finally:
        STATE.enabled = False
        restore(original)

    summary = {
        "benchmark": "h800_real_workload_diversity",
        "status": "ok",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "h_ref": f"log({args.num_bins})={math.log(args.num_bins):.4f}",
        "prompts": len(rows),
        "regime_summary": _summarize(rows),
    }
    (args.output_dir / "records.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
