"""Mixed-regime discovery for the post-H800 scheduler pivot.

The goal is to stop repeating homogeneous scheduler matrices and first ask
whether a realistic serving mix contains regimes that could prefer different
static chunks.  The script has two modes:

1. ``prompt_pool`` builds a JSONL candidate serving trace without requiring GPU.
2. ``feature_scan`` runs the existing inline hook without routing chunks, records
   selected chunks/features by regime, and writes a stop/go summary for H800.

Example no-GPU prompt-pool build:
    python -m src.experiments.run_mixed_regime_discovery --mode prompt_pool

Example CUDA feature scan:
    python -m src.experiments.run_mixed_regime_discovery \\
        --mode feature_scan \\
        --prompt-pool src/outputs/mixed_regime_discovery/prompt_pool.jsonl \\
        --model mamba-370m --new-tokens 1 --warmup 0 --repeats 1
"""

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prompt_pool", "feature_scan"), default="prompt_pool")
    parser.add_argument("--model", default="mamba-370m", choices=list(MODEL_REGISTRY))
    parser.add_argument("--output-dir", type=Path, default=Path("src/outputs/mixed_regime_discovery"))
    parser.add_argument("--prompt-pool", type=Path)
    parser.add_argument("--samples-per-regime", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--new-tokens", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--num-bins", type=int, default=256)
    parser.add_argument("--scheduler-modes", default="sampled_hist,variance_proxy,kurtosis_proxy,token_hist")
    parser.add_argument("--entropy-stride", type=int, default=8)
    return parser.parse_args()


def _repeat_to_words(text: str, target_words: int) -> str:
    words = text.split()
    if not words:
        return text
    out: list[str] = []
    while len(out) < target_words:
        out.extend(words)
    return " ".join(out[:target_words])


def _prompt_records(samples_per_regime: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    def add(regime: str, prompt: str, index: int) -> None:
        records.append({
            "id": f"{regime.replace('/', '_')}:{index:03d}",
            "regime": regime,
            "prompt": prompt,
            "chars": len(prompt),
        })

    for i in range(samples_per_regime):
        add(
            "short_chat",
            (
                "User asks a concise assistant question about project status. "
                f"Variant {i}: summarize the key blocker and next action in two sentences."
            ),
            i,
        )

        doc = _repeat_to_words(
            "Quarterly planning memo. The deployment team tracks latency, quality, "
            "incident risk, launch criteria, owner handoff, and rollback plans.",
            900 + 20 * (i % 5),
        )
        add("long_doc_qa", f"Read the document and answer the final question.\n{doc}\nQuestion: What should be escalated?", i)

        code = "\n".join(
            [
                "def route_event(event, state):",
                "    key = (event.customer_id, event.region, event.service)",
                "    bucket = state.setdefault(key, [])",
                "    if event.timestamp >= state.watermark(key):",
                "        bucket.append((event.timestamp, event.payload_hash))",
                "    return sorted(bucket)[-32:]",
            ]
            * (24 + i % 4)
        )
        add("code_repo", f"Explain this code path and identify edge cases.\n{code}\nAnalysis:", i)

        log_lines = []
        for j in range(180 + i % 20):
            uuid = f"{rng.getrandbits(128):032x}"
            log_lines.append(
                f"2026-04-30T12:{j % 60:02d}:00Z service=router shard={j % 8} "
                f"status={200 if j % 17 else 503} trace={uuid} latency_ms={18 + (j * 7) % 311}"
            )
        add("logs_uuid", "Find anomalies in this production log stream.\n" + "\n".join(log_lines), i)

        rows = [
            f"{j},acct_{j % 31:03d},region_{j % 5},score={(j * 13 + i) % 997},flag={j % 2}"
            for j in range(220 + i % 30)
        ]
        add("tables_csv", "Summarize outliers in this CSV extract.\nidx,account,region,score,flag\n" + "\n".join(rows), i)

        repeated = _repeat_to_words(
            "Policy clause 7.3 requires reviewer approval before escalation. "
            "The request must include risk tier, owner, deadline, and rollback.",
            1000 + 10 * (i % 7),
        )
        add("repetition_policy", f"Compress the following repeated policy text.\n{repeated}", i)

        mixed = _repeat_to_words(
            "请用中文总结这段 English incident report, then list action items in English. "
            "系统延迟升高但质量指标稳定，需要检查 routing, cache, and batch scheduling.",
            650 + 15 * (i % 5),
        )
        add("mixed_zh_en", mixed, i)

        form_rows = "\n".join(
            f"FIELD_{j:03d}: value={rng.choice(['approved', 'pending', 'missing', 'n/a'])}; confidence={(j * 19) % 100}%"
            for j in range(180 + i % 15)
        )
        add("ocr_forms", "Extract missing fields from this OCR-like form dump.\n" + form_rows, i)

    return records


def _write_prompt_pool(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "prompt_pool.jsonl"
    rows = _prompt_records(args.samples_per_regime, args.seed)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    manifest = {
        "mode": "prompt_pool",
        "records": len(rows),
        "regimes": sorted({row["regime"] for row in rows}),
        "samples_per_regime": args.samples_per_regime,
        "path": str(path),
    }
    (args.output_dir / "prompt_pool_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return path


def _load_prompt_pool(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summarize(rows: list[dict[str, Any]], num_bins: int) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    for mode in sorted({row["scheduler_mode"] for row in rows}):
        for regime in sorted({row["regime"] for row in rows if row["scheduler_mode"] == mode}):
            subset = [row for row in rows if row["scheduler_mode"] == mode and row["regime"] == regime]
            chunks: dict[int, int] = {}
            entropies: list[float] = []
            prompt_lens = [int(row["prompt_len"]) for row in subset]
            for row in subset:
                for rec in row["scheduler_records"]:
                    chunk = int(rec["chunk"])
                    chunks[chunk] = chunks.get(chunk, 0) + 1
                    if rec.get("entropy_nats") is not None:
                        entropies.append(float(rec["entropy_nats"]))
            total = sum(chunks.values())
            dominant = max(chunks.values()) / total if total else 0.0
            summary_rows.append({
                "scheduler_mode": mode,
                "regime": regime,
                "prompts": len(subset),
                "prompt_len_mean": round(statistics.mean(prompt_lens), 2) if prompt_lens else None,
                "entropy_mean": round(statistics.mean(entropies), 6) if entropies else None,
                "entropy_std": round(statistics.pstdev(entropies), 6) if len(entropies) > 1 else 0.0,
                "chunk_counts": dict(sorted(chunks.items())),
                "dominant_chunk_share": round(dominant, 4),
                "feature_diverse": len(chunks) > 1 and dominant < 0.95,
            })
    diverse = [row for row in summary_rows if row["feature_diverse"]]
    return {
        "benchmark": "mixed_regime_discovery",
        "status": "ok",
        "platform": platform.platform(),
        "h_ref": f"log({num_bins})={math.log(num_bins):.4f}",
        "summary_rows": summary_rows,
        "go_for_static_sweep": len(diverse) >= 3,
        "go_reason": (
            "at least three regime/mode rows show non-degenerate chunk diversity"
            if len(diverse) >= 3
            else "feature scan did not find three diverse regime/mode rows; do not open H800 yet"
        ),
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Mixed-Regime Discovery Summary",
        "",
        f"Decision: **{'GO' if summary['go_for_static_sweep'] else 'STOP'}**",
        "",
        summary["go_reason"],
        "",
        "| scheduler | regime | prompts | prompt len | entropy mean | entropy std | chunk counts | diverse |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in summary["summary_rows"]:
        lines.append(
            "| {scheduler_mode} | {regime} | {prompts} | {prompt_len_mean} | "
            "{entropy_mean} | {entropy_std} | `{chunk_counts}` | {feature_diverse} |".format(**row)
        )
    lines.extend([
        "",
        "H800 gate: only run static chunk sweeps when feature diversity suggests at least three useful regimes.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_feature_scan(args: argparse.Namespace) -> None:
    import torch

    pool = args.prompt_pool or (args.output_dir / "prompt_pool.jsonl")
    if not pool.exists():
        pool = _write_prompt_pool(args)
    prompts = _load_prompt_pool(pool)

    if not torch.cuda.is_available():
        raise RuntimeError("feature_scan requires CUDA. Use --mode prompt_pool for the no-GPU stage.")

    device = torch.device("cuda")
    model, tokenizer = _load(args.model, device)
    modes = [part.strip() for part in args.scheduler_modes.split(",") if part.strip()]

    STATE.num_bins = args.num_bins
    STATE.h_ref = None
    STATE.entropy_stride = max(args.entropy_stride, 1)
    original = install_patch()
    rows: list[dict[str, Any]] = []
    try:
        for mode in modes:
            STATE.scheduler_mode = mode
            STATE.enabled = True
            STATE.route_chunk = False
            for idx, item in enumerate(prompts):
                STATE.records.clear()
                timing = _time_generate(
                    model,
                    tokenizer,
                    item["prompt"],
                    args.new_tokens,
                    device,
                    args.warmup if idx == 0 else 0,
                    args.repeats,
                    args.max_prompt_length,
                )
                rows.append({
                    "id": item["id"],
                    "regime": item["regime"],
                    "scheduler_mode": mode,
                    "prompt_chars": item["chars"],
                    "prompt_len": timing["prompt_len"],
                    "lat_mean_ms": timing["lat_mean_ms"],
                    "scheduler_records": list(STATE.records),
                })
                print(f"[mixed] {mode} {idx + 1}/{len(prompts)} {item['regime']}")
    finally:
        STATE.enabled = False
        restore(original)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "feature_records.jsonl"
    records_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    summary = _summarize(rows, args.num_bins)
    summary["gpu"] = torch.cuda.get_device_name(0)
    summary["torch"] = torch.__version__
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(summary, args.output_dir / "summary_table.md")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = _parse_args()
    if args.mode == "prompt_pool":
        _write_prompt_pool(args)
    else:
        _run_feature_scan(args)


if __name__ == "__main__":
    main()
