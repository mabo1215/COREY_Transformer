"""
Unified scheduler ablation matrix for routed Mamba end-to-end timing.

The script is intentionally orchestration-only. Each row calls
run_integrated_end_to_end with matched prompt/model settings, then aggregates
the resulting active+routed latency into CSV, Markdown, and JSON summaries.

Typical smoke run on a free RTX 3090 box:
    python -m src.experiments.run_scheduler_ablation_matrix \
        --model mamba-370m \
        --prompt-repeat 1 \
        --max-prompt-length 1024 \
        --new-tokens 2 \
        --warmup 0 \
        --repeats 1 \
        --rows static,no_entropy,random,sampled_hist \
        --sweep-chunks 128,256 \
        --output-dir src/outputs/scheduler_ablation_3090_smoke

Full H800 run, after the patched runtime-chunk CUDA extension is available:
    python -m src.experiments.run_scheduler_ablation_matrix \
        --model mamba-370m \
        --prompt-repeat 8 \
        --max-prompt-length 4096 \
        --new-tokens 32 \
        --warmup 3 \
        --repeats 50 \
        --sweep-chunks 128,256,512,1024,2048 \
        --selective-scan-dispatch-module src.corey_selective_scan_dispatch \
        --output-dir src/outputs/scheduler_ablation_h800
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.run_integrated_end_to_end import DEFAULT_PROMPT, MODEL_REGISTRY


VALID_ROWS = (
    "static",
    "no_entropy",
    "random",
    "hist",
    "sampled_hist",
    "token_hist",
    "cheap_proxy",
    "variance_proxy",
    "kurtosis_proxy",
    "guarded_sampled_hist",
    "guarded_variance_proxy",
    "learned_table",
)


@dataclass(frozen=True)
class RowSpec:
    name: str
    kind: str
    scheduler_mode: str
    force_chunk: int | None = None
    entropy_stride: int = 1
    chunk_min: int | None = None
    chunk_max: int | None = None
    guard_fallback_chunk: int | None = None
    guard_min_delta_buckets: int | None = None


def _parse_chunks(raw: str) -> list[int]:
    chunks: list[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        value = int(text)
        if value <= 0:
            raise argparse.ArgumentTypeError("chunk sizes must be positive")
        chunks.append(value)
    if not chunks:
        raise argparse.ArgumentTypeError("at least one chunk size is required")
    return sorted(dict.fromkeys(chunks))


def _parse_rows(raw: str) -> list[str]:
    rows: list[str] = []
    for part in raw.split(","):
        row = part.strip()
        if not row:
            continue
        if row not in VALID_ROWS:
            raise argparse.ArgumentTypeError(
                f"unknown row {row!r}; expected one of {', '.join(VALID_ROWS)}"
            )
        rows.append(row)
    if not rows:
        raise argparse.ArgumentTypeError("at least one row is required")
    return list(dict.fromkeys(rows))


def _add_common_child_args(cmd: list[str], args: argparse.Namespace) -> None:
    cmd.extend([
        "--model", args.model,
        "--prompt", args.prompt,
        "--prompt-repeat", str(args.prompt_repeat),
        "--max-prompt-length", str(args.max_prompt_length),
        "--new-tokens", str(args.new_tokens),
        "--warmup", str(args.warmup),
        "--repeats", str(args.repeats),
        "--num-bins", str(args.num_bins),
        "--random-seed", str(args.random_seed),
    ])
    if args.selective_scan_dispatch_module:
        cmd.extend([
            "--selective-scan-dispatch-module",
            args.selective_scan_dispatch_module,
        ])


def _build_specs(args: argparse.Namespace) -> list[RowSpec]:
    chunks = list(args.sweep_chunks)
    adaptive_min = args.adaptive_chunk_min if args.adaptive_chunk_min is not None else min(chunks)
    adaptive_max = args.adaptive_chunk_max if args.adaptive_chunk_max is not None else max(chunks)
    specs: list[RowSpec] = []
    for row in args.rows:
        if row == "static":
            specs.extend(
                RowSpec(
                    name=f"static_chunk_{chunk}",
                    kind="static",
                    scheduler_mode="constant",
                    force_chunk=chunk,
                    chunk_min=chunk,
                    chunk_max=chunk,
                )
                for chunk in chunks
            )
        elif row == "no_entropy":
            specs.append(
                RowSpec(
                    name="no_entropy_mid",
                    kind="no_entropy",
                    scheduler_mode="no_entropy",
                    chunk_min=adaptive_min,
                    chunk_max=adaptive_max,
                )
            )
        elif row == "random":
            specs.append(
                RowSpec(
                    name=f"random_seed_{args.random_seed}",
                    kind="random",
                    scheduler_mode="random",
                    chunk_min=adaptive_min,
                    chunk_max=adaptive_max,
                )
            )
        elif row in {"sampled_hist", "token_hist"}:
            specs.append(
                RowSpec(
                    name=f"{row}_s{max(args.entropy_stride, 1)}",
                    kind="adaptive",
                    scheduler_mode=row,
                    entropy_stride=max(args.entropy_stride, 1),
                    chunk_min=adaptive_min,
                    chunk_max=adaptive_max,
                )
            )
        elif row == "guarded_sampled_hist":
            specs.append(
                RowSpec(
                    name=f"guarded_sampled_hist_s{max(args.entropy_stride, 1)}",
                    kind="guarded",
                    scheduler_mode="guarded_sampled_hist",
                    entropy_stride=max(args.entropy_stride, 1),
                    chunk_min=adaptive_min,
                    chunk_max=adaptive_max,
                    guard_fallback_chunk=args.guard_fallback_chunk,
                    guard_min_delta_buckets=args.guard_min_delta_buckets,
                )
            )
        elif row == "guarded_variance_proxy":
            specs.append(
                RowSpec(
                    name="guarded_variance_proxy",
                    kind="guarded",
                    scheduler_mode="guarded_variance_proxy",
                    chunk_min=adaptive_min,
                    chunk_max=adaptive_max,
                    guard_fallback_chunk=args.guard_fallback_chunk,
                    guard_min_delta_buckets=args.guard_min_delta_buckets,
                )
            )
        elif row == "learned_table":
            specs.append(
                RowSpec(
                    name="learned_table",
                    kind="learned",
                    scheduler_mode="learned_table",
                    entropy_stride=max(args.entropy_stride, 1),
                    chunk_min=adaptive_min,
                    chunk_max=adaptive_max,
                )
            )
        else:
            specs.append(
                RowSpec(
                    name=row,
                    kind="adaptive" if row in {"hist", "variance_proxy", "kurtosis_proxy"} else "proxy",
                    scheduler_mode=row,
                    chunk_min=adaptive_min,
                    chunk_max=adaptive_max,
                )
            )
    return specs


def _run_child(
    spec: RowSpec,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any] | None:
    summary_path = output_dir / "summary.json"
    log_path = output_dir / "run.log"

    if args.rerun and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if summary_path.exists() and not args.rerun:
        print(f"[ablation] Reusing {spec.name}: {summary_path}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    child_args: list[str] = []
    _add_common_child_args(child_args, args)
    child_args.extend([
        "--scheduler-mode", spec.scheduler_mode,
        "--entropy-stride", str(spec.entropy_stride),
    ])
    if spec.force_chunk is not None:
        child_args.extend(["--force-chunk", str(spec.force_chunk)])
    if spec.chunk_min is not None:
        child_args.extend(["--chunk-min", str(spec.chunk_min)])
    if spec.chunk_max is not None:
        child_args.extend(["--chunk-max", str(spec.chunk_max)])
    if spec.guard_fallback_chunk is not None:
        child_args.extend(["--guard-fallback-chunk", str(spec.guard_fallback_chunk)])
    if spec.guard_min_delta_buckets is not None:
        child_args.extend(["--guard-min-delta-buckets", str(spec.guard_min_delta_buckets)])
    if args.learned_policy_json and spec.scheduler_mode == "learned_table":
        child_args.extend(["--learned-policy-json", str(args.learned_policy_json)])

    cmd = [
        sys.executable,
        "-m",
        "src.experiments.run_integrated_end_to_end",
        *child_args,
        "--output-dir",
        str(output_dir),
    ]

    print(f"[ablation] Running {spec.name}")
    print(f"[ablation] Command: {' '.join(cmd)}")
    if args.dry_run:
        return None

    env = os.environ.copy()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    with log_path.open("w", encoding="utf-8") as log:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{spec.name} failed with exit code {rc}; see {log_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"{spec.name} did not produce {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _row_from_summary(spec: RowSpec, summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    integrated = summary["integrated"]
    active_only = summary["active_only"]
    passive = summary["passive"]
    lat = float(integrated["lat_mean_ms"])
    passive_lat = float(passive["lat_mean_ms"])
    active_lat = float(active_only["lat_mean_ms"])
    return {
        "name": spec.name,
        "kind": spec.kind,
        "scheduler_mode": summary.get("scheduler_mode"),
        "entropy_stride": summary.get("entropy_stride"),
        "force_chunk": summary.get("force_chunk"),
        "lat_mean_ms": lat,
        "lat_std_ms": float(integrated["lat_std_ms"]),
        "active_only_lat_mean_ms": active_lat,
        "passive_lat_mean_ms": passive_lat,
        "vs_passive_ratio": lat / passive_lat if passive_lat else None,
        "active_overhead_ratio": active_lat / passive_lat if passive_lat else None,
        "chunk_dist": integrated.get("chunk_dist", {}),
        "eligible_for_w1_speedup": summary.get("eligible_for_w1_speedup"),
        "dispatch_info": summary.get("dispatch_info"),
        "summary_path": str(summary_path),
    }


def _format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}x"


def _write_tables(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    csv_path = output_dir / "summary_table.csv"
    md_path = output_dir / "summary_table.md"
    fields = [
        "name",
        "kind",
        "scheduler_mode",
        "entropy_stride",
        "force_chunk",
        "lat_mean_ms",
        "lat_std_ms",
        "vs_passive_ratio",
        "vs_best_static_ratio",
        "speedup_over_best_static",
        "active_overhead_ratio",
        "chunk_dist",
        "eligible_for_w1_speedup",
        "summary_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    lines = [
        "# Scheduler Ablation Matrix",
        "",
        "| configuration | kind | scheduler | latency ms | vs passive | vs best static | speedup over best static | active overhead | chunk distribution |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        sched = str(row.get("scheduler_mode"))
        if row.get("entropy_stride") not in (None, 1):
            sched += f" stride={row['entropy_stride']}"
        if row.get("force_chunk") is not None:
            sched += f" chunk={row['force_chunk']}"
        lines.append(
            "| {name} | {kind} | {sched} | {lat:.2f} +/- {std:.2f} | "
            "{vp} | {vbs} | {sbs} | {aor} | `{dist}` |".format(
                name=row["name"],
                kind=row["kind"],
                sched=sched,
                lat=row["lat_mean_ms"],
                std=row["lat_std_ms"],
                vp=_format_ratio(row.get("vs_passive_ratio")),
                vbs=_format_ratio(row.get("vs_best_static_ratio")),
                sbs=_format_ratio(row.get("speedup_over_best_static")),
                aor=_format_ratio(row.get("active_overhead_ratio")),
                dist=row.get("chunk_dist", {}),
            )
        )
    best = summary.get("best_static_oracle")
    if best is not None:
        lines.extend([
            "",
            f"Best static oracle: `{best['name']}` at {best['lat_mean_ms']:.2f} ms.",
        ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="mamba-370m", choices=list(MODEL_REGISTRY))
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--prompt-repeat", type=int, default=1)
    p.add_argument("--max-prompt-length", type=int, default=1024)
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--num-bins", type=int, default=256)
    p.add_argument("--sweep-chunks", type=_parse_chunks, default=_parse_chunks("128,256,512,1024,2048"))
    p.add_argument(
        "--rows",
        type=_parse_rows,
        default=_parse_rows("static,no_entropy,random,hist,sampled_hist,cheap_proxy,variance_proxy,kurtosis_proxy,token_hist"),
        help=f"Comma-separated subset of: {', '.join(VALID_ROWS)}",
    )
    p.add_argument("--entropy-stride", type=int, default=8)
    p.add_argument("--adaptive-chunk-min", type=int, default=None)
    p.add_argument("--adaptive-chunk-max", type=int, default=None)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--guard-fallback-chunk", type=int, default=512)
    p.add_argument("--guard-min-delta-buckets", type=int, default=2)
    p.add_argument("--learned-policy-json", type=Path, default=None)
    p.add_argument(
        "--selective-scan-dispatch-module",
        default=None,
        help="Dispatch module exposing selective_scan_fn(..., chunk_size=...).",
    )
    p.add_argument("--output-dir", type=Path, default=Path("src/outputs/scheduler_ablation_matrix"))
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = _build_specs(args)
    print("[ablation] Plan: unified scheduler ablation matrix")
    print(f"[ablation] Rows: {[spec.name for spec in specs]}")

    rows: list[dict[str, Any]] = []
    child_summaries: dict[str, dict[str, Any]] = {}
    for spec in specs:
        out_dir = args.output_dir / spec.name
        result = _run_child(spec, args, out_dir)
        if result is None:
            continue
        summary_path = out_dir / "summary.json"
        child_summaries[spec.name] = result
        rows.append(_row_from_summary(spec, result, summary_path))

    if args.dry_run:
        print("[ablation] Dry run complete; no summary table written.")
        return
    if not rows:
        raise RuntimeError("No rows were produced.")

    static_rows = [row for row in rows if row["kind"] == "static"]
    best_static = min(static_rows, key=lambda row: row["lat_mean_ms"]) if static_rows else None
    best_lat = float(best_static["lat_mean_ms"]) if best_static is not None else None
    for row in rows:
        lat = float(row["lat_mean_ms"])
        row["vs_best_static_ratio"] = lat / best_lat if best_lat else None
        row["speedup_over_best_static"] = best_lat / lat if best_lat else None

    summary = {
        "plan": "unified_scheduler_ablation_matrix",
        "output_dir": str(args.output_dir),
        "rows_requested": list(args.rows),
        "sweep_chunks": list(args.sweep_chunks),
        "entropy_stride": max(args.entropy_stride, 1),
        "random_seed": args.random_seed,
        "best_static_oracle": best_static,
        "rows": rows,
        "child_summaries": child_summaries,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_tables(args.output_dir, rows, summary)

    print()
    print("[ablation] === Scheduler ablation matrix ===")
    print(f"{'Configuration':<28} {'Kind':<12} {'Latency (ms)':>16} {'vs best static':>16}")
    for row in rows:
        print(
            f"{row['name']:<28} {row['kind']:<12} "
            f"{row['lat_mean_ms']:>9.2f} +/- {row['lat_std_ms']:<5.2f} "
            f"{_format_ratio(row.get('vs_best_static_ratio')):>16}"
        )
    print(f"[ablation] Summary saved to {summary_path}")
    print(f"[ablation] Markdown table saved to {args.output_dir / 'summary_table.md'}")


if __name__ == "__main__":
    main()
