"""
Static chunk sweep -> best static oracle -> adaptive scheduler summary.

This orchestration layer runs the integrated end-to-end benchmark multiple
times with matched prompts/seeds/arguments:

  1. static routed chunk sweep, using scheduler_mode=constant + force_chunk
  2. best static oracle selection from the sweep
  3. adaptive routed scheduler run
  4. compact JSON/Markdown/CSV summary tables

Example:
    python -m src.experiments.run_static_oracle_adaptive \
        --model mamba-370m \
        --prompt-repeat 8 \
        --max-prompt-length 4096 \
        --new-tokens 32 --warmup 2 --repeats 50 \
        --sweep-chunks 128,256,512,1024,2048 \
        --adaptive-scheduler-mode sampled_hist \
        --adaptive-entropy-stride 8 \
        --selective-scan-dispatch-module src.corey_selective_scan_dispatch \
        --output-dir src/outputs/static_oracle_adaptive_h800
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.run_integrated_end_to_end import DEFAULT_PROMPT, MODEL_REGISTRY


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


def _run_child(
    name: str,
    child_args: list[str],
    output_dir: Path,
    *,
    rerun: bool,
    dry_run: bool,
) -> dict[str, Any] | None:
    summary_path = output_dir / "summary.json"
    log_path = output_dir / "run.log"

    if rerun and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if summary_path.exists() and not rerun:
        print(f"[oracle] Reusing existing {name}: {summary_path}")
        return json.loads(summary_path.read_text())

    cmd = [
        sys.executable,
        "-m",
        "src.experiments.run_integrated_end_to_end",
        *child_args,
        "--output-dir",
        str(output_dir),
    ]

    print(f"[oracle] Running {name}")
    print(f"[oracle] Command: {' '.join(cmd)}")
    if dry_run:
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
        raise RuntimeError(f"{name} failed with exit code {rc}; see {log_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"{name} did not produce {summary_path}")
    return json.loads(summary_path.read_text())


def _latency_row(
    name: str,
    kind: str,
    summary: dict[str, Any],
    *,
    forced_chunk: int | None = None,
) -> dict[str, Any]:
    integrated = summary["integrated"]
    passive = summary["passive"]
    lat = float(integrated["lat_mean_ms"])
    passive_lat = float(passive["lat_mean_ms"])
    return {
        "name": name,
        "kind": kind,
        "forced_chunk": forced_chunk,
        "scheduler_mode": summary.get("scheduler_mode"),
        "entropy_stride": summary.get("entropy_stride"),
        "lat_mean_ms": lat,
        "lat_std_ms": float(integrated["lat_std_ms"]),
        "lat_min_ms": float(integrated["lat_min_ms"]),
        "lat_max_ms": float(integrated["lat_max_ms"]),
        "vs_passive_ratio": lat / passive_lat if passive_lat else None,
        "passive_lat_mean_ms": passive_lat,
        "chunk_dist": integrated.get("chunk_dist", {}),
        "prompt_len": summary.get("prompt_len"),
        "new_tokens": summary.get("new_tokens"),
        "summary_path": str(summary.get("_summary_path", "")),
    }


def _format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}x"


def _write_tables(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    csv_path = output_dir / "summary_table.csv"
    md_path = output_dir / "summary_table.md"
    fields = [
        "name",
        "kind",
        "forced_chunk",
        "scheduler_mode",
        "entropy_stride",
        "lat_mean_ms",
        "lat_std_ms",
        "vs_passive_ratio",
        "vs_best_static_ratio",
        "speedup_over_best_static",
        "chunk_dist",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    lines = [
        "# Static Oracle vs Adaptive Summary",
        "",
        "| configuration | kind | chunk | scheduler | latency ms | vs passive | vs best static | speedup over best static | chunk distribution |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        chunk = "" if row.get("forced_chunk") is None else str(row["forced_chunk"])
        sched = str(row.get("scheduler_mode"))
        if row.get("entropy_stride") not in (None, 1):
            sched += f" stride={row['entropy_stride']}"
        lines.append(
            "| {name} | {kind} | {chunk} | {sched} | "
            "{lat:.2f} +/- {std:.2f} | {vp} | {vbs} | {sbs} | `{dist}` |".format(
                name=row["name"],
                kind=row["kind"],
                chunk=chunk,
                sched=sched,
                lat=row["lat_mean_ms"],
                std=row["lat_std_ms"],
                vp=_format_ratio(row.get("vs_passive_ratio")),
                vbs=_format_ratio(row.get("vs_best_static_ratio")),
                sbs=_format_ratio(row.get("speedup_over_best_static")),
                dist=row.get("chunk_dist", {}),
            )
        )
    best = summary["best_static_oracle"]
    adaptive = summary["adaptive"]
    lines.extend([
        "",
        f"Best static oracle: `{best['name']}` at {best['lat_mean_ms']:.2f} ms.",
        (
            "Adaptive vs best static: "
            f"{adaptive['vs_best_static_ratio']:.4f}x latency ratio, "
            f"{adaptive['speedup_over_best_static']:.4f}x speedup."
        ),
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
        "--adaptive-scheduler-mode",
        choices=(
            "hist",
            "sampled_hist",
            "token_hist",
            "cheap_proxy",
            "variance_proxy",
            "kurtosis_proxy",
            "no_entropy",
            "random",
        ),
        default="sampled_hist",
    )
    p.add_argument("--adaptive-entropy-stride", type=int, default=8)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--adaptive-chunk-min", type=int, default=None)
    p.add_argument("--adaptive-chunk-max", type=int, default=None)
    p.add_argument(
        "--selective-scan-dispatch-module",
        default=None,
        help="Dispatch module exposing selective_scan_fn(..., chunk_size=...).",
    )
    p.add_argument("--output-dir", type=Path, default=Path("src/outputs/static_oracle_adaptive"))
    p.add_argument("--rerun", action="store_true", help="Delete and rerun existing child outputs.")
    p.add_argument("--dry-run", action="store_true", help="Print child commands without running them.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    chunks = list(args.sweep_chunks)
    adaptive_min = args.adaptive_chunk_min if args.adaptive_chunk_min is not None else min(chunks)
    adaptive_max = args.adaptive_chunk_max if args.adaptive_chunk_max is not None else max(chunks)

    print("[oracle] Plan: static chunk sweep -> best static oracle -> adaptive scheduler -> summary table")
    print(f"[oracle] Static chunks: {chunks}")
    print(
        "[oracle] Adaptive: "
        f"{args.adaptive_scheduler_mode}, stride={args.adaptive_entropy_stride}, "
        f"chunk range=[{adaptive_min}, {adaptive_max}]"
    )

    rows: list[dict[str, Any]] = []
    child_summaries: dict[str, dict[str, Any]] = {}

    for chunk in chunks:
        child_args: list[str] = []
        _add_common_child_args(child_args, args)
        child_args.extend([
            "--scheduler-mode", "constant",
            "--force-chunk", str(chunk),
            "--chunk-min", str(chunk),
            "--chunk-max", str(chunk),
        ])
        name = f"static_chunk_{chunk}"
        out_dir = args.output_dir / name
        result = _run_child(name, child_args, out_dir, rerun=args.rerun, dry_run=args.dry_run)
        if result is None:
            continue
        result["_summary_path"] = str(out_dir / "summary.json")
        child_summaries[name] = result
        rows.append(_latency_row(name, "static", result, forced_chunk=chunk))

    adaptive_args: list[str] = []
    _add_common_child_args(adaptive_args, args)
    adaptive_args.extend([
        "--scheduler-mode", args.adaptive_scheduler_mode,
        "--entropy-stride", str(max(args.adaptive_entropy_stride, 1)),
        "--chunk-min", str(adaptive_min),
        "--chunk-max", str(adaptive_max),
    ])
    adaptive_name = f"adaptive_{args.adaptive_scheduler_mode}_s{max(args.adaptive_entropy_stride, 1)}"
    adaptive_dir = args.output_dir / adaptive_name
    adaptive_summary = _run_child(
        adaptive_name,
        adaptive_args,
        adaptive_dir,
        rerun=args.rerun,
        dry_run=args.dry_run,
    )
    if adaptive_summary is not None:
        adaptive_summary["_summary_path"] = str(adaptive_dir / "summary.json")
        child_summaries[adaptive_name] = adaptive_summary
        rows.append(_latency_row(adaptive_name, "adaptive", adaptive_summary))

    if args.dry_run:
        print("[oracle] Dry run complete; no summary table written.")
        return
    if not rows or not any(row["kind"] == "static" for row in rows):
        raise RuntimeError("No static rows were produced; cannot select best static oracle.")
    adaptive_rows = [row for row in rows if row["kind"] == "adaptive"]
    if not adaptive_rows:
        raise RuntimeError("Adaptive row was not produced.")

    static_rows = [row for row in rows if row["kind"] == "static"]
    best_static = min(static_rows, key=lambda row: row["lat_mean_ms"])
    adaptive = adaptive_rows[0]
    best_lat = float(best_static["lat_mean_ms"])
    for row in rows:
        lat = float(row["lat_mean_ms"])
        row["vs_best_static_ratio"] = lat / best_lat if best_lat else None
        row["speedup_over_best_static"] = best_lat / lat if lat else None

    summary = {
        "plan": "static_chunk_sweep__best_static_oracle__adaptive_scheduler",
        "output_dir": str(args.output_dir),
        "sweep_chunks": chunks,
        "adaptive_scheduler_mode": args.adaptive_scheduler_mode,
        "adaptive_entropy_stride": max(args.adaptive_entropy_stride, 1),
        "adaptive_chunk_min": adaptive_min,
        "adaptive_chunk_max": adaptive_max,
        "best_static_oracle": best_static,
        "adaptive": adaptive,
        "rows": rows,
        "child_summaries": child_summaries,
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_tables(args.output_dir, rows, summary)

    print()
    print("[oracle] === Static oracle vs adaptive summary ===")
    print(f"{'Configuration':<34} {'Latency (ms)':>16} {'vs best static':>16} {'speedup':>10}")
    for row in rows:
        print(
            f"{row['name']:<34} "
            f"{row['lat_mean_ms']:>9.2f} +/- {row['lat_std_ms']:<5.2f} "
            f"{_format_ratio(row.get('vs_best_static_ratio')):>16} "
            f"{_format_ratio(row.get('speedup_over_best_static')):>10}"
        )
    print(f"[oracle] Best static oracle: {best_static['name']}")
    print(f"[oracle] Summary saved to {summary_path}")
    print(f"[oracle] Markdown table saved to {args.output_dir / 'summary_table.md'}")


if __name__ == "__main__":
    main()
