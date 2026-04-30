"""Per-regime static chunk sweep on local GPU.

Loads representative prompts from the mixed-regime pool and runs
static chunk sweep (128, 256, 512) for short_chat, long_doc_qa, and code_repo.
Goal: determine whether per-regime optimal chunk differs → GO/STOP for H800 guarded sweep.

Usage:
    python -m src.experiments.run_regime_static_sweep \
        --prompt-pool src/outputs/mixed_regime_discovery_20260501_nocard/prompt_pool.jsonl \
        --output-dir src/outputs/mixed_regime_regime_static_sweep_3090
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.run_integrated_end_to_end import (
    MODEL_REGISTRY,
    STATE,
    _load,
    _time_generate,
    install_patch,
    restore,
)


TARGET_REGIMES = [
    "short_chat",
    "long_doc_qa",
    "code_repo",
    "logs_uuid",
    "tables_csv",
    "repetition_policy",
    "mixed_zh_en",
    "ocr_forms",
]
SWEEP_CHUNKS = [128, 256, 512]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt-pool", type=Path,
                   default=Path("src/outputs/mixed_regime_discovery_20260501_nocard/prompt_pool.jsonl"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("src/outputs/mixed_regime_regime_static_sweep_3090"))
    p.add_argument("--model", default="mamba-370m", choices=list(MODEL_REGISTRY))
    p.add_argument("--new-tokens", type=int, default=4)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--samples-per-regime", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    pool = [json.loads(l) for l in args.prompt_pool.read_text(encoding="utf-8").splitlines() if l.strip()]
    device = torch.device("cuda")
    model, tokenizer = _load(args.model, device)

    original = install_patch()
    results: list[dict] = []
    try:
        for chunk in SWEEP_CHUNKS:
            STATE.enabled = True
            STATE.route_chunk = True
            STATE.force_chunk = chunk
            for regime in TARGET_REGIMES:
                samples = [r for r in pool if r["regime"] == regime][: args.samples_per_regime]
                lats: list[float] = []
                for idx, item in enumerate(samples):
                    STATE.records.clear()
                    timing = _time_generate(
                        model, tokenizer, item["prompt"],
                        args.new_tokens, device,
                        args.warmup if idx == 0 else 0,
                        args.repeats,
                        args.max_prompt_length,
                    )
                    lats.append(timing["lat_mean_ms"])
                    print(f"[sweep] chunk={chunk} regime={regime} sample={idx} "
                          f"lat={timing['lat_mean_ms']:.2f}ms prompt_len={timing['prompt_len']}")
                results.append({
                    "chunk": chunk,
                    "regime": regime,
                    "samples": len(lats),
                    "lat_mean_ms": round(statistics.mean(lats), 3),
                    "lat_std_ms": round(statistics.pstdev(lats), 3),
                })
                print(f"[sweep] chunk={chunk} regime={regime} mean={results[-1]['lat_mean_ms']:.2f}ms")
    finally:
        STATE.enabled = False
        STATE.route_chunk = False
        STATE.force_chunk = None
        restore(original)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== PER-REGIME STATIC CHUNK SWEEP ===")
    print(f"{'chunk':>6} {'regime':20} {'lat_mean_ms':>12} {'lat_std_ms':>10}")
    for r in results:
        print(f"{r['chunk']:>6} {r['regime']:20} {r['lat_mean_ms']:>12.2f} {r['lat_std_ms']:>10.2f}")

    per_regime_best: dict[str, tuple[int, float]] = {}
    for regime in TARGET_REGIMES:
        regime_rows = [r for r in results if r["regime"] == regime]
        best = min(regime_rows, key=lambda x: x["lat_mean_ms"])
        per_regime_best[regime] = (int(best["chunk"]), float(best["lat_mean_ms"]))

    print("\n=== PER-REGIME BEST STATIC CHUNK ===")
    chunks_set = {v[0] for v in per_regime_best.values()}
    for regime, (chunk, lat) in per_regime_best.items():
        print(f"  {regime:20} best_chunk={chunk} lat={lat:.2f}ms")

    go = len(chunks_set) > 1
    print(f"\nGO for H800 guarded sweep: {'YES' if go else 'NO'}")
    if go:
        print("  Reason: per-regime best chunks differ — guarded/learned scheduler can beat global static oracle")
    else:
        print("  Reason: same chunk wins for all regimes — guarded scheduler adds no benefit")

    summary = {
        "go_for_h800_guarded_sweep": go,
        "per_regime_best": {k: {"chunk": v[0], "lat_mean_ms": v[1]} for k, v in per_regime_best.items()},
        "all_results": results,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
