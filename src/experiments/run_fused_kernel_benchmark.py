#!/usr/bin/env python
"""
run_fused_kernel_benchmark.py

Benchmark the entropy-guided fusion algorithm across operator-chain lengths,
tau thresholds, and entropy regimes.  Measures wall-clock time of the Python
fusion solver itself and reports surrogate latency reduction (kernel-launch
count) relative to no-fusion and static-fusion baselines.

Outputs results to src/outputs/fused_kernel_benchmark/
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

PYTHONPATH_NOTE = "Run with: PYTHONPATH=src python src/experiments/run_fused_kernel_benchmark.py"


def _build_chain(num_ops: int, entropy_regime: str):
    from algorithms.fusion import OperatorSpec

    if entropy_regime == "low":
        # Low-entropy regime: operators barely exceed fusion threshold
        return [
            OperatorSpec(
                name=f"op{i}",
                entropy=1.5 + i * 0.05,
                arithmetic_intensity=0.8 + i * 0.1,
                memory_traffic=0.6 + i * 0.04,
                register_cost=14,
                shared_memory_cost=28,
                occupancy=0.75,
            )
            for i in range(num_ops)
        ]
    elif entropy_regime == "high":
        # High-entropy regime: operators strongly benefit from fusion
        return [
            OperatorSpec(
                name=f"op{i}",
                entropy=3.5 + i * 0.15,
                arithmetic_intensity=1.8 + i * 0.25,
                memory_traffic=0.4 + i * 0.03,
                register_cost=18,
                shared_memory_cost=36,
                occupancy=0.82,
            )
            for i in range(num_ops)
        ]
    else:
        # Mixed regime: matches the paper's synthetic benchmark
        return [
            OperatorSpec(
                name=f"op{i}",
                entropy=2.0 + i * 0.1,
                arithmetic_intensity=1.0 + i * 0.2,
                memory_traffic=0.5 + i * 0.05,
                register_cost=16,
                shared_memory_cost=32,
                occupancy=0.80,
            )
            for i in range(num_ops)
        ]


def _surrogate_latency_ms(num_launches: int, base_dispatch_ms: float = 0.413) -> float:
    """Approximate dispatch latency from kernel-launch count.

    The constant 0.413 ms is the measured Python dispatch overhead per kernel
    call on RTX 3070 (WSL2, CUDA 12.8), derived from the chunk-sweep harness
    (Table A.12, static-64 row: 3.30 ms / 8 calls).  This is used only as a
    surrogate metric to illustrate launch-count reduction; actual fused-kernel
    timing would require the Triton implementation.
    """
    return num_launches * base_dispatch_ms


def benchmark_fusion(num_ops: int, tau: float, entropy_regime: str, repeats: int = 200):
    from algorithms.fusion import (
        ResourceModel,
        build_no_fusion_groups,
        build_static_fusion_groups,
        select_fusion_groups,
    )

    chain = _build_chain(num_ops, entropy_regime)
    resource_model = ResourceModel(max_registers=256, max_shared_memory=512, min_occupancy=0.5)

    # --- Entropy-guided fusion (COREY) ---
    times_corey = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        groups_corey = select_fusion_groups(chain, tau=tau, resource_model=resource_model)
        times_corey.append(time.perf_counter() - t0)

    # --- No-fusion baseline ---
    groups_nofuse = build_no_fusion_groups(chain)

    # --- Static fusion baseline ---
    groups_static = build_static_fusion_groups(chain, resource_model, group_size=3)

    launches_corey = len(groups_corey)
    launches_nofuse = len(groups_nofuse)
    launches_static = len(groups_static)

    mean_solver_us = (sum(times_corey) / len(times_corey)) * 1e6

    return {
        "num_ops": num_ops,
        "tau": tau,
        "entropy_regime": entropy_regime,
        "launches_nofuse": launches_nofuse,
        "launches_static": launches_static,
        "launches_corey": launches_corey,
        "surrogate_lat_nofuse_ms": round(_surrogate_latency_ms(launches_nofuse), 4),
        "surrogate_lat_static_ms": round(_surrogate_latency_ms(launches_static), 4),
        "surrogate_lat_corey_ms": round(_surrogate_latency_ms(launches_corey), 4),
        "reduction_vs_nofuse_pct": round(
            (1.0 - launches_corey / launches_nofuse) * 100.0, 2
        ),
        "reduction_vs_static_pct": round(
            (1.0 - launches_corey / max(launches_static, 1)) * 100.0, 2
        ),
        "solver_mean_us": round(mean_solver_us, 3),
        "group_depths": [g.depth for g in groups_corey],
        "group_scores": [round(g.score, 4) for g in groups_corey],
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark entropy-guided fusion algorithm.")
    parser.add_argument("--num-ops", type=int, default=8, help="Operator chain length.")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (metadata only).")
    parser.add_argument("--tau", type=float, default=0.5, help="Fusion score threshold.")
    parser.add_argument("--repeats", type=int, default=200, help="Solver timing repeats.")
    parser.add_argument(
        "--sweep",
        action="store_true",
        default=True,
        help="Run a sweep over chain lengths and entropy regimes (default: True).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/outputs/fused_kernel_benchmark",
        help="Output directory.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sweep_configs = [
        (4, 0.5, "low"),
        (4, 0.5, "mixed"),
        (4, 0.5, "high"),
        (8, 0.5, "low"),
        (8, 0.5, "mixed"),
        (8, 0.5, "high"),
        (16, 0.5, "low"),
        (16, 0.5, "mixed"),
        (16, 0.5, "high"),
        (32, 0.5, "mixed"),
        (64, 0.5, "mixed"),
    ] if args.sweep else [(args.num_ops, args.tau, "mixed")]

    all_results = []
    for num_ops, tau, regime in sweep_configs:
        print(f"[INFO] Benchmarking: num_ops={num_ops}, tau={tau}, regime={regime}")
        res = benchmark_fusion(num_ops, tau, regime, repeats=args.repeats)
        all_results.append(res)
        print(
            f"[RESULT] launches: nofuse={res['launches_nofuse']}  "
            f"static={res['launches_static']}  corey={res['launches_corey']}  "
            f"reduction={res['reduction_vs_nofuse_pct']:.1f}%  "
            f"solver={res['solver_mean_us']:.1f}µs"
        )

    # Single-entry summary for the default num_ops (backward compat)
    default_entry = next(
        (r for r in all_results if r["num_ops"] == args.num_ops and r["entropy_regime"] == "mixed"),
        all_results[0],
    )
    single_result = {
        "num_ops": default_entry["num_ops"],
        "fusion_groups": default_entry["group_depths"],
        "latency_ms": default_entry["surrogate_lat_corey_ms"],
    }

    out_path = Path(args.output_dir) / "fused_kernel_benchmark_result.json"
    with open(out_path, "w") as f:
        json.dump(single_result, f, indent=2)

    sweep_path = Path(args.output_dir) / "fused_kernel_sweep.json"
    with open(sweep_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"[DONE] Single-entry result saved to {out_path}")
    print(f"[DONE] Sweep results saved to {sweep_path}")


if __name__ == "__main__":
    main()
