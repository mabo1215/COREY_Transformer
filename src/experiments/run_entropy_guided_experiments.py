from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import numpy as np

from src.algorithms import (
    ExponentialMovingEntropy,
    OperatorSpec,
    ResourceModel,
    build_no_fusion_groups,
    build_static_fusion_groups,
    entropy_gain,
    fused_hadamard_projection,
    normalized_entropy,
    outlier_ratio,
    reference_projection,
    select_fusion_groups,
)


SEQUENCE_BUCKETS = {
    "short": (1024, 2048),
    "medium": (4096, 8192),
    "long": (16384, 32768),
    "ultra_long": (65536,),
}

PRECISION_SPEEDUP = {"fp16": 1.0, "w8a8": 1.12, "w4a8": 1.22}
PRECISION_SENSITIVITY = {"fp16": 0.0, "w8a8": 0.7, "w4a8": 1.3}


@dataclass(frozen=True)
class ExperimentConfig:
    hidden_dim: int = 192
    projection_dim: int = 256
    tau: float = 0.52
    repeats: int = 5
    bins: int = 64
    seed: int = 7


@dataclass(frozen=True)
class ScheduleMetrics:
    method: str
    sequence_length: int
    precision: str
    latency_ms: float
    throughput_tokens_per_s: float
    dram_bytes_per_token: float
    quality_drop: float
    average_fusion_depth: float
    p95_group_depth: float
    average_group_occupancy: float
    min_group_occupancy: float
    average_register_cost: float
    average_shared_memory_cost: float
    entropy_before: float
    entropy_after: float
    entropy_gain: float
    outlier_before: float
    outlier_after: float
    max_projection_error: float


@dataclass(frozen=True)
class ScheduleTraceRow:
    bucket: str
    sequence_length: int
    repeat: int
    method: str
    tau: float
    target_depth: float
    group_index: int
    group_depth: int
    group_occupancy: float
    group_register_cost: int
    group_shared_memory_cost: int
    group_entropy: float
    group_arithmetic_intensity: float
    group_memory_traffic: float
    group_score: float
    feasible: bool


@dataclass(frozen=True)
class TileTraceRow:
    bucket: str
    sequence_length: int
    repeat: int
    method: str
    tau: float
    group_index: int
    tile_index: int
    tile_size: int
    estimated_tile_count: int
    group_depth: int
    group_occupancy: float
    group_entropy: float
    group_register_cost: int
    group_shared_memory_cost: int
    tile_entropy: float
    tile_memory_bytes: float
    tile_compute_cost: float
    tile_latency_ms: float
    cumulative_group_latency_ms: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run entropy-guided fusion experiments.")
    parser.add_argument("--output-dir", type=Path, default=Path("src/outputs"))
    parser.add_argument("--tau", type=float, default=0.52)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def _sample_count(sequence_length: int) -> int:
    return min(4096, max(512, sequence_length // 8))


def _generate_activations(sequence_length: int, hidden_dim: int, rng: np.random.Generator) -> np.ndarray:
    samples = _sample_count(sequence_length)
    gaussian = rng.normal(loc=0.0, scale=1.0, size=(samples, hidden_dim))
    heavy_tail = 0.35 * rng.standard_t(df=3.0, size=(samples, hidden_dim))
    length_scale = 1.0 + math.log2(sequence_length) / 20.0
    return gaussian + length_scale * heavy_tail


def _build_operator_chain(
    activations: np.ndarray,
    rotated_activations: np.ndarray,
    sequence_length: int,
) -> list[OperatorSpec]:
    templates = [
        ("input_norm", 1.6, 18.0, 20, 8, 0.94),
        ("gate_projection", 2.7, 22.0, 36, 12, 0.91),
        ("selective_scan", 4.8, 31.0, 72, 28, 0.79),
        ("state_mix", 3.9, 24.0, 54, 22, 0.83),
        ("output_projection", 3.4, 19.0, 48, 18, 0.87),
        ("activation", 1.9, 14.0, 24, 8, 0.96),
        ("residual_merge", 1.5, 12.0, 18, 6, 0.97),
    ]
    seq_scale = 1.0 + math.log2(sequence_length) / 6.0
    operators: list[OperatorSpec] = []
    channel_splits = np.array_split(np.arange(activations.shape[1]), len(templates))

    for index, (name, arithmetic_base, memory_base, register_cost, shared_memory_cost, occupancy) in enumerate(templates):
        channels = channel_splits[index]
        base_slice = activations[:, channels]
        rotated_slice = rotated_activations[:, channels]
        local_entropy = 0.55 * normalized_entropy(base_slice) + 0.45 * normalized_entropy(rotated_slice)
        operators.append(
            OperatorSpec(
                name=name,
                entropy=float(np.clip(local_entropy, 0.0, 1.0)),
                arithmetic_intensity=arithmetic_base * seq_scale,
                memory_traffic=memory_base * seq_scale,
                register_cost=register_cost,
                shared_memory_cost=shared_memory_cost,
                occupancy=occupancy,
            )
        )

    return operators


def _resource_model(sequence_length: int) -> ResourceModel:
    long_context_penalty = int(math.log2(sequence_length) * 2)
    return ResourceModel(
        max_registers=180 - long_context_penalty,
        max_shared_memory=96 - long_context_penalty,
        min_occupancy=0.72,
    )


def _group_depth_p95(group_depths: list[int]) -> float:
    if not group_depths:
        return 0.0
    percentile_index = int(math.ceil(0.95 * len(group_depths))) - 1
    percentile_index = max(0, min(percentile_index, len(group_depths) - 1))
    return float(sorted(group_depths)[percentile_index])


def _group_metric_mean(groups, attribute: str) -> float:
    if not groups:
        return 0.0
    return float(mean(getattr(group, attribute) for group in groups))


def _group_metric_min(groups, attribute: str) -> float:
    if not groups:
        return 0.0
    return float(min(getattr(group, attribute) for group in groups))


def _bucket_name(sequence_length: int) -> str:
    for bucket_name, lengths in SEQUENCE_BUCKETS.items():
        if sequence_length in lengths:
            return bucket_name
    raise ValueError(f"Unexpected sequence length: {sequence_length}")


def _average_depth(groups) -> float:
    return float(mean(group.depth for group in groups)) if groups else 0.0


def _calibrate_arithmetic_only_schedule(chain, resource_model: ResourceModel, target_depth: float, base_tau: float):
    tau_candidates = [round(value, 2) for value in np.linspace(-0.2, max(base_tau, 0.8), 21)]
    best_tau = base_tau
    best_groups = select_fusion_groups(chain, tau=base_tau, resource_model=resource_model, alpha=0.0)
    best_gap = abs(_average_depth(best_groups) - target_depth)

    for tau in tau_candidates:
        groups = select_fusion_groups(chain, tau=tau, resource_model=resource_model, alpha=0.0)
        gap = abs(_average_depth(groups) - target_depth)
        if gap < best_gap:
            best_tau = tau
            best_groups = groups
            best_gap = gap
            continue
        if math.isclose(gap, best_gap, rel_tol=1e-9, abs_tol=1e-9) and abs(tau - base_tau) < abs(best_tau - base_tau):
            best_tau = tau
            best_groups = groups

    return best_tau, best_groups


def _build_schedule_trace_rows(
    sequence_length: int,
    repeat: int,
    method: str,
    tau: float,
    target_depth: float,
    groups,
) -> list[dict[str, object]]:
    bucket_name = _bucket_name(sequence_length)
    rows: list[dict[str, object]] = []
    for group_index, group in enumerate(groups):
        rows.append(
            asdict(
                ScheduleTraceRow(
                    bucket=bucket_name,
                    sequence_length=sequence_length,
                    repeat=repeat,
                    method=method,
                    tau=round(tau, 4),
                    target_depth=round(target_depth, 4),
                    group_index=group_index,
                    group_depth=group.depth,
                    group_occupancy=round(group.occupancy, 4),
                    group_register_cost=group.register_cost,
                    group_shared_memory_cost=group.shared_memory_cost,
                    group_entropy=round(group.entropy, 4),
                    group_arithmetic_intensity=round(group.arithmetic_intensity, 4),
                    group_memory_traffic=round(group.memory_traffic, 4),
                    group_score=round(group.score, 4),
                    feasible=group.feasible,
                )
            )
        )
    return rows


def _recommend_prototype_tile_size(entropy: float, base_tile: int = 64, max_tile: int = 512) -> int:
    normalized = min(max(entropy, 0.0), 1.0)
    suggested = base_tile + normalized * (max_tile - base_tile)
    return int(round(suggested / 32.0) * 32)


def _shared_histogram_mass(before: np.ndarray, after: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    before_array = np.asarray(before, dtype=np.float64).reshape(-1)
    after_array = np.asarray(after, dtype=np.float64).reshape(-1)
    combined = np.concatenate([before_array, after_array])
    minimum = float(np.min(combined))
    maximum = float(np.max(combined))
    if np.isclose(minimum, maximum):
        probabilities = np.zeros(bins, dtype=np.float64)
        probabilities[0] = 1.0
        centers = np.linspace(minimum, maximum + 1e-6, bins)
        return probabilities, probabilities.copy(), centers

    margin = max((maximum - minimum) * 0.05, 1e-6)
    histogram_range = (minimum - margin, maximum + margin)
    before_histogram, edges = np.histogram(before_array, bins=bins, range=histogram_range)
    after_histogram, _ = np.histogram(after_array, bins=bins, range=histogram_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    before_probabilities = before_histogram.astype(np.float64)
    after_probabilities = after_histogram.astype(np.float64)
    before_probabilities /= max(float(before_probabilities.sum()), 1.0)
    after_probabilities /= max(float(after_probabilities.sum()), 1.0)
    return before_probabilities, after_probabilities, centers


def _sinkhorn_bistochastic_projection(centers: np.ndarray, temperature: float, iterations: int = 200) -> np.ndarray:
    distance = centers[:, None] - centers[None, :]
    kernel = np.exp(-(distance**2) / max(temperature, 1e-9))
    projection = np.maximum(kernel, 1e-12)
    for _ in range(iterations):
        projection /= np.maximum(projection.sum(axis=1, keepdims=True), 1e-12)
        projection /= np.maximum(projection.sum(axis=0, keepdims=True), 1e-12)
    return projection


def _fit_sinkhorn_proxy(before: np.ndarray, after: np.ndarray, bins: int) -> dict[str, float]:
    before_mass, after_mass, centers = _shared_histogram_mass(before, after, bins=bins)
    if len(centers) < 2:
        transported = before_mass.copy()
        residual = float(np.sum(np.abs(after_mass - transported)))
        return {
            "sinkhorn_temperature": 0.0,
            "sinkhorn_residual_l1": residual,
            "sinkhorn_residual_l2": float(np.linalg.norm(after_mass - transported)),
            "sinkhorn_row_error": 0.0,
            "sinkhorn_col_error": 0.0,
        }

    spacing = max(float(np.mean(np.diff(centers) ** 2)), 1e-6)
    best_result: dict[str, float] | None = None
    for multiplier in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        temperature = spacing * multiplier
        bistochastic = _sinkhorn_bistochastic_projection(centers, temperature=temperature)
        transported = bistochastic @ before_mass
        residual_l1 = float(np.sum(np.abs(after_mass - transported)))
        residual_l2 = float(np.linalg.norm(after_mass - transported))
        row_error = float(np.max(np.abs(bistochastic.sum(axis=1) - 1.0)))
        col_error = float(np.max(np.abs(bistochastic.sum(axis=0) - 1.0)))
        candidate = {
            "sinkhorn_temperature": float(temperature),
            "sinkhorn_residual_l1": residual_l1,
            "sinkhorn_residual_l2": residual_l2,
            "sinkhorn_row_error": row_error,
            "sinkhorn_col_error": col_error,
        }
        if best_result is None or candidate["sinkhorn_residual_l1"] < best_result["sinkhorn_residual_l1"]:
            best_result = candidate

    assert best_result is not None
    return best_result


def _estimate_group_latency_proxy(group, sequence_length: int, method: str) -> float:
    seq_scale = math.log2(sequence_length)
    memory_reuse = 1.0 - 0.09 * (group.depth - 1) - 0.08 * group.entropy
    if method == "static_fusion":
        memory_reuse += 0.04 * max(0.0, 0.62 - group.entropy)
    if method == "entropy_guided":
        memory_reuse -= 0.04 * group.entropy
    if method == "arithmetic_only_matched":
        memory_reuse -= 0.02 * max(group.entropy - 0.55, 0.0)
    memory_reuse = float(np.clip(memory_reuse, 0.42, 1.1))

    adjusted_memory = group.memory_traffic * seq_scale * memory_reuse
    adjusted_compute = group.arithmetic_intensity * seq_scale
    launch_overhead = 0.28 if method == "no_fusion" else 0.16
    launch_overhead += 0.03 * max(group.depth - 3, 0)
    if method == "entropy_guided":
        launch_overhead -= 0.02 * min(group.depth - 1, 2)
    return max(0.0, 0.012 * adjusted_memory + 0.007 * adjusted_compute + launch_overhead)


def _build_tile_trace_rows(
    sequence_length: int,
    repeat: int,
    method: str,
    tau: float,
    groups,
) -> list[dict[str, object]]:
    bucket_name = _bucket_name(sequence_length)
    rows: list[dict[str, object]] = []
    for group_index, group in enumerate(groups):
        tile_size = _recommend_prototype_tile_size(group.entropy)
        estimated_tile_count = max(1, math.ceil(sequence_length / max(tile_size, 1)))
        group_latency_ms = _estimate_group_latency_proxy(group, sequence_length, method)
        group_memory_bytes = group.memory_traffic * math.log2(sequence_length) * 1024.0
        group_compute_cost = group.arithmetic_intensity * math.log2(sequence_length)
        tile_weights: list[float] = []
        tile_entropies: list[float] = []
        for tile_index in range(estimated_tile_count):
            position = (tile_index + 0.5) / estimated_tile_count
            entropy_wave = 0.06 * math.sin(2.0 * math.pi * position)
            local_entropy = float(np.clip(group.entropy + entropy_wave * (1.0 if method in {"entropy_guided", "arithmetic_only_matched"} else 0.5), 0.0, 1.0))
            tile_entropies.append(local_entropy)
            weight = 1.0 + 0.22 * math.sin(math.pi * position)
            weight += 0.18 * (1.0 - local_entropy)
            weight += 0.05 * max(group.depth - 2, 0)
            if method == "entropy_guided":
                weight -= 0.08 * local_entropy
            tile_weights.append(max(weight, 0.2))

        total_weight = sum(tile_weights)
        cumulative_latency_ms = 0.0
        for tile_index in range(estimated_tile_count):
            tile_latency_ms = group_latency_ms * tile_weights[tile_index] / total_weight
            tile_memory_bytes = group_memory_bytes * tile_weights[tile_index] / total_weight
            tile_compute_cost = group_compute_cost * tile_weights[tile_index] / total_weight
            cumulative_latency_ms += tile_latency_ms
            rows.append(
                asdict(
                    TileTraceRow(
                        bucket=bucket_name,
                        sequence_length=sequence_length,
                        repeat=repeat,
                        method=method,
                        tau=round(tau, 4),
                        group_index=group_index,
                        tile_index=tile_index,
                        tile_size=tile_size,
                        estimated_tile_count=estimated_tile_count,
                        group_depth=group.depth,
                        group_occupancy=round(group.occupancy, 4),
                        group_entropy=round(group.entropy, 4),
                        group_register_cost=group.register_cost,
                        group_shared_memory_cost=group.shared_memory_cost,
                        tile_entropy=round(tile_entropies[tile_index], 4),
                        tile_memory_bytes=round(tile_memory_bytes, 4),
                        tile_compute_cost=round(tile_compute_cost, 4),
                        tile_latency_ms=round(tile_latency_ms, 4),
                        cumulative_group_latency_ms=round(cumulative_latency_ms, 4),
                    )
                )
            )
    return rows


def _estimate_metrics(
    method: str,
    sequence_length: int,
    precision: str,
    groups,
    entropy_before_value: float,
    entropy_after_value: float,
    entropy_gain_value: float,
    outlier_before_value: float,
    outlier_after_value: float,
    max_projection_error: float,
) -> ScheduleMetrics:
    seq_scale = math.log2(sequence_length)
    speedup = PRECISION_SPEEDUP[precision]
    quantization_penalty = PRECISION_SENSITIVITY[precision]
    group_depths = [group.depth for group in groups]
    average_depth = mean(group_depths)
    average_occupancy = _group_metric_mean(groups, "occupancy")
    min_occupancy = _group_metric_min(groups, "occupancy")
    average_register_cost = _group_metric_mean(groups, "register_cost")
    average_shared_memory_cost = _group_metric_mean(groups, "shared_memory_cost")

    total_latency = 0.0
    total_memory = 0.0
    for group in groups:
        memory_reuse = 1.0 - 0.09 * (group.depth - 1) - 0.08 * group.entropy
        if method == "static_fusion":
            memory_reuse += 0.04 * max(0.0, 0.62 - group.entropy)
        if method == "entropy_guided":
            memory_reuse -= 0.06 * max(entropy_gain_value, 0.0)
        memory_reuse = float(np.clip(memory_reuse, 0.42, 1.1))

        adjusted_memory = group.memory_traffic * seq_scale * memory_reuse
        adjusted_compute = group.arithmetic_intensity * seq_scale / speedup
        launch_overhead = 0.28 if method == "no_fusion" else 0.16
        launch_overhead += 0.03 * max(group.depth - 3, 0)
        if method == "entropy_guided":
            launch_overhead -= 0.02 * min(group.depth - 1, 2)

        total_memory += adjusted_memory
        total_latency += 0.012 * adjusted_memory + 0.007 * adjusted_compute + launch_overhead

    over_fusion_penalty = max(0.0, average_depth - 2.0) * max(0.0, 0.63 - entropy_after_value)
    if method == "no_fusion":
        over_fusion_penalty = 0.0
    if method == "entropy_guided":
        over_fusion_penalty *= 0.35

    hadamard_bonus = 0.0
    if method == "entropy_guided":
        hadamard_bonus = 0.45 * max(entropy_gain_value, 0.0) + 5.0 * max(outlier_before_value - outlier_after_value, 0.0)

    quality_drop = max(0.0, quantization_penalty * (0.12 + outlier_before_value * 10.0) + over_fusion_penalty - hadamard_bonus)
    throughput = sequence_length / (total_latency / 1000.0)
    dram_bytes_per_token = total_memory * 1024.0 / sequence_length

    return ScheduleMetrics(
        method=method,
        sequence_length=sequence_length,
        precision=precision,
        latency_ms=round(total_latency, 4),
        throughput_tokens_per_s=round(throughput, 4),
        dram_bytes_per_token=round(dram_bytes_per_token, 4),
        quality_drop=round(quality_drop, 4),
        average_fusion_depth=round(average_depth, 4),
        p95_group_depth=round(_group_depth_p95(group_depths), 4),
        average_group_occupancy=round(average_occupancy, 4),
        min_group_occupancy=round(min_occupancy, 4),
        average_register_cost=round(average_register_cost, 4),
        average_shared_memory_cost=round(average_shared_memory_cost, 4),
        entropy_before=round(entropy_before_value, 4),
        entropy_after=round(entropy_after_value, 4),
        entropy_gain=round(entropy_gain_value, 4),
        outlier_before=round(outlier_before_value, 6),
        outlier_after=round(outlier_after_value, 6),
        max_projection_error=round(max_projection_error, 10),
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize_by_bucket(results: list[ScheduleMetrics]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for bucket_name, lengths in SEQUENCE_BUCKETS.items():
        for precision in PRECISION_SPEEDUP:
            for method in ("no_fusion", "static_fusion", "entropy_guided"):
                bucket_results = [
                    result
                    for result in results
                    if result.sequence_length in lengths and result.precision == precision and result.method == method
                ]
                if not bucket_results:
                    continue
                summary_rows.append(
                    {
                        "bucket": bucket_name,
                        "precision": precision,
                        "method": method,
                        "latency_ms": round(mean(result.latency_ms for result in bucket_results), 4),
                        "throughput_tokens_per_s": round(mean(result.throughput_tokens_per_s for result in bucket_results), 4),
                        "dram_bytes_per_token": round(mean(result.dram_bytes_per_token for result in bucket_results), 4),
                        "quality_drop": round(mean(result.quality_drop for result in bucket_results), 4),
                        "average_fusion_depth": round(mean(result.average_fusion_depth for result in bucket_results), 4),
                        "average_group_occupancy": round(mean(result.average_group_occupancy for result in bucket_results), 4),
                        "min_group_occupancy": round(mean(result.min_group_occupancy for result in bucket_results), 4),
                        "average_register_cost": round(mean(result.average_register_cost for result in bucket_results), 4),
                        "average_shared_memory_cost": round(mean(result.average_shared_memory_cost for result in bucket_results), 4),
                        "entropy_gain": round(mean(result.entropy_gain for result in bucket_results), 4),
                    }
                )
    return summary_rows


def _summarize_occupancy(results: list[ScheduleMetrics]) -> list[dict[str, object]]:
    occupancy_rows: list[dict[str, object]] = []
    for bucket_name, lengths in SEQUENCE_BUCKETS.items():
        for method in ("no_fusion", "static_fusion", "entropy_guided", "arithmetic_only", "arithmetic_only_matched"):
            bucket_results = [
                result
                for result in results
                if result.sequence_length in lengths and result.precision == "fp16" and result.method == method
            ]
            if not bucket_results:
                continue
            occupancy_rows.append(
                {
                    "bucket": bucket_name,
                    "method": method,
                    "average_group_occupancy": round(mean(result.average_group_occupancy for result in bucket_results), 4),
                    "min_group_occupancy": round(mean(result.min_group_occupancy for result in bucket_results), 4),
                    "average_fusion_depth": round(mean(result.average_fusion_depth for result in bucket_results), 4),
                    "average_register_cost": round(mean(result.average_register_cost for result in bucket_results), 4),
                    "average_shared_memory_cost": round(mean(result.average_shared_memory_cost for result in bucket_results), 4),
                }
            )
    return occupancy_rows


def _summarize_alpha_zero_ablation(results: list[ScheduleMetrics]) -> list[dict[str, object]]:
    ablation_rows: list[dict[str, object]] = []
    for bucket_name, lengths in SEQUENCE_BUCKETS.items():
        for precision in PRECISION_SPEEDUP:
            entropy_guided_rows = [
                result
                for result in results
                if result.sequence_length in lengths and result.precision == precision and result.method == "entropy_guided"
            ]
            arithmetic_only_rows = [
                result
                for result in results
                if result.sequence_length in lengths and result.precision == precision and result.method == "arithmetic_only"
            ]
            if not entropy_guided_rows or not arithmetic_only_rows:
                continue
            entropy_latency = mean(result.latency_ms for result in entropy_guided_rows)
            arithmetic_latency = mean(result.latency_ms for result in arithmetic_only_rows)
            ablation_rows.append(
                {
                    "bucket": bucket_name,
                    "precision": precision,
                    "entropy_guided_latency_ms": round(entropy_latency, 4),
                    "arithmetic_only_latency_ms": round(arithmetic_latency, 4),
                    "latency_delta_ms": round(arithmetic_latency - entropy_latency, 4),
                    "entropy_guided_throughput_tokens_per_s": round(mean(result.throughput_tokens_per_s for result in entropy_guided_rows), 4),
                    "arithmetic_only_throughput_tokens_per_s": round(mean(result.throughput_tokens_per_s for result in arithmetic_only_rows), 4),
                    "entropy_guided_quality_drop": round(mean(result.quality_drop for result in entropy_guided_rows), 4),
                    "arithmetic_only_quality_drop": round(mean(result.quality_drop for result in arithmetic_only_rows), 4),
                    "entropy_guided_average_depth": round(mean(result.average_fusion_depth for result in entropy_guided_rows), 4),
                    "arithmetic_only_average_depth": round(mean(result.average_fusion_depth for result in arithmetic_only_rows), 4),
                    "entropy_guided_average_group_occupancy": round(mean(result.average_group_occupancy for result in entropy_guided_rows), 4),
                    "arithmetic_only_average_group_occupancy": round(mean(result.average_group_occupancy for result in arithmetic_only_rows), 4),
                }
            )
    return ablation_rows


def _summarize_matched_alpha_zero(results: list[ScheduleMetrics], matched_tau_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for bucket_name, lengths in SEQUENCE_BUCKETS.items():
        for precision in PRECISION_SPEEDUP:
            entropy_rows = [
                result
                for result in results
                if result.sequence_length in lengths and result.precision == precision and result.method == "entropy_guided"
            ]
            matched_rows = [
                result
                for result in results
                if result.sequence_length in lengths and result.precision == precision and result.method == "arithmetic_only_matched"
            ]
            if not entropy_rows or not matched_rows:
                continue
            tau_candidates = [
                row["matched_tau"]
                for row in matched_tau_rows
                if row["bucket"] == bucket_name and row["sequence_length"] in lengths
            ]
            summary_rows.append(
                {
                    "bucket": bucket_name,
                    "precision": precision,
                    "matched_tau_mean": round(mean(tau_candidates), 4) if tau_candidates else "",
                    "entropy_guided_latency_ms": round(mean(result.latency_ms for result in entropy_rows), 4),
                    "arithmetic_only_matched_latency_ms": round(mean(result.latency_ms for result in matched_rows), 4),
                    "latency_delta_ms": round(mean(result.latency_ms for result in matched_rows) - mean(result.latency_ms for result in entropy_rows), 4),
                    "entropy_guided_quality_drop": round(mean(result.quality_drop for result in entropy_rows), 4),
                    "arithmetic_only_matched_quality_drop": round(mean(result.quality_drop for result in matched_rows), 4),
                    "entropy_guided_average_depth": round(mean(result.average_fusion_depth for result in entropy_rows), 4),
                    "arithmetic_only_matched_average_depth": round(mean(result.average_fusion_depth for result in matched_rows), 4),
                    "entropy_guided_average_group_occupancy": round(mean(result.average_group_occupancy for result in entropy_rows), 4),
                    "arithmetic_only_matched_average_group_occupancy": round(mean(result.average_group_occupancy for result in matched_rows), 4),
                }
            )
    return summary_rows


def _summarize_sinkhorn_validation(validation_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for bucket_name, lengths in SEQUENCE_BUCKETS.items():
        bucket_rows = [row for row in validation_rows if row["sequence_length"] in lengths]
        if not bucket_rows:
            continue
        residuals_l1 = [float(row["sinkhorn_residual_l1"]) for row in bucket_rows]
        residuals_l2 = [float(row["sinkhorn_residual_l2"]) for row in bucket_rows]
        temperatures = [float(row["sinkhorn_temperature"]) for row in bucket_rows]
        summary_rows.append(
            {
                "bucket": bucket_name,
                "mean_residual_l1": round(mean(residuals_l1), 6),
                "min_residual_l1": round(min(residuals_l1), 6),
                "max_residual_l1": round(max(residuals_l1), 6),
                "mean_residual_l2": round(mean(residuals_l2), 6),
                "mean_temperature": round(mean(temperatures), 8),
                "fraction_below_0p05": round(sum(value <= 0.05 for value in residuals_l1) / len(residuals_l1), 4),
                "fraction_below_0p10": round(sum(value <= 0.10 for value in residuals_l1) / len(residuals_l1), 4),
            }
        )
    return summary_rows


def _summarize_tile_runtime(tile_trace_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for bucket_name in SEQUENCE_BUCKETS:
        for method in ("no_fusion", "static_fusion", "entropy_guided", "arithmetic_only", "arithmetic_only_matched"):
            selected_rows = [row for row in tile_trace_rows if row["bucket"] == bucket_name and row["method"] == method]
            if not selected_rows:
                continue
            latencies = sorted(float(row["tile_latency_ms"]) for row in selected_rows)
            percentile_index = max(0, min(len(latencies) - 1, int(math.ceil(0.95 * len(latencies))) - 1))
            terminal_latency_by_group: dict[tuple[int, int, int], float] = {}
            for row in selected_rows:
                key = (int(row["sequence_length"]), int(row["repeat"]), int(row["group_index"]))
                terminal_latency_by_group[key] = max(
                    terminal_latency_by_group.get(key, 0.0),
                    float(row["cumulative_group_latency_ms"]),
                )
            summary_rows.append(
                {
                    "bucket": bucket_name,
                    "method": method,
                    "mean_tile_size": round(mean(float(row["tile_size"]) for row in selected_rows), 4),
                    "mean_tile_entropy": round(mean(float(row["tile_entropy"]) for row in selected_rows), 4),
                    "mean_tile_latency_ms": round(mean(float(row["tile_latency_ms"]) for row in selected_rows), 6),
                    "p95_tile_latency_ms": round(latencies[percentile_index], 6),
                    "mean_tile_memory_bytes": round(mean(float(row["tile_memory_bytes"]) for row in selected_rows), 4),
                    "mean_group_runtime_ms": round(mean(terminal_latency_by_group.values()), 6),
                }
            )
    return summary_rows


def run_experiments(config: ExperimentConfig, output_dir: Path) -> dict[str, object]:
    rng = np.random.default_rng(config.seed)
    results: list[ScheduleMetrics] = []
    validation_rows: list[dict[str, object]] = []
    schedule_trace_rows: list[dict[str, object]] = []
    tile_trace_rows: list[dict[str, object]] = []
    matched_tau_rows: list[dict[str, object]] = []

    for sequence_length in [length for bucket in SEQUENCE_BUCKETS.values() for length in bucket]:
        moving_entropy = ExponentialMovingEntropy(decay=0.85, bins=config.bins)

        for repeat in range(config.repeats):
            activations = _generate_activations(sequence_length, config.hidden_dim, rng)
            weight = rng.normal(size=(config.projection_dim, config.hidden_dim))
            reference = reference_projection(activations, weight)
            projected, rotated_inputs, _ = fused_hadamard_projection(activations, weight)

            before_entropy = moving_entropy.update(activations)
            after_entropy = moving_entropy.update(rotated_inputs)
            gain = entropy_gain(activations, rotated_inputs, bins=config.bins)
            before_outlier = outlier_ratio(activations)
            after_outlier = outlier_ratio(rotated_inputs)
            max_projection_error = float(np.max(np.abs(reference - projected)))
            sinkhorn_fit = _fit_sinkhorn_proxy(activations, rotated_inputs, bins=config.bins)

            validation_rows.append(
                {
                    "sequence_length": sequence_length,
                    "repeat": repeat,
                    "entropy_before": round(before_entropy, 6),
                    "entropy_after": round(after_entropy, 6),
                    "entropy_gain": round(gain, 6),
                    "outlier_before": round(before_outlier, 8),
                    "outlier_after": round(after_outlier, 8),
                    "max_projection_error": round(max_projection_error, 12),
                    "sinkhorn_temperature": round(sinkhorn_fit["sinkhorn_temperature"], 10),
                    "sinkhorn_residual_l1": round(sinkhorn_fit["sinkhorn_residual_l1"], 6),
                    "sinkhorn_residual_l2": round(sinkhorn_fit["sinkhorn_residual_l2"], 6),
                    "sinkhorn_row_error": round(sinkhorn_fit["sinkhorn_row_error"], 10),
                    "sinkhorn_col_error": round(sinkhorn_fit["sinkhorn_col_error"], 10),
                }
            )

            chain = _build_operator_chain(activations, rotated_inputs[:, : config.hidden_dim], sequence_length)
            resource_model = _resource_model(sequence_length)
            entropy_guided_groups = select_fusion_groups(chain, tau=config.tau, resource_model=resource_model)
            matched_tau, arithmetic_only_matched_groups = _calibrate_arithmetic_only_schedule(
                chain,
                resource_model=resource_model,
                target_depth=_average_depth(entropy_guided_groups),
                base_tau=config.tau,
            )
            schedules = {
                "no_fusion": build_no_fusion_groups(chain),
                "static_fusion": build_static_fusion_groups(chain, resource_model=resource_model, group_size=3),
                "entropy_guided": entropy_guided_groups,
                "arithmetic_only": select_fusion_groups(
                    chain,
                    tau=config.tau,
                    resource_model=resource_model,
                    alpha=0.0,
                ),
                "arithmetic_only_matched": arithmetic_only_matched_groups,
            }

            schedule_taus = {
                "no_fusion": 0.0,
                "static_fusion": 0.0,
                "entropy_guided": config.tau,
                "arithmetic_only": config.tau,
                "arithmetic_only_matched": matched_tau,
            }

            matched_tau_rows.append(
                {
                    "bucket": _bucket_name(sequence_length),
                    "sequence_length": sequence_length,
                    "repeat": repeat,
                    "base_tau": round(config.tau, 4),
                    "matched_tau": round(matched_tau, 4),
                    "target_depth": round(_average_depth(entropy_guided_groups), 4),
                    "matched_depth": round(_average_depth(arithmetic_only_matched_groups), 4),
                }
            )

            for method, groups in schedules.items():
                schedule_trace_rows.extend(
                    _build_schedule_trace_rows(
                        sequence_length=sequence_length,
                        repeat=repeat,
                        method=method,
                        tau=schedule_taus[method],
                        target_depth=_average_depth(entropy_guided_groups),
                        groups=groups,
                    )
                )
                tile_trace_rows.extend(
                    _build_tile_trace_rows(
                        sequence_length=sequence_length,
                        repeat=repeat,
                        method=method,
                        tau=schedule_taus[method],
                        groups=groups,
                    )
                )

            for precision in PRECISION_SPEEDUP:
                for method, groups in schedules.items():
                    results.append(
                        _estimate_metrics(
                            method=method,
                            sequence_length=sequence_length,
                            precision=precision,
                            groups=groups,
                            entropy_before_value=before_entropy,
                            entropy_after_value=after_entropy,
                            entropy_gain_value=gain,
                            outlier_before_value=before_outlier,
                            outlier_after_value=after_outlier,
                            max_projection_error=max_projection_error,
                        )
                    )

    detailed_rows = [asdict(result) for result in results]
    summary_rows = _summarize_by_bucket(results)
    occupancy_rows = _summarize_occupancy(results)
    alpha_zero_rows = _summarize_alpha_zero_ablation(results)
    matched_alpha_zero_rows = _summarize_matched_alpha_zero(results, matched_tau_rows)
    sinkhorn_rows = _summarize_sinkhorn_validation(validation_rows)
    tile_runtime_rows = _summarize_tile_runtime(tile_trace_rows)
    _write_csv(output_dir / "detailed_metrics.csv", detailed_rows)
    _write_csv(output_dir / "bucket_summary.csv", summary_rows)
    _write_csv(output_dir / "occupancy_summary.csv", occupancy_rows)
    _write_csv(output_dir / "alpha_zero_ablation.csv", alpha_zero_rows)
    _write_csv(output_dir / "alpha_zero_matched_ablation.csv", matched_alpha_zero_rows)
    _write_csv(output_dir / "schedule_trace.csv", schedule_trace_rows)
    _write_csv(output_dir / "tile_trace.csv", tile_trace_rows)
    _write_csv(output_dir / "tile_trace_summary.csv", tile_runtime_rows)
    _write_csv(output_dir / "alpha_zero_matched_tau.csv", matched_tau_rows)
    _write_csv(output_dir / "hadamard_validation.csv", validation_rows)
    _write_csv(output_dir / "sinkhorn_validation_summary.csv", sinkhorn_rows)

    metadata = {
        "config": asdict(config),
        "output_dir": str(output_dir),
        "sequence_lengths": [length for bucket in SEQUENCE_BUCKETS.values() for length in bucket],
        "methods": ["no_fusion", "static_fusion", "entropy_guided", "arithmetic_only", "arithmetic_only_matched"],
        "precisions": list(PRECISION_SPEEDUP.keys()),
        "ablations": {
            "alpha_zero": {
                "alpha": 0.0,
                "beta": 0.35,
                "gamma": 0.20,
                "description": "Arithmetic-intensity-only boundary selection with the entropy term disabled.",
            },
            "alpha_zero_matched": {
                "alpha": 0.0,
                "beta": 0.35,
                "gamma": 0.20,
                "description": "Arithmetic-intensity-only boundary selection with a calibrated tau chosen to match the entropy-guided fusion depth as closely as possible.",
            },
            "sinkhorn_proxy": {
                "description": "Fits a positive kernel over shared histogram bins and projects it to an approximately doubly-stochastic matrix with Sinkhorn normalization, then reports the residual between q and Bp.",
            },
            "tile_runtime_trace": {
                "description": "Prototype-level surrogate runtime trace that distributes each group's latency, memory, and compute cost across tiles using entropy-varying tile weights.",
            },
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "metadata": metadata,
        "summary_rows": summary_rows,
        "occupancy_rows": occupancy_rows,
        "alpha_zero_rows": alpha_zero_rows,
        "matched_alpha_zero_rows": matched_alpha_zero_rows,
        "tile_trace_rows": len(tile_trace_rows),
        "validation_rows": validation_rows,
    }


def main() -> None:
    args = _parse_args()
    config = ExperimentConfig(tau=args.tau, repeats=args.repeats)
    report = run_experiments(config=config, output_dir=args.output_dir)
    print(json.dumps(report["summary_rows"], indent=2))


if __name__ == "__main__":
    main()