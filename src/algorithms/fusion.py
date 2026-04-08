from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    entropy: float
    arithmetic_intensity: float
    memory_traffic: float
    register_cost: int
    shared_memory_cost: int
    occupancy: float


@dataclass(frozen=True)
class ResourceModel:
    max_registers: int
    max_shared_memory: int
    min_occupancy: float


@dataclass(frozen=True)
class FusionGroup:
    operators: tuple[OperatorSpec, ...]
    score: float
    feasible: bool

    @property
    def depth(self) -> int:
        return len(self.operators)

    @property
    def entropy(self) -> float:
        return mean(operator.entropy for operator in self.operators)

    @property
    def arithmetic_intensity(self) -> float:
        return sum(operator.arithmetic_intensity for operator in self.operators)

    @property
    def memory_traffic(self) -> float:
        return sum(operator.memory_traffic for operator in self.operators)


def _normalize(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.5
    return max(0.0, min(1.0, (value - lower) / (upper - lower)))


def _metric_bounds(chain: list[OperatorSpec], field_name: str) -> tuple[float, float]:
    values = [getattr(operator, field_name) for operator in chain]
    return min(values), max(values)


def _is_feasible(operators: tuple[OperatorSpec, ...], resource_model: ResourceModel) -> bool:
    register_cost = sum(operator.register_cost for operator in operators)
    shared_memory_cost = sum(operator.shared_memory_cost for operator in operators)
    occupancy = min(operator.occupancy for operator in operators)
    return (
        register_cost <= resource_model.max_registers
        and shared_memory_cost <= resource_model.max_shared_memory
        and occupancy >= resource_model.min_occupancy
    )


def estimate_fusion_score(
    chain: list[OperatorSpec],
    operators: tuple[OperatorSpec, ...],
    alpha: float = 0.45,
    beta: float = 0.35,
    gamma: float = 0.20,
) -> float:
    entropy_bounds = _metric_bounds(chain, "entropy")
    arithmetic_bounds = _metric_bounds(chain, "arithmetic_intensity")
    memory_bounds = _metric_bounds(chain, "memory_traffic")

    depth = len(operators)
    average_entropy = mean(operator.entropy for operator in operators)
    arithmetic_signal = sum(operator.arithmetic_intensity for operator in operators) / math.sqrt(depth)
    memory_signal = sum(operator.memory_traffic for operator in operators) / (depth ** 1.15)

    normalized_entropy = _normalize(average_entropy, *entropy_bounds)
    normalized_arithmetic = _normalize(arithmetic_signal, *arithmetic_bounds)
    normalized_memory = _normalize(memory_signal, *memory_bounds)
    depth_bonus = min(0.18 * (depth - 1), 0.36)
    return alpha * normalized_entropy + beta * normalized_arithmetic - gamma * normalized_memory + depth_bonus


def build_no_fusion_groups(chain: list[OperatorSpec]) -> list[FusionGroup]:
    return [
        FusionGroup(operators=(operator,), score=0.0, feasible=True)
        for operator in chain
    ]


def build_static_fusion_groups(
    chain: list[OperatorSpec],
    resource_model: ResourceModel,
    group_size: int = 3,
) -> list[FusionGroup]:
    groups: list[FusionGroup] = []
    start = 0
    while start < len(chain):
        end = min(start + group_size, len(chain))
        operators = tuple(chain[start:end])
        while len(operators) > 1 and not _is_feasible(operators, resource_model):
            operators = operators[:-1]
            end -= 1
        groups.append(
            FusionGroup(
                operators=operators,
                score=estimate_fusion_score(chain, operators),
                feasible=_is_feasible(operators, resource_model),
            )
        )
        start = end
    return groups


def select_fusion_groups(
    chain: list[OperatorSpec],
    tau: float,
    resource_model: ResourceModel,
    alpha: float = 0.45,
    beta: float = 0.35,
    gamma: float = 0.20,
) -> list[FusionGroup]:
    groups: list[FusionGroup] = []
    current_group: list[OperatorSpec] = []

    for operator in chain:
        candidate = tuple(current_group + [operator])
        score = estimate_fusion_score(chain, candidate, alpha=alpha, beta=beta, gamma=gamma)
        feasible = _is_feasible(candidate, resource_model)
        if score > tau and feasible:
            current_group.append(operator)
            continue

        if current_group:
            committed = tuple(current_group)
            groups.append(
                FusionGroup(
                    operators=committed,
                    score=estimate_fusion_score(chain, committed, alpha=alpha, beta=beta, gamma=gamma),
                    feasible=_is_feasible(committed, resource_model),
                )
            )
        current_group = [operator]

    if current_group:
        committed = tuple(current_group)
        groups.append(
            FusionGroup(
                operators=committed,
                score=estimate_fusion_score(chain, committed, alpha=alpha, beta=beta, gamma=gamma),
                feasible=_is_feasible(committed, resource_model),
            )
        )

    return groups