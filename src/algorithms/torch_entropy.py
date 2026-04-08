from __future__ import annotations

from typing import Iterable

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "torch_entropy requires PyTorch. Install torch before importing this module."
    ) from exc


def activation_entropy(
    values: torch.Tensor,
    num_bins: int = 256,
    per_channel: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    if per_channel:
        channel_count = values.shape[-1]
        flattened = values.reshape(-1, channel_count)
        return torch.stack([_hist_entropy(flattened[:, index], num_bins, eps) for index in range(channel_count)])
    return _hist_entropy(values.reshape(-1), num_bins, eps)


def _hist_entropy(values: torch.Tensor, num_bins: int, eps: float) -> torch.Tensor:
    value_min = values.min()
    value_max = values.max()
    if (value_max - value_min).abs() < eps:
        return torch.tensor(0.0, device=values.device, dtype=torch.float32)

    normalized = (values - value_min) / (value_max - value_min + eps)
    indices = (normalized * num_bins).long().clamp(0, num_bins - 1)
    counts = torch.zeros(num_bins, device=values.device, dtype=torch.float32)
    counts.scatter_add_(0, indices, torch.ones_like(values, dtype=torch.float32))
    probabilities = counts / (counts.sum() + eps)
    log_probabilities = torch.where(
        probabilities > eps,
        torch.log2(probabilities + eps),
        torch.zeros_like(probabilities),
    )
    return -(probabilities * log_probabilities).sum()


def entropy_delta(
    before: torch.Tensor,
    after: torch.Tensor,
    num_bins: int = 256,
    per_channel: bool = False,
) -> torch.Tensor:
    return activation_entropy(after, num_bins=num_bins, per_channel=per_channel) - activation_entropy(
        before,
        num_bins=num_bins,
        per_channel=per_channel,
    )


def outlier_ratio(values: torch.Tensor, sigma_threshold: float = 3.0) -> torch.Tensor:
    center = values.mean()
    scale = values.std().clamp(min=1e-8)
    return ((values - center).abs() > sigma_threshold * scale).float().mean()


def kurtosis(values: torch.Tensor) -> torch.Tensor:
    flattened = values.reshape(-1).float()
    standardized = (flattened - flattened.mean()) / flattened.std().clamp(min=1e-8)
    return standardized.pow(4).mean() - 3.0


def fusion_boundary_selector(
    layer_entropies: Iterable[float],
    entropy_threshold: float = 5.0,
    max_fused_layers: int = 8,
) -> list[list[int]]:
    groups: list[list[int]] = []
    current_group: list[int] = []

    for index, entropy in enumerate(layer_entropies):
        if entropy >= entropy_threshold and len(current_group) < max_fused_layers:
            current_group.append(index)
            continue
        if current_group:
            groups.append(current_group)
            current_group = []
        groups.append([index])

    if current_group:
        groups.append(current_group)
    return groups


def recommend_tile_size(
    entropy: float,
    base_tile: int = 64,
    max_tile: int = 512,
    entropy_max: float = 8.0,
) -> int:
    normalized = min(max(entropy / entropy_max, 0.0), 1.0)
    suggested = base_tile + normalized * (max_tile - base_tile)
    return int(round(suggested / 32.0) * 32)


class EntropyProfiler:
    def __init__(self, num_bins: int = 256):
        self.num_bins = num_bins
        self.records: dict[str, float] = {}

    def record(self, name: str, values: torch.Tensor) -> float:
        entropy = float(activation_entropy(values, num_bins=self.num_bins).item())
        self.records[name] = entropy
        return entropy

    def summary(self) -> dict[str, float]:
        return dict(self.records)