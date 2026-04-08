from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def histogram_entropy(
    values: np.ndarray,
    bins: int = 64,
    value_range: tuple[float, float] | None = None,
    epsilon: float = 1e-12,
) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return 0.0

    if value_range is None:
        minimum = float(np.min(array))
        maximum = float(np.max(array))
        if np.isclose(minimum, maximum):
            return 0.0
        margin = max((maximum - minimum) * 0.05, 1e-6)
        value_range = (minimum - margin, maximum + margin)

    histogram, _ = np.histogram(array, bins=bins, range=value_range)
    total = int(np.sum(histogram))
    if total == 0:
        return 0.0

    probabilities = histogram.astype(np.float64) / total
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log(probabilities + epsilon)).sum())


def normalized_entropy(values: np.ndarray, bins: int = 64) -> float:
    entropy = histogram_entropy(values, bins=bins)
    normalizer = np.log(max(bins, 2))
    if normalizer == 0:
        return 0.0
    return float(np.clip(entropy / normalizer, 0.0, 1.0))


def entropy_gain(before: np.ndarray, after: np.ndarray, bins: int = 64) -> float:
    return normalized_entropy(after, bins=bins) - normalized_entropy(before, bins=bins)


@dataclass
class ExponentialMovingEntropy:
    decay: float = 0.9
    bins: int = 64
    value: float | None = None

    def update(self, values: np.ndarray) -> float:
        current = normalized_entropy(values, bins=self.bins)
        if self.value is None:
            self.value = current
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * current
        return self.value