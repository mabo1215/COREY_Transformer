from __future__ import annotations

import math

import numpy as np


def _next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def _pad_last_dimension(array: np.ndarray, target_size: int | None = None) -> tuple[np.ndarray, int]:
    matrix = np.asarray(array, dtype=np.float64)
    original_size = matrix.shape[-1]
    padded_size = target_size or _next_power_of_two(original_size)
    if padded_size < original_size:
        raise ValueError("target_size must be at least the original size")
    if padded_size == original_size:
        return matrix.copy(), original_size

    padding_width = [(0, 0)] * matrix.ndim
    padding_width[-1] = (0, padded_size - original_size)
    padded = np.pad(matrix, padding_width, mode="constant")
    return padded, original_size


def normalized_hadamard_transform(array: np.ndarray) -> np.ndarray:
    padded, _ = _pad_last_dimension(array)
    last_dimension = padded.shape[-1]
    transformed = padded.reshape(-1, last_dimension).copy()

    block_size = 1
    while block_size < last_dimension:
        step = block_size * 2
        for start in range(0, last_dimension, step):
            left = transformed[:, start : start + block_size].copy()
            right = transformed[:, start + block_size : start + step].copy()
            transformed[:, start : start + block_size] = left + right
            transformed[:, start + block_size : start + step] = left - right
        block_size = step

    transformed /= math.sqrt(last_dimension)
    return transformed.reshape(padded.shape)


def reparameterize_weight(weight: np.ndarray) -> np.ndarray:
    padded_weight, _ = _pad_last_dimension(weight)
    return normalized_hadamard_transform(padded_weight)


def reference_projection(inputs: np.ndarray, weight: np.ndarray) -> np.ndarray:
    padded_inputs, _ = _pad_last_dimension(inputs)
    padded_weight, _ = _pad_last_dimension(weight, target_size=padded_inputs.shape[-1])
    return padded_inputs @ padded_weight.T


def fused_hadamard_projection(inputs: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    padded_inputs, _ = _pad_last_dimension(inputs)
    padded_weight, _ = _pad_last_dimension(weight, target_size=padded_inputs.shape[-1])
    rotated_inputs = normalized_hadamard_transform(padded_inputs)
    rotated_weight = normalized_hadamard_transform(padded_weight)
    projected = rotated_inputs @ rotated_weight.T
    return projected, rotated_inputs, rotated_weight


def outlier_ratio(values: np.ndarray, sigma_threshold: float = 6.0) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return 0.0
    sigma = float(np.std(array))
    if sigma == 0:
        return 0.0
    return float(np.mean(np.abs(array) > sigma_threshold * sigma))