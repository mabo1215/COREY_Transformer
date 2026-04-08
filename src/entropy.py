"""
Entropy Measurement utilities for guiding operator fusion in SSM models.

The core idea is to measure the Shannon entropy of an activation tensor's
distribution before and after applying the Hadamard transform.  High entropy
indicates a more uniform distribution (outliers suppressed), signalling that:
  1. The Hadamard transform has been effective at redistributing activation energy.
  2. The tensor is now suitable for aggressive operator fusion and fine-grained
     quantization without accuracy loss.

Key functions:
  - activation_entropy: compute per-channel or per-tensor entropy.
  - entropy_delta: measure the change in entropy induced by a transform.
  - fusion_boundary_selector: given a sequence of layer entropies, return the
    optimal fusion boundary (the set of layers to fuse into a single kernel).

Reference: "Entropy-Guided Operator Fusion with Hadamard Transform for SSM
           Acceleration" (this work).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core entropy computation
# ---------------------------------------------------------------------------


def activation_entropy(
    x: torch.Tensor,
    num_bins: int = 256,
    per_channel: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Estimate the Shannon entropy (in bits) of an activation tensor.

    Entropy is computed from a histogram approximation of the marginal
    distribution of activation values.

    Args:
        x: Activation tensor of arbitrary shape.
        num_bins: Number of histogram bins used to approximate the distribution.
        per_channel: If True, compute entropy independently for each channel
                     (last dimension).  Returns a 1-D tensor of length C.
                     If False, treat the whole tensor as a single distribution
                     and return a scalar tensor.
        eps: Small constant to avoid log(0).

    Returns:
        Scalar (per_channel=False) or 1-D tensor of shape (C,) (per_channel=True)
        representing Shannon entropy in bits.
    """
    if per_channel:
        C = x.shape[-1]
        x_flat = x.reshape(-1, C)  # (N, C)
        entropies = []
        for c in range(C):
            entropies.append(_hist_entropy(x_flat[:, c], num_bins, eps))
        return torch.stack(entropies)
    else:
        return _hist_entropy(x.flatten(), num_bins, eps)


def _hist_entropy(values: torch.Tensor, num_bins: int, eps: float) -> torch.Tensor:
    """
    Compute the Shannon entropy of a 1-D tensor via histogram binning.

    H(X) = -sum_i p_i * log2(p_i)

    Args:
        values: 1-D float tensor.
        num_bins: Number of histogram bins.
        eps: Smoothing constant.

    Returns:
        Scalar tensor representing entropy in bits.
    """
    v_min = values.min()
    v_max = values.max()

    if (v_max - v_min).abs() < eps:
        # Degenerate distribution: all mass in one bin => entropy = 0
        return torch.tensor(0.0, device=values.device, dtype=torch.float32)

    # Normalize to [0, 1], scale to [0, num_bins), clamp to valid bin indices
    normed = (values - v_min) / (v_max - v_min + eps)
    indices = (normed * num_bins).long().clamp(0, num_bins - 1)

    counts = torch.zeros(num_bins, device=values.device, dtype=torch.float32)
    counts.scatter_add_(0, indices, torch.ones_like(values, dtype=torch.float32))

    probs = counts / (counts.sum() + eps)
    # Shannon entropy: H = -sum p * log2(p), ignoring zero-probability bins
    log_probs = torch.where(probs > eps, torch.log2(probs + eps), torch.zeros_like(probs))
    entropy = -(probs * log_probs).sum()
    return entropy


# ---------------------------------------------------------------------------
# Entropy delta (before / after transform)
# ---------------------------------------------------------------------------


def entropy_delta(
    x_before: torch.Tensor,
    x_after: torch.Tensor,
    num_bins: int = 256,
    per_channel: bool = False,
) -> torch.Tensor:
    """
    Compute the change in entropy: ΔH = H(x_after) - H(x_before).

    A positive ΔH indicates that the transform increased entropy (more uniform
    distribution), which is the desired effect of the Hadamard rotation.

    Args:
        x_before: Activation tensor before the transform.
        x_after: Activation tensor after the transform.
        num_bins: Histogram bins.
        per_channel: Whether to compute per-channel entropy.

    Returns:
        Scalar or 1-D tensor of entropy deltas (in bits).
    """
    h_before = activation_entropy(x_before, num_bins=num_bins, per_channel=per_channel)
    h_after = activation_entropy(x_after, num_bins=num_bins, per_channel=per_channel)
    return h_after - h_before


# ---------------------------------------------------------------------------
# Outlier metrics
# ---------------------------------------------------------------------------


def outlier_ratio(x: torch.Tensor, sigma_threshold: float = 3.0) -> torch.Tensor:
    """
    Compute the fraction of activation values that are considered outliers.

    A value is an outlier if |x_i - mean(x)| > sigma_threshold * std(x).

    Args:
        x: Activation tensor.
        sigma_threshold: Number of standard deviations defining the outlier
                         boundary.

    Returns:
        Scalar tensor in [0, 1] representing the outlier fraction.
    """
    mean = x.mean()
    std = x.std().clamp(min=1e-8)
    outliers = ((x - mean).abs() > sigma_threshold * std).float()
    return outliers.mean()


def kurtosis(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the excess kurtosis of the activation distribution.

    High kurtosis (> 0) indicates heavy tails (outlier-prone distributions).
    After a successful Hadamard rotation the kurtosis should approach 0
    (Gaussian-like, mesokurtic distribution).

    Args:
        x: Activation tensor (any shape, flattened internally).

    Returns:
        Scalar tensor of excess kurtosis.
    """
    x_flat = x.flatten().float()
    mean = x_flat.mean()
    std = x_flat.std().clamp(min=1e-8)
    z = (x_flat - mean) / std
    excess_kurtosis = z.pow(4).mean() - 3.0
    return excess_kurtosis


# ---------------------------------------------------------------------------
# Fusion boundary selection
# ---------------------------------------------------------------------------


def fusion_boundary_selector(
    layer_entropies: List[float],
    entropy_threshold: float = 5.0,
    max_fused_layers: int = 8,
) -> List[List[int]]:
    """
    Given per-layer entropy values, select which contiguous layers should be
    fused into a single kernel.

    Fusion strategy:
      - Start a new fusion group when entropy drops below `entropy_threshold`
        (the distribution is no longer well-conditioned for fusion), or when the
        current group already contains `max_fused_layers` layers.
      - Layers with entropy >= threshold are considered "high-entropy" and safe
        to fuse aggressively.

    Args:
        layer_entropies: List of scalar entropy values, one per layer in the
                         computation graph (in execution order).
        entropy_threshold: Minimum entropy (bits) required to include a layer in
                           the current fusion group.
        max_fused_layers: Maximum number of consecutive layers in one fused group.

    Returns:
        A list of fusion groups, where each group is a list of 0-based layer
        indices that should be compiled into the same kernel.

    Example:
        >>> entropies = [6.1, 6.3, 5.8, 3.2, 6.7, 6.5]
        >>> fusion_boundary_selector(entropies, entropy_threshold=5.0)
        [[0, 1, 2], [3], [4, 5]]
    """
    groups: List[List[int]] = []
    current_group: List[int] = []

    for i, h in enumerate(layer_entropies):
        if h >= entropy_threshold and len(current_group) < max_fused_layers:
            current_group.append(i)
        else:
            # Flush the current high-entropy group (if any)
            if current_group:
                groups.append(current_group)
                current_group = []
            # Low-entropy (or max-size) layers always become their own solo group
            groups.append([i])

    if current_group:
        groups.append(current_group)

    return groups


# ---------------------------------------------------------------------------
# Entropy-based tiling size recommendation
# ---------------------------------------------------------------------------


def recommend_tile_size(
    entropy: float,
    base_tile: int = 64,
    max_tile: int = 512,
    entropy_max: float = 8.0,
) -> int:
    """
    Recommend a CUDA/Triton tiling block size based on entropy.

    Higher entropy ⟹ more uniform distribution ⟹ less spill / irregular access ⟹
    larger tile sizes are safe.  The recommended tile size scales linearly with
    entropy up to `max_tile`.

    Args:
        entropy: Shannon entropy of the activation tensor (bits).
        base_tile: Minimum tile size (used when entropy = 0).
        max_tile: Maximum tile size (used when entropy >= entropy_max).
        entropy_max: Entropy value at which the maximum tile size is reached.

    Returns:
        Recommended tile size (rounded down to nearest power of 2).
    """
    ratio = min(entropy / entropy_max, 1.0)
    raw = base_tile + ratio * (max_tile - base_tile)
    # Round down to nearest power of 2
    tile = 2 ** int(math.log2(raw))
    return int(tile)


# ---------------------------------------------------------------------------
# Profiling hook
# ---------------------------------------------------------------------------


class EntropyProfiler:
    """
    A context-manager / hook registry that records the entropy of intermediate
    activation tensors during a forward pass.

    Usage::

        profiler = EntropyProfiler()
        handles = profiler.register_hooks(model)
        with torch.no_grad():
            _ = model(x)
        profiler.remove_hooks(handles)
        print(profiler.records)

    After collecting entropy records, pass `profiler.entropy_sequence()` to
    `fusion_boundary_selector` to determine optimal fusion boundaries.
    """

    def __init__(self, num_bins: int = 256):
        self.num_bins = num_bins
        self.records: List[Tuple[str, float]] = []  # (layer_name, entropy)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                h = activation_entropy(output.detach().float(), self.num_bins).item()
                self.records.append((name, h))
        return hook

    def register_hooks(self, model: torch.nn.Module):
        """
        Register forward hooks on all leaf modules.

        Returns:
            List of hook handles (pass to `remove_hooks` for cleanup).
        """
        handles = []
        for name, module in model.named_modules():
            if not list(module.children()):  # leaf module
                h = module.register_forward_hook(self._make_hook(name))
                handles.append(h)
        return handles

    @staticmethod
    def remove_hooks(handles) -> None:
        for h in handles:
            h.remove()

    def entropy_sequence(self) -> List[float]:
        """Return only the entropy values in recording order."""
        return [h for _, h in self.records]

    def reset(self) -> None:
        self.records.clear()
