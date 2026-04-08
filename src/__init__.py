"""
ADAMA Transformer – src package init.

Exposes the core modules for entropy-guided operator fusion with Hadamard
transform in SSM (Mamba-series) models.
"""

from .hadamard import (
    hadamard_matrix,
    fast_hadamard_transform,
    inverse_fast_hadamard_transform,
    FusedHadamardLinear,
    HadamardRotation,
)

from .entropy import (
    activation_entropy,
    entropy_delta,
    outlier_ratio,
    kurtosis,
    fusion_boundary_selector,
    recommend_tile_size,
    EntropyProfiler,
)

from .fused_ops import (
    SSMBlock,
    EntropyGuidedSSMBlock,
    FusionScheduler,
)

__all__ = [
    # Hadamard
    "hadamard_matrix",
    "fast_hadamard_transform",
    "inverse_fast_hadamard_transform",
    "FusedHadamardLinear",
    "HadamardRotation",
    # Entropy
    "activation_entropy",
    "entropy_delta",
    "outlier_ratio",
    "kurtosis",
    "fusion_boundary_selector",
    "recommend_tile_size",
    "EntropyProfiler",
    # Fused ops
    "SSMBlock",
    "EntropyGuidedSSMBlock",
    "FusionScheduler",
]
