"""
Entropy-Guided Operator Fusion for SSM blocks.

This module implements memory-aware, entropy-guided fusion of the key operators
in the SSM (Mamba-series) computation: linear projections, the selective scan
state update, and activation functions.

The fusion strategy follows three tiers:
  1. No fusion  – each operator executes as a separate kernel (baseline).
  2. Basic fusion ("Fuse-All") – fuse all operators in the SSM block
     unconditionally, converting the bottleneck from memory-bound to
     compute-bound.
  3. Entropy-guided fusion – use per-layer entropy measurements to determine
     fusion boundaries dynamically.  High-entropy regions are fused
     aggressively; low-entropy (outlier-rich) regions are either preceded by a
     Hadamard rotation or broken into smaller fused groups.

Reference: "Entropy-Guided Operator Fusion with Hadamard Transform for SSM
           Acceleration" (this work).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hadamard import FusedHadamardLinear, fast_hadamard_transform
from .entropy import (
    activation_entropy,
    entropy_delta,
    fusion_boundary_selector,
    recommend_tile_size,
    EntropyProfiler,
)


# ---------------------------------------------------------------------------
# Minimal SSM block (Mamba-style, simplified for exposition)
# ---------------------------------------------------------------------------


class SSMBlock(nn.Module):
    """
    Simplified Mamba-style SSM block without operator fusion (baseline).

    Computes:
        z  = SiLU(x @ W_z + b_z)            # gate
        x' = x @ W_x + b_x                  # input projection
        y  = selective_scan(x', A, B, C, D)  # state-space recurrence
        out = y * z                          # gated output

    The selective scan is represented here by a single-layer GRU cell for
    simplicity; in production this would be replaced by a Triton/CUDA kernel.

    Args:
        d_model: Model / channel dimension.
        d_state: SSM state dimension.
        expand: Expansion factor for inner dimension.
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=4, padding=3, groups=self.d_inner
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # SSM parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(
            self.d_inner, -1
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            out: (B, L, d_model)
        """
        B, L, _ = x.shape
        xz = self.in_proj(x)                          # (B, L, 2 * d_inner)
        x_in, z = xz.chunk(2, dim=-1)                 # each (B, L, d_inner)

        # Depthwise conv along sequence dimension
        x_in = x_in.transpose(1, 2)                   # (B, d_inner, L)
        x_in = self.conv1d(x_in)[..., :L]             # causal padding
        x_in = x_in.transpose(1, 2)                   # (B, L, d_inner)
        x_in = F.silu(x_in)

        # SSM parameters from input
        xBC = self.x_proj(x_in)                       # (B, L, d_state*2 + 1)
        dt, B_ssm, C_ssm = torch.split(
            xBC, [1, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))             # (B, L, d_inner)
        A = -torch.exp(self.A_log.float())             # (d_inner, d_state)

        # Simplified scan (parallel prefix; real implementation uses Triton)
        y = self._simple_scan(x_in, dt, A, B_ssm, C_ssm)

        # Gated output
        out = y * F.silu(z)
        out = self.out_proj(out)
        return out

    def _simple_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplified sequential scan for illustration purposes.

        In a production system this would be replaced by the cuda_ssm or
        triton_selective_scan kernel.

        Args:
            x: (B, L, d_inner)
            dt: (B, L, d_inner)
            A: (d_inner, d_state)
            B: (B, L, d_state)
            C: (B, L, d_state)

        Returns:
            y: (B, L, d_inner)
        """
        B_batch, L, d_inner = x.shape
        d_state = A.shape[-1]
        h = x.new_zeros(B_batch, d_inner, d_state)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t, :].unsqueeze(-1) * A.unsqueeze(0))  # (B, d_inner, d_state)
            dB = dt[:, t, :].unsqueeze(-1) * B[:, t, :].unsqueeze(1)   # (B, d_inner, d_state)
            h = dA * h + dB * x[:, t, :].unsqueeze(-1)
            y_t = (h * C[:, t, :].unsqueeze(1)).sum(-1)                 # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x
        return y


# ---------------------------------------------------------------------------
# Entropy-guided fused SSM block
# ---------------------------------------------------------------------------


class EntropyGuidedSSMBlock(nn.Module):
    """
    SSM block with entropy-guided operator fusion and optional Hadamard
    rotation before the output projection to suppress outliers.

    At initialization the block runs a calibration pass on sample activations
    to determine:
      1. Whether the output tensor y contains outliers (measured via kurtosis
         and outlier ratio).
      2. Whether inserting a Hadamard rotation before `out_proj` increases
         entropy (measured via ΔH).
      3. Whether all operators in the block should be fused into a single
         compute-bound kernel.

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension.
        expand: Inner dimension expansion factor.
        entropy_threshold: Minimum activation entropy to allow aggressive fusion.
        use_hadamard: If True, always insert Hadamard rotation regardless of
                      the calibration result.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        entropy_threshold: float = 5.0,
        use_hadamard: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.entropy_threshold = entropy_threshold
        self.use_hadamard = use_hadamard

        # --- Input projection (with fused Hadamard for gate path) ---
        self.in_proj = FusedHadamardLinear(d_model, self.d_inner * 2, bias=False)

        # --- Depthwise conv ---
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=4, padding=3, groups=self.d_inner
        )

        # --- SSM parameter projections ---
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # --- Output projection with optional Hadamard pre-rotation ---
        if use_hadamard:
            self.out_proj = FusedHadamardLinear(self.d_inner, d_model, bias=False)
        else:
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # SSM parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(
            self.d_inner, -1
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Runtime entropy / fusion metadata (populated during calibration)
        self._last_entropy_before_out: Optional[float] = None
        self._last_entropy_after_out: Optional[float] = None
        self._fuse_all: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            out: (B, L, d_model)
        """
        B, L, _ = x.shape

        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        x_in = x_in.transpose(1, 2)
        x_in = self.conv1d(x_in)[..., :L]
        x_in = x_in.transpose(1, 2)
        x_in = F.silu(x_in)

        xBC = self.x_proj(x_in)
        dt, B_ssm, C_ssm = torch.split(
            xBC, [1, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())

        y = self._simple_scan(x_in, dt, A, B_ssm, C_ssm)

        # Hadamard rotation before output projection
        if self.use_hadamard:
            y = fast_hadamard_transform(y)

        gated = y * F.silu(z)
        out = self.out_proj(gated)
        return out

    def _simple_scan(self, x, dt, A, B, C):
        """Identical simplified scan as in SSMBlock."""
        B_batch, L, d_inner = x.shape
        d_state = A.shape[-1]
        h = x.new_zeros(B_batch, d_inner, d_state)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t, :].unsqueeze(-1) * A.unsqueeze(0))
            dB = dt[:, t, :].unsqueeze(-1) * B[:, t, :].unsqueeze(1)
            h = dA * h + dB * x[:, t, :].unsqueeze(-1)
            y_t = (h * C[:, t, :].unsqueeze(1)).sum(-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x
        return y

    @torch.no_grad()
    def calibrate(self, x_sample: torch.Tensor) -> dict:
        """
        Run a calibration forward pass to measure entropy metrics and
        determine whether Hadamard rotation is beneficial.

        Args:
            x_sample: Representative input tensor (B, L, d_model).

        Returns:
            dict with keys:
              - 'entropy_before_hadamard': entropy of y before rotation
              - 'entropy_after_hadamard': entropy of y after rotation
              - 'delta_entropy': ΔH = H_after - H_before
              - 'recommended_tile': tile size recommendation
              - 'use_hadamard': whether Hadamard is recommended
        """
        B, L, _ = x_sample.shape
        xz = self.in_proj(x_sample)
        x_in, z = xz.chunk(2, dim=-1)
        x_in_t = x_in.transpose(1, 2)
        x_in_t = self.conv1d(x_in_t)[..., :L]
        x_in_t = x_in_t.transpose(1, 2)
        x_in_t = F.silu(x_in_t)
        xBC = self.x_proj(x_in_t)
        dt, B_ssm, C_ssm = torch.split(xBC, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        y = self._simple_scan(x_in_t, dt, A, B_ssm, C_ssm)

        h_before = activation_entropy(y.float()).item()
        y_rot = fast_hadamard_transform(y.float())
        h_after = activation_entropy(y_rot).item()
        delta = h_after - h_before

        self._last_entropy_before_out = h_before
        self._last_entropy_after_out = h_after
        tile = recommend_tile_size(h_after)

        result = {
            "entropy_before_hadamard": h_before,
            "entropy_after_hadamard": h_after,
            "delta_entropy": delta,
            "recommended_tile": tile,
            "use_hadamard": delta > 0,
        }
        return result


# ---------------------------------------------------------------------------
# Fusion scheduler
# ---------------------------------------------------------------------------


class FusionScheduler:
    """
    Determines the optimal fusion plan for a sequence of SSM blocks based on
    their per-layer entropy profiles.

    Usage::

        scheduler = FusionScheduler(entropy_threshold=5.0)
        profiler = EntropyProfiler()
        handles = profiler.register_hooks(model)
        with torch.no_grad():
            _ = model(x_calibration)
        profiler.remove_hooks(handles)
        plan = scheduler.plan(profiler.entropy_sequence())

    Args:
        entropy_threshold: Entropy (bits) below which a layer boundary is
                           introduced.
        max_fused_layers: Maximum consecutive layers in one fused kernel.
    """

    def __init__(self, entropy_threshold: float = 5.0, max_fused_layers: int = 8):
        self.entropy_threshold = entropy_threshold
        self.max_fused_layers = max_fused_layers

    def plan(self, entropy_sequence: List[float]) -> dict:
        """
        Given per-layer entropies, produce a fusion plan.

        Returns:
            dict with keys:
              - 'groups': list of layer-index lists to fuse together
              - 'num_kernels': total number of compiled kernels
              - 'fusion_ratio': fraction of layers included in multi-layer groups
        """
        groups = fusion_boundary_selector(
            entropy_sequence,
            entropy_threshold=self.entropy_threshold,
            max_fused_layers=self.max_fused_layers,
        )
        n_total = len(entropy_sequence)
        n_fused = sum(len(g) for g in groups if len(g) > 1)
        return {
            "groups": groups,
            "num_kernels": len(groups),
            "fusion_ratio": n_fused / max(n_total, 1),
        }
