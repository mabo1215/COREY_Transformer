"""
Fused Hadamard Layer for SSM (Mamba-series) Operator Fusion.

This module implements the Hadamard transform and its inverse, fused with
linear projection layers to suppress outliers in SSM activations.
The Hadamard transform redistributes activation energy across all dimensions,
converting heavy-tailed, outlier-dominated distributions into smoother ones
that are amenable to fine-grained quantization and operator fusion.

Reference: "Entropy-Guided Operator Fusion with Hadamard Transform for SSM
           Acceleration" (this work).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def hadamard_matrix(n: int, device: torch.device = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Construct the normalized Hadamard matrix of order n (n must be a power of 2).

    H_n satisfies: H_n @ H_n^T = n * I, so the normalized version H_n / sqrt(n)
    is orthogonal.

    Args:
        n: Matrix size (must be a power of 2).
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tensor of shape (n, n) representing the normalized Hadamard matrix.
    """
    if n == 1:
        return torch.ones(1, 1, device=device, dtype=dtype)
    if n & (n - 1) != 0:
        raise ValueError(f"n must be a power of 2, got {n}")

    H = torch.ones(1, 1, device=device, dtype=dtype)
    base = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=dtype)
    log2n = int(math.log2(n))
    for _ in range(log2n):
        H = torch.kron(H, base)

    return H / math.sqrt(n)


def fast_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Apply the Fast Walsh-Hadamard Transform (FWHT) in-place along the last
    dimension of x, with O(n log n) complexity.

    The last dimension must be a power of 2.

    Args:
        x: Input tensor of shape (..., n).

    Returns:
        Transformed tensor of shape (..., n), normalized by 1/sqrt(n).
    """
    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError(f"Last dimension must be a power of 2, got {n}")

    x = x.clone()
    h = 1
    while h < n:
        x_view = x.view(*x.shape[:-1], n // (2 * h), 2 * h)
        # Clone before assignment to avoid aliasing: a and b are views of x_view,
        # so modifying x_view[..., :h] in-place would corrupt the value of a
        # before x_view[..., h:] is computed.
        a = x_view[..., :h].clone()
        b = x_view[..., h:].clone()
        x_view[..., :h] = a + b
        x_view[..., h:] = a - b
        h *= 2

    return x / math.sqrt(n)


def inverse_fast_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Apply the inverse Fast Walsh-Hadamard Transform.

    For the normalized Hadamard transform, H^{-1} = H^T = H (self-inverse up to
    scaling), so the inverse transform is identical to the forward transform.

    Args:
        x: Input tensor of shape (..., n).

    Returns:
        Reconstructed tensor of shape (..., n).
    """
    return fast_hadamard_transform(x)


class FusedHadamardLinear(nn.Module):
    """
    A linear projection layer with fused Hadamard pre- and post-rotations.

    Given a weight matrix W, the fused layer computes:
        y = H_out @ (W @ (H_in^T @ x)) + b
          = (H_out @ W @ H_in^T) @ x + b
          = W_fused @ x + b

    where H_in and H_out are Hadamard rotation matrices. This is mathematically
    equivalent to the original linear layer but applied in the Hadamard-rotated
    feature space, which suppresses activation outliers and enables more
    aggressive quantization.

    In practice, the rotation can be *pre-absorbed* into the weight matrix W
    at initialization, so inference cost is identical to a standard linear layer.

    Args:
        in_features: Input feature dimension (must be power of 2).
        out_features: Output feature dimension (must be power of 2).
        bias: Whether to include a bias term.
        absorb_rotation: If True, pre-compute W_fused = H_out @ W @ H_in^T and
                         store it as the effective weight (zero runtime overhead).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        absorb_rotation: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.absorb_rotation = absorb_rotation

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if not absorb_rotation:
            H_in = hadamard_matrix(in_features)
            H_out = hadamard_matrix(out_features)
            self.register_buffer("H_in", H_in)
            self.register_buffer("H_out", H_out)

    def fuse_hadamard_into_weight(self) -> None:
        """
        Pre-absorb the Hadamard rotations into the weight matrix.
        After calling this method the layer behaves identically to a standard
        nn.Linear with the rotated weight.
        """
        with torch.no_grad():
            H_in = hadamard_matrix(
                self.in_features, device=self.weight.device, dtype=self.weight.dtype
            )
            H_out = hadamard_matrix(
                self.out_features, device=self.weight.device, dtype=self.weight.dtype
            )
            # W_fused = H_out @ W @ H_in^T  (H^T = H for normalized Hadamard)
            self.weight.data = H_out @ self.weight.data @ H_in.T
        self.absorb_rotation = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.absorb_rotation:
            return F.linear(x, self.weight, self.bias)
        else:
            # Runtime application: rotate input, apply weight, rotate output
            x_rot = fast_hadamard_transform(x)
            out = F.linear(x_rot, self.weight, self.bias)
            return fast_hadamard_transform(out)


class HadamardRotation(nn.Module):
    """
    A standalone Hadamard rotation module that can be inserted between
    arbitrary layers in an SSM block.

    This is useful during profiling / entropy measurement: inserting this
    module lets us measure the entropy change induced by the Hadamard transform
    without restructuring the surrounding computation graph.

    Args:
        dim: Feature dimension (must be power of 2).
        inverse: If True, apply the inverse (identical for normalized Hadamard).
    """

    def __init__(self, dim: int, inverse: bool = False):
        super().__init__()
        self.dim = dim
        self.inverse = inverse
        if dim & (dim - 1) != 0:
            raise ValueError(f"dim must be a power of 2, got {dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inverse:
            return inverse_fast_hadamard_transform(x)
        return fast_hadamard_transform(x)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, inverse={self.inverse}"
