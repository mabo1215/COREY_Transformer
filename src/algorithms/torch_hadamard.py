from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "torch_hadamard requires PyTorch. Install torch before importing this module."
    ) from exc


def hadamard_matrix(
    size: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if size == 1:
        return torch.ones(1, 1, device=device, dtype=dtype)
    if size & (size - 1) != 0:
        raise ValueError(f"size must be a power of two, got {size}")

    matrix = torch.ones(1, 1, device=device, dtype=dtype)
    base = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=dtype)
    for _ in range(int(math.log2(size))):
        matrix = torch.kron(matrix, base)
    return matrix / math.sqrt(size)


def fast_hadamard_transform(values: torch.Tensor) -> torch.Tensor:
    size = values.shape[-1]
    if size & (size - 1) != 0:
        raise ValueError(f"Last dimension must be a power of two, got {size}")

    transformed = values.clone()
    step = 1
    while step < size:
        view = transformed.view(*transformed.shape[:-1], size // (2 * step), 2 * step)
        left = view[..., :step].clone()
        right = view[..., step:].clone()
        view[..., :step] = left + right
        view[..., step:] = left - right
        step *= 2
    return transformed / math.sqrt(size)


def inverse_fast_hadamard_transform(values: torch.Tensor) -> torch.Tensor:
    return fast_hadamard_transform(values)


class FusedHadamardLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        absorb_rotation: bool = True,
    ) -> None:
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
            self.register_buffer("hadamard_in", hadamard_matrix(in_features))
            self.register_buffer("hadamard_out", hadamard_matrix(out_features))

    def fuse_hadamard_into_weight(self) -> None:
        with torch.no_grad():
            hadamard_in = hadamard_matrix(self.in_features, device=self.weight.device, dtype=self.weight.dtype)
            hadamard_out = hadamard_matrix(self.out_features, device=self.weight.device, dtype=self.weight.dtype)
            self.weight.data = hadamard_out @ self.weight.data @ hadamard_in.T
        self.absorb_rotation = True

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.absorb_rotation:
            return F.linear(values, self.weight, self.bias)
        rotated = fast_hadamard_transform(values)
        projected = F.linear(rotated, self.weight, self.bias)
        return fast_hadamard_transform(projected)


class HadamardRotation(nn.Module):
    def __init__(self, dim: int, inverse: bool = False) -> None:
        super().__init__()
        if dim & (dim - 1) != 0:
            raise ValueError(f"dim must be a power of two, got {dim}")
        self.dim = dim
        self.inverse = inverse

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.inverse:
            return inverse_fast_hadamard_transform(values)
        return fast_hadamard_transform(values)