from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "torch_fused_ops requires PyTorch. Install torch before importing this module."
    ) from exc

from .torch_entropy import EntropyProfiler, activation_entropy, entropy_delta, fusion_boundary_selector, recommend_tile_size
from .torch_hadamard import FusedHadamardLinear, fast_hadamard_transform


class SSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=4, padding=3, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        basis = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(basis))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = values.shape
        xz = self.in_proj(values)
        x_in, z = xz.chunk(2, dim=-1)
        x_in = self.conv1d(x_in.transpose(1, 2))[..., :length].transpose(1, 2)
        x_in = F.silu(x_in)
        xbc = self.x_proj(x_in)
        dt, b_state, c_state = torch.split(xbc, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        a_matrix = -torch.exp(self.A_log.float())
        state_output = self._simple_scan(x_in, dt, a_matrix, b_state, c_state)
        gated = state_output * F.silu(z)
        return self.out_proj(gated)

    def _simple_scan(
        self,
        values: torch.Tensor,
        dt: torch.Tensor,
        a_matrix: torch.Tensor,
        b_state: torch.Tensor,
        c_state: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, length, d_inner = values.shape
        state = values.new_zeros(batch_size, d_inner, a_matrix.shape[-1])
        outputs = []
        for index in range(length):
            delta_a = torch.exp(dt[:, index, :].unsqueeze(-1) * a_matrix.unsqueeze(0))
            delta_b = dt[:, index, :].unsqueeze(-1) * b_state[:, index, :].unsqueeze(1)
            state = delta_a * state + delta_b * values[:, index, :].unsqueeze(-1)
            outputs.append((state * c_state[:, index, :].unsqueeze(1)).sum(-1))
        stacked = torch.stack(outputs, dim=1)
        return stacked + self.D.unsqueeze(0).unsqueeze(0) * values


class EntropyGuidedSSMBlock(SSMBlock):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, entropy_threshold: float = 5.0) -> None:
        super().__init__(d_model=d_model, d_state=d_state, expand=expand)
        self.entropy_threshold = entropy_threshold
        self.in_proj = FusedHadamardLinear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = FusedHadamardLinear(self.d_inner, d_model, bias=False)
        self.profiler = EntropyProfiler()

    def calibrate(self, sample: torch.Tensor) -> dict[str, float | list[list[int]] | int]:
        with torch.no_grad():
            projected = self.in_proj(sample)
            entropy_before = float(activation_entropy(projected).item())
            rotated = fast_hadamard_transform(projected)
            entropy_after = float(activation_entropy(rotated).item())
            delta = float(entropy_delta(projected, rotated).item())
            groups = fusion_boundary_selector([entropy_before, entropy_after], entropy_threshold=self.entropy_threshold)
            tile_size = recommend_tile_size(max(entropy_before, entropy_after))
            return {
                "entropy_before": entropy_before,
                "entropy_after": entropy_after,
                "entropy_gain": delta,
                "fusion_groups": groups,
                "tile_size": tile_size,
            }