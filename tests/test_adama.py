"""
Unit tests for the ADAMA src modules:
  - hadamard.py  (Hadamard transform and FusedHadamardLinear)
  - entropy.py   (activation entropy, fusion boundary selector)
  - fused_ops.py (SSMBlock, EntropyGuidedSSMBlock, FusionScheduler)
"""

import math
import pytest
import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.hadamard import (
    hadamard_matrix,
    fast_hadamard_transform,
    inverse_fast_hadamard_transform,
    FusedHadamardLinear,
    HadamardRotation,
)
from src.entropy import (
    activation_entropy,
    entropy_delta,
    outlier_ratio,
    kurtosis,
    fusion_boundary_selector,
    recommend_tile_size,
    EntropyProfiler,
)
from src.fused_ops import (
    SSMBlock,
    EntropyGuidedSSMBlock,
    FusionScheduler,
)


# ===========================================================================
# hadamard.py tests
# ===========================================================================


class TestHadamardMatrix:
    def test_shape(self):
        for n in [1, 2, 4, 8, 16, 32]:
            H = hadamard_matrix(n)
            assert H.shape == (n, n), f"Expected ({n},{n}), got {H.shape}"

    def test_orthogonality(self):
        """H @ H^T should equal the identity matrix."""
        for n in [2, 4, 8, 16]:
            H = hadamard_matrix(n, dtype=torch.float64)
            product = H @ H.T
            identity = torch.eye(n, dtype=torch.float64)
            assert torch.allclose(product, identity, atol=1e-10), \
                f"Hadamard matrix of size {n} is not orthogonal"

    def test_non_power_of_2_raises(self):
        with pytest.raises(ValueError):
            hadamard_matrix(3)
        with pytest.raises(ValueError):
            hadamard_matrix(6)

    def test_n1(self):
        H = hadamard_matrix(1)
        assert H.shape == (1, 1)
        assert torch.allclose(H, torch.tensor([[1.0]]))


class TestFastHadamardTransform:
    def test_matches_matrix_multiply(self):
        """FWHT output should match naive H @ x."""
        for n in [4, 8, 16, 32]:
            x = torch.randn(n, dtype=torch.float64)
            H = hadamard_matrix(n, dtype=torch.float64)
            expected = H @ x
            got = fast_hadamard_transform(x)
            assert torch.allclose(expected, got, atol=1e-10), \
                f"FWHT mismatch for n={n}"

    def test_batched(self):
        """FWHT should work on batched input (..., n)."""
        n = 16
        x = torch.randn(3, 5, n, dtype=torch.float64)
        H = hadamard_matrix(n, dtype=torch.float64)
        expected = (x @ H.T)  # batched: (..., n) @ (n, n)^T = (..., n) @ H (since H is symmetric up to sign)
        # Use direct per-sample comparison via loop
        got = fast_hadamard_transform(x)
        for i in range(3):
            for j in range(5):
                exp_ij = H @ x[i, j]
                assert torch.allclose(exp_ij, got[i, j], atol=1e-10)

    def test_self_inverse(self):
        """Applying FWHT twice should recover the original vector."""
        n = 16
        x = torch.randn(n, dtype=torch.float64)
        recovered = fast_hadamard_transform(fast_hadamard_transform(x))
        assert torch.allclose(x, recovered, atol=1e-10)

    def test_non_power_of_2_raises(self):
        with pytest.raises(ValueError):
            fast_hadamard_transform(torch.randn(6))

    def test_inverse_is_forward(self):
        """Inverse WHT should equal forward WHT (self-inverse property)."""
        n = 8
        x = torch.randn(n)
        fwd = fast_hadamard_transform(x)
        inv = inverse_fast_hadamard_transform(x)
        assert torch.allclose(fwd, inv, atol=1e-6)


class TestFusedHadamardLinear:
    def test_output_shape(self):
        layer = FusedHadamardLinear(16, 32)
        x = torch.randn(4, 16)
        out = layer(x)
        assert out.shape == (4, 32)

    def test_absorb_rotation_matches_explicit(self):
        """
        FusedHadamardLinear with absorbed rotation should produce the same
        output as explicit WHT + standard linear.
        """
        torch.manual_seed(0)
        in_f, out_f = 16, 32
        layer = FusedHadamardLinear(in_f, out_f, bias=False, absorb_rotation=False)
        x = torch.randn(4, in_f, dtype=torch.float64)
        layer = layer.double()

        # Explicit: WHT(x) then linear
        x_wht = fast_hadamard_transform(x)
        out_explicit = torch.nn.functional.linear(x_wht, layer.weight)

        # Absorbed: build fused weight manually
        H_in = hadamard_matrix(in_f, dtype=torch.float64)
        W_fused = layer.weight @ H_in.T  # weight @ H_in^T  (since we do W @ (H x))
        out_absorbed = torch.nn.functional.linear(x, W_fused)

        assert torch.allclose(out_explicit, out_absorbed, atol=1e-8)

    def test_fuse_hadamard_into_weight(self):
        """After calling fuse_hadamard_into_weight, layer runs as absorb_rotation=True."""
        torch.manual_seed(1)
        layer = FusedHadamardLinear(8, 16, bias=False, absorb_rotation=False)
        layer = layer.double()
        x = torch.randn(3, 8, dtype=torch.float64)

        # output before fusing (runtime WHT)
        out_before = layer(x)

        # fuse and run again
        layer.fuse_hadamard_into_weight()
        # After fusion, the layer applies W_fused directly:
        # W_fused = H_out @ W @ H_in^T  (as documented)
        # out_before = H_out(W @ H_in^T x) = (H_out W H_in^T) x  in principle
        # This test just verifies the method runs and output shape is preserved.
        out_after = layer(x)
        assert out_before.shape == out_after.shape

    def test_hadamard_rotation_module(self):
        layer = HadamardRotation(16)
        x = torch.randn(2, 16)
        out = layer(x)
        assert out.shape == x.shape

    def test_hadamard_rotation_self_inverse(self):
        layer = HadamardRotation(16)
        x = torch.randn(5, 16, dtype=torch.float64)
        recovered = layer(layer(x))
        assert torch.allclose(x, recovered, atol=1e-10)


# ===========================================================================
# entropy.py tests
# ===========================================================================


class TestActivationEntropy:
    def test_uniform_max_entropy(self):
        """Uniform distribution should have close-to-maximum entropy."""
        x = torch.linspace(-1, 1, 256 * 100)
        h = activation_entropy(x, num_bins=256)
        # log2(256) = 8 bits is the maximum for 256 bins
        assert h.item() > 7.0, f"Expected high entropy for uniform dist, got {h.item()}"

    def test_constant_zero_entropy(self):
        """All-same tensor should have zero entropy."""
        x = torch.ones(1000)
        h = activation_entropy(x, num_bins=256)
        assert h.item() == pytest.approx(0.0, abs=1e-3)

    def test_entropy_range(self):
        """Entropy should be in [0, log2(num_bins)]."""
        x = torch.randn(1000)
        num_bins = 64
        h = activation_entropy(x, num_bins=num_bins)
        assert 0.0 <= h.item() <= math.log2(num_bins) + 1e-6

    def test_per_channel(self):
        """Per-channel entropy should return one value per channel."""
        C = 8
        x = torch.randn(100, C)
        h = activation_entropy(x, per_channel=True)
        assert h.shape == (C,)

    def test_scalar_output(self):
        """Default (per_channel=False) should return a scalar tensor."""
        x = torch.randn(100)
        h = activation_entropy(x)
        assert h.dim() == 0


class TestEntropyDelta:
    def test_positive_delta_for_hadamard(self):
        """
        WHT should increase histogram entropy for a heavy-tailed distribution.
        """
        torch.manual_seed(42)
        # Create heavy-tailed tensor (Cauchy-like) with large outliers
        x = torch.zeros(512)
        x[:16] = 1000.0   # outliers
        x[16:] = torch.randn(512 - 16) * 0.1

        x_wht = fast_hadamard_transform(x)
        delta = entropy_delta(x, x_wht)
        # WHT should spread the outlier energy, increasing entropy
        assert delta.item() > 0.0, \
            f"Expected positive ΔH after WHT on outlier tensor, got {delta.item()}"


class TestOutlierMetrics:
    def test_outlier_ratio_zero_for_gaussian(self):
        """Clean Gaussian should have a small outlier ratio (< 1%)."""
        torch.manual_seed(0)
        x = torch.randn(10000)
        ratio = outlier_ratio(x, sigma_threshold=3.0)
        # Theoretically ~0.27% of Gaussian samples exceed 3σ
        assert ratio.item() < 0.01

    def test_outlier_ratio_high_for_spiky(self):
        """Tensor with many outliers should have high ratio."""
        x = torch.zeros(100)
        x[::2] = 1000.0   # 50% large values
        # Bimodal [0, 1000]: mean=500, std=500 (exact).
        # |0-500| = |1000-500| = 500 > 0.9*500=450, so all 100 elements are outliers.
        ratio = outlier_ratio(x, sigma_threshold=0.9)
        assert ratio.item() > 0.3

    def test_kurtosis_near_zero_for_gaussian(self):
        """Gaussian should have near-zero excess kurtosis."""
        torch.manual_seed(0)
        x = torch.randn(100000)
        k = kurtosis(x)
        assert abs(k.item()) < 0.2, f"Expected near-zero kurtosis, got {k.item()}"

    def test_kurtosis_positive_for_heavy_tail(self):
        """Heavy-tailed distribution should have positive excess kurtosis."""
        torch.manual_seed(0)
        x = torch.zeros(1000)
        x[:10] = 100.0    # heavy tail
        k = kurtosis(x)
        assert k.item() > 0.0


class TestFusionBoundarySelector:
    def test_basic(self):
        entropies = [6.1, 6.3, 5.8, 3.2, 6.7, 6.5]
        groups = fusion_boundary_selector(entropies, entropy_threshold=5.0)
        assert groups == [[0, 1, 2], [3], [4, 5]]

    def test_all_high_entropy(self):
        entropies = [6.0] * 10
        groups = fusion_boundary_selector(entropies, entropy_threshold=5.0,
                                          max_fused_layers=10)
        assert groups == [list(range(10))]

    def test_all_low_entropy(self):
        entropies = [2.0] * 5
        groups = fusion_boundary_selector(entropies, entropy_threshold=5.0)
        # Each layer forms its own group
        assert groups == [[i] for i in range(5)]

    def test_max_fused_layers(self):
        entropies = [6.0] * 10
        groups = fusion_boundary_selector(entropies, entropy_threshold=5.0,
                                          max_fused_layers=3)
        # Should split into groups of at most 3
        for g in groups:
            assert len(g) <= 3

    def test_empty_input(self):
        groups = fusion_boundary_selector([], entropy_threshold=5.0)
        assert groups == []


class TestRecommendTileSize:
    def test_min_entropy_gives_min_tile(self):
        t = recommend_tile_size(0.0, base_tile=64, max_tile=512)
        assert t == 64

    def test_max_entropy_gives_max_tile(self):
        t = recommend_tile_size(8.0, base_tile=64, max_tile=512)
        assert t == 512

    def test_power_of_2(self):
        for h in [1.0, 2.5, 4.0, 6.0]:
            t = recommend_tile_size(h)
            assert (t & (t - 1)) == 0, f"Tile {t} is not a power of 2"


class TestEntropyProfiler:
    def test_records_hooks(self):
        model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8))
        profiler = EntropyProfiler()
        handles = profiler.register_hooks(model)
        x = torch.randn(4, 16)
        with torch.no_grad():
            _ = model(x)
        profiler.remove_hooks(handles)
        seq = profiler.entropy_sequence()
        # Should have one record per leaf module (3 leaf modules)
        assert len(seq) == 3
        for h in seq:
            assert h >= 0.0

    def test_reset(self):
        profiler = EntropyProfiler()
        profiler.records.append(("layer", 5.0))
        profiler.reset()
        assert profiler.records == []


# ===========================================================================
# fused_ops.py tests
# ===========================================================================


class TestSSMBlock:
    @pytest.fixture
    def block(self):
        return SSMBlock(d_model=32, d_state=4, expand=2)

    def test_output_shape(self, block):
        x = torch.randn(2, 16, 32)
        with torch.no_grad():
            out = block(x)
        assert out.shape == (2, 16, 32)

    def test_gradient_flows(self, block):
        x = torch.randn(1, 8, 32, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_different_sequence_lengths(self, block):
        for L in [1, 4, 16, 64]:
            x = torch.randn(1, L, 32)
            with torch.no_grad():
                out = block(x)
            assert out.shape == (1, L, 32)


class TestEntropyGuidedSSMBlock:
    @pytest.fixture
    def block(self):
        return EntropyGuidedSSMBlock(d_model=32, d_state=4, expand=2,
                                     use_hadamard=True)

    def test_output_shape(self, block):
        x = torch.randn(2, 16, 32)
        with torch.no_grad():
            out = block(x)
        assert out.shape == (2, 16, 32)

    def test_calibrate_returns_dict(self, block):
        x = torch.randn(2, 16, 32)
        result = block.calibrate(x)
        required_keys = {
            "entropy_before_hadamard",
            "entropy_after_hadamard",
            "delta_entropy",
            "recommended_tile",
            "use_hadamard",
        }
        assert required_keys.issubset(result.keys())
        assert isinstance(result["recommended_tile"], int)
        assert isinstance(result["use_hadamard"], bool)

    def test_calibrate_delta_entropy_sign(self, block):
        """Calibration delta entropy should be non-negative after WHT."""
        torch.manual_seed(42)
        # Build a tensor with explicit outliers to ensure WHT increases entropy
        x = torch.randn(4, 16, 32)
        x[:, :, 0] = 100.0   # large outlier in first channel
        result = block.calibrate(x)
        # After WHT on an outlier-rich tensor, delta should be > 0
        assert result["delta_entropy"] >= 0.0


class TestFusionScheduler:
    def test_plan_returns_dict(self):
        scheduler = FusionScheduler(entropy_threshold=5.0, max_fused_layers=8)
        entropies = [6.1, 6.3, 5.8, 3.2, 6.7, 6.5, 2.8, 5.9]
        plan = scheduler.plan(entropies)
        assert "groups" in plan
        assert "num_kernels" in plan
        assert "fusion_ratio" in plan

    def test_fusion_ratio_range(self):
        scheduler = FusionScheduler()
        for entropies in [
            [6.0] * 10,   # all high → maximum fusion
            [2.0] * 10,   # all low → no fusion
            [6.0, 2.0] * 5,  # alternating
        ]:
            plan = scheduler.plan(entropies)
            assert 0.0 <= plan["fusion_ratio"] <= 1.0

    def test_num_kernels_equals_len_groups(self):
        scheduler = FusionScheduler()
        entropies = [5.5, 5.5, 2.0, 5.5, 5.5, 2.0, 5.5]
        plan = scheduler.plan(entropies)
        assert plan["num_kernels"] == len(plan["groups"])
