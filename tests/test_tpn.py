"""Unit tests for Tokenization Policy Network (TPN).

Tests verify:
- Output shapes: idx=[B,k], s_N=[B,576]
- Parameter count ~887K
- Gradient flow through STE (critical for training)
- Deterministic behavior in eval mode
"""

import torch
import pytest

from tc_vitok.models.tpn import TokenizationPolicyNetwork


@pytest.fixture
def tpn():
    """Create a default TPN instance."""
    return TokenizationPolicyNetwork()


@pytest.fixture
def dummy_inputs():
    """Create dummy coarse visual and text embedding inputs."""
    B = 4
    v_coarse = torch.randn(B, 49, 384)
    q_embed = torch.randn(B, 512)
    return v_coarse, q_embed


class TestTPNOutputShape:
    """Test TPN output shapes match architecture spec."""

    def test_idx_shape(self, tpn, dummy_inputs):
        """idx should be [B, k]."""
        v_coarse, q_embed = dummy_inputs
        k = 144
        idx, s_N = tpn(v_coarse, q_embed, k=k, tau=1.0, training=True)
        assert idx.shape == (4, k), f"Expected idx shape (4, {k}), got {idx.shape}"

    def test_scores_shape(self, tpn, dummy_inputs):
        """s_N should be [B, 576]."""
        v_coarse, q_embed = dummy_inputs
        idx, s_N = tpn(v_coarse, q_embed, k=144, tau=1.0, training=True)
        assert s_N.shape == (4, 576), f"Expected s_N shape (4, 576), got {s_N.shape}"

    def test_idx_values_in_range(self, tpn, dummy_inputs):
        """All indices should be in [0, 575]."""
        v_coarse, q_embed = dummy_inputs
        idx, _ = tpn(v_coarse, q_embed, k=144, tau=1.0, training=True)
        assert idx.min() >= 0, f"idx has negative values: min={idx.min()}"
        assert idx.max() < 576, f"idx exceeds 575: max={idx.max()}"

    def test_idx_unique_per_sample(self, tpn, dummy_inputs):
        """Each sample should have k unique indices (no duplicates)."""
        v_coarse, q_embed = dummy_inputs
        k = 144
        idx, _ = tpn(v_coarse, q_embed, k=k, tau=1.0, training=True)
        for b in range(idx.shape[0]):
            unique_count = idx[b].unique().numel()
            assert unique_count == k, (
                f"Sample {b}: expected {k} unique indices, got {unique_count}"
            )

    @pytest.mark.parametrize("k", [72, 144, 216, 288])
    def test_different_token_budgets(self, tpn, dummy_inputs, k):
        """TPN should work with different token budgets."""
        v_coarse, q_embed = dummy_inputs
        idx, s_N = tpn(v_coarse, q_embed, k=k, tau=1.0, training=True)
        assert idx.shape == (4, k)
        assert s_N.shape == (4, 576)


class TestTPNParamCount:
    """Test TPN parameter count matches paper spec."""

    def test_param_count_approximately_887k(self, tpn):
        """TPN should have ~887,955 trainable parameters."""
        num_params = tpn.get_num_params()
        assert 800_000 < num_params < 1_000_000, (
            f"Expected ~887K params, got {num_params:,}"
        )

    def test_param_count_exact(self, tpn):
        """TPN should have exactly 887,955 parameters (paper Appendix A.1)."""
        num_params = tpn.get_num_params()
        assert num_params == 887_955, (
            f"Expected 887,955 params, got {num_params:,}"
        )


class TestTPNGradientFlow:
    """Test gradient flow through TPN and STE."""

    def test_scores_have_grad_fn(self, tpn):
        """s_N must have grad_fn for gradient-based optimization."""
        v_coarse = torch.randn(2, 49, 384)
        q_embed = torch.randn(2, 512)
        _, s_N = tpn(v_coarse, q_embed, k=72, tau=1.0, training=True)
        assert s_N.grad_fn is not None, "s_N has no grad_fn — TPN is not differentiable"

    def test_backward_produces_gradients(self, tpn):
        """Backward pass through s_N should produce parameter gradients."""
        v_coarse = torch.randn(2, 49, 384, requires_grad=True)
        q_embed = torch.randn(2, 512, requires_grad=True)
        _, s_N = tpn(v_coarse, q_embed, k=72, tau=1.0, training=True)

        loss = s_N.sum()
        loss.backward()

        assert v_coarse.grad is not None, "No gradient on v_coarse"
        assert q_embed.grad is not None, "No gradient on q_embed"

        # Check at least some TPN parameters received gradients
        grads_exist = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in tpn.parameters()
        )
        assert grads_exist, "No TPN parameters received gradients"


class TestTPNEvalMode:
    """Test TPN behavior in evaluation mode."""

    def test_eval_is_deterministic(self, tpn):
        """Eval mode should produce identical outputs for same input."""
        v_coarse = torch.randn(2, 49, 384)
        q_embed = torch.randn(2, 512)
        tpn.eval()

        idx1, s_N1 = tpn(v_coarse, q_embed, k=144, tau=1.0, training=False)
        idx2, s_N2 = tpn(v_coarse, q_embed, k=144, tau=1.0, training=False)

        assert torch.equal(idx1, idx2), "Eval mode is not deterministic (idx differ)"
        assert torch.equal(s_N1, s_N2), "Eval mode is not deterministic (s_N differ)"
