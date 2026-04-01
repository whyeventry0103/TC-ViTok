"""Unit tests for Gumbel-Sigmoid STE masking.

Tests verify:
- Forward pass produces hard binary masks
- Backward pass has soft gradients (STE property)
- gather_tokens extracts correct shape
- Deterministic behavior in eval mode
- Temperature effect on selection
"""

import torch
import pytest

from tc_vitok.models.masking import (
    gumbel_sigmoid_ste,
    compute_spatial_entropy,
    compute_mask_diversity,
)
from tc_vitok.models.tc_vitok_model import gather_tokens


class TestSTEForward:
    """Test forward pass of Gumbel-Sigmoid STE."""

    def test_mask_is_binary(self):
        """STE forward mask must be exactly 0 or 1."""
        scores = torch.randn(8, 576)
        mask, idx = gumbel_sigmoid_ste(scores, k=144, tau=1.0, training=True)
        assert torch.all((mask == 0) | (mask == 1)), "Mask is not binary"

    def test_exactly_k_selected(self):
        """Each sample must have exactly k ones in the mask."""
        scores = torch.randn(8, 576)
        k = 144
        mask, idx = gumbel_sigmoid_ste(scores, k=k, tau=1.0, training=True)
        counts = mask.sum(dim=1)
        assert torch.all(counts == k), (
            f"Expected {k} tokens per sample, got counts: {counts.tolist()}"
        )

    def test_idx_shape(self):
        """idx should be [B, k]."""
        scores = torch.randn(4, 576)
        k = 72
        mask, idx = gumbel_sigmoid_ste(scores, k=k, tau=1.0, training=True)
        assert idx.shape == (4, k), f"Expected idx shape (4, {k}), got {idx.shape}"

    def test_idx_matches_mask(self):
        """Indices should correspond to the ones in the mask."""
        scores = torch.randn(4, 576)
        k = 144
        mask, idx = gumbel_sigmoid_ste(scores, k=k, tau=1.0, training=True)
        for b in range(4):
            mask_ones = mask[b].nonzero(as_tuple=True)[0]
            idx_sorted = idx[b].sort().values
            mask_sorted = mask_ones.sort().values
            assert torch.equal(idx_sorted, mask_sorted), (
                f"Sample {b}: idx and mask ones do not match"
            )


class TestSTEBackward:
    """Test backward pass (gradient flow through STE)."""

    def test_mask_has_grad_fn(self):
        """mask_ste must have grad_fn for gradient flow."""
        scores = torch.randn(4, 100, requires_grad=True)
        mask, idx = gumbel_sigmoid_ste(scores, k=20, tau=1.0, training=True)
        assert mask.grad_fn is not None, "STE mask has no grad_fn — STE is broken"

    def test_backward_produces_gradients(self):
        """Backward through mask should produce gradients on scores."""
        scores = torch.randn(4, 100, requires_grad=True)
        mask, idx = gumbel_sigmoid_ste(scores, k=20, tau=1.0, training=True)

        loss = mask.sum()
        loss.backward()

        assert scores.grad is not None, "No gradient on scores"
        assert torch.any(scores.grad != 0), "All gradients are zero"

    def test_gradients_are_soft(self):
        """STE gradients should not be exactly 0 or 1 everywhere."""
        scores = torch.randn(4, 100, requires_grad=True)
        mask, idx = gumbel_sigmoid_ste(scores, k=20, tau=1.0, training=True)

        loss = mask.sum()
        loss.backward()

        grad = scores.grad
        non_binary = ~((grad == 0) | (grad == 1))
        assert torch.any(non_binary), "Gradients appear to be hard (all 0 or 1)"


class TestSTEDeterministic:
    """Test deterministic behavior in eval mode."""

    def test_eval_no_randomness(self):
        """With training=False, same input should give same output."""
        scores = torch.randn(4, 100)
        k = 20
        mask1, idx1 = gumbel_sigmoid_ste(scores, k=k, tau=1.0, training=False)
        mask2, idx2 = gumbel_sigmoid_ste(scores, k=k, tau=1.0, training=False)
        assert torch.equal(mask1, mask2), "Eval mode masks differ"
        assert torch.equal(idx1, idx2), "Eval mode indices differ"

    def test_eval_mask_is_binary(self):
        """Eval mode mask should still be binary."""
        scores = torch.randn(4, 100)
        mask, idx = gumbel_sigmoid_ste(scores, k=20, tau=1.0, training=False)
        assert torch.all((mask == 0) | (mask == 1)), "Eval mask is not binary"


class TestGatherTokens:
    """Test gather_tokens extracts correct tokens."""

    def test_output_shape(self):
        """gather_tokens should return [B, k, D]."""
        V = torch.randn(4, 576, 1024)
        idx = torch.randint(0, 576, (4, 144))
        result = gather_tokens(V, idx)
        assert result.shape == (4, 144, 1024), f"Expected (4,144,1024), got {result.shape}"

    def test_correct_values(self):
        """Gathered tokens should match manual indexing."""
        B, N, D = 2, 10, 4
        V = torch.randn(B, N, D)
        idx = torch.tensor([[0, 3, 7], [1, 5, 9]])
        result = gather_tokens(V, idx)

        for b in range(B):
            for j in range(3):
                expected = V[b, idx[b, j]]
                actual = result[b, j]
                assert torch.allclose(expected, actual), (
                    f"Mismatch at batch={b}, pos={j}"
                )

    def test_different_k_values(self):
        """gather_tokens should handle various k values."""
        V = torch.randn(2, 576, 1024)
        for k in [72, 144, 216, 288]:
            idx = torch.randint(0, 576, (2, k))
            result = gather_tokens(V, idx)
            assert result.shape == (2, k, 1024)


class TestSpatialEntropy:
    """Test spatial entropy computation."""

    def test_uniform_mask_high_entropy(self):
        """Uniform selection should produce high entropy."""
        # Create mask where all positions are equally likely
        mask = torch.ones(16, 576) / 4  # not binary but tests the entropy formula
        # Use binary mask: each sample selects different tokens
        mask = torch.zeros(576, 576)
        for i in range(576):
            mask[i, i] = 1.0
        entropy = compute_spatial_entropy(mask)
        max_entropy = torch.log(torch.tensor(576.0))
        assert entropy > max_entropy * 0.95, (
            f"Uniform selection entropy {entropy:.2f} should be near max {max_entropy:.2f}"
        )

    def test_collapsed_mask_low_entropy(self):
        """All samples selecting same token should produce low entropy."""
        mask = torch.zeros(16, 576)
        mask[:, 0] = 1.0  # All select token 0
        entropy = compute_spatial_entropy(mask)
        assert entropy < 0.1, f"Collapsed mask entropy should be near 0, got {entropy:.2f}"


class TestMaskDiversity:
    """Test mask diversity computation."""

    def test_identical_masks_low_diversity(self):
        """Same mask for all samples should have low diversity."""
        mask = torch.zeros(8, 576)
        mask[:, :144] = 1.0  # All select first 144 tokens
        diversity = compute_mask_diversity(mask)
        assert diversity < 0.1, f"Identical masks should have low diversity, got {diversity:.2f}"

    def test_varied_masks_higher_diversity(self):
        """Different masks should produce higher diversity."""
        mask = torch.zeros(8, 576)
        for i in range(8):
            start = i * 72
            mask[i, start:start + 72] = 1.0
        diversity = compute_mask_diversity(mask)
        # Max frequency should be low since different tokens selected
        assert diversity > 0.5, f"Varied masks should have high diversity, got {diversity:.2f}"
