"""Unit tests for Gumbel-Sigmoid STE masking."""

import torch
import pytest

# TODO (Phase C): Import masking functions
# from tc_vitok.models.masking import gumbel_sigmoid_ste


def test_ste_forward_hard():
    """Test that STE produces hard binary mask in forward pass."""
    # TODO (Phase C): Implement test

    batch_size = 8
    num_tokens = 576
    k = 144

    scores = torch.randn(batch_size, num_tokens)

    # Forward pass
    # mask = gumbel_sigmoid_ste(scores, k=k, tau=1.0, training=True, hard=True)

    # Check mask is binary
    # assert torch.all((mask == 0) | (mask == 1))

    # Check exactly k tokens selected
    # assert torch.all(mask.sum(dim=1) == k)

    pass


def test_ste_backward_soft():
    """Test that STE uses soft gradients in backward pass."""
    # TODO (Phase C): Implement test
    # This is the key property of straight-through estimators

    batch_size = 4
    num_tokens = 100
    k = 20

    scores = torch.randn(batch_size, num_tokens, requires_grad=True)

    # Forward pass with STE
    # mask = gumbel_sigmoid_ste(scores, k=k, tau=1.0, training=True)

    # Backward pass
    # loss = mask.sum()
    # loss.backward()

    # Gradients should exist and be non-zero
    # assert scores.grad is not None
    # assert torch.any(scores.grad != 0)

    # Gradients should NOT be exactly 0 or 1 (soft, not hard)
    # assert not torch.all((scores.grad == 0) | (scores.grad == 1))

    pass


def test_ste_temperature_effect():
    """Test that temperature affects selection sharpness."""
    # TODO (Phase C): Implement test

    scores = torch.randn(16, 576)
    k = 144

    # High temperature → softer selection
    # mask_high_tau = gumbel_sigmoid_ste(scores, k=k, tau=10.0, training=True, hard=False)

    # Low temperature → sharper selection
    # mask_low_tau = gumbel_sigmoid_ste(scores, k=k, tau=0.1, training=True, hard=False)

    # Low tau should be closer to binary
    # assert mask_low_tau.std() > mask_high_tau.std()

    pass


def test_ste_deterministic_eval():
    """Test that STE is deterministic in eval mode (no Gumbel noise)."""
    # TODO (Phase C): Implement test

    scores = torch.randn(4, 100)
    k = 20

    # Eval mode (training=False)
    # mask1 = gumbel_sigmoid_ste(scores, k=k, tau=1.0, training=False)
    # mask2 = gumbel_sigmoid_ste(scores, k=k, tau=1.0, training=False)

    # Should be identical (no randomness)
    # assert torch.all(mask1 == mask2)

    pass
