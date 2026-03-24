"""Unit tests for Tokenization Policy Network (TPN)."""

import torch
import pytest

# TODO (Phase C): Import TPN
# from tc_vitok.models import TokenizationPolicyNetwork


def test_tpn_output_shape():
    """Test that TPN outputs correct mask and score shapes."""
    # TODO (Phase C): Implement test
    # TPN should output:
    # - mask: [batch_size, 576]
    # - scores: [batch_size, 576]

    batch_size = 4
    k = 144

    # tpn = TokenizationPolicyNetwork()

    # Dummy inputs
    v_coarse = torch.randn(batch_size, 49, 384)
    q_embed = torch.randn(batch_size, 512)

    # Forward pass
    # mask, scores = tpn(v_coarse, q_embed, k=k, tau=1.0, training=True)

    # Check shapes
    # assert mask.shape == (batch_size, 576)
    # assert scores.shape == (batch_size, 576)

    # Check mask is binary
    # assert torch.all((mask == 0) | (mask == 1))

    # Check exactly k tokens selected per sample
    # assert torch.all(mask.sum(dim=1) == k)

    pass


def test_tpn_gradient_flow():
    """Test that gradients flow through TPN (critical for training)."""
    # TODO (Phase C): Implement test
    # Verify mask.grad_fn is not None

    batch_size = 2
    k = 72

    # tpn = TokenizationPolicyNetwork()

    # Dummy inputs
    v_coarse = torch.randn(batch_size, 49, 384, requires_grad=True)
    q_embed = torch.randn(batch_size, 512, requires_grad=True)

    # Forward pass
    # mask, scores = tpn(v_coarse, q_embed, k=k, tau=1.0, training=True)

    # CRITICAL CHECK: mask must have grad_fn
    # assert mask.grad_fn is not None, "STE is broken! No gradient flow."

    # Verify backward pass works
    # loss = mask.sum()
    # loss.backward()

    # assert v_coarse.grad is not None
    # assert q_embed.grad is not None

    pass


def test_tpn_num_params():
    """Test that TPN has approximately 3M parameters."""
    # TODO (Phase C): Implement test
    # tpn = TokenizationPolicyNetwork()
    # num_params = tpn.get_num_params()

    # Should be around 3M params (lightweight)
    # assert 2_000_000 < num_params < 5_000_000

    pass
