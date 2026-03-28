"""Lightweight core verification — no model downloads required.

Tests the TPN architecture, losses, masking, and gather logic
using only synthetic tensors. No DINOv2/CLIP/LLaVA downloads.

Usage:
    python scripts/test_core.py
"""

import sys
sys.path.insert(0, ".")

import torch
from src.tc_vitok.models.tpn import TokenizationPolicyNetwork
from src.tc_vitok.models.masking import gumbel_sigmoid_ste, compute_spatial_entropy, compute_mask_diversity
from src.tc_vitok.models.tc_vitok_model import gather_selected_tokens
from src.tc_vitok.training.losses import l1_sparsity_loss, compute_total_loss
from src.tc_vitok.training.curriculum import get_tau, get_lambda


def test_tpn():
    """TPN architecture matches paper."""
    print("TEST 1: TPN Architecture")
    tpn = TokenizationPolicyNetwork(
        d_model=384, d_text=512, d_coarse=384,
        n_heads=4, d_h=512, M=49, N=576,
    )
    num_params = tpn.get_num_params()
    print(f"  Params: {num_params:,} (paper: ~887K)")
    assert 800_000 < num_params < 1_000_000, f"Got {num_params:,}"

    B = 4
    v = torch.randn(B, 49, 384)
    q = torch.randn(B, 512)
    idx, scores = tpn(v, q, k=144, tau=1.0, training=True)

    assert idx.shape == (B, 144), f"Expected idx shape (4, 144), got {idx.shape}"
    assert scores.shape == (B, 576)
    # Check that indices are in valid range
    assert idx.min() >= 0 and idx.max() < 576
    print("  PASSED\n")


def test_ste():
    """Gumbel-Sigmoid STE correctness."""
    print("TEST 2: Gumbel-Sigmoid STE")
    B, N, k = 4, 576, 144
    scores = torch.randn(B, N, requires_grad=True)

    mask, idx = gumbel_sigmoid_ste(scores, k=k, tau=1.0, training=True)
    assert mask.grad_fn is not None, "Mask should have gradients for backprop"
    assert idx.shape == (B, k), f"Expected idx shape (4, 144), got {idx.shape}"
    vals = mask.detach().unique().tolist()
    assert set(vals) == {0.0, 1.0}, f"Expected binary mask, got {vals}"

    loss = mask.sum()
    loss.backward()
    assert scores.grad is not None
    assert scores.grad.abs().sum() > 0
    print("  PASSED\n")


def test_gather():
    """Gather-based token extraction."""
    print("TEST 3: Gather Token Extraction")
    B, N, D, k = 4, 576, 1024, 144
    v = torch.randn(B, N, D)
    # Create indices for first k tokens
    idx = torch.arange(k).unsqueeze(0).expand(B, -1)

    selected = gather_selected_tokens(v, idx)
    assert selected.shape == (B, k, D)
    assert torch.allclose(selected, v[:, :k, :])
    print(f"  [{B}, {N}, {D}] -> [{B}, {k}, {D}] (real compression via gather)")
    print("  PASSED\n")


def test_sparsity_loss():
    """L1 sparsity = mean(|scores|), NOT |sum(mask)-k|."""
    print("TEST 4: Sparsity Loss (L1 on raw scores)")
    scores = torch.tensor([[1.0, -2.0, 0.5, 3.0]])
    loss = l1_sparsity_loss(scores)
    expected = (1.0 + 2.0 + 0.5 + 3.0) / 4.0
    assert abs(loss.item() - expected) < 1e-5, f"Got {loss.item()}, expected {expected}"
    print(f"  mean(|[1,-2,0.5,3]|) = {loss.item():.4f} (expected {expected})")
    print("  PASSED\n")


def test_entropy():
    """Shannon entropy of normalized spatial distribution."""
    print("TEST 5: Spatial Entropy (Shannon)")
    B, N = 4, 576

    # Uniform mask -> maximum entropy
    uniform_mask = torch.ones(B, N) * (144.0 / N)
    entropy_uniform = compute_spatial_entropy(uniform_mask)
    max_entropy = torch.log(torch.tensor(float(N)))
    print(f"  Uniform entropy: {entropy_uniform.item():.4f} (max: {max_entropy.item():.4f})")
    assert abs(entropy_uniform.item() - max_entropy.item()) < 0.1

    # Concentrated mask -> low entropy
    concentrated_mask = torch.zeros(B, N)
    concentrated_mask[:, :10] = 1.0
    entropy_conc = compute_spatial_entropy(concentrated_mask)
    print(f"  Concentrated entropy: {entropy_conc.item():.4f}")
    assert entropy_conc.item() < entropy_uniform.item()

    print("  PASSED\n")


def test_full_loss():
    """Total loss with all terms, gradient to scores."""
    print("TEST 6: Total Loss Computation")
    B, N, k = 4, 576, 144
    scores = torch.randn(B, N, requires_grad=True)
    mask, idx = gumbel_sigmoid_ste(scores, k=k, tau=1.0, training=True)
    logits = torch.randn(B, k, 1000)
    labels = torch.randint(0, 1000, (B, k))

    loss_dict = compute_total_loss(
        logits=logits, labels=labels,
        scores=scores, mask=mask, k=k,
        lambda_sparsity=0.1, lambda_entropy=0.01, lambda_diversity=0.01,
    )

    assert loss_dict["total"].grad_fn is not None
    loss_dict["total"].backward()
    assert scores.grad is not None
    assert scores.grad.abs().sum() > 0

    for key in ["total", "vqa", "sparsity", "entropy", "diversity"]:
        print(f"  {key}: {loss_dict[key].item():.4f}")
    print("  PASSED\n")


def test_curriculum():
    """Curriculum matches paper values."""
    print("TEST 7: Curriculum Schedules")
    T = 50000
    assert abs(get_tau(0, T) - 1.0) < 0.01
    assert abs(get_tau(T, T) - 0.1) < 0.01
    assert abs(get_lambda(0, 5000, 0.1)) < 1e-6
    assert abs(get_lambda(5000, 5000, 0.1) - 0.1) < 0.01
    print(f"  tau: 1.0 -> 0.1, lambda: 0.0 -> 0.1")
    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TC-ViTok Core Verification (no model downloads)")
    print("=" * 60 + "\n")

    test_tpn()
    test_ste()
    test_gather()
    test_sparsity_loss()
    test_entropy()
    test_full_loss()
    test_curriculum()

    print("=" * 60)
    print("ALL 7 CORE TESTS PASSED")
    print("=" * 60)
