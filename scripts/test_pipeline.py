"""End-to-end pipeline verification script.

Runs a minimal forward + backward pass through the full TC-ViTok pipeline
using synthetic data and the StubLLM. Verifies:
1. All shapes are correct
2. Gather-based token extraction produces [B, k, D]
3. Gradient flows through the STE to TPN parameters
4. Loss computation works with all terms

Usage:
    python scripts/test_pipeline.py
"""

import sys
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, ".")

from src.tc_vitok.models.tc_vitok_model import TCViTokModel, gather_selected_tokens
from src.tc_vitok.models.tpn import TokenizationPolicyNetwork
from src.tc_vitok.models.masking import gumbel_sigmoid_ste, compute_spatial_entropy
from src.tc_vitok.training.losses import compute_total_loss, l1_sparsity_loss
from src.tc_vitok.training.curriculum import get_tau, get_lambda
from src.tc_vitok.data.datasets import DummyVQADataset, collate_fn


def test_tpn_architecture():
    """Verify TPN matches paper architecture."""
    print("=" * 60)
    print("TEST 1: TPN Architecture")
    print("=" * 60)

    tpn = TokenizationPolicyNetwork(
        d_model=384, d_text=512, d_coarse=384,
        n_heads=4, d_h=512, M=49, N=576,
    )

    # Check parameter count (~887K per paper)
    num_params = tpn.get_num_params()
    print(f"  TPN parameters: {num_params:,}")
    assert 800_000 < num_params < 1_000_000, (
        f"Expected ~887K params, got {num_params:,}"
    )

    # Check forward pass shapes
    B = 4
    v_coarse = torch.randn(B, 49, 384)
    q_embed = torch.randn(B, 512)

    mask, scores = tpn(v_coarse, q_embed, k=144, tau=1.0, training=True)

    assert mask.shape == (B, 576), f"Mask shape: {mask.shape}, expected ({B}, 576)"
    assert scores.shape == (B, 576), f"Scores shape: {scores.shape}, expected ({B}, 576)"

    # Check exactly k tokens selected
    k_actual = mask.sum(dim=-1)
    assert torch.allclose(k_actual, torch.full_like(k_actual, 144.0), atol=1.0), (
        f"Expected 144 tokens selected, got {k_actual}"
    )

    # Check gradient flow
    assert mask.grad_fn is not None, "Mask has no gradient! STE is broken."

    print("  PASSED: shapes correct, ~887K params, gradient flows")
    print()


def test_gumbel_sigmoid_ste():
    """Verify Gumbel-Sigmoid STE forward/backward."""
    print("=" * 60)
    print("TEST 2: Gumbel-Sigmoid STE")
    print("=" * 60)

    B, N, k = 4, 576, 144
    scores = torch.randn(B, N, requires_grad=True)

    # Training mode
    mask = gumbel_sigmoid_ste(scores, k=k, tau=1.0, training=True)
    assert mask.shape == (B, N)
    assert mask.grad_fn is not None, "STE has no gradient in training mode"

    # Check binary values in forward
    hard_values = mask.detach()
    assert set(hard_values.unique().tolist()).issubset({0.0, 1.0}), (
        f"Forward pass should be binary, got values: {hard_values.unique()}"
    )

    # Check gradient flows
    loss = mask.sum()
    loss.backward()
    assert scores.grad is not None, "No gradient on scores after backward"
    assert scores.grad.abs().sum() > 0, "Gradient is all zeros"

    # Inference mode
    scores2 = torch.randn(B, N)
    mask_inf = gumbel_sigmoid_ste(scores2, k=k, tau=1.0, training=False)
    assert mask_inf.sum(dim=-1).allclose(torch.full((B,), float(k)))

    print("  PASSED: STE is binary forward, soft backward, correct k")
    print()


def test_gather_extraction():
    """Verify gather-based token extraction."""
    print("=" * 60)
    print("TEST 3: Gather-Based Token Extraction")
    print("=" * 60)

    B, N, D, k = 4, 576, 1024, 144

    # Create dummy visual tokens and mask
    v_full = torch.randn(B, N, D)
    mask = torch.zeros(B, N)
    # Set first k positions to 1 for simplicity
    mask[:, :k] = 1.0

    selected = gather_selected_tokens(v_full, mask, k)

    assert selected.shape == (B, k, D), (
        f"Selected shape: {selected.shape}, expected ({B}, {k}, {D})"
    )

    # Verify the correct tokens were extracted
    expected = v_full[:, :k, :]
    assert torch.allclose(selected, expected), "Gathered tokens don't match expected"

    print(f"  PASSED: output shape [{B}, {k}, {D}] (real compression from [{B}, {N}, {D}])")
    print()


def test_losses():
    """Verify loss functions match paper formulations."""
    print("=" * 60)
    print("TEST 4: Loss Functions")
    print("=" * 60)

    B, N, k = 4, 576, 144

    # Create dummy data
    scores = torch.randn(B, N, requires_grad=True)
    mask = gumbel_sigmoid_ste(scores, k=k, tau=1.0, training=True)

    # Test L1 sparsity on raw scores (NOT |sum(mask) - k|)
    loss_sp = l1_sparsity_loss(scores)
    assert loss_sp.item() > 0, "Sparsity loss should be > 0 for non-zero scores"
    print(f"  L1 sparsity (mean|scores|): {loss_sp.item():.4f}")

    # Test entropy (Shannon entropy of spatial distribution)
    entropy = compute_spatial_entropy(mask.detach())
    max_entropy = torch.log(torch.tensor(float(N)))
    print(f"  Spatial entropy: {entropy.item():.4f} (max possible: {max_entropy.item():.4f})")
    assert 0 < entropy.item() <= max_entropy.item() + 0.1

    # Test full loss computation (with dummy logits)
    logits = torch.randn(B, k, 1000)
    labels = torch.randint(0, 1000, (B, k))

    loss_dict = compute_total_loss(
        logits=logits, labels=labels,
        scores=scores, mask=mask, k=k,
        lambda_sparsity=0.1, lambda_entropy=0.01, lambda_diversity=0.01,
    )

    assert "total" in loss_dict
    assert loss_dict["total"].grad_fn is not None, "Total loss has no gradient"
    print(f"  Total loss: {loss_dict['total'].item():.4f}")
    print(f"  VQA loss: {loss_dict['vqa'].item():.4f}")
    print(f"  Sparsity: {loss_dict['sparsity'].item():.4f}")
    print(f"  Entropy: {loss_dict['entropy'].item():.4f}")
    print(f"  Diversity: {loss_dict['diversity'].item():.4f}")

    # Verify gradient flows to scores
    loss_dict["total"].backward()
    assert scores.grad is not None, "No gradient on scores from total loss"
    print("  PASSED: all loss terms computed, gradients flow")
    print()


def test_curriculum():
    """Verify curriculum schedules match paper."""
    print("=" * 60)
    print("TEST 5: Curriculum Schedules")
    print("=" * 60)

    T = 50000

    # Temperature: starts at 1.0, ends at 0.1 (paper values)
    tau_start = get_tau(0, T, tau_min=0.1, tau_max=1.0)
    tau_end = get_tau(T, T, tau_min=0.1, tau_max=1.0)
    print(f"  tau(0) = {tau_start:.2f} (expect 1.0)")
    print(f"  tau(T) = {tau_end:.2f} (expect 0.1)")
    assert abs(tau_start - 1.0) < 0.01
    assert abs(tau_end - 0.1) < 0.01

    # Lambda: starts at 0, ends at lambda_max
    lam_start = get_lambda(0, T_warmup=5000, lambda_max=0.1)
    lam_mid = get_lambda(2500, T_warmup=5000, lambda_max=0.1)
    lam_end = get_lambda(5000, T_warmup=5000, lambda_max=0.1)
    print(f"  lambda(0) = {lam_start:.4f} (expect 0.0)")
    print(f"  lambda(2500) = {lam_mid:.4f} (expect 0.05)")
    print(f"  lambda(5000) = {lam_end:.4f} (expect 0.1)")
    assert abs(lam_start) < 1e-6
    assert abs(lam_mid - 0.05) < 0.01
    assert abs(lam_end - 0.1) < 0.01

    print("  PASSED: schedules match paper values")
    print()


def test_full_pipeline():
    """Full end-to-end forward + backward with stub LLM."""
    print("=" * 60)
    print("TEST 6: Full Pipeline (Stub LLM)")
    print("=" * 60)

    # Create model with stub LLM (no real LLaVA needed)
    model = TCViTokModel(
        k=144,
        tpn_config={
            "d_model": 384, "d_text": 512, "d_coarse": 384,
            "n_heads": 4, "d_h": 512, "M": 49, "N": 576,
        },
        use_stub_llm=True,
    )
    model.set_training_phase(0)

    trainable = model.get_num_params(trainable_only=True)
    total = model.get_num_params(trainable_only=False)
    print(f"  Trainable params (Phase 0): {trainable:,}")
    print(f"  Total params: {total:,}")

    # Create dummy data
    dataset = DummyVQADataset(num_samples=8, vocab_size=1000)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    # Forward pass
    batch = next(iter(loader))
    model.train()

    outputs = model(
        images_coarse=batch["images_coarse"],
        images_full=batch["images_full"],
        questions=batch["questions"],
        labels=batch["labels"],
        tau=1.0,
        return_mask=True,
    )

    print(f"  Output keys: {list(outputs.keys())}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Selected tokens shape: {outputs['selected_tokens'].shape}")
    print(f"  Mask shape: {outputs['mask'].shape}")
    print(f"  Scores shape: {outputs['scores'].shape}")

    # Verify selected_tokens has correct shape [B, k, D]
    assert outputs["selected_tokens"].shape == (4, 144, 1024), (
        f"Expected [4, 144, 1024], got {outputs['selected_tokens'].shape}"
    )

    # Compute loss
    loss_dict = compute_total_loss(
        logits=outputs["logits"],
        labels=batch["labels"],
        scores=outputs["scores"],
        mask=outputs["mask"],
        k=144,
    )

    # Backward pass
    loss_dict["total"].backward()

    # Verify gradient flow to TPN
    tpn_grad_norm = 0.0
    for p in model.tpn.parameters():
        if p.grad is not None:
            tpn_grad_norm += p.grad.norm().item()
    print(f"  TPN gradient norm: {tpn_grad_norm:.6f}")
    assert tpn_grad_norm > 0, "TPN received no gradients!"

    print("  PASSED: full pipeline forward + backward works")
    print()


if __name__ == "__main__":
    print("\nTC-ViTok Pipeline Verification")
    print("=" * 60)
    print()

    test_tpn_architecture()
    test_gumbel_sigmoid_ste()
    test_gather_extraction()
    test_losses()
    test_curriculum()
    test_full_pipeline()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print()
    print("System is ready for first experiment.")
    print("Next steps:")
    print("  1. Implement real dataset loaders (VQAv2, GQA)")
    print("  2. Integrate full LLaVA-1.5 (replace StubLLM)")
    print("  3. Run Phase 0 training")
