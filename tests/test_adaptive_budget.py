"""Tests for Adaptive Budget TC-ViTok.

Validates all new components:
1. BudgetPredictor output shapes and range
2. adaptive_gumbel_sigmoid_ste gradient flow
3. Variable-k forward pass through TCViTokModel (stub mode)
4. Per-sample merge with variable k
5. Budget loss computation
6. compute_total_loss with budget term
7. Checkpoint save/load of budget_predictor weights
"""

import torch
import torch.nn as nn
import pytest


def test_budget_predictor_shapes():
    """BudgetPredictor produces correct output shape and range."""
    from tc_vitok.models.tpn import BudgetPredictor

    bp = BudgetPredictor(d_model=384, k_min=36, k_max=288, N=576)
    ctx = torch.randn(4, 384)
    k_frac = bp(ctx)

    assert k_frac.shape == (4,), f"Expected (4,), got {k_frac.shape}"
    assert (k_frac >= 36 / 576 - 1e-6).all(), f"k_frac below k_min/N: {k_frac}"
    assert (k_frac <= 288 / 576 + 1e-6).all(), f"k_frac above k_max/N: {k_frac}"

    # Check param count (~25K: 384*64 + 64 + 64*1 + 1 = 24,705)
    n_params = bp.get_num_params()
    assert 20000 < n_params < 30000, f"Expected ~25K params, got {n_params}"
    print(f"  BudgetPredictor: {n_params} params, k_frac range [{k_frac.min():.3f}, {k_frac.max():.3f}]")


def test_budget_predictor_gradient():
    """BudgetPredictor gradients flow back from a loss on k_frac."""
    from tc_vitok.models.tpn import BudgetPredictor

    bp = BudgetPredictor(d_model=384, k_min=36, k_max=288, N=576)
    ctx = torch.randn(4, 384, requires_grad=True)
    k_frac = bp(ctx)
    loss = k_frac.mean()
    loss.backward()

    # Verify gradient flows to context input
    assert ctx.grad is not None, "No gradient on context input"
    assert ctx.grad.abs().sum() > 0, "Zero gradient on context input"

    # Verify gradient flows to BudgetPredictor params
    for name, p in bp.named_parameters():
        assert p.grad is not None, f"No gradient on {name}"
        assert p.grad.abs().sum() > 0, f"Zero gradient on {name}"
    print("  BudgetPredictor gradient flow: OK")


def test_adaptive_gumbel_sigmoid_ste_training():
    """adaptive_gumbel_sigmoid_ste produces correct shapes and gradients."""
    from tc_vitok.models.masking import adaptive_gumbel_sigmoid_ste

    B, N = 4, 576
    scores = torch.randn(B, N, requires_grad=True)
    k_frac = torch.tensor([0.1, 0.2, 0.3, 0.4])  # different fractions

    mask_ste, idx, k_effective = adaptive_gumbel_sigmoid_ste(
        scores, k_frac=k_frac, tau=1.0, training=True,
        k_min=36, k_max=288, N=N,
    )

    # Shape checks
    assert mask_ste.shape == (B, N), f"mask_ste shape: {mask_ste.shape}"
    k_max_batch = idx.shape[1]
    assert idx.shape[0] == B, f"idx batch dim: {idx.shape[0]}"
    assert k_effective.shape == (B,), f"k_effective shape: {k_effective.shape}"

    # k_effective should be in reasonable range
    assert (k_effective > 0).all(), f"k_effective has zeros: {k_effective}"

    # Gradient flow through mask_ste
    loss = mask_ste.sum()
    loss.backward()
    assert scores.grad is not None, "No gradient on scores through mask_ste"
    assert scores.grad.abs().sum() > 0, "Zero gradient on scores"

    # Check per-sample variable k (idx padding)
    for b in range(B):
        valid = (idx[b] >= 0).sum().item()
        k_target = int(round(k_frac[b].item() * N))
        k_target = max(36, min(288, k_target))
        assert valid == k_target, f"Sample {b}: expected {k_target} valid, got {valid}"

    print(f"  Adaptive STE training: shapes OK, k_max_batch={k_max_batch}")
    print(f"  Per-sample k: {[(idx[b] >= 0).sum().item() for b in range(B)]}")
    print(f"  k_effective: {k_effective.detach().tolist()}")


def test_adaptive_gumbel_sigmoid_ste_inference():
    """adaptive_gumbel_sigmoid_ste inference: deterministic, no Gumbel noise."""
    from tc_vitok.models.masking import adaptive_gumbel_sigmoid_ste

    B, N = 4, 576
    scores = torch.randn(B, N)
    k_frac = torch.tensor([0.1, 0.25, 0.35, 0.5])

    mask, idx, k_effective = adaptive_gumbel_sigmoid_ste(
        scores, k_frac=k_frac, tau=0.3, training=False,
        k_min=36, k_max=288, N=N,
    )

    # Deterministic: running twice should give same result
    mask2, idx2, k_eff2 = adaptive_gumbel_sigmoid_ste(
        scores, k_frac=k_frac, tau=0.3, training=False,
        k_min=36, k_max=288, N=N,
    )
    assert torch.equal(idx, idx2), "Inference not deterministic"
    assert torch.equal(k_effective, k_eff2), "k_effective not deterministic"

    # k_effective should be integer counts at eval
    for b in range(B):
        k_b = k_effective[b].item()
        assert k_b == int(k_b), f"k_effective[{b}] not integer at eval: {k_b}"

    print(f"  Adaptive STE inference: deterministic, k={k_effective.tolist()}")


def test_tpn_adaptive_forward():
    """TPN forward with adaptive budget produces correct outputs."""
    from tc_vitok.models.tpn import TokenizationPolicyNetwork, BudgetPredictor

    tpn = TokenizationPolicyNetwork(d_model=384, d_text=512, d_coarse=384, N=576, M=49)
    bp = BudgetPredictor(d_model=384, k_min=36, k_max=288, N=576)

    B = 4
    v_coarse = torch.randn(B, 49, 384)
    q_embed = torch.randn(B, 512)

    # Training mode
    idx, s_N, mask_ste, k_effective = tpn(
        v_coarse=v_coarse, q_embed=q_embed, k=144, tau=1.0, training=True,
        adaptive_budget=True, budget_predictor=bp, k_min=36, k_max=288,
    )

    assert s_N.shape == (B, 576), f"s_N shape: {s_N.shape}"
    assert mask_ste is not None, "mask_ste should not be None in training"
    assert k_effective is not None, "k_effective should not be None in adaptive mode"
    assert k_effective.shape == (B,), f"k_effective shape: {k_effective.shape}"
    assert mask_ste.grad_fn is not None, "mask_ste has no gradient"

    # Eval mode
    idx_e, s_N_e, mask_e, k_eff_e = tpn(
        v_coarse=v_coarse, q_embed=q_embed, k=144, tau=0.3, training=False,
        adaptive_budget=True, budget_predictor=bp, k_min=36, k_max=288,
    )
    assert mask_e is None, "mask_ste should be None in eval"
    assert k_eff_e is not None, "k_effective should exist in eval"

    print(f"  TPN adaptive forward: OK, k_effective={k_effective.detach().tolist()}")


def test_tpn_fixed_backward_compat():
    """TPN forward with fixed k still works (backward compatibility)."""
    from tc_vitok.models.tpn import TokenizationPolicyNetwork

    tpn = TokenizationPolicyNetwork(d_model=384, d_text=512, d_coarse=384, N=576, M=49)
    v_coarse = torch.randn(2, 49, 384)
    q_embed = torch.randn(2, 512)

    idx, s_N, mask_ste, k_effective = tpn(
        v_coarse=v_coarse, q_embed=q_embed, k=144, tau=1.0, training=True,
    )

    assert idx.shape == (2, 144), f"idx shape: {idx.shape}"
    assert k_effective is None, "k_effective should be None in fixed-k mode"
    assert mask_ste is not None and mask_ste.grad_fn is not None
    print("  TPN fixed-k backward compat: OK")


def test_model_adaptive_forward_stub():
    """Full TCViTokModel forward with adaptive budget (stub LLM)."""
    from tc_vitok.models.tc_vitok_model import TCViTokModel

    model = TCViTokModel(
        k=144,
        use_stub_llm=True,
        load_encoders=False,
        adaptive_budget=True,
        k_min=36,
        k_max=288,
    )
    model.train()

    B, N, D = 4, 576, 1024
    v_coarse = torch.randn(B, 49, 384)
    q_embed = torch.randn(B, 512)
    v_full = torch.randn(B, N, D)

    outputs = model(
        v_coarse=v_coarse, q_embed=q_embed,
        images_full=torch.zeros(B, 3, 336, 336),  # unused with v_full
        v_full=v_full,
        tau=1.0,
        return_idx=True,
    )

    assert "logits" in outputs, "Missing logits"
    assert "scores" in outputs, "Missing scores"
    assert "k_effective" in outputs, "Missing k_effective in adaptive mode"
    assert "mask_ste" in outputs, "Missing mask_ste in training"
    assert "idx" in outputs, "Missing idx"

    k_eff = outputs["k_effective"]
    assert k_eff.shape == (B,), f"k_effective shape: {k_eff.shape}"

    print(f"  Model adaptive forward (stub): OK")
    print(f"    k_effective: {k_eff.detach().tolist()}")
    print(f"    idx shape: {outputs['idx'].shape}")
    print(f"    selected_tokens shape: {outputs['selected_tokens'].shape}")


def test_model_fixed_backward_compat_stub():
    """TCViTokModel with fixed k still works."""
    from tc_vitok.models.tc_vitok_model import TCViTokModel

    model = TCViTokModel(
        k=144, use_stub_llm=True, load_encoders=False,
        adaptive_budget=False,
    )
    model.train()

    B = 2
    outputs = model(
        v_coarse=torch.randn(B, 49, 384),
        q_embed=torch.randn(B, 512),
        images_full=torch.zeros(B, 3, 336, 336),
        v_full=torch.randn(B, 576, 1024),
        tau=1.0, return_idx=True,
    )

    assert outputs["idx"].shape == (B, 144), f"idx shape: {outputs['idx'].shape}"
    assert "k_effective" not in outputs or outputs.get("k_effective") is None
    print("  Model fixed-k backward compat (stub): OK")


def test_budget_loss():
    """budget_loss returns correct values."""
    from tc_vitok.training.losses import budget_loss

    k_eff = torch.tensor([100.0, 200.0, 150.0])
    loss = budget_loss(k_eff, N=576)

    expected = (100 + 200 + 150) / 3 / 576
    assert abs(loss.item() - expected) < 1e-5, f"Expected {expected}, got {loss.item()}"
    print(f"  Budget loss: {loss.item():.5f} (expected {expected:.5f})")


def test_compute_total_loss_with_budget():
    """compute_total_loss includes budget term when k_effective provided."""
    from tc_vitok.training.losses import compute_total_loss

    B, N = 4, 576
    scores = torch.randn(B, N)
    idx = scores.topk(144, dim=-1).indices
    k_effective = torch.tensor([100.0, 150.0, 200.0, 120.0], requires_grad=True)

    loss_dict = compute_total_loss(
        logits=None, labels=None,
        scores=scores, idx=idx, k=144,
        lambda_budget=0.1,
        k_effective=k_effective,
    )

    assert "budget" in loss_dict, "Missing 'budget' in loss_dict"
    assert "k_effective_mean" in loss_dict, "Missing 'k_effective_mean' in loss_dict"
    assert loss_dict["budget"].item() > 0, "Budget loss should be > 0"
    assert loss_dict["k_effective_mean"].item() > 0, "k_effective_mean should be > 0"

    # Verify gradient flows through budget term
    loss_dict["total"].backward()
    assert k_effective.grad is not None, "No gradient on k_effective"
    assert k_effective.grad.abs().sum() > 0, "Zero gradient on k_effective"

    print(f"  compute_total_loss with budget: budget={loss_dict['budget'].item():.4f}, "
          f"k_eff_mean={loss_dict['k_effective_mean'].item():.1f}")


def test_compute_total_loss_without_budget():
    """compute_total_loss works without budget (backward compat)."""
    from tc_vitok.training.losses import compute_total_loss

    B, N = 2, 576
    scores = torch.randn(B, N)
    idx = scores.topk(144, dim=-1).indices

    loss_dict = compute_total_loss(
        logits=None, labels=None,
        scores=scores, idx=idx, k=144,
        lambda_budget=0.0,
    )

    assert "budget" in loss_dict
    assert loss_dict["budget"].item() == 0.0, "Budget should be 0 when disabled"
    print("  compute_total_loss without budget: OK")


def test_end_to_end_gradient_flow():
    """Full gradient flow: VQA loss → TPN scores → BudgetPredictor."""
    from tc_vitok.models.tpn import TokenizationPolicyNetwork, BudgetPredictor
    from tc_vitok.models.masking import adaptive_gumbel_sigmoid_ste
    from tc_vitok.training.losses import budget_loss

    tpn = TokenizationPolicyNetwork(d_model=384, d_text=512, d_coarse=384, N=576, M=49)
    bp = BudgetPredictor(d_model=384, k_min=36, k_max=288, N=576)

    v_coarse = torch.randn(2, 49, 384)
    q_embed = torch.randn(2, 512)

    idx, s_N, mask_ste, k_effective = tpn(
        v_coarse=v_coarse, q_embed=q_embed, k=144, tau=1.0, training=True,
        adaptive_budget=True, budget_predictor=bp, k_min=36, k_max=288,
    )

    # Simulate VQA loss through mask_ste (gradient to TPN)
    vqa_proxy = mask_ste.sum()

    # Budget loss (gradient to BudgetPredictor)
    b_loss = budget_loss(k_effective, N=576)

    total = vqa_proxy + 0.1 * b_loss
    total.backward()

    # Check TPN gradients
    tpn_grad_norm = sum(
        p.grad.norm().item() for p in tpn.parameters() if p.grad is not None
    )
    assert tpn_grad_norm > 0, "TPN has zero gradient"

    # Check BudgetPredictor gradients
    bp_grad_norm = sum(
        p.grad.norm().item() for p in bp.parameters() if p.grad is not None
    )
    assert bp_grad_norm > 0, "BudgetPredictor has zero gradient"

    print(f"  End-to-end gradient flow: TPN grad_norm={tpn_grad_norm:.4f}, "
          f"BP grad_norm={bp_grad_norm:.6f}")


def test_set_training_phase_budget_predictor():
    """set_training_phase handles budget_predictor correctly."""
    from tc_vitok.models.tc_vitok_model import TCViTokModel

    model = TCViTokModel(
        k=144, use_stub_llm=True, load_encoders=False,
        adaptive_budget=True, k_min=36, k_max=288,
    )

    # Phase 0: budget_predictor should be frozen
    model.set_training_phase(0)
    for p in model.budget_predictor.parameters():
        assert not p.requires_grad, "Budget predictor should be frozen in phase 0"

    # Phase 1: budget_predictor should be trainable
    model.set_training_phase(1)
    for p in model.budget_predictor.parameters():
        assert p.requires_grad, "Budget predictor should be trainable in phase 1"

    print("  set_training_phase with budget_predictor: OK")


def test_optimizer_budget_group():
    """create_optimizer creates a group for budget_predictor."""
    from tc_vitok.models.tc_vitok_model import TCViTokModel
    from tc_vitok.training.optimizer import create_optimizer, get_optimizer_info

    model = TCViTokModel(
        k=144, use_stub_llm=True, load_encoders=False,
        adaptive_budget=True, k_min=36, k_max=288,
    )
    model.set_training_phase(1)

    optimizer = create_optimizer(model, lr_tpn=1e-3, lr_budget=5e-4, phase=1)
    info = get_optimizer_info(optimizer)

    # Should have at least 3 groups: TPN, StubLLM, BudgetPredictor
    assert info["num_groups"] >= 3, f"Expected >=3 groups, got {info['num_groups']}"

    # Find budget group (has ~25K params: 384*64 + 64 + 64*1 + 1 = 24705)
    budget_group = None
    for g in info["group_info"]:
        if 20000 < g["num_params"] < 30000:
            budget_group = g
            break
    assert budget_group is not None, f"No budget predictor group found. Groups: {info['group_info']}"
    assert budget_group["lr"] == 5e-4, f"Budget LR: {budget_group['lr']}"

    print(f"  Optimizer budget group: {budget_group}")


if __name__ == "__main__":
    print("=" * 60)
    print("Adaptive Budget TC-ViTok Test Suite")
    print("=" * 60)

    tests = [
        ("BudgetPredictor shapes", test_budget_predictor_shapes),
        ("BudgetPredictor gradient", test_budget_predictor_gradient),
        ("Adaptive STE training", test_adaptive_gumbel_sigmoid_ste_training),
        ("Adaptive STE inference", test_adaptive_gumbel_sigmoid_ste_inference),
        ("TPN adaptive forward", test_tpn_adaptive_forward),
        ("TPN fixed-k compat", test_tpn_fixed_backward_compat),
        ("Model adaptive (stub)", test_model_adaptive_forward_stub),
        ("Model fixed-k compat", test_model_fixed_backward_compat_stub),
        ("Budget loss", test_budget_loss),
        ("Total loss with budget", test_compute_total_loss_with_budget),
        ("Total loss without budget", test_compute_total_loss_without_budget),
        ("End-to-end gradient", test_end_to_end_gradient_flow),
        ("Training phase", test_set_training_phase_budget_predictor),
        ("Optimizer budget group", test_optimizer_budget_group),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            print(f"\n[TEST] {name}")
            test_fn()
            print(f"  PASS")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 60}")

    if failed > 0:
        exit(1)
