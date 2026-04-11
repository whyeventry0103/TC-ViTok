"""Phase 1.5 smoke tests: all new components verified with synthetic tensors.

Tests cover every new addition in the Phase 1.5 architecture:
1. complement_gather — correctness + disjoint-cover property
2. sinusoidal_2d_pe — shape, range, distinct rows/cols
3. UnselectedPooler — forward shape, gradient flow
4. Orthogonal deflation — complementarity (zero dot product)
5. spatial_spread_loss — spread > clustered
6. attention_distillation_loss — teacher_scale fix (non-flat distribution)
7. get_lambda_spread — schedule at key boundary steps
8. compute_total_loss — full call with all Phase 1.5 params
9. Stub integration — full forward with UnselectedPooler in stub mode
"""

import math

import pytest
import torch
import torch.nn.functional as F

from tc_vitok.models.tc_vitok_model import (
    UnselectedPooler,
    complement_gather,
    gather_tokens,
    sinusoidal_2d_pe,
)
from tc_vitok.training.curriculum import get_lambda_spread
from tc_vitok.training.losses import (
    attention_distillation_loss,
    compute_total_loss,
    spatial_spread_loss,
)


# ---------------------------------------------------------------------------
# 1. complement_gather
# ---------------------------------------------------------------------------
class TestComplementGather:
    """complement_gather must return exactly the N-k un-selected tokens."""

    def test_output_shapes(self):
        B, N, D, k = 4, 576, 1024, 144
        V = torch.randn(B, N, D)
        idx = torch.stack([torch.randperm(N)[:k] for _ in range(B)])
        v_unsel, unsel_idx = complement_gather(V, idx)
        assert v_unsel.shape == (B, N - k, D), f"v_unsel shape wrong: {v_unsel.shape}"
        assert unsel_idx.shape == (B, N - k), f"unsel_idx shape wrong: {unsel_idx.shape}"

    def test_disjoint_from_selected(self):
        """Selected and unselected index sets must be disjoint."""
        B, N, k = 2, 20, 5
        V = torch.randn(B, N, 8)
        idx = torch.stack([torch.randperm(N)[:k] for _ in range(B)])
        _, unsel_idx = complement_gather(V, idx)
        for b in range(B):
            sel_set = set(idx[b].tolist())
            unsel_set = set(unsel_idx[b].tolist())
            assert sel_set.isdisjoint(unsel_set), (
                f"Batch {b}: selected and unselected share tokens"
            )

    def test_full_cover(self):
        """Union of selected + unselected must cover all N positions."""
        B, N, k = 3, 30, 10
        V = torch.randn(B, N, 8)
        idx = torch.stack([torch.randperm(N)[:k] for _ in range(B)])
        _, unsel_idx = complement_gather(V, idx)
        all_positions = set(range(N))
        for b in range(B):
            combined = set(idx[b].tolist()) | set(unsel_idx[b].tolist())
            assert combined == all_positions, (
                f"Batch {b}: union of sel+unsel does not cover all {N} positions"
            )

    def test_values_match_original(self):
        """Unselected token values must match V at unsel_idx positions."""
        B, N, D, k = 2, 12, 4, 4
        V = torch.randn(B, N, D)
        idx = torch.stack([torch.randperm(N)[:k] for _ in range(B)])
        v_unsel, unsel_idx = complement_gather(V, idx)
        for b in range(B):
            for j in range(N - k):
                pos = unsel_idx[b, j].item()
                assert torch.allclose(v_unsel[b, j], V[b, pos]), (
                    f"Batch {b}, unsel pos {j}: value mismatch"
                )

    def test_no_overlap_with_gather_tokens(self):
        """gather_tokens and complement_gather return no shared positions."""
        B, N, D, k = 2, 576, 1024, 144
        V = torch.randn(B, N, D)
        idx = torch.stack([torch.randperm(N)[:k] for _ in range(B)])
        v_sel = gather_tokens(V, idx)
        v_unsel, unsel_idx = complement_gather(V, idx)
        # Check shapes
        assert v_sel.shape == (B, k, D)
        assert v_unsel.shape == (B, N - k, D)
        # Check indices are complementary
        for b in range(B):
            assert set(idx[b].tolist()).isdisjoint(set(unsel_idx[b].tolist()))


# ---------------------------------------------------------------------------
# 2. sinusoidal_2d_pe
# ---------------------------------------------------------------------------
class TestSinusoidal2DPE:
    """2D sinusoidal PE must have correct shape, dtype, and spatial variety."""

    def test_output_shape(self):
        B, L, D = 4, 432, 1024
        flat_idx = torch.randint(0, 576, (B, L))
        pe = sinusoidal_2d_pe(flat_idx, D=D)
        assert pe.shape == (B, L, D), f"PE shape wrong: {pe.shape}"

    def test_output_dtype(self):
        flat_idx = torch.randint(0, 576, (2, 10))
        pe = sinusoidal_2d_pe(flat_idx, D=64)
        assert pe.dtype == torch.float32, f"PE dtype should be float32, got {pe.dtype}"

    def test_D_must_be_divisible_by_4(self):
        """D=1024 is divisible by 4 — confirm no crash and full D used."""
        flat_idx = torch.randint(0, 576, (2, 10))
        pe = sinusoidal_2d_pe(flat_idx, D=1024)
        # All channels should be non-zero (sin/cos never all-zero unless input=0)
        assert pe.abs().sum() > 0

    def test_distinct_positions_have_distinct_pe(self):
        """Two spatially distinct positions should have different encodings."""
        # position 0 = (row=0, col=0), position 575 = (row=23, col=23)
        flat_idx = torch.tensor([[0, 575]])
        pe = sinusoidal_2d_pe(flat_idx, D=64)
        assert not torch.allclose(pe[0, 0], pe[0, 1]), (
            "Corner tokens (0,0) and (23,23) have identical PE — spatial info lost"
        )

    def test_same_row_different_col_differs(self):
        """Same row, different column must produce different PE."""
        # row 0, col 0 vs row 0, col 1
        flat_idx = torch.tensor([[0, 1]])
        pe = sinusoidal_2d_pe(flat_idx, D=64)
        assert not torch.allclose(pe[0, 0], pe[0, 1])

    def test_same_col_different_row_differs(self):
        """Same column, different row must produce different PE."""
        # row 0, col 0 vs row 1, col 0 (offset=24 for grid=24)
        flat_idx = torch.tensor([[0, 24]])
        pe = sinusoidal_2d_pe(flat_idx, D=64)
        assert not torch.allclose(pe[0, 0], pe[0, 1])

    def test_bounded_values(self):
        """All PE values must lie in [-1, 1] (sin/cos output range)."""
        flat_idx = torch.randint(0, 576, (8, 576))
        pe = sinusoidal_2d_pe(flat_idx, D=1024)
        assert pe.min() >= -1.0 - 1e-5, f"PE below -1: {pe.min()}"
        assert pe.max() <= 1.0 + 1e-5, f"PE above 1: {pe.max()}"


# ---------------------------------------------------------------------------
# 3. UnselectedPooler
# ---------------------------------------------------------------------------
class TestUnselectedPooler:
    """UnselectedPooler compresses N-k unselected tokens to n_slots vectors."""

    def test_output_shape(self):
        pooler = UnselectedPooler(d_model=64, n_slots=4, n_heads=4)
        B, L = 4, 432
        v_unsel = torch.randn(B, L, 64)
        unsel_idx = torch.randint(0, 576, (B, L))
        out = pooler(v_unsel, unsel_idx)
        assert out.shape == (B, 4, 64), f"Output shape wrong: {out.shape}"

    def test_gradient_flows_to_input(self):
        """Gradient must reach v_unsel_orth (i.e., cross-attn is differentiable)."""
        pooler = UnselectedPooler(d_model=64, n_slots=4, n_heads=4)
        v_unsel = torch.randn(4, 100, 64, requires_grad=True)
        unsel_idx = torch.randint(0, 576, (4, 100))
        out = pooler(v_unsel, unsel_idx)
        out.sum().backward()
        assert v_unsel.grad is not None, "No gradient on v_unsel"
        assert torch.any(v_unsel.grad != 0), "All v_unsel gradients are zero"

    def test_gradient_flows_to_slots(self):
        """Gradient must reach the learnable slot parameters."""
        pooler = UnselectedPooler(d_model=64, n_slots=4, n_heads=4)
        v_unsel = torch.randn(2, 50, 64)
        unsel_idx = torch.randint(0, 576, (2, 50))
        out = pooler(v_unsel, unsel_idx)
        out.sum().backward()
        assert pooler.slots.grad is not None, "No gradient on learnable slots"

    def test_different_inputs_different_outputs(self):
        """Different unselected tokens must produce different summaries."""
        pooler = UnselectedPooler(d_model=64, n_slots=4, n_heads=4)
        pooler.eval()
        v1 = torch.randn(1, 100, 64)
        v2 = torch.randn(1, 100, 64)
        idx = torch.randint(0, 576, (1, 100))
        with torch.no_grad():
            out1 = pooler(v1, idx)
            out2 = pooler(v2, idx)
        assert not torch.allclose(out1, out2, atol=1e-4), (
            "Different inputs produced identical pooler outputs"
        )

    def test_batch_independent(self):
        """Each batch element processed independently (no cross-batch contamination)."""
        pooler = UnselectedPooler(d_model=64, n_slots=4, n_heads=4)
        pooler.eval()
        B = 4
        v = torch.randn(B, 50, 64)
        idx = torch.randint(0, 576, (B, 50))
        with torch.no_grad():
            out_batch = pooler(v, idx)
            # Process first element alone
            out_single = pooler(v[:1], idx[:1])
        assert torch.allclose(out_batch[:1], out_single, atol=1e-5), (
            "Batch output for element 0 differs from single-element output"
        )


# ---------------------------------------------------------------------------
# 4. Orthogonal deflation
# ---------------------------------------------------------------------------
class TestOrthogonalDeflation:
    """After deflating v_sel_mean from v_unsel, the result should be orthogonal."""

    def _deflate(self, v_unsel, v_sel_mean):
        """Apply the orthogonal deflation formula from tc_vitok_model.py."""
        v_sel_norm_sq = (v_sel_mean * v_sel_mean).sum(-1, keepdim=True) + 1e-8
        dot = (v_unsel * v_sel_mean.unsqueeze(1)).sum(-1, keepdim=True)  # [B, L, 1]
        proj_coeff = dot / v_sel_norm_sq.unsqueeze(1)                    # [B, L, 1]
        return v_unsel - proj_coeff * v_sel_mean.unsqueeze(1)

    def test_deflated_orthogonal_to_sel_mean(self):
        """v_unsel_orth · v_sel_mean must be ≈ 0 for all samples."""
        B, L, D = 4, 100, 64
        v_unsel = torch.randn(B, L, D)
        v_sel_mean = torch.randn(B, D)  # Mean of selected tokens

        v_unsel_orth = self._deflate(v_unsel, v_sel_mean)

        # dot product of each unsel token with the sel mean should be near-zero
        dots = (v_unsel_orth * v_sel_mean.unsqueeze(1)).sum(-1)  # [B, L]
        assert dots.abs().max() < 1e-4, (
            f"Max dot product after deflation: {dots.abs().max():.6f} (should be ~0)"
        )

    def test_deflation_preserves_shape(self):
        B, L, D = 3, 200, 1024
        v_unsel = torch.randn(B, L, D)
        v_sel_mean = torch.randn(B, D)
        v_orth = self._deflate(v_unsel, v_sel_mean)
        assert v_orth.shape == (B, L, D), f"Shape changed after deflation: {v_orth.shape}"

    def test_already_orthogonal_unchanged(self):
        """If v_unsel is already orthogonal to v_sel_mean, deflation is identity."""
        B, D = 2, 8
        v_sel_mean = torch.zeros(B, D)
        v_sel_mean[:, 0] = 1.0  # unit vector in dim 0
        # v_unsel with zero in dim 0 → already orthogonal
        v_unsel = torch.randn(B, 10, D)
        v_unsel[..., 0] = 0.0  # zero out the shared dimension

        v_orth = self._deflate(v_unsel, v_sel_mean)
        assert torch.allclose(v_orth, v_unsel, atol=1e-5), (
            "Deflation modified already-orthogonal vectors"
        )


# ---------------------------------------------------------------------------
# 5. spatial_spread_loss
# ---------------------------------------------------------------------------
class TestSpatialSpreadLoss:
    """spatial_spread_loss should be lower (more negative) for spread distributions."""

    def _uniform_mask_soft(self, B: int, N: int) -> torch.Tensor:
        """Uniform soft mask: all tokens equally weighted."""
        return torch.ones(B, N) / N

    def _center_mask_soft(self, B: int, N: int, grid: int = 24) -> torch.Tensor:
        """Clustered soft mask: all weight on center tokens."""
        mask = torch.zeros(B, N)
        center = grid // 2 * grid + grid // 2  # center position
        mask[:, center] = 1.0
        return mask

    def test_uniform_more_negative_than_clustered(self):
        """Uniform distribution should minimize loss more (more negative)."""
        B, N, grid = 4, 576, 24
        uniform_loss = spatial_spread_loss(self._uniform_mask_soft(B, N), N=N, grid=grid)
        clustered_loss = spatial_spread_loss(self._center_mask_soft(B, N, grid), N=N, grid=grid)
        assert uniform_loss < clustered_loss, (
            f"Uniform loss {uniform_loss:.4f} should be < clustered {clustered_loss:.4f}"
        )

    def test_loss_is_negative(self):
        """Loss must be negative (we negate variance to minimize = maximize spread)."""
        B, N = 4, 576
        uniform_mask = self._uniform_mask_soft(B, N)
        loss = spatial_spread_loss(uniform_mask, N=N)
        assert loss.item() < 0, f"Spread loss should be negative, got {loss.item():.4f}"

    def test_gradient_flows_to_mask_soft(self):
        """Gradient must reach mask_soft so TPN can learn spatial spread."""
        B, N = 4, 576
        mask_soft = torch.sigmoid(torch.randn(B, N)) / N
        mask_soft = mask_soft.clone().detach().requires_grad_(True)
        loss = spatial_spread_loss(mask_soft, N=N)
        loss.backward()
        assert mask_soft.grad is not None, "No gradient on mask_soft"
        assert torch.any(mask_soft.grad != 0), "All gradients on mask_soft are zero"

    def test_output_is_scalar(self):
        mask_soft = torch.rand(4, 576)
        loss = spatial_spread_loss(mask_soft)
        assert loss.shape == torch.Size([]), f"Loss should be scalar, got shape {loss.shape}"

    def test_identical_batches_same_loss(self):
        """Same mask across all batch elements should give reproducible loss."""
        mask = torch.rand(1, 576).expand(4, -1).contiguous()
        loss1 = spatial_spread_loss(mask)
        loss2 = spatial_spread_loss(mask)
        assert torch.allclose(loss1, loss2), "Identical inputs gave different losses"


# ---------------------------------------------------------------------------
# 6. attention_distillation_loss (BUG-42 teacher_scale fix)
# ---------------------------------------------------------------------------
class TestAttentionDistillationLoss:
    """Verify teacher distribution is non-flat after teacher_scale fix."""

    def test_output_is_scalar(self):
        B, N = 4, 576
        s_N = torch.randn(B, N)
        teacher = torch.randn(B, N) * 0.1  # cosine-sim scale: std ≈ 0.1
        loss = attention_distillation_loss(s_N, teacher)
        assert loss.shape == torch.Size([]), f"Loss should be scalar, got {loss.shape}"

    def test_loss_is_nonnegative(self):
        """KL divergence is always ≥ 0."""
        B, N = 4, 576
        s_N = torch.randn(B, N)
        teacher = torch.randn(B, N) * 0.1
        loss = attention_distillation_loss(s_N, teacher)
        assert loss.item() >= -1e-6, f"KL loss should be ≥ 0, got {loss.item()}"

    def test_teacher_scale_produces_non_flat_distribution(self):
        """With teacher_scale=10, top teacher token >> uniform baseline.

        BUG-42: Without scaling, T=2.0 over cosine_sims ∈ [-1,1] gives
        p_top ≈ 1.16 × uniform. With teacher_scale=10.0, T=1.0:
        top token should be significantly above uniform (> 5×).
        """
        N = 576
        teacher = torch.zeros(1, N)
        teacher[0, 0] = 0.9   # strong peak at position 0 (cosine_sim scale)
        teacher[0, 1:] = -0.1  # low everywhere else

        p_t = F.softmax(teacher * 10.0 / 1.0, dim=-1)  # teacher_scale=10, T=1.0
        uniform_baseline = 1.0 / N
        concentration = (p_t[0, 0] / uniform_baseline).item()

        assert concentration > 5.0, (
            f"After teacher_scale=10, top token should be >5× uniform, "
            f"got {concentration:.1f}× (distribution too flat)"
        )

    def test_flat_without_scale_bug42(self):
        """Verify BUG-42 is real: without teacher_scale, distribution is near-flat."""
        N = 576
        teacher = torch.zeros(1, N)
        teacher[0, 0] = 0.9
        teacher[0, 1:] = -0.1
        # Bug: T=2.0, no scaling (teacher_scale=1)
        p_t_bug = F.softmax(teacher * 1.0 / 2.0, dim=-1)
        uniform_baseline = 1.0 / N
        concentration_bug = (p_t_bug[0, 0] / uniform_baseline).item()
        # Should be very close to uniform (original bug)
        assert concentration_bug < 2.0, (
            f"Expected near-flat distribution without scaling, got {concentration_bug:.1f}×"
        )

    def test_gradient_flows_to_s_N(self):
        """Gradient must reach s_N so TPN can learn to match teacher."""
        B, N = 4, 576
        s_N = torch.randn(B, N, requires_grad=True)
        teacher = torch.randn(B, N).detach() * 0.1
        loss = attention_distillation_loss(s_N, teacher)
        loss.backward()
        assert s_N.grad is not None, "No gradient on s_N"
        assert torch.any(s_N.grad != 0)

    def test_identical_distributions_zero_kl(self):
        """When student == teacher distribution, KL ≈ 0."""
        B, N = 4, 576
        teacher = torch.randn(B, N) * 0.1
        # Set s_N so that softmax(s_N) = softmax(teacher * 10)
        # This means s_N = teacher * 10 (up to additive constant)
        s_N = teacher * 10.0
        loss = attention_distillation_loss(s_N, teacher, T=1.0, teacher_scale=10.0)
        assert loss.item() < 0.01, (
            f"Identical distributions should give KL ≈ 0, got {loss.item():.4f}"
        )


# ---------------------------------------------------------------------------
# 7. get_lambda_spread
# ---------------------------------------------------------------------------
class TestGetLambdaSpread:
    """Annealing schedule: constant until decay_start, linear to 0 at decay_end."""

    def test_before_decay_start(self):
        """Steps before decay_start should return max_val exactly."""
        assert get_lambda_spread(0, max_val=0.005, decay_start=30000, decay_end=50000) == 0.005
        assert get_lambda_spread(1000, max_val=0.005, decay_start=30000, decay_end=50000) == 0.005
        assert get_lambda_spread(29999, max_val=0.005, decay_start=30000, decay_end=50000) == 0.005

    def test_at_decay_start(self):
        """At exactly decay_start, value is still max_val (decay hasn't begun)."""
        val = get_lambda_spread(30000, max_val=0.005, decay_start=30000, decay_end=50000)
        assert val == pytest.approx(0.005, abs=1e-9)

    def test_midpoint_of_decay(self):
        """At midpoint between decay_start and decay_end, value is max_val / 2."""
        val = get_lambda_spread(40000, max_val=0.005, decay_start=30000, decay_end=50000)
        assert val == pytest.approx(0.0025, abs=1e-9)

    def test_at_decay_end(self):
        """At decay_end, value must be exactly 0.0."""
        val = get_lambda_spread(50000, max_val=0.005, decay_start=30000, decay_end=50000)
        assert val == 0.0

    def test_after_decay_end(self):
        """After decay_end, value is clamped at 0.0."""
        assert get_lambda_spread(60000, max_val=0.005, decay_start=30000, decay_end=50000) == 0.0
        assert get_lambda_spread(100000, max_val=0.005, decay_start=30000, decay_end=50000) == 0.0

    def test_linear_decay(self):
        """Values at 25%, 50%, 75% of decay window should be linear."""
        max_val = 0.005
        ds, de = 30000, 50000
        window = de - ds
        for frac, expected_frac in [(0.25, 0.75), (0.50, 0.50), (0.75, 0.25)]:
            step = ds + int(frac * window)
            val = get_lambda_spread(step, max_val=max_val, decay_start=ds, decay_end=de)
            assert val == pytest.approx(max_val * expected_frac, rel=1e-5), (
                f"At step {step} (frac={frac}), expected {max_val * expected_frac:.5f}, got {val:.5f}"
            )

    def test_monotonically_decreasing_in_decay(self):
        """Values must be monotonically non-increasing during decay window."""
        vals = [
            get_lambda_spread(s, max_val=0.005, decay_start=30000, decay_end=50000)
            for s in range(29000, 51000, 1000)
        ]
        for i in range(1, len(vals)):
            assert vals[i] <= vals[i - 1] + 1e-9, (
                f"Non-monotone at index {i}: {vals[i-1]:.6f} → {vals[i]:.6f}"
            )


# ---------------------------------------------------------------------------
# 8. compute_total_loss with Phase 1.5 params
# ---------------------------------------------------------------------------
class TestComputeTotalLossPhase1p5:
    """compute_total_loss integrates all Phase 1.5 loss terms."""

    def _make_inputs(self, B=4, N=576, k=144, vocab=100, seq=20):
        scores = torch.randn(B, N)               # instance-normalized TPN scores
        idx = torch.stack([torch.randperm(N)[:k] for _ in range(B)])
        logits = torch.randn(B, seq, vocab)
        labels = torch.randint(0, vocab, (B, seq))
        mask_soft = torch.sigmoid(scores / 0.5)
        teacher_imp = torch.randn(B, N) * 0.1   # cosine-sim scale
        q_embed = torch.randn(B, 512)
        return scores, idx, logits, labels, mask_soft, teacher_imp, q_embed

    def test_returns_all_keys(self):
        scores, idx, logits, labels, mask_soft, teacher_imp, q_embed = self._make_inputs()
        loss_dict = compute_total_loss(
            logits=logits, labels=labels, scores=scores, idx=idx, k=144,
            lambda_entropy=0.02, lambda_diversity=0.1,
            lambda_spread=0.005, lambda_distill=0.2,
            mask_soft=mask_soft, teacher_importance=teacher_imp, q_embed=q_embed,
        )
        expected_keys = {"total", "vqa", "sparsity", "entropy", "diversity", "spread", "distill"}
        assert set(loss_dict.keys()) == expected_keys, (
            f"Missing keys: {expected_keys - set(loss_dict.keys())}"
        )

    def test_total_has_grad(self):
        """total loss must have grad_fn for backward()."""
        scores = torch.randn(4, 576, requires_grad=True)
        idx = torch.stack([torch.randperm(576)[:144] for _ in range(4)])
        logits = torch.randn(4, 20, 100)
        labels = torch.randint(0, 100, (4, 20))
        mask_soft = torch.sigmoid(scores / 0.5)
        loss_dict = compute_total_loss(
            logits=logits, labels=labels, scores=scores, idx=idx, k=144,
            lambda_entropy=0.02, lambda_diversity=0.1,
            lambda_spread=0.005, lambda_distill=0.0,  # distill off for simplicity
            mask_soft=mask_soft,
        )
        assert loss_dict["total"].grad_fn is not None, "total loss has no grad_fn"

    def test_spread_term_nonzero_when_enabled(self):
        """lambda_spread > 0 with mask_soft must produce nonzero spread loss."""
        scores, idx, logits, labels, mask_soft, _, _ = self._make_inputs()
        loss_dict = compute_total_loss(
            logits=logits, labels=labels, scores=scores, idx=idx, k=144,
            lambda_spread=0.005, mask_soft=mask_soft,
        )
        assert loss_dict["spread"].item() != 0.0, "Spread loss is zero when it should fire"

    def test_spread_term_zero_when_disabled(self):
        """lambda_spread=0 must give zero spread contribution."""
        scores, idx, logits, labels, mask_soft, _, _ = self._make_inputs()
        loss_dict = compute_total_loss(
            logits=logits, labels=labels, scores=scores, idx=idx, k=144,
            lambda_spread=0.0, mask_soft=mask_soft,
        )
        assert loss_dict["spread"].item() == 0.0

    def test_distill_term_nonzero_when_enabled(self):
        """lambda_distill > 0 with teacher_importance must fire."""
        scores, idx, logits, labels, mask_soft, teacher_imp, _ = self._make_inputs()
        loss_dict = compute_total_loss(
            logits=logits, labels=labels, scores=scores, idx=idx, k=144,
            lambda_distill=0.2, teacher_importance=teacher_imp,
        )
        assert loss_dict["distill"].item() >= 0.0
        assert loss_dict["distill"].item() != 0.0, "Distill loss is zero when it should fire"

    def test_distill_term_zero_when_disabled(self):
        """lambda_distill=0 must give zero distill contribution."""
        scores, idx, logits, labels, mask_soft, teacher_imp, _ = self._make_inputs()
        loss_dict = compute_total_loss(
            logits=logits, labels=labels, scores=scores, idx=idx, k=144,
            lambda_distill=0.0, teacher_importance=teacher_imp,
        )
        assert loss_dict["distill"].item() == 0.0

    def test_diversity_term_fires_with_q_embed(self):
        """Diversity loss must be nonzero with q_embed and lambda_diversity > 0."""
        scores, idx, logits, labels, mask_soft, _, q_embed = self._make_inputs()
        loss_dict = compute_total_loss(
            logits=logits, labels=labels, scores=scores, idx=idx, k=144,
            lambda_diversity=0.1, q_embed=q_embed,
        )
        # diversity should be non-trivially nonzero for random scores+queries
        assert loss_dict["diversity"].item() != 0.0

    def test_all_disabled_equals_vqa_plus_sparsity_plus_entropy(self):
        """With all Phase 1.5 terms disabled, total = vqa + sparsity + entropy."""
        B, N, k = 2, 576, 144
        scores = torch.randn(B, N)
        idx = torch.stack([torch.randperm(N)[:k] for _ in range(B)])
        logits = torch.randn(B, 10, 50)
        labels = torch.randint(0, 50, (B, 10))
        loss_dict = compute_total_loss(
            logits=logits, labels=labels, scores=scores, idx=idx, k=k,
            lambda_sparsity=0.0, lambda_entropy=0.01,
            lambda_diversity=0.0, lambda_spread=0.0, lambda_distill=0.0,
        )
        expected = (
            loss_dict["vqa"] + 0.0 * loss_dict["sparsity"] + 0.01 * loss_dict["entropy"]
        )
        assert torch.allclose(loss_dict["total"], expected, atol=1e-5), (
            f"total={loss_dict['total']:.6f}, expected={expected:.6f}"
        )

    def test_backward_does_not_crash(self):
        """Full backward pass through all Phase 1.5 terms must not raise."""
        scores = torch.randn(4, 576, requires_grad=True)
        idx = torch.stack([torch.randperm(576)[:144] for _ in range(4)])
        logits = torch.randn(4, 20, 100)
        labels = torch.randint(0, 100, (4, 20))
        mask_soft = torch.sigmoid(scores.detach() / 0.5).requires_grad_(False)
        teacher_imp = torch.randn(4, 576) * 0.1
        q_embed = torch.randn(4, 512)
        loss_dict = compute_total_loss(
            logits=logits, labels=labels, scores=scores, idx=idx, k=144,
            lambda_entropy=0.02, lambda_diversity=0.1,
            lambda_spread=0.005, lambda_distill=0.2,
            mask_soft=mask_soft, teacher_importance=teacher_imp, q_embed=q_embed,
        )
        loss_dict["total"].backward()  # should not raise
        assert scores.grad is not None


# ---------------------------------------------------------------------------
# 9. Stub integration: full forward with UnselectedPooler
# ---------------------------------------------------------------------------
class TestStubIntegrationPhase1p5:
    """End-to-end forward pass in stub mode verifies all Phase 1.5 glue code."""

    @pytest.fixture
    def stub_model(self):
        """TCViTokModel in stub mode with UnselectedPooler skipped (stub_llm=True)."""
        from tc_vitok.models.tc_vitok_model import TCViTokModel
        m = TCViTokModel(
            k=144,
            load_encoders=False,
            use_stub_llm=True,
        )
        m.eval()
        return m

    def test_model_has_no_unsel_pooler_in_stub(self, stub_model):
        """Stub mode must not instantiate UnselectedPooler (no real CLIP D=1024)."""
        assert stub_model.unsel_pooler is None, (
            "unsel_pooler should be None in stub mode"
        )

    def test_forward_returns_logits(self, stub_model):
        """Stub forward via cached path must return a dict with 'logits' key."""
        B = 2
        # Use cached path: provide pre-computed v_coarse, q_embed, v_full
        # This bypasses frozen encoders (which are None in stub mode)
        v_coarse = torch.randn(B, 49, 384)   # DINOv2-S coarse features
        q_embed = torch.randn(B, 512)         # CLIP text embeddings
        v_full = torch.randn(B, 576, 1024)   # CLIP ViT-L/14 full features
        out = stub_model(
            v_coarse=v_coarse,
            q_embed=q_embed,
            v_full=v_full,
            questions=["what color is the dog?"] * B,
            answer_strings=["brown"] * B,
        )
        assert out is not None, "Forward returned None"
        assert "logits" in out, f"'logits' key missing from output: {list(out.keys())}"

    def test_complement_gather_standalone(self):
        """complement_gather + gather_tokens partition V exactly."""
        B, N, D, k = 2, 576, 1024, 144
        V = torch.randn(B, N, D)
        idx = torch.stack([torch.randperm(N)[:k] for _ in range(B)])

        v_sel = gather_tokens(V, idx)
        v_unsel, unsel_idx = complement_gather(V, idx)

        # No index overlap
        for b in range(B):
            assert set(idx[b].tolist()).isdisjoint(set(unsel_idx[b].tolist()))

        # Perfect partition: sel + unsel = all N tokens
        for b in range(B):
            combined = sorted(idx[b].tolist() + unsel_idx[b].tolist())
            assert combined == list(range(N))

        # Values correct for sel
        for b in range(B):
            for j in range(k):
                assert torch.allclose(v_sel[b, j], V[b, idx[b, j]])

        # Values correct for unsel
        for b in range(B):
            for j in range(N - k):
                assert torch.allclose(v_unsel[b, j], V[b, unsel_idx[b, j]])

    def test_unselected_pooler_standalone(self):
        """UnselectedPooler with realistic dimensions: D=1024, 4 slots."""
        pooler = UnselectedPooler(d_model=1024, n_slots=4, n_heads=4)
        B, L = 4, 432   # N - k = 576 - 144
        v_unsel_orth = torch.randn(B, L, 1024)
        unsel_idx = torch.randint(0, 576, (B, L))
        summary = pooler(v_unsel_orth, unsel_idx)
        assert summary.shape == (B, 4, 1024), f"Summary shape wrong: {summary.shape}"

    def test_phase1p5_pipeline_smoke(self):
        """Smoke: complement_gather → deflation → UnselectedPooler → concat → no crash."""
        B, N, D, k = 2, 576, 1024, 144
        pooler = UnselectedPooler(d_model=D, n_slots=4, n_heads=4)

        V = torch.randn(B, N, D, requires_grad=True)
        idx = torch.stack([torch.randperm(N)[:k] for _ in range(B)])

        # Step 1: gather selected
        v_selected = gather_tokens(V, idx)          # [B, k, D]

        # Step 2: gather unselected
        v_unsel, unsel_idx = complement_gather(V, idx)  # [B, N-k, D]

        # Step 3: orthogonal deflation
        v_sel_mean = v_selected.mean(dim=1)         # [B, D]
        v_sel_norm_sq = (v_sel_mean * v_sel_mean).sum(-1, keepdim=True) + 1e-8
        dot = (v_unsel * v_sel_mean.unsqueeze(1)).sum(-1, keepdim=True)
        proj_coeff = dot / v_sel_norm_sq.unsqueeze(1)
        v_unsel_orth = v_unsel - proj_coeff * v_sel_mean.unsqueeze(1)  # [B, N-k, D]

        # Verify orthogonality
        dots = (v_unsel_orth.detach() * v_sel_mean.detach().unsqueeze(1)).sum(-1)
        assert dots.abs().max() < 1e-3, "Deflation not orthogonal enough"

        # Step 4: pooler → summary
        v_summary = pooler(v_unsel_orth, unsel_idx)  # [B, 4, D]
        assert v_summary.shape == (B, 4, D)

        # Step 5: concatenate for LLM input
        v_input = torch.cat([v_summary, v_selected], dim=1)  # [B, k+4, D]
        assert v_input.shape == (B, k + 4, D)

        # Step 6: backward must not crash
        v_input.sum().backward()
        assert V.grad is not None, "No gradient on V after full pipeline"
