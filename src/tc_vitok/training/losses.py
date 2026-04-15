"""Loss functions for TC-ViTok training.

This module implements all loss terms used during training:
1. VQA loss: Standard cross-entropy for answer generation
2. L1 sparsity: mean(|scores|) — encourages sparse raw score vectors
3. Entropy regularization: Shannon entropy of spatial selection distribution

Total loss (paper Eq. total_loss):
    L(t) = L_VQA + lambda(t) * ||s_N||_1/N + mu * L_ent

Diversity loss (paper Eq. div_loss) is a separate optional regularizer
for mitigating query-ignoring behavior; it requires paired question data
and is NOT part of the main training loss by default.

Paper reference: Section 7.1 (Training Objective)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def vqa_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Standard VQA cross-entropy loss.

    Args:
        logits: Language model logits, Shape: [B, seq_len, vocab_size]
        labels: Ground truth token IDs, Shape: [B, seq_len]
        ignore_index: Token ID to ignore (padding)

    Returns:
        loss: Scalar cross-entropy loss
    """
    # Align sequence lengths (handles stub mode where logits seq_len != labels seq_len)
    if logits.dim() == 3 and labels.dim() == 2:
        min_len = min(logits.shape[1], labels.shape[1])
        logits = logits[:, :min_len].contiguous()
        labels = labels[:, :min_len].contiguous()
    # nan_to_num + clamp: handles both NaN (from BF16/FP16 attention overflow)
    # and ±inf (from BF16 magnitude overflow). torch.clamp alone does NOT
    # remove NaN — clamp(nan, -100, 100) = nan. nan_to_num replaces nan→0,
    # +inf→100, -inf→-100 before CE loss.
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=100.0, neginf=-100.0)
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)
    return F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index)


def l1_sparsity_loss(scores: torch.Tensor) -> torch.Tensor:
    """L1 sparsity regularization on raw TPN scores.

    Paper formulation: L_comp = ||s_N||_1 / N = mean(|s_i|)

    This is NOT |sum(mask) - k| (which is constant under hard top-k).
    Minimizing mean(|scores|) pushes most scores toward zero; via the
    Gumbel-Sigmoid, near-zero scores map to near-zero selection probability.

    Args:
        scores: Raw TPN importance scores, Shape: [B, N]

    Returns:
        loss: Scalar mean absolute score
    """
    return scores.abs().mean()


def entropy_loss(mask: torch.Tensor) -> torch.Tensor:
    """Spatial entropy regularization to prevent selection collapse.

    Uses the paper's formulation (Section 7.1 / 9.1):
        p_i = m̄_i / Σ_j m̄_j   (normalized batch selection frequency)
        H = -Σ_i p_i log(p_i)    (Shannon entropy)

    We return -H so that minimizing the loss MAXIMIZES entropy.
    Higher entropy = more diverse spatial selection = less collapse.

    Args:
        mask: Binary selection mask, Shape: [B, N]

    Returns:
        loss: Scalar (-entropy), minimizing this maximizes spatial diversity
    """
    from ..models.masking import compute_spatial_entropy
    return -compute_spatial_entropy(mask)


def diversity_loss(
    mask_1: torch.Tensor,
    mask_2: torch.Tensor,
    q_embed_1: torch.Tensor,
    q_embed_2: torch.Tensor,
    gamma: float = 0.5,
) -> torch.Tensor:
    """Query-diversity loss to prevent query-ignoring behavior.

    Paper Eq. (div_loss):
        L_div = sum_b max(0, sim(m1, m2) - gamma) * (1 - sim(q1, q2))

    Requires paired questions for the same image. Use with PairedVQADataset.

    Args:
        mask_1: Binary mask for question 1, Shape: [B, N]
        mask_2: Binary mask for question 2, Shape: [B, N]
        q_embed_1: Text embedding for question 1, Shape: [B, d_q]
        q_embed_2: Text embedding for question 2, Shape: [B, d_q]
        gamma: Similarity threshold (default 0.5)

    Returns:
        loss: Scalar diversity loss
    """
    from ..models.masking import compute_diversity_loss
    return compute_diversity_loss(mask_1, mask_2, q_embed_1, q_embed_2, gamma)


def entropy_from_idx(idx: torch.Tensor, N: int = 576) -> torch.Tensor:
    """Compute Shannon entropy of spatial selection from gather indices.

    Reconstructs batch-mean selection frequency from top-k indices,
    then computes Shannon entropy. This matches the planv2 spec
    (Section Stage E: Loss Computation) where entropy is computed
    from idx rather than from a binary mask.

    Args:
        idx: Top-k gather indices from TPN, Shape: [B, k]
        N: Total number of spatial positions (576)

    Returns:
        entropy: Scalar Shannon entropy (higher = more diverse selection)
    """
    eps = 1e-10
    B = idx.shape[0]

    # Reconstruct batch-mean selection frequency from indices.
    # Filter out -1 padding from adaptive budget mode.
    flat_idx = idx.flatten()
    valid = flat_idx >= 0
    flat_idx = flat_idx[valid]

    m_bar = torch.zeros(N, device=idx.device)
    m_bar.scatter_add_(0, flat_idx,
                       torch.ones_like(flat_idx, dtype=torch.float))
    m_bar = m_bar / B  # normalize by batch size

    # Normalize to probability distribution
    p_hat = m_bar / (m_bar.sum() + eps)

    # Shannon entropy: H = -Σ p_i log(p_i)
    entropy = -(p_hat * torch.log(p_hat + eps)).sum()

    return entropy


def spatial_spread_loss(
    mask_soft: torch.Tensor,
    N: int = 576,
    grid: int = 24,
) -> torch.Tensor:
    """Penalize spatial clustering in the soft token selection distribution.

    Maximizes variance of the expected selection position, forcing TPN to
    spread selections across the image rather than clustering in one region.
    Differentiable via mask_soft (sigmoid(s_N / tau)) so gradient flows to
    ALL 576 TPN scores at every step.

    Anneal to zero after step 30k (see get_lambda_spread in curriculum.py) so
    late-training VQA gradient can freely zoom in for fine-grained questions.

    Args:
        mask_soft: Soft selection weights [B, N] — sigmoid(s_N / tau)
        N: Total tokens (576)
        grid: Grid side length (24 for 24×24)

    Returns:
        loss: -(var_row + var_col).mean() — minimizing maximizes spatial spread
    """
    device = mask_soft.device
    positions = torch.arange(N, device=device).float()
    row = (positions // grid)   # [N]
    col = (positions % grid)    # [N]

    # Normalize to probability distribution over spatial positions
    p = mask_soft / (mask_soft.sum(dim=-1, keepdim=True) + 1e-8)  # [B, N]

    E_row = (p * row).sum(dim=-1)    # [B]
    E_col = (p * col).sum(dim=-1)    # [B]

    var_row = (p * (row - E_row.unsqueeze(-1)) ** 2).sum(dim=-1)  # [B]
    var_col = (p * (col - E_col.unsqueeze(-1)) ** 2).sum(dim=-1)  # [B]

    return -(var_row + var_col).mean()


def attention_distillation_loss(
    s_N: torch.Tensor,
    teacher_importance: torch.Tensor,
    T: float = 1.0,
    teacher_scale: float = 10.0,
) -> torch.Tensor:
    """KL divergence between TPN score distribution and teacher importance map.

    The teacher (frozen base LLaVA with all 576 tokens) produces a per-token
    importance score via cosine similarity of answer hidden states with visual
    hidden states. TPN is trained to match this distribution — dense supervision
    at every step vs sparse VQA cross-entropy.

    Scale mismatch problem (BUG-42):
        teacher_importance ∈ [-1, 1] (cosine similarity), typical std ≈ 0.05-0.15.
        s_N is instance-normalized: mean=0, std=1 by construction.
        With T=2.0, softmax(cosine_sim / 2.0) over 576 tokens produces a near-
        uniform distribution (top token is only 1.16× the uniform baseline).
        The teacher gives essentially no signal — every token looks equally important.

    Fix: multiply teacher_importance by teacher_scale (default 10.0) before
    softmax, mapping cosine sims from std≈0.1 to std≈1, matching s_N's scale.
    Use T=1.0 for both sides — at matched scales, T=1.0 is genuinely "no
    extra smoothing". The 19× concentration at the important token (verified
    empirically) provides a clear, non-degenerate training target.

    Args:
        s_N: TPN instance-normalized scores [B, N], std=1
        teacher_importance: Teacher cosine similarity scores [B, N], detached, std≈0.1
        T: Softmax temperature, applied to both teacher and student (default 1.0)
        teacher_scale: Multiplicative scale applied to teacher_importance before
            softmax to match s_N's dynamic range (default 10.0)

    Returns:
        loss: KL(teacher || student) * T² (temperature-scaled, scalar)
    """
    # Scale teacher from cosine-sim range (std≈0.1) into s_N range (std≈1)
    # before applying the shared temperature T.
    p_t  = F.softmax(teacher_importance * teacher_scale / T, dim=-1)  # [B, N]
    lp_s = F.log_softmax(s_N / T, dim=-1)                            # [B, N]
    return F.kl_div(lp_s, p_t, reduction="batchmean") * (T ** 2)


def boundary_margin_loss(
    s_N: torch.Tensor,
    teacher_importance: torch.Tensor,
    k: int = 144,
    margin: float = 0.1,
) -> torch.Tensor:
    """Enforce correct TPN ordering at the teacher's selection boundary.

    Only constrains TWO tokens: the teacher's k-th and (k+1)-th ranked tokens.
    Does NOT push scores toward ±∞ (compatible with instance normalization).
    Hinge loss: exactly 0 when boundary is correct, bounded positive otherwise.

    Max loss = relu(margin - (-2*max_score)) ≈ 6.1 for instance-norm scores.
    Typical loss ≈ 0.0-0.1 (boundary tokens are usually close in score).

    Args:
        s_N: Instance-normalized TPN scores [B, N], mean=0 std=1
        teacher_importance: Teacher cosine similarity scores [B, N], raw [-1,1]
        k: Token budget (144)
        margin: Minimum score gap at boundary (0.1 ≈ 0.1σ in inst-norm space)

    Returns:
        loss: hinge loss scalar, 0 when boundary correct, >0 when inverted
    """
    # Sort teacher scores descending → get indices of k-th and (k+1)-th tokens
    _, teacher_sorted_idx = teacher_importance.sort(dim=-1, descending=True)
    kth_idx = teacher_sorted_idx[:, k - 1 : k]       # [B, 1] — teacher's last selected
    kp1_idx = teacher_sorted_idx[:, k : k + 1]       # [B, 1] — teacher's first rejected

    s_kth = s_N.gather(1, kth_idx)                    # [B, 1] — TPN score at boundary-in
    s_kp1 = s_N.gather(1, kp1_idx)                    # [B, 1] — TPN score at boundary-out

    return F.relu(margin - (s_kth - s_kp1)).mean()


def slot_orthogonality_loss(v_summary_prenorm: torch.Tensor) -> torch.Tensor:
    """Penalize cosine similarity between all pairs of slot output vectors.

    Operates on pre-LayerNorm slot embeddings (raw cross-attention outputs).
    In D=1024, random vectors have E[cos²] ≈ 1/1024 ≈ 0.001, so the loss
    starts near zero and fires only when slots genuinely over-specialize.

    Loss ∈ [0, 1]: 0 when all slots orthogonal, 1 when all identical.
    Max contribution with lambda=0.01: 0.01 (50× smaller than typical VQA).

    Args:
        v_summary_prenorm: Pre-norm slot outputs [B, n_slots, D]

    Returns:
        loss: mean squared cosine similarity across all slot pairs (scalar)
    """
    v_norm = F.normalize(v_summary_prenorm.float(), dim=-1)   # [B, n_slots, D]
    gram = torch.bmm(v_norm, v_norm.transpose(1, 2))          # [B, n_slots, n_slots]
    n = v_summary_prenorm.shape[1]
    mask = ~torch.eye(n, dtype=torch.bool, device=gram.device) # off-diagonal only
    return gram[:, mask].pow(2).mean()


def residual_pooler_distill_loss(
    pooler_attn: torch.Tensor,
    teacher_importance: torch.Tensor,
    unsel_idx: torch.Tensor,
    teacher_scale: float = 10.0,
) -> torch.Tensor:
    """Rank-banded KL: each pooler slot matches a distinct quartile of teacher residual.

    v5 bug: aggregating all slots via sum(dim=1) gave identical gradients to
    every slot → forced them toward the same distribution → slot collapse
    (slot_orth rose from 0.001 to 0.45, val_vqa regressed from 0.4230 to 0.4650).

    v6 fix: divide the teacher's N-k residual importance into n_slots rank bands.
    Slot 0 must attend to the teacher's top-25% of unselected tokens.
    Slot 1 must attend to 25-50%, Slot 2 to 50-75%, Slot 3 to bottom-25%.
    Each slot gets a DIFFERENT gradient signal → diversity is structurally enforced.

    Implementation detail: rank bands are defined by sorting the teacher's scores
    for the N-k unselected tokens, then masking each slot's target to only its
    assigned quartile positions. The target distribution is the teacher's softmax
    probabilities within that quartile, renormalized to sum=1. Spatial positions
    are preserved (we mask the original N-k distribution, not a sorted list).

    Args:
        pooler_attn: Head-averaged cross-attention weights [B, n_slots, N-k]
        teacher_importance: Teacher cosine similarity scores [B, N], raw ∈ [-1, 1]
        unsel_idx: Indices of unselected tokens [B, N-k] (from complement_gather)
        teacher_scale: Sharpening factor before softmax (matches BUG-41 fix)

    Returns:
        loss: Mean per-slot KL divergence (scalar, bounded ≥ 0, clamped ≤ 10)
    """
    B = pooler_attn.shape[0]
    n_slots = pooler_attn.shape[1]
    Nk = pooler_attn.shape[2]  # N - k

    # Step 1: teacher cosine sims → softmax distribution over all N tokens
    teacher_dist = F.softmax(
        teacher_importance.float() * teacher_scale, dim=-1
    )  # [B, N]

    # Step 2: extract teacher's distribution over the N-k unselected tokens only
    teacher_residual = teacher_dist.gather(1, unsel_idx)                # [B, N-k]

    # Step 3: sort teacher's residual scores to define rank bands
    # Higher teacher score → higher rank → assigned to slot 0 (most important)
    _, rank_order = teacher_residual.sort(dim=-1, descending=True)      # [B, N-k]

    # Step 4: assign each of the N-k positions to one of n_slots bands
    # rank_order[b, 0..Nk/4-1] → slot 0 (top quartile)
    # rank_order[b, Nk/4..Nk/2-1] → slot 1 (second quartile)
    # etc.
    band_size = Nk // n_slots  # 432 // 4 = 108
    remainder = Nk % n_slots   # 432 % 4 = 0

    total_kl = pooler_attn.new_zeros(())

    for slot_i in range(n_slots):
        # Determine which positions belong to this slot's band
        start = slot_i * band_size + min(slot_i, remainder)
        end = (slot_i + 1) * band_size + min(slot_i + 1, remainder)

        # Indices within the N-k ordering that belong to this band
        band_positions = rank_order[:, start:end]                       # [B, band_size]

        # Step 5: build per-slot target — teacher probs at this band, renormalized
        target = teacher_residual.gather(1, band_positions)             # [B, band_size]
        target = target / (target.sum(dim=-1, keepdim=True) + 1e-8)    # sums to 1

        # Step 6: extract this slot's attention at the same positions
        slot_attn = pooler_attn[:, slot_i, :]                           # [B, N-k]
        slot_attn_band = slot_attn.float().gather(1, band_positions)    # [B, band_size]
        slot_attn_band = slot_attn_band / (
            slot_attn_band.sum(dim=-1, keepdim=True) + 1e-8
        )                                                                # sums to 1

        # Step 7: KL(slot_attention_on_band || teacher_target_on_band)
        log_slot = torch.log(slot_attn_band + 1e-8)
        kl = F.kl_div(log_slot, target, reduction="batchmean")
        total_kl = total_kl + kl

    # Average across slots and clamp to prevent runaway KL (defensive, see audit)
    return (total_kl / n_slots).clamp(max=10.0)


def budget_loss(
    k_effective: torch.Tensor,
    N: int = 576,
) -> torch.Tensor:
    """Budget penalty: encourages the model to use fewer tokens.

    Penalizes the fraction of tokens selected, averaged over the batch.
    Fully differentiable through k_effective (sigmoid sum from adaptive STE).

    The budget loss provides direct gradient to the BudgetPredictor:
        L_budget = mean(k_effective / N)  ∈ [k_min/N, k_max/N]

    Higher λ_budget pushes toward fewer tokens (more compression).
    Lower λ_budget allows more tokens (higher accuracy).

    Args:
        k_effective: Differentiable expected token count per sample [B]
        N: Total token positions (576)

    Returns:
        loss: Scalar mean fraction of tokens used
    """
    return (k_effective / N).mean()


def compute_total_loss(
    logits: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
    scores: torch.Tensor,
    idx: torch.Tensor,
    k: int,
    N: int = 576,
    lambda_sparsity: float = 0.0,
    lambda_entropy: float = 0.01,
    lambda_diversity: float = 0.0,
    lambda_spread: float = 0.0,
    lambda_distill: float = 0.0,
    lambda_boundary: float = 0.0,
    lambda_slot_orth: float = 0.0,
    lambda_residual_distill: float = 0.0,
    lambda_budget: float = 0.0,
    boundary_margin: float = 0.1,
    mask_ste: Optional[torch.Tensor] = None,
    mask_soft: Optional[torch.Tensor] = None,
    q_embed: Optional[torch.Tensor] = None,
    teacher_importance: Optional[torch.Tensor] = None,
    v_summary_prenorm: Optional[torch.Tensor] = None,
    pooler_attn: Optional[torch.Tensor] = None,
    unsel_idx: Optional[torch.Tensor] = None,
    k_effective: Optional[torch.Tensor] = None,
    gamma: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Compute total training loss with all regularization terms.

    Total loss:
        L = L_VQA + λ_ent * L_ent + λ_div * L_div + λ_budget * L_budget
              + λ_boundary * L_boundary + λ_slot_orth * L_slot_orth
              + λ_residual * L_residual
              + λ_spread * L_spread + λ_distill * L_distill

    All terms are non-negative or bounded:
        VQA ≥ 0, budget ∈ [k_min/N, k_max/N], boundary ≥ 0, slot_orth ∈ [0,1],
        residual ≥ 0, entropy ∈ [-6.36, 0]
    No unbounded negative terms → VQA dominates the gradient budget.

    Args:
        logits: Language model logits [B, seq_len, vocab_size] (None → VQA=0)
        labels: Ground truth token IDs [B, seq_len] (None → VQA=0)
        scores: Instance-normalized TPN scores [B, N]
        idx: Top-k gather indices [B, k]
        k: Token budget
        N: Total spatial positions (576)
        lambda_sparsity: L1 sparsity weight (0.0 = disabled)
        lambda_entropy: Entropy weight — 0.02 to encourage diverse selection
        lambda_diversity: Within-batch diversity weight (0.1)
        lambda_spread: Spatial spread weight (0.0 = disabled)
        lambda_distill: KL distillation weight (0.0 = disabled)
        lambda_boundary: Boundary margin loss weight
        lambda_slot_orth: Slot orthogonality penalty weight
        lambda_residual_distill: Residual pooler distillation weight
        lambda_budget: Budget penalty weight for adaptive budget mode
        boundary_margin: Hinge margin for boundary loss (0.1)
        mask_ste: STE selection mask [B, N] (optional)
        mask_soft: sigmoid(s_N / tau) [B, N] (optional)
        q_embed: CLIP text embeddings [B, d_q] for diversity loss (optional)
        teacher_importance: Teacher cosine sim scores [B, N] (optional)
        v_summary_prenorm: Pre-LayerNorm pooler outputs [B, n_slots, D] (optional)
        pooler_attn: Head-averaged cross-attn weights [B, n_slots, N-k] (optional)
        unsel_idx: Unselected token indices [B, N-k] for residual distill (optional)
        k_effective: Differentiable expected token count [B] (adaptive budget)
        gamma: Mask similarity threshold for diversity loss (default 0.5)

    Returns:
        loss_dict: {total, vqa, sparsity, entropy, diversity, spread, distill,
                    boundary, slot_orth, residual_distill, budget, k_effective_mean}
    """
    from ..models.masking import compute_spatial_entropy, compute_diversity_loss

    # Main VQA loss (or zero if no logits available)
    if logits is not None and labels is not None:
        loss_vqa = vqa_loss(logits, labels)
    else:
        loss_vqa = torch.tensor(0.0, device=scores.device, requires_grad=True)

    # L1 sparsity on instance-normalized scores (constant ≈ 0.798 due to norm)
    loss_sparsity = l1_sparsity_loss(scores)

    # Entropy: use mask_ste for gradient to ALL N tokens when available.
    if mask_ste is not None and mask_ste.requires_grad:
        ent = compute_spatial_entropy(mask_ste)
    else:
        ent = entropy_from_idx(idx, N=N)
    loss_ent = -ent

    # Within-batch diversity loss on raw score vectors.
    loss_div = scores.new_zeros(())
    if q_embed is not None and lambda_diversity > 0 and scores.shape[0] >= 2:
        B = scores.shape[0]
        half = B // 2
        score_sim = F.cosine_similarity(
            scores[:half].float(), scores[half : 2 * half].float(), dim=-1
        )
        q_sim = F.cosine_similarity(
            q_embed[:half].float(), q_embed[half : 2 * half].float(), dim=-1
        )
        loss_div = (score_sim * (1.0 - q_sim.detach())).mean()

    # Spatial spread loss (v1-v3 legacy, disabled in v4: lambda_spread=0)
    loss_spread = scores.new_zeros(())
    if mask_soft is not None and lambda_spread > 0:
        loss_spread = spatial_spread_loss(mask_soft, N=N)

    # KL distillation (v1-v3 legacy, disabled in v4: lambda_distill=0)
    loss_distill = scores.new_zeros(())
    if teacher_importance is not None and lambda_distill > 0:
        loss_distill = attention_distillation_loss(scores, teacher_importance)

    # Boundary margin loss (v4): enforce TPN agrees with teacher at selection boundary.
    # Hinge loss ≥ 0 always. Typical contribution: 0.05 × 0.05 = 0.0025.
    loss_boundary = scores.new_zeros(())
    if teacher_importance is not None and lambda_boundary > 0:
        loss_boundary = boundary_margin_loss(
            scores, teacher_importance, k=k, margin=boundary_margin,
        )

    # Slot orthogonality penalty (v4): penalize cosine similarity between pooler slots.
    # Loss ∈ [0, 1]. Typical contribution: 0.01 × 0.001 = 0.00001.
    loss_slot_orth = scores.new_zeros(())
    if v_summary_prenorm is not None and lambda_slot_orth > 0:
        loss_slot_orth = slot_orthogonality_loss(v_summary_prenorm)

    # Residual pooler distillation (v5): KL(pooler attention || teacher's residual dist).
    # Teaches the pooler to attend to the background tokens the teacher valued most.
    # Requires: pooler_attn [B, n_slots, N-k], teacher_importance [B, N], unsel_idx [B, N-k].
    # Typical contribution: 0.02 × 0.5 = 0.01 (KL ≈ 0.5 at random initialization).
    loss_residual_distill = scores.new_zeros(())
    if (
        pooler_attn is not None
        and teacher_importance is not None
        and unsel_idx is not None
        and lambda_residual_distill > 0
    ):
        loss_residual_distill = residual_pooler_distill_loss(
            pooler_attn, teacher_importance, unsel_idx
        )

    # Budget penalty (adaptive budget mode): penalizes fraction of tokens used.
    # Provides direct gradient to BudgetPredictor to reduce token count.
    # Typical contribution: 0.1 × 0.25 = 0.025 (at k=144, k/N=0.25).
    loss_budget_val = scores.new_zeros(())
    k_eff_mean = scores.new_zeros(())
    if k_effective is not None and lambda_budget > 0:
        loss_budget_val = budget_loss(k_effective, N=N)
        k_eff_mean = k_effective.detach().mean()

    loss_total = (
        loss_vqa
        + lambda_sparsity * loss_sparsity
        + lambda_entropy * loss_ent
        + lambda_diversity * loss_div
        + lambda_spread * loss_spread
        + lambda_distill * loss_distill
        + lambda_boundary * loss_boundary
        + lambda_slot_orth * loss_slot_orth
        + lambda_residual_distill * loss_residual_distill
        + lambda_budget * loss_budget_val
    )

    return {
        "total": loss_total,
        "vqa": loss_vqa.detach(),
        "sparsity": loss_sparsity.detach(),
        "entropy": loss_ent.detach(),
        "diversity": loss_div.detach() if isinstance(loss_div, torch.Tensor) else torch.tensor(0.0),
        "spread": loss_spread.detach() if isinstance(loss_spread, torch.Tensor) else torch.tensor(0.0),
        "distill": loss_distill.detach() if isinstance(loss_distill, torch.Tensor) else torch.tensor(0.0),
        "boundary": loss_boundary.detach() if isinstance(loss_boundary, torch.Tensor) else torch.tensor(0.0),
        "slot_orth": loss_slot_orth.detach() if isinstance(loss_slot_orth, torch.Tensor) else torch.tensor(0.0),
        "residual_distill": loss_residual_distill.detach() if isinstance(loss_residual_distill, torch.Tensor) else torch.tensor(0.0),
        "budget": loss_budget_val.detach() if isinstance(loss_budget_val, torch.Tensor) else torch.tensor(0.0),
        "k_effective_mean": k_eff_mean if isinstance(k_eff_mean, torch.Tensor) else torch.tensor(0.0),
    }
