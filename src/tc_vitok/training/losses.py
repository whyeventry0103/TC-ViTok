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

    # Reconstruct batch-mean selection frequency from indices
    m_bar = torch.zeros(N, device=idx.device)
    m_bar.scatter_add_(0, idx.flatten(),
                       torch.ones_like(idx.flatten(), dtype=torch.float))
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
    boundary_margin: float = 0.1,
    mask_ste: Optional[torch.Tensor] = None,
    mask_soft: Optional[torch.Tensor] = None,
    q_embed: Optional[torch.Tensor] = None,
    teacher_importance: Optional[torch.Tensor] = None,
    v_summary_prenorm: Optional[torch.Tensor] = None,
    gamma: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Compute total training loss with Phase 1.5 v4 regularization terms.

    Total loss (v4):
        L = L_VQA + λ_ent * L_ent + λ_div * L_div
              + λ_boundary * L_boundary + λ_slot_orth * L_slot_orth
              + λ_spread * L_spread + λ_distill * L_distill  (both 0 in v4)

    All terms are non-negative or bounded:
        VQA ≥ 0, boundary ≥ 0, slot_orth ∈ [0,1], entropy ∈ [-6.36, 0]
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
        lambda_spread: Spatial spread weight (0.0 in v4 — removed)
        lambda_distill: KL distillation weight (0.0 in v4 — replaced by boundary)
        lambda_boundary: Boundary margin loss weight (0.05 in v4)
        lambda_slot_orth: Slot orthogonality penalty weight (0.01 in v4)
        boundary_margin: Hinge margin for boundary loss (0.1)
        mask_ste: STE selection mask [B, N] (optional)
        mask_soft: sigmoid(s_N / tau) [B, N] (optional, unused in v4)
        q_embed: CLIP text embeddings [B, d_q] for diversity loss (optional)
        teacher_importance: Teacher cosine sim scores [B, N] (optional)
        v_summary_prenorm: Pre-LayerNorm pooler outputs [B, n_slots, D] (optional)
        gamma: Mask similarity threshold for diversity loss (default 0.5)

    Returns:
        loss_dict: {total, vqa, sparsity, entropy, diversity, spread, distill,
                    boundary, slot_orth}
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

    loss_total = (
        loss_vqa
        + lambda_sparsity * loss_sparsity
        + lambda_entropy * loss_ent
        + lambda_diversity * loss_div
        + lambda_spread * loss_spread
        + lambda_distill * loss_distill
        + lambda_boundary * loss_boundary
        + lambda_slot_orth * loss_slot_orth
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
    }
