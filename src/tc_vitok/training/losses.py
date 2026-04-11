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
    T: float = 2.0,
) -> torch.Tensor:
    """KL divergence between TPN score distribution and teacher importance map.

    The teacher (frozen base LLaVA with all 576 tokens) produces a per-token
    importance score via cosine similarity of answer hidden states with visual
    hidden states. TPN is trained to match this distribution — dense supervision
    at every step vs sparse VQA cross-entropy.

    Args:
        s_N: TPN instance-normalized scores [B, N]
        teacher_importance: Teacher cosine similarity scores [B, N], detached
        T: Temperature for softening distributions (default 2.0)

    Returns:
        loss: KL(teacher || student) * T² (temperature-scaled, scalar)
    """
    p_t = F.softmax(teacher_importance / T, dim=-1)          # [B, N]
    lp_s = F.log_softmax(s_N / T, dim=-1)                   # [B, N]
    return F.kl_div(lp_s, p_t, reduction="batchmean") * (T ** 2)


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
    mask_ste: Optional[torch.Tensor] = None,
    mask_soft: Optional[torch.Tensor] = None,
    q_embed: Optional[torch.Tensor] = None,
    teacher_importance: Optional[torch.Tensor] = None,
    gamma: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Compute total training loss with Phase 1.5 regularization terms.

    Total loss:
        L = L_VQA + λ_spar * ||s_N||_1/N + λ_ent * L_ent + λ_div * L_div
              + λ_spread * L_spread + λ_distill * L_distill

    Args:
        logits: Language model logits [B, seq_len, vocab_size] (None → VQA=0)
        labels: Ground truth token IDs [B, seq_len] (None → VQA=0)
        scores: Instance-normalized TPN scores [B, N]
        idx: Top-k gather indices [B, k]
        k: Token budget
        N: Total spatial positions (576)
        lambda_sparsity: L1 sparsity weight (0.0 = disabled)
        lambda_entropy: Entropy weight — 0.01-0.05 to compete with VQA
        lambda_diversity: Within-batch diversity weight (0.0 = disabled)
        lambda_spread: Spatial spread penalty weight (anneal 0.005→0 after step 30k)
        lambda_distill: Teacher distillation KL weight (0.2 nominal)
        mask_ste: STE selection mask [B, N] — binary fwd, m_soft gradient bwd (optional)
        mask_soft: sigmoid(s_N / tau) [B, N] — for spatial_spread_loss (optional)
        q_embed: CLIP text embeddings [B, d_q] for diversity loss (optional)
        teacher_importance: Frozen teacher per-token cosine sim [B, N] (optional)
        gamma: Mask similarity threshold for diversity loss (default 0.5)

    Returns:
        loss_dict: {total, vqa, sparsity, entropy, diversity, spread, distill}
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
    # mask_ste routes gradient via m_soft STE to both selected AND unselected tokens.
    # Without this, 432 unselected tokens have zero VQA gradient → trapped in static crop.
    if mask_ste is not None and mask_ste.requires_grad:
        ent = compute_spatial_entropy(mask_ste)   # differentiable: grad flows to all N scores
    else:
        ent = entropy_from_idx(idx, N=N)           # fallback for eval (non-differentiable)
    loss_ent = -ent  # negate so minimizing total loss = maximizing entropy

    # Within-batch diversity loss: penalize query-agnostic score distributions.
    #
    # WHY NOT mask_ste: gumbel-noisy binary masks have low cosine similarity by
    # pure chance (different noise per sample). clamp(sim - gamma, 0) always hits 0.
    # The diversity loss was firing 0.0000 every step — dead signal.
    #
    # WHY SCORES: instance-normalized s_N (mean=0, std=1) captures the TRUE spatial
    # bias of TPN regardless of gumbel noise. Static-crop TPN → both samples produce
    # score vectors with same spatial peaks (center) → cosine_sim ≈ 1.0 → loss fires.
    # Query-conditioned TPN → different peaks for different queries → sim ≈ low.
    # No gamma/clamp threshold: always differentiable, gradient active every step.
    #
    #   L_div = mean( cosine_sim(s_N_i, s_N_j) * (1 - q_sim(q_i, q_j)) )
    #
    # High when TPN ignores the query (static bias). Low when TPN routes per query.
    loss_div = scores.new_zeros(())
    if q_embed is not None and lambda_diversity > 0 and scores.shape[0] >= 2:
        B = scores.shape[0]
        half = B // 2
        score_sim = F.cosine_similarity(
            scores[:half].float(), scores[half : 2 * half].float(), dim=-1
        )  # [B//2] — always in [-1, 1], near 1.0 for static-crop TPN
        q_sim = F.cosine_similarity(
            q_embed[:half].float(), q_embed[half : 2 * half].float(), dim=-1
        )  # [B//2] — detached (frozen CLIP), used as per-pair weight
        loss_div = (score_sim * (1.0 - q_sim.detach())).mean()

    # Spatial spread loss: differentiable spatial diversity penalty via mask_soft.
    # Maximizes variance of selection position → TPN spreads across image.
    # Requires mask_soft = sigmoid(s_N / tau). Annealed to 0 after step 30k.
    loss_spread = scores.new_zeros(())
    if mask_soft is not None and lambda_spread > 0:
        loss_spread = spatial_spread_loss(mask_soft, N=N)

    # Teacher distillation: KL(teacher_importance || s_N distribution).
    # Dense token-level supervision from frozen base LLaVA with all 576 tokens.
    loss_distill = scores.new_zeros(())
    if teacher_importance is not None and lambda_distill > 0:
        loss_distill = attention_distillation_loss(scores, teacher_importance)

    loss_total = (
        loss_vqa
        + lambda_sparsity * loss_sparsity
        + lambda_entropy * loss_ent
        + lambda_diversity * loss_div
        + lambda_spread * loss_spread
        + lambda_distill * loss_distill
    )

    return {
        "total": loss_total,
        "vqa": loss_vqa.detach(),
        "sparsity": loss_sparsity.detach(),
        "entropy": loss_ent.detach(),
        "diversity": loss_div.detach() if isinstance(loss_div, torch.Tensor) else torch.tensor(0.0),
        "spread": loss_spread.detach() if isinstance(loss_spread, torch.Tensor) else torch.tensor(0.0),
        "distill": loss_distill.detach() if isinstance(loss_distill, torch.Tensor) else torch.tensor(0.0),
    }
