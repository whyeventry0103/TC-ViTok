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
    mask_ste: Optional[torch.Tensor] = None,
    q_embed: Optional[torch.Tensor] = None,
    gamma: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Compute total training loss with entropy + diversity regularization.

    Total loss:
        L = L_VQA + lambda * ||s_N||_1/N + mu * L_ent + nu * L_div

    Entropy via mask_ste (NOT idx):
        Using mask_ste (binary forward, m_soft STE backward) provides gradient
        to ALL N token scores, including the N-k unselected tokens. This is the
        key escape route from static crop: unselected tokens receive entropy
        gradient d(H)/d(s_N[b,i]) ∝ m_soft[b,i]*(1-m_soft[b,i])/tau, which
        pushes tokens that are never selected to have higher scores, giving them
        a chance to displace the current static selection.
        Fallback to entropy_from_idx (non-differentiable) when mask_ste is None.

    Diversity via within-batch pairs:
        Pairs adjacent samples in the batch. Penalizes TPN when two samples with
        different query embeddings produce similar selection masks (mask cosine
        sim > gamma). Forces TPN to develop query-conditioned routing.

    Args:
        logits: Language model logits [B, seq_len, vocab_size] (None → VQA=0)
        labels: Ground truth token IDs [B, seq_len] (None → VQA=0)
        scores: Instance-normalized TPN scores [B, N]
        idx: Top-k gather indices [B, k]
        k: Token budget
        N: Total spatial positions (576)
        lambda_sparsity: L1 sparsity weight (0.0 = disabled; instance norm makes it ~constant)
        lambda_entropy: Entropy weight — should be O(0.01-0.05) to compete with VQA
        lambda_diversity: Within-batch diversity weight (0.0 = disabled)
        mask_ste: STE selection mask [B, N] — binary fwd, m_soft gradient bwd (optional)
        q_embed: CLIP text embeddings [B, d_q] for diversity loss (optional)
        gamma: Mask similarity threshold for diversity loss (default 0.5)

    Returns:
        loss_dict: {total, vqa, sparsity, entropy, diversity}
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

    # Within-batch diversity loss: penalize query-agnostic token selection.
    # Pairs adjacent samples in the batch. If two samples with different queries
    # produce similar masks (sim > gamma), the loss fires. Gradient flows via mask_ste
    # to TPN scores, pushing TPN to produce different masks for different queries.
    loss_div = scores.new_zeros(())  # scalar zero, no gradient
    if (mask_ste is not None and q_embed is not None
            and lambda_diversity > 0 and mask_ste.shape[0] >= 2):
        B = mask_ste.shape[0]
        half = B // 2
        loss_div = compute_diversity_loss(
            mask_ste[:half],
            mask_ste[half : 2 * half],
            q_embed[:half],
            q_embed[half : 2 * half],
            gamma=gamma,
        )

    loss_total = (
        loss_vqa
        + lambda_sparsity * loss_sparsity
        + lambda_entropy * loss_ent
        + lambda_diversity * loss_div
    )

    return {
        "total": loss_total,
        "vqa": loss_vqa.detach(),
        "sparsity": loss_sparsity.detach(),
        "entropy": loss_ent.detach(),
        "diversity": loss_div.detach() if isinstance(loss_div, torch.Tensor) else torch.tensor(0.0),
    }
