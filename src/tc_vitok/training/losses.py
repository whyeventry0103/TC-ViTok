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
    lambda_sparsity: float = 0.01,
    lambda_entropy: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Compute total training loss (paper Eq. total_loss).

    Total loss:
        L(t) = L_VQA + lambda(t) * ||s_N||_1/N + mu * L_ent

    The paper defines exactly 3 loss terms. Diversity loss (Eq. div_loss) is
    a separate optional regularizer for query-ignoring mitigation and requires
    paired question data from PairedVQADataset; it is NOT included here.

    Entropy is computed from gather indices (idx), not from a binary mask.
    This matches the gather-based (Option B) approach throughout.

    Args:
        logits: Language model logits (None if pipeline stub)
        labels: Ground truth token IDs (None if pipeline stub)
        scores: Raw TPN importance scores, Shape: [B, N]
        idx: Top-k gather indices from TPN, Shape: [B, k]
        k: Token budget
        N: Total number of spatial positions (576)
        lambda_sparsity: Weight for L1 sparsity (paper: lambda)
        lambda_entropy: Weight for entropy (paper: |mu|)

    Returns:
        loss_dict: Dictionary with all loss components
    """
    # Main VQA loss (or zero if no logits available)
    if logits is not None and labels is not None:
        loss_vqa = vqa_loss(logits, labels)
    else:
        loss_vqa = torch.tensor(0.0, device=scores.device, requires_grad=True)

    # L1 sparsity on RAW SCORES (not mask)
    loss_sparsity = l1_sparsity_loss(scores)

    # Entropy regularization from gather indices
    # Returns positive entropy; we negate to maximize (minimizing -H = maximizing H)
    ent = entropy_from_idx(idx, N=N)
    loss_ent = -ent

    # Total loss: 3 terms per paper Eq. total_loss
    loss_total = (
        loss_vqa
        + lambda_sparsity * loss_sparsity
        + lambda_entropy * loss_ent
    )

    return {
        "total": loss_total,
        "vqa": loss_vqa.detach(),
        "sparsity": loss_sparsity.detach(),
        "entropy": loss_ent.detach(),
    }
