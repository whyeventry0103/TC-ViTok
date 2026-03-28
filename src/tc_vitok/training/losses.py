"""Loss functions for TC-ViTok training.

This module implements all loss terms used during training:
1. VQA loss: Standard cross-entropy for answer generation
2. L1 sparsity: mean(|scores|) — encourages sparse raw score vectors
3. Entropy regularization: Shannon entropy of spatial selection distribution
4. Diversity regularization: Prevents query-ignoring behavior

Total loss (paper Eq. 15):
    L = L_VQA + λ(t) * ||s_N||_1/N - |μ| * H(p) + ν * L_div

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


def diversity_loss(mask: torch.Tensor) -> torch.Tensor:
    """Diversity regularization to prevent query-ignoring behavior.

    Returns negative diversity so that minimizing loss increases diversity.

    Args:
        mask: Binary selection mask, Shape: [B, N]

    Returns:
        loss: Scalar (-diversity)
    """
    from ..models.masking import compute_mask_diversity
    return -compute_mask_diversity(mask)


def compute_total_loss(
    logits: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
    scores: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    lambda_sparsity: float = 0.01,
    lambda_entropy: float = 0.01,
    lambda_diversity: float = 0.001,
) -> dict[str, torch.Tensor]:
    """Compute total training loss with all regularization terms.

    Total loss (paper Eq. 15):
        L = L_VQA + λ * mean(|s_N|) - |μ| * H(p) + ν * L_div

    Args:
        logits: Language model logits (None if pipeline stub)
        labels: Ground truth token IDs (None if pipeline stub)
        scores: Raw TPN importance scores, Shape: [B, N]
        mask: Token selection mask from TPN, Shape: [B, N]
        k: Token budget
        lambda_sparsity: Weight for L1 sparsity (paper: λ)
        lambda_entropy: Weight for entropy (paper: |μ|)
        lambda_diversity: Weight for diversity (paper: ν)

    Returns:
        loss_dict: Dictionary with all loss components
    """
    # Main VQA loss (or zero if no logits available)
    if logits is not None and labels is not None:
        loss_vqa = vqa_loss(logits, labels)
    else:
        loss_vqa = torch.tensor(0.0, device=mask.device, requires_grad=True)

    # L1 sparsity on RAW SCORES (not mask)
    loss_sparsity = l1_sparsity_loss(scores)

    # Entropy regularization (returns -entropy, so minimizing = maximizing entropy)
    loss_ent = entropy_loss(mask)

    # Diversity regularization
    loss_div = diversity_loss(mask)

    # Total loss
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
        "diversity": loss_div.detach(),
    }
