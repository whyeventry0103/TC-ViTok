"""Loss functions for TC-ViTok training.

This module implements all loss terms used during training:
1. VQA loss: Standard cross-entropy for answer generation
2. L1 sparsity regularization: Encourages using fewer tokens
3. Entropy regularization: Prevents selection collapse
4. Diversity regularization: Prevents query-ignoring behavior

Total loss (Eq. 15):
    L = L_VQA + λ * L_sparsity + λ_ent * L_entropy + λ_div * L_diversity

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

    Computes cross-entropy between predicted answer logits and ground truth labels.

    Args:
        logits: Language model logits, Shape: [batch_size, seq_len, vocab_size]
        labels: Ground truth token IDs, Shape: [batch_size, seq_len]
        ignore_index: Token ID to ignore (padding, etc.)

    Returns:
        loss: Scalar cross-entropy loss

    Paper reference: L_VQA in Eq. 15
    """
    # TODO (Phase C): Implement VQA loss
    # Standard cross-entropy for language modeling

    # Flatten tensors for cross-entropy
    logits_flat = logits.view(-1, logits.size(-1))  # [B*seq_len, vocab_size]
    labels_flat = labels.view(-1)  # [B*seq_len]

    # Compute cross-entropy
    loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=ignore_index,
        reduction="mean",
    )

    return loss


def l1_sparsity_reg(
    mask: torch.Tensor,
    k: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """L1 sparsity regularization on token selection.

    Encourages the model to use exactly k tokens (no more, no less).
    Penalizes deviation from the target token budget.

    Loss (Eq. 16):
        L_sparsity = |sum(mask) - k|

    Args:
        mask: Binary selection mask, Shape: [batch_size, num_tokens]
        k: Target number of tokens (budget)
        reduction: How to reduce across batch ("mean", "sum", "none")

    Returns:
        loss: Scalar sparsity loss

    Notes:
        - In practice, hard top-k ensures exactly k tokens are selected
        - This loss is more useful during soft selection (high τ)
        - Can be replaced with L2 for smoother gradients

    Paper reference: L_sparsity in Eq. 15
    """
    # TODO (Phase C): Implement sparsity regularization

    # Count selected tokens per sample
    num_selected = mask.sum(dim=-1)  # [batch_size]

    # L1 distance from target k
    loss = torch.abs(num_selected - k)

    # Reduction
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def entropy_reg(
    mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Entropy regularization to prevent selection collapse.

    Maximizes the entropy of token selection distribution to encourage
    diverse selection patterns across spatial locations.

    High entropy = tokens selected uniformly across space
    Low entropy = all tokens selected from same region (collapse)

    Args:
        mask: Binary selection mask, Shape: [batch_size, num_tokens]
        reduction: How to reduce across batch

    Returns:
        loss: Scalar entropy loss (negated, so minimizing increases entropy)

    Paper reference: Section 9.1 (Failure Mode #1: Spatial Collapse)
    """
    # TODO (Phase D): Implement entropy regularization
    # Maximize H(mask) = -sum(p * log(p))

    from ..models.masking import compute_mask_entropy

    # Compute entropy (higher is better)
    entropy = compute_mask_entropy(mask)

    # Return negative (so minimizing loss maximizes entropy)
    loss = -entropy

    return loss


def diversity_loss(
    mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Diversity regularization to prevent query-ignoring behavior.

    Penalizes the model for always selecting the same tokens regardless of query.
    Encourages query-conditioned selection patterns.

    Diversity metric:
        D = 1 - max_i P(token i selected)

    High diversity = different tokens selected for different queries
    Low diversity = same tokens always selected (query-ignoring)

    Args:
        mask: Binary selection mask, Shape: [batch_size, num_tokens]
        reduction: How to reduce (currently always mean across batch)

    Returns:
        loss: Scalar diversity loss (negated, so minimizing increases diversity)

    Paper reference: Section 9.1 (Failure Mode #2: Query-Ignoring Policy)
    """
    # TODO (Phase D): Implement diversity loss
    # Maximize diversity across batch

    from ..models.masking import compute_mask_diversity

    # Compute diversity (higher is better)
    diversity = compute_mask_diversity(mask)

    # Return negative (so minimizing loss maximizes diversity)
    loss = -diversity

    return loss


def compute_total_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    lambda_sparsity: float = 0.01,
    lambda_entropy: float = 0.0001,
    lambda_diversity: float = 0.001,
) -> dict[str, torch.Tensor]:
    """Compute total training loss with all regularization terms.

    Total loss (Eq. 15):
        L = L_VQA + λ * L_sparsity + λ_ent * L_entropy + λ_div * L_diversity

    Args:
        logits: Language model logits
        labels: Ground truth token IDs
        mask: Token selection mask from TPN
        k: Token budget
        lambda_sparsity: Weight for sparsity regularization
        lambda_entropy: Weight for entropy regularization
        lambda_diversity: Weight for diversity regularization

    Returns:
        loss_dict: Dictionary with all loss components:
            - total: Total loss for backprop
            - vqa: VQA cross-entropy
            - sparsity: Sparsity regularization
            - entropy: Entropy regularization
            - diversity: Diversity regularization

    Paper reference: Eq. 15, Section 7.1
    """
    # TODO (Phase C): Combine all loss terms

    # Main VQA loss
    loss_vqa = vqa_loss(logits, labels)

    # Sparsity regularization
    loss_sparsity = l1_sparsity_reg(mask, k)

    # Entropy regularization
    loss_ent = entropy_reg(mask)

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
        "vqa": loss_vqa,
        "sparsity": loss_sparsity,
        "entropy": loss_ent,
        "diversity": loss_div,
    }
