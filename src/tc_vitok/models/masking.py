"""Gumbel-Sigmoid Straight-Through Estimator for differentiable top-k selection.

This module implements the core masking operation that enables gradient flow
through discrete token selection. The forward pass performs hard top-k selection,
while the backward pass uses soft Gumbel-Sigmoid gradients.

Paper Section 6: "Differentiable Token Selection via Gumbel-Sigmoid STE"
"""

from typing import Optional

import torch
import torch.nn.functional as F


def gumbel_sigmoid_ste(
    scores: torch.Tensor,
    k: int,
    tau: float = 1.0,
    training: bool = True,
    hard: bool = True,
) -> torch.Tensor:
    """Gumbel-Sigmoid Straight-Through Estimator for differentiable top-k selection.

    This function implements the key technique that makes token selection differentiable.
    It combines:
    1. Gumbel noise injection for exploration (training only)
    2. Sigmoid activation with temperature for soft selection
    3. Hard top-k selection in forward pass
    4. Soft gradient in backward pass (straight-through trick)

    Forward pass (Eq. 7-9):
        g_i ~ Gumbel(0, 1)
        z_i = σ((s_i + g_i) / τ)
        m_i = 1[z_i ∈ top-k(z)]

    Backward pass (Eq. 10-12):
        ∂L/∂s_i = ∂L/∂z_i * ∂z_i/∂s_i
        where z_i is the SOFT Gumbel-Sigmoid output (not the hard mask)

    Args:
        scores: Raw importance scores from TPN
                Shape: [batch_size, num_tokens] (typically [B, 576])
        k: Number of tokens to select (budget)
        tau: Temperature parameter
             - High τ → softer selection, more exploration
             - Low τ → sharper selection, more exploitation
             - Annealed during training via curriculum (see curriculum.py)
        training: If True, inject Gumbel noise; else deterministic
        hard: If True, use hard top-k in forward; else use soft selection

    Returns:
        mask: Binary selection mask, Shape: [batch_size, num_tokens]
              - Forward: hard top-k selection (0s and 1s)
              - Backward: gradients from soft Gumbel-Sigmoid

    Notes:
        - The straight-through trick is implemented via:
          mask = hard_mask - soft_mask.detach() + soft_mask
        - This makes forward = hard, backward = soft
        - Critical for training stability (Section 9.2)

    Paper references:
        - Gumbel-Sigmoid formulation: Eq. 7-9
        - Gradient flow: Eq. 10-12
        - Temperature scheduling: Eq. 13
    """
    batch_size, num_tokens = scores.shape

    # TODO (Phase C): Implement Gumbel-Sigmoid STE

    # Step 1: Add Gumbel noise during training (Eq. 7-8)
    if training:
        # Sample Gumbel noise: g ~ Gumbel(0, 1)
        # Using the log(-log(uniform)) trick
        uniform = torch.rand_like(scores)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-10) + 1e-10)
        noisy_scores = scores + gumbel_noise
    else:
        noisy_scores = scores

    # Step 2: Apply Gumbel-Sigmoid with temperature (Eq. 9)
    # z_i = σ((s_i + g_i) / τ)
    soft_mask = torch.sigmoid(noisy_scores / tau)

    if not hard:
        # Return soft mask directly (useful for debugging)
        return soft_mask

    # Step 3: Hard top-k selection in forward pass
    # Find k-th largest value per sample
    topk_values, topk_indices = torch.topk(soft_mask, k=k, dim=-1)
    threshold = topk_values[:, -1:]  # [B, 1], k-th largest value

    # Create hard binary mask
    hard_mask = (soft_mask >= threshold).float()

    # Handle ties: if multiple values equal threshold, randomly break ties
    # This is important when tau is very low
    if training:
        # Count how many tokens are selected (should be exactly k)
        num_selected = hard_mask.sum(dim=-1)  # [B]
        # If more than k selected (due to ties), need to break ties
        # For simplicity, we rely on topk's tie-breaking behavior
        # Reconstruct hard mask from topk indices
        hard_mask = torch.zeros_like(soft_mask)
        hard_mask.scatter_(1, topk_indices, 1.0)

    # Step 4: Straight-through estimator trick
    # Forward: hard_mask
    # Backward: soft_mask gradients
    # Implementation: mask = hard - soft.detach() + soft
    mask = hard_mask - soft_mask.detach() + soft_mask

    # Sanity check: verify exactly k tokens selected per sample
    if training:
        actual_k = mask.sum(dim=-1)
        if not torch.allclose(actual_k, torch.full_like(actual_k, float(k)), atol=1.0):
            print(f"Warning: Expected {k} tokens, got {actual_k.tolist()}")

    return mask


def compute_mask_entropy(mask: torch.Tensor) -> torch.Tensor:
    """Compute entropy of token selection distribution (for regularization).

    High entropy = diverse selection across spatial locations
    Low entropy = collapsed selection (pathological)

    Args:
        mask: Binary mask, Shape: [batch_size, num_tokens]

    Returns:
        entropy: Scalar tensor, averaged over batch

    Used in: losses.py for entropy regularization
    """
    # TODO (Phase D): Implement entropy computation
    # H(p) = -Σ p_i log(p_i)
    # where p_i = mean selection probability for token i across batch

    # Average selection probability per token
    prob = mask.mean(dim=0)  # [num_tokens]

    # Compute entropy
    prob = torch.clamp(prob, min=1e-10, max=1.0 - 1e-10)  # Numerical stability
    entropy = -(prob * torch.log(prob) + (1 - prob) * torch.log(1 - prob))
    entropy = entropy.mean()

    return entropy


def compute_mask_diversity(mask: torch.Tensor) -> torch.Tensor:
    """Compute diversity of token selection across batch (for regularization).

    Diversity = 1 - max_i P(token i selected)
    High diversity = tokens selected vary across samples
    Low diversity = same tokens always selected (query-ignoring pathology)

    Args:
        mask: Binary mask, Shape: [batch_size, num_tokens]

    Returns:
        diversity: Scalar tensor in [0, 1]

    Used in: losses.py for diversity regularization
    Paper reference: Section 9.1 (Failure Mode #2: Query-Ignoring Policy)
    """
    # TODO (Phase D): Implement diversity metric
    # Diversity = 1 - max_i (mean selection frequency of token i)

    # Selection frequency per token
    freq = mask.mean(dim=0)  # [num_tokens]

    # Find most frequently selected token
    max_freq = freq.max()

    # Diversity: 1 = all tokens equally likely, 0 = one token always selected
    diversity = 1.0 - max_freq

    return diversity
