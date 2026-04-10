"""Gumbel-Sigmoid Straight-Through Estimator for differentiable top-k selection.

This module implements the core masking operation that enables gradient flow
through discrete token selection. The forward pass performs hard top-k selection,
while the backward pass uses soft Gumbel-Sigmoid gradients.

Paper Section 6: "Differentiable Token Selection via Gumbel-Sigmoid STE"
"""

from typing import Tuple

import torch


def gumbel_sigmoid_ste(
    scores: torch.Tensor,
    k: int,
    tau: float = 1.0,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gumbel-Sigmoid Straight-Through Estimator for differentiable top-k selection.

    Forward pass: hard top-k binary mask (exact k ones per sample).
    Backward pass: gradients flow through soft Gumbel-Sigmoid.

    Implementation matches paper Appendix A.2 and planv2 spec exactly:
        eps = 1e-8
        u ~ Uniform(0, 1), clamped to [eps, 1-eps]
        g = -log(-log(u))                    # Gumbel(0,1)
        m_soft = sigmoid((scores + g) / tau)  # soft mask
        idx = scores.topk(k).indices          # hard top-k on RAW scores
        m_hard = scatter(idx, 1.0)
        mask_ste = m_hard - m_soft.detach() + m_soft  # STE trick
        return mask_ste, idx

    Args:
        scores: Raw importance scores from TPN, Shape: [B, N]
        k: Number of tokens to select
        tau: Temperature (annealed 1.0 → 0.1 during training)
        training: If True, inject Gumbel noise; else deterministic top-k

    Returns:
        mask_ste: STE mask, Shape: [B, N] — binary forward, soft gradients backward
        idx: Top-k indices, Shape: [B, k] — used for gather
    """
    if not training:
        # Inference: deterministic hard top-k, no gradients needed
        idx = scores.topk(k, dim=-1).indices
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, idx, 1.0)
        return mask, idx

    # Step 1: Sample Gumbel noise
    eps = 1e-8
    u = torch.rand_like(scores).clamp(eps, 1 - eps)
    g = -torch.log(-torch.log(u))  # Gumbel(0, 1)

    # Step 2: Noisy scores used for BOTH soft mask and hard top-k.
    # Using raw scores for topk (the prior bug) meant all-zero raw scores always
    # returned indices 0..k-1 via deterministic tiebreaking → all batch samples
    # select identical tokens → entropy = log(k) instead of log(N).
    noisy_scores = scores + g

    # Step 3: Soft mask via Gumbel-Sigmoid with temperature
    m_soft = torch.sigmoid(noisy_scores / tau)

    # Step 4: Hard top-k on NOISY scores — ensures per-sample diversity
    idx = noisy_scores.topk(k, dim=-1).indices
    m_hard = torch.zeros_like(scores)
    m_hard.scatter_(-1, idx, 1.0)

    # Step 4: STE trick — forward = m_hard, backward = grad through m_soft
    mask_ste = m_hard - m_soft.detach() + m_soft
    return mask_ste, idx


def compute_spatial_entropy(mask: torch.Tensor) -> torch.Tensor:
    """Compute Shannon entropy of the spatial selection distribution.

    This is the paper's entropy regularizer (Eq. in Section 7.1):
        p_i = m̄_i / Σ_j m̄_j
        H = -Σ_i p_i log(p_i)

    where m̄_i is the batch-averaged selection frequency at position i.

    Maximum entropy = log(N) ≈ 6.36 (uniform selection over 576 positions).
    Minimum entropy → 0 (all mass on one position).

    We MAXIMIZE entropy (by negating it in the loss) to prevent spatial collapse.

    Args:
        mask: Binary selection mask, Shape: [B, N]

    Returns:
        entropy: Scalar Shannon entropy of the normalized spatial distribution
    """
    eps = 1e-10

    # Batch-averaged selection frequency per position
    m_bar = mask.mean(dim=0)  # [N]

    # Normalize to a proper probability distribution
    p = m_bar / (m_bar.sum() + eps)  # [N], sums to 1

    # Shannon entropy: H = -Σ p_i log(p_i)
    entropy = -(p * torch.log(p + eps)).sum()

    return entropy


def compute_diversity_loss(
    mask_1: torch.Tensor,
    mask_2: torch.Tensor,
    q_embed_1: torch.Tensor,
    q_embed_2: torch.Tensor,
    gamma: float = 0.5,
) -> torch.Tensor:
    """Compute query-diversity loss to prevent query-ignoring behavior.

    Paper Eq. (div_loss):
        L_div = sum_b max(0, sim(m1, m2) - gamma) * (1 - sim(q1, q2))

    When two different questions for the same image produce very similar
    masks (sim > gamma), and the questions themselves are dissimilar,
    this loss penalizes the TPN for ignoring the query.

    Args:
        mask_1: Binary mask for question 1, Shape: [B, N]
        mask_2: Binary mask for question 2, Shape: [B, N]
        q_embed_1: Text embedding for question 1, Shape: [B, d_q]
        q_embed_2: Text embedding for question 2, Shape: [B, d_q]
        gamma: Similarity threshold (default 0.5)

    Returns:
        loss: Scalar diversity loss (minimize to encourage query sensitivity)
    """
    eps = 1e-8

    # Cosine similarity between mask pairs (per batch element)
    mask_sim = torch.nn.functional.cosine_similarity(mask_1, mask_2, dim=-1)  # [B]

    # Cosine similarity between query embedding pairs
    q_sim = torch.nn.functional.cosine_similarity(q_embed_1, q_embed_2, dim=-1)  # [B]

    # Hinge on mask similarity, weighted by query dissimilarity
    per_sample = torch.clamp(mask_sim - gamma, min=0.0) * (1.0 - q_sim)  # [B]

    return per_sample.sum()
