"""Gumbel-Sigmoid Straight-Through Estimator for differentiable top-k selection.

This module implements the core masking operation that enables gradient flow
through discrete token selection. The forward pass performs hard top-k selection,
while the backward pass uses soft Gumbel-Sigmoid gradients.

Paper Section 6: "Differentiable Token Selection via Gumbel-Sigmoid STE"
"""

import torch


def gumbel_sigmoid_ste(
    scores: torch.Tensor,
    k: int,
    tau: float = 1.0,
    training: bool = True,
    hard: bool = True,
) -> torch.Tensor:
    """Gumbel-Sigmoid Straight-Through Estimator for differentiable top-k selection.

    Forward pass: hard top-k binary mask (exact k ones per sample).
    Backward pass: gradients flow through soft Gumbel-Sigmoid.

    Implementation matches paper Appendix A.2 exactly:
        eps = 1e-8
        u ~ Uniform(0, 1), clamped to [eps, 1-eps]
        g = -log(-log(u))                    # Gumbel(0,1)
        m_soft = sigmoid((scores + g) / tau)  # soft mask
        topk_idx = scores.topk(k).indices     # hard top-k on RAW scores
        m_hard = scatter(topk_idx, 1.0)
        return m_hard - m_soft.detach() + m_soft  # STE trick

    Args:
        scores: Raw importance scores from TPN, Shape: [B, N]
        k: Number of tokens to select
        tau: Temperature (annealed 1.0 → 0.1 during training)
        training: If True, inject Gumbel noise; else deterministic top-k
        hard: If True, use hard top-k in forward; else return soft mask

    Returns:
        mask: Selection mask, Shape: [B, N]
    """
    if not training:
        # Inference: deterministic hard top-k, no gradients needed
        topk_idx = scores.topk(k, dim=-1).indices
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, topk_idx, 1.0)
        return mask

    # Step 1: Sample Gumbel noise
    eps = 1e-8
    u = torch.rand_like(scores).clamp(eps, 1 - eps)
    g = -torch.log(-torch.log(u))  # Gumbel(0, 1)

    # Step 2: Soft mask via Gumbel-Sigmoid with temperature
    m_soft = torch.sigmoid((scores + g) / tau)

    if not hard:
        return m_soft

    # Step 3: Hard top-k on RAW scores (not noisy scores)
    topk_idx = scores.topk(k, dim=-1).indices
    m_hard = torch.zeros_like(scores)
    m_hard.scatter_(-1, topk_idx, 1.0)

    # Step 4: STE trick — forward = m_hard, backward = grad through m_soft
    return m_hard - m_soft.detach() + m_soft


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


def compute_mask_diversity(mask: torch.Tensor) -> torch.Tensor:
    """Compute diversity of token selection across batch.

    Diversity = 1 - max_i P(token i selected)
    High diversity = tokens selected vary across samples.
    Low diversity = same tokens always selected (query-ignoring pathology).

    Args:
        mask: Binary mask, Shape: [B, N]

    Returns:
        diversity: Scalar in [0, 1]
    """
    freq = mask.mean(dim=0)  # [N]
    max_freq = freq.max()
    return 1.0 - max_freq
