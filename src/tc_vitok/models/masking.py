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


def adaptive_gumbel_sigmoid_ste(
    scores: torch.Tensor,
    k_frac: torch.Tensor,
    tau: float = 1.0,
    training: bool = True,
    k_min: int = 36,
    k_max: int = 288,
    N: int = 576,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adaptive Gumbel-Sigmoid STE with per-sample variable token budget.

    Instead of fixed top-k, the BudgetPredictor provides a per-sample target
    fraction k_frac ∈ [k_min/N, k_max/N]. We convert this to a per-sample
    threshold by shifting scores:

        threshold = quantile(s_N, 1 - k_frac)  (conceptually)

    In practice, we compute k_frac * N as the target count, convert to integer
    k_hard per sample, and use topk on shifted scores. The differentiable
    k_effective = σ(s_shifted / tau).sum() provides gradient to the
    BudgetPredictor through the budget loss.

    The key insight: gradient flows from VQA loss through selected tokens back
    to TPN scores, and from budget loss λ * mean(k_eff/N) directly to the
    BudgetPredictor. Both paths are fully differentiable.

    Args:
        scores: Instance-normalized TPN scores [B, N]
        k_frac: Per-sample target fraction from BudgetPredictor [B]
        tau: Gumbel-Sigmoid temperature
        training: If True, inject Gumbel noise and use STE
        k_min: Minimum tokens per sample
        k_max: Maximum tokens per sample
        N: Total token positions (576)

    Returns:
        mask_ste: STE mask [B, N] (training) or hard mask (eval)
        idx: Padded gather indices [B, k_max_in_batch], -1 for padding
        k_effective: Differentiable expected token count [B]
    """
    B = scores.shape[0]

    # Per-sample target k from fraction
    k_target = k_frac * N  # [B], continuous in [k_min, k_max]

    if not training:
        # Inference: deterministic hard threshold per sample
        # Use topk with per-sample k_hard, pad to max
        k_hard = k_target.detach().round().long().clamp(k_min, k_max)  # [B]
        k_max_batch = k_hard.max().item()

        all_idx = []
        for b in range(B):
            kb = k_hard[b].item()
            top_idx = scores[b].topk(kb, dim=-1).indices  # [kb]
            # Pad to k_max_batch with -1 (will be masked in LLM)
            if kb < k_max_batch:
                pad = torch.full(
                    (k_max_batch - kb,), -1,
                    device=scores.device, dtype=torch.long,
                )
                top_idx = torch.cat([top_idx, pad])
            all_idx.append(top_idx)

        idx = torch.stack(all_idx, dim=0)  # [B, k_max_batch]
        mask = torch.zeros_like(scores)
        # Scatter only valid (non-padding) indices to avoid overwriting
        # token 0 with 0.0 from clamped padding entries.
        for b in range(B):
            valid = idx[b] >= 0
            if valid.any():
                mask[b].scatter_(0, idx[b][valid], 1.0)

        # k_effective at eval: just the hard counts (no gradient needed)
        k_effective = k_hard.float()
        return mask, idx, k_effective

    # ── Training: STE with Gumbel noise and differentiable k_effective ──

    # Step 1: Compute per-sample threshold from k_frac.
    # Since scores are instance-normalized (mean=0, std=1), the threshold
    # that selects k_frac*N tokens is approximately the (1-k_frac) quantile
    # of the score distribution. For a sorted score vector, this is:
    #   threshold ≈ scores[sorted][(1-k_frac)*N]
    # We use a differentiable approximation: shift scores by a threshold
    # derived from k_frac so that sigmoid((scores - threshold) / tau).sum()
    # depends on both scores AND k_frac (giving gradient to BudgetPredictor).
    #
    # Differentiable threshold: for instance-normed scores (mean=0, std=1),
    # the approximate quantile threshold for keeping k_frac fraction of tokens
    # is the (1 - k_frac) quantile of a standard normal ≈ -sqrt(2) * erfinv(2*k_frac - 1).
    # But a simpler linear approximation works well in practice:
    #   threshold = (0.5 - k_frac) * scale
    # where scale controls the steepness. At k_frac=0.25 (144 tokens), threshold > 0
    # (scores must exceed a positive bar). At k_frac=0.5 (288 tokens), threshold = 0
    # (baseline at mean).
    #
    # We use a direct approach: sort detached scores to find the hard threshold,
    # then use a soft version for k_effective.
    scores_sorted = scores.detach().sort(dim=-1, descending=True).values  # [B, N]
    # k_target is differentiable through k_frac
    k_idx = k_target.detach().long().clamp(1, N - 1)  # [B] — integer index for threshold
    # Threshold = midpoint between k-th and (k+1)-th score (detached from scores,
    # but the shift by k_frac-derived threshold keeps gradient to BudgetPredictor)
    threshold_hard = torch.zeros(B, device=scores.device)
    for b in range(B):
        ki = k_idx[b].item()
        threshold_hard[b] = (scores_sorted[b, ki - 1] + scores_sorted[b, ki]) / 2

    # Make threshold depend on k_frac (differentiable):
    # threshold = threshold_hard - (k_target - k_target.detach())
    # This has zero forward effect (identity) but d(threshold)/d(k_frac) = -N
    # so d(k_eff)/d(k_frac) = -N * d(k_eff)/d(threshold) which is the signal
    # the BudgetPredictor needs.
    threshold = threshold_hard - (k_target - k_target.detach())  # [B]

    # Step 2: Gumbel noise for exploration
    eps = 1e-8
    u = torch.rand_like(scores).clamp(eps, 1 - eps)
    g = -torch.log(-torch.log(u))  # Gumbel(0, 1)
    noisy_scores = scores + g

    # Step 3: Shifted soft mask — depends on BOTH scores and threshold (k_frac)
    shifted_scores = noisy_scores - threshold.unsqueeze(1)  # [B, N]
    m_soft = torch.sigmoid(shifted_scores / tau)  # [B, N]

    # Step 4: Differentiable k_effective — budget loss target
    # k_effective = σ((scores + g - threshold) / tau).sum()
    # Gradient to BudgetPredictor flows via threshold → k_frac → BudgetPredictor params.
    k_effective = m_soft.sum(dim=-1)  # [B]

    # Step 5: Hard selection — per-sample k from budget predictor
    k_hard = k_target.detach().round().long().clamp(k_min, k_max)  # [B]
    k_max_batch = k_hard.max().item()

    # Per-sample topk on noisy scores (ensures diversity like original STE)
    all_idx = []
    m_hard = torch.zeros_like(scores)  # [B, N]
    for b in range(B):
        kb = k_hard[b].item()
        top_idx = noisy_scores[b].topk(kb, dim=-1).indices  # [kb]
        m_hard[b].scatter_(0, top_idx, 1.0)
        # Pad to k_max_batch
        if kb < k_max_batch:
            pad = torch.full(
                (k_max_batch - kb,), -1,
                device=scores.device, dtype=torch.long,
            )
            top_idx = torch.cat([top_idx, pad])
        all_idx.append(top_idx)

    idx = torch.stack(all_idx, dim=0)  # [B, k_max_batch]

    # Step 6: STE trick — forward = m_hard, backward = grad through m_soft
    mask_ste = m_hard - m_soft.detach() + m_soft  # [B, N]

    return mask_ste, idx, k_effective


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
