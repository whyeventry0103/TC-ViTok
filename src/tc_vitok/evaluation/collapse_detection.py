"""Collapse detection diagnostics for TC-ViTok.

This module provides utilities to detect and diagnose the three known failure
modes during TPN training:

1. Spatial Collapse: All tokens selected from same spatial region
2. Query-Ignoring Policy: Same tokens selected regardless of question
3. Gradient Vanishing: STE not propagating gradients correctly

Paper reference: Section 9 (Known Failure Modes & Debugging)
"""

from typing import Dict, List, Tuple
import numpy as np
import torch


def detect_spatial_collapse(
    masks: torch.Tensor,
    threshold: float = 0.8,
    grid_size: Tuple[int, int] = (24, 24),
) -> Dict[str, any]:
    """Detect spatial collapse in token selection masks.

    Spatial collapse occurs when all selected tokens concentrate in one region
    (e.g., all in top-left quadrant). This indicates the TPN is not learning
    to spread attention across the image.

    Detection strategy:
        1. Reshape mask to spatial grid (24×24)
        2. Compute entropy of spatial distribution
        3. Check if any quadrant contains > 80% of selected tokens

    Args:
        masks: Selection masks, Shape: [batch_size, 576]
        threshold: Collapse threshold (default 0.8 = 80%)
        grid_size: Spatial grid dimensions (24, 24) for CLIP ViT-L/14

    Returns:
        diagnostics: Dictionary with:
            - is_collapsed: Boolean, True if collapse detected
            - max_quadrant_ratio: Ratio of tokens in most dense quadrant
            - spatial_entropy: Entropy of spatial distribution
            - quadrant_counts: Tokens per quadrant [TL, TR, BL, BR]

    Paper reference: Section 9.1 (Failure Mode #1)
    """

    batch_size, num_tokens = masks.shape
    h, w = grid_size

    # Reshape to spatial grid: [batch_size, 24, 24]
    masks_spatial = masks.view(batch_size, h, w)

    # Divide into quadrants
    h_mid, w_mid = h // 2, w // 2

    quadrant_counts = {
        "top_left": masks_spatial[:, :h_mid, :w_mid].sum(dim=(1, 2)),
        "top_right": masks_spatial[:, :h_mid, w_mid:].sum(dim=(1, 2)),
        "bottom_left": masks_spatial[:, h_mid:, :w_mid].sum(dim=(1, 2)),
        "bottom_right": masks_spatial[:, h_mid:, w_mid:].sum(dim=(1, 2)),
    }

    # Find maximum quadrant ratio
    total_selected = masks.sum(dim=1)  # [batch_size]
    max_quadrant = max(
        quadrant_counts.values(),
        key=lambda x: x.mean(),
    )
    max_ratio = (max_quadrant / total_selected).mean().item()

    # Compute spatial entropy
    # Flatten spatial grid and compute entropy
    probs = masks_spatial.mean(dim=0)  # [24, 24], average over batch
    probs_flat = probs.flatten()
    probs_flat = torch.clamp(probs_flat, min=1e-10, max=1.0)
    spatial_entropy = -(probs_flat * torch.log(probs_flat)).sum().item()

    return {
        "is_collapsed": max_ratio > threshold,
        "max_quadrant_ratio": max_ratio,
        "spatial_entropy": spatial_entropy,
        "quadrant_counts": {k: v.mean().item() for k, v in quadrant_counts.items()},
    }


def detect_query_ignoring(
    masks: torch.Tensor,
    questions: List[str],
    threshold: float = 0.9,
) -> Dict[str, any]:
    """Detect query-ignoring behavior in token selection.

    Query-ignoring occurs when the TPN selects the same tokens for all questions,
    ignoring the query input. This is a pathological policy that reduces to
    a fixed mask.

    Detection strategy:
        1. Compute pairwise mask similarity across batch
        2. Check if average similarity exceeds threshold (0.9)
        3. Identify most frequently selected tokens

    Args:
        masks: Selection masks, Shape: [batch_size, 576]
        questions: List of question strings (for logging)
        threshold: Similarity threshold (default 0.9 = 90%)

    Returns:
        diagnostics: Dictionary with:
            - is_ignoring: Boolean, True if query-ignoring detected
            - avg_mask_similarity: Average pairwise Jaccard similarity
            - most_frequent_tokens: Indices of 10 most frequent tokens
            - token_selection_variance: Variance of selection frequency

    Paper reference: Section 9.1 (Failure Mode #2)
    """

    batch_size = masks.shape[0]

    # Compute pairwise Jaccard similarity
    # Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    similarities = []
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            intersection = (masks[i] * masks[j]).sum()
            union = ((masks[i] + masks[j]) > 0).sum()
            similarity = (intersection / union).item() if union > 0 else 0.0
            similarities.append(similarity)

    avg_similarity = np.mean(similarities) if similarities else 0.0

    # Compute selection frequency per token
    selection_freq = masks.mean(dim=0)  # [576]
    most_frequent_indices = torch.argsort(selection_freq, descending=True)[:10]

    # Compute variance of selection frequency
    token_variance = selection_freq.std().item()

    return {
        "is_ignoring": avg_similarity > threshold,
        "avg_mask_similarity": avg_similarity,
        "most_frequent_tokens": most_frequent_indices.tolist(),
        "token_selection_variance": token_variance,
    }


def check_gradient_flow(
    mask: torch.Tensor,
    scores: torch.Tensor,
) -> Dict[str, bool]:
    """Verify gradient flow through Gumbel-Sigmoid STE.

    The STE must propagate gradients from mask to scores. If grad_fn is None,
    the STE is broken and training will fail.

    Args:
        mask: Selection mask with grad_fn
        scores: TPN output scores

    Returns:
        diagnostics: Dictionary with:
            - mask_has_grad: True if mask.grad_fn is not None
            - scores_has_grad: True if scores.grad_fn is not None
            - ste_working: True if both have gradients

    Critical check:
        assert mask.grad_fn is not None, "STE is broken!"

    Paper reference: Section 9.2 (Debugging Critical Path)
    """

    mask_has_grad = mask.grad_fn is not None
    scores_has_grad = scores.grad_fn is not None

    return {
        "mask_has_grad": mask_has_grad,
        "scores_has_grad": scores_has_grad,
        "ste_working": mask_has_grad and scores_has_grad,
    }


def run_full_diagnostics(
    masks: torch.Tensor,
    scores: torch.Tensor,
    questions: List[str],
) -> Dict[str, Dict]:
    """Run all collapse diagnostics and return comprehensive report.

    Args:
        masks: Selection masks, Shape: [batch_size, 576]
        scores: TPN scores, Shape: [batch_size, 576]
        questions: List of question strings

    Returns:
        report: Dictionary with all diagnostic results:
            - spatial_collapse: Results from detect_spatial_collapse
            - query_ignoring: Results from detect_query_ignoring
            - gradient_flow: Results from check_gradient_flow
            - overall_health: "healthy" / "warning" / "critical"

    Usage in training loop:
        if step % 1000 == 0:
            diagnostics = run_full_diagnostics(masks, scores, questions)
            if diagnostics["overall_health"] == "critical":
                print("WARNING: Training pathology detected!")

    Paper reference: Section 9 (full diagnostic suite)
    """

    spatial = detect_spatial_collapse(masks)
    ignoring = detect_query_ignoring(masks, questions)
    gradients = check_gradient_flow(masks, scores)

    # Determine overall health
    is_critical = (
        spatial["is_collapsed"]
        or ignoring["is_ignoring"]
        or not gradients["ste_working"]
    )

    is_warning = (
        spatial["max_quadrant_ratio"] > 0.6
        or ignoring["avg_mask_similarity"] > 0.7
    )

    if is_critical:
        overall_health = "critical"
    elif is_warning:
        overall_health = "warning"
    else:
        overall_health = "healthy"

    return {
        "spatial_collapse": spatial,
        "query_ignoring": ignoring,
        "gradient_flow": gradients,
        "overall_health": overall_health,
    }
