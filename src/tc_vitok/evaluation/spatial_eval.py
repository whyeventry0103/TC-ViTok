"""Spatial specificity evaluation for GQA.

This module implements the spatial vs. global question stratification protocol.
It allows us to measure whether token selection performance degrades on
spatially-specific questions.

Protocol:
1. Classify each GQA question as spatial or global
2. Compute accuracy separately for each subset
3. Report both spatial and global accuracy

Paper reference: Section 8.3 (Spatial Specificity Evaluation)
"""

from typing import List, Dict, Tuple

from ..data.spatial_labels import classify_spatial
from .metrics import gqa_accuracy


def evaluate_spatial_stratified(
    predictions: List[str],
    ground_truth: List[str],
    questions: List[str],
    gqa_annotations: List[Dict] = None,
) -> Dict[str, float]:
    """Evaluate GQA with spatial/global stratification.

    Args:
        predictions: Model predictions
        ground_truth: Reference answers
        questions: Question strings for classification
        gqa_annotations: Optional GQA scene graph annotations

    Returns:
        metrics: Dictionary with:
            - overall_acc: Overall accuracy
            - spatial_acc: Accuracy on spatial questions
            - global_acc: Accuracy on global questions
            - spatial_count: Number of spatial questions
            - global_count: Number of global questions

    This reveals whether token selection hurts spatial reasoning.

    Expected result:
        If TC-ViTok is working well, spatial_acc should be close to global_acc.
        Large gap indicates spatial collapse failure mode.

    Paper reference: Section 8.3, Table 2
    """

    assert len(predictions) == len(ground_truth) == len(questions)

    # Classify questions
    spatial_labels = [
        classify_spatial(q, gqa_annotations[i] if gqa_annotations else None)
        for i, q in enumerate(questions)
    ]

    # Separate spatial and global subsets
    spatial_preds = [p for p, is_sp in zip(predictions, spatial_labels) if is_sp]
    spatial_gt = [g for g, is_sp in zip(ground_truth, spatial_labels) if is_sp]

    global_preds = [p for p, is_sp in zip(predictions, spatial_labels) if not is_sp]
    global_gt = [g for g, is_sp in zip(ground_truth, spatial_labels) if not is_sp]

    # Compute accuracies
    overall_acc = sum(
        gqa_accuracy(p, g) for p, g in zip(predictions, ground_truth)
    ) / len(predictions)

    spatial_acc = (
        sum(gqa_accuracy(p, g) for p, g in zip(spatial_preds, spatial_gt))
        / len(spatial_preds)
        if spatial_preds else 0.0
    )

    global_acc = (
        sum(gqa_accuracy(p, g) for p, g in zip(global_preds, global_gt))
        / len(global_preds)
        if global_preds else 0.0
    )

    return {
        "overall_acc": overall_acc,
        "spatial_acc": spatial_acc,
        "global_acc": global_acc,
        "spatial_count": len(spatial_preds),
        "global_count": len(global_preds),
        "spatial_global_gap": spatial_acc - global_acc,
    }


def analyze_selection_by_spatial(
    masks: List,  # [num_samples, num_tokens]
    spatial_labels: List[bool],
) -> Dict[str, float]:
    """Analyze token selection patterns for spatial vs. global questions.

    Computes statistics on how many tokens are selected for each question type.
    This helps diagnose spatial collapse.

    Args:
        masks: Token selection masks from TPN
        spatial_labels: Boolean labels (True = spatial)

    Returns:
        stats: Dictionary with:
            - spatial_mean_tokens: Average tokens selected for spatial questions
            - global_mean_tokens: Average tokens selected for global questions
            - spatial_std_tokens: Std of tokens for spatial
            - global_std_tokens: Std of tokens for global

    Expected behavior:
        Spatial questions should select similar or slightly more tokens than
        global questions. Large difference indicates pathology.

    Paper reference: Section 9.1 (Failure Mode Diagnosis)
    """

    import numpy as np

    # Convert to numpy
    masks_np = np.array([m.cpu().numpy() if hasattr(m, 'cpu') else m for m in masks])

    # Count tokens per sample
    token_counts = masks_np.sum(axis=1)

    # Separate by spatial label
    spatial_counts = token_counts[[i for i, is_sp in enumerate(spatial_labels) if is_sp]]
    global_counts = token_counts[[i for i, is_sp in enumerate(spatial_labels) if not is_sp]]

    return {
        "spatial_mean_tokens": float(np.mean(spatial_counts)) if len(spatial_counts) > 0 else 0.0,
        "global_mean_tokens": float(np.mean(global_counts)) if len(global_counts) > 0 else 0.0,
        "spatial_std_tokens": float(np.std(spatial_counts)) if len(spatial_counts) > 0 else 0.0,
        "global_std_tokens": float(np.std(global_counts)) if len(global_counts) > 0 else 0.0,
    }
