"""Evaluation metrics for VQA benchmarks.

This module implements standard VQA metrics:
- VQA-score: Soft accuracy with multiple reference answers
- GQA accuracy: Exact match accuracy
- POPE F1: Binary classification F1 score
- TextVQA accuracy: Answer accuracy with normalization

Paper reference: Section 8 (Evaluation Protocols)
"""

from typing import List, Dict, Any
from collections import Counter
import re


def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison.

    Applies standard VQA normalization:
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Strip whitespace

    Args:
        answer: Raw answer string

    Returns:
        normalized: Normalized answer

    Used by all VQA metrics for fair comparison.
    """
    # TODO (Phase C): Implement answer normalization

    # Lowercase
    answer = answer.lower()

    # Remove punctuation
    answer = re.sub(r"[^\w\s]", "", answer)

    # Remove articles
    articles = ["a", "an", "the"]
    words = answer.split()
    words = [w for w in words if w not in articles]
    answer = " ".join(words)

    # Strip whitespace
    answer = answer.strip()

    return answer


def vqa_score(
    predicted: str,
    ground_truth: List[str],
    weights: List[float] = None,
) -> float:
    """Compute VQA-score with soft accuracy.

    VQA-score allows partial credit if answer matches multiple annotators.
    Score = min(# annotators who said predicted / 3, 1.0)

    Args:
        predicted: Model's predicted answer
        ground_truth: List of reference answers (from multiple annotators)
        weights: Optional weights for each reference (default: uniform)

    Returns:
        score: Soft accuracy score in [0, 1]

    Example:
        >>> vqa_score("cat", ["cat", "cat", "dog"])
        0.67  # 2/3 annotators said "cat"

    Paper reference: VQAv2 official metric
    """
    # TODO (Phase C): Implement VQA-score

    predicted_norm = normalize_answer(predicted)

    # Count how many annotators gave this answer
    gt_norm = [normalize_answer(ans) for ans in ground_truth]
    count = sum(1 for ans in gt_norm if ans == predicted_norm)

    # VQA-score: min(count / 3, 1.0)
    score = min(count / 3.0, 1.0)

    return score


def gqa_accuracy(
    predicted: str,
    ground_truth: str,
) -> float:
    """Compute GQA exact match accuracy.

    GQA uses binary 0/1 accuracy (exact match after normalization).

    Args:
        predicted: Model's predicted answer
        ground_truth: Reference answer

    Returns:
        accuracy: 1.0 if match, 0.0 otherwise
    """
    # TODO (Phase C): Implement GQA accuracy

    predicted_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    return 1.0 if predicted_norm == gt_norm else 0.0


def textvqa_accuracy(
    predicted: str,
    ground_truth: List[str],
) -> float:
    """Compute TextVQA accuracy.

    TextVQA uses soft accuracy like VQAv2 but with stricter normalization
    due to OCR answers.

    Args:
        predicted: Model's predicted answer
        ground_truth: List of reference answers

    Returns:
        accuracy: Soft accuracy score
    """
    # TODO (Phase C): Implement TextVQA accuracy
    # Similar to VQA-score but with different threshold

    predicted_norm = normalize_answer(predicted)
    gt_norm = [normalize_answer(ans) for ans in ground_truth]

    # Count matches
    count = sum(1 for ans in gt_norm if ans == predicted_norm)

    # TextVQA uses same formula as VQAv2
    score = min(count / 3.0, 1.0)

    return score


def pope_f1(
    predictions: List[str],
    ground_truth: List[str],
) -> Dict[str, float]:
    """Compute POPE F1, precision, and recall.

    POPE is a binary yes/no task for object hallucination detection.

    Args:
        predictions: List of predicted answers ("yes" or "no")
        ground_truth: List of reference answers ("yes" or "no")

    Returns:
        metrics: Dictionary with:
            - f1: F1 score
            - precision: Precision
            - recall: Recall
            - accuracy: Exact match accuracy

    Paper reference: POPE benchmark protocol
    """
    # TODO (Phase C): Implement POPE metrics

    # Normalize answers
    preds_norm = [normalize_answer(p) for p in predictions]
    gt_norm = [normalize_answer(g) for g in ground_truth]

    # Binary classification metrics
    true_pos = sum(1 for p, g in zip(preds_norm, gt_norm)
                   if p == "yes" and g == "yes")
    false_pos = sum(1 for p, g in zip(preds_norm, gt_norm)
                    if p == "yes" and g == "no")
    false_neg = sum(1 for p, g in zip(preds_norm, gt_norm)
                    if p == "no" and g == "yes")
    true_neg = sum(1 for p, g in zip(preds_norm, gt_norm)
                   if p == "no" and g == "no")

    # Compute metrics
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (true_pos + true_neg) / len(predictions)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def aggregate_scores(
    scores: List[float],
) -> Dict[str, float]:
    """Aggregate individual scores into summary statistics.

    Args:
        scores: List of per-sample scores

    Returns:
        stats: Dictionary with mean, median, std, etc.
    """
    # TODO (Phase D): Implement score aggregation

    import numpy as np

    scores_arr = np.array(scores)

    return {
        "mean": float(np.mean(scores_arr)),
        "median": float(np.median(scores_arr)),
        "std": float(np.std(scores_arr)),
        "min": float(np.min(scores_arr)),
        "max": float(np.max(scores_arr)),
        "count": len(scores),
    }
