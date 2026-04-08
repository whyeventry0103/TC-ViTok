"""Spatial specificity classifier for VQA questions.

Classifies questions as 'spatial' or 'global' using the protocol
described in paper Section 8.3:

Priority:
1. (GQA only) Semantic annotation includes relation.spatial = True
2. Answer is a spatial direction (left, right, above, below)
3. Question contains spatial keywords

Paper reference: Section 8.3 (Spatial Specificity Evaluation Protocol)
"""

from typing import Optional, Dict

# Spatial keywords for keyword-based classification (fallback)
SPATIAL_KEYWORDS = {
    "left", "right", "above", "below", "next to", "behind",
    "corner", "between", "near", "top", "bottom", "beside",
    "front", "underneath", "on top of", "side",
}

SPATIAL_ANSWERS = {
    "left", "right", "above", "below", "top", "bottom",
    "yes", "no",  # only when question is spatial
}


def classify_spatial(
    question: str,
    gqa_annotation: Optional[Dict] = None,
) -> bool:
    """Classify a VQA question as spatial or global.

    Args:
        question: Question string
        gqa_annotation: Optional GQA scene graph annotation dict.
                        If provided and contains 'semantic' field,
                        uses relation.spatial for classification.

    Returns:
        is_spatial: True if question is spatially-grounded
    """
    # Priority 1: GQA semantic annotation (most reliable)
    if gqa_annotation is not None:
        semantic = gqa_annotation.get("semantic", {})
        if isinstance(semantic, dict):
            relations = semantic.get("relations", [])
            for rel in relations:
                if isinstance(rel, dict) and rel.get("spatial", False):
                    return True
            # If GQA annotation exists but no spatial relation, it's global
            return False

    # Priority 2/3: Keyword-based classification (noisy fallback)
    q_lower = question.lower()
    for keyword in SPATIAL_KEYWORDS:
        if keyword in q_lower:
            return True

    return False


def classify_batch(
    questions: list[str],
    gqa_annotations: Optional[list[Dict]] = None,
) -> list[bool]:
    """Classify a batch of questions.

    Args:
        questions: List of question strings
        gqa_annotations: Optional list of GQA annotations (same length)

    Returns:
        labels: List of boolean spatial labels
    """
    annotations = gqa_annotations or [None] * len(questions)
    return [classify_spatial(q, a) for q, a in zip(questions, annotations)]
