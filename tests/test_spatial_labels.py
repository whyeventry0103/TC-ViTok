"""Unit tests for spatial specificity classification."""

import pytest

# TODO (Phase C): Import classification function
# from tc_vitok.data.spatial_labels import classify_spatial


def test_classify_spatial_keywords():
    """Test keyword-based spatial classification."""
    # TODO (Phase C): Implement test

    # Spatial questions (should return True)
    spatial_questions = [
        "what is on the left side of the image?",
        "is the dog behind the fence?",
        "what color is the object in the top right corner?",
        "which direction is the car facing?",
    ]

    # for q in spatial_questions:
    #     assert classify_spatial(q) == True

    # Global questions (should return False)
    global_questions = [
        "what color is the car?",
        "how many people are there?",
        "is this a beach?",
        "what time of day is it?",
    ]

    # for q in global_questions:
    #     assert classify_spatial(q) == False

    pass


def test_classify_spatial_distribution():
    """Test that spatial classification produces reasonable distribution."""
    # TODO (Phase D): Implement test
    # On GQA, expect roughly 30-40% spatial questions

    # from tc_vitok.data.spatial_labels import compute_spatial_statistics

    # Mock GQA questions
    # questions = [...]

    # stats = compute_spatial_statistics(questions)

    # Sanity check: not all spatial, not all global
    # assert 0.2 < stats["spatial_ratio"] < 0.6

    pass


def test_classify_spatial_with_annotations():
    """Test annotation-based classification (priority 1)."""
    # TODO (Phase D): Implement test
    # GQA annotations should override keyword matching

    question = "what is the color?"  # No spatial keywords
    annotations = {"types": ["select spatial"]}  # But annotation says spatial

    # result = classify_spatial(question, annotations)
    # assert result == True  # Annotation takes priority

    pass
