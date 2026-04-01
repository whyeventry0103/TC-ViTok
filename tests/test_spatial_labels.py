"""Unit tests for spatial specificity classification.

Tests verify:
- Keyword-based spatial detection
- GQA annotation-based classification (priority 1)
- Label distribution sanity (not all spatial, not all global)
- Batch classification consistency
"""

import pytest

from tc_vitok.data.spatial_labels import classify_spatial, classify_batch


class TestKeywordClassification:
    """Test keyword-based spatial classification (fallback method)."""

    @pytest.mark.parametrize("question", [
        "What is on the left side of the image?",
        "Is the dog behind the fence?",
        "What color is the object in the top right corner?",
        "Is the cup next to the plate?",
        "What is below the table?",
        "Is there a cat near the window?",
        "What is between the two chairs?",
        "What is at the bottom of the image?",
    ])
    def test_spatial_questions_detected(self, question):
        """Questions with spatial keywords should be classified as spatial."""
        assert classify_spatial(question) is True, (
            f"Expected spatial=True for: {question}"
        )

    @pytest.mark.parametrize("question", [
        "What color is the car?",
        "How many people are there?",
        "Is this a beach?",
        "What time of day is it?",
        "What animal is shown?",
        "Is the sky blue?",
        "What are they doing?",
    ])
    def test_global_questions_detected(self, question):
        """Questions without spatial keywords should be classified as global."""
        assert classify_spatial(question) is False, (
            f"Expected spatial=False for: {question}"
        )


class TestAnnotationClassification:
    """Test GQA annotation-based classification (priority 1)."""

    def test_annotation_with_spatial_relation(self):
        """Spatial relation in annotation should return True."""
        annotation = {
            "semantic": {
                "relations": [
                    {"spatial": True, "name": "to the left of"}
                ]
            }
        }
        result = classify_spatial("what color is it?", annotation)
        assert result is True, "Annotation with spatial relation should be spatial"

    def test_annotation_without_spatial_relation(self):
        """No spatial relation in annotation should return False."""
        annotation = {
            "semantic": {
                "relations": [
                    {"spatial": False, "name": "wearing"}
                ]
            }
        }
        result = classify_spatial("what is on the left?", annotation)
        assert result is False, (
            "Annotation without spatial relation overrides keywords"
        )

    def test_annotation_empty_relations(self):
        """Empty relations list should return False (global)."""
        annotation = {"semantic": {"relations": []}}
        result = classify_spatial("what is on the left?", annotation)
        assert result is False, "Empty relations should be global"

    def test_no_annotation_falls_back_to_keywords(self):
        """Without annotation, should fall back to keyword matching."""
        assert classify_spatial("what is on the left?", None) is True
        assert classify_spatial("what color is it?", None) is False


class TestLabelDistribution:
    """Test that classification produces reasonable label distributions."""

    def test_mixed_questions_produce_both_labels(self):
        """A realistic mix of questions should have both spatial and global."""
        questions = [
            "What is to the left of the dog?",
            "What color is the car?",
            "Is the cat behind the couch?",
            "How many people are there?",
            "What is near the window?",
            "Is the sky blue?",
            "What is between the trees?",
            "What animal is shown?",
            "What is at the bottom?",
            "What is the weather like?",
        ]
        labels = classify_batch(questions)
        spatial_count = sum(labels)
        global_count = len(labels) - spatial_count

        assert spatial_count > 0, "Expected at least some spatial questions"
        assert global_count > 0, "Expected at least some global questions"

    def test_spatial_ratio_reasonable(self):
        """Spatial ratio should be between 20-60% for diverse questions."""
        questions = [
            "What is on the left?",
            "What color is it?",
            "Is the dog behind the cat?",
            "How many are there?",
            "What is near the table?",
            "Is this a kitchen?",
            "What is at the top?",
            "What time is it?",
            "What is next to the door?",
            "What shape is it?",
        ]
        labels = classify_batch(questions)
        spatial_ratio = sum(labels) / len(labels)
        assert 0.2 <= spatial_ratio <= 0.6, (
            f"Spatial ratio {spatial_ratio:.2f} outside expected range [0.2, 0.6]"
        )


class TestBatchClassification:
    """Test batch classification consistency."""

    def test_batch_matches_individual(self):
        """Batch results should match individual classification."""
        questions = [
            "What is on the left?",
            "What color is the car?",
            "Is the cat behind the table?",
        ]
        batch_labels = classify_batch(questions)
        individual_labels = [classify_spatial(q) for q in questions]
        assert batch_labels == individual_labels, (
            "Batch and individual classification disagree"
        )

    def test_batch_with_annotations(self):
        """Batch with annotations should use annotation priority."""
        questions = ["what color?", "what shape?"]
        annotations = [
            {"semantic": {"relations": [{"spatial": True}]}},
            {"semantic": {"relations": []}},
        ]
        labels = classify_batch(questions, annotations)
        assert labels == [True, False]
