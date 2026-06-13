"""Unit tests for the data model classes."""

import pytest  # Test framework
from pydantic import ValidationError  # Pydantic validation errors

from src.models.evaluation_result import EvaluationResult, MetricScore  # Models under test
from src.models.test_case import EvaluationTestCase  # Model under test


class TestEvaluationTestCase:
    """Test suite for EvaluationTestCase data model."""

    def test_create_valid_test_case(self) -> None:
        """Verify a test case with all required fields is created successfully."""
        # Act: create test case with required fields
        test_case = EvaluationTestCase(
            input_text="What is AI?",
            actual_output="AI is artificial intelligence.",
        )

        # Assert: fields are set correctly
        assert test_case.input_text == "What is AI?"
        assert test_case.actual_output == "AI is artificial intelligence."
        assert test_case.expected_output == ""
        assert test_case.retrieval_context == []

    def test_create_test_case_with_all_fields(self) -> None:
        """Verify a fully populated test case preserves all values."""
        # Act: create test case with all fields populated
        test_case = EvaluationTestCase(
            input_text="What is AI?",
            actual_output="AI is artificial intelligence.",
            expected_output="Artificial intelligence",
            retrieval_context=["AI refers to machine intelligence."],
            context=["AI is a broad field of computer science."],
            name="ai_definition_test",
        )

        # Assert: all fields match provided values
        assert test_case.name == "ai_definition_test"
        assert len(test_case.retrieval_context) == 1
        assert len(test_case.context) == 1

    def test_empty_input_text_raises_validation_error(self) -> None:
        """Verify empty input_text triggers validation error."""
        # Act & Assert: empty string should fail min_length validation
        with pytest.raises(ValidationError):
            EvaluationTestCase(
                input_text="",
                actual_output="Some output",
            )

    def test_empty_actual_output_raises_validation_error(self) -> None:
        """Verify empty actual_output triggers validation error."""
        # Act & Assert: empty string should fail min_length validation
        with pytest.raises(ValidationError):
            EvaluationTestCase(
                input_text="Some input",
                actual_output="",
            )


class TestMetricScore:
    """Test suite for MetricScore data model."""

    def test_create_passing_metric_score(self) -> None:
        """Verify a passing metric score is constructed correctly."""
        # Act: create a metric score above threshold
        score = MetricScore(
            metric_name="AnswerRelevancy",
            score=0.85,
            passed=True,
            threshold=0.7,
            reason="High relevancy to query",
        )

        # Assert: all fields set correctly
        assert score.metric_name == "AnswerRelevancy"
        assert score.score == 0.85
        assert score.passed is True
        assert score.threshold == 0.7

    def test_create_failing_metric_score(self) -> None:
        """Verify a failing metric score is constructed correctly."""
        # Act: create a metric score below threshold
        score = MetricScore(
            metric_name="Faithfulness",
            score=0.3,
            passed=False,
            threshold=0.7,
        )

        # Assert: score indicates failure
        assert score.passed is False
        assert score.score == 0.3

    def test_score_below_zero_raises_validation_error(self) -> None:
        """Verify score below 0.0 triggers validation error."""
        # Act & Assert: negative score should fail ge=0.0 constraint
        with pytest.raises(ValidationError):
            MetricScore(
                metric_name="Test",
                score=-0.1,
                passed=False,
            )

    def test_score_above_one_raises_validation_error(self) -> None:
        """Verify score above 1.0 triggers validation error."""
        # Act & Assert: score > 1.0 should fail le=1.0 constraint
        with pytest.raises(ValidationError):
            MetricScore(
                metric_name="Test",
                score=1.5,
                passed=True,
            )


class TestEvaluationResult:
    """Test suite for EvaluationResult data model."""

    def test_compute_overall_status_all_passed(self) -> None:
        """Verify overall status is True when all metrics pass."""
        # Arrange: create result with passing scores
        result = EvaluationResult(
            test_case_name="test_case_1",
            metric_scores=[
                MetricScore(metric_name="A", score=0.9, passed=True),
                MetricScore(metric_name="B", score=0.8, passed=True),
            ],
        )

        # Act: compute overall status
        result.compute_overall_status()

        # Assert: overall passes when all metrics pass
        assert result.overall_passed is True

    def test_compute_overall_status_one_failed(self) -> None:
        """Verify overall status is False when any metric fails."""
        # Arrange: create result with one failing score
        result = EvaluationResult(
            test_case_name="test_case_2",
            metric_scores=[
                MetricScore(metric_name="A", score=0.9, passed=True),
                MetricScore(metric_name="B", score=0.3, passed=False),
            ],
        )

        # Act: compute overall status
        result.compute_overall_status()

        # Assert: overall fails when any metric fails
        assert result.overall_passed is False

    def test_compute_overall_status_empty_scores(self) -> None:
        """Verify overall status is False when no metrics are present."""
        # Arrange: create result with no scores
        result = EvaluationResult(
            test_case_name="empty_test",
            metric_scores=[],
        )

        # Act: compute overall status
        result.compute_overall_status()

        # Assert: no scores means failure
        assert result.overall_passed is False

    def test_evaluated_at_is_set_automatically(self) -> None:
        """Verify timestamp is auto-populated on creation."""
        # Act: create a result
        result = EvaluationResult(test_case_name="timestamp_test")

        # Assert: timestamp is populated
        assert result.evaluated_at is not None
