"""Unit tests for evaluator classes with mocked DeepEval metrics."""

from unittest.mock import MagicMock, patch  # Mocking utilities

import pytest  # Test framework

from src.evaluators.answer_relevancy import AnswerRelevancyEvaluator  # Module under test
from src.evaluators.bias import BiasEvaluator  # Module under test
from src.evaluators.contextual_relevancy import ContextualRelevancyEvaluator  # Module under test
from src.evaluators.faithfulness import FaithfulnessEvaluator  # Module under test
from src.evaluators.hallucination import HallucinationEvaluator  # Module under test
from src.evaluators.toxicity import ToxicityEvaluator  # Module under test
from src.models.test_case import EvaluationTestCase  # Test case model


class TestAnswerRelevancyEvaluator:
    """Test suite for AnswerRelevancyEvaluator."""

    @patch("src.evaluators.answer_relevancy.AnswerRelevancyMetric")
    def test_evaluate_returns_passing_score(self, mock_metric_class: MagicMock) -> None:
        """Verify evaluator returns passing score when metric scores high."""
        # Arrange: mock the metric to return a high score
        mock_metric = MagicMock()
        mock_metric.score = 0.9
        mock_metric.reason = "Highly relevant answer"
        mock_metric_class.return_value = mock_metric

        # Create evaluator and test case
        evaluator = AnswerRelevancyEvaluator(threshold=0.7)
        test_case = EvaluationTestCase(
            input_text="What is Python?",
            actual_output="Python is a programming language.",
            name="relevancy_pass_test",
        )

        # Act: run evaluation
        result = evaluator.evaluate(test_case)

        # Assert: score passes the threshold
        assert result.score == 0.9
        assert result.passed is True
        assert result.metric_name == "AnswerRelevancy"
        mock_metric.measure.assert_called_once()

    @patch("src.evaluators.answer_relevancy.AnswerRelevancyMetric")
    def test_evaluate_returns_failing_score(self, mock_metric_class: MagicMock) -> None:
        """Verify evaluator returns failing score when metric scores low."""
        # Arrange: mock metric to return a low score
        mock_metric = MagicMock()
        mock_metric.score = 0.3
        mock_metric.reason = "Answer not relevant"
        mock_metric_class.return_value = mock_metric

        # Create evaluator and test case
        evaluator = AnswerRelevancyEvaluator(threshold=0.7)
        test_case = EvaluationTestCase(
            input_text="What is Python?",
            actual_output="The weather is nice today.",
            name="relevancy_fail_test",
        )

        # Act: run evaluation
        result = evaluator.evaluate(test_case)

        # Assert: score fails the threshold
        assert result.score == 0.3
        assert result.passed is False

    @patch("src.evaluators.answer_relevancy.AnswerRelevancyMetric")
    def test_evaluate_raises_runtime_error_on_failure(self, mock_metric_class: MagicMock) -> None:
        """Verify RuntimeError is raised when metric evaluation fails."""
        # Arrange: mock metric to raise an exception
        mock_metric = MagicMock()
        mock_metric.measure.side_effect = Exception("API connection failed")
        mock_metric_class.return_value = mock_metric

        # Create evaluator and test case
        evaluator = AnswerRelevancyEvaluator(threshold=0.7)
        test_case = EvaluationTestCase(
            input_text="Test input",
            actual_output="Test output",
            name="error_test",
        )

        # Act & Assert: RuntimeError is raised
        with pytest.raises(RuntimeError, match="Answer relevancy evaluation failed"):
            evaluator.evaluate(test_case)


class TestFaithfulnessEvaluator:
    """Test suite for FaithfulnessEvaluator."""

    @patch("src.evaluators.faithfulness.FaithfulnessMetric")
    def test_evaluate_with_context_returns_score(self, mock_metric_class: MagicMock) -> None:
        """Verify faithfulness evaluator works with provided context."""
        # Arrange: mock metric with passing score
        mock_metric = MagicMock()
        mock_metric.score = 0.85
        mock_metric.reason = "Output is faithful to context"
        mock_metric_class.return_value = mock_metric

        # Create evaluator with context-bearing test case
        evaluator = FaithfulnessEvaluator(threshold=0.7)
        test_case = EvaluationTestCase(
            input_text="What is Paris?",
            actual_output="Paris is the capital of France.",
            retrieval_context=["Paris is the capital city of France."],
            name="faithfulness_pass_test",
        )

        # Act: run evaluation
        result = evaluator.evaluate(test_case)

        # Assert: score is returned correctly
        assert result.score == 0.85
        assert result.passed is True

    @patch("src.evaluators.faithfulness.FaithfulnessMetric")
    def test_evaluate_without_context_raises_value_error(self, mock_metric_class: MagicMock) -> None:
        """Verify ValueError is raised when retrieval context is empty."""
        # Arrange: create evaluator and test case without context
        evaluator = FaithfulnessEvaluator(threshold=0.7)
        test_case = EvaluationTestCase(
            input_text="What is AI?",
            actual_output="AI is artificial intelligence.",
            retrieval_context=[],
            name="faithfulness_no_context",
        )

        # Act & Assert: ValueError raised for missing context
        with pytest.raises(ValueError, match="requires non-empty retrieval_context"):
            evaluator.evaluate(test_case)


class TestHallucinationEvaluator:
    """Test suite for HallucinationEvaluator."""

    @patch("src.evaluators.hallucination.HallucinationMetric")
    def test_evaluate_with_context_returns_score(self, mock_metric_class: MagicMock) -> None:
        """Verify hallucination evaluator works with provided context."""
        # Arrange: mock metric with passing score
        mock_metric = MagicMock()
        mock_metric.score = 0.95
        mock_metric.reason = "No hallucinations detected"
        mock_metric_class.return_value = mock_metric

        # Create evaluator with context
        evaluator = HallucinationEvaluator(threshold=0.7)
        test_case = EvaluationTestCase(
            input_text="What is the Sun?",
            actual_output="The Sun is a star.",
            context=["The Sun is the star at the center of our solar system."],
            name="hallucination_pass_test",
        )

        # Act: run evaluation
        result = evaluator.evaluate(test_case)

        # Assert: no hallucinations gives a high score
        assert result.score == 0.95
        assert result.passed is True

    @patch("src.evaluators.hallucination.HallucinationMetric")
    def test_evaluate_without_context_raises_value_error(self, mock_metric_class: MagicMock) -> None:
        """Verify ValueError is raised when context is empty."""
        # Arrange: create evaluator and test case without context
        evaluator = HallucinationEvaluator(threshold=0.7)
        test_case = EvaluationTestCase(
            input_text="What is gravity?",
            actual_output="Gravity pulls objects toward Earth.",
            context=[],
            name="hallucination_no_context",
        )

        # Act & Assert: ValueError raised for missing context
        with pytest.raises(ValueError, match="requires non-empty context"):
            evaluator.evaluate(test_case)


class TestToxicityEvaluator:
    """Test suite for ToxicityEvaluator."""

    @patch("src.evaluators.toxicity.ToxicityMetric")
    def test_evaluate_non_toxic_output(self, mock_metric_class: MagicMock) -> None:
        """Verify non-toxic output receives a high score."""
        # Arrange: mock metric with passing score
        mock_metric = MagicMock()
        mock_metric.score = 0.95
        mock_metric.reason = "No toxic content detected"
        mock_metric_class.return_value = mock_metric

        # Create evaluator
        evaluator = ToxicityEvaluator(threshold=0.7)
        test_case = EvaluationTestCase(
            input_text="Tell me about dogs.",
            actual_output="Dogs are loyal companion animals.",
            name="toxicity_clean_test",
        )

        # Act: run evaluation
        result = evaluator.evaluate(test_case)

        # Assert: clean content passes
        assert result.score == 0.95
        assert result.passed is True


class TestBiasEvaluator:
    """Test suite for BiasEvaluator."""

    @patch("src.evaluators.bias.BiasMetric")
    def test_evaluate_unbiased_output(self, mock_metric_class: MagicMock) -> None:
        """Verify unbiased output receives a high score."""
        # Arrange: mock metric with passing score
        mock_metric = MagicMock()
        mock_metric.score = 0.9
        mock_metric.reason = "No bias detected"
        mock_metric_class.return_value = mock_metric

        # Create evaluator
        evaluator = BiasEvaluator(threshold=0.7)
        test_case = EvaluationTestCase(
            input_text="Describe software engineers.",
            actual_output="Software engineers write and maintain code for applications.",
            name="bias_clean_test",
        )

        # Act: run evaluation
        result = evaluator.evaluate(test_case)

        # Assert: unbiased content passes
        assert result.score == 0.9
        assert result.passed is True


class TestContextualRelevancyEvaluator:
    """Test suite for ContextualRelevancyEvaluator."""

    @patch("src.evaluators.contextual_relevancy.ContextualRelevancyMetric")
    def test_evaluate_relevant_context(self, mock_metric_class: MagicMock) -> None:
        """Verify relevant context receives a high score."""
        # Arrange: mock metric with passing score
        mock_metric = MagicMock()
        mock_metric.score = 0.88
        mock_metric.reason = "Context is highly relevant"
        mock_metric_class.return_value = mock_metric

        # Create evaluator with relevant context
        evaluator = ContextualRelevancyEvaluator(threshold=0.7)
        test_case = EvaluationTestCase(
            input_text="What is DNA?",
            actual_output="DNA is the molecule that carries genetic information.",
            retrieval_context=["DNA carries the genetic instructions for life."],
            name="context_relevancy_pass",
        )

        # Act: run evaluation
        result = evaluator.evaluate(test_case)

        # Assert: relevant context scores well
        assert result.score == 0.88
        assert result.passed is True

    @patch("src.evaluators.contextual_relevancy.ContextualRelevancyMetric")
    def test_evaluate_without_retrieval_context_raises_error(self, mock_metric_class: MagicMock) -> None:
        """Verify ValueError is raised when retrieval context is empty."""
        # Arrange: create evaluator without context
        evaluator = ContextualRelevancyEvaluator(threshold=0.7)
        test_case = EvaluationTestCase(
            input_text="What is RNA?",
            actual_output="RNA carries genetic information.",
            retrieval_context=[],
            name="context_relevancy_no_context",
        )

        # Act & Assert: ValueError for missing context
        with pytest.raises(ValueError, match="requires non-empty retrieval_context"):
            evaluator.evaluate(test_case)
