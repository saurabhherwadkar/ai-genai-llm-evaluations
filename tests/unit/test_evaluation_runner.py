"""Unit tests for the EvaluationRunner orchestration module."""

from unittest.mock import MagicMock, patch  # Mocking utilities

import pytest  # Test framework

from src.evaluators.base_evaluator import BaseEvaluator  # Evaluator base class
from src.models.evaluation_result import MetricScore  # Metric score model
from src.models.test_case import EvaluationTestCase  # Test case model
from src.runners.evaluation_runner import EvaluationRunner  # Module under test


class TestEvaluationRunner:
    """Test suite for EvaluationRunner orchestration logic."""

    @patch("src.runners.evaluation_runner.ConfigLoader")
    def test_add_evaluator_increases_count(self, mock_config: MagicMock) -> None:
        """Verify adding evaluators increments the evaluator count."""
        # Arrange: create runner and mock evaluator
        mock_config.return_value.get.return_value = 0.7
        runner = EvaluationRunner(config=mock_config.return_value)
        mock_evaluator = MagicMock(spec=BaseEvaluator)
        mock_evaluator.metric_name = "TestMetric"

        # Act: add the evaluator
        runner.add_evaluator(mock_evaluator)

        # Assert: count increases by one
        assert runner.evaluator_count == 1

    @patch("src.runners.evaluation_runner.ConfigLoader")
    def test_add_evaluators_adds_multiple(self, mock_config: MagicMock) -> None:
        """Verify adding multiple evaluators at once works correctly."""
        # Arrange: create runner and multiple mock evaluators
        mock_config.return_value.get.return_value = 0.7
        runner = EvaluationRunner(config=mock_config.return_value)
        evaluators = [
            MagicMock(spec=BaseEvaluator, metric_name="Metric1"),
            MagicMock(spec=BaseEvaluator, metric_name="Metric2"),
            MagicMock(spec=BaseEvaluator, metric_name="Metric3"),
        ]

        # Act: add all evaluators
        runner.add_evaluators(evaluators)

        # Assert: all evaluators are registered
        assert runner.evaluator_count == 3

    @patch("src.runners.evaluation_runner.ConfigLoader")
    def test_run_executes_all_evaluators(self, mock_config: MagicMock) -> None:
        """Verify all registered evaluators are executed during a run."""
        # Arrange: set up runner with two mock evaluators
        mock_config.return_value.get.return_value = 0.7
        runner = EvaluationRunner(config=mock_config.return_value)

        # Create mock evaluators that return passing scores
        evaluator_1 = MagicMock(spec=BaseEvaluator)
        evaluator_1.metric_name = "Metric1"
        evaluator_1.evaluate.return_value = MetricScore(
            metric_name="Metric1", score=0.9, passed=True, threshold=0.7
        )

        evaluator_2 = MagicMock(spec=BaseEvaluator)
        evaluator_2.metric_name = "Metric2"
        evaluator_2.evaluate.return_value = MetricScore(
            metric_name="Metric2", score=0.85, passed=True, threshold=0.7
        )

        runner.add_evaluators([evaluator_1, evaluator_2])

        # Create test case
        test_case = EvaluationTestCase(
            input_text="Test question",
            actual_output="Test answer",
            name="runner_test",
        )

        # Act: run evaluation
        result = runner.run(test_case)

        # Assert: both evaluators were called and result is correct
        evaluator_1.evaluate.assert_called_once_with(test_case)
        evaluator_2.evaluate.assert_called_once_with(test_case)
        assert len(result.metric_scores) == 2
        assert result.overall_passed is True

    @patch("src.runners.evaluation_runner.ConfigLoader")
    def test_run_handles_evaluator_failure_gracefully(self, mock_config: MagicMock) -> None:
        """Verify a failing evaluator doesn't prevent others from running."""
        # Arrange: create runner with one failing and one passing evaluator
        mock_config.return_value.get.return_value = 0.7
        runner = EvaluationRunner(config=mock_config.return_value)

        # First evaluator raises an error
        failing_evaluator = MagicMock(spec=BaseEvaluator)
        failing_evaluator.metric_name = "FailMetric"
        failing_evaluator.threshold = 0.7
        failing_evaluator.evaluate.side_effect = RuntimeError("Metric failed")

        # Second evaluator succeeds
        passing_evaluator = MagicMock(spec=BaseEvaluator)
        passing_evaluator.metric_name = "PassMetric"
        passing_evaluator.evaluate.return_value = MetricScore(
            metric_name="PassMetric", score=0.9, passed=True, threshold=0.7
        )

        runner.add_evaluators([failing_evaluator, passing_evaluator])

        # Create test case
        test_case = EvaluationTestCase(
            input_text="Test input",
            actual_output="Test output",
            name="failure_handling_test",
        )

        # Act: run evaluation
        result = runner.run(test_case)

        # Assert: both evaluators attempted, overall fails due to one failure
        assert len(result.metric_scores) == 2
        assert result.overall_passed is False
        assert result.metric_scores[0].score == 0.0
        assert result.metric_scores[1].score == 0.9

    @patch("src.runners.evaluation_runner.ConfigLoader")
    def test_run_batch_processes_all_test_cases(self, mock_config: MagicMock) -> None:
        """Verify batch execution processes every test case in the list."""
        # Arrange: create runner with a mock evaluator
        mock_config.return_value.get.return_value = 0.7
        runner = EvaluationRunner(config=mock_config.return_value)

        mock_evaluator = MagicMock(spec=BaseEvaluator)
        mock_evaluator.metric_name = "BatchMetric"
        mock_evaluator.evaluate.return_value = MetricScore(
            metric_name="BatchMetric", score=0.8, passed=True, threshold=0.7
        )
        runner.add_evaluator(mock_evaluator)

        # Create batch of test cases
        test_cases = [
            EvaluationTestCase(input_text="Q1", actual_output="A1", name="batch_1"),
            EvaluationTestCase(input_text="Q2", actual_output="A2", name="batch_2"),
            EvaluationTestCase(input_text="Q3", actual_output="A3", name="batch_3"),
        ]

        # Act: run batch evaluation
        results = runner.run_batch(test_cases)

        # Assert: all test cases processed
        assert len(results) == 3
        assert mock_evaluator.evaluate.call_count == 3

    @patch("src.runners.evaluation_runner.ConfigLoader")
    def test_run_records_duration(self, mock_config: MagicMock) -> None:
        """Verify execution duration is recorded in the result."""
        # Arrange: create runner with a mock evaluator
        mock_config.return_value.get.return_value = 0.7
        runner = EvaluationRunner(config=mock_config.return_value)

        mock_evaluator = MagicMock(spec=BaseEvaluator)
        mock_evaluator.metric_name = "DurationMetric"
        mock_evaluator.evaluate.return_value = MetricScore(
            metric_name="DurationMetric", score=0.9, passed=True, threshold=0.7
        )
        runner.add_evaluator(mock_evaluator)

        # Create test case
        test_case = EvaluationTestCase(
            input_text="Test", actual_output="Output", name="duration_test"
        )

        # Act: run evaluation
        result = runner.run(test_case)

        # Assert: duration is positive
        assert result.duration_seconds >= 0.0

    @patch("src.runners.evaluation_runner.ConfigLoader")
    def test_run_with_no_evaluators_returns_failed_result(self, mock_config: MagicMock) -> None:
        """Verify running with no evaluators returns a failed result."""
        # Arrange: create runner without any evaluators
        mock_config.return_value.get.return_value = 0.7
        runner = EvaluationRunner(config=mock_config.return_value)

        # Create test case
        test_case = EvaluationTestCase(
            input_text="Test", actual_output="Output", name="no_evaluator_test"
        )

        # Act: run evaluation without evaluators
        result = runner.run(test_case)

        # Assert: result fails with no scores
        assert len(result.metric_scores) == 0
        assert result.overall_passed is False
