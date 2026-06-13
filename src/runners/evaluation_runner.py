"""Evaluation runner module orchestrating metric evaluation execution."""

import time  # Timing evaluation duration

from src.evaluators.base_evaluator import BaseEvaluator  # Evaluator base class
from src.models.evaluation_result import EvaluationResult, MetricScore  # Result models
from src.models.test_case import EvaluationTestCase  # Test case model
from src.utils.config_loader import ConfigLoader  # Configuration access
from src.utils.logger import get_logger  # Application logger

# Initialize module-level logger instance
logger = get_logger(__name__)


class EvaluationRunner:
    """Orchestrates running multiple evaluation metrics against test cases.

    Manages evaluator instances, executes them against test cases,
    collects results, and determines overall pass/fail status.
    """

    def __init__(self, config: ConfigLoader | None = None) -> None:
        """Initialize the evaluation runner.

        Args:
            config: Optional configuration loader instance.
        """
        # Store the config loader for reading settings
        self._config = config or ConfigLoader()
        # List of registered evaluator instances
        self._evaluators: list[BaseEvaluator] = []
        # Read the passing threshold from configuration
        self._threshold = self._config.get("evaluation.passing_threshold", 0.7)
        logger.info("EvaluationRunner initialized with threshold=%.2f", self._threshold)

    def add_evaluator(self, evaluator: BaseEvaluator) -> None:
        """Register an evaluator instance for use during evaluation.

        Args:
            evaluator: The evaluator to add to the evaluation pipeline.
        """
        # Append the evaluator to the list of active evaluators
        self._evaluators.append(evaluator)
        logger.info("Added evaluator: %s", evaluator.metric_name)

    def add_evaluators(self, evaluators: list[BaseEvaluator]) -> None:
        """Register multiple evaluator instances at once.

        Args:
            evaluators: List of evaluators to add to the pipeline.
        """
        # Iterate and add each evaluator individually for logging
        for evaluator in evaluators:
            self.add_evaluator(evaluator)

    def run(self, test_case: EvaluationTestCase) -> EvaluationResult:
        """Execute all registered evaluators against a single test case.

        Args:
            test_case: The test case to evaluate with all metrics.

        Returns:
            EvaluationResult containing all metric scores and overall status.
        """
        logger.info("Running evaluation for test case: %s", test_case.name)
        # Record start time for duration tracking
        start_time = time.time()
        # Collect metric scores from each evaluator
        metric_scores: list[MetricScore] = []

        # Iterate through each registered evaluator
        for evaluator in self._evaluators:
            try:
                # Execute the evaluator and collect the score
                score = evaluator.evaluate(test_case)
                metric_scores.append(score)
                logger.debug(
                    "Metric '%s' scored %.4f (passed=%s)",
                    score.metric_name,
                    score.score,
                    score.passed,
                )
            except (ValueError, RuntimeError) as eval_error:
                # Log failed evaluations but continue with remaining metrics
                logger.warning(
                    "Evaluator '%s' failed for '%s': %s",
                    evaluator.metric_name,
                    test_case.name,
                    eval_error,
                )
                # Record a zero score for failed evaluations
                failed_score = MetricScore(
                    metric_name=evaluator.metric_name,
                    score=0.0,
                    passed=False,
                    threshold=evaluator.threshold,
                    reason=f"Evaluation failed: {eval_error}",
                )
                metric_scores.append(failed_score)

        # Calculate total duration in seconds
        duration = time.time() - start_time
        # Build the evaluation result object
        result = EvaluationResult(
            test_case_name=test_case.name or "unnamed",
            metric_scores=metric_scores,
            duration_seconds=round(duration, 3),
        )
        # Determine overall pass/fail from individual scores
        result.compute_overall_status()
        logger.info(
            "Evaluation complete for '%s': overall_passed=%s, duration=%.3fs",
            test_case.name,
            result.overall_passed,
            duration,
        )
        return result

    def run_batch(self, test_cases: list[EvaluationTestCase]) -> list[EvaluationResult]:
        """Execute evaluations against a batch of test cases.

        Args:
            test_cases: List of test cases to evaluate.

        Returns:
            List of EvaluationResult instances, one per test case.
        """
        logger.info("Running batch evaluation for %d test cases", len(test_cases))
        # Collect results for each test case in order
        results: list[EvaluationResult] = []
        # Process each test case sequentially
        for test_case in test_cases:
            result = self.run(test_case)
            results.append(result)
        # Log summary statistics for the batch
        passed_count = sum(1 for r in results if r.overall_passed)
        logger.info(
            "Batch evaluation complete: %d/%d passed",
            passed_count,
            len(results),
        )
        return results

    @property
    def evaluator_count(self) -> int:
        """Return the number of registered evaluators.

        Returns:
            Integer count of evaluators.
        """
        return len(self._evaluators)
