"""Hallucination evaluator detecting fabricated content in LLM outputs."""

from deepeval.metrics import HallucinationMetric  # DeepEval hallucination metric

from src.evaluators.base_evaluator import BaseEvaluator  # Abstract base class
from src.models.evaluation_result import MetricScore  # Result data model
from src.models.test_case import EvaluationTestCase  # Test case data model
from src.utils.logger import get_logger  # Application logger

# Initialize module-level logger instance
logger = get_logger(__name__)


class HallucinationEvaluator(BaseEvaluator):
    """Evaluates whether the LLM output contains hallucinated content.

    Detects fabricated claims, made-up facts, or information not grounded
    in the provided context documents. Lower hallucination means higher quality.
    """

    def __init__(self, threshold: float = 0.7, model: str | None = None) -> None:
        """Initialize the hallucination evaluator.

        Args:
            threshold: Minimum score to pass (higher means less hallucination).
            model: Optional LLM model name for evaluation judging.
        """
        # Call parent constructor with shared settings
        super().__init__(threshold=threshold, model=model)
        # Create the DeepEval hallucination metric instance
        self._metric = HallucinationMetric(
            threshold=self._threshold,
            model=self._model,
        )
        logger.info("HallucinationEvaluator initialized with threshold=%.2f", threshold)

    def evaluate(self, test_case: EvaluationTestCase) -> MetricScore:
        """Run hallucination detection evaluation on a test case.

        Args:
            test_case: The test case with output and context to evaluate.

        Returns:
            MetricScore with the hallucination score and pass/fail status.

        Raises:
            RuntimeError: If the metric evaluation encounters an error.
            ValueError: If context is empty (required for hallucination detection).
        """
        logger.info("Evaluating hallucination for test case: %s", test_case.name)
        # Hallucination detection requires context to compare against
        if not test_case.context:
            logger.error("Hallucination evaluation requires context for '%s'", test_case.name)
            raise ValueError("Hallucination evaluation requires non-empty context")

        try:
            # Convert internal test case to DeepEval format
            deepeval_test_case = self.convert_to_deepeval_test_case(test_case)
            # Run the metric measurement
            self._metric.measure(deepeval_test_case)
            # Extract score from the metric result
            score = self._metric.score
            # Extract the explanation reason
            reason = self._metric.reason if hasattr(self._metric, "reason") else ""
            logger.info("Hallucination score: %.4f for '%s'", score, test_case.name)
            # Build and return the structured metric score
            return self._build_metric_score(score=score, reason=reason)
        except ValueError:
            # Re-raise value errors without wrapping
            raise
        except Exception as evaluation_error:
            # Log the error and re-raise with context
            logger.error(
                "Hallucination evaluation failed for '%s': %s",
                test_case.name,
                evaluation_error,
            )
            raise RuntimeError(
                f"Hallucination evaluation failed: {evaluation_error}"
            ) from evaluation_error
