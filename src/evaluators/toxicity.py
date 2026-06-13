"""Toxicity evaluator detecting harmful or offensive content in LLM outputs."""

from deepeval.metrics import ToxicityMetric  # DeepEval toxicity metric

from src.evaluators.base_evaluator import BaseEvaluator  # Abstract base class
from src.models.evaluation_result import MetricScore  # Result data model
from src.models.test_case import EvaluationTestCase  # Test case data model
from src.utils.logger import get_logger  # Application logger

# Initialize module-level logger instance
logger = get_logger(__name__)


class ToxicityEvaluator(BaseEvaluator):
    """Evaluates whether the LLM output contains toxic or harmful content.

    Detects offensive language, hate speech, threats, or other harmful
    content that should not appear in LLM-generated responses.
    """

    def __init__(self, threshold: float = 0.7, model: str | None = None) -> None:
        """Initialize the toxicity evaluator.

        Args:
            threshold: Minimum score to pass (higher means less toxic).
            model: Optional LLM model name for evaluation judging.
        """
        # Call parent constructor with shared settings
        super().__init__(threshold=threshold, model=model)
        # Create the DeepEval toxicity metric instance
        self._metric = ToxicityMetric(
            threshold=self._threshold,
            model=self._model,
        )
        logger.info("ToxicityEvaluator initialized with threshold=%.2f", threshold)

    def evaluate(self, test_case: EvaluationTestCase) -> MetricScore:
        """Run toxicity evaluation on a test case.

        Args:
            test_case: The test case with output to check for toxicity.

        Returns:
            MetricScore with the toxicity score and pass/fail status.

        Raises:
            RuntimeError: If the metric evaluation encounters an error.
        """
        logger.info("Evaluating toxicity for test case: %s", test_case.name)
        try:
            # Convert internal test case to DeepEval format
            deepeval_test_case = self.convert_to_deepeval_test_case(test_case)
            # Run the metric measurement
            self._metric.measure(deepeval_test_case)
            # Extract score from the metric result
            score = self._metric.score
            # Extract the explanation reason
            reason = self._metric.reason if hasattr(self._metric, "reason") else ""
            logger.info("Toxicity score: %.4f for '%s'", score, test_case.name)
            # Build and return the structured metric score
            return self._build_metric_score(score=score, reason=reason)
        except Exception as evaluation_error:
            # Log the error and re-raise with context
            logger.error(
                "Toxicity evaluation failed for '%s': %s",
                test_case.name,
                evaluation_error,
            )
            raise RuntimeError(
                f"Toxicity evaluation failed: {evaluation_error}"
            ) from evaluation_error
