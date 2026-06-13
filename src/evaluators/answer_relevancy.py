"""Answer relevancy evaluator measuring how well outputs address inputs."""

from deepeval.metrics import AnswerRelevancyMetric  # DeepEval answer relevancy metric

from src.evaluators.base_evaluator import BaseEvaluator  # Abstract base class
from src.models.evaluation_result import MetricScore  # Result data model
from src.models.test_case import EvaluationTestCase  # Test case data model
from src.utils.logger import get_logger  # Application logger

# Initialize module-level logger instance
logger = get_logger(__name__)


class AnswerRelevancyEvaluator(BaseEvaluator):
    """Evaluates whether the LLM output is relevant to the input question.

    Uses DeepEval's AnswerRelevancyMetric to score how directly the
    response addresses the original query without unnecessary information.
    """

    def __init__(self, threshold: float = 0.7, model: str | None = None) -> None:
        """Initialize the answer relevancy evaluator.

        Args:
            threshold: Minimum relevancy score to pass (0.0 to 1.0).
            model: Optional LLM model name for evaluation judging.
        """
        # Call parent constructor with shared settings
        super().__init__(threshold=threshold, model=model)
        # Create the DeepEval metric instance with configured threshold
        self._metric = AnswerRelevancyMetric(
            threshold=self._threshold,
            model=self._model,
        )
        logger.info("AnswerRelevancyEvaluator initialized with threshold=%.2f", threshold)

    def evaluate(self, test_case: EvaluationTestCase) -> MetricScore:
        """Run answer relevancy evaluation on a test case.

        Args:
            test_case: The test case with input and output to evaluate.

        Returns:
            MetricScore with the relevancy score and pass/fail status.

        Raises:
            RuntimeError: If the metric evaluation encounters an error.
        """
        logger.info("Evaluating answer relevancy for test case: %s", test_case.name)
        try:
            # Convert internal test case to DeepEval format
            deepeval_test_case = self.convert_to_deepeval_test_case(test_case)
            # Run the metric measurement
            self._metric.measure(deepeval_test_case)
            # Extract score from the metric result
            score = self._metric.score
            # Extract the explanation reason
            reason = self._metric.reason if hasattr(self._metric, "reason") else ""
            logger.info("Answer relevancy score: %.4f for '%s'", score, test_case.name)
            # Build and return the structured metric score
            return self._build_metric_score(score=score, reason=reason)
        except Exception as evaluation_error:
            # Log the error and re-raise with context
            logger.error(
                "Answer relevancy evaluation failed for '%s': %s",
                test_case.name,
                evaluation_error,
            )
            raise RuntimeError(
                f"Answer relevancy evaluation failed: {evaluation_error}"
            ) from evaluation_error
