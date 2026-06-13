"""Contextual relevancy evaluator measuring retrieved context quality."""

from deepeval.metrics import ContextualRelevancyMetric  # DeepEval contextual relevancy metric

from src.evaluators.base_evaluator import BaseEvaluator  # Abstract base class
from src.models.evaluation_result import MetricScore  # Result data model
from src.models.test_case import EvaluationTestCase  # Test case data model
from src.utils.logger import get_logger  # Application logger

# Initialize module-level logger instance
logger = get_logger(__name__)


class ContextualRelevancyEvaluator(BaseEvaluator):
    """Evaluates whether the retrieved context is relevant to the input query.

    Measures how well the retrieval system selects documents that contain
    information needed to answer the original question.
    """

    def __init__(self, threshold: float = 0.7, model: str | None = None) -> None:
        """Initialize the contextual relevancy evaluator.

        Args:
            threshold: Minimum score to pass (0.0 to 1.0).
            model: Optional LLM model name for evaluation judging.
        """
        # Call parent constructor with shared settings
        super().__init__(threshold=threshold, model=model)
        # Create the DeepEval contextual relevancy metric instance
        self._metric = ContextualRelevancyMetric(
            threshold=self._threshold,
            model=self._model,
        )
        logger.info("ContextualRelevancyEvaluator initialized with threshold=%.2f", threshold)

    def evaluate(self, test_case: EvaluationTestCase) -> MetricScore:
        """Run contextual relevancy evaluation on a test case.

        Args:
            test_case: The test case with input and retrieval context to evaluate.

        Returns:
            MetricScore with the relevancy score and pass/fail status.

        Raises:
            RuntimeError: If the metric evaluation encounters an error.
            ValueError: If retrieval context is empty (required for this metric).
        """
        logger.info("Evaluating contextual relevancy for test case: %s", test_case.name)
        # This metric requires retrieval context to evaluate
        if not test_case.retrieval_context:
            logger.error("Contextual relevancy requires retrieval_context for '%s'", test_case.name)
            raise ValueError("Contextual relevancy evaluation requires non-empty retrieval_context")

        try:
            # Convert internal test case to DeepEval format
            deepeval_test_case = self.convert_to_deepeval_test_case(test_case)
            # Run the metric measurement
            self._metric.measure(deepeval_test_case)
            # Extract score from the metric result
            score = self._metric.score
            # Extract the explanation reason
            reason = self._metric.reason if hasattr(self._metric, "reason") else ""
            logger.info("Contextual relevancy score: %.4f for '%s'", score, test_case.name)
            # Build and return the structured metric score
            return self._build_metric_score(score=score, reason=reason)
        except ValueError:
            # Re-raise value errors without wrapping
            raise
        except Exception as evaluation_error:
            # Log the error and re-raise with context
            logger.error(
                "Contextual relevancy evaluation failed for '%s': %s",
                test_case.name,
                evaluation_error,
            )
            raise RuntimeError(
                f"Contextual relevancy evaluation failed: {evaluation_error}"
            ) from evaluation_error
