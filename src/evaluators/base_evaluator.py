"""Abstract base class for all evaluation metric implementations."""

from abc import ABC, abstractmethod  # Abstract base class support

from deepeval.test_case import LLMTestCase  # DeepEval test case type

from src.models.evaluation_result import MetricScore  # Internal metric result model
from src.models.test_case import EvaluationTestCase  # Internal test case model
from src.utils.logger import get_logger  # Application logger

# Initialize module-level logger instance
logger = get_logger(__name__)


class BaseEvaluator(ABC):
    """Abstract base class defining the interface for all evaluators.

    Each evaluator wraps a specific DeepEval metric and provides
    a consistent interface for running evaluations across metric types.
    """

    def __init__(self, threshold: float = 0.7, model: str | None = None) -> None:
        """Initialize the base evaluator with common settings.

        Args:
            threshold: Minimum score required to pass (0.0 to 1.0).
            model: Optional LLM model name for evaluation judging.
        """
        # Store the minimum passing threshold
        self._threshold = threshold
        # Store the model identifier for LLM-based metrics
        self._model = model
        # Log evaluator creation with configuration
        logger.debug(
            "Initialized %s with threshold=%.2f, model=%s",
            self.__class__.__name__,
            threshold,
            model,
        )

    @property
    def threshold(self) -> float:
        """Read-only access to the passing threshold.

        Returns:
            The configured minimum passing score.
        """
        return self._threshold

    @property
    def metric_name(self) -> str:
        """Provide the human-readable name of this evaluation metric.

        Returns:
            String name of the metric.
        """
        return self.__class__.__name__.replace("Evaluator", "")

    def convert_to_deepeval_test_case(self, test_case: EvaluationTestCase) -> LLMTestCase:
        """Convert internal test case model to DeepEval's LLMTestCase format.

        Args:
            test_case: The internal evaluation test case to convert.

        Returns:
            DeepEval-compatible LLMTestCase instance.
        """
        # Map internal fields to DeepEval's expected fields
        return LLMTestCase(
            input=test_case.input_text,
            actual_output=test_case.actual_output,
            expected_output=test_case.expected_output if test_case.expected_output else None,
            retrieval_context=test_case.retrieval_context if test_case.retrieval_context else None,
            context=test_case.context if test_case.context else None,
        )

    @abstractmethod
    def evaluate(self, test_case: EvaluationTestCase) -> MetricScore:
        """Execute the evaluation metric against a test case.

        Args:
            test_case: The test case containing input and output to evaluate.

        Returns:
            MetricScore containing the evaluation result.

        Raises:
            EvaluationError: If the evaluation cannot be completed.
        """
        ...

    def _build_metric_score(self, score: float, reason: str = "") -> MetricScore:
        """Construct a MetricScore result from raw evaluation output.

        Args:
            score: The numeric score from the evaluation (0.0 to 1.0).
            reason: Optional explanation text from the metric.

        Returns:
            Fully populated MetricScore instance.
        """
        # Determine pass/fail by comparing against threshold
        passed = score >= self._threshold
        # Build and return the structured result
        return MetricScore(
            metric_name=self.metric_name,
            score=score,
            passed=passed,
            threshold=self._threshold,
            reason=reason,
        )
