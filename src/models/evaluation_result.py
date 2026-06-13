"""Data model for storing evaluation results and metric scores."""

from datetime import datetime, timezone  # Timestamp handling

from pydantic import BaseModel, Field  # Data validation framework


class MetricScore(BaseModel):
    """Represents a single metric score from an evaluation.

    Stores the metric name, numeric score, and pass/fail status.
    """

    # Name identifier of the metric that was evaluated
    metric_name: str = Field(
        ...,
        description="Name of the evaluation metric",
    )

    # Numeric score value between 0.0 and 1.0
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Numeric score from 0.0 to 1.0",
    )

    # Whether this metric score meets the passing threshold
    passed: bool = Field(
        ...,
        description="Whether the score meets the minimum threshold",
    )

    # Threshold that was used to determine pass/fail
    threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum threshold for a passing score",
    )

    # Explanation or reason provided by the evaluation metric
    reason: str = Field(
        default="",
        description="Explanation of the score from the evaluator",
    )


class EvaluationResult(BaseModel):
    """Aggregates all metric scores for a single test case evaluation.

    Provides overall pass/fail status and detailed per-metric results.
    """

    # Name of the test case that was evaluated
    test_case_name: str = Field(
        ...,
        description="Identifier of the evaluated test case",
    )

    # Collection of individual metric scores
    metric_scores: list[MetricScore] = Field(
        default_factory=list,
        description="List of scores for each metric evaluated",
    )

    # Overall pass/fail for the entire test case
    overall_passed: bool = Field(
        default=False,
        description="Whether all metrics passed their thresholds",
    )

    # Timestamp when the evaluation was completed
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of when the evaluation ran",
    )

    # Total time taken for evaluation in seconds
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken to complete the evaluation",
    )

    def compute_overall_status(self) -> None:
        """Calculate overall pass/fail based on individual metric results.

        Sets overall_passed to True only if all metrics passed.
        """
        # If no metrics were evaluated, mark as failed
        if not self.metric_scores:
            self.overall_passed = False
            return
        # Pass only if every metric individually passed
        self.overall_passed = all(score.passed for score in self.metric_scores)
