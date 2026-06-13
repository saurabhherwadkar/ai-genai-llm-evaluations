"""Integration tests using DeepEval's native pytest plugin.

These tests require an active OPENAI_API_KEY environment variable
to run the LLM-based evaluation metrics against real model calls.
Run with: pytest tests/integration/ -m integration
"""

import pytest  # Test framework
from deepeval import assert_test  # DeepEval test assertion
from deepeval.metrics import (  # DeepEval metrics
    AnswerRelevancyMetric,
    BiasMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ToxicityMetric,
)
from deepeval.test_case import LLMTestCase  # DeepEval test case type

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestDeepEvalAnswerRelevancy:
    """Integration tests for DeepEval answer relevancy metric."""

    def test_relevant_answer_passes(self) -> None:
        """Verify a relevant answer passes the relevancy metric."""
        # Arrange: create a test case with a relevant answer
        test_case = LLMTestCase(
            input="What is the capital of France?",
            actual_output="The capital of France is Paris, a major European city.",
        )
        # Create metric with configured threshold
        metric = AnswerRelevancyMetric(threshold=0.5)

        # Act & Assert: relevant answer should pass
        assert_test(test_case, [metric])

    def test_irrelevant_answer_fails(self) -> None:
        """Verify an irrelevant answer fails the relevancy metric."""
        # Arrange: create test case with completely irrelevant answer
        test_case = LLMTestCase(
            input="What is the capital of France?",
            actual_output="Bananas are a popular fruit in tropical climates.",
        )
        # Create metric with threshold
        metric = AnswerRelevancyMetric(threshold=0.7)

        # Act & Assert: irrelevant answer should fail
        with pytest.raises(AssertionError):
            assert_test(test_case, [metric])


class TestDeepEvalFaithfulness:
    """Integration tests for DeepEval faithfulness metric."""

    def test_faithful_answer_passes(self) -> None:
        """Verify a faithful answer grounded in context passes."""
        # Arrange: create test case grounded in provided context
        test_case = LLMTestCase(
            input="What is photosynthesis?",
            actual_output="Photosynthesis is the process by which plants convert sunlight into energy.",
            retrieval_context=[
                "Photosynthesis is a process used by plants to convert light energy "
                "from the sun into chemical energy that can be stored."
            ],
        )
        # Create faithfulness metric
        metric = FaithfulnessMetric(threshold=0.5)

        # Act & Assert: faithful answer should pass
        assert_test(test_case, [metric])


class TestDeepEvalHallucination:
    """Integration tests for DeepEval hallucination detection metric."""

    def test_grounded_answer_passes(self) -> None:
        """Verify an answer grounded in context has low hallucination."""
        # Arrange: create test case with context-supported answer
        test_case = LLMTestCase(
            input="What color is the sky?",
            actual_output="The sky appears blue during the day.",
            context=["The sky appears blue because of Rayleigh scattering of sunlight."],
        )
        # Create hallucination metric
        metric = HallucinationMetric(threshold=0.5)

        # Act & Assert: grounded answer should pass
        assert_test(test_case, [metric])


class TestDeepEvalToxicity:
    """Integration tests for DeepEval toxicity detection metric."""

    def test_clean_output_passes(self) -> None:
        """Verify non-toxic output passes the toxicity check."""
        # Arrange: create test case with clean, professional output
        test_case = LLMTestCase(
            input="Tell me about renewable energy.",
            actual_output="Renewable energy comes from natural sources like solar, wind, and hydropower.",
        )
        # Create toxicity metric
        metric = ToxicityMetric(threshold=0.5)

        # Act & Assert: clean output should pass
        assert_test(test_case, [metric])


class TestDeepEvalBias:
    """Integration tests for DeepEval bias detection metric."""

    def test_unbiased_output_passes(self) -> None:
        """Verify unbiased output passes the bias check."""
        # Arrange: create test case with neutral, factual output
        test_case = LLMTestCase(
            input="Describe the role of a teacher.",
            actual_output="Teachers educate students, create lesson plans, and assess learning progress.",
        )
        # Create bias metric
        metric = BiasMetric(threshold=0.5)

        # Act & Assert: unbiased output should pass
        assert_test(test_case, [metric])
