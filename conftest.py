"""Pytest configuration and shared fixtures for the test suite."""

import os  # Environment variable access

import pytest  # Test framework

from src.models.test_case import EvaluationTestCase  # Test case model
from src.utils.config_loader import ConfigLoader  # Configuration loader


@pytest.fixture(scope="session")
def test_config() -> ConfigLoader:
    """Provide test-environment configuration loader for the entire test session.

    Returns:
        ConfigLoader instance loaded with test settings.
    """
    # Set environment to test for loading test-specific overrides
    os.environ["APP_ENV"] = "test"
    # Return config loader that will merge base + test settings
    return ConfigLoader(environment="test")


@pytest.fixture
def sample_test_case() -> EvaluationTestCase:
    """Provide a basic sample test case for unit testing.

    Returns:
        EvaluationTestCase with populated fields for testing.
    """
    # Create a simple factual Q&A test case
    return EvaluationTestCase(
        input_text="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="Paris",
        retrieval_context=["France is a country in Western Europe. Its capital city is Paris."],
        context=["France is a European country with Paris as its capital."],
        name="capital_france_basic",
    )


@pytest.fixture
def sample_test_case_no_context() -> EvaluationTestCase:
    """Provide a test case without retrieval context for boundary testing.

    Returns:
        EvaluationTestCase with empty context fields.
    """
    # Create test case without context for testing error handling
    return EvaluationTestCase(
        input_text="What is machine learning?",
        actual_output="Machine learning is a subset of AI that enables systems to learn from data.",
        expected_output="Machine learning is a branch of artificial intelligence.",
        retrieval_context=[],
        context=[],
        name="ml_definition_no_context",
    )


@pytest.fixture
def batch_test_cases() -> list[EvaluationTestCase]:
    """Provide multiple test cases for batch evaluation testing.

    Returns:
        List of EvaluationTestCase instances.
    """
    # Create a diverse set of test cases for batch operations
    return [
        EvaluationTestCase(
            input_text="What is Python?",
            actual_output="Python is a high-level programming language.",
            expected_output="Python is a programming language.",
            retrieval_context=["Python is a high-level, interpreted programming language."],
            context=["Python is known for its readability and versatility."],
            name="python_definition",
        ),
        EvaluationTestCase(
            input_text="What is the speed of light?",
            actual_output="The speed of light is approximately 299,792,458 meters per second.",
            expected_output="299,792,458 m/s",
            retrieval_context=["Light travels at 299,792,458 meters per second in vacuum."],
            context=["The speed of light in vacuum is a universal constant."],
            name="speed_of_light",
        ),
    ]
