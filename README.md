# LLM & Agent Evaluation Framework

A comprehensive Python framework for evaluating Large Language Model (LLM) and AI agent responses using [DeepEval](https://github.com/confident-ai/deepeval) and [pytest](https://pytest.org/). This project provides structured evaluation pipelines across multiple quality dimensions including relevancy, faithfulness, hallucination detection, toxicity, and bias.

---

## Project Structure

```
ai-genai-llm-evaluations/
├── config/
│   ├── settings.yaml              # Base application configuration
│   └── settings.test.yaml         # Test environment overrides
├── src/
│   ├── __init__.py
│   ├── evaluators/
│   │   ├── __init__.py
│   │   ├── base_evaluator.py      # Abstract base class for all evaluators
│   │   ├── answer_relevancy.py    # Answer relevancy metric evaluator
│   │   ├── faithfulness.py        # Faithfulness metric evaluator
│   │   ├── hallucination.py       # Hallucination detection evaluator
│   │   ├── toxicity.py            # Toxicity detection evaluator
│   │   ├── bias.py                # Bias detection evaluator
│   │   └── contextual_relevancy.py # Contextual relevancy evaluator
│   ├── models/
│   │   ├── __init__.py
│   │   ├── test_case.py           # Pydantic model for test cases
│   │   └── evaluation_result.py   # Pydantic model for results
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config_loader.py       # YAML configuration loader
│   │   └── logger.py              # Centralized logging setup
│   └── runners/
│       ├── __init__.py
│       └── evaluation_runner.py   # Orchestrates multi-metric evaluation
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_config_loader.py  # Config loader unit tests
│   │   ├── test_models.py         # Data model unit tests
│   │   ├── test_evaluators.py     # Evaluator unit tests (mocked)
│   │   └── test_evaluation_runner.py # Runner unit tests
│   └── integration/
│       ├── __init__.py
│       └── test_deepeval_integration.py # Real API integration tests
├── conftest.py                    # Shared pytest fixtures
├── pyproject.toml                 # Project metadata and tool config
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore patterns
└── README.md                      # This file
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `deepeval` >= 2.5.0 | LLM evaluation metrics framework |
| `pytest` >= 8.3.0 | Test framework and runner |
| `openai` >= 1.50.0 | OpenAI API client for LLM-based metrics |
| `anthropic` >= 0.40.0 | Anthropic API client (optional) |
| `pydantic` >= 2.9.0 | Data validation and serialization |
| `pyyaml` >= 6.0.2 | YAML configuration parsing |
| `python-dotenv` >= 1.0.1 | Environment variable management |
| `tenacity` >= 9.0.0 | Retry logic for API calls |
| `pytest-cov` >= 6.0.0 | Test coverage reporting |
| `pytest-mock` >= 3.14.0 | Mocking utilities for pytest |
| `ruff` >= 0.7.0 | Python linter and formatter |
| `mypy` >= 1.13.0 | Static type checking |

---

## Deployment & Setup

### Prerequisites

- Python 3.11 or higher
- An OpenAI API key (required for DeepEval's LLM-based metrics)
- Git

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-genai-llm-evaluations
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

5. **Verify installation:**
   ```bash
   pytest tests/unit/ -v
   ```

### Running Tests

**Unit tests (no API key required):**
```bash
pytest tests/unit/ -v
```

**Integration tests (requires OPENAI_API_KEY):**
```bash
pytest tests/integration/ -m integration -v
```

**All tests with coverage:**
```bash
pytest --cov=src --cov-report=html
```

**Run specific metric tests:**
```bash
pytest tests/unit/test_evaluators.py::TestAnswerRelevancyEvaluator -v
```

### Configuration

The framework uses YAML-based configuration in `config/settings.yaml`. Override values per environment by creating `config/settings.<env>.yaml` files.

Set the active environment via the `APP_ENV` environment variable:
```bash
export APP_ENV=test
```

Key configuration options:
- `logging.level` - Log verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `evaluation.passing_threshold` - Minimum score to pass (0.0 to 1.0)
- `llm.default_model` - Model used for evaluation judging
- `deepeval.telemetry_enabled` - Whether to report to Confident AI

### Available Evaluation Metrics

| Metric | Description | Requires Context |
|--------|-------------|:---:|
| Answer Relevancy | How well the output addresses the input | No |
| Faithfulness | Factual consistency with retrieval context | Yes (retrieval_context) |
| Hallucination | Detection of fabricated content | Yes (context) |
| Toxicity | Detection of harmful or offensive content | No |
| Bias | Detection of discriminatory language | No |
| Contextual Relevancy | Quality of retrieved context documents | Yes (retrieval_context) |

### Usage Example

```python
from src.evaluators import AnswerRelevancyEvaluator, ToxicityEvaluator
from src.models.test_case import EvaluationTestCase
from src.runners.evaluation_runner import EvaluationRunner

# Create a test case
test_case = EvaluationTestCase(
    input_text="What is machine learning?",
    actual_output="Machine learning is a subset of AI that learns from data.",
    expected_output="Machine learning is a branch of artificial intelligence.",
    name="ml_definition",
)

# Set up the evaluation runner
runner = EvaluationRunner()
runner.add_evaluators([
    AnswerRelevancyEvaluator(threshold=0.7),
    ToxicityEvaluator(threshold=0.7),
])

# Run evaluation
result = runner.run(test_case)
print(f"Overall Passed: {result.overall_passed}")
for score in result.metric_scores:
    print(f"  {score.metric_name}: {score.score:.2f} ({'PASS' if score.passed else 'FAIL'})")
```

---

## End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION PIPELINE FLOW                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────────┐     ┌─────────────────────────────┐
│  config/     │     │  .env            │     │  User Code / Test Suite     │
│  settings.   │     │  (API keys,      │     │  (pytest or direct import)  │
│  yaml        │     │   environment)   │     │                             │
└──────┬───────┘     └────────┬─────────┘     └──────────────┬──────────────┘
       │                      │                               │
       ▼                      ▼                               │
┌──────────────────────────────────┐                          │
│         ConfigLoader             │                          │
│  • Reads base YAML settings      │                          │
│  • Merges env-specific overrides │                          │
│  • Provides dot-notation access  │                          │
└──────────────┬───────────────────┘                          │
               │                                              │
               ▼                                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        EvaluationRunner                                    │
│  • Initialized with config (threshold, model settings)                    │
│  • Accepts one or more Evaluator instances via add_evaluators()           │
│  • Orchestrates run() or run_batch() execution                            │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
               ┌───────────────────┼───────────────────┐
               │                   │                   │
               ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  EvaluationTest  │ │  EvaluationTest  │ │  EvaluationTest  │
│  Case #1         │ │  Case #2         │ │  Case #N         │
│  • input_text    │ │  • input_text    │ │  • input_text    │
│  • actual_output │ │  • actual_output │ │  • actual_output │
│  • context (opt) │ │  • context (opt) │ │  • context (opt) │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                               ▼
         ┌─────────────────────────────────────────────┐
         │          For each test case, run all         │
         │          registered evaluators:              │
         └─────────────────────┬───────────────────────┘
                               │
       ┌───────────┬───────────┼───────────┬───────────┐
       ▼           ▼           ▼           ▼           ▼
┌───────────┐┌───────────┐┌───────────┐┌───────────┐┌───────────┐
│  Answer   ││ Faithful- ││ Hallucin- ││ Toxicity  ││   Bias    │
│ Relevancy ││   ness    ││  ation    ││ Evaluator ││ Evaluator │
│ Evaluator ││ Evaluator ││ Evaluator ││           ││           │
└─────┬─────┘└─────┬─────┘└─────┬─────┘└─────┬─────┘└─────┬─────┘
      │            │            │            │            │
      │            │            │            │            │
      ▼            ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BaseEvaluator.evaluate()                       │
│  1. Convert EvaluationTestCase → DeepEval LLMTestCase            │
│  2. Invoke DeepEval metric (calls LLM judge via OpenAI API)      │
│  3. Build MetricScore (score, passed, threshold, reason)         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                      EvaluationResult                              │
│  • test_case_name: str                                            │
│  • metric_scores: [MetricScore, MetricScore, ...]                 │
│  • overall_passed: bool (all metrics must pass)                   │
│  • duration_seconds: float                                        │
│  • evaluated_at: datetime (UTC)                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Flow Summary:**

1. **Configuration** — `ConfigLoader` reads `config/settings.yaml`, merges environment overrides, and provides settings (thresholds, model names) to the runner.
2. **Test Case Creation** — User constructs `EvaluationTestCase` objects containing the LLM input, output, and optional context.
3. **Runner Setup** — `EvaluationRunner` is initialized with config and loaded with one or more evaluator instances.
4. **Execution** — For each test case, the runner iterates through all registered evaluators. Each evaluator converts the test case to DeepEval's format and calls the underlying LLM-as-judge metric via the OpenAI API.
5. **Results** — Individual `MetricScore` objects are collected into an `EvaluationResult`, which computes an overall pass/fail status (all metrics must pass their thresholds).

---

## Code Quality

**Linting:**
```bash
ruff check .
```

**Formatting:**
```bash
ruff format .
```

**Type checking:**
```bash
mypy src/
```
