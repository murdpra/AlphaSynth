from unittest.mock import MagicMock

import pytest

from src.financial_analysis.analysis.synthesizer import SynthAgent


@pytest.fixture
def set_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture
def sample_inputs():
    return {
        "query": "Should I invest in AAPL?",
        "research": "Apple has strong cash flows and brand loyalty.",
        "market": "Stock is trending above the 50-day moving average.",
        "news": "Recent earnings beat expectations.",
        "risk": "Moderate valuation risk with macro uncertainty.",
    }


@pytest.fixture
def mock_llm_success():
    mock = MagicMock()
    mock.invoke.return_value.content = """
## Recommendation:
Buy

## Executive Summary
Apple shows strong fundamentals supported by positive earnings and favorable market trends.

## Key Observations
- Strong brand and ecosystem
- Positive price momentum
- Earnings beat expectations
- Manageable macro risks

## Investment Thesis
Apple remains a solid long-term investment given its fundamentals and innovation pipeline.

## Risk Summary
- Valuation risk
- Macro headwinds
- Regulatory pressure

## Suggested Next Steps
- Monitor next earnings
- Track macro indicators
"""
    return mock


@pytest.fixture
def mock_llm_failure():
    mock = MagicMock()
    mock.invoke.side_effect = RuntimeError("LLM crashed")
    return mock


def test_synthesize_success(set_openai_key, sample_inputs, mock_llm_success):
    agent = SynthAgent()
    agent.llm = mock_llm_success

    result = agent.synthesize(**sample_inputs)

    assert "Recommendation" in result  # noqa: S101
    assert "Buy" in result  # noqa: S101
    assert "Executive Summary" in result  # noqa: S101
    assert "Risk Summary" in result  # noqa: S101
    assert isinstance(result, str)  # noqa: S101


def test_synthesize_llm_failure_returns_fallback(set_openai_key, sample_inputs, mock_llm_failure):
    agent = SynthAgent()
    agent.llm = mock_llm_failure

    result = agent.synthesize(**sample_inputs)

    assert "Synthesis LLM failed" in result  # noqa: S101
    assert "Research:" in result  # noqa: S101
    assert "Market:" in result  # noqa: S101
    assert "News:" in result  # noqa: S101
    assert "Risk:" in result  # noqa: S101
    assert "Error:" in result  # noqa: S101


def test_missing_openai_key_raises_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(OSError):  # noqa: PT011
        SynthAgent()


def test_prompt_contains_all_inputs(set_openai_key, sample_inputs):
    agent = SynthAgent()

    captured_prompt = {}

    def fake_invoke(messages):
        captured_prompt["text"] = messages[0].content
        mock_response = MagicMock()
        mock_response.content = "Recommendation:\nHold"
        return mock_response

    agent.llm = MagicMock()
    agent.llm.invoke.side_effect = fake_invoke

    agent.synthesize(**sample_inputs)

    prompt = captured_prompt["text"]
    assert sample_inputs["query"] in prompt  # noqa: S101
    assert sample_inputs["research"] in prompt  # noqa: S101
    assert sample_inputs["market"] in prompt  # noqa: S101
    assert sample_inputs["news"] in prompt  # noqa: S101
    assert sample_inputs["risk"] in prompt  # noqa: S101
