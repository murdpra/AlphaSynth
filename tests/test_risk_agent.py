import json
from unittest.mock import MagicMock

import pytest
from src.agent.risk_agent import RiskAgent


@pytest.fixture
def valid_inputs():
    return {
        "research_summary": "Company has moderate debt and stable revenue.",
        "market_summary": "Price is above 20-day moving average.",
        "news_summary": "No major negative news reported.",
    }


@pytest.fixture
def mock_llm_success():
    mock = MagicMock()
    mock.invoke.return_value.content = json.dumps(
        {
            "risk_score": 35,
            "risk_drivers": ["Moderate leverage", "Market volatility", "Sector competition"],
            "confidence_level": "High",
            "quantitative_flag": "Price_Above_MA_Bullish",
        }
    )
    return mock


@pytest.fixture
def mock_llm_with_noise():
    mock = MagicMock()
    mock.invoke.return_value.content = """
    Sure! Here's your risk assessment:

    {
        "risk_score": 60,
        "risk_drivers": [
            "Earnings uncertainty",
            "Macroeconomic risk",
            "Interest rate sensitivity"
        ],
        "confidence_level": "Medium",
        "quantitative_flag": "Neutral"
    }

    Let me know if you need more details.
    """
    return mock


@pytest.fixture
def mock_llm_invalid_json():
    mock = MagicMock()
    mock.invoke.return_value.content = "This is not JSON at all."
    return mock


@pytest.fixture
def set_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

def test_compute_risk_success(set_openai_key, valid_inputs, mock_llm_success):
    agent = RiskAgent()
    agent.llm = mock_llm_success

    result = agent.compute_risk(**valid_inputs)

    assert result["risk_score"] == 35  # noqa: S101
    assert result["confidence_level"] == "High"  # noqa: S101
    assert result["quantitative_flag"] == "Price_Above_MA_Bullish"  # noqa: S101
    assert len(result["risk_drivers"]) >= 1  # noqa: S101


def test_compute_risk_with_extra_text(set_openai_key, valid_inputs, mock_llm_with_noise):
    agent = RiskAgent()
    agent.llm = mock_llm_with_noise

    result = agent.compute_risk(**valid_inputs)

    assert result["risk_score"] == 60  # noqa: S101
    assert result["confidence_level"] == "Medium"  # noqa: S101
    assert result["quantitative_flag"] == "Neutral"  # noqa: S101


def test_compute_risk_invalid_json_returns_fallback(
    set_openai_key, valid_inputs, mock_llm_invalid_json
):
    agent = RiskAgent()
    agent.llm = mock_llm_invalid_json

    result = agent.compute_risk(**valid_inputs)

    assert result["risk_score"] == 75  # noqa: S101
    assert result["confidence_level"] == "Low"  # noqa: S101
    assert result["quantitative_flag"] == "Error"  # noqa: S101
    assert "error" in result  # noqa: S101


def test_schema_validation_failure_returns_fallback(set_openai_key, valid_inputs):
    agent = RiskAgent()

    agent.llm = MagicMock()
    agent.llm.invoke.return_value.content = json.dumps(
        {
            "risk_score": 150,  # invalid > 100
            "risk_drivers": [],
            "confidence_level": "Extreme",  # invalid enum
            "quantitative_flag": "Neutral",
        }
    )

    result = agent.compute_risk(**valid_inputs)

    assert result["confidence_level"] == "Low"  # noqa: S101
    assert result["quantitative_flag"] == "Error"  # noqa: S101


def test_missing_openai_key_raises_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(OSError):  # noqa: PT011
        RiskAgent()
