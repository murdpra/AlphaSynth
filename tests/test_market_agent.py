import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.financial_analysis.analysis.market import MarketAgent


@pytest.fixture
def agent():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "testkey"}, clear=True):
        return MarketAgent(llm_model="gpt-4o-mini")


@patch("yfinance.download")
def test_load_market_success(mock_download, agent):
    mock_download.return_value = pd.DataFrame(
        {
            "Close": [100, 101, 102],
            "Open": [99, 100, 101],
        }
    )

    df = agent.load_market("AAPL")

    assert isinstance(df, pd.DataFrame)  # noqa: S101
    assert "close" in df.columns  # noqa: S101
    assert "open" in df.columns  # noqa: S101


@patch("yfinance.download")
def test_load_market_empty(mock_download, agent):
    mock_download.return_value = pd.DataFrame()

    df = agent.load_market("AAPL")
    assert df.empty  # noqa: S101


@patch("yfinance.download")
def test_analyze_ticker_empty_data(mock_download, agent):
    mock_download.return_value = pd.DataFrame()

    result = agent.analyze_ticker("AAPL")

    assert "No market data found" in result  # noqa: S101


@patch("yfinance.download")
def test_analyze_ticker_missing_close(mock_download, agent):
    mock_download.return_value = pd.DataFrame({"Open": [10, 11, 12]})

    result = agent.analyze_ticker("AAPL")

    assert "Market data fetched for AAPL but could not find the required 'close'" in result  # noqa: S101


@patch("yfinance.download")
@patch("yfinance.Ticker")
@patch("langchain_openai.ChatOpenAI.invoke")
def test_analyze_ticker_success(mock_invoke, mock_ticker, mock_download, agent):
    # Mock market data
    mock_download.return_value = pd.DataFrame(
        {
            "Close": [100, 102, 104, 103, 105, 110],
        }
    )

    ticker_instance = MagicMock()
    ticker_instance.info = {"sector": "Technology", "marketCap": 250000000000, "forwardPE": 28.7}
    mock_ticker.return_value = ticker_instance

    mock_invoke.return_value.content = "LLM RESULT"

    result = agent.analyze_ticker("AAPL")

    assert result == "LLM RESULT"  # noqa: S101


@patch("yfinance.download")
@patch("yfinance.Ticker")
@patch("langchain_openai.ChatOpenAI.invoke")
def test_analyze_ticker_llm_failure(mock_invoke, mock_ticker, mock_download, agent):
    mock_download.return_value = pd.DataFrame({"Close": [100, 101, 102, 103, 104, 105]})

    mock_ticker.return_value.info = {
        "sector": "Tech",
        "marketCap": 1000000,
        "forwardPE": 15,
    }

    mock_invoke.side_effect = Exception("LLM crashed")

    result = agent.analyze_ticker("AAPL")

    assert "LLM analysis failed" in result  # noqa: S101
    assert "LLM crashed" in result  # noqa: S101


@patch("yfinance.download", side_effect=Exception("Network down"))
def test_analyze_ticker_fetch_exception(mock_download, agent):
    result = agent.analyze_ticker("AAPL")

    assert "CRITICAL ERROR FETCHING MARKET DATA FOR AAPL" in result  # noqa: S101
    assert "Network down" in result  # noqa: S101
