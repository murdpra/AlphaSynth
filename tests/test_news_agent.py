import os
from unittest.mock import patch

import pytest

from src.financial_analysis.analysis.news import NewsAgent


@pytest.fixture
def agent():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "testkey"}, clear=True):
        return NewsAgent(llm_model="gpt-4o-mini")


@patch("langchain_community.tools.DuckDuckGoSearchRun.run")
def test_fetch_live_news_success(mock_run, agent):
    mock_run.return_value = "Apple launches new AI chip impacting stock prices."

    result = agent.fetch_live_news("Apple")

    assert "Apple launches new AI chip" in result  # noqa: S101


@patch("langchain_community.tools.DuckDuckGoSearchRun.run")
def test_fetch_live_news_empty(mock_run, agent):
    mock_run.return_value = "   "

    result = agent.fetch_live_news("Apple")

    assert "No relevant news snippets found for Apple" in result  # noqa: S101


@patch("langchain_community.tools.DuckDuckGoSearchRun.run")
def test_fetch_live_news_exception(mock_run, agent):
    mock_run.side_effect = Exception("Search API down")

    result = agent.fetch_live_news("Apple")

    assert "Error fetching news with DuckDuckGo" in result  # noqa: S101
    assert "Search API down" in result  # noqa: S101


@patch.object(NewsAgent, "fetch_live_news")
def test_top_headlines_for_fetch_error(mock_fetch, agent):
    mock_fetch.return_value = "Error fetching news with DuckDuckGo: timeout"

    result = agent.top_headlines_for("Apple")

    assert result.startswith("Error fetching news")  # noqa: S101


@patch.object(NewsAgent, "fetch_live_news")
def test_top_headlines_for_no_news(mock_fetch, agent):
    mock_fetch.return_value = "No relevant news snippets found for Apple in the past 7 days."

    result = agent.top_headlines_for("Apple")

    assert result.startswith("No relevant news snippets")  # noqa: S101


@patch.object(NewsAgent, "fetch_live_news")
@patch("langchain_openai.ChatOpenAI.invoke")
def test_top_headlines_for_success(mock_invoke, mock_fetch, agent):
    mock_fetch.return_value = "Apple faces regulatory scrutiny in EU markets."

    mock_invoke.return_value.content = (
        "### Key Themes\n- Regulatory pressure\n- Market uncertainty"
    )

    result = agent.top_headlines_for("Apple")

    assert "Key Themes" in result  # noqa: S101
    assert "Regulatory" in result  # noqa: S101


@patch.object(NewsAgent, "fetch_live_news")
@patch("langchain_openai.ChatOpenAI.invoke")
def test_top_headlines_for_llm_failure(mock_invoke, mock_fetch, agent):
    mock_fetch.return_value = "Apple reports mixed quarterly earnings."

    mock_invoke.side_effect = Exception("LLM timeout")

    result = agent.top_headlines_for("Apple")

    assert "LLM analysis failed for news sentiment" in result  # noqa: S101
    assert "LLM timeout" in result  # noqa: S101
