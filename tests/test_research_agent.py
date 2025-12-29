from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# Import your agent
from src.financial_analysis.analysis.research import ResearchAgent

# -------------------------------------------------------------------
# FIXTURES
# -------------------------------------------------------------------


@pytest.fixture(autouse=True)
def set_openai_env(monkeypatch):
    """Ensure OPENAI_API_KEY is always set during tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")


@pytest.fixture
def mock_faiss():
    """Mock FAISS vectorstore."""
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = [
        Document(
            page_content="This is a mock 10-K filing content.",
            metadata={"company": "TEST_CORP", "date": "2023"},
        )
    ]
    return mock_vs


# -------------------------------------------------------------------
# INIT TESTS
# -------------------------------------------------------------------


@patch("src.financial_analysis.analysis.research.FAISS.load_local")
@patch("src.financial_analysis.analysis.research.ChatOpenAI")
@patch("src.financial_analysis.analysis.research.OpenAIEmbeddings")
def test_research_agent_initialization_success(
    mock_embeddings, mock_chatopenai, mock_faiss_load, mock_faiss
):
    """Agent initializes correctly when FAISS index exists."""
    mock_faiss_load.return_value = mock_faiss

    agent = ResearchAgent(index_path="vectorstore")

    assert agent.vectorstore is not None  # noqa: S101
    assert agent.llm_analyst is not None  # noqa: S101
    assert agent.llm_summarizer is not None  # noqa: S101
    assert agent.embeddings is not None  # noqa: S101


@patch(
    "src.financial_analysis.analysis.research.FAISS.load_local",
    side_effect=Exception("Missing index"),
)
@patch("src.financial_analysis.analysis.research.ChatOpenAI")
@patch("src.financial_analysis.analysis.research.OpenAIEmbeddings")
def test_research_agent_init_fails_if_vectorstore_missing(
    mock_embeddings, mock_chatopenai, mock_faiss_load
):
    """Initialization should fail if FAISS index cannot be loaded."""
    with pytest.raises(FileNotFoundError):
        ResearchAgent(index_path="vectorstore")


# -------------------------------------------------------------------
# RETRIEVAL TESTS
# -------------------------------------------------------------------


@patch("src.financial_analysis.analysis.research.FAISS.load_local")
@patch("src.financial_analysis.analysis.research.ChatOpenAI")
@patch("src.financial_analysis.analysis.research.OpenAIEmbeddings")
def test_retrieve_documents(mock_embeddings, mock_chatopenai, mock_faiss_load, mock_faiss):
    mock_faiss_load.return_value = mock_faiss
    agent = ResearchAgent()

    results = agent.retrieve_documents("revenue risk", k=1)

    assert len(results) == 1  # noqa: S101
    assert isinstance(results[0], Document)  # noqa: S101
    mock_faiss.similarity_search.assert_called_once_with("revenue risk", k=1)


# -------------------------------------------------------------------
# SUMMARIZATION TESTS
# -------------------------------------------------------------------


@patch("src.financial_analysis.analysis.research.FAISS.load_local")
@patch("src.financial_analysis.analysis.research.ChatOpenAI")
@patch("src.financial_analysis.analysis.research.OpenAIEmbeddings")
def test_summarize_chunk(mock_embeddings, mock_chatopenai, mock_faiss_load, mock_faiss):
    mock_faiss_load.return_value = mock_faiss

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "Mock summary"
    mock_chatopenai.return_value = mock_llm

    agent = ResearchAgent()

    summary = agent.summarize_chunk("Very long 10-K text" * 100)

    assert "Mock summary" in summary  # noqa: S101
    assert mock_llm.invoke.called  # noqa: S101


# -------------------------------------------------------------------
# ANALYSIS TESTS
# -------------------------------------------------------------------


@patch("src.financial_analysis.analysis.research.FAISS.load_local")
@patch("src.financial_analysis.analysis.research.ChatOpenAI")
@patch("src.financial_analysis.analysis.research.OpenAIEmbeddings")
def test_analyze_happy_path(mock_embeddings, mock_chatopenai, mock_faiss_load, mock_faiss):
    mock_faiss_load.return_value = mock_faiss

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "Final financial analysis"
    mock_chatopenai.return_value = mock_llm

    agent = ResearchAgent()

    output = agent.analyze("What are the risks?", k=1)

    assert "Final financial analysis" in output  # noqa: S101
    assert mock_llm.invoke.called  # noqa: S101


@patch("src.financial_analysis.analysis.research.FAISS.load_local")
@patch("src.financial_analysis.analysis.research.ChatOpenAI")
@patch("src.financial_analysis.analysis.research.OpenAIEmbeddings")
def test_analyze_llm_failure_returns_fallback(
    mock_embeddings, mock_chatopenai, mock_faiss_load, mock_faiss
):
    mock_faiss_load.return_value = mock_faiss

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM down")
    mock_chatopenai.return_value = mock_llm

    agent = ResearchAgent()

    output = agent.analyze("What are the risks?", k=1)

    assert "Final analysis failed" in output  # noqa: S101
