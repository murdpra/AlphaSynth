# AlphaSynth: Financial RAG Orchestrator

AlphaSynth is a sophisticated multi-agent system designed for in-depth financial analysis of publicly traded companies. It orchestrates a suite of specialized AI agents—Research, Market, News, Risk, and Synthesis—to generate comprehensive, human-readable analyst reports. This system leverages Retrieval-Augmented Generation (RAG) for deep document insights, real-time market data, and current news sentiment to provide a holistic view of a company's financial health and prospects.

## Features

*   **RAG-based Research:** Utilizes a FAISS vector store pre-loaded with 10-K filings to provide detailed, document-based insights into a company's fundamental performance and disclosures.
*   **Real-time Market Data:** Fetches current stock prices, key technical indicators, and essential fundamental metrics using the `yfinance` library.
*   **News Sentiment Analysis:** Gathers and analyzes up-to-the-minute financial news from DuckDuckGo, offering qualitative insights into market perception and recent events.
*   **Structured Risk Assessment:** Consolidates and synthesizes information from the Research, Market, and News agents into a quantifiable and structured risk profile.
*   **Comprehensive Analyst Reports:** The Synthesis agent compiles all gathered insights and assessments into a clear, concise, and human-readable analyst report, suitable for informed decision-making.
*   **Modular Architecture:** Built with an easily extensible agent-based design, allowing for the addition of new data sources, analytical methods, or agent types.
*   **FastAPI Backend:** Provides a robust, scalable, and asynchronous API for seamless interaction and integration with other systems.

## Architecture

The system operates on a modular, agent-based architecture coordinated by a FastAPI application.

*   **Core Orchestrator (`src/financial_analysis/api/main.py`):** The main entry point for the application. It defines the `/analyze` endpoint and orchestrates the sequential execution and data flow between the various agents.
*   **Agents:**
    *   **`ResearchAgent`:** Responsible for retrieving and analyzing information from the indexed 10-K filings stored in the FAISS vector store.
    *   **`MarketAgent`:** Collects and interprets real-time stock market data, including price movements and financial metrics.
    *   **`NewsAgent`:** Gathers and analyzes current financial news headlines and their sentiment relevant to the target company.
    *   **`RiskAgent`:** Assesses and quantifies financial and operational risks based on the combined outputs and insights from the Research, Market, and News agents.
    *   **`SynthAgent`:** The final agent in the pipeline, responsible for compiling all preceding analyses and insights into the comprehensive, human-readable analyst report.
*   **Data Store:** A FAISS vector store (`src/rag/vectorstore/`) efficiently stores and enables semantic search over company 10-K filings, powering the RAG capabilities.

## Setup and Installation

To set up AlphaSynth locally, follow these steps:

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   `git` (for cloning the repository)

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/AlphaSynth.git # Replace with actual repo URL
    cd AlphaSynth
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    make install
    ```
    If you plan to contribute, use `make edit-install` for editable mode.

4.  **Prepare the RAG Data (CRITICAL STEP):**
    The `ResearchAgent` relies on a pre-built FAISS vector store of 10-K filings. You must run the `setup_data.py` script to generate this vector store. This process may take some time depending on your internet connection and computational resources.
    ```bash
    python setup-data.py
    ```
    This script will download necessary data and build the `index.faiss` and `index.pkl` files in `src/financial_analysis/rag/vectorstore/`.

## Usage

Once the application is set up and the data is prepared, you can start the FastAPI server and use the `/analyze` endpoint.

### Starting the Application

```bash
make run
```
This will start the server, typically accessible at `http://127.0.0.1:8000`. The `--reload` flag enables auto-reloading upon code changes (useful for development).

### Making an Analysis Request

You can send a POST request to the `/analyze` endpoint with the company's stock ticker and name.

**Endpoint:** `POST /analyze`

**Request Body Example:**

```json
{
"query":"Is it a good time to buy APPLE stock?",
"company":"AAPL",
"k":4
}
```

**Example `curl` command:**

```bash
curl --location 'http://127.0.0.1:8000/analyze' \
--header 'Content-Type: application/json' \
--data '{"query":"Is it a good time to buy APPLE stock?","company":"AAPL","k":4}'
```

### Expected Response

The API will return a JSON object containing the comprehensive analyst report generated by the Synthesis Agent. The structure will vary based on the agents' outputs but will typically include sections for research findings, market analysis, news sentiment, risk assessment, and an overall summary.

## Development and Contribution

AlphaSynth's modular design makes it easy to extend. You can:
*   Add new agents to incorporate different data sources or analytical models.
*   Enhance existing agents with more sophisticated logic.
*   Improve the RAG data preparation process or integrate with different document types.

Contributions are welcome! Please feel free to fork the repository, make your changes, and submit a pull request.
