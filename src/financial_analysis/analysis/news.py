import os

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()


class NewsAgent:
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        """
        Initializes the NewsAgent with the cost-efficient gpt-4o-mini model.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OSError("OPENAI_API_KEY environment variable is not set.")
        self.llm = ChatOpenAI(model=llm_model, api_key=SecretStr(api_key))
        self.search_tool = DuckDuckGoSearchRun()

    def fetch_live_news(self, company: str) -> str:
        """
        Invokes the DuckDuckGo Search tool to fetch current financial news snippets.

        Searches for the last 7 days to ensure relevance and adds the 'stock' term.
        """

        query = f"latest {company} stock financial news and headlines past 7 days"

        try:
            search_output = self.search_tool.run(query)

            if not search_output.strip():
                return f"No relevant news snippets found for {company} in the past 7 days."
            return search_output
        except Exception as e:
            return f"Error fetching news with DuckDuckGo: {e}"

    def top_headlines_for(self, company: str) -> str | list[str | dict]:
        """
        Fetches live news and asks the LLM to analyze the sentiment and impact.
        """
        search_data = self.fetch_live_news(company)

        if search_data.startswith("Error fetching news") or search_data.startswith(
            "No relevant news snippets"
        ):
            return search_data

        prompt = f"""
            You are a highly professional financial sentiment analyst.
            Here is the raw, current web search output containing recent news items and
            financial snippets related to {company}:

            Raw Search Data:
            {search_data}

            Task:
            1. Summarize the **top 3 key news themes** (e.g., AI investment, CEO uncertainty,
            regulatory issues).
            2. Explain the **potential market impact** (bullish/bearish/neutral) of these themes
            in brief sentences.
            3. Identify the most pressing **risk or opportunity** mentioned in the news.

            Provide a short, structured analysis in markdown format using headings and
            bullet points.
            The tone must be professional, impartial, and concise. Do not use generic
            market phrases.
        """

        try:
            return str(self.llm.invoke([HumanMessage(content=prompt)]).content)
        except Exception as e:
            return f"LLM analysis failed for news sentiment. Raw data:\n{search_data}\nError: {e}"
