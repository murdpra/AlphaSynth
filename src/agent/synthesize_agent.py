import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()


class SynthAgent:
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OSError("OPENAI_API_KEY is not set")
        self.llm = ChatOpenAI(model=llm_model, api_key=SecretStr(api_key))

    def synthesize(self, query: str, research: str, market: str, news: str, risk: str) -> str:
        prompt = f"""
You are a senior investment analyst for a major fund. Your task is to produce a definitive,
actionable analyst note that fully addresses the query: **'{query}'**.

---
**INPUTS:**
## Research Summary (10-K Analysis)
{research}

## Market Summary (Technical/Fundamentals)
{market}

## News Analysis (Sentiment/Headlines)
{news}

## Risk Evaluation (Score/Drivers)
{risk}

---
**OUTPUT INSTRUCTIONS (STRICTLY FOLLOW THIS FORMAT):**

1.  **Recommendation**: Start the response with one of the following exact phrases:
"Recommendation:
Buy", "Recommendation: Hold", or "Recommendation: Sell".
2.  **Executive Summary**:
A maximum 4-sentence summary covering the main conclusion and the primary support factors
(market/news/risk).
3.  **Key Observations**: 4-6 concise bullet points combining insights from all four inputs.
4.  **Investment Thesis**: A brief (2-3 sentence) argument FOR or AGAINST the investment.
5.  **Risk Summary**: A bulleted summary of the top 3-5 drivers of risk.
6.  **Suggested Next Steps**: 2-3 specific data points or research tasks required for a follow-up.

- The final output **MUST** be in clean markdown format with
appropriate headings and bullet points.
- The tone should be professional, data-driven, and concise.
"""
        try:
            return str(self.llm.invoke([HumanMessage(content=prompt)]).content)
        except Exception as e:
            return f"Synthesis LLM failed. Inputs were:\nResearch: {research}\nMarket: \
                {market}\nNews: {news}\nRisk: {risk}\nError: {e}"
