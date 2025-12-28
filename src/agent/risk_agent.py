import json
import os
import re
from typing import Annotated, Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

load_dotenv()


class RiskAssessment(BaseModel):
    risk_score: Annotated[
        int,
        Field(
            ge=0,
            le=100,
            description="Overall risk score from 0 (low risk) to 100 (high risk).",
        ),
    ]

    risk_drivers: Annotated[
        list[str],
        Field(
            min_items=1,
            max_items=5,
            description="The top 5 most critical risk factors identified.",
        ),
    ]

    confidence_level: Annotated[
        str,
        Field(
            pattern=r"^(High|Medium|Low)$",
            description="Confidence level for the assessment.",
        ),
    ]

    quantitative_flag: Annotated[
        str,
        Field(
            description="A simple, data-driven flag (e.g., 'MA_Crossover_Bearish', "
            "'Price_Above_MA_Bullish', 'Neutral', 'Error').",
        ),
    ]


class RiskAgent:
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OSError("OPENAI_API_KEY environment variable is not set.")

        self.llm = ChatOpenAI(model=llm_model, api_key=SecretStr(api_key))

    def compute_risk(
        self, research_summary: str, market_summary: str, news_summary: str
    ) -> dict[str, Any]:
        """
        Computes a structured risk assessment based on combined inputs.
        Returns a dictionary based on the RiskAssessment Pydantic model.
        """
        schema_json = json.dumps(RiskAssessment.model_json_schema(), indent=2)
        prompt = f"""
            You are an Investment Risk Evaluator AI. Your task is to synthesize the provided Research, Market,
            and News summaries to produce a structured, machine-readable risk assessment.

            Analyze the three inputs for potential downside factors, volatility, and uncertainty.

            Inputs:
            ---
            ## Research Analysis (10-K RAG)
            {research_summary}

            ## Market Analysis (Price/Fundamentals/Indicators)
            {market_summary}

            ## News Analysis (Sentiment/Headlines)
            {news_summary}
            ---

            Task:
            1.  **Risk Score**: Assign an integer risk score from 0 (very low risk) to 100 (extreme risk).
            2.  **Risk Drivers**: List the top 5 most critical, distinct risk factors.
            3.  **Confidence Level**: State your confidence in the assessment: 'High', 'Medium', or 'Low'.
            4.  **Quantitative Flag**: Extract a simple flag from the Market Analysis.
                Look for price vs. 20-Day Moving Average: 'Price_Below_MA_Bearish', 'Price_Above_MA_Bullish',
                or 'Neutral'.

            Format your entire response STRICTLY as a single JSON object matching the following schema:
            {schema_json}
            """
        try:
            response_text = str(self.llm.invoke([HumanMessage(content=prompt)]).content)

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                risk_data = json.loads(json_string)

                validated_risk = RiskAssessment(**risk_data)
                return validated_risk.model_dump()
            raise ValueError(
                f"LLM response did not contain a valid JSON object. Raw response: {response_text}"
            )

        except Exception as e:
            error_message = f"Error in Risk Agent LLM or parsing: {e}"
            print(f"CRITICAL ERROR: {error_message}")

            return {
                "risk_score": 75,
                "risk_drivers": [
                    "Risk calculation failed due to error.",
                    "Manual review required.",
                    "High LLM error risk.",
                    "Data integrity concern.",
                    "Returning default risk.",
                ],
                "confidence_level": "Low",
                "quantitative_flag": "Error",
                "error": error_message,
            }
