import os
from typing import Any

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()


class MarketAgent:
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OSError("OPENAI_API_KEY environment variable is not set.")

        self.llm = ChatOpenAI(model=llm_model, api_key=SecretStr(api_key))

    def load_market(self, ticker: str, period: str = "60d") -> pd.DataFrame:
        """
        Fetches historical market data for a single ticker and cleans the columns.
        Uses a 60-day period for more robust 20-day MA calculation.
        """

        df = yf.download(
            ticker, period=period, interval="1d", progress=False, multi_level_index=False
        )

        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()

        df.columns = ["_".join(col.lower().split(" ")) for col in df.columns]
        return df

    def analyze_ticker(self, ticker: str) -> str:
        try:
            df = self.load_market(ticker, period="60d")
        except Exception as e:
            return f"**CRITICAL ERROR FETCHING MARKET DATA FOR {ticker}: {e}**"

        if df.empty:
            return f"No market data found for {ticker} (Check ticker name and connectivity). \
            Data frame was empty."

        price_col = "close"

        if price_col not in df.columns:
            return (
                f"Market data fetched for {ticker} but could not find the required '{price_col}' \
            column after cleaning."
            )

        df["ret_5d"] = df[price_col].pct_change(periods=5).fillna(0)

        df["ma_20"] = df[price_col].rolling(window=20).mean()

        last: pd.Series = df.tail(1).iloc[0]

        indicators: dict[str, float] = {
            "last_price": last.get(price_col, 0.0),
            "ret_5d": last.get("ret_5d", 0.0),
            "ma_20": last.get("ma_20", 0.0) if pd.notna(last.get("ma_20")) else 0.0,
        }

        fundamentals: dict[str, Any] = {"sector": "N/A", "market_cap": "N/A", "forward_pe": "N/A"}
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            fundamentals = {
                "sector": info.get("sector", "N/A"),
                "market_cap": info.get("marketCap", "N/A"),
                "forward_pe": info.get("forwardPE", "N/A"),
            }
        except Exception as e:
            print(f"Warning: Could not fetch fundamentals for {ticker}. Error: {e}")

        mc_str = fundamentals["market_cap"]
        if isinstance(mc_str, (int, float)):
            mc_str = f"${mc_str:,.0f}"

        market_summary = f"""
Market Analysis for {ticker}:
- Current Price: ${indicators["last_price"]:.2f}
- 5-Day Return: {indicators["ret_5d"] * 100:.2f}%
- 20-Day Moving Average: ${indicators["ma_20"]:.2f}
- Sector: {fundamentals["sector"]}
- Market Cap: {mc_str}
- Forward P/E: {fundamentals["forward_pe"]}
"""

        prompt = f"""
Given this market and fundamental summary, act as a financial quant.
Write a concise interpretation (3-4 sentences) and highly actionable trading-research style bullet
points (2-3):
\n\n{market_summary}
"""
        try:
            return str(self.llm.invoke([HumanMessage(content=prompt)]).content)
        except Exception as e:
            return (
                f"LLM analysis failed for market summary. Raw data:\n{market_summary}\nError: {e}"
            )
