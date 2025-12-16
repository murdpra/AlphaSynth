import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agent.market_agent import MarketAgent
from src.agent.news_agent import NewsAgent
from src.agent.research_agent import ResearchAgent
from src.agent.risk_agent import RiskAgent
from src.agent.synthesize_agent import SynthAgent

app = FastAPI(title="Financial RAG Orchestrator")

research_agent = ResearchAgent()
market_agent = MarketAgent()
news_agent = NewsAgent()
risk_agent = RiskAgent()
synth_agent = SynthAgent()


class QueryIn(BaseModel):
    query: str
    company: str
    k: int = 4


@app.post("/analyze")
def analyze(q: QueryIn):
    try:
        research_out = research_agent.analyze(q.query, k=q.k)
        market_out = market_agent.analyze_ticker(q.company)
        news_out = news_agent.top_headlines_for(q.company)
        risk_out = risk_agent.compute_risk(research_out, market_out, news_out)
        final = synth_agent.synthesize(q.query, research_out, market_out, news_out, risk_out)
        return {"synthesis": final}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
