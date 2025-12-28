import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import uvicorn
from fastapi import FastAPI, HTTPException

from financial_analysis.analysis.market import MarketAgent
from financial_analysis.analysis.news import NewsAgent
from financial_analysis.analysis.research import ResearchAgent
from financial_analysis.analysis.risk import RiskAgent
from financial_analysis.analysis.synthesizer import SynthAgent

from .models import QueryIn

app = FastAPI(title="Financial RAG Orchestrator")

research_agent = ResearchAgent()
market_agent = MarketAgent()
news_agent = NewsAgent()
risk_agent = RiskAgent()
synth_agent = SynthAgent()


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
    uvicorn.run("financial_analysis.api.main:app", host="127.0.0.1", port=8000, reload=True)
