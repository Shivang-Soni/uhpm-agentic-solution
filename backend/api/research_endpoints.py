from fastapi import APIRouter
from pydantic import BaseModel
from agents.research_agent import ResearchAgent

router = APIRouter()
agent = ResearchAgent()


class ResearchRequest(BaseModel):
    product_text: str
    competitor_text: str | None = None


@router.post("/analyse")
def analyse_product(request: ResearchRequest):
    result = agent.analyse_product(
        product_text=request.product_text,
        competitor_text=request.competitor_text
    )
    return {"analysis": result}