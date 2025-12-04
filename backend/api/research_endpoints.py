import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.research_agent import ResearchAgent

router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

agent = ResearchAgent()


class ResearchRequest(BaseModel):
    product_text: str
    competitor_text: str | None = None


@router.post("/analyse")
def analyse_product(request: ResearchRequest):
    try:
        logger.info("Received research analysis request.")

        result = agent.analyse_product(
        product_text=request.product_text,
        competitor_text=request.competitor_text
        )
        if not result:
            logger.warning("No analysis result returned by the agent.")
            raise HTTPException(
                status_code=500,
                detail="No analysis result has been returned by the agent."
            )
        return {
            "status": "success",
            "analysis": result
        }
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occured during product analysis: {e}"
        )