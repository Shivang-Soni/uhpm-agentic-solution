import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.reasoner import ReasonerAgent
from agents.dispatcher import Dispatcher
# agent imports
from agents.research_agent import ResearchAgent
from agents.analytics_agent import AnalyticsAgent
from agents.content_agent import ContentAgent
from agents.experiment_agent import ExperimentationAgent
from agents.persona_agent import PersonaAgent

router = APIRouter()
# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

reasoner = ReasonerAgent()

research_agent = ResearchAgent()
analytics_agent = AnalyticsAgent()
content_agent = ContentAgent()
experiment_agent = ExperimentationAgent()
persona_agent = PersonaAgent()

dispatcher = Dispatcher(
    research_agent,
    persona_agent,
    content_agent,
    experiment_agent,
    analytics_agent
    )


class ReasonRequest(BaseModel):
    task: str
    product_text: str | None = None
    customer_text: str | None = None
    market_text: str | None = None
    competitor_text: str | None = None
    persona_text: str | None = None
    channel: str | None = None
    variants: list | None = None
    campaign_results: str | None = None


@router.post("/reason")
async def reason_endpoint(request: ReasonRequest):
    """
    Endpoint to handle reasoning requests.
    Decides what needs to happen and executes the correct agent.
    """
    try:
        logger.info(f"Reasoning endpoint called with task: {request.task}")
        # Run core reasoning layer
        reasoning = reasoner.decide(request.task)
        # Dispatch to appropriate agent
        dispatch_result = dispatcher.run(
            reason_output=reasoning,
            user_payload=request.dict()
            )

        return {
            "status": "ok",
            "reasoning": reasoning,
            "dispatch": dispatch_result
        }
    except Exception as e:
        logger.exception(f"Reasoning endpoint failed due to {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Reasoning failed: {e}"
            )
