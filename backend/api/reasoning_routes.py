import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.reasoner import ReasonerAgent
from agents.dispatcher import Dispatcher

router = APIRouter()
# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

reasoner = ReasonerAgent()
dispatcher = Dispatcher()


class ReasonRequest(BaseModel):
    task: str
    product_text: str | None = None
    customer_text: str | None = None


@router.post("/reason")
async def reason_endpoint(request: ReasonRequest):
    """
    Endpoint to handle reasoning requests.
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
            "reasoning": reasoning,
            "dispatch": dispatch_result
        }
    except Exception as e:
        logger.exception(f"Reasoning endpoint failed due to {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Reasoning failed: {e}"
            )
