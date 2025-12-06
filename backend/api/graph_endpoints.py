import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from graph.runner import run_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


class GraphRequest(BaseModel):
    task: str,
    product_text: Optional[str] | None
    competitor_text: Optional[str] | None
    market_text: Optional[str] | None
    customer_text: Optional[str] | None
    persona_text: Optional[str] | None
    channel: Optional[str] | None
    variants: Optional[List[str]] | None
    campaign_results: Optional[str] | None


@router.post("/run-graph")
async def run_graph_endpoint(request: GraphRequest):
    """
    Endpoint which runs the full UHPM Graph asynchronously and return its result.
    """
    payload: Dict[str, Any] = request.dict()
    try:
        # call async runner
        result = await run_graph(payload, timeout=60)
        return {
            "status": "ok",
            "result": result
        }
    except Exception as e:
        logger.exception(f"Graph execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))