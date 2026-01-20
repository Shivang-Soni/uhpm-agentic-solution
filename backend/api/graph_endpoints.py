import asyncio
import logging
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from graph.runner import run_graph
from agents.schemas import GraphRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


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
