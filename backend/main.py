from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from agents.run_campaign import CampaignRunner
from actions import Action
from agents.schemas import CampaignState

app = FastAPI(title="UHPM Agent API")

runner = CampaignRunner()


class RunCampaignRequest(BaseModel):
    state: CampaignState
    execution_plan: Optional[List[Action]] = [Action.PLAN]


@app.get("/")
def root():
    return {"status": "ok", "msg": "UHPM Agent API is running."}


@app.post("/run_campaign")
def run_campaign_endpoint(request: RunCampaignRequest):
    """
    Run a campaign using the given state and execution plan.
    Stores objective in vector memory and retrieves context from previous campaigns.
    """
    updated_state = runner.run_campaign(
        state=request.state,
        execution_plan=request.execution_plan
    )

    # JSON-serializable return
    return updated_state
