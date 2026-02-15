from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

from agents.run_campaign import CampaignRunner
from agents.actions import Action
from agents.schemas import CampaignState
from agents.repositories.in_memory_campaign_repository import (
    InMemoryCampaignRepository
)

repository = InMemoryCampaignRepository()

app = FastAPI(title="UHPM Agent API")

runner = CampaignRunner()


class RunCampaignRequest(BaseModel):
    state: CampaignState
    execution_plan: Optional[List[Action]] = [Action.PLAN]


@app.get("/")
def root():
    return {"status": "ok", "msg": "UHPM Agent API is running."}


@app.get("/campaigns")
def list_campaigns():
    """
    List all the campaigns in the repository
    """
    campaigns = repository.list_all()
    return [
        {
            "campaign_id": c["id"],
            **c
        }
        for c in campaigns
    ]


@app.post("/run-campaign")
def run_campaign_endpoint(request: RunCampaignRequest):
    # Save campaign initial
    campaign_record = repository.create(
        {
            "objective": request.state.brief,
            "status": "running",
            "created_at": datetime.utcnow().isoformat()
        }
    )

    campaign_id = campaign_record["id"]
    request.state.campaign_id = campaign_id

    try:
        # Carry out the camopaign execution
        updated_state = runner.run_campaign(
            state=request.state,
            execution_plan=request.execution_plan
        )
        # Update status
        final_status = "failed" if updated_state.errors else "completed"

        repository.update(
            campaign_id,
            {
                "status": final_status
            }
        )

        # Add IDs to response
        return {
            "campaign_id": campaign_id,
            **updated_state.dict()
        }

    except Exception as e:
        repository.update(
            campaign_id,
            {
                'status': "failed"
            }
        )
        raise e

@app.get("/campaigns/{campaign_id}")
def get_campaign(campaign_id: str):
    try:
        return repository.get(campaign_id)
    except KeyError:
        return {
            "error": "Campaign not found."
        }
