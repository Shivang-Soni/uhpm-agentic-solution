from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.run_campaign import CampaignRunner
from agents.actions import Action
from agents.schemas import CampaignState
from agents.repositories.in_memory_campaign_repository import (
    InMemoryCampaignRepository
)

repository = InMemoryCampaignRepository()

app = FastAPI(title="UHPM Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

runner = CampaignRunner()


class RunCampaignRequest(BaseModel):
    state: CampaignState
    execution_plan: Optional[List[Action]] = None


@app.get("/")
def root():
    return {"status": "ok", "msg": "UHPM Agent API is running."}


@app.get("/campaigns")
def list_campaigns():
    campaigns = repository.list_all()
    return campaigns


@app.post("/run-campaign")
def run_campaign_endpoint(request: RunCampaignRequest):

    # Default plan if none given
    execution_plan = request.execution_plan or [Action.PLAN]

    # Use parsed state directly from request
    state = request.state

    # Save campaign initial
    campaign_record = repository.create(
        {
            "objective": state["brief"],
            "status": "running",
            "created_at": datetime.utcnow().isoformat()
        }
    )

    campaign_id = campaign_record["campaign_id"]
    state["campaign_id"] = campaign_id

    try:
        updated_state = runner.run_campaign(
            state=state,
            execution_plan=execution_plan
        )

        final_status = "failed" if updated_state.errors else "completed"

        repository.update(
            campaign_id,
            {
                "status": final_status
            }
        )

        return {
            "campaign_id": campaign_id,
            **updated_state.dict()
        }

    except Exception as e:
        repository.update(
            campaign_id,
            {"status": "failed"}
        )
        raise e


@app.get("/campaigns/{campaign_id}")
def get_campaign(campaign_id: str):
    try:
        return repository.get(campaign_id)
    except KeyError:
        return {"error": "Campaign not found."}
