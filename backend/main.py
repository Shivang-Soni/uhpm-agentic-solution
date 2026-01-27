from fastapi import FastAPI
from api.routes import router
from agents.schemas import CampaignState
from agents.run_campaign import CampaignRunner
from actions import Action

app = FastAPI(title="UHPM Agent API")
app.include_router(router)

runner = CampaignRunner()


@app.get("/")
def root():
    return {"status": "ok", "msg": "UHPM Agent API is running."}


@app.post("/run_campaign")
def run_campaign_endpoint(state: CampaignState):
    # Start action = PLAN
    updated_state = runner.run_campaign(
        state=state,
        execution_plan=[Action.PLAN],
    )

    return updated_state
