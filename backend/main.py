from fastapi import FastAPI
from api.routes import router
from agents.bootstrap_runtime import build_registry
from agents.agent_runner import AgentRunner
from agents.dispatcher import Dispatcher
from agents.schemas import CampaignState
from actions import Action

app = FastAPI(title="UHPM Agent API")
app.include_router(router)


@app.get("/")
def root():
    return {"status": "ok", "msg": "UHPM Agent API is running."}


@app.post("/run_campaign")
def run_campaign_endpoint(state: CampaignState):
    registry = build_registry()
    dispatcher = Dispatcher(registry)
    runner = AgentRunner(dispatcher)

    # Wir starten mit Action.PLAN als Startpunkt
    result = dispatcher.run(state, Action.PLAN)
    execution_plan = result.data.get("execution_plan", [])
    runner.run(state, execution_plan)

    return state
