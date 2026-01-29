import logging
from typing import List

from agents.bootstrap_runtime import build_registry
from agents.dispatcher import Dispatcher
from agents.agent_runner import AgentRunner
from actions import Action
from agents.schemas import CampaignState

# Vector Store importieren
from vectorstore.store import search, add_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CampaignRunner:
    """
    Run a campaign using agents and optionally leverage vector memory for
    past campaigns / audience insights.
    """

    def __init__(self):
        self.registry = build_registry()
        self.dispatcher = Dispatcher(self.registry)
        self.agent_runner = AgentRunner(self.dispatcher)

    def run_campaign(self, state: CampaignState, execution_plan: List[Action]) -> CampaignState:
        """
        Run the campaign based on the provided state and execution plan.
        Uses vector memory to provide context if available.
        """
        logger.info("Starting campaign run.")

        # Optional: Kontext aus vergangene Kampagnen abrufen
        context_results = search(state.get("objective", ""), k=3)
        state["vector_memory"] = context_results

        updated_state = self.agent_runner.run(state, execution_plan)

        add_document(
            text=state.get("objective", "No objective"),
            metadata={"campaign_id": str(state.get("id", "unknown"))}
        )

        logger.info("Campaign run finished.")
        return updated_state
