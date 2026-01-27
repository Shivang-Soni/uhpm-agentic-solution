import logging
from typing import List

from agents.bootstrap_runtime import build_registry
from agents.dispatcher import Dispatcher
from agents.agent_runner import AgentRunner
from actions import Action
from agents.schemas import CampaignState


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CampaignRunner:
    """
    Run a campaign using agents.
    """

    def __init__(self):
        self.registry = build_registry()
        self.dispatcher = Dispatcher(self.registry)
        self.agent_runner = AgentRunner(self.dispatcher)

    def run_campaign(self, state: CampaignState, execution_plan: List[Action]) -> CampaignState:
        """
        Run the campaign based on the provided state and execution plan.
        """
        logger.info("Starting campaign run.")
        updated_state = self.agent_runner.run(state, execution_plan)
        logger.info("Campaign run finished.")
        return updated_state
