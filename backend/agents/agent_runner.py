import logging
from typing import List

from actions import Action
from agents.dispatcher import Dispatcher
from agents.schemas import CampaignState

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AgentRunner:
    """
    Executes dispatcher actions sequentially.

    Contract:
    - Receives ordered list of Actions
    - Calls Dispatcher.run for each
    - Dispatcher owns state mutation
    - Runner controls flow + failure handling
    """

    def __init__(self, dispatcher: Dispatcher):
        self.dispatcher = dispatcher

    def run(self, state: CampaignState, execution_plan: List[Action]) -> CampaignState:
        logger.info("[AgentRunner] Started execution plan")

        for step_index, action in enumerate(execution_plan):
            logger.info(f"[AgentRunner] Step {step_index + 1}/{len(execution_plan)}: {action.value}")

            result = self.dispatcher.run(state, action)

            if not result.success:
                logger.error(f"[AgentRunner] Execution failed at {action.value}: {result.error}")
                break

        logger.info("[AgentRunner] Finished execution plan")
        return state
