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

    def run(
        self,
        state: CampaignState,
        execution_plan: List[Action],
    ) -> CampaignState:

        logger.info("AgentRunner started")

        state.setdefault("execution_results", [])

        for step_index, action in enumerate(execution_plan):
            logger.info(f"Running step {step_index + 1}: {action.value}")

            result = self.dispatcher.run(
                state=state,
                action=action,
            )

            dumped = result.model_dump()

            state["execution_results"].append(dumped)
            state["last_result"] = dumped

            if not result.success:
                logger.error(
                    f"Execution failed at step {action.value}: {result.error}"
                )
                break

        logger.info("AgentRunner finished")
        return state
