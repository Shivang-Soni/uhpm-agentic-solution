import logging
from typing import List

from agents.actions import Action
from agents.dispatcher import Dispatcher
from agents.schemas import CampaignState

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AgentRunner:
    """
    Executes dispatcher actions sequentially.

    Guarantees:
    - Ordered execution
    - Early abort on failure
    - Explicit execution trace in state
    """

    def __init__(self, dispatcher: Dispatcher):
        self.dispatcher = dispatcher

    def run(
        self,
        state: CampaignState,
        execution_plan: List[Action]
    ) -> CampaignState:
        logger.info("[AgentRunner] Started execution plan")

        state["execution_plan"] = [a.value for a in execution_plan]
        state["executed_actions"] = []
        state["aborted"] = False

        for step_index, action in enumerate(execution_plan):
            logger.info(
                f"[AgentRunner] Step {step_index + 1}/"
                f"{len(execution_plan)}: {action.value}"
            )

            result = self.dispatcher.run(state, action)
            state["executed_actions"].append(action.value)

            if not result.success:
                logger.error(
                    f"[AgentRunner] Execution failed at "
                    f"{action.value}: {result.error}"
                )

                state["aborted"] = True
                state["failed_action"] = action.value
                state["failure_reason"] = result.error
                break

        state["completed"] = not state["aborted"]

        logger.info(
            "[AgentRunner] Finished execution plan "
            f"(completed={state['completed']})"
        )

        return state
