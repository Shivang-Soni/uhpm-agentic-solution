import logging
from typing import Dict, Any, List

from actions import Action
from agents.dispatcher import Dispatcher

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AgentRunner:
    """
    Execute dispatcher actions sequentially.
    """

    def __init__(self, dispatcher: Dispatcher):
        self.dispatcher = dispatcher

    def run(
        self,
        state: Dict[str, Any],
        execution_plan: List[Action],
    ) -> Dict[str, Any]:

        logger.info("AgentRunner started")

        state.setdefault("execution_results", [])

        for step_index, action in enumerate(execution_plan):
            logger.info(f"Running step {step_index + 1}: {action.value}")

            result = self.dispatcher.run(
                state=state,
                action=action,
            )

            # Persist raw ExecutionResult
            state["execution_results"].append(result)
            state["last_result"] = result

            # Stop on failure
            if not result.success:
                logger.error(
                    f"Execution failed at step {action.value}: {result.error}"
                )
                break

        logger.info("AgentRunner finished")
        return state
