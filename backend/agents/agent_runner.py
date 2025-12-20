import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class AgentRunner:
    """
    Executes dispatcher actions sequentially
    and persists ExecutionResults into state.
    """

    def __init__(self, dispatcher):
        self.dispatcher = dispatcher

    def run(
        self,
        state: Dict[str, Any],
        execution_plan: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        logger.info("AgentRunner started")

        state.setdefault("execution_results", [])

        for step_index, step in enumerate(execution_plan):
            action = step.get("action")
            payload = step.get("payload", {})

            logger.info(f"Running step {step_index + 1}: {action}")

            result = self.dispatcher.run(
                state=state,
                reason_output={"action": action},
                user_payload=payload,
                plan=state.get("plan"),
            )

            # Persist
            state["execution_results"].append(result)
            state["last_result"] = result

            # Failure detection
            if not result.get("success", False):
                logger.error(
                    f"Execution failed at step {action}: {result.get('error')}"
                )
                break

        logger.info("AgentRunner finished")
        return state
