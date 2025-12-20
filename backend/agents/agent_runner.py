import logging
from typing import Dict, Any, List

from agents.dispatcher import Dispatcher
from agents.execution_context_agent import ExecutionContextAgent
from agents.planner_context_agent import PlannerContextAgent
from agents.planner_agent import PlannerAgent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AgentRunner:
    """
    Orchestrates a full Agentic run:
    Context -> Plan -> Execution -> State update
    """

    def __init__(
            self,
            dispatcher: Dispatcher
    ):
        self.dispatcher = dispatcher
        self.planner = PlannerAgent()
        self.planner_context_agent = PlannerContextAgent()
        self.execution_context_agent = ExecutionContextAgent()

    def run(
            self,
            state: Dict[str, Any],
            execution_plan: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        
        logger.info("Starting execution runner.")

        for step_index, step in enumerate(execution_plan):
            action = step.get("action")
            payload = step.get("payload", {})

            logger.info(f"Executing step: {step_index + 1}: {action}")

            result = self.dispatcher.run(
                state=state,
                reason_output={
                    "action": action
                },
                user_payload=payload
            )

            # Persist result into state
            state.setdefault("execution_results", []).append(
                {
                    "action": action,
                    "result": result
                }
            )

            # Expose last result for downstream agents
            state["last_result"] = result

            # Fail fast strategy
            if "error" in result:
                logger.error(f"Execution terminated due to error in: {action}")
                break

        logger.info("Execution runner finished.")
        return state
