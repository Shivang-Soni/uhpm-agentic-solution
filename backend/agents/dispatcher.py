import logging
import time
from typing import Dict, Any

from actions import Action
from agents.registry import AgentRegistry
from agents.schemas import CampaignState, ExecutionResult
from agents.state import apply_execution_result

logger = logging.getLogger(__name__)


class Dispatcher:
    """
    Executes exactly one Action against CampaignState.

    Guarantees:
    - Single commit boundary
    - Idempotent action execution
    - Structured history
    - Deterministic state mutation
    """

    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def run(self, state: CampaignState, action: Action) -> ExecutionResult:
        timestamp = time.time()

        # Idempotency: skip already successful actions
        for entry in state.get("history", []):
            if entry["action"] == action.value and entry["success"]:
                logger.info(f"Skipping already completed action: {action.value}")
                return ExecutionResult(
                    action=action.value,
                    success=True,
                    data=entry.get("data", {})
                )

        try:
            agent = self.registry.get(action)

            logger.info(
                f"Dispatcher executing action={action.value} agent={agent.__class__.__name__}"
            )

            result = agent.execute(state)

            history_entry: Dict[str, Any] = {
                "action": action.value,
                "success": result.success,
                "timestamp": timestamp,
                "data_keys": list(result.data.keys()) if result.data else [],
            }

            if result.success:
                if result.data:
                    apply_execution_result(state, result.data)

                history_entry["data"] = result.data
            else:
                history_entry["error"] = result.error

                state.setdefault("errors", []).append({
                    "action": action.value,
                    "timestamp": timestamp,
                    "error": result.error
                })

            # Single commit boundary
            state.setdefault("history", []).append(history_entry)
            state["current_action"] = action.value

            return result

        except Exception as e:
            logger.exception("Dispatcher execution failed.")

            error_entry = {
                "action": action.value,
                "timestamp": timestamp,
                "error": str(e)
            }

            state.setdefault("errors", []).append(error_entry)
            state.setdefault("history", []).append({
                "action": action.value,
                "success": False,
                "timestamp": timestamp,
                "error": str(e)
            })

            return ExecutionResult(
                action=action.value,
                success=False,
                error=str(e)
            )
