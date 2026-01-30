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
    Execute exactly one Action against CampaignState.
    Guarantee Single commit boundary, Idempotent action execution,
    Structured history, Deterministic state mutation,
    Retry with self-reflection
    """

    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def run(self, state: CampaignState, action: Action) -> ExecutionResult:
        MAX_RETRIES = 2

        # Idempotency
        for entry in state.get("history", []):
            if entry["action"] == action.value and entry["success"]:
                logger.info(f"Skipping already completed action: {action.value}")
                return ExecutionResult(
                    action=action.value,
                    success=True,
                    data=entry.get("data", {})
                )

        agent = self.registry.get(action)

        # Retry loop
        for attempt in range(MAX_RETRIES + 1):
            timestamp = time.time()

            try:
                logger.info(
                    f"Dispatcher executing action={action.value} "
                    f"agent={agent.__class__.__name__} attempt={attempt}"
                )

                result = agent.execute(state)

                history_entry: Dict[str, Any] = {
                    "action": action.value,
                    "attempt": attempt,
                    "success": result.success,
                    "timestamp": timestamp,
                }

                # Success
                if result.success:
                    if result.data:
                        apply_execution_result(state, result.data)
                        history_entry["data"] = result.data

                    state.setdefault("history", []).append(history_entry)
                    state["current_action"] = action.value
                    return result

                # Failure
                history_entry["error"] = result.error
                state.setdefault("history", []).append(history_entry)

                state.setdefault("errors", []).append({
                    "action": action.value,
                    "attempt": attempt,
                    "error": result.error,
                    "timestamp": timestamp,
                })

                # Self reflection
                state.setdefault("self_reflection", []).append({
                    "action": action.value,
                    "attempt": attempt,
                    "error": result.error,
                    "timestamp": timestamp,
                })

                logger.warning(
                    f"Action failed: {action.value} attempt={attempt} → retrying"
                )

                time.sleep(0.5)

            except Exception as e:
                logger.exception("Dispatcher execution crashed.")

                state.setdefault("errors", []).append({
                    "action": action.value,
                    "attempt": attempt,
                    "error": str(e),
                    "timestamp": timestamp,
                })

                state.setdefault("history", []).append({
                    "action": action.value,
                    "attempt": attempt,
                    "success": False,
                    "timestamp": timestamp,
                    "error": str(e),
                })

        # Total Failure after retries
        return ExecutionResult(
            action=action.value,
            success=False,
            error="Max retries exceeded",
        )
