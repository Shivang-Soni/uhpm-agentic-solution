import logging
from typing import Dict, Any

from actions import Action
from agents.registry import AgentRegistry
from agents.schemas import CampaignState, ExecutionResult
from agents.state import apply_execution_result

logger = logging.getLogger(__name__)


class Dispatcher:
    """
    Execute exactly one action on CampaignState via the AgentRegistry.
    Automatically applies execution results to CampaignState.
    """

    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def run(
            self,
            state: CampaignState,
            action: Action
    ) -> ExecutionResult:

        try:
            agent = self.registry.get(action)

            logger.info(
                f"Dispatcher executing action = {action.value} "
                f"with agent = {agent.__class__.__name__}"
            )

            result = agent.execute(state)

            # Lifecycle bookkeeping
            state.setdefault("history", []).append(action)
            state["current_action"] = action

            if result.success:
                # Controlled mutation of state
                if result.data:
                    apply_execution_result(state, result.data)
            else:
                state.setdefault("errors", []).append(result.error)

            return result

        except Exception as e:
            logger.exception("Dispatcher execution failed.")

            state.setdefault("errors", []).append(str(e))

            return ExecutionResult(
                action=action.value,
                success=False,
                error=str(e)
            )
