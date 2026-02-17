import logging
from typing import List
import uuid

from agents.registry import AgentRegistry
from agents.dispatcher import Dispatcher
from agents.agent_runner import AgentRunner
from agents.actions import Action
from agents.schemas import CampaignState

from vectorstore.store import search, add_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CampaignRunner:
    """
    Orchestrates campaign execution across agents
    and integrates vector memory for contextual learning.
    Automatically inserts EvaluationAgent steps after each action.
    """

    def __init__(self):
        self.registry = AgentRegistry()
        self.dispatcher = Dispatcher(self.registry)
        self.agent_runner = AgentRunner(self.dispatcher)

    def run_campaign(
        self,
        state: CampaignState,
        execution_plan: List[Action]
    ) -> CampaignState:
        logger.info("Starting campaign run.")

        if not state.get("campaign_id"):
            state["campaign_id"] = str(uuid.uuid4())
        # Retrieve vector memory context
        query_text = state.get("brief", "")
        memory_context = search(query_text, k=3) if query_text else {}
        state["memory_context"] = memory_context

        # Build execution plan with optional evaluation steps
        updated_execution_plan: List[Action] = []
        add_eval = state.get("add_evaluation_steps", True)

        for action in execution_plan:
            updated_execution_plan.append(action)
            if add_eval and Action.EVALUATE in self.registry.list_actions():
                updated_execution_plan.append(Action.EVALUATE)

        success = True
        error_message = None

        try:
            updated_state = self.agent_runner.run(
                state,
                updated_execution_plan
            )
        except Exception as e:
            success = False
            error_message = str(e)
            logger.exception("Campaign run failed.")
            raise
        finally:
            # Persist learning into vector memory
            add_document(
                text=query_text or "No campaign brief provided",
                action=execution_plan[0].value if execution_plan else "unknown",
                success=success,
                campaign_id=str(state.get("campaign_id", "unknown")),
                metadata={
                    "last_action": execution_plan[-1].value
                    if execution_plan else None,
                    "error": error_message,
                },
            )

        logger.info("Campaign run finished.")
        return updated_state
