import logging
from typing import Dict, Any

from actions import Action
from agents.base_agent import BaseAgent
from agents.schemas import CampaignState, ExecutionResult

logger = logging.getLogger(__name__)


class EvaluationAgent(BaseAgent):
    """
    Evaluate the outcome of the previous action and provide feedback.
    """

    action = Action.EVALUATE

    def execute(
            self,
            state: CampaignState,
            reflection = None
    ) -> ExecutionResult:
        """
        Return only evaulation metadata
        """
        last_history = state.get("history", [])
        if not last_history:
            return self._success(data={})
        last_step = last_history[-1]

        evaluation: Dict[str, Any] = {
            "action": last_step.get("action"),
            "success": last_step.get("success"),
            "confidence": 0.8 if last_step.get("success") else 0.3,
            "notes": "auto-eval placeholder"
        }

        return ExecutionResult(
            action=self.action.value,
            success=True,
            evaluation=evaluation
        )
