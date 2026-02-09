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
        