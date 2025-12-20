import logging
from typing import Dict, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ExecutionContextAgent:
    """
    Translates planner decisions and context
      into executable dispatcher actions.
    """
    def build_execution_plan(
            self,
            plan: Dict[str, object],
            social_context: Dict[str, object],
            user_payload: Dict[str, object]
    ) -> List[Dict[str, object]]:

        """
        Returns an ordered list of dispatcher actions.
        """

        logger.info("Building execution plan")

        execution_steps: List[Dict[str, object]] = []

        # Research
        if plan.get("needs_research"):
            execution_steps.append(
                {
                    "action": "call_research_agent",
                    "payload": user_payload
                }
            )

        # Persona
        if plan.get("needs_persona"):
            execution_steps.append(
                {
                    "action": "call_persona_agent",
                    "payload": {
                        **user_payload,
                        "market_context": social_context
                    }
                }
            )
        
        # Content Generation
        if plan.get("needs_content"):
            execution_steps.append(
                {
                    "action": "call_content_agent",
                    "payload": {
                        **user_payload,
                        "strategy_hint": social_context.get("social_verdict")
                    }
                }
            )
        
        # Experimentation
        if plan.get("needs_experimentation"):
            execution_steps.append(
                {
                    "action": "call_experiment_agent",
                    "payload": user_payload
                }
            )

        if plan.get("needs_analytics"):
            execution_steps.append(
                {
                    "action": "call_analytics_agent",
                    "payload": user_payload
                }
            )

        return execution_steps
