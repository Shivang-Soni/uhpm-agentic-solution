import json
import logging
from typing import Dict, Any, List

from agents.base_agent import BaseAgent
from actions import Action
from agents.schemas import CampaignState, ExecutionResult, PlannerOutput
from llm.gemini_pipeline import invoke
from agents.planner_context_agent import PlannerContextAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PlannerAgent(BaseAgent):
    """
    Converts user task + enriched context into executable action plan.
    """

    action = Action.PLAN

    def __init__(self):
        self.context_agent = PlannerContextAgent()

    def execute(self, state: CampaignState) -> ExecutionResult:
        try:
            user_task = state.get("task") or state.get("brief") or ""

            context = self.context_agent.build_context(user_task)

            prompt = f"""
You are the Planner Agent of an ULTRA HIGH PERFORMANCE MARKETING AI SYSTEM.

Return ONLY valid JSON:

{{
  "task": "...",
  "needs_research": true/false,
  "needs_persona": true/false,
  "needs_content": true/false,
  "needs_experimentation": true/false,
  "needs_analytics": true/false,
  "additional_context": "optional"
}}

User request:
{user_task}

External context:
{json.dumps(context, indent=2)}
"""

            response = invoke(prompt)

            if not response:
                plan_dict = self._fallback(user_task, "Empty LLM response")
            else:
                try:
                    parsed = json.loads(response)
                    plan_dict = PlannerOutput(**parsed).model_dump()
                except Exception as e:
                    logger.warning(f"Planner JSON invalid: {e}")
                    plan_dict = self._fallback(user_task, "Invalid JSON")

            execution_plan = self._build_execution_plan(plan_dict)

            return self._success(
                {
                    "plan": plan_dict,
                    "execution_plan": execution_plan,
                }
            )

        except Exception as e:
            logger.exception("PlannerAgent execution failed")
            return self._failure(str(e))

    def _build_execution_plan(self, plan: Dict[str, Any]) -> List[Action]:
        actions: List[Action] = []

        if plan.get("needs_research"):
            actions.append(Action.RESEARCH)

        if plan.get("needs_persona"):
            actions.append(Action.GENERATE_PERSONA)

        if plan.get("needs_content"):
            actions.append(Action.GENERATE_CONTENT)

        if plan.get("needs_experimentation"):
            actions.append(Action.RUN_EXPERIMENT)

        if plan.get("needs_analytics"):
            actions.append(Action.ANALYSE_PERFORMANCE)

        # lifecycle always last
        actions.append(Action.PREVIEW_CAMPAIGN)
        actions.append(Action.PUBLISH_CAMPAIGN)

        return actions

    def _fallback(self, task: str, reason: str) -> Dict[str, Any]:
        return PlannerOutput(
            task=task,
            needs_research=True,
            needs_persona=False,
            needs_content=False,
            needs_experimentation=False,
            needs_analytics=False,
            additional_context=f"Fallback used: {reason}",
        ).model_dump()
