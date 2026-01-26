import json
import logging
from typing import Dict, Any

from agents.base_agent import BaseAgent
from actions import Action
from agents.schemas import CampaignState, ExecutionResult, PlannerOutput
from llm.gemini_pipeline import invoke
from agents.planner_context_agent import PlannerContextAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PlannerAgent(BaseAgent):
    """
    Converts user task + enriched context into an executable plan.
    """

    action = Action.PLAN

    def __init__(self):
        self.context_agent = PlannerContextAgent()

    def execute(self, state: CampaignState) -> ExecutionResult:
        try:
            user_task = state.get("task") or state.get("brief") or ""

            # build external context (social sentiment etc.)
            context = self.context_agent.build_context(user_task)

            prompt = f"""
You are the Planner Agent of an ULTRA HIGH PERFORMANCE MARKETING AI SYSTEM.

Your job is to analyze the user's request and output a JSON plan that tells the system
which agents must be activated.

Agents available:
- research_agent
- persona_agent
- content_agent
- experiment_agent
- analytics_agent

Return ONLY valid JSON in this format:

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
                plan = self._fallback(user_task, "Empty LLM response")
            else:
                try:
                    parsed = json.loads(response)
                    plan = PlannerOutput(**parsed).model_dump()
                except Exception as e:
                    logger.warning(f"Planner JSON invalid: {e}")
                    plan = self._fallback(user_task, "Invalid JSON")

            # Agent returns only — Dispatcher commits to state
            return self._success({"plan": plan})

        except Exception as e:
            logger.exception("PlannerAgent execution failed")
            return self._failure(str(e))

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
