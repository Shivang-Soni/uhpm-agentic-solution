import logging
import json
from llm.gemini_pipeline import invoke

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlannerAgent:
    """
    LLM-driven planner that converts the user's raw request
    into a structured plan for the multi-agent pipeline.
    """

    def __init__(self):
        pass

    def plan(self, user_task: str) -> dict:
        """
        Uses the LLM to classify the task and decide which agents are needed.
        """

        prompt = f"""
        You are the Planner Agent of an Ultra High Performance Marketing AI System.
        Your job is to analyze the user's request and output a JSON plan that
        tells the system which agents must be activated.

        Agents available:
        - research_agent (product analysis, competitor analysis)
        - persona_agent (target audience creation)
        - content_agent (ads, posts, scripts, long-form content)
        - experiment_agent (A/B tests, variations, test ideas)
        - analytics_agent (ROI analysis, performance insights)

        The output MUST be valid JSON with this structure:
        {{
            "task": "...",
            "needs_research": true/false,
            "needs_persona": true/false,
            "needs_content": true/false,
            "needs_experimentation": true/false,
            "needs_analytics": true/false,
            "additional_context": "Optional description or extracted info"
        }}

        User request:
        {user_task}
        """

        response = invoke(prompt)

        if not response:
            logger.error("PlannerAgent: LLM returned no response. Using fallback plan.")
            return {
                "task": user_task,
                "needs_research": True,
                "needs_persona": False,
                "needs_content": False,
                "needs_experimentation": False,
                "needs_analytics": False,
                "additional_context": "Fallback: No LLM response"
            }

        try:
            plan = json.loads(response)
        except Exception as e:
            logger.error(f"PlannerAgent failed to JSON parse: {e}")
            logger.error(f"Raw LLM response: {response}")

            # Fallback if JSON fails
            return {
                "task": user_task,
                "needs_research": True,
                "needs_persona": False,
                "needs_content": False,
                "needs_experimentation": False,
                "needs_analytics": False,
                "additional_context": "Invalid LLM JSON, fallback used."
            }

        logger.info(f"PLANNER PLAN GENERATED: {plan}")
        return plan
