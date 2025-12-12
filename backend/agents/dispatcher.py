import logging
import traceback
from typing import Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Dispatcher:
    """
    Minimal-stable Dispatcher for the UHPM MVP system.

    Responsibilities:
    - Validate required inputs
    - Route tasks to the correct agent
    - Return predictable structured outputs
    - Fail gracefully without breaking the graph
    """

    def __init__(
        self,
        research_agent,
        persona_agent,
        content_agent,
        experiment_agent,
        analytics_agent
    ):
        self.research_agent = research_agent
        self.persona_agent = persona_agent
        self.content_agent = content_agent
        self.experiment_agent = experiment_agent
        self.analytics_agent = analytics_agent

    def run(self, plan: Dict[str, Any], reason_output: Dict[str, Any], user_payload: Dict[str, Any]):
        """
        Executes the action selected by Reasoner.
        """

        task_type = reason_output.get("task_type")
        action = reason_output.get("action")
        inputs_needed = reason_output.get("inputs_needed", {})

        logger.info(f"[Dispatcher] Task: {task_type}, Action: {action}")

        # Normalize to list of required inputs
        if isinstance(inputs_needed, dict):
            required_inputs = list(inputs_needed.keys())
        elif isinstance(inputs_needed, (list, set, tuple)):
            required_inputs = list(inputs_needed)
        else:
            required_inputs = []

        # Validate required inputs
        missing_inputs = [x for x in required_inputs if x not in user_payload]

        if missing_inputs:
            return {
                "status": "waiting_for_inputs",
                "missing_inputs": missing_inputs,
                "required": inputs_needed,
                "plan": plan
            }

        # Execute agent
        try:
            # RESEARCH
            if action == "call_research_agent":
                result = self.research_agent.analyse_product(
                    product_text=user_payload.get("product_text", ""),
                    competitor_text=user_payload.get("competitor_text", None)
                )
                return {"status": "research_done", "result": result, "plan": plan}

            # PERSONA
            elif action == "call_persona_agent":
                result = self.persona_agent.generate_persona(
                    product_text=user_payload.get("product_text", ""),
                    market_text=user_payload.get("market_text", None)
                )
                return {"status": "persona_done", "result": result, "plan": plan}

            # CONTENT
            elif action == "call_content_agent":
                result = self.content_agent.generate_content(
                    product_text=user_payload.get("product_text", ""),
                    persona_text=user_payload.get("persona_text", ""),
                    channel=user_payload.get("channel", "social_media")
                )
                return {"status": "content_done", "result": result, "plan": plan}

            # EXPERIMENT
            elif action == "call_experiment_agent":
                result = self.experiment_agent.run_experiment(
                    persona_text=user_payload.get("persona_text", ""),
                    channel=user_payload.get("channel", ""),
                    variants=user_payload.get("variants", [])
                )
                return {"status": "experiment_done", "result": result, "plan": plan}

            # ANALYTICS
            elif action == "call_analytics_agent":
                result = self.analytics_agent.analyse_campaign(
                    campaign_results=user_payload.get("campaign_results", "")
                )
                return {"status": "analytics_done", "result": result, "plan": plan}

            else:
                return {
                    "status": "unknown_action",
                    "action": action,
                    "reason_output": reason_output,
                    "plan": plan
                }

        except Exception as e:
            logger.error(f"[Dispatcher] Agent crashed: {e}")
            return {
                "status": "agent_error",
                "error": str(e),
                "trace": traceback.format_exc(),
                "plan": plan
            }
