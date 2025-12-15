import logging
import traceback
from typing import Dict, Any

from agents.schemas import DispatcherOutput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Dispatcher:
    """
    Routes actions decided by Reasoner to the correct agent
    and normalizes outputs into a consistent structure.
    """

    ACTION_STATUS_MAP = {
        "call_research_agent": ("research_done", "research"),
        "call_whatsapp_agent": ("whatsapp_messages_generated", "whatsapp"),
    }

    def __init__(
        self,
        research_agent,
        persona_agent,
        content_agent,
        experiment_agent,
        analytics_agent,
        whatsapp_agent
    ):
        self.research_agent = research_agent
        self.persona_agent = persona_agent
        self.content_agent = content_agent
        self.experiment_agent = experiment_agent
        self.analytics_agent = analytics_agent
        self.whatsapp_agent = whatsapp_agent

    def run(
        self,
        plan: Dict[str, Any],
        reason_output: Dict[str, Any],
        user_payload: Dict[str, Any]
    ):
        action = reason_output.get("action")
        inputs_needed = reason_output.get("inputs_needed", [])

        # Normalize inputs_needed
        if isinstance(inputs_needed, dict):
            required_inputs = list(inputs_needed.keys())
        elif isinstance(inputs_needed, (list, set, tuple)):
            required_inputs = list(inputs_needed)
        else:
            required_inputs = []

        missing_inputs = [x for x in required_inputs if x not in user_payload]

        if missing_inputs:
            return self._build_output(
                status="waiting_for_inputs",
                agent="dispatcher",
                data={
                    "missing_inputs": missing_inputs,
                    "required": required_inputs
                },
                plan=plan
            )

        try:
            # Research
            if action == "call_research_agent":
                result = self.research_agent.analyse_product(
                    product_text=user_payload.get("product_text", ""),
                    competitor_text=user_payload.get("competitor_text")
                )
                status, agent = self.ACTION_STATUS_MAP[action]

                return self._build_output(
                    status=status,
                    agent=agent,
                    data=result,
                    plan=plan
                )

            # WhatsApp
            if action == "call_whatsapp_agent":
                result = self.whatsapp_agent.generate_messages(
                    product_text=user_payload.get("product_text", ""),
                    persona_text=user_payload.get("persona_text", ""),
                    intent=user_payload.get("intent", "lead"),
                    tone=user_payload.get("tone", "friendly")
                )
                status, agent = self.ACTION_STATUS_MAP[action]

                return self._build_output(
                    status=status,
                    agent=agent,
                    data=result,
                    plan=plan
                )

            # ---- Unknown ----
            return self._build_output(
                status="unknown_action",
                agent="dispatcher",
                data={"action": action},
                plan=plan
            )

        except Exception as e:
            logger.error(f"[Dispatcher] Agent crashed: {e}")
            return self._build_output(
                status="agent_error",
                agent="dispatcher",
                data={
                    "error": str(e),
                    "trace": traceback.format_exc()
                },
                plan=plan
            )

    def _build_output(
        self,
        status: str,
        agent: str,
        data: Dict[str, Any],
        plan: Dict[str, Any]
    ):
        return DispatcherOutput(
            status=status,
            agent=agent,
            data=data,
            plan=plan
        ).model_dump()
