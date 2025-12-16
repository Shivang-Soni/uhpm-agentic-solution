import logging
import traceback
from typing import Dict, Any, Callable

from agents.schemas import DispatcherOutput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Dispatcher:
    """
    Routes actions decided by the Reasoner to the correct agent
    and normalizes outputs into a consistent structure.
    """

    ACTION_STATUS_MAP = {
        "call_research_agent": ("research_done", "research"),
        "call_whatsapp_agent": ("whatsapp_messages_generated", "whatsapp"),
        "call_google_ads_agent": ("google_ads_campaign_generated", "google_ads"),
        "call_meta_ads_agent": ("meta_ads_campaign_generated", "meta_ads"),
        "call_email_agent": ("email_campaign_generated", "email"),
    }

    def __init__(
        self,
        research_agent,
        persona_agent,
        content_agent,
        experiment_agent,
        analytics_agent,
        whatsapp_agent,
        googleads_agent,
        metaads_agent,
        email_agent
    ):
        self.research_agent = research_agent
        self.persona_agent = persona_agent
        self.content_agent = content_agent
        self.experiment_agent = experiment_agent
        self.analytics_agent = analytics_agent
        self.whatsapp_agent = whatsapp_agent
        self.googleads_agent = googleads_agent
        self.metaads_agent = metaads_agent
        self.email_agent = email_agent

        self._action_handlers: Dict[
            str, Callable[[Dict[str, Any]], Dict[str, Any]]
            ] = {
            "call_research_agent": self._handle_research,
            "call_whatsapp_agent": self._handle_whatsapp,
            "call_google_ads_agent": self._handle_google_ads,
            "call_meta_ads_agent": self._handle_meta_ads,
            "call_email_agent": self._handle_email
        }

    def run(
        self,
        plan: Dict[str, Any],
        reason_output: Dict[str, Any],
        user_payload: Dict[str, Any]
    ) -> Dict[str, Any]:

        action = reason_output.get("action")
        inputs_needed = reason_output.get("inputs_needed", {})

        # Normalize required inputs
        required_inputs = (
            list(inputs_needed.keys())
            if isinstance(inputs_needed, dict)
            else list(inputs_needed)
            if isinstance(inputs_needed, (list, set, tuple))
            else []
        )

        missing_inputs = [x for x in required_inputs if x not in user_payload]

        if missing_inputs:
            return self._build_output(
                status="waiting_for_inputs",
                agent="dispatcher",
                data={
                    "missing_inputs": missing_inputs,
                    "required": inputs_needed
                },
                plan=plan
            )

        if action not in self.ACTION_STATUS_MAP:
            return self._build_output(
                status="unknown_action",
                agent="dispatcher",
                data={"action": action},
                plan=plan
            )

        try:
            handler = self._action_handlers.get(action)
            if not handler:
                raise ValueError(f"No handler registered for action: {action}")

            result = handler(user_payload)

            status, agent = self.ACTION_STATUS_MAP[action]

            return self._build_output(
                status=status,
                agent=agent,
                data=result,
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

    # Action handlers
    def _handle_research(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.research_agent.analyse_product(
            product_text=payload.get("product_text", ""),
            competitor_text=payload.get("competitor_text")
        )

    def _handle_whatsapp(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.whatsapp_agent.generate_messages(
            product_text=payload.get("product_text", ""),
            persona_text=payload.get("persona_text", ""),
            intent=payload.get("intent", "lead"),
            tone=payload.get("tone", "friendly")
        )

    def _handle_google_ads(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.googleads_agent.generate_campaign(
            product_text=payload.get("product_text", ""),
            persona_text=payload.get("persona_text", ""),
            campaign_budget=payload.get("campaign_budget", ""),
            tone=payload.get("tone", "neutral")
        )

    def _handle_meta_ads(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.metaads_agent.generate_campaign(
            product_text=payload.get("product_text", ""),
            persona_text=payload.get("persona_text", ""),
            campaign_budget=payload.get("campaign_budget", ""),
            tone=payload.get("tone", "neutral")
        )

    def _handle_email(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.email_agent.generate_campaign(
            product_text=payload.get("product_text", ""),
            persona_text=payload.get("persona_text", ""),
            email_template=payload.get("email_template", ""),
            tone=payload.get("tone", "friendly")
        )

    # Output Normalization
    def _build_output(
        self,
        status: str,
        agent: str,
        data: Dict[str, Any],
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        return DispatcherOutput(
            status=status,
            agent=agent,
            data=data,
            plan=plan
        ).model_dump()
