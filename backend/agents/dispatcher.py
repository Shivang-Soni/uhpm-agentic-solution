import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Dispatcher:
    def __init__(
        self,
        research_agent=None,
        persona_agent=None,
        content_agent=None,
        experiment_agent=None,
        analytics_agent=None,
        whatsapp_agent=None,
        google_ads_agent=None,
        meta_ads_agent=None,
        email_agent=None,
    ):
        self.research_agent = research_agent
        self.persona_agent = persona_agent
        self.content_agent = content_agent
        self.experiment_agent = experiment_agent
        self.analytics_agent = analytics_agent
        self.whatsapp_agent = whatsapp_agent
        self.google_ads_agent = google_ads_agent
        self.meta_ads_agent = meta_ads_agent
        self.email_agent = email_agent

    def run(
        self,
        state: Dict[str, Any],
        reason_output: Dict[str, Any],
        user_payload: Dict[str, Any],
        plan: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:

        plan = plan or {}

        action = reason_output.get("action")
        inputs_needed = reason_output.get("inputs_needed", {})

        # required vs optional inputs
        if isinstance(inputs_needed, dict):
            required_inputs = [
                k for k, v in inputs_needed.items() if v == "required"
            ]
        else:
            required_inputs = list(inputs_needed)

        missing_inputs = [
            key for key in required_inputs if key not in user_payload
        ]

        if missing_inputs:
            return {
                "status": "waiting_for_inputs",
                "agent": "dispatcher",
                "data": {"missing_inputs": missing_inputs},
                "plan": plan,
            }

        try:
            logger.info(f"[Dispatcher] Executing action: {action}")

            # Research
            if action == "call_research_agent":
                result = self.research_agent.analyse_product(
                    product_text=user_payload.get("product_text"),
                    competitor_text=user_payload.get("competitor_text"),
                )
                return {
                    "status": "research_done",
                    "agent": "research",
                    "data": result,
                    "plan": plan,
                }

            # Persona
            if action == "call_persona_agent":
                result = self.persona_agent.build_persona(
                    product_text=user_payload.get("product_text"),
                    research_insights=user_payload.get("research_insights"),
                )
                return {
                    "status": "persona_created",
                    "agent": "persona",
                    "data": result,
                    "plan": plan,
                }

            # Content
            if action == "call_content_agent":
                result = self.content_agent.generate_content(
                    product_text=user_payload.get("product_text"),
                    persona_text=user_payload.get("persona_text"),
                    channel=user_payload.get("channel"),
                    tone=user_payload.get("tone"),
                )
                return {
                    "status": "content_generated",
                    "agent": "content",
                    "data": result,
                    "plan": plan,
                }

            # WhatsApp
            if action == "call_whatsapp_agent":
                result = self.whatsapp_agent.generate_messages(
                    product_text=user_payload.get("product_text"),
                    persona_text=user_payload.get("persona_text"),
                    intent=user_payload.get("intent"),
                    tone=user_payload.get("tone"),
                )
                return {
                    "status": "whatsapp_messages_generated",
                    "agent": "whatsapp",
                    "data": result,
                    "plan": plan,
                }

            # Google Ads
            if action == "call_google_ads_agent":
                result = self.google_ads_agent.generate_campaign(
                    product_text=user_payload.get("product_text"),
                    persona_text=user_payload.get("persona_text"),
                    campaign_budget=user_payload.get("campaign_budget"),
                    tone=user_payload.get("tone"),
                )
                return {
                    "status": "google_ads_generated",
                    "agent": "google_ads",
                    "data": result,
                    "plan": plan,
                }

            # Meta Ads
            if action == "call_meta_ads_agent":
                result = self.meta_ads_agent.generate_campaign(
                    product_text=user_payload.get("product_text"),
                    persona_text=user_payload.get("persona_text"),
                    campaign_budget=user_payload.get("campaign_budget"),
                    tone=user_payload.get("tone"),
                )
                return {
                    "status": "meta_ads_generated",
                    "agent": "meta_ads",
                    "data": result,
                    "plan": plan,
                }

            # Email
            if action == "call_email_agent":
                result = self.email_agent.generate_campaign(
                    product_text=user_payload.get("product_text"),
                    persona_text=user_payload.get("persona_text"),
                    email_template=user_payload.get("email_template"),
                    tone=user_payload.get("tone"),
                )
                return {
                    "status": "email_campaign_generated",
                    "agent": "email",
                    "data": result,
                    "plan": plan,
                }

            # Unknown
            return {
                "status": "unknown_action",
                "agent": "dispatcher",
                "data": {"action": action},
                "plan": plan,
            }

        except Exception as e:
            logger.exception("Dispatcher caught agent error")
            return {
                "status": "agent_error",
                "agent": "dispatcher",
                "data": {"error": str(e)},
                "plan": plan,
            }
