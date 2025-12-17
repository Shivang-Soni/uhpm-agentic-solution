import logging
from typing import Dict, Any

from agents.adapters.channel_adapter_registry import ChannelAdapterRegistry

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
        campaign_agent=None,
        channel_adapter_registry: ChannelAdapterRegistry | None = None
    ):
        self.research_agent = research_agent
        self.persona_agent = persona_agent
        self.content_agent = content_agent
        self.experiment_agent = experiment_agent
        self.analytics_agent = analytics_agent
        self.campaign_agent = campaign_agent
        self.channel_adapter_registry = (
            channel_adapter_registry or ChannelAdapterRegistry()
        )

    def run(
        self,
        reason_output: Dict[str, Any],
        user_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deterministic dispatcher.
        Routes execution strictly based on reason_output["action"].
        """

        action = reason_output.get("action")
        inputs_needed = reason_output.get("inputs_needed", [])

        logger.info(f"[Dispatcher] Received action: {action}")

        # Normalize required inputs
        if isinstance(inputs_needed, dict):
            required_inputs = [
                k for k, v in inputs_needed.items() if v == "required"
            ]
        else:
            required_inputs = list(inputs_needed)

        missing_inputs = [
            key for key in required_inputs
            if key not in user_payload or user_payload.get(key) is None
        ]

        if missing_inputs:
            return {
                "status": "waiting_for_inputs",
                "agent": "dispatcher",
                "data": {"missing_inputs": missing_inputs}
            }

        try:
            # Research
            if action == "call_research_agent":
                result = self.research_agent.analyse_product(
                    product_text=user_payload.get("product_text"),
                    competitor_text=user_payload.get("competitor_text"),
                )
                return {
                    "status": "research_done",
                    "agent": "research",
                    "data": result
                }

            # Persona
            elif action == "call_persona_agent":
                result = self.persona_agent.build_persona(
                    product_text=user_payload.get("product_text"),
                    research_insights=user_payload.get("research_insights"),
                )
                return {
                    "status": "persona_created",
                    "agent": "persona",
                    "data": result
                }

            # Content
            elif action == "call_content_agent":
                result = self.content_agent.generate_content(
                    product_text=user_payload.get("product_text"),
                    persona_text=user_payload.get("persona_text"),
                    channel=user_payload.get("channel"),
                    tone=user_payload.get("tone"),
                )
                return {
                    "status": "content_generated",
                    "agent": "content",
                    "data": result
                }

            # Campaign
            elif action == "call_campaign_agent":
                result = self.campaign_agent.generate(
                    product_text=user_payload.get("product_text"),
                    persona_text=user_payload.get("persona_text"),
                    channel=user_payload.get("channel"),
                    budget=user_payload.get("budget"),
                    tone=user_payload.get("tone"),
                    intent=user_payload.get("intent"),
                )
                return {
                    "status": "campaign_generated",
                    "agent": "campaign",
                    "channel": user_payload.get("channel"),
                    "data": result
                }

            # Preview
            elif action == "preview_campaign":
                channel = user_payload.get("channel")
                artifacts = user_payload.get("artifacts")

                adapter = self.channel_adapter_registry.get(channel)
                preview = adapter.preview(artifacts)

                return {
                    "status": "campaign_preview_ready",
                    "agent": "dispatcher",
                    "channel": channel,
                    "data": preview
                }

            # Publish
            elif action == "publish_campaign":
                channel = user_payload.get("channel")
                artifacts = user_payload.get("artifacts")

                adapter = self.channel_adapter_registry.get(channel)
                result = adapter.publish(artifacts)

                return {
                    "status": "campaign_published",
                    "agent": "dispatcher",
                    "channel": channel,
                    "data": result
                }

            # Unknown
            else:
                return {
                    "status": "unknown_action",
                    "agent": "dispatcher",
                    "data": {"action": action}
                }

        except Exception as e:
            logger.exception("[Dispatcher] Agent execution failed")
            return {
                "status": "agent_error",
                "agent": "dispatcher",
                "data": {"error": str(e)}
            }
