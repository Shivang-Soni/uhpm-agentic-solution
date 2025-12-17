import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class Dispatcher:
    def __init__(
        self,
        research_agent,
        persona_agent,
        content_agent,
        experiment_agent,
        analytics_agent,
        whatsapp_agent,
        channel_adapter_dispatcher
    ):
        self.research_agent = research_agent
        self.persona_agent = persona_agent
        self.content_agent = content_agent
        self.experiment_agent = experiment_agent
        self.analytics_agent = analytics_agent
        self.whatsapp_agent = whatsapp_agent
        self.channel_dispatcher = channel_adapter_dispatcher

    def run(
        self,
        state: Dict[str, Any],
        reason_output: Dict[str, Any],
        user_payload: Dict[str, Any],
        plan: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:

        action = reason_output.get("action")

        try:
            if action == "preview_campaign":
                return {
                    "channel_result": self.channel_dispatcher.preview(
                        channel=user_payload.get("channel"),
                        artifacts=user_payload.get("artifacts"),
                    )
                }

            if action == "publish_campaign":
                return {
                    "channel_result": self.channel_dispatcher.publish(
                        channel=user_payload.get("channel"),
                        artifacts=user_payload.get("artifacts"),
                    )
                }

            if action == "call_content_agent":
                return self.content_agent.generate_content(**user_payload)

            if action == "call_research_agent":
                return self.research_agent.analyse_product(**user_payload)

            if action == "call_persona_agent":
                return self.persona_agent.build_persona(**user_payload)

            if action == "call_experiment_agent":
                return self.experiment_agent.evaluate(**user_payload)

            if action == "call_whatsapp_agent":
                return self.whatsapp_agent.generate(**user_payload)

            return {"error": f"Unknown action: {action}"}

        except Exception as e:
            logger.exception("Dispatcher error")
            return {"error": str(e)}
