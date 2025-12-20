import logging
from typing import Dict, Any

from agents.schemas import ExecutionResult

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
                data = {
                    "channel_result": self.channel_dispatcher.preview(
                        channel=user_payload.get("channel"),
                        artifacts=user_payload.get("artifacts"),
                    )
                }
                return ExecutionResult(
                    action=action,
                    success=True,
                    data=data
                ).model_dump()

            if action == "publish_campaign":
                data = {
                    "channel_result": self.channel_dispatcher.publish(
                        channel=user_payload.get("channel"),
                        artifacts=user_payload.get("artifacts"),
                    )
                }
                return ExecutionResult(
                    action=action,
                    success=True,
                    data=data
                ).model_dump()

            if action == "call_content_agent":
                result = self.content_agent.generate_content(**user_payload)
                return ExecutionResult(
                    action=action,
                    success=True,
                    data=result if isinstance(result, dict) else result.model_dump()
                ).model_dump()

            if action == "call_research_agent":
                result = self.research_agent.analyse_product(**user_payload)
                return ExecutionResult(
                    action=action,
                    success=True,
                    data=result if isinstance(result, dict) else result.model_dump()
                ).model_dump()

            if action == "call_persona_agent":
                result = self.persona_agent.build_persona(**user_payload)
                return ExecutionResult(
                    action=action,
                    success=True,
                    data=result if isinstance(result, dict) else result.model_dump()
                ).model_dump()

            if action == "call_experiment_agent":
                result = self.experiment_agent.evaluate(**user_payload)
                return ExecutionResult(
                    action=action,
                    success=True,
                    data=result if isinstance(result, dict) else result.model_dump()
                ).model_dump()

            if action == "call_analytics_agent":
                result = self.analytics_agent.analyse(**user_payload)
                return ExecutionResult(
                    action=action,
                    success=True,
                    data=result if isinstance(result, dict) else result.model_dump()
                ).model_dump()

            if action == "call_whatsapp_agent":
                result = self.whatsapp_agent.generate(**user_payload)
                return ExecutionResult(
                    action=action,
                    success=True,
                    data=result if isinstance(result, dict) else result.model_dump()
                ).model_dump()

            return ExecutionResult(
                action=action or "unknown",
                success=False,
                error=f"Unknown action: {action}"
            ).model_dump()

        except Exception as e:
            logger.exception("Dispatcher execution failed")
            return ExecutionResult(
                action=action or "unknown",
                success=False,
                error=str(e)
            ).model_dump()
