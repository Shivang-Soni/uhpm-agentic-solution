from agents.base_agent import BaseAgent
from actions import Action
from agents.schemas import CampaignState, ExecutionResult
from agents.dispatcher import Dispatcher
from agents.state import apply_execution_result
import logging

logger = logging.getLogger(__name__)


class PublishAgent(BaseAgent):
    action = Action.PUBLISH_CAMPAIGN

    def __init__(self, channel_dispatcher: Dispatcher):
        self._channel_dispatcher = channel_dispatcher

    def execute(self, state: CampaignState) -> ExecutionResult:
        try:
            channel = state.get("channel")
            artifacts = state.get("content", {})

            result = self._channel_dispatcher.publish(
                channel=channel,
                artifacts=artifacts
            )

            apply_execution_result(state, {"published": True, "publish_result": result})

            return self._success({"publish_result": result})

        except Exception as e:
            logger.exception("PublishAgent execution failed")
            return self._failure(str(e))
