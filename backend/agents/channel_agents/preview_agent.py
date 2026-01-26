from agents.base_agent import BaseAgent
from actions import Action
from agents.schemas import CampaignState, ExecutionResult
from agents.dispatcher import Dispatcher
import logging

logger = logging.getLogger(__name__)


class PreviewAgent(BaseAgent):
    action = Action.PREVIEW_CAMPAIGN

    def __init__(self, channel_dispatcher: Dispatcher):
        self._channel_dispatcher = channel_dispatcher

    def execute(self, state: CampaignState) -> ExecutionResult:
        try:
            channel = state.get("channel")
            artifacts = state.get("content", {})

            result = self._channel_dispatcher.preview(
                channel=channel,
                artifacts=artifacts
            )

            # ONLY return result — Dispatcher commits
            return self._success({"preview_result": result})

        except Exception as e:
            logger.exception("PreviewAgent execution failed")
            return self._failure(str(e))