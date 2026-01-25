from typing import Dict, Any

from agents.base_agent import BaseAgent
from actions import Action
from agents.schemas import CampaignState
from agents.dispatcher import Dispatcher


class PreviewAgent(BaseAgent):

    action = Action.PREVIEW_CAMPAIGN

    def __init__(self, channel_dispatcher: Dispatcher):
        self._channel_dispatcher = channel_dispatcher

    def execute(self, state: CampaignState):

        try:
            result = self.channel_dispatcher.preview(
                channel=state.get("channel"),
                artifacts=state.get("content")
            )

            return self._success(
                {
                    "preview_result": result
                }
            )

        except Exception as e:
            return self._failure(str(e))
