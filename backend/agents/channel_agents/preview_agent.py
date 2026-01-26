from agents.base_agent import BaseAgent
from actions import Action
from agents.schemas import CampaignState


class PreviewAgent(BaseAgent):

    action = Action.PREVIEW_CAMPAIGN

    def __init__(self, channel_dispatcher):
        self._channel_dispatcher = channel_dispatcher

    def execute(self, state: CampaignState):

        try:
            result = self._channel_dispatcher.preview(
                channel=state.get("channel"),
                artifacts=state.get("content") or {},
            )

            return self._success({
                "preview_result": result
            })

        except Exception as e:
            return self._failure(str(e))