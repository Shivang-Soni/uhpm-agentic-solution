from agents.base_agent import BaseAgent
from actions import Action
from agents.schemas import CampaignState
from agents.dispatcher import Dispatcher

class PublishAgent(BaseAgent):

    action = Action.PUBLISH_CAMPAIGN

    def __init__(self, channel_dispatcher: Dispatcher):
        self._channel_dispatcher = channel_dispatcher
    
    def execute(self, state: CampaignState):

        try:
            result = self._channel_dispatcher.publish(
                channel=state.get("channel"),
                artifacts=state.get("content") or {}
            )

            return self._success(
                {
                    "preview_result": result
                }
            )

        except Exception as e:
            return self._failure(
                str(e)
            )