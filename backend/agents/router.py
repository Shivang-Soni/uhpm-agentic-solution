from typing import Dict, Callable, Any

from actions import Action
from schemas import CampaignState


class ActionRouter:
    """
    Routes current_action in state to the responsible agent/service.
    """

    def __init__(self):
        self._routes: Dict[Action, Callable[[CampaignState], str]] = {
            Action.GENERATE_CONTENT: self._route_content,
            Action.GENERATE_PERSONA: self._route_persona,
            Action.RUN_EXPERIMENT: self._route_experiment,
            Action.ANALYSE_PERFORMANCE: self._route_analytics,
            Action.PREVIEW_CAMPAIGN: self._route_preview,
            Action.PUBLISH_CAMPAIGN: self._route_publish,
        }

    def route(self, state: CampaignState) -> str:
        action = state.get("current_action")
        if not action:
            raise ValueError("State missing 'current_action'")

        if action not in self._routes:
            raise ValueError(f"No route exists for action: {action}")

        return self._routes[action](state)

    # Route handlers

    def _route_content(self, state: CampaignState) -> str:
        return "content_agent"

    def _route_persona(self, state: CampaignState) -> str:
        return "persona_agent"

    def _route_experiment(self, state: CampaignState) -> str:
        return "experiment_agent"

    def _route_analytics(self, state: CampaignState) -> str:
        return "analytics_agent"

    def _route_preview(self, state: CampaignState) -> str:
        return "route_preview"

    def _route_publish(self, state: CampaignState) -> str:
        return "publish_agent"
