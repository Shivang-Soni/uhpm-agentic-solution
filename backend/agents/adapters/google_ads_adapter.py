from typing import Dict, Any

from agents.adapters.base_channel_adapter import BaseChannelAdapter
from agents.schemas import GoogleAdsAgentOutput


class GoogleAdsAdapter(BaseChannelAdapter):
    """
    Adapter for the Google Ads platform.
    """
    def validate(self, artifact: Dict[str, Any]) -> bool:
        required_keys = [
            "headline",
            "description",
            "keywords",
            "daily_budget_estimate",
            "landing_page_angle"
            ]

        return all(key in artifact for key in required_keys)

    def preview(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        return GoogleAdsAgentOutput(
            headline=artifact.get("headline", ""),
            description=artifact.get("description", ""),
            keywords=artifact.get("keywords", []),
            daily_budget_estimate=artifact.get("daily_budget_estimate", ""),
            landing_page_angle=artifact.get('landing_page_angle', "")
        ).model_dump()

    def publish(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate(artifact):
            raise ValueError("Invalid campaign artifact.")
        return {
            "status": "published_mock",
            "artifact": artifact
        }
