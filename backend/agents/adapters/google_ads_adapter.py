from typing import Dict, Any
from agents.adapters.base_channel_adapter import BaseChannelAdapter
from agents.schemas import GoogleAdsAgentOutput
from agents.publishers.mock_publisher import MockPublisher


class GoogleAdsAdapter(BaseChannelAdapter):
    """
    Adapter for the Google Ads platform.
    """

    def __init__(self):
        self.publisher = MockPublisher()

    def validate(self, artifacts: Dict[str, Any]) -> bool:
        required_keys = [
            "headline",
            "description",
            "keywords",
            "daily_budget_estimate",
            "landing_page_angle"
        ]
        return all(key in artifacts for key in required_keys)

    def preview(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        return GoogleAdsAgentOutput(
            headline=artifacts.get("headline", ""),
            description=artifacts.get("description", ""),
            keywords=artifacts.get("keywords", []),
            daily_budget_estimate=artifacts.get("daily_budget_estimate", ""),
            landing_page_angle=artifacts.get("landing_page_angle", "")
        ).model_dump()

    def publish(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate(artifacts):
            raise ValueError("Invalid campaign artifact.")
        return self.publisher.publish(artifacts)
