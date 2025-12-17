from typing import Dict, Any

from agents.adapters.base_channel_adapter import BaseChannelAdapter
from agents.schemas import GoogleAdsAgentOutput
from agents.publishers.mock_publisher import MockPublisher


class GoogleAdsAdapter(BaseChannelAdapter):
    """
    Adapter for Google Ads campaigns.
    Responsible for validation, preview formatting, and publishing.
    """

    REQUIRED_FIELDS = {
        "headline",
        "description",
        "keywords",
        "daily_budget_estimate",
        "landing_page_angle",
    }

    def __init__(self, publisher=None):
        self.publisher = publisher or MockPublisher()

    def validate(self, artifacts: Dict[str, Any]) -> bool:
        if not isinstance(artifacts, dict):
            return False

        missing = self.REQUIRED_FIELDS - artifacts.keys()
        return len(missing) == 0

    def preview(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate(artifacts):
            raise ValueError("Invalid Google Ads artifact for preview")

        preview = GoogleAdsAgentOutput(
            headline=artifacts.get("headline"),
            description=artifacts.get("description"),
            keywords=artifacts.get("keywords"),
            daily_budget_estimate=artifacts.get("daily_budget_estimate"),
            landing_page_angle=artifacts.get("landing_page_angle"),
        )

        return preview.model_dump()

    def publish(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate(artifacts):
            raise ValueError("Invalid Google Ads artifact for publishing")

        # TODO: Replace MockPublisher with real Google Ads API publisher
        return self.publisher.publish(
            channel="google_ads",
            payload=artifacts
        )
