from typing import Dict, Any

from agents.adapters.base_channel_adapter import BaseChannelAdapter
from agents.schemas import MetaAdsAgentOutput
from agents.publishers.mock_publisher import MockPublisher


class MetaAdsAdapter(BaseChannelAdapter):
    """
    Adapter for Meta Ads campaigns (Facebook / Instagram).
    """

    REQUIRED_FIELDS = {
        "platform",
        "headline",
        "persona",
        "budget",
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
            raise ValueError("Invalid Meta Ads artifact for preview")

        preview = MetaAdsAgentOutput(
            platform=artifacts.get("platform", "meta"),
            headline=artifacts.get("headline"),
            persona=artifacts.get("persona"),
            budget=artifacts.get("budget"),
            tone=artifacts.get("tone", "neutral"),
            call_to_action=artifacts.get("call_to_action"),
            creative_angle=artifacts.get("creative_angle"),
        )

        return preview.model_dump()

    def publish(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate(artifacts):
            raise ValueError("Invalid Meta Ads artifact for publishing")

        # TODO: Replace MockPublisher with Meta Marketing API
        return self.publisher.publish(
            channel="meta_ads",
            payload=artifacts
        )
