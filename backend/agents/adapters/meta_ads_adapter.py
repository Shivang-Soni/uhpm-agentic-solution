from typing import Dict, Any

from agents.adapters.base_channel_adapter import BaseChannelAdapter
from agents.schemas import MetaAdsAgentOutput


class MetaAdsAdapter(BaseChannelAdapter):
    """
    Adapter for Meta Ads campaigns (Facebook / Instagram).
    """

    def validate(self, artifacts: Dict[str, Any]) -> bool:
        required_keys = [
            "platform",
            "headline",
            "persona",
            "budget",
            "tone",
        ]
        return all(key in artifacts for key in required_keys)

    def preview(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        return MetaAdsAgentOutput(
            platform=artifacts.get("platform", "meta"),
            headline=artifacts.get("headline", ""),
            persona=artifacts.get("persona", ""),
            budget=artifacts.get("budget", ""),
            tone=artifacts.get("tone", "neutral"),
        ).model_dump()

    def publish(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate(artifacts):
            raise ValueError("Invalid Meta Ads campaign artifact.")

        # Placeholder for real Meta Ads API integration
        return {
            "status": "published_mock",
            "channel": "meta_ads",
            "artifact": artifacts,
        }
