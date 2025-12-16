from typing import Dict, Any

from agents.adapters.base_channel_adapter import BaseChannelAdapter
from agents.schemas import MetaAdsAgentOutput


class MetaAdsAdapter(BaseChannelAdapter):
    """
    Adapter for Meta Ads campaigns.
    """
    def validate(self, artifact: Dict[str, Any]) -> bool:
        required_keys = ["platform", "headline", "persona", "budget", "tone"]
        return all(key in artifact for key in required_keys)

    def preview(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        return MetaAdsAgentOutput(
            platform=artifact.get("platform", "meta"),
            headline=artifact.get("headline", ""),
            persona=artifact.get("persona", ""),
            budget=artifact.get("budget", ""),
            tone=artifact.get("tone", "neutral")
        ).model_dump()

    def publish(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate(artifact):
            raise ValueError("Invalid Meta Ads campaign artifact.")
        # Actual API call
        return {
            "status": "published_mock",
            "artifact": artifact
        }
