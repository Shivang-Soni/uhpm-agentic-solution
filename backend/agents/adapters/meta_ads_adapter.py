from typing import Dict, Any
import httpx

from backend.core.config import Settings
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

        url = f"https://graph.facebook.com/v17.0/{Settings.META_ADS_ACCOUNT_ID}/ads"
        headers = {
            "Authorization": f"Bearer {Settings.META_ADS_TOKEN}"
            }
        data = {
            "name": artifacts["headline"],
            "creative": {
                "title": artifacts["headline"],
                "body": artifacts["persona"]
            },
            "status": "PAUSED",
            "daily_budget": artifacts["budget"]
        }

        try:
            response = httpx.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            return {
                "status": "published",
                "response": response.json()
            }
        except httpx.HTTPStatusError as e:
            return {
                "status": "failed",
                "error": str(e),
                "response": e.response.text
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
