from typing import Dict, Any

from agents.adapters.base_channel_adapter import BaseChannelAdapter
from agents.schemas import EmailAgentOutput


class EmailAdapter(BaseChannelAdapter):
    """
    Adapter for Email marketing campaigns.
    """

    def validate(self, artifacts: Dict[str, Any]) -> bool:
        required_keys = [
            "subject_line",
            "body",
            "tone",
        ]
        return all(key in artifacts for key in required_keys)

    def preview(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        return EmailAgentOutput(
            subject_line=artifacts.get("subject_line", ""),
            body=artifacts.get("body", ""),
            tone=artifacts.get("tone", "friendly"),
        ).model_dump()

    def publish(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate(artifacts):
            raise ValueError("Invalid email campaign artifact.")

        # Placeholder for real Email Service Provider (ESP) integration
        return {
            "status": "published_mock",
            "channel": "email",
            "artifact": artifacts,
        }
