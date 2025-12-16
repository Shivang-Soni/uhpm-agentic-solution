from typing import Dict, Any

from agents.schemas import WhatsappAgentOutput
from agents.adapters.base_channel_adapter import BaseChannelAdapter


class WhatsappAdapter(BaseChannelAdapter):
    """
    Adapter for Whatsapp messaging campaigns.
    Handles validation, preview and publishing.
    """

    def validate(self, artifacts: Dict[str, Any]) -> bool:
        """
        Validate message required fields.
        """
        required_keys = [
            "initial_message",
            "follow_up_message",
            "closing_message"
            ]
        return all(key in artifacts for key in required_keys)

    def preview(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalise Whatsapp messages for previewing.
        """
        return WhatsappAgentOutput(
            initial_message=artifacts.get("initial_message", ""),
            follow_up_message=artifacts.get("follow_up_message", ""),
            closing_message=artifacts.get("closing_message", ""),
            error=""
        ).model_dump()

    def publish(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the Whatsapp Business API and publish
        """
        if not self.validate(artifacts):
            raise ValueError("Invalid whatsapp campaign artifact.")
        return {
            "status": "published_mock",
            "channel": "whatsapp",
            "artifact": artifacts
        }
