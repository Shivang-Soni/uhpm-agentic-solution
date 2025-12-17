from typing import Dict, Any

from agents.adapters.base_channel_adapter import BaseChannelAdapter
from agents.schemas import WhatsappAgentOutput
from agents.publishers.mock_publisher import MockPublisher


class WhatsappAdapter(BaseChannelAdapter):
    """
    Adapter for WhatsApp messaging campaigns.
    Handles validation, preview formatting, and publishing.
    """

    REQUIRED_FIELDS = {
        "initial_message",
        "follow_up_message",
        "closing_message",
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
            raise ValueError("Invalid WhatsApp artifact for preview")

        preview = WhatsappAgentOutput(
            initial_message=artifacts.get("initial_message"),
            follow_up_message=artifacts.get("follow_up_message"),
            closing_message=artifacts.get("closing_message"),
            intent=artifacts.get("intent"),
            tone=artifacts.get("tone"),
        )

        return preview.model_dump()

    def publish(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate(artifacts):
            raise ValueError("Invalid WhatsApp artifact for publishing")

        # TODO: Replace MockPublisher with WhatsApp Business API
        return self.publisher.publish(
            channel="whatsapp",
            payload=artifacts
        )
