from typing import Dict, Any

from agents.adapters.base_channel_adapter import BaseChannelAdapter
from agents.schemas import EmailAgentOutput


class EmailAdapter(BaseChannelAdapter):
    """
    Adapter for Email marketing campaigns.
    Handles validation, preview formatting, and publishing.
    """

    REQUIRED_FIELDS = {"subject_line", "body"}

    def validate(self, artifacts: Dict[str, Any]) -> bool:
        """
        Validate that the email artifact contains all required fields.
        """
        if not isinstance(artifacts, dict):
            return False

        missing = self.REQUIRED_FIELDS - artifacts.keys()
        return len(missing) == 0

    def preview(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a normalized preview of the email campaign.
        """
        if not self.validate(artifacts):
            raise ValueError("Invalid email campaign artifact for preview")

        preview = EmailAgentOutput(
            subject_line=artifacts.get("subject_line"),
            body=artifacts.get("body"),
            tone=artifacts.get("tone", "friendly"),
            cta=artifacts.get("cta"),
            footer=artifacts.get("footer"),
        )

        return preview.model_dump()

    def publish(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publish the email campaign.
        Currently mocked — replace with real ESP integration.
        """
        if not self.validate(artifacts):
            raise ValueError("Invalid email campaign artifact for publishing")

        # TODO: integrate with SendGrid / SES / Mailgun
        return {
            "status": "published",
            "channel": "email",
            "provider": "mock",
            "artifact": artifacts,
        }
