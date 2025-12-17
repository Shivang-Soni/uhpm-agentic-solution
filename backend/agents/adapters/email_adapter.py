from typing import Dict, Any
from agents.adapters.base_channel_adapter import BaseChannelAdapter
from agents.schemas import EmailAgentOutput


class EmailAdapter(BaseChannelAdapter):
    """
    Adapter for E-mail marketing campaigns.
    """

    def validate(self, artifacts: Dict[str, Any]) -> bool:
        """
        Validate that the email campaign artifact contains
        all necessary fields.
        """
        required_keys = ["subject_line", "body", "tone"]
        return all(key in artifacts for key in required_keys)

    def preview(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a normalized preview of the email campaign.
        """
        return EmailAgentOutput(
            subject_line=artifacts.get("subject_line", ""),
            body=artifacts.get("body", ""),
            tone=artifacts.get("tone", "friendly")
        ).model_dump()

    def publish(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the email API and publish the campaign.
        """
        if not self.validate(artifacts):
            raise ValueError("Invalid email campaign artifact.")

        # Mock publish for now
        return {
            "status": "published_mock",
            "artifact": artifacts
        }
