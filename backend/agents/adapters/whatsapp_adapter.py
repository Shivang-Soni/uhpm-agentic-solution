from typing import Dict, Any

from agents.adapters.base_channel_adapter import BaseChannelAdapter
from agents.schemas import WhatsappAgentOutput
from agents.adapters.whatsapp_client import send_whatsapp_message


class WhatsappAdapter(BaseChannelAdapter):
    """
    Adapter for WhatsApp messaging campaigns.
    Handles validation, preview formatting, and publishing.
    """

    REQUIRED_FIELDS = {
        "to",
        "initial_message",
        "follow_up_message",
        "closing_message",
    }   

    def validate(self, artifacts: Dict[str, Any]) -> bool:
        if not isinstance(artifacts, dict):
            return False
        return self.REQUIRED_FIELDS.issubset(artifacts.keys())

    def preview(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate(artifacts):
            raise ValueError("Invalid WhatsApp artifact for preview")

        preview = WhatsappAgentOutput(
            initial_message=artifacts.get("initial_message"),
            follow_up_message=artifacts.get("follow_up_message"),
            closing_message=artifacts.get("closing_message"),
            intent=artifacts.get("intent"),
            tone=artifacts.get("tone"),
            error=artifacts.get("error", "")
        )

        return preview.model_dump()

    def publish(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate(artifacts):
            raise ValueError("Invalid WhatsApp artifact for publishing")
        
        to = artifacts.get("to")

        try:
            responses = []
            responses.append(
                send_whatsapp_message(to, artifacts.get()"initial_message")
            )
            responses.append(
                send_whatsapp_message(to, artifacts.get("follow_up_message"))
            )
            responses.append(
                send_whatsapp_message(to, artifacts.get("closing_message"))
            )

            return {
                "status": "sent",
                "channel": "whatsapp",
                "responses": responses
            }
        
        except Exception as e:
            return {
                "status": "failed",
                "channel": "whatsapp",
                "error": str(e)
            }