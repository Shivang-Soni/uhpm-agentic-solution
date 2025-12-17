from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseChannelAdapter(ABC):
    """
    Base class for all channel adapters (WhatsApp, Meta, Google, Email, etc.).

    Responsibilities:
    - Validate campaign artifacts
    - Provide preview-ready output
    - Publish artifacts via channel-specific APIs
    """

    channel_name: str

    @abstractmethod
    def validate(self, artifacts: Dict[str, Any]) -> bool:
        """
        Validate that the required artifacts for this channel exist
        and are structurally correct.

        Returns:
            bool: True if valid, False otherwise
        """
        pass

    @abstractmethod
    def preview(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform artifacts into a UI-friendly preview format.
        Must NOT have side effects.
        """
        pass

    @abstractmethod
    def publish(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publish artifacts to the actual channel.
        Can be mocked in MVP stage.
        """
        pass
