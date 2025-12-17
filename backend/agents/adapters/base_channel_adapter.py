from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseChannelAdapter:
    """
    Base class for all marketing channel adapters.
    Responsibilities:
    - Enforce implementation of validate, preview, and publish methods
    - Provide optional flags for capabilities
    """

    # Flags to indicate adapter capabilities
    supports_preview: bool = True
    supports_publish: bool = True

    def validate(self, artifacts: Dict[str, Any]) -> bool:
        """
        Validate the artifact before previewing or publishing.
        Must be implemented by subclass.
        """
        raise NotImplementedError("validate() must be implemented by subclass")

    def preview(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a normalized preview of the campaign artifacts.
        Must be implemented by subclass.
        """
        raise NotImplementedError("preview() must be implemented by subclass")

    def publish(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publish the campaign artifacts through the actual channel.
        Must be implemented by subclass.
        """
        raise NotImplementedError("publish() must be implemented by subclass")
