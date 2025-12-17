import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class BaseChannelAdapter:
    """
    Base class for all marketing channel adapters.

    Responsibilities:
    - Enforce validate / preview / publish interface
    - Provide safe execution wrappers
    - Declare adapter capabilities
    """

    supports_preview: bool = True
    supports_publish: bool = True

    # ---- Core interface ----

    def validate(self, artifacts: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def preview(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def publish(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    # ---- Safe execution wrappers ----

    def safe_preview(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        if not self.supports_preview:
            raise RuntimeError("Preview not supported by this adapter")

        if not self.validate(artifacts):
            raise ValueError("Artifact validation failed")

        logger.info(
            f"[{self.__class__.__name__}] Preview executed successfully"
        )
        return self.preview(artifacts)

    def safe_publish(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        if not self.supports_publish:
            raise RuntimeError("Publish not supported by this adapter")

        if not self.validate(artifacts):
            raise ValueError("Artifact validation failed")

        logger.info(
            f"[{self.__class__.__name__}] Publish executed successfully"
        )
        return self.publish(artifacts)
