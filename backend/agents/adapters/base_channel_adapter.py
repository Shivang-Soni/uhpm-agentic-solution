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

    # Core interface
    def validate(self, artifacts: Dict[str, Any]) -> bool:
        raise NotImplementedError("validate() must be implemented by subclass")

    def preview(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("preview() must be implemented by subclass")

    def publish(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("publish() must be implemented by subclass")

    # Safe execution wrappers
    def safe_preview(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        if not self.supports_preview:
            raise RuntimeError(
                f"{self.__class__.__name__} does not support preview"
            )

        logger.info(
            f"[{self.__class__.__name__}] Preview started"
        )

        try:
            if not self.validate(artifacts):
                raise ValueError("Artifact validation failed")

            result = self.preview(artifacts)

            logger.info(
                f"[{self.__class__.__name__}] Preview completed successfully"
            )

            return result

        except Exception:
            logger.exception(
                f"[{self.__class__.__name__}] Preview execution failed"
            )
            raise

    def safe_publish(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        if not self.supports_publish:
            raise RuntimeError(
                f"{self.__class__.__name__} does not support publish"
            )

        logger.info(
            f"[{self.__class__.__name__}] Publish started"
        )

        try:
            if not self.validate(artifacts):
                raise ValueError("Artifact validation failed")

            result = self.publish(artifacts)

            logger.info(
                f"[{self.__class__.__name__}] Publish completed successfully"
            )

            return result

        except Exception:
            logger.exception(
                f"[{self.__class__.__name__}] Publish execution failed"
            )
            raise
