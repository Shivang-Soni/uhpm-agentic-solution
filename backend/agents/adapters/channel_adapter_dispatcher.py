import logging
from typing import Dict, Any

from agents.adapters.channel_adapter_registry import ChannelAdapterRegistry

logger = logging.getLogger(__name__)


class ChannelAdapterDispatcher:
    """
    Central dispatcher for all channel adapters.

    Responsibilities:
    - Resolve adapter by channel
    - Enforce adapter capabilities
    - Execute preview or publish safely
    - Normalize execution responses
    """

    def __init__(self, registry: ChannelAdapterRegistry):
        self.registry = registry

    def preview(
        self,
        channel: str,
        artifacts: Dict[str, Any],
    ) -> Dict[str, Any]:
        adapter = self.registry.get(channel)

        logger.info(
            f"[ChannelAdapterDispatcher] Preview request | channel={channel}"
        )

        try:
            result = adapter.safe_preview(artifacts)

            return {
                "status": "preview_ready",
                "channel": channel,
                "data": result,
            }

        except Exception as e:
            logger.exception(
                f"[ChannelAdapterDispatcher] Preview failed | channel={channel}"
            )
            return {
                "status": "preview_failed",
                "channel": channel,
                "error": str(e),
            }

    def publish(
        self,
        channel: str,
        artifacts: Dict[str, Any],
    ) -> Dict[str, Any]:
        adapter = self.registry.get(channel)

        logger.info(
            f"[ChannelAdapterDispatcher] Publish request | channel={channel}"
        )

        try:
            result = adapter.safe_publish(artifacts)

            return {
                "status": "published",
                "channel": channel,
                "data": result,
            }

        except Exception as e:
            logger.exception(
                f"[ChannelAdapterDispatcher] Publish failed | channel={channel}"
            )
            return {
                "status": "publish_failed",
                "channel": channel,
                "error": str(e),
            }
