import logging
from typing import Dict, Any

from agents.adapters.channel_adapter_registry import ChannelAdapterRegistry

logger = logging.getLogger(__name__)


class ChannelAdapterDispatcher:
    """
    Routes artifacts to the correct channel adapter
    and enforces validation + safe execution.
    """

    def __init__(self, registry: ChannelAdapterRegistry):
        self.registry = registry

    def preview(
        self,
        channel: str,
        artifacts: Dict[str, Any],
    ) -> Dict[str, Any]:
        adapter = self.registry.get(channel)

        if not adapter.validate(artifacts):
            logger.error(
                f"[ChannelAdapterDispatcher] Validation failed | "
                f"channel={channel} artifacts={artifacts}"
            )
            raise ValueError(f"Invalid artifacts for channel '{channel}'")

        logger.info(
            f"[ChannelAdapterDispatcher] Preview generated | channel={channel}"
        )
        return adapter.preview(artifacts)

    def publish(
        self,
        channel: str,
        artifacts: Dict[str, Any],
    ) -> Dict[str, Any]:
        adapter = self.registry.get(channel)

        if not adapter.validate(artifacts):
            logger.error(
                f"[ChannelAdapterDispatcher] Validation failed | "
                f"channel={channel} artifacts={artifacts}"
            )
            raise ValueError(f"Invalid artifacts for channel '{channel}'")

        logger.info(
            f"[ChannelAdapterDispatcher] Publishing campaign | channel={channel}"
        )
        return adapter.publish(artifacts)
