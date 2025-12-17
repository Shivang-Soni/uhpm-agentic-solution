from typing import Dict, Any
from agents.adapters.channel_adapter_registry import ChannelAdapterRegistry


class ChannelAdapterDispatcher:
    """
    Dispatches campaign artifacts to the correct channel adapter.
    Handles validation and forwards to preview/publish methods.
    """

    def __init__(self, registry: ChannelAdapterRegistry):
        self.registry = registry

    def preview(self, channel: str, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and preview artifacts for the given channel.
        """
        adapter = self.registry.get(channel)

        if not adapter.validate(artifacts):
            raise ValueError(
                f"Validation failed for channel '{channel}' with artifacts: {artifacts}"
            )

        return adapter.preview(artifacts)

    def publish(self, channel: str, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and publish artifacts for the given channel.
        """
        adapter = self.registry.get(channel)

        if not adapter.validate(artifacts):
            raise ValueError(
                f"Validation failed for channel '{channel}' with artifacts: {artifacts}"
            )

        return adapter.publish(artifacts)
