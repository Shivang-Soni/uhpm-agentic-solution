from typing import Dict, Any

from agents.adapters.channel_adapter_registry import ChannelAdapterRegistry


class ChannelAdapterDispatcher:
    """
    Dispatches artifacts to the correct channel adapter
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
            raise ValueError(
                f"Validation failed for channel: '{channel}' with artifact:" \
                f"'{artifacts}"
            )
        return adapter.preview(artifacts)
