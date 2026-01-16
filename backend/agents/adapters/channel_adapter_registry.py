import logging
from typing import Dict, List

from agents.adapters.base_channel_adapter import BaseChannelAdapter
from agents.adapters.google_ads_adapter import GoogleAdsAdapter
from agents.adapters.meta_ads_adapter import MetaAdsAdapter
from agents.adapters.email_adapter import EmailAdapter
from agents.adapters.whatsapp_adapter import WhatsappAdapter

logger = logging.getLogger(__name__)


class ChannelAdapterRegistry:
    """
    Central registry mapping channel identifiers to channel adapters.

    Responsibilities:
    - Register and resolve channel adapters
    - Normalize channel identifiers
    - Act as single source of truth for supported channels
    """

    def __init__(self):
        self._adapters: Dict[str, BaseChannelAdapter] = {}
        self._bootstrap_default_adapters()

    def _bootstrap_default_adapters(self) -> None:
        """
        Register all built-in adapters.
        """
        self.register("google_ads", GoogleAdsAdapter())
        self.register("meta_ads", MetaAdsAdapter())
        self.register("email", EmailAdapter())
        self.register("whatsapp", WhatsappAdapter())

        logger.info(
            f"[ChannelAdapterRegistry] Initialized with channels: "
            f"{list(self._adapters.keys())}"
        )

    def register(self, channel: str, adapter: BaseChannelAdapter) -> None:
        """
        Register a new channel adapter.
        """
        if not channel:
            raise ValueError("Channel name must be provided")

        normalized = channel.lower().strip()

        self._adapters[normalized] = adapter
        logger.info(
            f"[ChannelAdapterRegistry] Adapter registered | channel={normalized}"
        )

    def get(self, channel: str) -> BaseChannelAdapter:
        """
        Resolve adapter for the given channel.
        """
        if not channel:
            raise ValueError("Channel must be provided")

        normalized = channel.lower().strip()

        adapter = self._adapters.get(normalized)
        if not adapter:
            logger.error(
                f"[ChannelAdapterRegistry] No adapter registered | channel={normalized}"
            )
            raise ValueError(f"No adapter registered for channel: {normalized}")

        return adapter

    def list_channels(self) -> List[str]:
        """
        List all supported channels.
        """
        return list(self._adapters.keys())
