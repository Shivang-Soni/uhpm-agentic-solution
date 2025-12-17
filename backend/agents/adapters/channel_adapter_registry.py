import logging
from typing import Dict

from agents.adapters.base_channel_adapter import BaseChannelAdapter
from agents.adapters.google_ads_adapter import GoogleAdsAdapter
from agents.adapters.meta_ads_adapter import MetaAdsAdapter
from agents.adapters.email_adapter import EmailAdapter
from agents.adapters.whatsapp_adapter import WhatsappAdapter

logger = logging.getLogger(__name__)


class ChannelAdapterRegistry:
    """
    Central registry mapping channel identifiers to channel adapters.
    """

    def __init__(self):
        self._adapters: Dict[str, BaseChannelAdapter] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register("google_ads", GoogleAdsAdapter())
        self.register("meta_ads", MetaAdsAdapter())
        self.register("whatsapp", WhatsappAdapter())
        self.register("email", EmailAdapter())

    def register(self, channel: str, adapter: BaseChannelAdapter):
        if not channel:
            raise ValueError("Channel name must be provided")

        if not isinstance(adapter, BaseChannelAdapter):
            raise TypeError("Adapter must extend BaseChannelAdapter")

        normalized = channel.lower().strip()
        self._adapters[normalized] = adapter

        logger.info(f"Registered channel adapter: {normalized}")

    def get(self, channel: str) -> BaseChannelAdapter:
        if not channel:
            raise ValueError("Channel must be provided")

        normalized = channel.lower().strip()
        adapter = self._adapters.get(normalized)

        if not adapter:
            logger.error(f"No adapter registered for channel: {normalized}")
            raise ValueError(f"No adapter registered for channel: {normalized}")

        logger.info(f"Resolved adapter for channel: {normalized}")
        return adapter

    def list_channels(self) -> list[str]:
        return list(self._adapters.keys())
