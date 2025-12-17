import logging
from typing import Dict
from agents.adapters.base_channel_adapter import BaseChannelAdapter
from agents.adapters.google_ads_adapter import GoogleAdsAdapter
from agents.adapters.meta_ads_adapter import MetaAdsAdapter
from agents.adapters.email_adapter import EmailAdapter
from agents.adapters.whatsapp_adapter import WhatsappAdapter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ChannelAdapterRegistry:
    """
    Central registry mapping channel identifiers to their adapter instances.
    """

    def __init__(self):
        # Register default adapters
        self._adapters: Dict[str, BaseChannelAdapter] = {
            "google_ads": GoogleAdsAdapter(),
            "meta_ads": MetaAdsAdapter(),
            "whatsapp": WhatsappAdapter(),
            "email": EmailAdapter(),
        }

    def get(self, channel: str) -> BaseChannelAdapter:
        """
        Retrieve the adapter for a given channel.
        """
        if not channel:
            raise ValueError("Channel must be provided")

        normalized_channel = channel.lower().strip()
        adapter = self._adapters.get(normalized_channel)

        if not adapter:
            logger.error(f"No adapter registered for channel: {normalized_channel}")
            raise ValueError(f"No adapter registered for channel: {normalized_channel}")

        logger.info(f"Resolved adapter for channel: {normalized_channel}")
        return adapter

    def list_channels(self) -> list[str]:
        """
        Return a list of all registered channel names.
        """
        return list(self._adapters.keys())
