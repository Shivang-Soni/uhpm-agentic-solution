from typing import Dict

from agents.adapters.base_channel_adapter import BaseChannelAdapter
from agents.adapters.google_ads_adapter import GoogleAdsAdapter
from agents.adapters.meta_ads_adapter import MetaAdsAdapter
from agents.adapters.email_adapter import EmailAdapter
from agents.adapters.whatsapp_adapter import WhatsappAdapter


class ChannelAdapterRegistry:
    """
    Central registry mapping channel names to adapters.
    """

    def __init__(self):
        self._adapters: Dict[str, BaseChannelAdapter] = {
            "google_ads": GoogleAdsAdapter(),
            "meta_ads": MetaAdsAdapter(),
            "whatsapp": WhatsappAdapter(),
            "email": EmailAdapter()
        }

    def get(self, channel: str) -> BaseChannelAdapter:
        if channel not in self._adapters:
            raise ValueError(f"No adapter registered for {channel}")
        return self._adapters[channel]
