import httpx

from backend.core.config import Settings
from agents.schemas import CampaignPerformance
from agents.analytics_collectors.base_collector import BaseAnalyticsCollector


class MetaAnalyticsCollector(CampaignPerformance):

    def fetch_performance(self, campaign_id: str) -> CampaignPerformance: