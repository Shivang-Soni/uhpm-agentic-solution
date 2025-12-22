from abc import ABC, abstractmethod
from agents.schemas import CampaignPerformance


class BaseAnalyticsCollector(ABC):

    @abstractmethod
    def fetch_performance(self, campaign_id: str) -> CampaignPerformance:
        pass
