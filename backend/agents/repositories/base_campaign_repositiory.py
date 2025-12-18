from typing import Dict, Any
from abc import ABC, abstractmethod


class BaseCampaignRepository(ABC):
    """
    Abstact repository for campaign persistence.
    """

    @abstractmethod
    def create(self, campaign: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def update(
        self, campaign_id: str, updates: Dict[str, Any]
        ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get(self, campaign_id: str) -> Dict[str, Any]:
        pass
