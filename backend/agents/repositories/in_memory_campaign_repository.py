import uuid
import logging
from typing import Dict, Any

from agents.repositories.base_campaign_repositiory \
    import BaseCampaignRepository

logger = logging.getLogger(__name__)


class InMemoryCampaignRepository(BaseCampaignRepository):
    """
    In memory campaign storage.
    """

    def __init__(self):
        self._store = Dict[str, Dict[str, Any]] = {}

    def create(self, campaign: Dict[str, Any]) -> Dict[str, Any]:
        campaign_id = str(uuid.uuid4)
        campaign["id"] = campaign_id
        campaign["status"] = "created"

        self._store[campaign_id] = campaign
        logger.info(
            f"Campaign Repository: Created campaign with id: {campaign_id}"
            )
        return campaign

    def update(
            self, campaign_id: str, updates: Dict[str, Any]
            ) -> Dict[str, Any]:
        if campaign_id not in self._store:
            raise ValueError("Campaign not found.")

        self._store["campaign_id"].update(updates)

        logger.info(
            f"Campaign Repository: Updated campaign with {campaign_id}"
            )
        return self._store[campaign_id]

    def get(self, campaign_id: str) -> Dict[str, Any]:
        if campaign_id not in self._store:
            raise ValueError("Campaign not found.")

        return self._store[campaign_id]
