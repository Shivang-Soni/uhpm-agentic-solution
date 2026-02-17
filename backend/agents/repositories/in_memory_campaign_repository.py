import uuid
import logging
from typing import Dict, Any

from agents.repositories.base_campaign_repository import (
    BaseCampaignRepository
)

logger = logging.getLogger(__name__)


class InMemoryCampaignRepository(BaseCampaignRepository):
    """
    In-memory campaign repository.
    Used for local development and MVP phase.
    """

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def create(self, campaign: Dict[str, Any]) -> Dict[str, Any]:
        campaign_id = str(uuid.uuid4())

        record = {
            "campaign_id": campaign_id,
            **campaign
        }

        self._store[campaign_id] = record

        logger.info(
            f"[CampaignRepository] Created campaign | id={campaign_id}"
        )
        return record

    def update(
        self,
        campaign_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        if campaign_id not in self._store:
            raise KeyError(f"Campaign not found: {campaign_id}")

        self._store[campaign_id].update(updates)

        logger.info(
            f"[CampaignRepository] Updated campaign | id={campaign_id}"
        )
        return self._store[campaign_id]

    def get(self, campaign_id: str) -> Dict[str, Any]:
        if campaign_id not in self._store:
            raise KeyError(f"Campaign not found: {campaign_id}")

        return self._store[campaign_id]

    def list_all(self) -> list[Dict[str, Any]]:
        return list(self._store.values())
