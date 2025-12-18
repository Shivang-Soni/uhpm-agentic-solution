import logging
from typing import Dict, Any

from agents.adapters.channel_adapter_dispatcher import ChannelAdapterDispatcher
from agents.repositories.base_campaign_repository import BaseCampaignRepository
from agents.campaigns.campaign_status import CampaignStatus

logger = logging.getLogger(__name__)


class CampaignService:
    """
    Application layer for campaign execution.

    Responsiblities:
    - Validate high-level campaign intent
    - Delegate execution to dispatcher
    - Guard invalid operations
    """

    def __init__(
            self,
            dispatcher: ChannelAdapterDispatcher,
            repositiory: BaseCampaignRepository
    ):
        self.dispatcher = dispatcher
        self.repository = repositiory

    # Preview
    def preview_campaign(
            self,
            channel: str,
            artifacts: Dict[str, Any]
    ) -> Dict[str, Any]:

        if not channel:
            raise ValueError(
                "Channel must be provided."
            )

        logger.info(
            f"CampaignService: Preview requested | channel = {channel}"
        )

        return self.dispatcher.preview(
            channel=channel,
            artifacts=artifacts,
        )

    # Publish
    def publish_campaign(
            self,
            campaign_id: str,
    ) -> Dict[str, Any]:
        
        campaign = self.repository.get(campaign_id)

        if campaign["status"] != CampaignStatus.PREVIEWED:
            raise RuntimeError(
                f"Campaign {campaign_id} is not ready for publishing"
                f"(status = {campaign.get("status")})"
            )
        
        logger.info(
            f"CampaignService: Publish requested | campaign_id: {campaign_id}"
        )

        self.dispatcher.publish(campaign_id)