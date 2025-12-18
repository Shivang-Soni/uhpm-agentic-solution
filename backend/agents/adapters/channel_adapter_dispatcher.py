import logging
from typing import Dict, Any

from agents.adapters.channel_adapter_registry import ChannelAdapterRegistry
from agents.repositories.base_campaign_repository import BaseCampaignRepository
from agents.campaigns.campaign_status import CampaignStatus

logger = logging.getLogger(__name__)


class ChannelAdapterDispatcher:
    """
    Central dispatcher for all channel adapters.

    Responsibilities:
    - Resolve adapter by channel
    - Create and manage campaign lifecycle
    - Execute preview and publish safely
    - Persist campaign state transitions
    - Return normalized responses
    """

    def __init__(
        self,
        registry: ChannelAdapterRegistry,
        repository: BaseCampaignRepository,
    ):
        self.registry = registry
        self.repository = repository

    def preview(
        self,
        channel: str,
        artifacts: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Create campaign
        campaign = self.repository.create(
            {
                "channel": channel,
                "artifacts": artifacts,
                "status": CampaignStatus.CREATED,
            }
        )

        campaign_id = campaign["id"]
        adapter = self.registry.get(channel)

        # Update status to PREVIEWING
        self.repository.update(
            campaign_id=campaign_id,
            updates={"status": CampaignStatus.PREVIEWING},
        )

        logger.info(
            f"[ChannelAdapterDispatcher] Preview started | "
            f"campaign_id={campaign_id} channel={channel}"
        )

        try:
            # Execute preview
            result = adapter.safe_preview(artifacts)

            # Update status to PREVIEWED
            self.repository.update(
                campaign_id=campaign_id,
                updates={
                    "status": CampaignStatus.PREVIEWED,
                    "preview": result,
                },
            )

            return {
                "campaign_id": campaign_id,
                "status": CampaignStatus.PREVIEWED,
                "channel": channel,
                "data": result,
            }

        except Exception as e:
            logger.exception(
                f"[ChannelAdapterDispatcher] Preview failed | "
                f"campaign_id={campaign_id} channel={channel}"
            )

            self.repository.update(
                campaign_id=campaign_id,
                updates={
                    "status": CampaignStatus.PREVIEW_FAILED,
                    "error": str(e),
                },
            )

            return {
                "campaign_id": campaign_id,
                "status": CampaignStatus.PREVIEW_FAILED,
                "channel": channel,
                "error": str(e),
            }

    def publish(
        self,
        campaign_id: str,
    ) -> Dict[str, Any]:
        # Load campaign
        campaign = self.repository.get(campaign_id=campaign_id)
        channel = campaign["channel"]
        artifacts = campaign["artifacts"]

        adapter = self.registry.get(channel)

        # Update status to PUBLISHING
        self.repository.update(
            campaign_id=campaign_id,
            updates={"status": CampaignStatus.PUBLISHING},
        )

        logger.info(
            f"[ChannelAdapterDispatcher] Publishing started | "
            f"campaign_id={campaign_id} channel={channel}"
        )

        try:
            # Execute publish
            result = adapter.safe_publish(artifacts)

            # Update status to PUBLISHED
            self.repository.update(
                campaign_id=campaign_id,
                updates={
                    "status": CampaignStatus.PUBLISHED,
                    "publish_result": result,
                },
            )

            return {
                "campaign_id": campaign_id,
                "status": CampaignStatus.PUBLISHED,
                "channel": channel,
                "data": result,
            }

        except Exception as e:
            logger.exception(
                f"[ChannelAdapterDispatcher] Publish failed | "
                f"campaign_id={campaign_id} channel={channel}"
            )

            self.repository.update(
                campaign_id=campaign_id,
                updates={
                    "status": CampaignStatus.PUBLISH_FAILED,
                    "error": str(e),
                },
            )

            return {
                "campaign_id": campaign_id,
                "status": CampaignStatus.PUBLISH_FAILED,
                "channel": channel,
                "error": str(e),
            }
