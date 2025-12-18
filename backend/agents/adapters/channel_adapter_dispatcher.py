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
    - Enforce adapter capabilities
    - Execute preview or publish safely
    - Normalize execution responses
    """

    def __init__(
            self,
            registry: ChannelAdapterRegistry,
            repository: BaseCampaignRepository
            ):
        self.registry = registry
        self.repository = repository

    def preview(
        self,
        channel: str,
        artifacts: Dict[str, Any],
    ) -> Dict[str, Any]:
        campaign = self.repository.create(
            {
                "channel": channel,
                "artifacts": artifacts,
                "status": CampaignStatus.CREATED
            }
        )

        campaign_id = campaign["id"]
        adapter = self.registry.get(channel)

        self.repository.update(
            campaign_id=campaign_id,
            updates={
                "status": CampaignStatus.PREVIEWING
            }
        )

        logger.info(
            f"[ChannelAdapterDispatcher] Preview request | channel={channel}"
        )

        try:
            result = adapter.safe_preview(artifacts)

            self.repository.update(
                campaign_id,
                {
                    "status": CampaignStatus.PREVIEWED,
                    "preview": result
                }
            )

            return {
                "campaign_id": campaign_id,
                "status": "preview_ready",
                "channel": channel,
                "data": result,
            }

        except Exception as e:
            logger.exception(
                f"[ChannelAdapterDispatcher] Preview failed"
                f" | channel={channel}"
            )

            self.repository.update(
                campaign_id=campaign_id,
                updates={
                    "status": CampaignStatus.PREVIEW_FAILED,
                    "error": str(e)
                }
            )

            return {
                "campaign_id": campaign_id,
                "status": "preview_failed",
                "channel": channel,
                "error": str(e),
            }

    def publish(
        self,
        campaign_id: str,
    ) -> Dict[str, Any]:
        campaign = self.repository.get(campaign_id=campaign_id)
        channel = campaign["channel"]
        artifacts = campaign["artifacts"]

        adapter = self.registry.get(channel)

        self.repository.update(
            campaign_id=campaign_id,
            updates={
                "status": CampaignStatus.PUBLISHING
            }
        )

        logger.info(
            f"[ChannelAdapterDispatcher] Publishing started | channel={channel}"
        )

        try:
            result = adapter.safe_publish(artifacts)

            self.repository.update(
                campaign_id=campaign_id,
                updates={
                    "status": CampaignStatus.PUBLISHED,
                    'publish_result': result
                }
            )

            return {
                "campaign_id": campaign_id,
                "status": CampaignStatus.PUBLISHED,
                "channel": channel,
                "data": result,
            }

        except Exception as e:
            logger.exception(
                f"[ChannelAdapterDispatcher] Publish failed | channel={channel}"
            )

            self.repository.update(
                campaign_id=campaign_id,
                updates={
                    "status": CampaignStatus.PUBLISH_FAILED,
                    "error": str(e)
                }
            )

            return {
                "campaign_id": campaign_id,
                "status": CampaignStatus.PUBLISH_FAILED,
                "channel": channel,
                "error": str(e),
            }
