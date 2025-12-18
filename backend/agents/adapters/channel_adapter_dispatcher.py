import logging
from typing import Dict, Any

from agents.adapters.channel_adapter_registry import ChannelAdapterRegistry
from agents.repositories.base_campaign_repository import BaseCampaignRepository
from agents.campaigns.campaign_status import CampaignStatus
from agents.campaigns.campaign_state_machine import CampaignStateMachine

logger = logging.getLogger(__name__)


class ChannelAdapterDispatcher:
    """
    Central dispatcher for all channel adapters.

    Responsibilities:
    - Resolve adapter by channel
    - Enforce campaign state transitions
    - Execute preview and publish safely
    - Persist campaign lifecycle
    """

    def __init__(
        self,
        registry: ChannelAdapterRegistry,
        repository: BaseCampaignRepository,
    ):
        self.registry = registry
        self.repository = repository

    # Preview

    def preview(
        self,
        channel: str,
        artifacts: Dict[str, Any],
    ) -> Dict[str, Any]:

        campaign = self.repository.create(
            {
                "channel": channel,
                "artifacts": artifacts,
                "status": CampaignStatus.CREATED,
            }
        )

        campaign_id = campaign["id"]
        adapter = self.registry.get(channel)

        # Transition: CREATED to PREVIEWING
        next_status = CampaignStateMachine.transition(
            campaign["status"], "start_preview"
        )
        self.repository.update(
            campaign_id=campaign_id,
            updates={"status": next_status},
        )

        logger.info(
            f"[ChannelAdapterDispatcher] Preview started | "
            f"campaign_id={campaign_id} channel={channel}"
        )

        try:
            result = adapter.safe_preview(artifacts)

            # Transition: PREVIEWING to PREVIEWED
            next_status = CampaignStateMachine.transition(
                next_status, "preview_success"
            )
            self.repository.update(
                campaign_id=campaign_id,
                updates={
                    "status": next_status,
                    "preview": result,
                },
            )

            return {
                "campaign_id": campaign_id,
                "status": next_status,
                "channel": channel,
                "data": result,
            }

        except Exception as e:
            logger.exception(
                f"[ChannelAdapterDispatcher] Preview failed | "
                f"campaign_id={campaign_id} channel={channel}"
            )

            failed_status = CampaignStateMachine.transition(
                next_status, "preview_failed"
            )
            self.repository.update(
                campaign_id=campaign_id,
                updates={
                    "status": failed_status,
                    "error": str(e),
                },
            )

            return {
                "campaign_id": campaign_id,
                "status": failed_status,
                "channel": channel,
                "error": str(e),
            }

    # Publish

    def publish(self, campaign_id: str) -> Dict[str, Any]:

        campaign = self.repository.get(campaign_id=campaign_id)
        channel = campaign["channel"]
        artifacts = campaign["artifacts"]

        adapter = self.registry.get(channel)

        # Transition: PREVIEWED to PUBLISHING
        next_status = CampaignStateMachine.transition(
            campaign["status"], "start_publish"
        )
        self.repository.update(
            campaign_id=campaign_id,
            updates={"status": next_status},
        )

        logger.info(
            f"[ChannelAdapterDispatcher] Publishing started | "
            f"campaign_id={campaign_id} channel={channel}"
        )

        try:
            result = adapter.safe_publish(artifacts)

            # Transition: PUBLISHING to PUBLISHED
            final_status = CampaignStateMachine.transition(
                next_status, "publish_success"
            )
            self.repository.update(
                campaign_id=campaign_id,
                updates={
                    "status": final_status,
                    "publish_result": result,
                },
            )

            return {
                "campaign_id": campaign_id,
                "status": final_status,
                "channel": channel,
                "data": result,
            }

        except Exception as e:
            logger.exception(
                f"[ChannelAdapterDispatcher] Publish failed | "
                f"campaign_id={campaign_id} channel={channel}"
            )

            failed_status = CampaignStateMachine.transition(
                next_status, "publish_failed"
            )
            self.repository.update(
                campaign_id=campaign_id,
                updates={
                    "status": failed_status,
                    "error": str(e),
                },
            )

            return {
                "campaign_id": campaign_id,
                "status": failed_status,
                "channel": channel,
                "error": str(e),
            }
