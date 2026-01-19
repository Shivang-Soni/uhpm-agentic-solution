import logging
from typing import Dict

from agents.campaigns.campaign_status import CampaignStatus

logger = logging.getLogger(__name__)


class CampaignStateMachine:
    """
    Enforces valid campaign state transitions using explicit actions.
    """

    _transitions = {
        CampaignStatus.CREATED: {
            "start_preview": CampaignStatus.PREVIEWING,
        },
        CampaignStatus.PREVIEWING: {
            "preview_success": CampaignStatus.PREVIEWED,
            "preview_failed": CampaignStatus.PREVIEW_FAILED,
        },
        CampaignStatus.PREVIEWED: {
            "start_publish": CampaignStatus.PUBLISHING,
        },
        CampaignStatus.PUBLISHING: {
            "publish_success": CampaignStatus.PUBLISHED,
            "publish_failed": CampaignStatus.PUBLISH_FAILED,
        },
    }

    @classmethod
    def transition(cls, current_status: CampaignStatus, action: str) -> CampaignStatus:
        if current_status not in cls._transitions:
            raise ValueError(
                f"No transitions defined for state: {current_status}"
            )

        if action not in cls._transitions[current_status]:
            raise ValueError(
                f"Illegal transition: {current_status} --({action})--> ?"
            )

        next_status = cls._transitions[current_status][action]

        logger.info(
            f"[CampaignStateMachine] {current_status} --({action})--> {next_status}"
        )

        return next_status
