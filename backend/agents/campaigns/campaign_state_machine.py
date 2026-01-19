import logging
from typing import Dict

logger = logging.getLogger(__name__)


class CampaignStateMachine:
    """
    Deterministic state machine for campaign lifecycle.
    """

    _transitions: Dict[str, Dict[str, str]] = {
        "created": {
            "start_preview": "previewing",
        },
        "previewing": {
            "preview_success": "previewed",
            "preview_failed": "failed",
        },
        "previewed": {
            "start_publish": "publishing",
        },
        "publishing": {
            "publish_success": "published",
            "publish_failed": "failed",
        },
        "published": {},
        "failed": {},
    }

    @classmethod
    def transition(cls, current_state: str, action: str) -> str:
        if current_state not in cls._transitions:
            raise ValueError(f"Unknown campaign state: {current_state}")

        state_actions = cls._transitions[current_state]

        if action not in state_actions:
            raise ValueError(
                f"Illegal campaign transition: "
                f"state={current_state}, action={action}"
            )

        next_state = state_actions[action]

        logger.info(
            f"[CampaignStateMachine] {current_state} --({action})--> {next_state}"
        )

        return next_state
