import logging
from typing import Dict

logger = logging.getLogger(__name__)


class CampaignStateMachine:
    """
    Event-driven finite state machine for campaign lifecycle.
    """

    # (current_state, event) -> next_state
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
        "failed": {},
        "published": {},
    }

    @classmethod
    def transition(cls, current_state: str, event: str) -> str:
        if current_state not in cls._transitions:
            raise ValueError(f"Unknown campaign state: {current_state}")

        state_events = cls._transitions[current_state]

        if event not in state_events:
            raise ValueError(
                f"Illegal campaign transition: "
                f"state={current_state}, event={event}"
            )

        next_state = state_events[event]

        logger.info(
            f"[CampaignStateMachine] "
            f"{current_state} --({event})--> {next_state}"
        )

        return next_state
