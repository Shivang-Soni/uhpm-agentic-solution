import logging
from typing import Dict, Set

logger = logging.getLogger(__name__)


class CampaignStateMachine:
    """
    Enforces campaign state transitions.
    """

    # Allowed state transitions
    _transitions = Dict[str, Set(str)] = {
        "created": {"validated"},
        "validated": {"previewed", "failed"},
        "previewed": {"published", "failed"},
        "published": set(),
        "failed": set(),
    }

    @classmethod
    def can_transition(cls, from_state: str, to_state: str) -> bool:
        return to_state in cls._transitions.get(from_state, set())
    
    @classmethod
    def transition(cls, campaign: Dict, to_state: str) -> Dict:
        current_state = campaign.get("status")

        if current_state is None:
            raise ValueError("Campaign has no status.")
        
        if not cls.can_transition(current_state, to_state):
            raise ValueError (
                f"Illegal campaign state transition: ",
                f"{current_state} -> {to_state}"
            )
        
        campaign["status"] = to_state

        logger.info(
            f"CampaignStateMachine: Transitioned campaign"
            f"{campaign.get('id')} from {current_state} to {to_state}"
        )

        return campaign