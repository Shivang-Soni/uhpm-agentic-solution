# backend/agents/reflection_updater.py
from typing import Dict
from actions import Action


def add_reflection(state: Dict, action: Action, text: str, success: bool, campaign_id: str | None = None):
    """
    Save a reflection entry to the agent's state.
    """
    if "self_reflections" not in state:
        state["self_reflections"] = []

    reflection_entry = {
        "action": action.value,
        "text": text,
        "success": success,
        "campaign_id": campaign_id,
    }

    state["self_reflections"].append(reflection_entry)

    return reflection_entry
