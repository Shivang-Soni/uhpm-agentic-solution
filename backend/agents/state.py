from typing import Dict, Any

from agents.actions import Action
from agents.schemas import CampaignState


def create_initial_state(
        brief: str,
        target_audience: str | None = None,
) -> CampaignState:
    """
    Single entry point for campaign state creation.
    """
    return CampaignState(
        campaign_id="",
        current_action=None,
        brief=brief,
        target_audience=target_audience,
        persona=None,
        content=None,
        experiments=[],
        performance_metrics=None,
        published=False,
        errors=[],
        history=[]
    )


def apply_execution_result(
        state: CampaignState,
        result_data: Dict[str, Any]
):
    """
    Controlled mutation of CampaignState after agent execution.
    """
    if "persona" in result_data:
        state["persona"] = result_data["persona"]

    if "content" in result_data:
        state["content"] = result_data["content"]

    if "experiments" in result_data:
        state["experiments"] = result_data["experiments"]

    if "performance_metrics" in result_data:
        state["performance_metrics"] = result_data["performance_metrics"]

    if "published" in result_data:
        state["published"] = result_data["published"]
