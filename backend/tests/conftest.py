import pytest
from backend.graph.uhpm_graph import create_uhpm_graph
from backend.agents.planner_agent import PlannerAgent


@pytest.fixture
def graph_app():
    """
    Shared fixture to create the full UHPM graph.
    """
    return create_uhpm_graph()


@pytest.fixture
def planner():
    """
    PlannerAgent fixture
    """
    return PlannerAgent()
