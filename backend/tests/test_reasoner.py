import pytest
from unittest.mock import patch

from backend.agents.reasoner import ReasonerAgent


@pytest.fixture
def agent():
    return ReasonerAgent(retriever=None)


@patch("backend.agents.reasoner.invoke")
def test_decide_return_dict(mock_invoke, agent):
    mock_invoke.return_value = """
    {
    "task_type": "research",
    "reasoning": "ok,
    "action": "call_research_agent",
    "inputs_needed": []
    }
    """

    result = agent.decide("run research task")

    assert isinstance(result, dict)


@patch("agents.reasoner.invoke")
def test_decide_contains_required_keys(mock_invoke, agent):
    mock_invoke.return_value = """
    {
    "task_type": "persona",
    "reasoning": "ok",
    "action": "call_persona_agent",
    "inputs_needed": ["product_text"]
    }
    """

    result = agent.decide("create persona")

    for key in ["task_type", "reasoning", "action", "inputs_needed"]:
        assert key in result
