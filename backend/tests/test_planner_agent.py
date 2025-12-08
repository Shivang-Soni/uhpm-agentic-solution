import pytest
from unittest.mock import patch
from backend.agents.planner_agent import PlannerAgent


@pytest.fixture
def agent():
    return PlannerAgent()


@patch("backend.agents.planner_agent.invoke")
def test_plan_valid_json(mock_invoke, agent):
    mock_invoke.return_value = """
    {
        "task": "create ad copy",
        "needs_research": false,
        "needs_persona": true,
        "needs_content": true,
        "needs_experimentation": false,
        "needs_analytics": false,
        "additional_context": "Persona + content required."
    }
    """

    result = agent.plan("create ad copy")

    assert isinstance(result, dict)
    assert result["needs_persona"] is True
    assert result["needs_content"] is True
    assert result["task"] == "create ad copy"


@patch("backend.agents.planner_agent.invoke")
def test_plan_no_response(mock_invoke, agent):
    mock_invoke.return_value = None

    result = agent.plan("generate content")

    assert isinstance(result, dict)
    assert result["needs_research"] is True
    assert result["additional_context"].startswith("Fallback")


@patch("backend.agents.planner_agent.invoke")
def test_plan_invalid_json(mock_invoke, agent):
    mock_invoke.return_value = "INVALID_JSON"

    result = agent.plan("fix json")

    assert isinstance(result, dict)
    assert result["needs_research"] is True
    assert "fallback" in result["additional_context"].lower()


@patch("backend.agents.planner_agent.invoke")
def test_plan_invalid_schema(mock_invoke, agent):
    mock_invoke.return_value = """
    {
        "task": "weird output",
        "needs_research": "not_boolean",
        "needs_persona": [],
        "needs_content": 123,
        "needs_experimentation": false,
        "needs_analytics": false,
        "additional_context": "invalid schema"
    }
    """

    result = agent.plan("weird output")

    assert isinstance(result, dict)
    assert result["needs_research"] is True
    assert "fallback" in result["additional_context"].lower()
