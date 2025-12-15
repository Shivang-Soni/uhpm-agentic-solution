import pytest
from unittest.mock import patch, MagicMock
from backend.agents.reasoner import ReasonerAgent


@pytest.fixture
def agent():
    return ReasonerAgent(retriever=None)


@patch("backend.agents.reasoner.invoke")
def test_decide_return_dict(mock_invoke, agent):
    mock_invoke.return_value = """
    {
        "task_type": "research",
        "reasoning": "ok",
        "action": "call_research_agent",
        "inputs_needed": {}
    }
    """
    result = agent.decide("run research task")
    assert isinstance(result, dict)


@patch("backend.agents.reasoner.invoke")
def test_decide_contains_required_keys(mock_invoke, agent):
    mock_invoke.return_value = """
    {
        "task_type": "persona",
        "reasoning": "ok",
        "action": "call_persona_agent",
        "inputs_needed": {"product_text": "required"}
    }
    """
    result = agent.decide("create persona")
    for key in ["task_type", "reasoning", "action", "inputs_needed"]:
        assert key in result


@patch("backend.agents.reasoner.invoke")
def test_decide_fallback_on_invalid_json(mock_invoke, agent):
    mock_invoke.side_effect = [
        "NOT JSON",  # first call invalid
        """
        {
            "task_type": "research",
            "reasoning": "ok",
            "action": "call_research_agent",
            "inputs_needed": {}
        }
        """  # fallback valid
    ]
    result = agent.decide("run task")
    assert result["task_type"] == "research"


@patch("backend.agents.reasoner.invoke")
def test_decide_no_response_returns_default(mock_invoke, agent):
    mock_invoke.return_value = None
    result = agent.decide("any task")
    assert result["task_type"] == "research"
    assert result["action"] == "call_research_agent"


def test_decide_with_retriever_error(monkeypatch):
    # Retriever raises Exception
    faulty_retriever = MagicMock()
    faulty_retriever.search_docs.side_effect = Exception("Retriever crash")
    agent = ReasonerAgent(retriever=faulty_retriever)

    with patch("backend.agents.reasoner.invoke") as mock_invoke:
        mock_invoke.return_value = """
        {
            "task_type": "research",
            "reasoning": "ok",
            "action": "call_research_agent",
            "inputs_needed": {}
        }
        """
        result = agent.decide("task")
        assert result["task_type"] == "research"
