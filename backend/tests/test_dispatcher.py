import pytest
from unittest.mock import MagicMock

from backend.agents.dispatcher import Dispatcher


@pytest.fixture
def mock_agents():
    """
    Create mocked agents
    """
    return {
        "research_agent": MagicMock(),
        "persona_agent": MagicMock(),
        "content_agent": MagicMock(),
        "experiment_agent": MagicMock(),
        "analytics_agent": MagicMock()
    }


@pytest.fixture
def dispatcher(mock_agents):
    """
    Return a dispatcher with mocked agents.
    """
    return Dispatcher(
        research_agent=mock_agents["research_agent"],
        persona_agent=mock_agents["persona_agent"],
        content_agent=mock_agents["content_agent"],
        experiment_agent=mock_agents["experiment_agent"],
        analytics_agent=mock_agents["analytics_agent"]
    )


def test_dispatch_research_agent(dispatcher, mock_agents):
    """
    Test whether Dispatcher actually calls ResearchAgent correctly.
    """
    reason_output = {
        "task_type": "research",
        "action": "call_research_agent",
        "inputs_needed": ["product_text"]
    }
    
    user_payload = {
        "product_text": "Some Product"
    }

    mock_agents["research_agent"].analyse_product.return_value = {
        "insights": "good"
    }

    result = dispatcher.run({}, reason_output, user_payload)

    assert result["status"] == "research_done"
    assert result["result"] == {"insights": "good"}
    mock_agents["research_agent"].analyse_product.assert_called_once_with(
        product_text="Some Product",
        competitor_text=""
    )


def test_dispatch_missing_inputs(dispatcher):
    """
    Test dispatcher returns 'waiting_for_inputs' when required inputs missing
    """
    reason_output = {
        "task_type": "research",
        "action": "call_research_agent",
        "inputs_needed": ["product_text", "competitor_text"]
    }
    user_payload = {
        "product_text": "Some Product"
    }
    result = dispatcher.run({}, reason_output, user_payload)

    assert result["status"] == "waiting_for_inputs"
    assert "competitor_text" in result["missing_inputs"]


def test_dispatch_unknown_action(dispatcher):
    """
    Test Dispatcher handles unknown actions gracefully.
    """
    reason_output = {
        "task_type": "unknown",
        "action": "call_unknown_agent",
        "inputs_needed": []
    }
    user_payload = {}

    result = dispatcher.run({}, reason_output, user_payload)

    assert result["status"] == "unknown_action"
    assert result["action"] == "call_unknown_agent"


def test_dispatch_agent_error(dispatcher, mock_agents):
    """
    Test Dispatcher returns 'agent_error' if an agent raises Exception.
    """
    reason_output = {
        "task_type": "research",
        "action": "call_research_agent",
        "inputs_needed": ["product_text"]
    }
    user_payload = {
        "product_text": "Some Product"
    }

    mock_agents["research_agent"].analyse_product.side_effect = Exception("Crash!")

    result = dispatcher.run({}, reason_output, user_payload)

    assert result["status"] == "agent_error"
    assert "Crash!" in result["error"]
