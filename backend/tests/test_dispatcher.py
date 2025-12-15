import pytest
from unittest.mock import MagicMock

from backend.agents.dispatcher import Dispatcher


@pytest.fixture
def mock_agents():
    return {
        "research_agent": MagicMock(),
        "persona_agent": MagicMock(),
        "content_agent": MagicMock(),
        "experiment_agent": MagicMock(),
        "analytics_agent": MagicMock(),
        "whatsapp_agent": MagicMock()
    }


@pytest.fixture
def dispatcher(mock_agents):
    return Dispatcher(
        research_agent=mock_agents["research_agent"],
        persona_agent=mock_agents["persona_agent"],
        content_agent=mock_agents["content_agent"],
        experiment_agent=mock_agents["experiment_agent"],
        analytics_agent=mock_agents["analytics_agent"],
        whatsapp_agent=mock_agents["whatsapp_agent"]
    )


def test_dispatch_research_agent(dispatcher, mock_agents):
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
    assert result["agent"] == "research"
    assert result["data"] == {"insights": "good"}

    mock_agents["research_agent"].analyse_product.assert_called_once_with(
        product_text="Some Product",
        competitor_text=None
    )


def test_dispatch_missing_inputs(dispatcher):
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
    assert result["agent"] == "dispatcher"
    assert "competitor_text" in result["data"]["missing_inputs"]


def test_dispatch_unknown_action(dispatcher):
    reason_output = {
        "task_type": "unknown",
        "action": "call_unknown_agent",
        "inputs_needed": []
    }

    result = dispatcher.run({}, reason_output, {})

    assert result["status"] == "unknown_action"
    assert result["agent"] == "dispatcher"
    assert result["data"]["action"] == "call_unknown_agent"


def test_dispatch_agent_error(dispatcher, mock_agents):
    reason_output = {
        "task_type": "research",
        "action": "call_research_agent",
        "inputs_needed": ["product_text"]
    }

    user_payload = {
        "product_text": "Some Product"
    }

    mock_agents["research_agent"].analyse_product.side_effect = Exception(
        "Crash!"
        )

    result = dispatcher.run({}, reason_output, user_payload)

    assert result["status"] == "agent_error"
    assert result["agent"] == "dispatcher"
    assert "Crash!" in result["data"]["error"]


def test_dispatch_whatsapp_agent(dispatcher, mock_agents):
    reason_output = {
        "task_type": "messaging",
        "action": "call_whatsapp_agent",
        "inputs_needed": ["product_text", "persona_text"]
    }

    user_payload = {
        "product_text": "Local bakery offering fresh bread.",
        "persona_text": "Budget-conscious local customers",
        "intent": "lead",
        "tone": "friendly"
    }

    mock_agents["whatsapp_agent"].generate_messages.return_value = {
        "initial_message": "Hi! We bake fresh bread daily.",
        "follow_up_message": "Would you like today’s specials?",
        "closing_message": "Reply YES to order."
    }

    result = dispatcher.run({}, reason_output, user_payload)

    assert result["status"] == "whatsapp_messages_generated"
    assert result["agent"] == "whatsapp"
    assert "initial_message" in result["data"]

    mock_agents["whatsapp_agent"].generate_messages.assert_called_once_with(
        product_text="Local bakery offering fresh bread.",
        persona_text="Budget-conscious local customers",
        intent="lead",
        tone="friendly"
    )
