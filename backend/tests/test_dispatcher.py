import pytest
from unittest.mock import MagicMock

from backend.agents.dispatcher import Dispatcher


# Fixtures
@pytest.fixture
def mock_agents():
    return {
        "research_agent": MagicMock(),
        "persona_agent": MagicMock(),
        "content_agent": MagicMock(),
        "experiment_agent": MagicMock(),
        "analytics_agent": MagicMock(),
        "whatsapp_agent": MagicMock(),
        "googleads_agent": MagicMock(),
        "metaads_agent": MagicMock(),
        "email_agent": MagicMock(),
    }


@pytest.fixture
def dispatcher(mock_agents):
    return Dispatcher(
        research_agent=mock_agents["research_agent"],
        persona_agent=mock_agents["persona_agent"],
        content_agent=mock_agents["content_agent"],
        experiment_agent=mock_agents["experiment_agent"],
        analytics_agent=mock_agents["analytics_agent"],
        whatsapp_agent=mock_agents["whatsapp_agent"],
        google_ads_agent=mock_agents["googleads_agent"],
        meta_ads_agent=mock_agents["metaads_agent"],
        email_agent=mock_agents["email_agent"],
    )


# Research Agent
def test_dispatch_research_agent(dispatcher, mock_agents):
    reason_output = {
        "task_type": "research",
        "action": "call_research_agent",
        "inputs_needed": {
            "product_text": "required"
        }
    }

    user_payload = {
        "product_text": "Some Product"
    }

    mock_agents["research_agent"].analyse_product.return_value = {
        "insights": "market is growing"
    }

    result = dispatcher.run({}, reason_output, user_payload)

    assert result["status"] == "research_done"
    assert result["agent"] == "research"
    assert result["data"]["insights"] == "market is growing"

    mock_agents["research_agent"].analyse_product.assert_called_once_with(
        product_text="Some Product",
        competitor_text=None
    )


# Missing Inputs
def test_dispatch_missing_inputs(dispatcher):
    reason_output = {
        "task_type": "research",
        "action": "call_research_agent",
        "inputs_needed": {
            "product_text": "required",
            "competitor_text": "optional competitor context"
        }
    }

    user_payload = {} # Empty payload

    result = dispatcher.run({}, reason_output, user_payload)

    assert result["status"] == "waiting_for_inputs"
    assert result["agent"] == "dispatcher"
    assert "product_text" in result["data"]["missing_inputs"]


# Unknown Action
def test_dispatch_unknown_action(dispatcher):
    reason_output = {
        "task_type": "unknown",
        "action": "call_unknown_agent",
        "inputs_needed": {}
    }

    result = dispatcher.run({}, reason_output, {})

    assert result["status"] == "unknown_action"
    assert result["agent"] == "dispatcher"
    assert result["data"]["action"] == "call_unknown_agent"


# Agent Error Handling
def test_dispatch_agent_error(dispatcher, mock_agents):
    reason_output = {
        "task_type": "research",
        "action": "call_research_agent",
        "inputs_needed": {
            "product_text": "required"
        }
    }

    user_payload = {
        "product_text": "Some Product"
    }

    mock_agents["research_agent"].analyse_product.side_effect = Exception("Crash!")

    result = dispatcher.run({}, reason_output, user_payload)

    assert result["status"] == "agent_error"
    assert result["agent"] == "dispatcher"
    assert "Crash!" in result["data"]["error"]


# WhatsApp Agent
def test_dispatch_whatsapp_agent(dispatcher, mock_agents):
    reason_output = {
        "task_type": "content",
        "action": "call_whatsapp_agent",
        "inputs_needed": {
            "product_text": "required",
            "persona_text": "required"
        }
    }

    user_payload = {
        "product_text": "Local bakery offering fresh bread",
        "persona_text": "Budget-conscious locals",
        "intent": "lead",
        "tone": "friendly"
    }

    mock_agents["whatsapp_agent"].generate_messages.return_value = {
        "initial_message": "Fresh bread today!",
        "follow_up_message": "Want to order?",
        "closing_message": "Reply YES"
    }

    result = dispatcher.run({}, reason_output, user_payload)

    assert result["status"] == "whatsapp_messages_generated"
    assert result["agent"] == "whatsapp"
    assert "initial_message" in result["data"]

    mock_agents["whatsapp_agent"].generate_messages.assert_called_once()


# Google Ads Agent
def test_dispatch_google_ads_agent(dispatcher, mock_agents):
    reason_output = {
        "task_type": "channel",
        "action": "call_google_ads_agent",
        "inputs_needed": {
            "product_text": "required",
            "persona_text": "required",
            "campaign_budget": "required"
        }
    }

    user_payload = {
        "product_text": "AI Marketing Tool",
        "persona_text": "Startup founders",
        "campaign_budget": "1000 USD"
    }

    mock_agents["googleads_agent"].generate_campaign.return_value = {
        "headline": "Grow Faster with AI",
        "description": "Automate marketing",
        "keywords": ["ai marketing"],
        "daily_budget_estimate": "30 USD",
        "landing_page_angle": "ROI focused"
    }

    result = dispatcher.run({}, reason_output, user_payload)

    assert result["status"] == "google_ads_generated"
    assert result["agent"] == "google_ads"
    assert "headline" in result["data"]


# Meta Ads Agent
def test_dispatch_meta_ads_agent(dispatcher, mock_agents):
    reason_output = {
        "task_type": "channel",
        "action": "call_meta_ads_agent",
        "inputs_needed": {
            "product_text": "required",
            "persona_text": "required",
            "campaign_budget": "required"
        }
    }

    user_payload = {
        "product_text": "Fitness App",
        "persona_text": "Busy professionals",
        "campaign_budget": "500 USD"
    }

    mock_agents["metaads_agent"].generate_campaign.return_value = {
        "platform": "meta",
        "headline": "Train Smarter",
        "persona": "Busy professionals",
        "budget": "500 USD",
        "tone": "motivational"
    }

    result = dispatcher.run({}, reason_output, user_payload)

    assert result["status"] == "meta_ads_generated"
    assert result["agent"] == "meta_ads"
    assert result["data"]["platform"] == "meta"


# Email Agent
def test_dispatch_email_agent(dispatcher, mock_agents):
    reason_output = {
        "task_type": "channel",
        "action": "call_email_agent",
        "inputs_needed": {
            "product_text": "required",
            "persona_text": "required",
            "email_template": "required"
        }
    }

    user_payload = {
        "product_text": "CRM Tool",
        "persona_text": "Sales Managers",
        "email_template": "Hello {{name}}"
    }

    mock_agents["email_agent"].generate_campaign.return_value = {
        "subject_line": "CRM Tool Special Offer",
        "body": "Hello Sales Managers",
        "tone": "friendly"
    }

    result = dispatcher.run({}, reason_output, user_payload)

    assert result["status"] == "email_campaign_generated"
    assert result["agent"] == "email"
    assert "subject_line" in result["data"]
