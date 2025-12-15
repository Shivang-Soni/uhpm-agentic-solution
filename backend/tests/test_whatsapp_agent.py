import pytest
from unittest.mock import patch

from agents.whatsapp_agent import WhatsappAgent


@ pytest.fixture
def agent():
    return WhatsappAgent()


@ patch("agents.whatsapp_agent.invoke")
def test_generate_messages_valid_json(mock_invoke, agent):
    mock_invoke.return_value = """
    {
    "initial_message": "Hello, this is a test.",
    "follow_up_message": "Just following up.",
    "closing_message": "Looking forward to know whether you are interested."
    }
    """

    result = agent.generate_messages(
        product_text="Test Product",
        persona_text="Test Persona"
    )

    assert isinstance(result, dict)
    assert "initial_message" in result
    assert "follow_up_message" in result
    assert "closing_message" in result


@patch("agents.whatsapp_agent.invoke")
def test_generate_messages_fallback_on_invalid_json(mock_invoke, agent):
    mock_invoke.side_effect = [
        "INVALID RESPONSE",
        """
        {
        "initial_message": "Fallback initial",
        "follow_up_message": "Fallback follow up",
        "closing_message": "Fallback closing"
        }
        """
    ]

    result = agent.generate_messages(
        product_text="Test Product",
        persona_text="Test Persona"
    )

    assert result["initial_message"] == "Fallback initial"
