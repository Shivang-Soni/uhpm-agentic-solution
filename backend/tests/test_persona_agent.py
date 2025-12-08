import pytest
from unittest.mock import patch

from backend.agents.persona_agent import PersonaAgent


@pytest.fixture
def agent():
    return PersonaAgent()


def test_generate_persona_valid_json(agent):
    fake_response = """
    {
        "persona_name": "Tech Tom",
        "age_range": "25-40",
        "demographics": "Urban professionals",
        "lifestyle": "Active",
        "deep_motivations": "Career growth",
        "pain_points": "Time shortage",
        "buying_triggers": "Efficiency tools",
        "objections": "Price",
        "language_and_tone": "Professional",
        "recommended_channels": ["LinkedIn", "YouTube"],
        "summary": "A motivated tech buyer."
    }
    """

    with patch("backend.agents.persona_agent.invoke", return_value=fake_response) as mock_invoke, \
         patch("backend.agents.persona_agent.add_document") as mock_add:

        result = agent.generate_persona("Product X")

        assert result["persona_name"] == "Tech Tom"
        assert result["recommended_channels"] == ["LinkedIn", "YouTube"]

        mock_invoke.assert_called_once()
        mock_add.assert_called_once()


def test_generate_persona_invalid_json(agent):
    with patch("backend.agents.persona_agent.invoke", return_value="NOT_JSON"):

        result = agent.generate_persona("Product X")

        assert result["error"] == "Invalid JSON response from agent."
        assert isinstance(result["recommended_channels"], list)
        assert result["recommended_channels"] == []


def test_generate_persona_empty_response(agent):
    with patch("backend.agents.persona_agent.invoke", return_value=None):

        result = agent.generate_persona("Product X")

        assert result["error"] == "Empty LLM response"
        assert result["persona_name"] == ""
        assert result["recommended_channels"] == []
