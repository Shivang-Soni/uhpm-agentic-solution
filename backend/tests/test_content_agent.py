import json
import pytest
from unittest.mock import patch

from backend.agents.content_agent import ContentAgent


@pytest.fixture
def agent():
    return ContentAgent()


def test_generate_content_success(agent):
    fake_llm_response = json.dumps({
        "product_text": "prod",
        "persona_text": "persona",
        "channel": "social_media",
        "tone": "professional",
        "variants": [
            {
                "headline": "Great post",
                "primary_text": "This is a great product",
                "cta": "Buy now"
            }
        ],
        "metadata": {}
    })

    with patch("backend.agents.content_agent.invoke", return_value=fake_llm_response) as mock_invoke, \
         patch("backend.agents.content_agent.add_document") as mock_add:

        result = agent.generate_content("prod", "persona")

        assert result["channel"] == "social_media"
        assert result["tone"] == "professional"
        assert len(result["variants"]) == 1
        assert result["variants"][0]["headline"] == "Great post"

        mock_invoke.assert_called_once()
        mock_add.assert_called_once()


def test_generate_content_invalid_json(agent):
    with patch("backend.agents.content_agent.invoke", return_value="NOT JSON"), \
         patch("backend.agents.content_agent.add_document") as mock_add:

        result = agent.generate_content("prod", "persona")

        # Fallback greift
        assert result["metadata"]["fallback"] is True
        assert len(result["variants"]) == 1
        assert "headline" in result["variants"][0]

        mock_add.assert_called_once()


def test_generate_content_no_response(agent):
    with patch("backend.agents.content_agent.invoke", return_value=None), \
         patch("backend.agents.content_agent.add_document") as mock_add:

        result = agent.generate_content("prod", "persona", channel="email")

        assert result["channel"] == "email"
        assert result["metadata"]["fallback"] is True
        assert len(result["variants"]) == 1

        mock_add.assert_called_once()
