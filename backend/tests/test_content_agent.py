import pytest
from unittest.mock import patch

from agents.content_agent import ContentAgent


@pytest.fixture
def agent():
    return ContentAgent()


def test_generate_content_success(agent):
    fake_json = {"text": "Great post", "cta": "Buy now!"}

    with patch("backend.agents.content_agent.invoke", return_value=fake_json) \
        as mock_invoke, \
            patch("backend.agents.content_agent.add_document") as mock_add:

        result = agent.generate_content("prod", "persona")

        assert result["content"] == fake_json
        assert result["channel"] == "social_media"
        mock_invoke.assert_called_once()
        mock_add.assert_called_once()


def test_generate_content_invalid_json(agent):
    invalid_response = "NOT_JSON"

    with patch("backend.agents.content_agent.invoke", return_value=invalid_response), \
            patch("backend.agents.content_agent.add_document") as mock_add:

        result = agent.generate_content("prod", "persona")

        assert result["content"] == {"text": invalid_response}
        mock_add.assert_called_once()


def test_generate_content_no_response(agent):
    with patch("backend.agents.content_agent.invoke", return_value=None):
        result = agent.generate_content("prod", "persona", "email")

        assert result["content"] is None
        assert result["channel"] == "email"
