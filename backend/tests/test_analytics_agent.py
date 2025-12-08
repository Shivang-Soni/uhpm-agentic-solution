import json
import pytest
from unittest.mock import patch, MagicMock

from backend.agents.analytics_agent import AnalyticsAgent


@pytest.fixture
def agent():
    return AnalyticsAgent()


def test_campaign_success(agent):
    fake_json = {
        "summary": "good",
        "persona_changes": ["more_detail"],
        "content_improvements": ["improve CTA"],
        "channel_recommendations": ["TikTok"],
        "next_steps": ["A/B test"]
    }

    with patch("backend.agents.analytics_agent.invoke", return_value=json.dumps(fake_json)) \
        as mock_invoke, \
            patch("backend.agents.analytics_agent.add_document") as mock_add:
        result = agent.analyse_campaign("test data")

        assert result == fake_json
        mock_invoke.assert_called_once()
        mock_add.assert_called_once()


def test_analyse_campaign_no_response(agent):
    with patch("backend.agents.analytics_agent.invoke", return_value=None):

        result = agent.analyse_campaign("test data")

        assert result["error"] == "No response from Agent"
        assert result["summary"] == ""
        assert isinstance(result["persona_changes"], list)


def test_analyse_campaign_invalid_json(agent):
    with patch("backend.agents.analytics_agent.invoke", return_value="NOT JSON"):

        result = agent.analyse_campaign("test data")

        assert result["error"] == "Invalid JSON from Agent"
        assert result["summary"] == ""
        assert isinstance(result["content_improvements"], list)
