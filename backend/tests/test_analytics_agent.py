import json
import pytest
from unittest.mock import patch

from agents.analytics_agent import AnalyticsAgent
from agents.schemas import AnalyticsOutput


@pytest.fixture
def agent():
    return AnalyticsAgent()


def test_analyse_campaign_success(agent):
    fake_json = {
        "summary": "Campaign performed well",
        "persona_changes": ["Focus more on price-sensitive users"],
        "content_improvements": ["Improve CTA clarity"],
        "channel_recommendations": ["TikTok"],
        "next_steps": ["Run A/B test"]
    }

    with patch(
        "agents.analytics_agent.invoke",
        return_value=json.dumps(fake_json)
    ), patch(
        "agents.analytics_agent.add_document"
    ) as mock_add:

        result = agent.analyse_campaign("test data")

        # Schema-level correctness
        validated = AnalyticsOutput(**result)
        assert validated.summary == "Campaign performed well"
        assert "TikTok" in validated.channel_recommendations

        # Side-effect check
        mock_add.assert_called_once()


def test_analyse_campaign_empty_llm_response(agent):
    with patch(
        "agents.analytics_agent.invoke",
        return_value=None
    ):
        result = agent.analyse_campaign("test data")

        # Still valid schema
        validated = AnalyticsOutput(**result)

        assert "Analytics unavailable" in validated.summary
        assert len(validated.persona_changes) > 0
        assert len(validated.content_improvements) > 0
        assert len(validated.channel_recommendations) > 0
        assert len(validated.next_steps) > 0


def test_analyse_campaign_invalid_json(agent):
    with patch(
        "agents.analytics_agent.invoke",
        return_value="NOT JSON"
    ):
        result = agent.analyse_campaign("test data")

        validated = AnalyticsOutput(**result)

        assert "Analytics unavailable" in validated.summary
        assert isinstance(validated.persona_changes, list)
        assert isinstance(validated.content_improvements, list)
        assert isinstance(validated.channel_recommendations, list)
        assert isinstance(validated.next_steps, list)
