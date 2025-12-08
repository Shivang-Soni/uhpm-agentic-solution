import pytest
from unittest.mock import patch
import asyncio

from backend.graph.runner import run_graph


@pytest.mark.asyncio
async def test_full_uhpm_graph_integration():
    """
    Integration test for the full UHPM graph.
    Simulates a complete marketing task flow.
    """

    user_task = "Create a marketing campaign for Product X targeting young professionals"

    fake_llm_response = '{"task_type": "persona", "reasoning": "Task requires persona creation", "action": "call_persona_agent", "inputs_needed": ["product_text", "market_text"]}'
    fake_content_response = '{"text": "Generated marketing content"}'
    fake_research_response = '{"product_summary": "Summary", "usps": ["USP1"], "target_audience": ["Young professionals"], "competitor_comparision": "Competitor analysis"}'
    fake_experiment_response = '[{"variant": "Ad1", "score": 80, "reason": "Good fit"}]'
    fake_analytics_response = '{"summary": "Campaign did well", "persona_changes": [], "content_improvements": [], "channel_recommendations": [], "next_steps": []}'

    with patch("backend.agents.planner_agent.invoke", return_value=fake_llm_response), \
         patch("backend.agents.reasoner.invoke", return_value=fake_llm_response), \
         patch("backend.agents.research_agent.invoke", return_value=fake_research_response), \
         patch("backend.agents.persona_agent.invoke", return_value=fake_llm_response), \
         patch("backend.agents.content_agent.invoke", return_value=fake_content_response), \
         patch("backend.agents.experiment_agent.invoke", return_value=fake_experiment_response), \
         patch("backend.agents.analytics_agent.invoke", return_value=fake_analytics_response), \
         patch("backend.vectorstore.store.add_document") as mock_add:

        result = await run_graph({"task": user_task}, timeout=10)

        # Basic assertions
        assert "plan" in result
        assert "reasoning" in result
        assert "agent_output" in result

        # Ensure Memory node ran
        mock_add.assert_called()

        # Check that agent_output contains at least one key
        assert isinstance(result["agent_output"], dict)
