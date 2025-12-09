import pytest
from unittest.mock import patch
from backend.graph import runner


@pytest.mark.asyncio
async def test_full_uhpm_graph_integration():

    runner._graph_app = None

    user_task = "Create a marketing campaign for Product X"

    fake_plan = {"task_type": "persona", "action": "call_persona_agent"}
    fake_reason = {
        "task_type": "persona",
        "action": "call_persona_agent",
        "inputs_needed": ["product_text", "market_text"]
    }
    fake_research = {"product_summary": "Summary"}
    fake_persona = {"persona": "Young Professional Persona"}
    fake_content = {"text": "Marketing Content"}
    fake_experiment = {"variants": [{"variant": "A", "score": 90}]}
    fake_analytics = {"summary": "Good campaign"}


    with patch("backend.graph.uhpm_graph.planner_agent.plan", return_value=fake_plan), \
         patch("backend.graph.uhpm_graph.reasoner.decide", return_value=fake_reason), \
         patch("backend.graph.uhpm_graph.research_agent.analyse_product", return_value=fake_research), \
         patch("backend.graph.uhpm_graph.persona_agent.generate_persona", return_value=fake_persona), \
         patch("backend.graph.uhpm_graph.content_agent.generate_content", return_value=fake_content), \
         patch("backend.graph.uhpm_graph.experiment_agent.score_variants", return_value=fake_experiment), \
         patch("backend.graph.uhpm_graph.analytics_agent.analyse_campaign", return_value=fake_analytics), \
         patch("vectorstore.store.add_document") as mock_add:

        result = await runner.run_graph({"task": user_task}, timeout=10)

        assert "plan" in result
        assert "reasoning" in result
        assert "agent_output" in result
        assert isinstance(result["agent_output"], dict)

        mock_add.assert_called_once()
