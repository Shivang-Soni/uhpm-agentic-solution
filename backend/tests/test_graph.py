import pytest
from unittest.mock import patch
from backend.graph import runner


@pytest.mark.asyncio
async def test_full_uhpm_graph_integration():
    runner._graph_app = None  # Reset lazy-loaded graph

    user_task = "Create a marketing campaign for Product X"

    fake_plan = {
        "task": user_task,
        "needs_research": False,
        "needs_persona": True,
        "needs_content": False,
        "needs_experimentation": False,
        "needs_analytics": False,
        "additional_context": ""
    }

    fake_reason = {
        "task_type": "persona",
        "reasoning": "Persona creation required",
        "action": "call_persona_agent",
        "inputs_needed": {
            "product_text": "Required",
            "market_text": "Required"
        }
    }

    fake_persona = {
        "persona_name": "Young Professional",
        "summary": "Tech-savvy early adopter"
    }

    with patch("backend.graph.uhpm_graph.PlannerAgent") as MockPlanner, \
         patch("backend.graph.uhpm_graph.ReasonerAgent") as MockReasoner, \
         patch("backend.graph.uhpm_graph.PersonaAgent") as MockPersona, \
         patch("vectorstore.store.add_document") as mock_add:

        MockPlanner.return_value.plan.return_value = fake_plan
        MockReasoner.return_value.decide.return_value = fake_reason
        MockPersona.return_value.generate_persona.return_value = fake_persona

        result = await runner.run_graph({"task": user_task}, timeout=10)

        # Assertions
        assert "plan" in result
        assert "reasoning" in result
        assert "agent_output" in result
        assert result["agent_output"] == fake_persona
        mock_add.assert_called_once()
