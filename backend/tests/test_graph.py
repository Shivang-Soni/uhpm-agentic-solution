import pytest
from unittest.mock import patch

from backend.graph.runner import run_graph

@pytest.mark.asyncio
async def test_uhpm_graph_end_to_end():
    input_dict = {"task": "Test Product"}

    fake_planner_response = {
        "task": "Test Product",
        "needs_research": True,
        "needs_persona": True,
        "needs_content": True,
        "needs_experimentation": False,
        "needs_analytics": False,
        "additional_context": "Mocked plan"
    }

    fake_reasoner_response = {"reasoning": "Mocked reasoning"}

    fake_dispatcher_response = {"status": "success", "details": "Mocked agent output"}

    with patch("backend.agents.planner_agent.invoke", return_value=fake_planner_response), \
         patch("backend.agents.reasoner.invoke", return_value=fake_reasoner_response), \
         patch("backend.agents.research_agent.invoke", return_value="Research done"), \
         patch("backend.agents.persona_agent.invoke", return_value={
             "persona_name": "Mock Persona",
             "recommended_channels": ["Email"],
             "age_range": "",
             "demographics": "",
             "lifestyle": "",
             "deep_motivations": "",
             "pain_points": "",
             "buying_triggers": "",
             "objections": "",
             "language_and_tone": "",
             "summary": ""
         }), \
         patch("backend.agents.content_agent.invoke", return_value="Content generated"), \
         patch("backend.agents.experiment_agent.invoke", return_value="Experiments created"), \
         patch("backend.agents.analytics_agent.invoke", return_value="Analytics computed"):

        result = await run_graph(input_dict, timeout=30)

    assert "plan" in result
    assert "reasoning" in result
    assert "agent_output" in result

    agent_output = result["agent_output"]
    assert isinstance(agent_output, dict)
    assert "status" in agent_output or "error" in agent_output
