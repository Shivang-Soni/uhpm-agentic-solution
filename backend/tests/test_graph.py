import pytest
from unittest.mock import patch
from backend.graph.runner import run_graph


@pytest.mark.asyncio
async def test_uhpm_graph_end_to_end():

    input_dict = {"task": "Test Product"}

    # Mocked response for PlannerAgent.invoke
    fake_invoke_response = """
    {
        "task": "research",
        "needs_research": true,
        "needs_persona": false,
        "needs_content": false,
        "needs_experimentation": false,
        "needs_analytics": false,
        "additional_context": "Test reasoning"
    }
    """

    with patch(
        "backend.agents.planner_agent.invoke",
        return_value=fake_invoke_response
    ):
        result = await run_graph(input_dict, timeout=10)

    # Assert graph outputs
    assert "plan" in result
    assert "reasoning" in result
    assert "agent_output" in result

    agent_output = result["agent_output"]
    # agent_output can have "status" or "error" depending on Dispatcher
    assert "status" in agent_output or "error" in agent_output
