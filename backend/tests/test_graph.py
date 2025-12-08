import pytest
from unittest.mock import patch

from backend.graph.runner import run_graph


@pytest.mark.asyncio
async def test_uhpm_graph_end_to_end():

    input_dict = {"task": "Test Product"}

    fake_invoke_response = '{"task_type": "research", "reasoning": "Test reasoning", "action": "call_research_agent", "inputs_needed": ["product_text"]}'

    with patch("backend.llm.gemini_pipeline.invoke", return_value=fake_invoke_response):
        result = await run_graph(input_dict, timeout=10)

    assert "plan" in result
    assert "reasoning" in result
    assert "agent_output" in result

    agent_output = result["agent_output"]
    assert "status" in agent_output or "error" in agent_output
