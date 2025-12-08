def test_graph_runs(graph_app):
    state = {"task": "Write a social media add for a new fitness app."}

    result = graph_app.invoke(state)

    assert "plan" in result
    assert "reasoning" in result
    assert "agent_output" in result
    
    assert isinstance(result["plan"], dict)