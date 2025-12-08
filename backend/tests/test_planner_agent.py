def test_planner_agent_executes_json(planner):
    user_task = "Generate a marketing plan for a skincare product."
    plan = planner.plan(user_task)

    assert isinstance(plan, dict)
    assert "task" in plan
    assert "needs_research" in plan
    assert "needs_persona" in plan
    assert "needs_content" in plan
    assert "needs_experimentation" in plan
    assert "needs_analytics" in plan
