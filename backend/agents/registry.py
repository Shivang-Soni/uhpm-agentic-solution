from typing import Dict

from actions import Action
from agents.base_agent import BaseAgent


class AgentRegistry:
    """
    Single source of truth for agent resolution.
    """

    def __init__(self):
        self._agents: Dict[Action, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        if not isinstance(agent, BaseAgent):
            raise TypeError("Only BaseAgent subclasses may be registered")

        if agent.action is None:
            raise ValueError("Agent.action must not be None")

        if agent.action in self._agents:
            raise ValueError(
                f"Agent already registered for action {agent.action}"
            )

        self._agents[agent.action] = agent

    def get(self, action: Action) -> BaseAgent:
        if action not in self._agents:
            available = ", ".join(a.value for a in self._agents.keys())
            raise KeyError(
                f"No agent registered for action: {action}. "
                f"Available actions: [{available}]"
            )

        return self._agents[action]

    def list_actions(self):
        return [a.value for a in self._agents.keys()]