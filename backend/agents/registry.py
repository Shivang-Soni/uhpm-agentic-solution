from typing import Dict, Type

from actions import Action
from agents.base_agent import BaseAgent


class AgentRegistry:
    """
    Single source of truth for agent resolution.
    """

    def __init__(self):
        self._agents: Dict[Action, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        if agent.action in self._agents:
            raise ValueError(f"Agent already registered for action {agent.action}")

        self._agents[agent.action] = agent

    def get(self, action: Action) -> BaseAgent:
        if action not in self._agents:
            raise KeyError(
                f"No agent registered for action: {action}"
            )

        return self._agents[action]

    def list_actions(self):
        return list(self._agents.keys())
