import logging
from typing import Dict

from actions import Action
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Explicit agent registry.
    All agents must be registered manually (via bootstrap_runtime).
    """

    def __init__(self):
        self._agents: Dict[Action, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        action = agent.action

        if action in self._agents:
            raise RuntimeError(
                f"Duplicate agent for action {action.value}: "
                f"{agent.__class__.__name__}"
            )

        self._agents[action] = agent

        logger.info(
            f"Registered agent {agent.__class__.__name__} for action {action.value}"
        )

    def validate(self) -> None:
        missing = []
        for action in Action:
            if action not in self._agents:
                missing.append(action.value)

        if missing:
            raise RuntimeError(
                f"Missing agents for actions: {missing}"
            )

        logger.info("AgentRegistry validation successful")

    def get(self, action: Action) -> BaseAgent:
        return self._agents[action]

    def list_actions(self):
        return [a.value for a in self._agents.keys()]
