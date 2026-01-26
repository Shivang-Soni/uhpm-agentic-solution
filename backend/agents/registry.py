import inspect
import logging
from typing import Dict, Type

from actions import Action
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Single source of truth for agent resolution.
    """

    def __init__(self):
        self._agents: Dict[Action, BaseAgent] = {}
        self._discover_agents()
        self._validate()

    def _discover_agents(self) -> None:
        """
        Automatically find all subclasses of BaseAgent and register them.
        """
        for cls in BaseAgent.__subclasses__():
            if inspect.isabstract(cls):
                continue

            try:
                instance = cls()
            except TypeError as e:
                raise RuntimeError(
                    f"Agent {cls.__name__} must have parameterless constructor"
                    f"or be manually injected"
                ) from e

            self._register(instance)

    def _register(self, agent: BaseAgent) -> None:
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

    def _validate(self) -> None:
        """
        Ensure every Action enum has exactly one Agent.
        """
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
