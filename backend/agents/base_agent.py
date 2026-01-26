from abc import ABC, abstractmethod
from typing import Dict, Any

from actions import Action
from agents.schemas import CampaignState, ExecutionResult


class BaseAgent(ABC):
    """
    Base class for all agents.

    Contract:
    - Every agent MUST define class attribute: action: Action
    - Agent NEVER mutates state directly
    - Agent returns ExecutionResult only
    - Dispatcher owns state mutation
    """

    action: Action

    def __init_subclass__(cls):
        """
        Enforce that every subclass defines a valid Action.
        Fails fast at import time.
        """
        super().__init_subclass__()

        if not hasattr(cls, "action"):
            raise TypeError(f"{cls.__name__} must define class attribute 'action'")

        if not isinstance(cls.action, Action):
            raise TypeError(
                f"{cls.__name__}.action must be instance of Action enum"
            )

    @abstractmethod
    def execute(self, state: CampaignState) -> ExecutionResult:
        """
        Execute exactly one action.

        Must NOT mutate state.
        Must return ExecutionResult.
        """
        pass

    def _success(self, data: Dict[str, Any] | None = None) -> ExecutionResult:
        return ExecutionResult(
            action=self.action.value,
            success=True,
            data=data or {}
        )

    def _failure(self, error: str) -> ExecutionResult:
        return ExecutionResult(
            action=self.action.value,
            success=False,
            error=error
        )