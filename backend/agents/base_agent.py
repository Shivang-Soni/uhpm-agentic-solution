from abc import ABC, abstractmethod
from typing import Dict, Any, List

from actions import Action
from agents.schemas import CampaignState, ExecutionResult


class BaseAgent(ABC):
    """
    Base class for all agents.
    """

    action: Action

    def __init_subclass__(cls):
        super().__init_subclass__()

        if not hasattr(cls, "action"):
            raise TypeError(
                f"{cls.__name__} must define class attribute 'action'"
                )

        if not isinstance(cls.action, Action):
            raise TypeError(
                f"{cls.__name__}.action must be instance of Action enum"
            )

    @abstractmethod
    def execute(
        self,
        state: CampaignState,
        reflection: List[Dict] | None = None
    ) -> ExecutionResult:
        """
        Execute exactly one action.

        Must NOT mutate state.
        Must return ExecutionResult.
        """
        pass

    def reflect(self, state: CampaignState) -> List[Dict]:
        """
        Extract past reflection entries for this agent action.
        """
        reflections = state.get("self_reflection", [])

        return [
            r for r in reflections
            if r.get("action") == self.action.value
        ]

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
