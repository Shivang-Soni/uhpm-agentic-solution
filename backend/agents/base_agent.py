from abc import ABC, abstractmethod
from typing import Dict, Any

from schemas import CampaignState, ExecutionResult
from actions import Action


class BaseAgent(ABC):
    """
    Reads from CampaignState, updates CampaignState
    """

    action: Action

    @abstractmethod
    def execute(self, state: CampaignState) -> ExecutionResult:
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
