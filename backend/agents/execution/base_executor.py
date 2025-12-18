from typing import Dict, Any, Callable
import logging

logging = logging.getLogger(__name__)


class BaseExecutor:
    """
    Base executor for running publish jobs.
    """
    def execute(
            self,
            job: Callable[..., Dict[str, Any]],
            payload: Dict[str, Any],
    ) - Dict[str, Any]:
        raise NotImplementedError("execute() must be implemented.")