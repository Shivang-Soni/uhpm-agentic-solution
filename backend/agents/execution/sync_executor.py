from typing import Dict, Callable, Any
import logging

from agents.execution.base_executor import BaseExecutor

logger = logging.getLogger(__name__)


class SyncExecutor(BaseExecutor):
    """
    A temporary synchronous executor.
    Act as a placeholder for async execution. 
    """

    def execute(
            self,
            job: Callable[..., Dict[str, Any]],
            payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        logging.info("Sync Executor: executing job synchronously.")
        return job(payload)
