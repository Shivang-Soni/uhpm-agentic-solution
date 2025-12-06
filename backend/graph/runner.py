import asyncio
import logging
from typing import Any, Dict

from graph.uhpm_graph import create_uhpm_graph, GraphState

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy load graph app
_graph_app = None


def _get_graph_app():
    global _graph_app
    if _graph_app is None:
        logger.info("Initialising UHPM Langgraph app...")
        _graph_app = create_uhpm_graph()
    return _graph_app


async def run_graph(input_dict: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    """
    Runs the UHPM Langgraph pipeline asynchronously with a timeout.

    Args:
    - input_dict
    - timeout

    Returns:
    - The final Langgraph state as a plain dict.
    """
    if "task" not in input_dict:
        raise ValueError("Missing required field 'task'")
    app = _get_graph_app()

    # ensure GraphState shape
    state = GraphState(input_dict)

    # run graph with a timeout
    try:
        coro = app.ainvoke(state)
        result_state = await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error("Graph execution timed out")
        raise
    except Exception as e:
        logger.exception(f"Graph execution failed: {e}")
        raise

    return dict(result_state)
