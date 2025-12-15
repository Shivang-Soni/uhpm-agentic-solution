import logging
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Agents
from agents.planner_agent import PlannerAgent
from agents.reasoner import ReasonerAgent
from agents.dispatcher import Dispatcher
from agents.research_agent import ResearchAgent
from agents.persona_agent import PersonaAgent
from agents.content_agent import ContentAgent
from agents.experiment_agent import ExperimentationAgent
from agents.analytics_agent import AnalyticsAgent
from agents.retriever_agent import RetrieverAgent

from vectorstore.store import add_document

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    """
    Shared state between nodes.
    """
    configurable: dict
    task: str
    plan: dict | Any
    reasoning: dict | str | Any
    agent_output: dict | str | Any


# Agent initialisation
retriever = RetrieverAgent()

planner_agent = PlannerAgent()
reasoner = ReasonerAgent(retriever)
research_agent = ResearchAgent()
persona_agent = PersonaAgent()
content_agent = ContentAgent()
experiment_agent = ExperimentationAgent()
analytics_agent = AnalyticsAgent()

dispatcher = Dispatcher(
    research_agent,
    persona_agent,
    content_agent,
    experiment_agent,
    analytics_agent
)

memory = MemorySaver()


def planner_node(state: GraphState):
    """Generate structured plan via PlannerAgent."""
    logger.info("[planner_node] Start")

    state.setdefault("configurable", {})
    user_task = state.get("task", "")

    plan_obj = planner_agent.plan(user_task)
    state["plan"] = plan_obj

    logger.info("[planner_node] Completed")
    return state


def reason_node(state: GraphState):
    """Run reasoning step."""
    logger.info("[reason_node] Start")

    user_task = state.get("task", "")
    reasoning = reasoner.decide(user_task)
    state["reasoning"] = reasoning

    logger.info("[reason_node] Completed")
    return state


def dispatch_node(state: GraphState):
    """Dispatcher chooses correct agent(s)."""
    logger.info("[dispatch_node] Start")

    result = dispatcher.run(
        plan=state.get("plan"),
        reason_output=state.get("reasoning"),
        user_payload=state
    )

    state["agent_output"] = result

    logger.info("[dispatch_node] Completed")
    return state


def write_memory_node(state: GraphState):
    """Persist data to vector store and memory."""
    logger.info("[memory_node] Start")

    payload = {
        "task": state.get("task"),
        "plan": state.get("plan"),
        "reasoning": state.get("reasoning"),
        "agent_output": state.get("agent_output")
    }

    agent_output = state.get("agent_output", {})

    has_error = (
        isinstance(agent_output, dict)
        and "error" in agent_output
    )

    add_document(
                text=str(payload),
                metadata={
                    "type": "graph_run",
                    "has_error": has_error
                }
                )

    logger.info("[memory_node] Completed")
    return state


def create_uhpm_graph(checkpointer=None):
    """
    Creates the UHPM LangGraph with optional checkpointing.

    Args:
        checkpointer: Optional LangGraph checkpointer instance.

    Returns:
        Compiled graph ready to run.
    """
    graph = StateGraph(GraphState)

    # Nodes
    graph.add_node("planner", planner_node)
    graph.add_node("reason", reason_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("memory", write_memory_node)

    # Entry point
    graph.set_entry_point("planner")

    # Edges
    graph.add_edge("planner", "reason")
    graph.add_edge("reason", "dispatch")
    graph.add_edge("dispatch", "memory")
    graph.add_edge("memory", END)

    # Compile graph with checkpointer
    return graph.compile(checkpointer=checkpointer)
