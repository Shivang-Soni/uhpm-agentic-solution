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

from vectorstore.store import add_document

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    """Shared state between nodes."""
    task: str
    plan: dict | str | Any
    reasoning: dict | str | Any
    agent_output: dict | str | Any


# Agent initialisation
planner_agent = PlannerAgent()
reasoner = ReasonerAgent()
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
    user_task = state.get("task", "")

    plan_obj = planner_agent.plan(user_task)
    state["plan"] = plan_obj if isinstance(plan_obj, dict) else plan_obj.dict()

    logger.info("[planner_node] Completed")
    return state


def reason_node(state: GraphState):
    """Run reasoning step."""
    logger.info("[reason_node] Start")

    plan_dict = state.get("plan", {})
    reasoning = reasoner.decide(plan_dict)
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
    """Persist data to vector store."""
    logger.info("[memory_node] Start")

    payload = {
        "task": state.get("task"),
        "plan": state.get("plan"),
        "reason": state.get("reasoning"),
        "output": state.get("agent_output")
    }

    add_document(str(payload))

    logger.info("[memory_node] Completed")
    return state


def create_uhpm_graph():
    graph = StateGraph(GraphState)

    graph.add_node("planner", planner_node)
    graph.add_node("reason", reason_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("memory", write_memory_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "reason")
    graph.add_edge("reason", "dispatch")
    graph.add_edge("dispatch", "memory")
    graph.add_edge("memory", END)

    return graph.compile(checkpointer=memory)
