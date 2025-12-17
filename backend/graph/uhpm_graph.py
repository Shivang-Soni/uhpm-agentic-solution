import logging
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agents.planner_agent import PlannerAgent
from agents.reasoner import ReasonerAgent
from agents.dispatcher import Dispatcher

from agents.research_agent import ResearchAgent
from agents.persona_agent import PersonaAgent
from agents.content_agent import ContentAgent
from agents.experiment_agent import ExperimentationAgent
from agents.analytics_agent import AnalyticsAgent
from agents.whatsapp_agent import WhatsappAgent

from agents.adapters.channel_adapter_registry import ChannelAdapterRegistry
from agents.adapters.channel_adapter_dispatcher import ChannelAdapterDispatcher

from vectorstore.store import add_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    task: str
    plan: dict
    reasoning: dict
    agent_output: dict

    channel: str
    artifacts: dict
    channel_result: dict


# Agents
retriever = None
planner_agent = PlannerAgent()
reasoner = ReasonerAgent(retriever)

research_agent = ResearchAgent()
persona_agent = PersonaAgent()
content_agent = ContentAgent()
experiment_agent = ExperimentationAgent()
analytics_agent = AnalyticsAgent()
whatsapp_agent = WhatsappAgent()

channel_registry = ChannelAdapterRegistry()
channel_dispatcher = ChannelAdapterDispatcher(channel_registry)

dispatcher = Dispatcher(
    research_agent,
    persona_agent,
    content_agent,
    experiment_agent,
    analytics_agent,
    whatsapp_agent,
    channel_adapter_dispatcher=channel_dispatcher
)


def planner_node(state: GraphState):
    state["plan"] = planner_agent.plan(state.get("task", ""))
    return state


def reason_node(state: GraphState):
    state["reasoning"] = reasoner.decide(state.get("task", ""))
    return state


def dispatch_node(state: GraphState):
    result = dispatcher.run(
        state=state,
        reason_output=state.get("reasoning", {}),
        user_payload=state,
        plan=state.get("plan")
    )

    if "channel_result" in result:
        state["channel_result"] = result["channel_result"]
    else:
        state["agent_output"] = result

    return state


def memory_node(state: GraphState):
    add_document(
        text=str(state),
        metadata={
            "type": "uhpm_graph_run",
            "has_error": "error" in str(state).lower()
        }
    )
    return state


def create_uhpm_graph(checkpointer=None):
    graph = StateGraph(GraphState)

    graph.add_node("planner", planner_node)
    graph.add_node("reason", reason_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("memory", memory_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "reason")
    graph.add_edge("reason", "dispatch")
    graph.add_edge("dispatch", "memory")
    graph.add_edge("memory", END)

    return graph.compile(checkpointer=checkpointer)
