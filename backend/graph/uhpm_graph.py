import logging
from typing import Dict, Any, List
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agents.planner_agent import PlannerAgent
from agents.reasoner import ReasonerAgent
from agents.execution_context_agent import ExecutionContextAgent
from agents.agent_runner import AgentRunner
from agents.dispatcher import Dispatcher

from agents.research_agent import ResearchAgent
from agents.persona_agent import PersonaAgent
from agents.content_agent import ContentAgent
from agents.experiment_agent import ExperimentationAgent
from agents.analytics_agent import AnalyticsAgent
from agents.whatsapp_agent import WhatsappAgent
from agents.retriever_agent import RetrieverAgent

from agents.adapters.channel_adapter_registry import ChannelAdapterRegistry
from agents.adapters.channel_adapter_dispatcher import ChannelAdapterDispatcher
from agents.repositories.in_memory_campaign_repository import (
    InMemoryCampaignRepository
)
from vectorstore.store import add_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# Graph State Definition
# =========================
class GraphState(TypedDict, total=False):
    # Core
    task: str
    plan: Dict[str, Any]
    reasoning: Dict[str, Any]

    # Memory
    memory_context: List[Dict[str, Any]]

    # Execution
    execution_plan: List[Dict[str, Any]]
    execution_results: List[Dict[str, Any]]
    last_result: Dict[str, Any]

    # Control
    replanned: bool

    # Channel outputs
    channel: str
    artifacts: Dict[str, Any]
    channel_result: Dict[str, Any]


# =========================
# Agent Initialization
# =========================

# Retriever exists but is not yet used by Planner / Reasoner directly
retriever = None

planner_agent = PlannerAgent()
reasoner_agent = ReasonerAgent(retriever)
execution_context_agent = ExecutionContextAgent()

research_agent = ResearchAgent()
persona_agent = PersonaAgent()
content_agent = ContentAgent()
experiment_agent = ExperimentationAgent()
analytics_agent = AnalyticsAgent()
whatsapp_agent = WhatsappAgent()
retriever_agent = RetrieverAgent()

channel_registry = ChannelAdapterRegistry()
repository = InMemoryCampaignRepository()
channel_dispatcher = ChannelAdapterDispatcher(channel_registry, repository)

dispatcher = Dispatcher(
    research_agent=research_agent,
    persona_agent=persona_agent,
    content_agent=content_agent,
    experiment_agent=experiment_agent,
    analytics_agent=analytics_agent,
    whatsapp_agent=whatsapp_agent,
    channel_adapter_dispatcher=channel_dispatcher,
)

execution_runner = AgentRunner(dispatcher)


# =========================
# Memory Retrieval Node
# =========================
def memory_retrieval_node(state: GraphState) -> GraphState:
    logging.info("Memory retrieval node started.")

    task = state.get("task", "")
    if not task:
        return state

    # Retrieve semantically similar past executions
    results = retriever_agent.search_docs(query=task, top_k=3)
    state["memory_context"] = results

    return state


# =========================
# Graph Nodes
# =========================
def planner_node(state: GraphState) -> GraphState:
    logger.info("Planner node started")

    state["plan"] = planner_agent.plan(state.get("task", ""))

    return state


def reason_node(state: GraphState) -> GraphState:
    logger.info("Reasoner node started")

    state["reasoning"] = reasoner_agent.decide(
        state.get("task", ""), state.get("memory_context", [])
        )

    return state


def execution_context_node(state: GraphState) -> GraphState:
    logger.info("Execution context node started")

    execution_plan = execution_context_agent.build_execution_plan(
        plan=state.get("plan", {}),
        social_context=state.get("reasoning", {}),
        user_payload=state,
    )

    state["execution_plan"] = execution_plan
    return state


def execution_runner_node(state: GraphState) -> GraphState:
    logger.info("Execution runner node started")

    # Execute steps sequentially and update last_result internally
    updated_state = execution_runner.run(
        state=state,
        execution_plan=state.get("execution_plan", []),
    )

    return updated_state


def memory_node(state: GraphState) -> GraphState:
    logger.info("Memory node started")

    # Persist full execution trace for future retrieval
    add_document(
        text=str(state),
        metadata={
            "type": "uhpm_graph_run",
            "has_error": "error" in str(state).lower(),
        },
    )

    return state


# =========================
# Failure Routing
# =========================
def failure_router(state: GraphState) -> str:
    last = state.get("last_result")

    if not last:
        return "memory"

    if last.get("success") is False and not state.get("replanned", False):
        logger.warning("Failure detected: triggering replan.")
        state["replanned"] = True
        return "planner"

    return "memory"


# =========================
# Graph Construction
# =========================
def create_uhpm_graph(checkpointer=None):
    graph = StateGraph(GraphState)

    graph.add_node("memory_retrieval", memory_retrieval_node)
    graph.add_node("planner", planner_node)
    graph.add_node("reason", reason_node)
    graph.add_node("execution_context", execution_context_node)
    graph.add_node("execution_runner", execution_runner_node)
    graph.add_node("memory", memory_node)

    # Memory-first entry to enable context-aware planning
    graph.set_entry_point("memory_retrieval")

    graph.add_edge("memory_retrieval", "planner")
    graph.add_edge("planner", "reason")
    graph.add_edge("reason", "execution_context")
    graph.add_edge("execution_context", "execution_runner")

    graph.add_conditional_edges(
        "execution_runner",
        failure_router,
        {
            "planner": "planner",
            "memory": "memory"
        }
    )

    graph.add_edge("memory", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
