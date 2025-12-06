from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import agents
from agents.reasoner import ReasonerAgent
from agents.dispatcher import Dispatcher
from agents.research_agent import ResearchAgent
from agents.persona_agent import PersonaAgent
from agents.content_agent import ContentAgent
from agents.experiment_agent import ExperimentationAgent
from agents.analytics_agent import AnalyticsAgent

# memory
from vectorstore.store import add_document


class GraphState(dict):
    """
    Shared state structure between nodes
    Everything the agents generate flows through here.
    """
    pass

# Initialise Agents
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


def reason_node(state: GraphState):
    task = state.get("task", "")
    reasoning = reasoner.decide(task)
    state["reasoning"] = reasoning
    return state


def dispatch_node(state: GraphState):
    result = dispatcher.run(
        reason_output=state["reasoning"],
        user_payload=state
    )
    state["agent_output"] = result
    return state


def write_memory_node(state: GraphState):
    payload = {
        "task": state.get("task"),
        "reason": state.get("reasoning"),
        "output": state.get("agent_output")
    }
    add_document(str(payload))
    return state


def create_uhpm_graph():
    graph = StateGraph(GraphState)

    graph.add_node("reason", reason_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("memory", write_memory_node)

    # edges
    graph.set_entry_point("reason")
    graph.add_edge("reason", "dispatch")
    graph.add_edge("dispatch", "memory")
    graph.add_edge("memory", END)

    # compile with memory
    app = graph.compile(checkpointer=memory)
    return app
