from langgraph.graph import StateGraph, END
import logging

# Initialise Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define state
class State(dict):
    message: str


# Create Node functions
def node_a(state: State):
    logging.info("Executing Node A.")
    return {"message": "Hello from Node A"}


def node_b(state: State):
    logging.info("Executing Node B.")
    return {"message": state["message"] + " -> directed to Node B"}


# Define Graph
def build_basic_graph():
    graph = StateGraph(State)
    graph.add_node("step_a", node_a)
    graph.add_node("step_b", node_b)

    # Define transitions
    graph.set_entry_point("step_a")
    graph.add_edge("step_a", "step_b")
    graph.add_edge("step_b", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_basic_graph()
    result = app.invoke({})
    logging.info(f"Graph execution result: {result}")