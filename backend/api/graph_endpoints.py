from fastapi import APIRouter
from graph.basic_graph import build_basic_graph


router = APIRouter()


@router.get("/run-basic-graph")
def run_basic_graph():
    """
    Endpoint which runs the graph and return its result.
    """
    app = build_basic_graph()
    result = app.invoke({})
    return {"result": result}