from fastapi import APIRouter

from api.graph_endpoints import router as graph_router
from api.vector_endpoints import router as vectordb_router
from api.research_endpoints import router as research_router
from api.reasoning_routes import router as reasoning_router

router = APIRouter()


@router.get("/health")
def health_check():
    return {"ok": True}


router.include_router(graph_router, prefix="/graph")
router.include_router(vectordb_router, prefix="/vectordb")
router.include_router(research_router, prefix="/research")
router.include_router(reasoning_router, prefix="/api")
