from fastapi import APIRouter
from api.graph_endpoints import router as graph_router
from api.vector_endpoints import router as vectordb_router

router = APIRouter()


@router.get("/health")
def health_check():
    return {"ok": True}


router.include_router(graph_router, prefix="/graph")
router.include_router(vectordb_router, prefix="/vectordb")
