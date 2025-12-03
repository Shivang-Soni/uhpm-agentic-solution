from fastapi import APIRouter
from api.graph_endpoints import router as graph_router

router = APIRouter()


@router.get("/health")
def health_check():
    return {"ok": True}


router.include_router(graph_router, prefix="/graph")
