from fastapi import APIRouter

# Existing routers
from api.graph_endpoints import router as graph_router
from api.vector_endpoints import router as vectordb_router
from api.research_endpoints import router as research_router
from api.reasoning_routes import router as reasoning_router

# Campaign wiring
from agents.adapters.channel_adapter_registry import ChannelAdapterRegistry
from agents.adapters.channel_adapter_dispatcher import ChannelAdapterDispatcher
from agents.repositories.in_memory_campaign_repository import (
    InMemoryCampaignRepository
)
from agents.campaigns.campaign_service import CampaignService
from agents.campaigns.campaign_controller import CampaignController

router = APIRouter()


# Health

@router.get("/health")
def health_check():
    return {"ok": True}


router.include_router(graph_router, prefix="/graph")
router.include_router(vectordb_router, prefix="/vectordb")
router.include_router(research_router, prefix="/research")
router.include_router(reasoning_router, prefix="/api")


# Campaign API

# Infrastructure
campaign_repository = InMemoryCampaignRepository()
channel_registry = ChannelAdapterRegistry()
campaign_dispatcher = ChannelAdapterDispatcher(
    registry=channel_registry,
    repository=campaign_repository,
)

# Application layer
campaign_service = CampaignService(
    dispatcher=campaign_dispatcher,
    repository=campaign_repository,
)

# Controller
campaign_controller = CampaignController(service=campaign_service)
campaign_controller.register_routes(router)
