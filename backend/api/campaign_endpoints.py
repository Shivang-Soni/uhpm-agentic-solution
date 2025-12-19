import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.schemas import PreviewRequest, PublishRequest
from agents.adapters.channel_adapter_dispatcher import ChannelAdapterDispatcher
from agents.adapters.channel_adapter_registry import ChannelAdapterRegistry
from agents.repositories.in_memory_campaign_repository import (
    InMemoryCampaignRepository
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency Wiring
repository = InMemoryCampaignRepository()
registry = ChannelAdapterRegistry()
dispatcher = ChannelAdapterDispatcher(
    registry=registry,
    repository=repository
)

# Routes
@router.post("/campaign/preview")

