from fastapi import APIRouter, HTTPException
from typing import Dict
from pydantic import BaseModel

from agents.campaigns.campaign_service import CampaignService
from agents.schemas import PreviewRequest, PublishRequest

router = APIRouter()


class CampaignController:
    """
    FastAPI endpoints for campaign operations/
    """

    def __init__(self, service: CampaignService):
        self.service = service

    def register_routes(self, router: APIRouter):
        @router.post("/campaign/preview")
        def preview_campaign(req: PreviewRequest):
            try:
                return self.service.preview_campaign(
                    channel=req.channel,
                    artifacts=req.artifacts
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
            
    @router.post("/campaign/publish")
    def publish_campaign(req: PublishRequest):
        try:
            return self.service.publish_campaign(campaign_id=req.campaign_id)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
