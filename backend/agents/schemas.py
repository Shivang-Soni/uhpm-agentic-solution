from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field


class PlannerOutput(BaseModel):
    task: str
    needs_research: bool
    needs_persona: bool
    needs_content: bool
    needs_experimentation: bool
    needs_analytics: bool
    additional_context: Optional[str] = None


class AnalyticsOutput(BaseModel):
    summary: str
    persona_changes: list
    content_improvements: list
    channel_recommendations: list
    next_steps: list


class ContentVariant(BaseModel):
    variant_id: Optional[str] = None
    headline: str
    primary_text: str
    cta: Optional[str] = ""
    extra: Optional[Dict[str, Any]] = None
    score: Optional[float] = None


class ContentOutput(BaseModel):
    product_text: str
    persona_text: str
    channel: str
    tone: Optional[str] = ""
    variants: List[ContentVariant] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class CampaignPerformance(BaseModel):
    channel: str = Field(..., description="Marketing channel name")
    spend: float = Field(..., ge=0)
    impressions: int = Field(..., ge=0)
    clicks: int = Field(..., ge=0)
    ctr: float = Field(..., ge=0)
    conversions: Optional[int] = Field(default=0, ge=0)
    cpa: Optional[float] = Field(default=None, ge=0)


class DispatcherOutput(BaseModel):
    status: str
    agent: str
    data: Dict[str, Any]
    plan: Dict[str, Any]


class WhatsappAgentOutput(BaseModel):
    initial_message: str
    follow_up_message: str
    closing_message: str
    error: str


class GoogleAdsAgentOutput(BaseModel):
    headline: str
    description: str
    keywords: list
    daily_budget_estimate: str
    landing_page_angle: str


class EmailAgentOutput(BaseModel):
    subject: str
    body: str
    tone: str


class MetaAdsAgentOutput(BaseModel):
    platform: str
    headline: str
    persona:  str
    budget: str
    tone: str