from typing import Optional, Dict, Any, List, TypedDict

from pydantic import BaseModel, Field

from actions import Action


class PlannerOutput(BaseModel):
    task: str
    needs_research: bool
    needs_persona: bool
    needs_content: bool
    needs_experimentation: bool
    needs_analytics: bool
    additional_context: Optional[str] = None
    add_evaluation_steps: bool = True


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
    error: Optional[str]
    intent: str
    tone: str


class GoogleAdsAgentOutput(BaseModel):
    headline: str
    description: str
    keywords: list
    daily_budget_estimate: str
    landing_page_angle: str


class EmailAgentOutput(BaseModel):
    subject_line: str
    body: str
    tone: str


class MetaAdsAgentOutput(BaseModel):
    platform: str
    headline: str
    persona:  str
    budget: str
    tone: str


class PreviewRequest(BaseModel):
    channel: str
    artifacts: Dict


class PublishRequest(BaseModel):
    campaign_id: str


class ExecutionResult(BaseModel):
    action: str
    success: bool = True
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    evaluation: Optional[Dict[str, Any]] = None


class GraphRequest(BaseModel):
    task: str
    product_text: Optional[str] | None
    competitor_text: Optional[str] | None
    market_text: Optional[str] | None
    customer_text: Optional[str] | None
    persona_text: Optional[str] | None
    channel: Optional[str] | None
    variants: Optional[List[str]] | None
    campaign_results: Optional[str] | None


class ReasonRequest(BaseModel):
    task: str
    product_text: str | None = None
    customer_text: str | None = None
    market_text: str | None = None
    competitor_text: str | None = None
    persona_text: str | None = None
    channel: str | None = None
    variants: list | None = None
    campaign_results: str | None = None


class ResearchRequest(BaseModel):
    product_text: str
    competitor_text: str | None = None


class CampaignState(TypedDict, total=False):
    # Core identifiers
    campaign_id: str
    current_action: Action

    # Inputs
    brief: str
    target_audience: Optional[str]

    # Generated artifacts
    persona: Optional[Dict[str, Any]]
    content: Optional[Dict[str, Any]]

    # Experimentation
    experiments: List[Dict[str, Any]]
    performance_metrics: Optional[Dict[str, float]]

    # Control and lifecycle
    published: bool
    errors: List[str]
    history: List[Action]
