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
    channel_recommendation: list
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
