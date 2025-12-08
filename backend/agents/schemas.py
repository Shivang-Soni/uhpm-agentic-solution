from pydantic import BaseModel
from typing import Optional


class PlannerOutput(BaseModel):
    task: str
    needs_research: bool
    needs_persona: bool
    needs_content: bool
    needs_experimentation: bool
    needs_analytics: bool
    additional_context: Optional[str] = None
