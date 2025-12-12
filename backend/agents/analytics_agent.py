import json
import logging
from typing import Dict, Any

from llm.gemini_pipeline import invoke
from vectorstore.store import add_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


EXPECTED_SCHEMA = {
    "summary": str,
    "persona_changes": list,
    "content_improvements": list,
    "channel_recommendations": list,
    "next_steps": list,
}


class AnalyticsAgent:
    """
    Analyses performance results and generates structured, actionable insights.
    """

    def __init__(self):
        pass

    def analyse_campaign(
        self,
        campaign_results: str | Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyse performance metrics (CTR, conversions, CPL, ROAS etc.)
        and return fully validated structured insights.

        Args:
            campaign_results: Raw dict or text summary of the results.

        Returns:
            JSON dict following EXACT schema:
                summary: str
                persona_changes: list[str]
                content_improvements: list[str]
                channel_recommendations: list[str]
                next_steps: list[str]
        """

        prompt = f"""
You are a SENIOR PERFORMANCE MARKETING OPTIMISATION ENGINE.

Analyse the following campaign results and return ONLY a **valid JSON object**
matching EXACTLY this schema:

{{
  "summary": "string",
  "persona_changes": ["string", ...],
  "content_improvements": ["string", ...],
  "channel_recommendations": ["string", ...],
  "next_steps": ["string", ...]
}}

Rules:
- Output ONLY valid JSON.
- No explanations, prose, markdown, or surrounding text.
- All fields must be actionable.
- Never leave fields empty.
        
Campaign Results:
{campaign_results}
"""

        response = invoke(prompt)

        if not response:
            logger.warning("AnalyticsAgent: No LLM response. Using fallback.")
            return self._fallback("No LLM response")

        # Parse JSON
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            logger.error("AnalyticsAgent: Invalid JSON. Using fallback.")
            parsed = self._fallback("JSON parsing error")

        validated = self._validate_response(parsed)

        # Persist to vectorstore
        try:
            add_document(
                json.dumps(validated),
                metadata={"type": "analytics", "source": "campaign_feedback"}
            )
        except Exception as e:
            logger.error(f"Failed to store analytics insights: {e}")

        return validated

    # --------------------------
    # Validation & fallback
    # --------------------------

    def _validate_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures all required fields exist, have correct types,
        and replaces invalid values with fallbacks.
        """

        validated = {}

        for key, expected_type in EXPECTED_SCHEMA.items():
            value = data.get(key)

            # Missing key
            if value is None:
                validated[key] = self._default_value(expected_type)
                continue

            # Wrong type
            if not isinstance(value, expected_type):
                validated[key] = self._default_value(expected_type)
                continue

            validated[key] = value

        return validated

    def _fallback(self, error_msg: str) -> Dict[str, Any]:
        """
        Fallback JSON in case LLM fails fully.
        """
        return {
            "summary": f"Analytics unavailable: {error_msg}",
            "persona_changes": [],
            "content_improvements": [],
            "channel_recommendations": [],
            "next_steps": [],
        }

    @staticmethod
    def _default_value(expected_type):
        if expected_type is str:
            return ""
        if expected_type is list:
            return []
        return None
