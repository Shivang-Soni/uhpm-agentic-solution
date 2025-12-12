import json
import logging
from typing import Dict, Any

from llm.gemini_pipeline import invoke
from vectorstore.store import add_document
from schemas import AnalyticsOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalyticsAgent:
    """
    Analyses performance results and produces structured optimisation insights.
    """

    def analyse_campaign(
            self,
            campaign_results: str | Dict[str, Any]
            ) -> Dict[str, Any]:
        """
        Analyse campaign performance
          and return structured optimisation suggestions.

        Args:
            campaign_results: dict or string description of performance.

        Returns:
            A dict matching EXACTLY the AnalyticsOutput schema.
        """

        prompt = f"""
You are a SENIOR PERFORMANCE MARKETING OPTIMISATION ENGINE.

Analyse the following campaign results and return ONLY a valid JSON object
matching EXACTLY this schema:

{{
  "summary": "string",
  "persona_changes": ["string", ...],
  "content_improvements": ["string", ...],
  "channel_recommendations": ["string", ...],
  "next_steps": ["string", ...]
}}

RULES:
- Output ONLY raw JSON.
- NO text outside the JSON object.
- NEVER leave any field empty.
- If information is missing, infer the most likely optimisation.

Campaign Results:
{campaign_results}
"""

        # 1. LLM Invocation
        response = invoke(prompt)
        if not response:
            logger.warning(
                "AnalyticsAgent: Empty LLM response. Using fallback."
                )
            return self._fallback("LLM returned empty response")

        # 2. Parse JSON
        try:
            parsed = json.loads(response)
        except Exception as e:
            logger.error(f"AnalyticsAgent: JSON parse failed: {e}")
            return self._fallback("Invalid JSON returned by LLM")

        # 3. Validate schema
        try:
            model = AnalyticsOutput(**parsed)
            validated = model.model_dump()
        except Exception as e:
            logger.error(f"AnalyticsAgent: Schema validation failed: {e}")
            return self._fallback("Schema validation error")

        # 4. Store in vector DB
        try:
            add_document(
                json.dumps(validated),
                metadata={"type": "analytics", "source": "campaign_feedback"},
            )
        except Exception as e:
            logger.error(
                f"AnalyticsAgent: Failed to store analytics insights: {e}"
                )

        return validated

    def _fallback(self, reason: str) -> Dict[str, Any]:
        """
        Deterministic fallback structure ensuring nothing breaks.
        """
        logger.warning(f"AnalyticsAgent fallback triggered: {reason}")

        return {
            "summary": f"Analytics unavailable: {reason}. \
              Providing generic improvements.",
            "persona_changes": [
                "Clarify primary customer motivation.",
                "Refine demographic focus for precision targeting."
            ],
            "content_improvements": [
                "Improve clarity of value proposition.",
                "Strengthen opening hook to increase attention.",
            ],
            "channel_recommendations": [
                "Focus on WhatsApp + Meta high-intent placements."
            ],
            "next_steps": [
                "Collect more granular performance metrics.",
                "Test 2–3 more creative or offer variations."
            ],
        }
