import json
import logging
from llm.gemini_pipeline import invoke
from vectorstore.store import add_document

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalyticsAgent:
    """
    Analyses performance results and generates improvement suggestions.
    """

    def __init__(self):
        pass

    def analyse_campaign(self, campaign_results: str):
        """
        Analyzes campaign results and produces structured insights.

        Args:
        - campaign_results: JSON or text of impressions, clicks, conversions, etc.

        Returns:
        - Structured JSON with summary, persona changes, content improvements, channel recommendations, next steps.
        """
        prompt = f"""
        You are a SENIOR AI MARKETING PERFORMANCE ANALYST.

        Analyse the following campaign results:
        {campaign_results}

        Provide structured insights:
        - What worked well
        - What underperformed and why
        - Persona adjustments
        - Content improvements
        - Channel recommendations
        - Clear steps to improve conversion

        Return clean JSON:
        {{
            "summary": "...",
            "persona_changes": [...],
            "content_improvements": [...],
            "channel_recommendations": [...],
            "next_steps": [...]
        }}
        """

        response = invoke(prompt)

        if not response:
            logger.warning("No response from Analytics Agent")
            return {
                "summary": "",
                "persona_changes": [],
                "content_improvements": [],
                "channel_recommendations": [],
                "next_steps": [],
                "error": "No response from Agent"
            }

        # Validate JSON and apply fallback if needed
        try:
            json_response = json.loads(response)
        except json.JSONDecodeError:
            logger.error("Analytics JSON invalid. Applying fallback structure.")
            json_response = {
                "summary": "",
                "persona_changes": [],
                "content_improvements": [],
                "channel_recommendations": [],
                "next_steps": [],
                "error": "Invalid JSON from Agent"
            }

        # Store in vector DB
        add_document(
            json.dumps(json_response),
            metadata={"type": "analytics", "source": "campaign_feedback"}
        )

        logger.info("Analytics insights stored successfully.")
        return json_response
