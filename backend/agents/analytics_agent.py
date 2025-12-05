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
        campaign_results: JSON or text of impressions, clicks,
          conversions, etc.
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
            return "No analytics could be generated."
        
        add_document(
            response,
            metadata={"type": "analytics", "source": "campaign_feedback"}
        )

        logger.info("Analytics insights stored successfully")
        return response