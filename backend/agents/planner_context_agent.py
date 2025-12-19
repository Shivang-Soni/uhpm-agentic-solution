import logging
from typing import Dict

from backend.agents.social_insight_agent import SocialInsightAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PlannerContextAgent:
    """
    Enriches planner input with external market and sentiment content.
    """

    def __init__(self):
        self.social_insight_agent = SocialInsightAgent()

    def build_context(self, user_task: str) -> Dict[str, object]:
        """
        Generate planner ready context from social sentiment data.
        """
        logger.info(
            "PlannerContextAgent: Building planner"
            " ready context from social insights."
            )

        insight = self.social_insight_agent.derive_marketing_insight(
            query=user_task
        )

        context = {
            "social_sentiment_summary": insight["sentiment_distribution"],
            "social_verdict": insight["verdict"],
            "confidence": insight["confidence"]
        }

        return context
