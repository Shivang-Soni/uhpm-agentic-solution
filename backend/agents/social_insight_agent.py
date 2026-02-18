import logging
from collections import defaultdict
from typing import Dict, List, Any

from agents.retriever_agent import RetrieverAgent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SocialInsightAgent:
    """
    Converts raw / social sentiment documents from memory
    into actionable marketing insights.
    """

    def __init__(self):
        self.retriever = RetrieverAgent()

    def gather_social_signals(
            self,
            query: str,
            top_k: int = 20,
    ) -> List[dict]:
        """
        Fetches relevant social documents from memory.
        """
        logger.info(f"Gathering social signals for {query}")
        return self.retriever.search_docs(query, top_k)

    def analyse_sentiment_distribution(
            self,
            documents: List[dict]
    ) -> Dict[str, int]:
        """
        Aggregates sentiment signals.
        """
        sentiment_buckets = defaultdict(int)

        for item in documents:
            sentiment = item["metadata"].get("sentiment", 0)

            if sentiment > 0.1:
                sentiment_buckets["positive"] += 1
            elif sentiment < -0.1:
                sentiment_buckets["negative"] += 1
            else:
                sentiment_buckets["neutral"] += 1

        return dict(sentiment_buckets)

    def derive_marketing_insight(
            self,
            query: str,
    ) -> Dict[str, object]:
        """
        Produces a marketing-ready insight from social data.
        """
        docs = self.gather_social_signals(query)
        distribution = self.analyse_sentiment_distribution(docs)

        total = sum(distribution.values()) or 1

        insight = {
            "query": query,
            "sentiment_distributiom": distribution,
            "negative_ratio": distribution.get("negative", 0) / total,
            "positive_ratio": distribution.get('positive', 0) / total,
            "confidence": min(1.0, total / 20)
        }

        insight["verdict"] = self._verdict(insight)

        return insight

    def _verdict(self, insight: Dict[str, Any]) -> str:
        """
        Converts sentiment stats into plain language marketing guidance.
        """
        if insight["negative_ratio"] > 0.4:
            return (
                "Strong negative sentiment detected. "
                "Address obejections and reposition messaging."
            )

        if insight["positive_ratio"] > 0.4:
            return (
                "Strong positive sentiment detected."
                "Double down on current positioning."
                )

        return (
            "Mixed sentiment. "
            "Run creative experiments to identify winning angles."
        )
