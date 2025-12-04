import logging

from llm.gemini_pipeline import invoke
from vectorstore.store import add_document

# logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchAgent:
    def __init__(self):
        pass

    def analyse_product(
            self, product_text: str, competitor_text: str | None
            ) -> str:
        """
        Analyses the product text and competitor text
        altogether and provides insights. Then saves these
        to the vector database.
        """
        prompt = f"""
        You are a marketing research agent.
        Analyse the following product:
        {product_text}
        Competitor information: {competitor_text if competitor_text else "N/A"}
        Give the results in following JSON format:
        {{
            "product_summary": "2 Sentences",
            "usps": ["List", "of", "USPs"],
            "target_audience": ["List", "of", "Target", "Audience"],
            "competitor_comparison": "2 Sentences"
        }}
        """
        response = invoke(prompt)
        if response:
            logger.info("Analysis completed successfully.")
            add_document(
                response, 
                metadata={
                    "product_text": product_text,
                    "competititor_text": competitor_text
                    })
        return response or "No response is available."

