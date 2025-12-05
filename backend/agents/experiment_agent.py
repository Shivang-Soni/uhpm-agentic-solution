import logging

from llm.gemini_pipeline import invoke
from vectorstore.store import add_document

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentationAgent:
    """
    A/B Testing + Variant Scoring Agent
    Scores multiple content variants and returns the best performer.
    """

    def __init__(self):
        pass

    def score_variants(self, persona_text: str, channel: str, variants: list[str]):
        """
        Scores content variants based on persona fit and conversion likelihood.
        """

        prompt = f"""
        You are an AI MARKETING EXPERIMENT EVAULATOR.
        Score each content variant from 0 to 100 based on:

        - Fit to Persona
        - Fit to Channel
        - Conversion Likelihood
        - Clarity and persuasiveness

        Persona:
        {persona_text}

        Channel:
        {channel}

        Variants:
        {variants}

        Return JSON with:
        [
        {{
        "variant": "...",
        "score": 0-100,
        "reason": "..."
        }}]
        """

        response = invoke(prompt)
   
        if not response:
            return "No evaluation could be generated."

        # Save experiment vectors into the Vectordb
        add_document(
            response,
            metadata={
                "type": "experiment",
                "channel": channel,
                "persona": persona_text
            }
        )

        logger.info("Experiment results stored successfully.")
        return response
