import json
import logging

from llm.gemini_pipeline import invoke
from vectorstore.store import add_document

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PersonaAgent:
    """
    Generates detailed marketing personas from product + market text.
    Ensures structured JSON output and stores results in vectorstore.
    """

    def __init__(self):
        pass

    def _fallback_persona(self, error_message: str) -> dict:
        """
        Return minimal persona structure if LLM fails or response is invalid
        """
        return {
            "persona_name": "",
            "age_range": "",
            "demographics": "",
            "lifestyle": "",
            "deep_motivations": "",
            "pain_points": "",
            "buying_triggers": "",
            "objections": "",
            "language_and_tone": "",
            "recommended_channels": [],
            "summary": "",
            "error": error_message
        }

    def _normalize_persona(self, persona: dict) -> dict:
        """Ensure all required fields exist and types are correct"""
        template = {
            "persona_name": "",
            "age_range": "",
            "demographics": "",
            "lifestyle": "",
            "deep_motivations": "",
            "pain_points": "",
            "buying_triggers": "",
            "objections": "",
            "language_and_tone": "",
            "recommended_channels": [],
            "summary": ""
        }

        for key, default in template.items():
            persona.setdefault(key, default)

        # recommended_channels normalization
        if isinstance(persona["recommended_channels"], str):
            persona["recommended_channels"] = [persona["recommended_channels"]]
        elif persona["recommended_channels"] is None:
            persona["recommended_channels"] = []

        # clear error field if exists
        persona.pop("error", None)

        return persona

    def generate_persona(
            self, product_text: str, market_text: str = None
            ) -> dict:
        """Generate persona JSON and store in vectorstore"""
        prompt = f"""
        You are a SENIOR MARKETING PERSONA MODELLER.
        Create a complete buyer persona based on the product below.

        Product:
        {product_text}

        Market/Customer Info:
        {market_text if market_text else "N/A"}

        Return ONLY the persona in this JSON schema:
        {{
            "persona_name": "",
            "age_range": "",
            "demographics": "",
            "lifestyle": "",
            "deep_motivations": "",
            "pain_points": "",
            "buying_triggers": "",
            "objections": "",
            "language_and_tone": "",
            "recommended_channels": [],
            "summary": ""
        }}

        Requirements:
        - NO TEXT outside JSON.
        - JSON must be valid and parseable.
        - Make persona actionable and insightful.
        """

        logger.info("PersonaAgent: sending prompt to LLM...")
        response = invoke(prompt)

        if not response:
            logger.error("PersonaAgent: empty LLM response")
            return self._fallback_persona("Empty LLM response")

        try:
            json_response = json.loads(response)
        except json.JSONDecodeError:
            logger.error("PersonaAgent: invalid JSON. Using fallback.")
            return self._fallback_persona("Invalid JSON response from agent.")

        json_response = self._normalize_persona(json_response)

        try:
            add_document(
                json.dumps(json_response),
                metadata={
                    "type": "persona",
                    "product_text": product_text,
                    "agent": "persona_agent"
                    }
            )
            logger.info("PersonaAgent: persona successfully stored")
        except Exception as e:
            logger.error(f"PersonaAgent: failed to store persona: {e}")

        return json_response
