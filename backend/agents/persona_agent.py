import json
import logging

from llm.gemini_pipeline import invoke
from vectorstore.store import add_document

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonaAgent:
    def __init__(self):
        pass

    def _fallback_persona(self, error_message: str) -> dict:
        """Fallback structure if the JSON fails or the model breaks"""
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
            "recommended_channels": [],      # <-- FIX
            "summary": "",
            "error": error_message
        }
    
    def _normalize_persona(self, persona: dict) -> dict:
        """Ensures all fields exist and have the correct type"""
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
            "recommended_channels": [],      # <-- FIX
            "summary": ""
        }

        # Insert defaults in missing fields
        for key, default in template.items():
            persona.setdefault(key, default)

        # Normalisation of recommended channels
        if isinstance(persona["recommended_channels"], str):
            persona["recommended_channels"] = [persona["recommended_channels"]]

        if persona["recommended_channels"] is None:
            persona["recommended_channels"] = []

        return persona

    def generate_persona(self, product_text: str, market_text: str = None) -> dict:
        """
        Creates a full persona profile for the given product and the market.
        Then automatically saves it to the vector database.
        """
        prompt = f"""
        You are a SENIOR MARKETING PERSONA MODELLER.
        Based on the product description below,
        create a complete, highly actionable buyer persona.

        Product:
        {product_text}

        Market/Customer Info:
        {market_text if market_text else "N/A"}

        Return the persona ONLY in the following strict JSON format:
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
            "recommended_channels": ["", ""],
            "summary": ""
        }}

        Requirements:
        - NO TEXT outside JSON.
        - No markdown.
        - JSON must be valid and parseable.
        - Make the persona extremely actionable and emotionally insightful.
        """

        logger.info("Sending persona prompt to model...")
        response = invoke(prompt)

        if not response:
            logger.error("PersonaAgent: empty LLM response.")
            return self._fallback_persona("Empty LLM response")

        # Try to parse JSON
        try:
            json_response = json.loads(response)
        except json.JSONDecodeError:
            logger.error("PersonaAgent: Invalid JSON. Using fallback.")
            return self._fallback_persona("Invalid JSON response from agent.")

        # Normalize fields
        json_response = self._normalize_persona(json_response)

        # Store in vector DB
        try:
            add_document(
                json.dumps(json_response),
                metadata={"type": "persona", "product_text": product_text}
            )
            logger.info("Persona has been successfully generated and stored.")
        except Exception as e:
            logger.error(f"PersonaAgent: Failed to store persona: {e}")

        return json_response
