import logging

from llm.gemini_pipeline import invoke
from vectorstore.store import add_document

# Logging configuration
logging.basicConfig(level=logging.INFO)


class PersonaAgent:
    def __init__(self):
        pass

    def generate_persona(self, product_text: str, market_text: str = None):
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
            "recommended_channels": "",
            "summary": ""
        }}

        Requirements:
        - No explanation outside JSON.
        - No markdown.
        - JSON must be valid and parseable.
        - Make the persona extremely actionable and emotionally insightful.
        """

        # Call the model
        response = invoke(prompt)

        # Save to vector store only if data was generated
        if response:
            add_document(response, metadata={"type": "persona"})
            logging.info("Persona has been successfully generated and stored.")

        return response or "No persona could be generated."
