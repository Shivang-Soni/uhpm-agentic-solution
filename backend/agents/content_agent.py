import logging
import json

from llm.gemini_pipeline import invoke
from vectorstore.store import add_document

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentAgent:
    """
    Generates marketing content automatically based on
    product info and persona insights.
    """

    def __init__(self):
        pass

    def generate_content(self, product_text: str, persona_text: str, channel: str = "social_media") -> dict:
        """
        Args:
        - product_text
        - persona-text
        - channel
        Returns:
        - Generated content as text
        """

        prompt = f"""
        You are a SENIOR MARKETING CONTENT GENERATOR.
        Create a high-converting content for the following:

        Product:
        {product_text}

        Target Persona:
        {persona_text}

        Channel: 
        {channel}

        Output should include:
        - Text
        - Structure / headings
        - Optional CTA
        - Tone suitable for the persona
        """

        # Templates
        templates = {
            "social_media": "Create short, punchy social posts.",
            "email": "Write a professional marketing email.",
            "ads": "Write a catchy ad headline and description."
        }
        template_instruction = templates.get(channel, "")
        prompt += "\n{template_instruction}"
        logger.info(f"Prompt sent to Agent: \n {prompt}")
        response = invoke(prompt)

        if response:
            try:
                structured_response = json.loads(response)
            except json.JSONDecodeError:
                structured_response = {"text": response}

            # Store in vectorstore for retrieval and experiments
            add_document(
                str(structured_response),
                metadata={
                    "type": "content",
                    "product_text": product_text,
                    "persona_text": persona_text,
                    "channel": channel
                }
            )
            logger.info(
                "Content successfully generated and stored."
            )
            return {
                    "product_text": product_text,
                    "persona_text": persona_text,
                    "channel": channel,
                    "content": structured_response
                }
        return {
                    "product_text": product_text,
                    "persona_text": persona_text,
                    "channel": channel,
                    "content": None
                }
