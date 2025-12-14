import json
import logging
import time
from typing import Optional, Dict, Any

from pydantic import ValidationError

from llm.gemini_pipeline import invoke
from vectorstore.store import add_document
from agents.schemas import ContentOutput, ContentVariant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentAgent:
    """
    Generates marketing content based on product and persona.
    Deterministic schema, safe fallback, production-ready.
    """

    VERSION = "1.0"

    def _build_prompt(
        self,
        product_text: str,
        persona_text: str,
        channel: str,
        tone: Optional[str],
        max_variants: int,
    ) -> str:

        templates = {
            "social_media": "Create short, punchy social posts.",
            "email": "Write a professional marketing email.",
            "ads": "Write a catchy ad headline and description.",
            "whatsapp": "Write short WhatsApp messages.",
        }

        instruction = templates.get(channel, templates["social_media"])

        schema_example = {
            "product_text": product_text,
            "persona_text": persona_text,
            "channel": channel,
            "tone": tone or "",
            "variants": [
                {
                    "headline": "string",
                    "primary_text": "string",
                    "cta": "string"
                }
            ]
        }

        return f"""
You are a senior marketing content generator.

Return ONLY a valid JSON object matching EXACTLY this schema:
{json.dumps(schema_example, indent=2)}

RULES:
- Output JSON only.
- Provide exactly {max_variants} variants.
- Each variant must include headline and primary_text.
- CTA is optional.
- Infer tone if missing.

Instruction:
{instruction}

Product:
{product_text}

Persona:
{persona_text}

Channel: {channel}
Tone: {tone or "neutral"}
"""

    def _fallback(
        self,
        product_text: str,
        persona_text: str,
        channel: str,
    ) -> ContentOutput:

        variant = ContentVariant(
            headline="Discover a better solution today",
            primary_text=product_text[:120],
            cta="Learn more"
        )

        return ContentOutput(
            product_text=product_text,
            persona_text=persona_text,
            channel=channel,
            tone="neutral",
            variants=[variant],
            metadata={"fallback": True},
        )

    def generate_content(
        self,
        product_text: str,
        persona_text: str,
        channel: str = "social_media",
        tone: Optional[str] = None,
        max_variants: int = 3,
    ) -> Dict[str, Any]:

        start = time.time()

        prompt = self._build_prompt(
            product_text,
            persona_text,
            channel,
            tone,
            max_variants
        )

        logger.info("ContentAgent: invoking LLM.")
        response = None

        try:
            response = invoke(prompt)
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")

        if not response:
            logger.warning("No LLM response. Using fallback.")
            validated = self._fallback(product_text, persona_text, channel)
        else:
            try:
                parsed = json.loads(response)
                validated = ContentOutput(**parsed)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Invalid content output: {e}")
                validated = self._fallback(product_text, persona_text, channel)

        try:
            add_document(
                json.dumps(validated.model_dump()),
                metadata={
                    "type": "content",
                    "channel": channel,
                    "version": self.VERSION
                },
            )
        except Exception as e:
            logger.error(f"Vectorstore write failed: {e}")

        duration = time.time() - start
        logger.info(
            f"ContentAgent finished in {duration:.2f}s "
            f"with {len(validated.variants)} variants"
        )

        return validated.model_dump()
