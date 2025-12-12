# backend/agents/content_agent.py
import logging
import json
import time
from typing import Optional, Dict, Any

from pydantic import ValidationError

from llm.gemini_pipeline import invoke
from vectorstore.store import add_document
from agents.schemas import ContentOutput, ContentVariant  # adjust import path if needed

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentAgent:
    """
    Generates marketing content automatically based on
    product info and persona insights.
    """
    VERSION = "1.0"

    def __init__(self):
        pass

    def _build_prompt(
            self,
            product_text: str,
            persona_text: str,
            channel: str,
            tone: Optional[str],
            max_variants: int = 3
            ) -> str:
        """
        Build a strict JSON-only prompt for the LLM.
        """

        templates = {
            "social_media": "Create short, punchy social posts suitable for the channel.",
            "email": "Write a professional marketing email (subject + body + CTA).",
            "ads": "Write a catchy ad headline and short description (headline + body + CTA).",
            "whatsapp": "Write short WhatsApp messages and follow ups (no markup).",
        }

        # Use the actual variable 'channel' (was a bug before)
        template_instruction = templates.get(channel, templates["social_media"])

        schema_example = {
            "product_text": product_text,
            "persona_text": persona_text,
            "channel": channel,
            "tone": tone or "",
            "variants": [
                {
                    "headline": "string",
                    "primary_text": "string",
                    "cta": "string",
                    "extra": {"notes": "optional"}
                }
            ]
        }

        prompt = f"""
You are a senior marketing content generator. Return ONLY a single valid JSON object EXACTLY matching the schema described below. Do NOT include any explanatory text.

SCHEMA:
{json.dumps(schema_example, indent=2)}

RULES:
- Output ONLY JSON and nothing else.
- Provide {max_variants} distinct variants in the "variants" array.
- Each variant must include "headline" and "primary_text". "cta" is optional.
- Keep tone consistent with the persona. If persona lacks details, infer a neutral professional tone.

INSTRUCTION:
{template_instruction}

Product text:
{product_text}

Persona text:
{persona_text}

Channel: {channel}
Tone: {tone or "not specified"}

Return valid JSON now.
"""
        return prompt

    def _parse_and_validate(
            self,
            response: str,
            product_text: str,
            persona_text: str,
            channel: str
            ) -> ContentOutput:
        """
        Parse JSON string and validate against the ContentOutput Schema.
        Raises ValidationError when fails.
        """
        parsed = json.loads(response)
        parsed.setdefault("product_text", product_text)
        parsed.setdefault("persona_text", persona_text)
        parsed.setdefault("channel", channel)
        parsed.setdefault("metadata", {})
        validated = ContentOutput(**parsed)
        return validated

    def _fallback(self, product_text: str, persona_text: str, channel: str) -> ContentOutput:
        """
        Deterministic fallback to avoid breaking the pipeline.
        Returns a minimal, predictable ContentOutput.
        """
        headline = f"{product_text[:40]} — Try now"
        primary = f"{product_text[:120]} — Short summary targeted at {persona_text[:60]}."
        variant = ContentVariant(headline=headline, primary_text=primary, cta="Learn more")
        out = ContentOutput(
            product_text=product_text,
            persona_text=persona_text,
            channel=channel,
            tone="neutral",
            variants=[variant],
            metadata={"fallback": True},
        )
        return out

    def generate_content(
            self,
            product_text: str,
            persona_text: str,
            channel: str = "social_media",
            tone: Optional[str] = None,
            persona_id: Optional[str] = None,
            brand_voice: Optional[str] = None,
            max_variants: int = 3,
            ) -> Dict[str, Any]:
        """
        Generates channel optimized content and returns a serializable dict.
        """

        start_ts = time.time()
        prompt = self._build_prompt(
            product_text,
            persona_text,
            channel,
            tone,
            max_variants
        )

        logger.info("ContentAgent: invoking LLM for content generation.")
        response = None
        try:
            response = invoke(prompt)
        except Exception as e:
            logger.error(f"ContentAgent: LLM invocation failed: {e}")

        # Default to fallback if no response
        if not response:
            logger.warning("ContentAgent: No response from LLM, applying fallback.")
            validated = self._fallback(product_text, persona_text, channel)
        else:
            # Try parse & validate; if fail, attempt repair once, then fallback
            try:
                validated = self._parse_and_validate(response, product_text, persona_text, channel)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"ContentAgent: parsing/validation error: {e}")
                # Attempt repair via LLM (single retry)
                try:
                    repair_prompt = f"""
Convert the following text into VALID JSON that matches the expected schema exactly.

Text:
{response}

Return only the corrected JSON.
"""
                    repaired = invoke(repair_prompt)
                    validated = self._parse_and_validate(repaired, product_text, persona_text, channel)
                    logger.info("ContentAgent: repair succeeded.")
                except Exception as e2:
                    logger.warning(f"ContentAgent: repair failed: {e2}. Falling back.")
                    validated = self._fallback(product_text, persona_text, channel)

        # Prepare metadata and store in vectorstore
        metadata = {
            "type": "content",
            "channel": channel,
            "version": self.VERSION,
            "persona_id": persona_id,
            "brand_voice": brand_voice
        }

        try:
            # store the serializable dict
            add_document(json.dumps(validated.model_dump()), metadata=metadata)
        except Exception as e:
            logger.error(f"ContentAgent: failed to store results in vectorstore: {e}")

        duration = time.time() - start_ts
        try:
            variants_count = len(validated.variants)
        except Exception:
            variants_count = 0

        logger.info(f"ContentAgent: completed in {duration:.2f}s. Variants: {variants_count}")

        # Return a plain dict (serializable)
        return validated.model_dump()
