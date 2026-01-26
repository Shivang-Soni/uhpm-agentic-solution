import json
import logging
from typing import Optional

from pydantic import ValidationError

from agents.base_agent import BaseAgent
from actions import Action
from agents.schemas import CampaignState, ContentOutput, ContentVariant, ExecutionResult
from llm.gemini_pipeline import invoke
from vectorstore.store import add_document

logger = logging.getLogger(__name__)


class ContentAgent(BaseAgent):

    action = Action.GENERATE_CONTENT
    VERSION = "1.0"

    def execute(self, state: CampaignState) -> ExecutionResult:
        try:
            product_text = state.get("brief") or ""
            persona_text = state.get("persona", {}).get("summary", "")

            channel = state.get("channel", "social_media")
            tone = state.get("tone")
            max_variants = state.get("max_variants", 3)

            prompt = f"""
You are a senior conversion copywriter.

Generate marketing content.

Product:
{product_text}

Persona:
{persona_text}

Channel: {channel}
Tone: {tone or "auto"}

Return ONLY valid JSON matching ContentOutput schema.
"""

            response = invoke(prompt)

            if not response:
                logger.warning("LLM empty response, using fallback")
                validated = self._fallback(product_text, persona_text, channel)
            else:
                try:
                    parsed = json.loads(response)
                    validated = ContentOutput(**parsed)
                except (json.JSONDecodeError, ValidationError):
                    logger.warning("Invalid JSON from LLM, using fallback")
                    validated = self._fallback(product_text, persona_text, channel)

            try:
                add_document(
                    json.dumps(validated.model_dump()),
                    metadata={"type": "content", "version": self.VERSION}
                )
            except Exception as e:
                logger.warning(f"Vectorstore write failed: {e}")

            # Agent returns only — Dispatcher commits
            return self._success({"content": validated.model_dump()})

        except Exception as e:
            logger.exception("ContentAgent execution failed")
            return self._failure(str(e))

    def _fallback(self, product_text: str, persona_text: str, channel: str) -> ContentOutput:
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
            metadata={"fallback": True}
        )
