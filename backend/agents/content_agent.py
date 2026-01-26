import json
import logging
import time
from typing import Dict, Any, Optional

from pydantic import ValidationError
from agents.base_agent import BaseAgent
from actions import Action
from agents.schemas import CampaignState, ContentOutput, ContentVariant, ExecutionResult
from llm.gemini_pipeline import invoke
from vectorstore.store import add_document
from agents.state import apply_execution_result

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ContentAgent(BaseAgent):
    action = Action.GENERATE_CONTENT
    VERSION = "1.0"

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

    def execute(
        self,
        state: CampaignState,
        channel: str = "social_media",
        tone: Optional[str] = None,
        max_variants: int = 3
    ) -> ExecutionResult:
        try:
            product_text = state.get("brief") or ""
            persona_text = state.get("persona", {}).get("summary", "")

            prompt = f"""
            Generate content for product:
            {product_text}
            Persona:
            {persona_text}
            Channel: {channel}
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

            apply_execution_result(state, {"content": validated.model_dump()})
            add_document(json.dumps(validated.model_dump()), metadata={"type": "content", "version": self.VERSION})

            return self._success(validated.model_dump())

        except Exception as e:
            logger.exception("ContentAgent execution failed")
            return self._failure(str(e))
