import json
import logging
from typing import Dict, Any

from agents.base_agent import BaseAgent
from actions import Action
from agents.schemas import CampaignState, ExecutionResult
from llm.gemini_pipeline import invoke
from vectorstore.store import add_document

logger = logging.getLogger(__name__)


class PersonaAgent(BaseAgent):

    action = Action.GENERATE_PERSONA

    def execute(self, state: CampaignState) -> ExecutionResult:
        try:
            product_text = state.get("brief") or ""
            market_text = state.get("market_text")

            prompt = f"""
You are a senior marketing persona modeler.

Generate a complete buyer persona.

Product:
{product_text}

Market / Customer Info:
{market_text if market_text else "N/A"}

Return ONLY valid JSON.
"""

            response = invoke(prompt)

            if not response:
                persona_result = self._fallback("Empty LLM response")
            else:
                try:
                    persona_result: Dict[str, Any] = json.loads(response)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from LLM")
                    persona_result = self._fallback("Invalid JSON")

            try:
                add_document(
                    json.dumps(persona_result),
                    metadata={"type": "persona"}
                )
            except Exception as e:
                logger.warning(f"Vectorstore write failed: {e}")

            # Agent returns only — Dispatcher commits
            return self._success({"persona": persona_result})

        except Exception as e:
            logger.exception("PersonaAgent execution failed")
            return self._failure(str(e))

    def _fallback(self, reason: str) -> Dict[str, Any]:
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
            "error": reason
        }
