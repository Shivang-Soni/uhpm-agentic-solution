import json
import logging
from typing import Dict, Any

from agents.base_agent import BaseAgent
from actions import Action
from agents.schemas import CampaignState, ExecutionResult
from llm.gemini_pipeline import invoke
from vectorstore.store import add_document
from agents.state import apply_execution_result

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PersonaAgent(BaseAgent):
    action = Action.GENERATE_PERSONA

    def execute(self, state: CampaignState) -> ExecutionResult:
        try:
            product_text = state.get("brief") or ""
            market_text = state.get("market_text")

            prompt = f"""
            You are a senior marketing persona modeler.
            Generate a complete persona for the product below:
            Product: {product_text}
            Market/Customer Info: {market_text if market_text else "N/A"}
            """

            response = invoke(prompt)
            persona_result: Dict[str, Any] = {}
            if response:
                try:
                    persona_result = json.loads(response)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from LLM, returning fallback")
                    persona_result = {
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
                        "error": "Invalid JSON"
                    }

            apply_execution_result(state, {"persona": persona_result})
            add_document(json.dumps(persona_result), metadata={"type": "persona"})

            return self._success(persona_result)

        except Exception as e:
            logger.exception("PersonaAgent execution failed")
            return self._failure(str(e))
