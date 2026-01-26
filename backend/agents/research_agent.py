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


class ResearchAgent(BaseAgent):
    action = Action.GENERATE_CONTENT

    def execute(self, state: CampaignState) -> ExecutionResult:
        try:
            product_text = state.get("brief") or ""
            competitor_text = state.get("competitor_text")

            prompt = f"""
            You are a marketing research agent.
            Analyse the following product:
            {product_text}
            Competitor information: {competitor_text if competitor_text else "N/A"}
            """

            response = invoke(prompt)
            result: Dict[str, Any] = {}
            if response:
                try:
                    result = json.loads(response)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from model, returning fallback")
                    result = {
                        "product_summary": "",
                        "usps": [],
                        "target_audience": [],
                        "competitor_comparision": "",
                        "error": "Invalid JSON from model"
                    }

            apply_execution_result(state, {"research": result})
            add_document(json.dumps(result), metadata={"type": "research"})

            return self._success(result)

        except Exception as e:
            logger.exception("ResearchAgent execution failed")
            return self._failure(str(e))
