import json
import logging
from typing import Dict, Any

from agents.base_agent import BaseAgent
from actions import Action
from agents.schemas import CampaignState, ExecutionResult
from llm.gemini_pipeline import invoke
from vectorstore.store import add_document

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):

    action = Action.GENERATE_RESEARCH

    def execute(self, state: CampaignState) -> ExecutionResult:
        try:
            product_text = state.get("brief") or ""
            competitor_text = state.get("competitor_text")

            prompt = f"""
You are a marketing research agent.

Analyse the following product:
{product_text}

Competitor information:
{competitor_text if competitor_text else "N/A"}

Return ONLY valid JSON.
"""

            response = invoke(prompt)

            if not response:
                result = {
                    "product_summary": "",
                    "usps": [],
                    "target_audience": [],
                    "competitor_comparison": "",
                    "error": "Empty LLM response"
                }
            else:
                try:
                    result: Dict[str, Any] = json.loads(response)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from model")
                    result = {
                        "product_summary": "",
                        "usps": [],
                        "target_audience": [],
                        "competitor_comparison": "",
                        "error": "Invalid JSON from model"
                    }

            try:
                add_document(
                    json.dumps(result),
                    metadata={"type": "research"}
                )
            except Exception as e:
                logger.warning(f"Vectorstore write failed: {e}")

            # Agent ONLY returns data
            return self._success({"research": result})

        except Exception as e:
            logger.exception("ResearchAgent execution failed")
            return self._failure(str(e))
