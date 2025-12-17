import logging
import json
from typing import Any, Dict, Optional

from llm.gemini_pipeline import invoke

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


ALLOWED_ACTIONS = {
    "call_research_agent",
    "call_persona_agent",
    "call_content_agent",
    "call_campaign_agent",
    "preview_campaign",
    "publish_campaign",
}

ALLOWED_INPUT_KEYS = {
    "product_text",
    "competitor_text",
    "research_insights",
    "persona_text",
    "channel",
    "budget",
    "tone",
    "intent",
    "artifacts",
}


class ReasonerAgent:
    """
    Determines the next system action in a strictly deterministic way.
    Produces Dispatcher-compatible JSON only.
    """

    def __init__(self, retriever=None):
        self.retriever = retriever
        self.max_retries = 2

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_prompt(self, user_task: str, retrieved: Any) -> str:
        return f"""
You are a routing engine inside a marketing automation system.

You MUST return VALID JSON ONLY.
NO explanations. NO markdown. NO comments.

Allowed actions:
- call_research_agent
- call_persona_agent
- call_content_agent
- call_campaign_agent
- preview_campaign
- publish_campaign

Allowed input keys:
product_text, competitor_text, research_insights,
persona_text, channel, budget, tone, intent, artifacts

Each input MUST be marked as:
"required" or "optional"

If uncertain, choose the simplest valid action.

User task:
{user_task}

Retrieved context:
{json.dumps(retrieved, indent=2)}

Return JSON in EXACT format:
{{
  "task_type": "research|persona|content|campaign",
  "action": "one_allowed_action",
  "inputs_needed": {{
    "input_name": "required|optional"
  }}
}}
"""

    # Parsing & Validation
    def _try_parse(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(text)

            if not isinstance(data, dict):
                return None

            if "action" not in data or "inputs_needed" not in data:
                return None

            if data["action"] not in ALLOWED_ACTIONS:
                logger.warning(f"Invalid action received: {data['action']}")
                return None

            if not isinstance(data["inputs_needed"], dict):
                return None

            data["inputs_needed"] = self._normalize_inputs(data["inputs_needed"])

            return data

        except Exception as e:
            logger.warning(f"Reasoner JSON parsing failed: {e}")
            return None

    def _normalize_inputs(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        normalized = {}
        for key, value in inputs.items():
            if key not in ALLOWED_INPUT_KEYS:
                continue
            if value not in ("required", "optional"):
                normalized[key] = "optional"
            else:
                normalized[key] = value
        return normalized

    # Fallback
    def _default_decision(self) -> Dict[str, Any]:
        return {
            "task_type": "research",
            "action": "call_research_agent",
            "inputs_needed": {
                "product_text": "required",
                "competitor_text": "optional"
            }
        }

    # Public API
    def decide(self, user_task: str) -> Dict[str, Any]:
        logger.info("[Reasoner] Starting decision process")

        retrieved = {}
        if self.retriever:
            try:
                retrieved = self.retriever.search_docs(user_task)
            except Exception as e:
                logger.warning(f"[Reasoner] Retriever failed: {e}")

        prompt = self._build_prompt(user_task, retrieved)
        response = invoke(prompt)

        if not response:
            return self._default_decision()

        parsed = self._try_parse(response)
        if parsed:
            return parsed

        # Retry once with hard fallback instruction
        fallback_prompt = f"""
Convert the following text into VALID JSON ONLY.
Use allowed actions and input keys only.

Text:
{response}
"""
        retry_response = invoke(fallback_prompt)

        if retry_response:
            parsed_retry = self._try_parse(retry_response)
            if parsed_retry:
                return parsed_retry

        return self._default_decision()
