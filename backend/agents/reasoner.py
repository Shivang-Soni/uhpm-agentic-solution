import logging
import json
from typing import Any, Dict, Optional

from llm.gemini_pipeline import invoke

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReasonerAgent:
    """
    ReasonerAgent:
    Classifies the user input and returns a structured JSON dictionary.
    Responsibilities:
    - Retrieval lookup (optional)
    - Task type classification
    - Agent selection
    - JSON validation & fallback
    """
    def __init__(self, retriever):
        self.max_retries = 2
        self.retriever = retriever

    def _build_prompt(self, user_task: str, retrieved: Any) -> str:
        return f"""
        You are the CORE REASONING AGENT of an advanced marketing AI system.

        Responsibilities:
        - Analyze the user's request
        - Use retrieved memory context
        - Decide the task_type
        - Select the agent to call
        - Determine required inputs
        - ALWAYS return valid JSON ONLY. NEVER EXPLAIN.

        === Classification Rules ===
        Persona related:
        - target audience, personas, segmentation
        → task_type: "persona"
        → action: "call_persona_agent"
        → inputs_needed:
        {{
            "product_text": "description of the product",
            "market_text": "market or customer context"
        }}

        WhatsApp / Messaging related:
        - WhatsApp messages, follow-ups, broadcasts
        → task_type: "content"
        → action: "call_whatsapp_agent"
        → inputs_needed:
        {{
            "product_text": "needed to describe the product or service",
            "persona_text": "needed to adapt tone and messaging",
            "intent": "optional goal of the message",
            "tone": "optional tone of voice"
        }}

        Meta Ads:
        - Meta / Facebook Ads campaigns
        → task_type: "channel"
        → action: "call_meta_ads_agent"
        → inputs_needed: {{
            "product_text": "description of product",
            "persona_text": "persona for campaign",
            "campaign_budget": "budget allocation"
            }}

        Google Ads:
        - Google / Search Ads
        → task_type: "channel"
        → action: "call_google_ads_agent"
        → inputs_needed: {{
            "product_text": "description of product",
            "persona_text": "persona for campaign",
            "campaign_budget": "budget allocation"
            }}

        E-Mail Marketing:
        - Email campaigns, templates
        → task_type: "channel"
        → action: "call_email_agent"
        → inputs_needed: {{
            "product_text": "description of product",
            "persona_text": "persona for campaign",
            "email_template": "base template to use"
            }}

        Other task types:
        - research   → call_research_agent
        - analysis   → call_analysis_agent
        - content    → call_content_agent
        - experiment → call_experiment_agent

        === INPUT ===
        User request:
        {user_task}

        Retrieved memory:
        {json.dumps(retrieved, indent=2)}

        === OUTPUT FORMAT ===
        {{
          "task_type": "research|persona|content|experiment|analysis",
          "reasoning": "short justification",
          "action": "agent action name",
          "inputs_needed": {{
            "input_name": "why this input is required"
          }}
        }}
        """

    def _fallback_prompt(self, raw_text: str) -> str:
        return f"""
        Previous response was not valid JSON.
        Convert the following into VALID JSON with fields:
        task_type, reasoning, action, inputs_needed.
        Return ONLY JSON - NEVER EXPLAIN.

        Raw text:
        {raw_text}
        """

    def _try_parse(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(text)
            required = {"task_type", "reasoning", "action", "inputs_needed"}
            if not required.issubset(set(data.keys())):
                logger.warning("Parsed JSON is missing required keys.")
                return None
            if not isinstance(data.get("inputs_needed"), dict):
                logger.warning("inputs_needed must be a dict.")
                return None
            return data
        except Exception as e:
            logger.warning(f"JSON parsing failed: {e}")
            return None

    def _default_decision(self) -> Dict[str, Any]:
        return {
            "task_type": "research",
            "reasoning": "fallback due to reasoning failure",
            "action": "call_research_agent",
            "inputs_needed": {
                "product_text": "required"
            }
        }

    def decide(self, user_task: str):
        """
        Main reasoning entrypoint.
        Returns structured JSON for Dispatcher.
        """
        logger.info("Starting reasoning process.")

        retrieved = {}
        if self.retriever is not None:
            try:
                retrieved = self.retriever.search_docs(user_task)
            except Exception as e:
                logger.warning(f"Retriever failed: {e}")
                retrieved = {}

        prompt = self._build_prompt(user_task, retrieved)
        response = invoke(prompt)

        if not response:
            logger.error("No reasoning response from Agent on initial call.")
            return self._default_decision()

        parsed = self._try_parse(response)
        if parsed:
            logger.info("Successfully parsed initial LLM response.")
            return parsed

        logger.warning("Initial parse failed - requesting JSON-only fallback from Agent")
        fallback = self._fallback_prompt(response)
        updated_response = invoke(fallback)

        if not updated_response:
            logger.error("No response from LLM on fallback.")
            return self._default_decision()

        parsed_fallback = self._try_parse(updated_response)
        if parsed_fallback:
            logger.info("Successfully parsed fallback LLM response.")
            return parsed_fallback

        logger.error("Both parsing attempts failed.")
        return self._default_decision()
