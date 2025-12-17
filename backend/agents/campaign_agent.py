from typing import Dict, Any
import logging
import json

from llm.gemini_pipeline import invoke

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CampaignAgent:
    """
    Channel-agnostic campaign generation.
    Produce raw campaign artifacts for channels.
    """

    def __init__(self):
        pass

    def generate(
            self,
            product_text: str,
            persona_text: str,
            channel: str,
            budget: str | None = None,
            tone: str | None = None,
            intent: str | None = None
            ) -> Dict[str, Any]:
        """
        Generate campaign artifacts dynamically.
        """
        prompt = self._build_prompt(product_text, persona_text, channel, budget, tone, intent)
        logger.info(f"[CampaignAgent] Generating campaign for channel: {channel}")

        response = invoke(prompt)

        if not response:
            return RuntimeError("Empty response from campaign agent.")
        
        parsed = self._try_parse(response)
        if parsed:
            return parsed
        
        repair_prompt = self._repair_prompt(response)
        repaired = invoke(repair_prompt)

        parsed_repaired = self._try_parse(repaired)
        if parsed_repaired:
            return parsed_repaired
        
        raise RuntimeError("Failed to generate valid campaign JSON")
    
    def _build_prompt(
            self,
            product_text: str,
            persona_text: str,
            channel: str,
            budget: str,
            tone: str,
            intent: str
    ) -> str:
        return f"""
    You are a SENIOR PERFORMANCE MARKETING ANALYST.

    RULES:
    - Output valid JSON only.
    - No markdown
    - No explanation
    - No extra keys
    - Channel specific structure
    - Conversion focused

    Context:
    Product:
    {product_text}

    Persona:
    {persona_text}

    Channel:
    {channel}

    Budget:
    {budget or "not specified"}

    Tone:
    {tone or "neutral"}

    Intent:
    {intent or "general"}

    Channel Output Schemas:

    if channel=="google_ads":
    {{
    "headline": "",
    "description": "",
    "keywords": [],
    "daily_budget_estimate": "",
    "landing_page_angle": ""
    }}

    if channel=="meta_ads":
    {{
    "platform": "meta",
    "headline": "",
    "persona": "",
    "budget": "",
    "tone": ""
    }}

    if channel=="whatsapp":
    {{
    "initial_message": "",
    "follow_up_message": "",
    "closing_message": ""
    }}

    if channel=="email":
    {{
    "subject_line": "",
    "body": "",
    "tone": ""
    }}

    Return ONLY the JSON Object for the selected channel.
    """

    def _repair_prompt(
            self,
            raw_text: str
    ) -> str:
        
        return f"""
    Convert the following text ONLY into VALID JSON.
    NEVER EXPLAIN.
    Do not add new keys.

    Text:
    {raw_text}
    """

    def _try_parse(
            self,
            text: str
    ) -> Dict[str, Any] | None :

        try:
            return json.loads(text)
        except Exception as e:
            logger.warning(f"[CampaignAgent] JSON parse failed: {e}")
            return None
