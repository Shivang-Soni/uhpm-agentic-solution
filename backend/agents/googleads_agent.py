import json
import logging
from typing import Dict, Optional

from llm.gemini_pipeline import invoke
from agents.schemas import GoogleAdsAgentOutput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GoogleAdsAgent:
    """
    Generate Google Ads campaign structures based on
    product, persona and budget.
    """

    def generate_campaign(
        self,
        product_text: str,
        persona_text: str,
        campaign_budget: str,
        tone: str = "neutral"
    ) -> Dict:

        prompt = self._build_prompt(
            product_text,
            persona_text,
            campaign_budget,
            tone
        )

        response = invoke(prompt)

        if not response:
            return self._fallback("empty_response")

        parsed = self._try_parse(response)
        if parsed:
            return parsed

        # JSON repair fallback
        fallback_prompt = self._fallback_prompt(response)
        fallback_response = invoke(fallback_prompt)

        if not fallback_response:
            return self._fallback("fallback_empty")

        parsed_fallback = self._try_parse(fallback_response)
        if parsed_fallback:
            return parsed_fallback

        return self._fallback("json_parse_failed")

    def _build_prompt(
        self,
        product_text: str,
        persona_text: str,
        campaign_budget: str,
        tone: str
    ) -> str:
        return f"""
        You are a Google Ads campaign strategist.

        Rules:
        - Output VALID JSON ONLY
        - No markdown
        - No explanations
        - Conversion-focused
        - Budget-aware

        Context:
        Product:
        {product_text}

        Persona:
        {persona_text}

        Budget:
        {campaign_budget}

        Tone:
        {tone}

        Output format:
        {{
        "headline": "",
        "description": "",
        "keywords": [],
        "daily_budget_estimate": "",
        "landing_page_angle": ""
        }}
        """

    def _try_parse(self, text: str) -> Optional[Dict]:
        try:
            data = json.loads(text)
            required = {
                "headline",
                "description",
                "keywords",
                "daily_budget_estimate",
                "landing_page_angle"
            }
            if not required.issubset(data.keys()):
                return None
            return data
        except Exception as e:
            logger.warning(f"GoogleAdsAgent JSON parse failed: {e}")
            return None

    def _fallback_prompt(self, raw_text: str) -> str:
        return f"""
        Convert the following text into VALID JSON with keys:
        headline, description, keywords, daily_budget_estimate,
        landing_page_angle.
        Return ONLY JSON.

        Text:
        {raw_text}
        """

    def _fallback(self, reason: str) -> Dict:
        logger.error(f"GoogleAdsAgent fallback used: {reason}")
        return GoogleAdsAgentOutput(
            headline="",
            description="",
            keywords=[],
            daily_budget_estimate="",
            landing_page_angle="",
            error=reason
        ).model_dump()
