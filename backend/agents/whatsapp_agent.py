import json
import logging

from llm.gemini_pipeline import invoke
from agents.schemas import WhatsappAgentOutput

# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class WhatsappAgent:
    """
    Generates Whatsapp ready messages, on the basis
    of product- and persona-information.
    """

    def generate_messages(
            self,
            product_text: str,
            persona_text: str,
            intent: str = "lead",
            tone: str = "friendly"
    ) -> dict:

        prompt = self._build_prompt(
            product_text=product_text,
            persona_text=persona_text,
            intent=intent,
            tone=tone
        )

        response = invoke(prompt)

        if not response:
            return self._fallback_response("empty response")

        parsed = self._try_parse(response)
        if parsed:
            return parsed

        # JSON Fallback
        fallback_prompt = self._fallback_prompt(response)
        fallback_response = invoke(fallback_prompt)

        if not fallback_response:
            return self._fallback_response("fallback empty")

        parsed_fallback = self._try_parse(fallback_response)
        if parsed_fallback:
            return parsed_fallback

        return self._fallback_response("JSON parsing failed.")

    def _build_prompt(
            self,
            product_text: str,
            persona_text: str,
            intent: str,
            tone: str
    ) -> str:
        return f"""
        You are a conversion-focused WhatsApp marketing assistant.

        Rules:
        - Write short, natural WhatsApp messages.
        - Max 3 sentences per message.
        - No emojis.
        - No markdown.
        - Clear call-to-action in every message.
        - Sound human, not like an ad.

        Context:
        Product:
        {product_text}

        Persona:
        {persona_text}

        Intent: {intent}
        Tone: {tone}

        Output format (VALID JSON ONLY):
        {{
        "initial_message": "",
        "follow_up_message": "",
        "closing_message": ""
        }}
        """

    def _try_parse(self, text: str) -> dict | None:
        try:
            data = json.loads(text)
            required_keys = {
                "initial_message",
                "follow_up_message",
                "closing_message"
            }
            if not required_keys.issubset(data.keys()):
                return None
            return data
        except Exception as e:
            logger.warning(f"Whatssapp Agent JSON parsing failed: {e}")
            return None

    def _fallback_prompt(
            self,
            raw_text: str
    ) -> str:
        return f"""
    The following text is NOT valid JSON.
    Convert it into valid JSON with exactly these keys:
    initial_message, follow_up_message, closing_message.

    Return ONLY JSON. Never explain.

    Text:
    {raw_text}
        """

    def _fallback_response(
            self,
            reason: str
    ) -> dict:
        logger.error(f"Whatsappagent fallback is being utilised: {reason}")
        return WhatsappAgentOutput(
            initial_message="",
            follow_up_message="",
            closing_message="",
            error=reason
        ).model_dump()
