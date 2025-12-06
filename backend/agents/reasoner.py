import logging
import json
from typing import Any, Dict

from llm.gemini_pipeline import invoke

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReasonerAgent:
    """
    ReasonerAgent: Classifies the user input and gives back a JSON dictionary.
    Includes:
    - retrieval lookup
    - classification
    - fallback JSON repair
    - validation
    """
    def __init__(self, retriever):
        self.max_retries = 2
        self.retriever = retriever
    
    def _build_prompt(self, user_task: str, retrieved: Any) -> str:
        return f"""
        You are the CORE REASONING AGENT of an advanced marketing AI system.

        Your responsibilities:
        - Analyze the user's request
        - Use retrieved memory context
        - Decide what type of task this is
        - Identify the correct subsystem (agent) to call
        - Determine what inputs are needed for the next action
        - ALWAYS return a VALID JSON OBJECT. Never return text outside JSON
        (NEVER EXPLAIN).

        === Classification Rules ===
        If the task is about:
        - target audience
        - personas
        - ideal customer
        - segmentation
        - buyer insights  
        → classify it as: "persona"
        → action: "call_persona_agent"
        → inputs_needed: ["product_text", "market_text"]

        Other categories:
        - research → action: "call_research_agent"
        - analysis → action: "call_analysis_agent"
        - content → action: "call_content_agent"
        - experiment → action: "call_experiment_agent"

        === INPUT ===
        User request:
        {user_task}

        Retrieved memory:
        {json.dumps(retrieved, indent=2)}

        === OUTPUT FORMAT ===
        {{
          "task_type": "research|persona|content|experiment|analysis",
          "reasoning": "short high-level justification",
          "action": "which agent to call next",
          "inputs_needed": ["list", "of", "required", "inputs"]
        }}
        """
    
    def _fallback_prompt(self, raw_text: str) -> str:
        return f"""
        The previous response was not valid JSON. Convert the following text
        into VALID JSON with fields:
        task_type, reasoning, action, inputs_needed.
        Return ONLY JSON - NEVER EXPLAIN.

        Raw text:
        {raw_text}
        """
    
    def _try_parse(self, text: str) -> Dict[str, Any] | None:
        """
        Tries to parse JSON and returns Dict when successful,
        otherwise None
        """
        try:
            data = json.loads(text)
            # Minimal validation
            required = {"task_type", "reasoning", "action", "inputs_needed"}
            if not required.issubset(set(data.keys())):
                logger.warning("Parson JSON is missing required keys.")
                return None
            # Ensure types
            if not isinstance(data.get("inputs_needed", []), list):
                logger.warning("inputs_needed is not a list.")
                return None
            return data
        except Exception as e:
            logger.warning(f"_try_parse() failed: {e}")
            return None

    def decide(self, user_task: str):
        """
        Core reasoning layer:
        - retrieves memory
        - prepares a structured reasoning prompt
        - calls LLM
        - returns a structured JSON response with task classification + next action
        """

        logger.info("Starting reasoning process.")

        retrieved = {}

        if self.retriever is not None:
            try:
                # Retrieve memory context
                retrieved = self.retriever.search_docs(user_task)
            except Exception as e:
                logger.warning(f"Retriever failed: {e}")
                retrieved = {}

        prompt = self._build_prompt(user_task, retrieved)

        # Call LLM
        response = invoke(prompt)

        if not response:
            logger.error("No reasoning response from Agent on initial call.")
            return {"error": "No reasoning response returned."}

        # Validate & parse JSON safely
        parsed = self._try_parse(response)
        if parsed:
            logger.info("Successfully parsed initial LLM response.")
            return parsed
        
        logger.warning(
            "Initial parse failed - requesting JSON-only fallback from Agent"
            )
        fallback = self._fallback_prompt(response)
        updated_response = invoke(fallback)

        if not updated_response:
            logger.error("No response from LLM on fallback.")
            return {"error": "no_response_on_fallback", "raw_first": response}
        
        parsed2 = self._try_parse(updated_response)
        if parsed2:
            logger.info("Successfully parsed fallback LLM response.")
            return parsed2
        
        logger.error("Both parsing attempts failed.")
        return {
            "error": "json_parsing_failed",
            "raw_first": response,
            "raw_fallback": updated_response

        }