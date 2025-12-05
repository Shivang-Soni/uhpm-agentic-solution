import logging
import json

from llm.gemini_pipeline import invoke
from agents.retriever_agent import RetrieverAgent

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReasonerAgent:
    def __init__(self):
        self.retriever = RetrieverAgent()

    def decide(self, user_task: str):
        """
        Core reasoning layer:
        - retrieves memory
        - prepares a structured reasoning prompt
        - calls LLM
        - returns a structured JSON response with task classification + next action
        """

        logger.info("Starting reasoning process.")

        # Retrieve memory context
        retrieved = self.retriever.search_docs(user_task)

        # LLM Prompt
        prompt = f"""
        You are the CORE REASONING AGENT of an advanced marketing AI system.

        Your responsibilities:
        - Analyze the user's request
        - Use retrieved memory context
        - Decide what type of task this is
        - Identify the correct subsystem (agent) to call
        - Determine what inputs are needed for the next action
        - ALWAYS return a VALID JSON OBJECT. Never return text outside JSON.

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

        # Call LLM
        response = invoke(prompt)

        if not response:
            return {"error": "No reasoning response returned."}

        # Validate & parse JSON safely
        try:
            structured = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("ReasonerAgent: Response was not valid JSON. Wrapping text.")
            structured = {"raw_text": response}

        return structured
