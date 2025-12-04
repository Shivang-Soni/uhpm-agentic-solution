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
        Core reasoning layer, that:
        - retrieves context
        - prepared a structured reasoning prompt
        - calls LLM
        - returns a structured JSON response
        """

        logger.info("Starting reasoning process.")

        # Memory retrieval
        retrieved = self.retriever.search_docs(user_task)

        # Prepare the prompt for the LLM call
        prompt = f"""
        You are the CORE REASONING AGENT of a marketing AI system.

        Your job is to:
        - Analyse the user's task
        - Use retrieved memory
        - Determine clearly structured next steps
        - ALWAYS return JSON only.

        User's request:
        {user_task}

        Retrieved memory (past research results, notes, insights):
        {json.dumps(retrieved, indent=2)}

        Respond ONLY in the following JSON format:
        {{
        "task_type": "research|persona|content|experiment|analysis",
        "reasoning": "your chain of thought (short, high-level)",
        "action": "what the system should do next",
        "inputs_needed": ["list", "of", "inputs", "required"]
        }}
        """

        response = invoke(prompt)

        if not response:
            return {"error": "No reasoning response returned."}
        
        # Try to parse the response as JSON
        try: 
            structured_response = json.loads(response)
        except json.JSONDecodeError:
            logger.warning(
                "Agent returned non-JSON response....Wrapping raw text"
                )
            structured_response = {"raw_text": response}
        
        return structured_response
