import logging

from agents.research_agent import ResearchAgent
from agents.persona_agent import PersonaAgent

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.info)


class Dispatcher:
    def __init__(self):
        self.research_agent = ResearchAgent()
        self.persona_agent = PersonaAgent()
        # self.content_agent = ContentAgent()
        # self.experiment_agent = ExperimentAgent()
    
    def run(self, reason_output: dict, user_payload: dict):
        """
        Orchestrates actual execution of the selected agent

        -reason_output: dict: The output from the Reasoning Agent
        -user_payload: dict: the user input payload from the API call
        """
        task_type = reason_output.get("task_type")
        action = reason_output.get("action")
        inputs_needed = reason_output.get("inputs_needed", {})

        logger.info(
            f"Dispatcher received task: {task_type} with action: {action}"
            )
        
        # Check if required inputs are present already
        missing_inputs = [
            field for field in inputs_needed
            if field not in user_payload
        ]

        if missing_inputs:
            return {
                "status": "waiting_for_inputs",
                "missing": missing_inputs,
                "what_is_needed": inputs_needed
            }

        if task_type == "research":
            result = self.research_agent.analyse_product(
                product_text=user_payload.get("product_text", ""),
                competitor_text=user_payload.get("competitor_text", "")
            )

            return {
                "status": "research_completed",
                "agent_result": result
            }
        elif task_type == "persona":
            result = self.persona_agent.generate_persona(
                product_text=user_payload.get("product_text", ""),
                market_text=user_payload.get("market_text", "")
            )

            return {
                "status": "persona_generated",
                "agent_result": result
            }
            
        # fallback for unrecognised task type(s)
        return {
            "status": "unrecognised_task_type",
            "details": reason_output
        }
