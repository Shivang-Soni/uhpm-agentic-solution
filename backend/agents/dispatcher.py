import logging

from agents.research_agent import ResearchAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.info)


class Dispatcher:
    def __init__(self):
        self.research_agent = ResearchAgent()
        # self.persona_agent = PersonaAgent()
        # self.content_agent = ContentAgent()
        # self.experiment_agent = ExperimentAgent()
    
    def run(self, reason_output: dict):
        task_type = reason_output.get("task_type")
        action = reason_output.get("action")
        inputs_needed = reason_output.get("inputs_needed", {})

        logger.info(
            f"Dispatcher received task: {task_type} with action: {action}"
            )

        if task_type == "research":
            return {
                "status": "running_research_agent",
                "expected_inputs": inputs_needed
            }
        # fallback for unrecognised task type(s)
        return {
            "status": "unrecognised_task_type",
            "details": reason_output
        }
