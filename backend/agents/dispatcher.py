import logging
import traceback

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.info)


class Dispatcher:
    """
        The Dispatcher recieves the Reasoner output
        and calls the correct agent.
        It ensures:
        - required inputs exist
        - clean error handling
        - structured responses
        - compatibility with ReasonerAgent (task_type + action)
    """
    def __init__(
            self,
            research_agent,
            persona_agent,
            content_agent,
            experiment_agent,
            analytics_agent
            ):
        self.research_agent = research_agent
        self.persona_agent = persona_agent
        self.content_agent = content_agent
        self.experiment_agent = experiment_agent
        self.analytics_agent = analytics_agent
    
    # Main Dispatcher Routing
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

        try:
            if action=="call_research_agent":
                result = self.research_agent.analyse_product(
                    product_text=user_payload.get("product_text", ""),
                    competitor_text=user_payload.get("competitor_text", "")
                )
                return {"status": "research_done", "result": result}
            elif action=="call_persona_agent":
                result = self.persona_agent.generate_persona(
                    product_text=user_payload.get("product_text", ""),
                    market_text=user_payload.get("market_text", "")
                )
                return {"status": "persona_done", "result": result}
            elif action=="call_experiment_agent":
                result = self.experiment_agent.run_experiment(
                    persona_text=user_payload.get("persona_text", ""),
                    channel=user_payload.get("channel", ""),
                    variants=user_payload.get("variants", [])
                )
                return {"status": "experiment_done", "result": result}
            elif action=="call_analytics_agent":
                result = self.analytics_agent.analyse_campaign(
                    campaign_results=user_payload.get("campaign_results", "")
                    )
                return {"status": "analytics_done", "result": result}
            else:
                return {
                    "status": "unknown_action",
                    "received_action": action,
                    "reasoner_output": reason_output
                }
        
        except Exception as e:
            logger.error(f"Agent crashed: {e}")
            return {
                "status": "agent_error",
                "error": str(e),
                "trace": traceback.format_exc()
            }
