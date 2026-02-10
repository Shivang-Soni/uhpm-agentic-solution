from enum import Enum


class Action(str, Enum):
    # Campaign Lifecycle
    PREVIEW_CAMPAIGN = "preview_campaign"
    PUBLISH_CAMPAIGN = "publish_campaign"

    # Agent calls
    GENERATE_CONTENT = "generate_content"
    ANALYSE_PERFORMANCE = "analyse_performance"

    GENERATE_PERSONA = "generate_persona"
    RUN_EXPERIMENT = "run_experiment"

    PLAN = "plan"
    EVALUATE = "evaluate"
