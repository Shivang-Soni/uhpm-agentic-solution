from agents.registry import AgentRegistry

from agents.planner_agent import PlannerAgent
from agents.research_agent import ResearchAgent
from agents.persona_agent import PersonaAgent
from agents.content_agent import ContentAgent
from agents.experiment_agent import ExperimentationAgent
from agents.analytics_agent import AnalyticsAgent
from agents.channel_agents.preview_agent import PreviewAgent
from agents.channel_agents.publish_agent import PublishAgent

from agents.adapters.channel_adapter_dispatcher import ChannelAdapterDispatcher
from agents.repositories.in_memory_campaign_repository import InMemoryCampaignRepository


def build_registry() -> AgentRegistry:
    registry = AgentRegistry()

    # infra
    repository = InMemoryCampaignRepository()
    channel_dispatcher = ChannelAdapterDispatcher(repository)

    # core agents
    registry.register(PlannerAgent())
    registry.register(ResearchAgent())
    registry.register(PersonaAgent())
    registry.register(ContentAgent())
    registry.register(ExperimentationAgent())
    registry.register(AnalyticsAgent())

    # lifecycle agents
    registry.register(PreviewAgent(channel_dispatcher))
    registry.register(PublishAgent(channel_dispatcher))

    # hard validation: all Actions must be wired
    registry.validate()

    return registry
