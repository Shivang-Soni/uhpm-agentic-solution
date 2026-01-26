import logging

from agents.registry import AgentRegistry
from agents.channel_agents.preview_agent import PreviewAgent
from agents.channel_agents.publish_agent import PublishAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class MockChannelDispatcher:
    def preview(self, channel, artifacts):
        return (
            "mock_preview": True,
            "channel": channel
        )

    def publish(self, channel, artifacts):
        return {
            "mock_publish": True,
            "channel": channel
        }
    

    registry = AgentRegistry()

    channel_dispatcher = MockChannelDispatcher()

    preview_agent = PreviewAgent(channel_dispatcher=channel_dispatcher)
    publish_agent = PublishAgent(channel_dispatcher=channel_dispatcher)

    registry.register(preview_agent)
    registry.register(publish_agent)

    logger.info(f"Registered Agents: {registry.list_actions()}")