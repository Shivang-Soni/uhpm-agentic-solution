from typing import Dict, Any

from agents.publishers.base_publisher import BasePublisher


class MockPublisher(BasePublisher):
    """
    Mock publisher for local testing and preview.
    """
    def publish(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "published_mock",
            "artifacts": artifacts
        }
