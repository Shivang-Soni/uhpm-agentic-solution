from typing import Dict, Any


class BasePublisher:
    """
    Base class for all channel publishers.
    Real API Integration will subclass this.
    """

    def publish(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        pass