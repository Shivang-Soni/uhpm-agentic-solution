from typing import Dict, Any


class BaseChannelAdapter:
    """
    Base Class for all Marketing pipelines.
    Responsiblities:
    - Validation of campaign output
    - Preview
    - Publishing through API calles
    """

    def validate(self, artifact: Dict[str, Any]) -> bool:
        pass

    def preview(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def publish(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        pass
