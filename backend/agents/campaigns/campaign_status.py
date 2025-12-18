from enum import Enum


class CampaignStatus(str, Enum):
    CREATED = "created"
    PREVIEWING = "previewing"
    PREVIEWED = "previewed"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    FAILED = "failed"

