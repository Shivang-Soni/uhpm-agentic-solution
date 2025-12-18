from enum import Enum


class CampaignStatus(str, Enum):
    CREATED = "created"
    PREVIEWED = "previewed"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    FAILED = "failed"

