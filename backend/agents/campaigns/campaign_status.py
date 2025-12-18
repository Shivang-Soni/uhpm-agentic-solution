from enum import Enum


class CampaignStatus(str, Enum):
    CREATED = "created"
    PREVIEWING = "previewing"
    PREVIEWED = "previewed"
    PREVIEW_FAILED = "preview_failed"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    PUBLISH_FAILED = "publish_failed"
    FAILED = "failed"
