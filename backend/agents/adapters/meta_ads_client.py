import httpx
from backend.core.config import settings


def get_meta_ads_client() -> httpx.Client:
    """
    Return a configured HTTP client for the Meta Graph API.
    """
    return httpx.Client(
        base_url="https://graph.facebook.com/v17.0",
        headers={
            "Authorization": f"Bearer {settings.META_ADS_TOKEN}",
            "Content-Type": "application/json"
        },
        timeout=10
    )
