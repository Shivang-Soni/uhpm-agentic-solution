from google.ads.googleads.client import GoogleAdsClient

from backend.core.config import Settings


def get_google_ads_client() -> GoogleAdsClient:
    """
    Load and return the GoogleAdsClient
    """

    config = {
        "developer_token": Settings.GOOGLE_ADS_DEVELOPER_TOKEN,
        "login_customer_id": Settings.GOOGLE_ADS_CUSTOMER_ID,
        "json_key_file_path": Settings.GOOGLE_ADS_SERVICE_ACCOUNT_JSON,
        "use_proto_plus": True
    }

    return GoogleAdsClient.load_from_dict(config)
