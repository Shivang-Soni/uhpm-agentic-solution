import logging
from typing import Dict, Any

from agents.adapters.base_channel_adapter import BaseChannelAdapter
from agents.schemas import GoogleAdsAgentOutput
from agents.adapters.google_ads_client import get_google_ads_client
from backend.core.config import settings

logger = logging.getLogger(__name__)


class GoogleAdsAdapter(BaseChannelAdapter):
    """
    Adapter for Google Ads (Search Ads only, MVP scope).

    Responsibilities:
    - Validate Google Ads campaign artifacts
    - Provide preview-ready normalized output
    - Publish campaign via Google Ads API
    """

    supports_preview = True
    supports_publish = True

    # Validation

    def validate(self, artifacts: Dict[str, Any]) -> bool:
        required_keys = [
            "headline",
            "description",
            "keywords",
            "daily_budget_estimate",
            "landing_page_angle",
        ]
        return all(key in artifacts for key in required_keys)

    # Preview

    def preview(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        return GoogleAdsAgentOutput(
            headline=artifacts["headline"],
            description=artifacts["description"],
            keywords=artifacts["keywords"],
            daily_budget_estimate=artifacts["daily_budget_estimate"],
            landing_page_angle=artifacts["landing_page_angle"],
        ).model_dump()

    # Publish

    def publish(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publishes a paused Search campaign to Google Ads.
        Ad groups, ads & keywords will be added in later iterations.
        """

        logger.info("[GoogleAdsAdapter] Publishing campaign")

        client = get_google_ads_client()
        customer_id = settings.GOOGLE_ADS_CUSTOMER_ID

        # Budget
        budget_service = client.get_service("CampaignBudgetService")
        budget_operation = client.get_type("CampaignBudgetOperation")

        budget = budget_operation.create
        budget.name = "UHPM Daily Budget"
        budget.delivery_method = (
            client.enums.BudgetDeliveryMethodEnum.STANDARD
        )
        budget.amount_micros = int(
            float(artifacts["daily_budget_estimate"]) * 1_000_000
        )

        budget_response = budget_service.mutate_campaign_budgets(
            customer_id=customer_id,
            operations=[budget_operation],
        )

        budget_resource_name = (
            budget_response.results[0].resource_name
        )

        # Campaign
        campaign_service = client.get_service("CampaignService")
        campaign_operation = client.get_type("CampaignOperation")

        campaign = campaign_operation.create
        campaign.name = "UHPM Search Campaign"
        campaign.advertising_channel_type = (
            client.enums.AdvertisingChannelTypeEnum.SEARCH
        )
        campaign.status = client.enums.CampaignStatusEnum.PAUSED
        campaign.campaign_budget = budget_resource_name

        campaign_response = campaign_service.mutate_campaigns(
            customer_id=customer_id,
            operations=[campaign_operation],
        )

        campaign_resource_name = (
            campaign_response.results[0].resource_name
        )

        logger.info(
            "[GoogleAdsAdapter] Campaign published successfully | "
            f"resource={campaign_resource_name}"
        )

        return {
            "provider": "google_ads",
            "status": "published",
            "external_campaign_id": campaign_resource_name,
            "mode": "paused",
        }
