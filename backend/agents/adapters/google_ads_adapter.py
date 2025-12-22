from typing import Dict, Any

from agents.adapters.base_channel_adapter import BaseChannelAdapter
from agents.schemas import GoogleAdsAgentOutput
from agents.adapters.google_ads_client import get_google_ads_client
from backend.core.config import settings


class GoogleAdsAdapter(BaseChannelAdapter):
    """
    Adapter for the Google Ads platform (Search Ads, MVP scope).
    """

    def validate(self, artifacts: Dict[str, Any]) -> bool:
        required_keys = [
            "headline",
            "description",
            "keywords",
            "daily_budget_estimate",
            "landing_page_angle",
        ]
        return all(key in artifacts for key in required_keys)

    def preview(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        return GoogleAdsAgentOutput(
            headline=artifacts.get("headline", ""),
            description=artifacts.get("description", ""),
            keywords=artifacts.get("keywords", []),
            daily_budget_estimate=artifacts.get("daily_budget_estimate", ""),
            landing_page_angle=artifacts.get("landing_page_angle", ""),
        ).model_dump()

    def publish(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate(artifacts):
            raise ValueError("Invalid Google Ads campaign artifact.")

        client = get_google_ads_client()
        customer_id = settings.GOOGLE_ADS_CUSTOMER_ID

        try:
            # --- Budget ---
            budget_service = client.get_service("CampaignBudgetService")
            budget_operation = client.get_type("CampaignBudgetOperation")
            budget = budget_operation.create
            budget.name = f"UHPM Budget"
            budget.delivery_method = (
                client.enums.BudgetDeliveryMethodEnum.STANDARD
            )
            budget.amount_micros = int(
                float(artifacts["daily_budget_estimate"]) * 1_000_000
            )

            budget_response = budget_service.mutate_campaign_budgets(
                customer_id=customer_id, operations=[budget_operation]
            )

            budget_resource = budget_response.results[0].resource_name

            # --- Campaign ---
            campaign_service = client.get_service("CampaignService")
            campaign_operation = client.get_type("CampaignOperation")
            campaign = campaign_operation.create
            campaign.name = f"UHPM Search Campaign"
            campaign.advertising_channel_type = (
                client.enums.AdvertisingChannelTypeEnum.SEARCH
            )
            campaign.status = client.enums.CampaignStatusEnum.PAUSED
            campaign.campaign_budget = budget_resource

            campaign_response = campaign_service.mutate_campaigns(
                customer_id=customer_id, operations=[campaign_operation]
            )

            return {
                "status": "published",
                "campaign_resource": (
                    campaign_response.results[0].resource_name,
                )
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
            }
