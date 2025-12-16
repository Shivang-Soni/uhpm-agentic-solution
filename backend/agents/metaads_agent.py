from agents.schemas import MetaAdsAgentOutput


class MetaAdsAgent:
    def generate_campaign(
            self, product_text, persona_text, campaign_budget, tone="neutral"):
        return MetaAdsAgentOutput(
            platform="meta",
            headline=f"Meta Ad for {product_text}",
            persona=persona_text,
            budget=campaign_budget,
            tone=tone
        ).model_dump()
