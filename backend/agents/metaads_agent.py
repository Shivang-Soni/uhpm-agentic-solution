class MetaAdsAgent:
    def generate_campaign(
            self, product_text, persona_text, campaign_budget, tone="neutral"):
        return {
            "headline": f"Meta Ad for {product_text}",
            "copy": f"Targeting {persona_text} with budget: {campaign_budget}",
            "cta": "Click here!"
        }