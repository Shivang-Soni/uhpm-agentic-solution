class GoogleAdsAgent:
    def generate_campaign(
            self,
            product_text,
            persona_text,
            campaign_budget,
            tone="neutral"
    ):
        return {
            "ad_title": f"Google Ad: {product_text}",
            "description": f"Persona: {persona_text}, Budget: {campaign_budget}",
            "url": "https://www.example.com"
        }