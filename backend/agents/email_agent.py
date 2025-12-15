class EmailAgent:
    def generate_campaign(
            self,
            product_text,
            persona_text,
            email_template,
            tone="friendly"
    ):
        return {
            "subject_line": f"{product_text} Special Offer!",
            "body": f"Hi {persona_text}, check out our product.\n{email_template}",
            "cta": "Shop now!"
        }