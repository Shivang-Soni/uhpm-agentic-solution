import os
import logging

from google import genai

from core.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialise Gemini client
client = genai.Client(api_key=settings.GEMINI_API_KEY)


def invoke(prompt: str) -> str | None:
    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=prompt,

    )

    logging.info(f"Gemini Response: {response}")
    return response.text if response else None
