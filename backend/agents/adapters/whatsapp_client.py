import httpx

from backend.core.config import Settings


def send_whatsapp_message(to: str, message:str) -> dict:
    """
    Send whatsapp messages to a specific person.
    """
    url = (
        f"https://graph.facebook.com/"
        f"{Settings.WHATSAPP_API_VERSION}/"
        f"{Settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    )

    headers = {
        "Authorization": f"Bearer {Settings.WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {
            "body": message
        }
    }

    response = httpx.post(url=url, headers=headers, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()
