import os
from dotenv import load_dotenv

# Load .env before anything else
load_dotenv()


class Settings:
    PROJECT_NAME: str = "UHPM-Agent"
    ENV: str = os.getenv("ENV", "dev")

    # Vectorstore path
    PERSIST_DIRECTORY: str = os.getenv(
        "PERSIST_DIRECTORY", "vectorstore/chroma_store"
    )

    # Telemetry flags
    ALLOW_TELEMETRY: bool = os.getenv("ALLOW_TELEMETRY", "false").lower() == "true"
    CHROMA_TELEMETRY_ENABLED: bool = os.getenv(
        "CHROMA_TELEMETRY_ENABLED", "false"
    ).lower() == "true"
    OTEL_SDK_DISABLED: bool = os.getenv(
        "OTEL_SDK_DISABLED", "true"
    ).lower() == "true"

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Meta Ads API
    META_ADS_TOKEN: str = os.getenv("META_ADS_TOKEN", "")
    META_ADS_ACCOUNT_ID = os.getenv("META_ADS_ACCOUNT_ID", "")

    # Google Ads API
    GOOGLE_ADS_DEVELOPER_TOKEN: str = os.getenv(
        "GOOGLE_ADS_DEVELOPER_TOKEN", ""
        )
    GOOGLE_ADS_CUSTOMER_ID: str = os.getenv(
        "GOOGLE_ADS_CUSTOMER_ID", ""
        )
    GOOGLE_ADS_LOGIN_CUSTOMER_ID: str = os.getenv(
        "GOOGLE_ADS_LOGIN_CUSTOMER_ID", ""
        )
    GOOGLE_ADS_SERVICE_ACCOUNT_JSON: str = os.getenv(
        "GOOGLE_ADS_SERVICE_ACCOUNT_JSON", ""
    )

    # Whatsapp API
    WHATSAPP_ACCESS_TOKEN: str = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
    WHATSAPP_PHONE_NUMBER: str = os.getenv("WHATSAPP_PHONE_NUMBER", "")
    WHATSAPP_API_NUMBER: str = os.getenv("WHATSAPP_API_NUMBER", "")

settings = Settings()
