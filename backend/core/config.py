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

settings = Settings()
