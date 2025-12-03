import os 
from dotenv import load_dotenv

load_dotenv()


class Settings:
    PROJECT_NAME: str = "UHPM-Agent"
    ENV: str = os.getenv("ENV", "dev")
    PERSIST_DIRECTORY: str = os.getenv(
        "PERSIST_DIRECTORY", "backend/chroma_store"
        )


settings = Settings()
