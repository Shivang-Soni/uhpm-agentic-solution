import os 
from dotenv import load_dotenv

load_dotenv()


class Settings:
    PROJECT_NAME: str = "UHPM-Agent"
    ENV: str = os.getenv("ENV", "dev")


settings = Settings()
