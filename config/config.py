import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    REPO_URL: str = os.getenv("REPO_URL", "")
    TARGET_DIR: str = os.getenv("TARGET_DIR", "")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/codebase_llm.log")

settings = Settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("codebase_llm")
