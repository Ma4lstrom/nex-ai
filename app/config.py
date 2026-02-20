from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    ANTHROPIC_API_KEY: str = ""
    API_KEY: str
    REFERENCE_IMAGE_DIR: str = "storage/references"
    TEMP_IMAGE_DIR: str = "storage/temp"
    MODEL_DIR: str = "models"
    ALLOWED_ORIGINS: List[str] = ["*"]
    VISUAL_SIMILARITY_WEIGHT: float = 0.5
    COLOR_SIMILARITY_WEIGHT: float = 0.25
    CLAUDE_SCORE_WEIGHT: float = 0.25
    MAX_IMAGE_SIZE_MB: int = 10
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]

    class Config:
        env_file = ".env"

settings = Settings()
