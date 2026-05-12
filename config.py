"""Configuration via .env file."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Registry: name -> path
    model_registry: dict[str, str] = {
        "arabic": "models/Arabic-sign-language-translation-CNN.h5",
        "asl":    "models/asl_model.h5",
    }
    default_model: str = "arabic"

    # Each model has its own image size
    image_size_per_model: dict[str, int] = {
        "arabic": 64,
        "asl":    32,
    }

    confidence_threshold: float = 0.50
    rolling_window: int = 16
    max_image_bytes: int = 10 * 1024 * 1024
    max_video_bytes: int = 100 * 1024 * 1024

    app_title: str = "Sign Language API"
    app_version: str = "2.0.0"
    cors_origins: list[str] = ["*"]


settings = Settings()
