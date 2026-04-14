from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables when available."""

    app_name: str = Field(default="Support API")
    environment: str = Field(default="development")
    api_v1_prefix: str = Field(default="")
    conversation_storage_dir: Path = Field(default=Path("data/conversations"))

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
