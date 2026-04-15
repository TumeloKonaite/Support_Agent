from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables when available."""

    app_name: str = Field(default="Support API")
    environment: str = Field(default="development")
    api_v1_prefix: str = Field(default="")
    content_data_dir: Path = Field(default=Path("data"))
    conversation_storage_dir: Path = Field(default=Path("data/conversations"))
    openai_api_key: SecretStr | None = Field(default=None)
    openai_model: str = Field(default="gpt-4.1-mini")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
