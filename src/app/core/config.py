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
    retrieval_embedding_provider: str = Field(default="hashing")
    retrieval_embedding_model: str = Field(default="hashing-v1")
    retrieval_top_k: int = Field(default=3, ge=1)
    retrieval_chunk_size: int = Field(default=500, ge=1)
    retrieval_chunk_overlap: int = Field(default=100, ge=0)
    retrieval_vector_store_path: Path = Field(
        default=Path("data/retrieval/vector_store.json")
    )
    support_observability_enabled: bool = Field(default=True)
    support_observability_prompt_preview_enabled: bool = Field(default=False)
    support_observability_redact_sensitive_fields: bool = Field(default=True)
    support_observability_max_preview_chars: int = Field(default=160, ge=16)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
