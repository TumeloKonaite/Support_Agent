from functools import lru_cache

from src.app.core.config import get_settings
from src.app.domain.support.prompt_builder import SupportPromptBuilder
from src.app.domain.support.service import SupportService
from src.app.infrastructure.content.business_profile_loader import (
    BusinessProfileLoader,
)
from src.app.infrastructure.content.knowledge_loader import KnowledgeLoader
from src.app.infrastructure.llm.openai_client import OpenAIClient
from src.app.infrastructure.storage.conversation_store import ConversationStore
from src.app.infrastructure.storage.file_conversation_store import (
    FileConversationStore,
)


@lru_cache
def get_conversation_store() -> ConversationStore:
    """Return the configured conversation store implementation."""
    settings = get_settings()
    return FileConversationStore(settings.conversation_storage_dir)


@lru_cache
def get_openai_client() -> OpenAIClient:
    """Return the configured OpenAI client."""
    settings = get_settings()
    if settings.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY must be configured to use chat endpoints")

    return OpenAIClient(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_model,
    )


@lru_cache
def get_support_prompt_builder() -> SupportPromptBuilder:
    """Return the support prompt builder."""
    settings = get_settings()
    return SupportPromptBuilder(
        business_profile_source=BusinessProfileLoader(settings.content_data_dir),
        knowledge_source=KnowledgeLoader(settings.content_data_dir),
    )


def get_support_service() -> SupportService:
    """Build the support service with its infrastructure dependencies."""
    return SupportService(
        conversation_store=get_conversation_store(),
        openai_client=get_openai_client(),
        prompt_builder=get_support_prompt_builder(),
    )
