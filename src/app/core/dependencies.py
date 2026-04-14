from functools import lru_cache

from src.app.core.config import get_settings
from src.app.domain.support.service import SupportService
from src.app.infrastructure.storage.conversation_store import ConversationStore
from src.app.infrastructure.storage.file_conversation_store import (
    FileConversationStore,
)


@lru_cache
def get_conversation_store() -> ConversationStore:
    """Return the configured conversation store implementation."""
    settings = get_settings()
    return FileConversationStore(settings.conversation_storage_dir)


def get_support_service() -> SupportService:
    """Build the support service with its infrastructure dependencies."""
    return SupportService(conversation_store=get_conversation_store())
