from functools import lru_cache

from src.app.core.config import Settings, get_settings
from src.app.domain.support.prompt_builder import (
    BusinessProfileSource,
    KnowledgeSource,
    SupportPromptBuilder,
)
from src.app.domain.support.retrieval import RetrievalPipeline
from src.app.domain.support.service import SupportService
from src.app.infrastructure.content.business_profile_loader import (
    BusinessProfileLoader,
)
from src.app.infrastructure.content.knowledge_loader import KnowledgeLoader
from src.app.infrastructure.llm.openai_client import LLMClient, OpenAIClient
from src.app.infrastructure.retrieval.indexer import build_embedder
from src.app.infrastructure.retrieval.retriever import Retriever, VectorStoreRetriever
from src.app.infrastructure.retrieval.vector_store import JsonVectorStore
from src.app.infrastructure.storage.conversation_store import ConversationStore
from src.app.infrastructure.storage.file_conversation_store import (
    FileConversationStore,
)


def get_config() -> Settings:
    """Return the runtime application settings."""
    return get_settings()


@lru_cache
def get_conversation_store() -> ConversationStore:
    """Return the configured conversation store implementation."""
    settings = get_config()
    return FileConversationStore(settings.conversation_storage_dir)


@lru_cache
def get_openai_client() -> LLMClient:
    """Return the configured OpenAI client."""
    settings = get_config()
    if settings.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY must be configured to use chat endpoints")

    return OpenAIClient(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_model,
    )


@lru_cache
def get_business_profile_loader() -> BusinessProfileSource:
    """Return the configured business profile content source."""
    settings = get_config()
    return BusinessProfileLoader(settings.content_data_dir)


@lru_cache
def get_knowledge_loader() -> KnowledgeSource:
    """Return the configured support knowledge content source."""
    settings = get_config()
    return KnowledgeLoader(settings.content_data_dir)


@lru_cache
def get_support_prompt_builder() -> SupportPromptBuilder:
    """Return the support prompt builder."""
    return SupportPromptBuilder(
        business_profile_source=get_business_profile_loader(),
        knowledge_source=get_knowledge_loader(),
    )


@lru_cache
def get_retriever() -> Retriever:
    """Return the configured retrieval implementation."""
    settings = get_config()
    return VectorStoreRetriever(
        embedder=build_embedder(settings),
        vector_store=JsonVectorStore(settings.retrieval_vector_store_path),
        default_top_k=settings.retrieval_top_k,
    )


@lru_cache
def get_support_retrieval_pipeline() -> RetrievalPipeline:
    """Return the support-domain retrieval pipeline."""
    return RetrievalPipeline(retriever=get_retriever())


def get_support_service() -> SupportService:
    """Build the support service with its infrastructure dependencies."""
    return SupportService(
        conversation_store=get_conversation_store(),
        openai_client=get_openai_client(),
        prompt_builder=get_support_prompt_builder(),
        retrieval_pipeline=get_support_retrieval_pipeline(),
    )
